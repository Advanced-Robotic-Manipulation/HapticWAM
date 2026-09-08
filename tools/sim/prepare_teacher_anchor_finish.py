#!/usr/bin/env python3
"""Emit a hash-pinned FINISH-only overlay for the successful frozen v1 source.

Never mutates the base or donor tree and never runs physics/model/hardware code.
Copy the two emitted modules onto a NEW copy of the frozen v1 source; enable
finish_after_release explicitly in a separate release JSON. The frozen v1
runner continues its declared horizon with live sensors and safety after FINISH.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_HASHES = {
    "phantom/sim/policy_adapter.py": "3ff10500ccab0ebd747207f95bdf3049ae001fe3d98a3ba59489b809d208ef7d",
    "phantom/sim/release_controller.py": "16c8853989c1c9bc7a99cc949de8757e29f79ca7022fd6494f23ce6f2194d3b8",
}
DONOR_HASHES = {
    "phantom/sim/policy_adapter.py": "c405c9e472252697d25e438b00f2505fe7cdbfa54df8b59168764c477549d8c6",
    "phantom/deploy/release_controller.py": "3405da4cf01617d42ff0348e8ad6d2e20b69801572b853588fb5fb8c1f867e26",
}
FINISH_METHODS = (
    "reset",
    "ready_for_replan",
    "replan",
    "submit",
    "placement_release_opening_mask",
    "step",
)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_pinned(root, expected):
    result = {}
    for name, digest in expected.items():
        value = (root / name).read_text()
        if sha(value) != digest:
            raise ValueError(
                f"Unreviewed source hash for {root / name}; expected {digest}"
            )
        result[name] = value
    return result


def replace_methods(base, donor):
    """Copy only reviewed completion methods; retain v1 construction and veto eligibility."""

    def methods(source):
        cls = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == "SimulationPolicyAdapter"
        )
        return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    old, new = methods(base), methods(donor)
    old_lines, new_lines = (
        base.splitlines(keepends=True),
        donor.splitlines(keepends=True),
    )
    for name in sorted(FINISH_METHODS, key=lambda n: old[n].lineno, reverse=True):
        a, b = old[name], new[name]
        if a.decorator_list or b.decorator_list:
            raise ValueError("Unexpected decorated completion method")
        old_lines[a.lineno - 1 : a.end_lineno] = new_lines[b.lineno - 1 : b.end_lineno]
    result = "".join(old_lines)
    compile(result, "phantom/sim/policy_adapter.py", "exec")
    return result


def prepare(base_source, output_overlay, donor_source=REPO, *, dry_run=False):
    base_source, output_overlay, donor_source = map(
        Path, (base_source, output_overlay, donor_source)
    )
    if output_overlay.exists():
        raise FileExistsError("Overlay output must be new; preserve existing evidence")
    for protected in (base_source, donor_source):
        if output_overlay.resolve().is_relative_to(protected.resolve()):
            raise ValueError("Overlay must be outside the protected source trees")
    base = read_pinned(base_source, BASE_HASHES)
    donor = read_pinned(donor_source, DONOR_HASHES)
    # The shared donor module's first two classes are exactly v1 plus FINISH.
    # Omit its native bridge helpers: the old adapter keeps its local constructor
    # and _original_policy_grip implementation and imports only these classes.
    release = donor["phantom/deploy/release_controller.py"]
    split = next(
        n.lineno
        for n in ast.parse(release).body
        if isinstance(n, ast.FunctionDef) and n.name == "make_release_controller"
    )
    release = "".join(release.splitlines(keepends=True)[: split - 1]).rstrip() + "\n"
    outputs = {
        "phantom/sim/policy_adapter.py": replace_methods(
            base["phantom/sim/policy_adapter.py"],
            donor["phantom/sim/policy_adapter.py"],
        ),
        "phantom/sim/release_controller.py": release,
    }
    for name, text in outputs.items():
        compile(text, name, "exec")
    manifest = {
        "variant": "frozen_v1_plus_finish_only",
        "base_source": str(base_source.resolve()),
        "donor_source": str(donor_source.resolve()),
        "base_sha256": BASE_HASHES,
        "donor_sha256": DONOR_HASHES,
        "output_sha256": {name: sha(value) for name, value in outputs.items()},
        "adapter_methods_replaced": list(FINISH_METHODS),
        "unchanged_by_overlay": [
            "deployment_filters.py (fd4a032/request-snapshot semantics)",
            "gel_contact.py/tactile_proxy.py",
            "wrist proxy",
            "run_waffles.py/scene/dynamics",
            "safety guards/limits/debounce",
            "model inputs and inference recipe",
        ],
        "enablement": {
            "separate_release_config_override": {"finish_after_release": True},
            "horizon": "keep original declared60s; no success early-stop",
        },
        "completion_semantics": "after policy release plus measured unloaded/open dwell and accepted open command; freeze achieved measured TCP/grip; no replans; live safety can still stop",
        "scope": "completion state is not object task success; no object scorer or object-state input",
        "assembly": "copy these two modules only onto a NEW copy of base_source; leave every other v1 file and matched input unchanged",
        "dry_run": dry_run,
    }
    if not dry_run:
        output_overlay.mkdir(parents=True)
        for name, text in outputs.items():
            path = output_overlay / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (output_overlay / "anchor_finish_overlay.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path, required=True)
    parser.add_argument("--donor-source", type=Path, default=REPO)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.base_source,
                args.output_overlay,
                args.donor_source,
                dry_run=args.dry_run,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
