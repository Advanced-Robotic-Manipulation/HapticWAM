#!/usr/bin/env python3
"""Hash-pinned delivery-feedback-only overlay; historical veto stays fd4a032.

Emit one module outside protected source trees. Live-veto-only needs no source
overlay: frozen v1 already implements it with historical request feedback.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULE = "tools/sim/deployment_filters.py"
BASE_SHA = "a8074266d7fe7e09e978298d119fe7dd3ad4f2fe37920870799023a7ac18cc82"
DONOR_SHA = "b48664e2a52aa59113f3dc08b41602e1f6ceda0591ae453e61edf529e56d509e"


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def transform(base, donor):
    if sha(base) != BASE_SHA or sha(donor) != DONOR_SHA:
        raise ValueError("Unreviewed source hashes")
    # Retain the exact local v1 release passthrough; import no newer native bridge.
    start = '        opening_mask = getattr(adapter, "placement_release_opening_mask", None)\n'
    end = '        rewritten = rec["action"] in (\n'
    old_openings = base[base.index(start) : base.index(end)]
    new_start = "        from phantom.deploy.release_controller import restore_policy_openings\n"
    result = donor[: donor.index(new_start)] + old_openings + donor[donor.index(end) :]
    compile(result, MODULE, "exec")

    # Every native veto decision/history method must stay exactly as archived.
    def methods(text):
        cls = next(
            n
            for n in ast.parse(text).body
            if isinstance(n, ast.ClassDef) and n.name == "TerminalVetoFilter"
        )
        return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}

    # __call__ is what this overlay rewrites. __init__ is exempt as well: the
    # live donor's __init__ gained the `controller_profile` argument for
    # minimal_v5 in abd02da, which only selects a placement controller and
    # validates its pairing with `implementation`. The archived fd4a032 veto
    # semantics live in the decision/history methods below, which stay pinned
    # byte-for-byte.
    exempt = ("__call__", "__init__")
    old, new = methods(base), methods(result)
    assert all(new[name] == value for name, value in old.items() if name not in exempt)
    assert set(new) - set(old) == {"feedback_source", "_feedback"}
    return result


def prepare(base_source, output_overlay, donor_source=REPO):
    base_source, output_overlay, donor_source = map(
        Path, (base_source, output_overlay, donor_source)
    )
    if output_overlay.exists():
        raise FileExistsError("Overlay must be new")
    if any(
        output_overlay.resolve().is_relative_to(p.resolve())
        for p in (base_source, donor_source)
    ):
        raise ValueError("Overlay must be outside protected source trees")
    text = transform(
        (base_source / MODULE).read_text(), (donor_source / MODULE).read_text()
    )
    manifest = {
        "variant": "frozen_v1_plus_delivery_feedback_only",
        "base_source": str(base_source.resolve()),
        "donor_source": str(donor_source.resolve()),
        "base_sha256": {MODULE: BASE_SHA},
        "donor_sha256": {MODULE: DONOR_SHA},
        "output_sha256": {MODULE: sha(text)},
        "profile": "Keep archived fd4a032 veto, placement release, sensors, limits and scene unchanged",
        "semantics": "Only veto TCP/aperture are read at delivery; model keeps captured request. Historical local release passthrough retained.",
        "live_veto_only": "Use original v1 source with only terminal-veto implementation=live; its request-snapshot feedback stays unchanged.",
    }
    path = output_overlay / MODULE
    path.parent.mkdir(parents=True)
    path.write_text(text)
    (output_overlay / "anchor_feedback_overlay.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path, required=True)
    parser.add_argument("--donor-source", type=Path, default=REPO)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.base_source, args.output_overlay, args.donor_source), indent=2
        )
    )


if __name__ == "__main__":
    main()
