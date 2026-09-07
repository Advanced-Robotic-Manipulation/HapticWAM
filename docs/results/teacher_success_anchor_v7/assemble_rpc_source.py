#!/usr/bin/env python3
"""Copy the immutable v6 runtime and backport only opt-in RPC delivery timing."""

import argparse
import ast
import copy
import difflib
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

FILES = ("phantom/sim/policy_adapter.py", "tools/sim/run_waffles.py")
BASE_PINS = (
    "97adde55eb1b6e619054c45426e571ab08fe39f5f2da16e51b3b5a02022eae88",
    "903eae23fe5f09e3b1b5fc6c2153edc3de1d43619d0f2719201eb5f373443be2",
)
DONOR_PINS = (
    "c68029e687c3d78d51d396e6ee95249f263cbb957bae86d5fb0603a486c0923b",
    "67895ec44ff25f7c9e5520fcb404f1a9ae9d0238baca634cd9990d892cc541ca",
)
HELPER_PIN = "41c0c6beebd805dda4d9bf7c66e507491eb1121cdb5cf70a701ce845b37e027f"
PATCH_PIN = "32ab34567127e9e5e621a92ed8e4730cb49afb6bb61650cc5da1ad89c4235df3"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def transplant(base, patch, file_index):
    sections = re.split(r"(?m)^diff --git ", patch)[1:]
    if len(sections) != 2:
        raise ValueError("Expected the two-file reviewed RPC delta")
    section = sections[file_index]
    if not section.startswith(f"a/{FILES[file_index]} b/{FILES[file_index]}\n"):
        raise ValueError("Unexpected patch file")
    hunks = re.split(r"(?m)^@@[^\n]*\n", section)[1:]
    if len(hunks) != (8 if file_index == 0 else 6):
        raise ValueError("Unexpected RPC hunk count")
    for index, hunk in enumerate(hunks):
        lines = hunk.splitlines(keepends=True)
        old = "".join(s[1:] for s in lines if s[:1] in (" ", "-"))
        new = "".join(s[1:] for s in lines if s[:1] in (" ", "+"))
        if file_index == 0 and index == 4:
            # The old runtime retains its original sim release controller.
            old = "        self.delivered_plan_callback = delivered_plan_callback\n"
            new = old + (
                "        self.policy_delivery_clock = policy_delivery_clock\n"
                "        self._replan_clock = time.perf_counter if replan_clock is None else replan_clock\n"
            )
        elif file_index == 1 and index == 4:
            # Do not import unrelated newer metadata/profile changes.
            old = '                        "inference_delivery_clock": "native inference latency plus explicitly configured response delay; RPC overhead is separately logged",\n'
            new = (
                '                        "policy_delivery_clock": args.policy_delivery_clock,\n'
                '                        "policy_call_wall_measurement": "Complete synchronous policy.replan client call; excludes observation callbacks and runner audit bookkeeping"\n'
                '                        if args.policy_delivery_clock == "rpc_wall"\n'
                "                        else None,\n"
                '                        "inference_delivery_clock": "measured complete client policy call plus configured response delay; native Plan.latency_s and action grid preserved"\n'
                '                        if args.policy_delivery_clock == "rpc_wall"\n'
                '                        else "native inference latency plus explicitly configured response delay; RPC overhead is separately logged",\n'
            )
        elif file_index == 1 and index == 5:
            old = '        "policy_stop_reason": adapter.stopped_reason if adapter else None,\n'
            new = (
                old
                + '        "policy_delivery_clock": args.policy_delivery_clock if adapter else None,\n'
            )
        base = replace_once(base, old, new)
    return base


class NativePath(ast.NodeTransformer):
    """Project only reviewed RPC-only syntax away; preserve all old runtime AST."""

    def visit_Import(self, node):
        if len(node.names) == 1 and node.names[0].name == "time":
            return None
        return node

    def visit_ClassDef(self, node):
        if node.name == "PolicyTimingError":
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "_read_replan_clock":
            return None
        if node.name == "__init__":
            kept = [
                (a, d)
                for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if a.arg not in ("policy_delivery_clock", "replan_clock")
            ]
            node.args.kwonlyargs = [a for a, _ in kept]
            node.args.kw_defaults = [d for _, d in kept]
        node = self.generic_visit(node)
        if node.name == "arguments":
            for i, item in enumerate(node.body):
                if (
                    isinstance(item, ast.Assign)
                    and ast.unparse(item) == "args = p.parse_args()"
                ):
                    if not (
                        i + 1 == len(node.body) - 1
                        and ast.unparse(node.body[i + 1]) == "return args"
                    ):
                        raise AssertionError("Unexpected argument-parser delta")
                    node.body[i:] = [ast.Return(value=item.value)]
                    break
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # docstrings do not alter either numerical path
        if (
            isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "--policy-delivery-clock"
        ):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        targets = [ast.unparse(n) for n in node.targets]
        if any(
            n in ("rpc_wall", "self.policy_delivery_clock", "self._replan_clock")
            for n in targets
        ):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        expression = ast.unparse(node.test)
        if (
            any(
                word in expression
                for word in ("policy_delivery_clock", "replan_clock", "rpc_wall")
            )
            or expression == "getattr(error, 'infrastructure_invalid', False)"
        ):
            return [self.visit(n) for n in node.orelse]
        return self.generic_visit(node)

    def visit_IfExp(self, node):
        expression = ast.unparse(node.test)
        if "policy_delivery_clock" in expression:
            return self.visit(node.orelse)
        if expression == "rpc_wall is None":
            return self.visit(node.body)
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords if k.arg != "policy_delivery_clock"]
        return self.generic_visit(node)

    def visit_Dict(self, node):
        kept = [
            (k, v)
            for k, v in zip(node.keys, node.values)
            if not (
                isinstance(k, ast.Constant)
                and k.value in ("policy_delivery_clock", "policy_call_wall_measurement")
            )
        ]
        node.keys = [k for k, _ in kept]
        node.values = [v for _, v in kept]
        return self.generic_visit(node)


def native_equal(base, overlay):
    trees = [NativePath().visit(copy.deepcopy(ast.parse(s))) for s in (base, overlay)]
    return ast.dump(trees[0], include_attributes=False) == ast.dump(
        trees[1], include_attributes=False
    )


def inventory(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() or args.audit_dir.exists():
        raise FileExistsError("Preserve existing source/audit; no overwrite")
    patch_path = args.inputs / "rpc_donor.patch"
    if sha(patch_path) != PATCH_PIN:
        raise ValueError("RPC patch differs from the reviewed commit")
    patch = patch_path.read_text()
    overlays, bases = {}, {}
    for i, name in enumerate(FILES):
        donor = args.inputs / (
            "donor_policy_adapter.py" if i == 0 else "donor_run_waffles.py"
        )
        if sha(args.base / name) != BASE_PINS[i] or sha(donor) != DONOR_PINS[i]:
            raise ValueError("Unreviewed base or donor")
        if any(
            (args.base / Path(*Path(name).parts[:j])).is_symlink()
            for j in range(1, len(Path(name).parts) + 1)
        ):
            raise ValueError("Refusing mutation through inherited symlink")
        bases[name] = (args.base / name).read_text()
        overlays[name] = transplant(bases[name], patch, i)
        if not native_equal(bases[name], overlays[name]):
            raise AssertionError(f"Default-native projected AST differs: {name}")
        compile(overlays[name], name, "exec")
    if sha(args.base / "phantom/drivers/servo_limiter.py") != HELPER_PIN:
        raise ValueError("Limiter helper changed")
    before = inventory(args.base)
    shutil.copytree(args.base, args.out, symlinks=True)
    for name, text in overlays.items():
        (args.out / name).write_text(text)
    after = inventory(args.out)
    changed = sorted(
        k for k in set(before) | set(after) if before.get(k) != after.get(k)
    )
    if changed != list(FILES) or inventory(args.base) != before:
        raise AssertionError("Unexpected source change")
    manifest = {
        "schema_version": 1,
        "status": "prepared_source_audited_no_GPU_or_policy_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_source": str(args.base),
        "output_source": str(args.out),
        "allowed_changed_files": changed,
        "default_native_projected_AST_equal": True,
        "base_unchanged_after_assembly": True,
        "base_sha256": before,
        "output_sha256": after,
        "source_digest": hashlib.sha256(
            json.dumps(after, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "donor_commit": "5c2f76b",
        "donor_sha256": dict(zip(FILES, DONOR_PINS)),
        "patch_sha256": sha(patch_path),
        "limiter_helper_unchanged_sha256": HELPER_PIN,
        "profile_changes": [
            "Opt-in full synchronous policy.replan duration for delivery only",
            "Native action grid, reported latency, CPK token and request-time feedback retained",
        ],
        "new_default_metadata_only": [
            "policy_delivery_clock=native",
            "policy_call_wall_measurement=null",
        ],
        "remaining_profile": "Original minimal gel-v2 + FINISH, historical request-time veto, pad-only wrist proxy, unchanged default-off shared limiter",
        "default_numerical_trajectory_not_tested_in_Isaac": True,
    }
    args.audit_dir.mkdir(parents=True)
    (args.audit_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for name in FILES:
        (args.audit_dir / (Path(name).stem + ".patch")).write_text(
            "".join(
                difflib.unified_diff(
                    bases[name].splitlines(keepends=True),
                    overlays[name].splitlines(keepends=True),
                    fromfile="limiter_v6/" + name,
                    tofile="rpc_v7/" + name,
                )
            )
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "changed": changed,
                "source_digest": manifest["source_digest"],
                "output_file_sha256": {k: after[k] for k in FILES},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
