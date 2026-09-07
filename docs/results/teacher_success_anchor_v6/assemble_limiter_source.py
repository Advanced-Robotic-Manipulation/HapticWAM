#!/usr/bin/env python3
"""Create an independent minimal-v5 copy with only the opt-in limiter hook.

CPU/source preparation only. No simulator, policy, device, or controller launch.
The supplied donor runner is never substituted wholesale for the v5 runner.
"""

import argparse
import ast
import copy
import difflib
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PINS = {
    "base_runner": "b5184eb802ed2d99e1ac2f678fe485b810c72d878b0a52f330d4fec9423ca923",
    "donor_runner": "91a578e060f1596cecee03aff271bad1ebe5da33b6a399134829826050e43a5f",
    "helper": "41c0c6beebd805dda4d9bf7c66e507491eb1121cdb5cf70a701ce845b37e027f",
    "native_audited_base": "b24cbdb9c21b6a5b113d59145e08b716655d9036f9bc2bc371e614f4879587a0",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact source anchor: {old[:90]!r}")
    return text.replace(old, new, 1)


def block(source, node):
    return "".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])


def transplant(base, donor):
    tree = ast.parse(donor)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    args = functions["arguments"]
    cli = next(
        n
        for n in args.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and n.value.args
        and isinstance(n.value.args[0], ast.Constant)
        and n.value.args[0].value == "--servo-reach-limiter"
    )
    old = '    p.add_argument("--max-play-steps", type=int, default=10)\n'
    base = replace_once(base, old, old + block(donor, cli))
    helper = block(donor, functions["report_servo_limiter_execution"])
    base = replace_once(
        base,
        "def apply_episode_overrides(",
        helper + "\n\ndef apply_episode_overrides(",
    )
    guard = functions["main"].body[1]
    assert isinstance(guard, ast.If) and "servo_reach_limiter" in ast.unparse(
        guard.test
    )
    old = "def main():\n    args = arguments()\n"
    base = replace_once(base, old, old + block(donor, guard))
    base = replace_once(
        base,
        "    hw = None\n",
        "    hw = None\n    servo_reach_limits = None\n    servo_limiter_rejects = 0\n",
    )
    run = functions["run"]
    enabled = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "args.servo_reach_limiter"
    )
    old = "            hw = apply_episode_overrides(hw, overrides)\n"
    base = replace_once(base, old, old + block(donor, enabled))
    begin = donor.index('                        "planner_stall_watchdog": True,\n')
    begin += len('                        "planner_stall_watchdog": True,\n')
    end = donor.index('                        "terminal_veto":', begin)
    metadata = donor[begin:end]
    assert "servo_reach_limiter" in metadata
    old = '                        "planner_stall_watchdog": True,\n'
    base = replace_once(base, old, old + metadata)
    limited = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "servo_reach_limits is not None"
    )
    # The selected elif node owns the original else subtree; copy only its body.
    new_branch = "".join(
        donor.splitlines(keepends=True)[
            limited.lineno - 1 : limited.body[-1].end_lineno
        ]
    )
    assert new_branch.lstrip().startswith("elif servo_reach_limits")
    old = "                    terminal_until = min(t + 2.0, duration)\n                else:\n"
    base = replace_once(
        base,
        old,
        "                    terminal_until = min(t + 2.0, duration)\n"
        + new_branch
        + "                else:\n",
    )
    begin = donor.index('        "policy_completion_is_task_success": False,\n')
    begin += len('        "policy_completion_is_task_success": False,\n')
    end = donor.index('        "post_stop_observation_s":', begin)
    metadata = donor[begin:end]
    assert "servo_reach_limiter" in metadata
    old = '        "policy_stop_reason": adapter.stopped_reason if adapter else None,\n'
    return replace_once(base, old, old + metadata)


class DisabledPath(ast.NodeTransformer):
    """Remove only declared disabled hooks; compare every remaining AST node."""

    def visit_FunctionDef(self, node):
        if node.name == "report_servo_limiter_execution":
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if (
            isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "--servo-reach-limiter"
        ):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(
            isinstance(t, ast.Name)
            and t.id in ("servo_reach_limits", "servo_limiter_rejects")
            for t in node.targets
        ):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        expression = ast.unparse(node.test)
        if expression == "servo_reach_limits is not None":
            return [self.visit(n) for n in node.orelse]
        if "args.servo_reach_limiter" in expression:
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        values = [
            (k, v)
            for k, v in zip(node.keys, node.values)
            if not (
                k is None
                and isinstance(v, ast.IfExp)
                and ast.unparse(v.test) == "servo_reach_limits is not None"
            )
        ]
        node.keys = [k for k, _ in values]
        node.values = [v for _, v in values]
        return self.generic_visit(node)


def disabled_equal(base, overlay):
    projected = DisabledPath().visit(copy.deepcopy(ast.parse(overlay)))
    return ast.dump(ast.parse(base), include_attributes=False) == ast.dump(
        projected, include_attributes=False
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
    p.add_argument("--donor-runner", type=Path, required=True)
    p.add_argument("--helper", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    args = p.parse_args()
    args.base, args.out = args.base.resolve(), args.out.resolve()
    base_runner = args.base / "tools/sim/run_waffles.py"
    for key, path in (
        ("base_runner", base_runner),
        ("donor_runner", args.donor_runner),
        ("helper", args.helper),
    ):
        if sha(path) != PINS[key]:
            raise ValueError(f"Unreviewed {key}: {path}")
    if args.out.exists() or args.audit_dir.exists():
        raise FileExistsError("Preserve existing source/audit; no overwrite or resume")
    for relative in ("tools/sim/run_waffles.py", "phantom/drivers/servo_limiter.py"):
        components = Path(relative).parts
        if any(
            (args.base.joinpath(*components[:i])).is_symlink()
            for i in range(1, len(components) + 1)
        ):
            raise ValueError(
                f"Refusing replacement through an inherited symlink: {relative}"
            )
    base, donor = base_runner.read_text(), args.donor_runner.read_text()
    overlay = transplant(base, donor)
    if not disabled_equal(base, overlay):
        raise AssertionError("Disabled command path AST differs")
    compile(overlay, str(args.out / "tools/sim/run_waffles.py"), "exec")
    before = inventory(args.base)
    # No hardlinks: modifications must never reach the immutable v5 source.
    shutil.copytree(args.base, args.out, symlinks=True)
    (args.out / "tools/sim/run_waffles.py").write_text(overlay)
    shutil.copy2(args.helper, args.out / "phantom/drivers/servo_limiter.py")
    after = inventory(args.out)
    changed = sorted(
        k for k in set(before) | set(after) if before.get(k) != after.get(k)
    )
    if changed != ["phantom/drivers/servo_limiter.py", "tools/sim/run_waffles.py"]:
        raise AssertionError(f"Unexpected source delta: {changed}")
    if inventory(args.base) != before:
        raise AssertionError("Immutable base changed during source preparation")
    native = "phantom/drivers/real/ur.py"
    manifest = {
        "schema_version": 1,
        "status": "prepared_CPU_audited_no_simulator_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_source": str(args.base),
        "output_source": str(args.out),
        "allowed_changed_files": changed,
        "default_off_AST_equal": True,
        "base_unchanged_after_assembly": True,
        "base_sha256": before,
        "output_sha256": after,
        "source_digest": hashlib.sha256(
            json.dumps(after, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "donor_sha256": {"runner": sha(args.donor_runner), "helper": sha(args.helper)},
        "native_driver": {
            "base_sha256": before[native],
            "output_sha256": after[native],
            "audited_extraction_base_sha256": PINS["native_audited_base"],
            "delegation_backported": False,
            "reason": "Frozen native UR base differs from the b24 extraction audit; preserve it unchanged. Simulator calls the shared pure helper directly.",
        },
        "settings": {
            "flag": "--servo-reach-limiter",
            "default": False,
            "elbow_min_rad": 0.40,
            "command_joint_speed_max_rad_s": 1.0,
            "consecutive_reject_limit": 25,
            "rejection_stop": "servo_limiter_stall",
            "ordinary_stop_observation_s": 2.0,
            "actual_tracking_guaranteed": False,
        },
        "hardware_or_GPU_actions": False,
    }
    args.audit_dir.mkdir(parents=True)
    (args.audit_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.audit_dir / "runner.patch").write_text(
        "".join(
            difflib.unified_diff(
                base.splitlines(keepends=True),
                overlay.splitlines(keepends=True),
                fromfile="minimal_v5/tools/sim/run_waffles.py",
                tofile="limiter_v6/tools/sim/run_waffles.py",
            )
        )
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "changed": changed,
                "runner_sha256": after["tools/sim/run_waffles.py"],
                "source_digest": manifest["source_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
