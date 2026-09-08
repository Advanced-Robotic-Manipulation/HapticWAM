#!/usr/bin/env python3
"""Prepare separate, hash-checked sensor overlays on the immutable v1 source.

No simulator/model/hardware launch. Unchanged files are symlinked to the base;
the launcher is copied unchanged so its repository root resolves to the overlay.
Use PYTHONDONTWRITEBYTECODE=1 when executing an overlay. Never edit its symlinks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

RUNNER = "tools/sim/run_waffles.py"
LAUNCHER = "tools/sim/launch_waffles.sh"
BASE_RUNNER_SHA = "355c6a76f00c55721ffad8744337863b6962b4777bb7df27ce5ff400dcba91a6"
MODULE_PINS = {
    "tools/sim/gel_contact.py": "d7f4cb1d4606aa0ced6d360be6073498e5505460248d75cf6d04c7cf52853d6e",
    "tools/sim/gripper_wrist.py": "819780c79bfbabb76c4f2b913b5774437255bd31b69e41eb9e05e026ab1c2846",
    "tools/sim/robot_environment_contacts.py": "7585e9474b681f460300d8b94b9c8bc25529dd45a85683aff83d5de7b87c47d0",
}
VARIANTS = {
    "gel_v2": {
        "modules": ("tools/sim/gel_contact.py",),
        "flags": ("--gel-contact-coverage", "manifold_patch_v2"),
    },
    "gripper_wrist": {
        "modules": (
            "tools/sim/gripper_wrist.py",
            "tools/sim/robot_environment_contacts.py",
        ),
        "flags": ("--wrist", "gripper_contact_proxy"),
    },
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"frozen runner patch anchor is not unique: {old[:80]!r}")
    return text.replace(old, new, 1)


def patch_runner(raw, variant):
    """Exact v1 input required; changes only sensor setup/readout/diagnostics."""
    if hashlib.sha256(raw).hexdigest() != BASE_RUNNER_SHA:
        raise ValueError("base runner hash differs from frozen successful v1")
    if variant not in VARIANTS:
        raise ValueError("choose one separate sensor variant")
    text = raw.decode()
    if variant == "gel_v2":
        text = replace_once(
            text,
            'choices=["point", "manifold_patch"],',
            'choices=["point", "manifold_patch", "manifold_patch_v2"],',
        )
        ast.parse(text)
        return text.encode()
    text = replace_once(
        text,
        'choices=["contact_proxy", "zero_ablation"],',
        'choices=["contact_proxy", "zero_ablation", "gripper_contact_proxy"],',
    )
    text = replace_once(
        text,
        "    gel_views = None\n",
        """    gripper_wrist = None
    if args.wrist == "gripper_contact_proxy":
        from tools.sim.gripper_wrist import GripperContactWrist

        gripper_wrist = GripperContactWrist(
            str(housing.GetPath()), paths["pad_paths"], rigid_prim_cls=RigidPrim
        )
    gel_views = None
""",
    )
    text = replace_once(
        text,
        "    tool_body.initialize()\n",
        "    if gripper_wrist is not None:\n        gripper_wrist.initialize()\n    tool_body.initialize()\n",
    )
    text = replace_once(
        text,
        "    wrist_bias = np.zeros(6)\n",
        """    wrist_bias = np.zeros(6)
    wrist_contact_file = None
    wrist_value = None
    wrist_sample_t = None
    next_wrist_t = 0.0
""",
    )
    text = replace_once(
        text,
        "        value = wrist_bias.copy()\n",
        """        if gripper_wrist is not None:
            if wrist_value is None:
                raise RuntimeError("Gripper wrist input has not been sampled")
            return wrist_value.copy()
        value = wrist_bias.copy()
""",
    )
    text = replace_once(
        text,
        "    def tactile_proxy(t, forces):\n",
        """    def sample_gripper_wrist(t, tcp):
        nonlocal wrist_value, wrist_sample_t, next_wrist_t
        wrist_value, record = gripper_wrist.sample(tcp, wrist_bias, dt)
        wrist_sample_t = float(t)
        wrist_contact_file.write(json.dumps({"t": t, **record}) + "\\n")
        next_wrist_t = t + control_dt

    def tactile_proxy(t, forces):
""",
    )
    text = replace_once(
        text,
        """    start = time.monotonic()
    try:
        if args.mode == "policy":
""",
        """    start = time.monotonic()
    try:
        if gripper_wrist is not None:
            from phantom.config.hardware import load_hardware

            wrist_hw = load_hardware(args.hardware_config, quiet=True)
            control_dt = 1 / wrist_hw.control.executor_rate_hz
            if dt > control_dt + 1e-9:
                raise ValueError("Physics dt exceeds wrist sample/control period")
            if "native_arm_ft" in data:
                ft_idx = int(np.argmin(np.abs(data["native_arm_ft_t"])))
                wrist_bias = np.asarray(data["native_arm_ft"][ft_idx], dtype=float)
            if initial_state is not None:
                wrist_bias = initial_wrist.copy()
            wrist_contact_file = (args.output / "wrist_contact_trace.jsonl").open("w")
        if args.mode == "policy":
""",
    )
    text = replace_once(
        text,
        '                        "wrist_model": args.wrist,\n',
        """                        "wrist_model": args.wrist,
                        "wrist_proxy_metadata": gripper_wrist.metadata()
                        if gripper_wrist is not None else None,
                        "wrist_sampling_rate_hz": 1 / control_dt
                        if gripper_wrist is not None else None,
""",
    )
    text = replace_once(
        text,
        """            tcp = tcp_measured(qactual)
            if args.mode in ("replay", "dynamics"):
""",
        """            tcp = tcp_measured(qactual)
            if (
                gripper_wrist is not None
                and (wrist_value is None or args.mode != "policy" or stop_after_step)
                and t + 1e-9 >= next_wrist_t
            ):
                sample_gripper_wrist(t, tcp)
            if args.mode in ("replay", "dynamics"):
""",
    )
    text = replace_once(
        text,
        "                forces = measured_pad_forces()\n",
        """                forces = measured_pad_forces()
                if gripper_wrist is not None:
                    sample_gripper_wrist(t, tcp)
                measured_wrist = wrist_proxy(tcp, forces)
""",
    )
    text = replace_once(
        text,
        "                    wrist_ft=wrist_proxy(tcp, forces),\n",
        "                    wrist_ft=measured_wrist,\n",
    )
    text = replace_once(
        text,
        '                    "measured_wrist_ft": wrist_proxy(tcp, forces),\n',
        """                    "measured_wrist_ft": measured_wrist,
                    "wrist_capture_t": wrist_sample_t
                    if gripper_wrist is not None else t,
""",
    )
    text = replace_once(
        text,
        "    finally:\n        writer.release()\n",
        "    finally:\n        if wrist_contact_file is not None:\n            wrist_contact_file.close()\n        writer.release()\n",
    )
    text = replace_once(
        text,
        '        "wrist_model": args.wrist if args.mode == "policy" else None,\n',
        """        "wrist_model": args.wrist
        if args.mode == "policy" or gripper_wrist is not None else None,
        "wrist_proxy_metadata": gripper_wrist.metadata()
        if gripper_wrist is not None else None,
        "wrist_sampling_rate_hz": 1 / control_dt
        if gripper_wrist is not None else None,
        "initial_recorded_wrist_bias": wrist_bias.tolist()
        if gripper_wrist is not None else None,
""",
    )
    ast.parse(text)
    return text.encode()


def prepare_overlay(base, output, module_source, frozen_inputs, variant):
    base, output, module_source = (
        Path(p).resolve() for p in (base, output, module_source)
    )
    if output.exists() or output == base or base in output.parents:
        raise ValueError("output must be new and outside the immutable base source")
    if variant not in VARIANTS:
        raise ValueError("choose one separate sensor variant")
    frozen = json.loads(Path(frozen_inputs).read_text())
    if Path(frozen["source_root"]).resolve() != base:
        raise ValueError("frozen manifest identifies a different base source")
    expected = frozen["source_sha256"]
    for name, digest in expected.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not (base / relative).is_file()
        ):
            raise ValueError(f"invalid or missing frozen source path: {name}")
        if sha(base / name) != digest:
            raise ValueError(f"frozen source hash mismatch: {name}")
    if expected.get(RUNNER) != BASE_RUNNER_SHA:
        raise ValueError("manifest does not pin the successful v1 runner")
    modules = VARIANTS[variant]["modules"]
    replacements = {RUNNER: patch_runner((base / RUNNER).read_bytes(), variant)}
    for name in modules:
        if sha(module_source / name) != MODULE_PINS[name]:
            raise ValueError(f"sensor module hash mismatch: {name}")
        replacements[name] = (module_source / name).read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent)
    )
    unchanged, changed = {}, {}
    try:
        for parent, dirs, files in os.walk(base):
            dirs[:] = [
                d
                for d in dirs
                if d not in ("__pycache__", ".git", ".pytest_cache", ".ruff_cache")
            ]
            for filename in files:
                original = Path(parent) / filename
                relative = original.relative_to(base)
                name = relative.as_posix()
                if name in replacements or original.suffix in (".pyc", ".pyo"):
                    continue
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if name == LAUNCHER:
                    shutil.copy2(original, target)
                else:
                    target.symlink_to(original)
                unchanged[name] = sha(original)
        for name, raw in replacements.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            if (base / name).exists():
                target.chmod((base / name).stat().st_mode & 0o777)
            changed[name] = {
                "base_sha256": sha(base / name) if (base / name).exists() else None,
                "overlay_sha256": sha(target),
            }
        manifest = {
            "schema_version": 1,
            "variant": variant,
            "base_source": str(base),
            "output_source": str(output),
            "frozen_manifest_sha256": sha(frozen_inputs),
            "verified_frozen_files": len(expected),
            "required_rollout_flags": list(VARIANTS[variant]["flags"]),
            "required_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            "changed_files": changed,
            "unchanged_files_sha256": unchanged,
            "physics_controller_or_profile_changes": False,
            "execution": "No simulator or policy executed during preparation. Defaults preserve v1; explicitly select this overlay's one sensor flag.",
            "limits": "Independent sensor intervention only. Added wrist contact reporting requires a separate physics invariance preflight. Gel v2 remains an uncalibrated uniform-patch estimate; wrist normal-contact model omits friction/inertia/gravity/proximal/self contacts.",
            "symlink_warning": "Unchanged files point to immutable base. Never edit overlay symlinks; verify these hashes before each diagnostic and suppress bytecode writes.",
        }
        (temporary / "sensor_overlay_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--module-source", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--frozen-inputs", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    manifest = prepare_overlay(
        args.base_source,
        args.output,
        args.module_source,
        args.frozen_inputs,
        args.variant,
    )
    print(
        json.dumps(
            {
                "variant": manifest["variant"],
                "output_source": manifest["output_source"],
                "changed_files": manifest["changed_files"],
                "required_rollout_flags": manifest["required_rollout_flags"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
