#!/usr/bin/env python3
"""CPU-only assembly of one matched v1 latency diagnostic; never launches it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(path):
    return {
        str(p.relative_to(path)): sha(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    prep, base = args.preparation.resolve(), args.base.resolve()
    archived = base / "source_teacher_pick_place_v1"
    source = base / "source_teacher_anchor_latency_v3"
    output = base / "runs/teacher_success_anchor_v3/latency_schedule_seed4242"
    if source.exists() or output.exists():
        raise FileExistsError("Prepared source and trial output must both be new")
    dependencies = json.loads((prep / "anchor_launch_dependencies.json").read_text())
    expected = dependencies["launch_dependencies"]
    checks = {}
    for flag, record in expected.items():
        if record["sha256"] is not None:
            checks[flag] = sha(record["path"]) == record["sha256"]
    if not all(checks.values()):
        raise ValueError(f"Original launch dependency hash mismatch: {checks}")
    helper = prep / "prepare_teacher_anchor_diagnostic.py"
    spec = importlib.util.spec_from_file_location("anchor_overlay_builder", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.prepare(
        archived, prep / "overlay", "latency_schedule", prep / "donor"
    )
    before = snapshot(archived)
    shutil.copytree(
        archived,
        source,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", ".git"),
    )
    for relative in manifest["output_sha256"]:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prep / "overlay" / relative, destination)
    copied = snapshot(source)
    differences = {
        name: {"base": before.get(name), "prepared": copied.get(name)}
        for name in sorted(set(before) | set(copied))
        if before.get(name) != copied.get(name)
    }
    if (
        set(differences) != set(manifest["output_sha256"])
        or snapshot(archived) != before
    ):
        raise RuntimeError("Unexpected prepared source difference or archive mutation")
    original = list(dependencies["original_command"])
    original_dir = Path(original[original.index("--output") + 1])
    trace = original_dir / "planner_trace.json"
    # CPU validation of all38 recorded rows, without importing policy/runtime.
    schedule_spec = importlib.util.spec_from_file_location(
        "anchor_schedule", source / "tools/sim/anchor_inference_timing.py"
    )
    schedule_module = importlib.util.module_from_spec(schedule_spec)
    schedule_spec.loader.exec_module(schedule_module)
    schedule = schedule_module.RecordedInferenceLatencies(trace)
    if len(schedule.rows) != 38:
        raise ValueError("Expected exactly38 original completed inference records")
    live = Path("/home/physicalai/phantom-icra-2027/phantom")
    port = 7799
    original[0] = str(source / "tools/sim/launch_waffles.sh")
    for flag, value in (
        ("--output", output / "rollout"),
        ("--policy-server", f"127.0.0.1:{port}"),
        ("--duration", "33.38"),
    ):
        original[original.index(flag) + 1] = str(value)
    original += ["--anchor-latency-trace", str(trace)]
    hw = expected["--hardware-config"]
    checkpoint = dependencies["checkpoints"]["fta1500"]
    server = [
        str(live / ".venv/bin/python"),
        str(source / "tools/sim/policy_server.py"),
        "--repo",
        str(live),
        "--ckpt",
        checkpoint["path"],
        "--expected-sha256",
        checkpoint["expected_sha256"],
        "--system",
        "teacher",
        "--port",
        str(port),
        "--out",
        str(output / "server"),
        "--hardware",
        hw["path"],
        "--nfe",
        "1",
        "--guidance",
        "1",
        "--k-seeds",
        "4",
        "--task-text",
        "waffles",
        "--parity-fixes",
        "--persistent-noise",
    ]
    inputs = {
        flag.lstrip("-").replace("-", "_"): value
        for flag, value in expected.items()
        if value["sha256"] is not None
    }
    inputs["hardware"] = hw
    inputs["historical_latency_trace"] = {"path": str(trace), "sha256": sha(trace)}
    plan = {
        "schema_version": 1,
        "status": "prepared_not_launched",
        "variant": "adapter_latency_schedule_only",
        "seed": 4242,
        "output": str(output),
        "source": str(source),
        "live_repo": str(live),
        "port": port,
        "commands": {"server": server, "rollout": original},
        "inputs": inputs,
        "environment": {
            "ISAAC_SIM_ROOT": "/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "source_files_sha256": copied,
        "coverage": "manifold_patch",
        "veto_version": "fd4a032",
        "checkpoint_weights": "EMA",
        "settings": {
            "nfe": 1,
            "guidance": 1.0,
            "k_seeds": 4,
            "parity_fixes": True,
            "persistent_noise": True,
            "task_text": "waffles",
            "drop_video": False,
            "close_p": 0.5,
        },
        "schedule": schedule.metadata,
        "declared_horizon_s": 33.38,
        "historical_actual_final_sample_s": 33.376,
        "horizon_note": "Frozen loop records a sample one physics tick before requested horizon; this reproduces the reference final33.376s observation, including original post-stop tail.",
        "limitation": "Only adapter delivery/action-grid latency is fixed. Native K4 candidate selection still uses measured current compute latency; next CPK offset receives overridden previous latency. All38 values used in order; fail before request39 rather than invent a delay.",
        "interpretation": "Diagnostic only, excluded from teacher selection and confirmation denominators. Source, model observations, raw policy predictions and current safety remain live; no object oracle and no FINISH.",
        "hardware_launcher_used": False,
    }
    (prep / "latency_diagnostic_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    audit = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_not_launched": True,
        "base_source": str(archived),
        "prepared_source": str(source),
        "all_original_source_files_unchanged": True,
        "base_file_count": len(before),
        "prepared_file_count": len(copied),
        "differences": differences,
        "original_input_checks": checks,
        "overlay": manifest,
        "preparation_helper_sha256": sha(__file__),
        "overlay_helper_sha256": sha(helper),
        "plan_sha256": sha(prep / "latency_diagnostic_plan.json"),
    }
    (prep / "latency_source_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                "prepared_source": str(source),
                "output": str(output),
                "files_different": len(differences),
                "plan_sha256": audit["plan_sha256"],
                "hardware_or_gpu_launched": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
