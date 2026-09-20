#!/usr/bin/env python3
"""Prepare/run two separate latest-real pickup replays; never launch hardware.

--prepare is CPU-only and freezes the standard replay/tactile exports.
--execute refuses to start until the frozen60-case policy campaign is complete.
No candidate scene changes, policy inference, or placement claim is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

R4_SHA = "deeca9e635bf7b404d483ab3edb633e35960566783a6e1309f0fdad25f6ee3df"
EPISODE = "ep_teacher_waffles_1788867073_000"
SOURCE_FILES = [
    "tools/sim/prepare_waffles.py", "tools/sim/tactile_panels.py",
    "tools/sim/compare_replay.py", "tools/sim/evaluate_pick_place.py",
    "tools/sim/launch_waffles.sh", "tools/sim/run_waffles.py",
    "tools/sim/gel_contact.py", "tools/sim/object_support.py",
    "tools/sim/robot_environment_contacts.py", "tools/sim/gripper_wrist.py",
    "phantom/sim/scene.py", "phantom/sim/kinematics.py", "phantom/sim/command_replay.py", "phantom/sim/camera.py",
    "phantom/sim/gripper_visual.py", "phantom/sim/geometry.py",
    "phantom/data/episode_store.py", "phantom/data/schema.py",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tree_hashes(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def prepare(args):
    import numpy as np
    from tools.sim.prepare_waffles import export_episode
    from tools.sim.tactile_panels import export_tactile_sidecar

    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        validate(args, protocol)
        print(json.dumps({"status": "reused_frozen_preparation", "protocol": str(protocol_path)}))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Partial export protected; inspect before replacing: {args.out}")
    assert args.episode.name == EPISODE
    config = args.source / "configs/sim/waffles_d435_factory_20260908_r4.json"
    assert sha(config) == R4_SHA
    before = tree_hashes(args.episode)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = export_episode(args.episode, args.out / "reference", "heldout", 15.0, None)
    tactile = export_tactile_sidecar(args.episode, args.out / "reference", args.out / "tactile")
    after = tree_hashes(args.episode)
    if before != after:
        raise RuntimeError("Source recording changed during read-only export")
    with np.load(args.out / "reference/replay.npz") as data:
        arm_t0 = float(data["t0_master"] + data["native_arm_tcp_pose_t"][0])
        delta = arm_t0 - float(data["t0_master"])
        time_origin = {"export_t0_master_s": float(data["t0_master"]), "original_arm_t0_master_s": arm_t0, "original_elapsed_to_export_elapsed_offset_s": delta}
    gates = json.loads((args.study_root / "replay_protocol.json").read_text())["gates"]
    commands = {}
    for mode in ["replay", "dynamics"]:
        directory = args.out / mode
        commands[mode] = [str(args.source / "tools/sim/launch_waffles.sh"),
            "--episode", str(args.out / "reference"), "--config", str(config),
            "--output", str(directory), "--mode", mode,
            "--robot-usd", str(args.robot_usd), "--skip-stage-export",
            "--record-packet-support", "--record-robot-environment-contacts",
            "--record-gel-contacts", "--gel-contact-coverage", "manifold_patch_v2",
            "--wrist", "gripper_contact_proxy"]
    protocol = {
        "schema_version": 1, "status": "frozen_prepared_not_executed",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Unassisted latest-real pickup/carry reproduction diagnostic; not placement success, not policy inference, not a test of the bounded native reach hotfix",
        "episode": str(args.episode), "source": str(args.source), "output": str(args.out),
        "source_episode_files_sha256": before, "source_recording_unchanged": True,
        "scene": str(config), "scene_sha256": R4_SHA,
        "runner_sha256": sha(Path(__file__)), "source_files_sha256": {f: sha(args.source / f) for f in SOURCE_FILES},
        "robot_usd": str(args.robot_usd), "robot_usd_sha256": sha(args.robot_usd),
        "export_files_sha256": {str(p.relative_to(args.out)): sha(p) for sub in ["reference", "tactile"] for p in sorted((args.out / sub).rglob("*")) if p.is_file()},
        "duration_s": manifest["duration_s"], "frames": manifest["frames"], "time_origin": time_origin,
        "kinematic_gates": gates,
        "real_adjudication": {
            "source": "latest_real_failure_audit.md/.json; scene RGB plus tactile/gripper/measured state inspection",
            "external_assistance_observed": False,
            "observed_pick_and_carry": True, "observed_place": False,
            "approximate_acquisition_original_elapsed_interval_s": [12.76, 12.87],
            "confirmed_lift_original_frame_elapsed_s": 15.97490904503502,
            "wrist_extension_stop_original_elapsed_s": 18.877,
            "diagnostic_carry_original_elapsed_interval_s": [16.0, 18.75],
            "reference_object_3d_pose": None,
        },
        "physical_diagnostic_thresholds": {
            "bilateral_packet_normal_n": .05, "bilateral_onset_hold_s": .20,
            "lift_onset_m": .02, "lift_onset_hold_s": .15, "minimum_peak_lift_m": .10,
            "retention_tcp_relative_p95_m": .020, "retention_tcp_relative_max_m": .050,
            "bilateral_carry_fraction_min": .85, "maximum_bilateral_carry_gap_s": .30,
            "note": "Existing replay thresholds; observed real 3D packet pose unavailable. No success event derived from a controller flag. These object checks do not alter kinematic readiness gates.",
        },
        "scene_pose_note": "Unchanged R4 reset: packet[-.398711456,-.273325117,.0355]m yaw.25955; tableZ-.0125 and mat top-.0095. Latest RGB table repeats previous image at0.438pxRMS, but camera-to-UR wrist residual5.714pxRMS remains. Latest packet pose is visually close, not independently measured or matched in3D; projection overlay must be reviewed. No per-episode object refit.",
        "tactile_note": "Native left/right infer_img plus available wrench/depth sidecars; comparison uses causal previous recorded tactile and labeled contact-force proxies. Gel-filtered packet contact is reported separately from body contact. No tactile optical transfer calibration claimed.",
        "execution_dependency": "All60 planned V10 policy trials completed and campaign runner finished; sequential owned replay processes only",
        "commands": commands,
    }
    write(protocol_path, protocol)
    print(json.dumps({"status": "prepared_not_executed", "protocol": str(protocol_path), "duration_s": manifest["duration_s"], "tactile_streams": list(tactile["streams"])}))


def validate(args, protocol):
    if Path(protocol["source"]) != args.source or Path(protocol["output"]) != args.out:
        raise ValueError("Frozen paths differ")
    if sha(Path(__file__)) != protocol["runner_sha256"]:
        raise ValueError("Replay runner changed after preparation")
    for relative, expected in protocol["source_files_sha256"].items():
        if sha(args.source / relative) != expected:
            raise ValueError(f"Frozen simulator source changed: {relative}")
    if sha(Path(protocol["scene"])) != R4_SHA or sha(args.robot_usd) != protocol["robot_usd_sha256"]:
        raise ValueError("Frozen scene or robot asset changed")
    for relative, expected in protocol["export_files_sha256"].items():
        if sha(args.out / relative) != expected:
            raise ValueError(f"Frozen export changed: {relative}")
    if tree_hashes(args.episode) != protocol["source_episode_files_sha256"]:
        raise ValueError("Source recording changed since preparation")


def baseline_finished(root):
    progress = json.loads((root / "campaign/progress.json").read_text())
    if progress.get("status") != "all_trials_completed":
        raise RuntimeError("Frozen policy campaign is not complete; replay GPU launch forbidden")
    trials = progress.get("trials", {})
    if len(trials) != 60 or any(t.get("status") != "completed" for t in trials.values()):
        raise RuntimeError("Need all60 completed policy outcomes before replay")
    if len(progress.get("analysis", {})) != 5 or any(r.get("status") != "completed" for r in progress["analysis"].values()):
        raise RuntimeError("Campaign analyses still incomplete")
    return progress


def diagnostics(args, protocol, mode):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tools.sim.compare_replay import state_metrics
    from tools.sim.evaluate_pick_place import sustained_onset, longest_gap

    directory = args.out / mode
    with np.load(args.out / "reference/replay.npz") as data:
        reference = {k: data[k] for k in data.files}
    with np.load(directory / "sim_trace.npz") as data:
        trace = {k: data[k] for k in data.files}
    state = state_metrics(reference, trace)
    g = protocol["kinematic_gates"]
    time = trace["t"]
    gates = {
        "joint_rmse": state["joint_error_rad"]["rmse_all"] <= g["replay_joint_rmse_rad" if mode == "replay" else "dynamics_joint_rmse_rad"],
        "tcp_rmse": state["tcp_translation"]["rmse"] / 1000 <= g["replay_tcp_rmse_m" if mode == "replay" else "dynamics_tcp_rmse_m"],
        "physics_clock": state["physics_clock_consistency"]["passed"] is True,
        "coverage": bool(time[0] <= .1 and time[-1] >= reference["t"][-1] - g["coverage_end_tolerance_s"]),
        "scene_hash": sha(directory / "effective_config.json") == R4_SHA,
    }
    if mode == "replay":
        gates["rotation_rmse"] = state["tcp_rotation_geodesic"]["rmse"] <= g["replay_rotation_rmse_deg"]
    cfg = json.loads((directory / "effective_config.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    gates["camera_readback"] = run.get("camera_projection", {}).get("readback_matches_config") is True
    obj = trace["waffle_position"]
    initial = np.median(obj[time <= min(1., time[-1])], axis=0)
    rise = obj[:, 2] - initial[2]
    thresholds = protocol["physical_diagnostic_thresholds"]
    normals = trace["pad_packet_normal_force"]
    bilateral = np.all(normals > thresholds["bilateral_packet_normal_n"], axis=1)
    lift = sustained_onset(time, rise > thresholds["lift_onset_m"], thresholds["lift_onset_hold_s"])
    body_onset = sustained_onset(time, bilateral, thresholds["bilateral_onset_hold_s"])
    offset = protocol["time_origin"]["original_elapsed_to_export_elapsed_offset_s"]
    carry_start, carry_end = np.array(protocol["real_adjudication"]["diagnostic_carry_original_elapsed_interval_s"]) + offset
    carry = (time >= carry_start) & (time <= carry_end)
    if carry.sum() < 3:
        raise ValueError("Replay does not cover frozen carry diagnostic interval")
    relative = Rotation.from_rotvec(trace["tcp"][carry, 3:]).inv().apply(obj[carry] - trace["tcp"][carry, :3])
    anchor = np.median(relative[:min(5, len(relative))], axis=0)
    slip = np.linalg.norm(relative - anchor, axis=1)
    gap = longest_gap(time, bilateral, float(carry_start), float(carry_end))
    gel_rows = json.loads((directory / "gel_contact_trace.json").read_text())
    gel_time = np.array([r["t"] for r in gel_rows])
    gel = np.array([[sum(c["gel_compression_n"] for c in p["per_filter_contacts"] if c["filter_path"] == "/World/Waffle") for p in r["per_pad"]] for r in gel_rows])
    gel_bilateral = np.all(gel > thresholds["bilateral_packet_normal_n"], axis=1)
    gel_onset = sustained_onset(gel_time, gel_bilateral, thresholds["bilateral_onset_hold_s"])
    object_gates = {
        "free_dynamic_object": run["object_dynamics"]["rigid_body_dynamic"] is True and run["object_dynamics"]["kinematic"] is False and run["object_dynamics"]["attachments"] == [] and run["object_dynamics"]["pose_writes_after_initialization"] == 0,
        "physical_lift_100mm": bool(lift is not None and rise.max() >= thresholds["minimum_peak_lift_m"]),
        "body_contact_acquisition": body_onset is not None,
        "body_contact_carry": bool(bilateral[carry].mean() >= thresholds["bilateral_carry_fraction_min"] and gap is not None and gap <= thresholds["maximum_bilateral_carry_gap_s"]),
        "retention": bool(np.quantile(slip, .95) <= thresholds["retention_tcp_relative_p95_m"] and slip.max() <= thresholds["retention_tcp_relative_max_m"]),
    }
    result = {
        "mode": mode, "kinematic_gates": gates, "kinematic_gates_passed": all(gates.values()), "state": state,
        "object_diagnostic_gates": object_gates,
        "physical_pick_carry_reproduced": bool(mode == "dynamics" and all(object_gates.values())),
        "placement_claim": False, "native_hotfix_validation_claim": False,
        "initial_packet_m": obj[0].tolist(), "initial_packet_offset_from_config_m": (obj[0] - cfg["waffle"]["center"]).tolist(),
        "maximum_lift_mm": float(1000 * rise.max()), "lift20mm_onset_s": lift,
        "body_bilateral_contact_onset_s": body_onset, "gel_bilateral_packet_contact_onset_s": gel_onset,
        "body_peak_normal_proxy_n": normals.max(0).tolist(), "gel_peak_packet_compression_proxy_n": gel.max(0).tolist(),
        "gel_any_packet_contact_rows": int(np.any(gel > .05, axis=1).sum()), "gel_bilateral_packet_contact_rows": int(gel_bilateral.sum()),
        "carry_interval_s": [float(carry_start), float(carry_end)],
        "carry_bilateral_body_fraction": float(bilateral[carry].mean()), "carry_longest_bilateral_gap_s": gap,
        "carry_tcp_relative_slip_p95_mm": float(1000 * np.quantile(slip, .95)), "carry_tcp_relative_slip_max_mm": float(1000 * slip.max()),
        "packet_robot_support_peak_n": float(np.max(trace["packet_robot_normal_force"])), "packet_bin_support_peak_n": float(np.max(trace["packet_bin_normal_force"])),
        "raw_sha256": {p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.suffix in [".json", ".npz", ".mp4"] and p.name != "diagnostic.json"},
        "limitations": ["Pad-body contact can include exterior/backing; active gel is separately filtered.", "Forces are uncalibrated normal-contact proxies, not measured real Newtons.", "Real packet 3D pose and exact initial scene correspondence unavailable; fixedR4 reset is preserved.", "This is measured motion, not a closed-loop policy or bounded-native-safety replay."],
    }
    write(directory / "diagnostic.json", result)
    return result


def execute(args):
    protocol = json.loads((args.out / "protocol.json").read_text())
    validate(args, protocol)
    baseline = baseline_finished(args.study_root)
    progress_path = args.out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"status": "running", "cases": {}, "protocol_sha256": sha(args.out / "protocol.json")}
    if progress["protocol_sha256"] != sha(args.out / "protocol.json"):
        raise ValueError("Replay protocol changed")
    progress["baseline_dependency"] = {"status": baseline["status"], "trial_count": len(baseline["trials"]), "progress_sha256": sha(args.study_root / "campaign/progress.json")}
    from tools.sim.run_policy_campaign import stop_owned
    child = None
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for mode in ["replay", "dynamics"]:
            directory = args.out / mode
            if progress["cases"].get(mode, {}).get("status") == "completed":
                result = diagnostics(args, protocol, mode)
                if not result["kinematic_gates_passed"]:
                    raise ValueError("Previously completed replay failed re-audit")
                continue
            if directory.exists():
                raise FileExistsError(f"Incomplete replay protected: {directory}")
            directory.mkdir()
            row = {"status": "running", "command": protocol["commands"][mode], "started_utc": datetime.now(timezone.utc).isoformat()}
            progress["cases"][mode] = row
            write(progress_path, progress)
            start = time.monotonic()
            with (directory / "launch.log").open("w") as log:
                child = subprocess.Popen(row["command"], cwd=args.source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait(timeout=900)
            child = None
            row.update(returncode=code, wall_s=time.monotonic() - start)
            if code:
                raise RuntimeError(f"{mode} replay exited{code}")
            comparison = [sys.executable, str(args.source / "tools/sim/compare_replay.py"), "--reference", str(args.out / "reference"), "--sim-trace", str(directory / "sim_trace.npz"), "--sim-video", str(directory / "sim.mp4"), "--tactile-reference", str(args.out / "tactile"), "--out", str(directory / "comparison")]
            with (directory / "comparison.log").open("w") as log:
                subprocess.run(comparison, cwd=args.source, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            result = diagnostics(args, protocol, mode)
            row["kinematic_gates_passed"] = result["kinematic_gates_passed"]
            row["physical_pick_carry_reproduced"] = result["physical_pick_carry_reproduced"]
            row["status"] = "completed" if result["kinematic_gates_passed"] else "completed_kinematic_gate_failed"
            write(progress_path, progress)
            if not result["kinematic_gates_passed"]:
                raise RuntimeError("Measured replay failed frozen kinematic gates; inspect before continuing")
        progress["status"] = "completed"
    except BaseException as error:
        progress["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        progress["error"] = repr(error)
        raise
    finally:
        stop_owned(child)
        write(progress_path, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--robot-usd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for name in ["source", "study_root", "episode", "robot_usd", "out"]:
        setattr(args, name, getattr(args, name).resolve())
    if args.out == args.episode or args.episode in args.out.parents or args.out == args.source or args.source in args.out.parents:
        raise ValueError("Outputs must be outside source episode and source checkout")
    sys.path.insert(0, str(args.source))
    # Lock is a sibling so a new output directory can remain empty for preparation.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            prepare(args)
        else:
            execute(args)


if __name__ == "__main__":
    main()
