#!/usr/bin/env python3
"""Read completed corrected-study start5928 cases on compute3; JSON to stdout.

No simulator, policy inference, sensor replay, or source/data mutation is run.
The independent wrench reconstruction covers the native wrench subguard only.
"""

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.dont_write_bytecode = True
BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
SOURCE = BASE / "source_teacher_v2_delivery"
ROOT = BASE / "runs/teacher_robustness_v2_delivery/screen"
sys.path.insert(0, str(SOURCE))
from phantom.sim.policy_metrics import evaluate_policy_trace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def native_wrench_component(executions, hardware):
    """Reproduce calm-only EMA and time-based debounce, without other guards."""
    limits = hardware["safety"]
    debounce = limits["wrench_debounce_ticks"] / hardware["control"]["action_rate_hz"]
    baseline, last_t, over_since = None, None, None
    rows = []
    for row in executions:
        t = row["t"]
        ft = np.asarray(row["measured_wrist_ft"], dtype=float)
        if baseline is None:
            baseline = ft.copy()
        deviation = ft - baseline
        force = float(np.linalg.norm(deviation[:3]))
        torque = float(np.linalg.norm(deviation[3:]))
        over = force > limits["wrench_limit_N"] or torque > limits["wrench_limit_Nm"]
        dt = 0 if last_t is None else t - last_t
        last_t = t
        if over:
            if over_since is None:
                over_since = t
        else:
            over_since = None
        rows.append(
            {
                "t_s": t,
                "baseline_n_nm": baseline.tolist(),
                "deviation_force_n": force,
                "deviation_torque_nm": torque,
                "over_threshold": over,
                "current_over_since_s": over_since,
                "debounced_wrench_stop": bool(over and t - over_since >= debounce),
            }
        )
        if not over and dt > 0:
            baseline += min(1.0, dt / limits["wrench_baseline_tau_s"]) * deviation
        if row["stopped"]:
            break
    first_over = next((r for r in rows if r["over_threshold"]), None)
    first_trip = next((r for r in rows if r["debounced_wrench_stop"]), None)
    return {
        "threshold_force_n": limits["wrench_limit_N"],
        "threshold_torque_nm": limits["wrench_limit_Nm"],
        "baseline_tau_s": limits["wrench_baseline_tau_s"],
        "debounce_s": debounce,
        "first_over_threshold": first_over,
        "first_debounced_stop": first_trip,
        "maximum_force_deviation": max(rows, key=lambda r: r["deviation_force_n"]),
        "maximum_torque_deviation": max(rows, key=lambda r: r["deviation_torque_nm"]),
        "last_guard_row": rows[-1],
    }


def body_contacts(wrist_row):
    output = []
    tcp = np.asarray(wrist_row["tcp_pose"][:3])
    for actor in wrist_row["per_actor"]:
        for contact in actor["contacts"]:
            signed = np.asarray(contact["normal_force_signed_n"])
            if not np.any(np.abs(signed) > 0):
                continue
            points = np.asarray(contact["points_world_m"])
            normals = np.asarray(contact["normals_world"])
            force = signed[:, None] * normals
            wrench = np.r_[force.sum(axis=0), np.cross(points - tcp, force).sum(axis=0)]
            output.append(
                {
                    "actor_path": actor["actor_path"],
                    "filter_path": contact["filter_path"],
                    "normal_force_magnitude_n": contact["normal_force_magnitude_n"],
                    "normal_wrench_world_n_nm": wrench.tolist(),
                    "loaded_centroid_world_m": np.average(
                        points, axis=0, weights=np.abs(signed)
                    ).tolist(),
                    "signed_forces_n": signed.tolist(),
                    "normals_world": normals.tolist(),
                    "points_world_m": points.tolist(),
                    "signed_separation_m": contact["signed_separation_m"],
                }
            )
    return output


def wrist_snapshot(row):
    return {
        "t_s": row["t"],
        "tcp_pose": row["tcp_pose"],
        "normal_wrench_world_n_nm": row["normal_wrench_world"],
        "wrist_ft_n_nm": row["wrist_ft"],
        "contacts": body_contacts(row),
    }


def summarize_case(case, design, progress, hardware):
    entry = progress["trials"][case]
    assert entry["status"] == "completed" and entry["exit_code"] == 0, case
    folder = ROOT / "rollouts" / case
    run = json.loads((folder / "run.json").read_text())
    assert run["mode"] == "policy" and run["wrist_model"] == "gripper_contact_proxy"
    config = json.loads((folder / "effective_config.json").read_text())
    executions = read_jsonl(folder / "execution_trace.jsonl")
    wrist = read_jsonl(folder / "wrist_contact_trace.jsonl")
    gel = json.loads((folder / "gel_contact_trace.json").read_text())
    plans = json.loads((folder / "planner_trace.json").read_text())
    delivered = read_jsonl(folder / "delivered_plans.jsonl")
    z = np.load(folder / "sim_trace.npz")
    metrics = evaluate_policy_trace(
        z,
        config,
        design["thresholds"],
        run=run,
        execution_trace=executions,
        planner_trace=plans,
    )
    stop = next((r for r in executions if r["stopped"]), None)
    stop_t = stop["t"] if stop else None
    end_t = stop_t if stop else executions[-1]["t"]
    wrist = [r for r in wrist if r["t"] <= end_t + 1e-9]
    guard = native_wrench_component(executions, hardware)
    expected_trip = guard["first_debounced_stop"]
    actual_wrench_stop = bool(
        stop and "wrench_limit" in stop["diagnostics"]["safety_events"]
    )
    assert actual_wrench_stop == (expected_trip is not None), case
    if actual_wrench_stop:
        assert expected_trip["t_s"] == stop_t, (case, expected_trip, stop_t)
    wrist_by_time = {round(r["t"], 9): r for r in wrist}
    relevant = [r for r in executions if r["t"] <= end_t]
    assert all(r["wrist_capture_t"] <= r["t"] + 1e-9 for r in relevant)
    wrist_cache_error = max(
        float(
            np.max(
                np.abs(
                    np.array(r["measured_wrist_ft"])
                    - wrist_by_time[round(r["wrist_capture_t"], 9)]["wrist_ft"]
                )
            )
        )
        for r in relevant
    )
    assert wrist_cache_error == 0
    peak_body, first_body = {}, {}
    maximum_wrench_arithmetic_error = 0.0
    maximum_impulse_force_conversion_error = 0.0
    for row in wrist:
        contacts = body_contacts(row)
        reconstructed = np.zeros(6)
        for actor in row["per_actor"]:
            for contact in actor["contacts"]:
                converted = (
                    np.asarray(contact["normal_impulse_signed_ns"])
                    / row["physics_dt_s"]
                )
                error = np.max(
                    np.abs(converted - contact["normal_force_signed_n"]), initial=0
                )
                maximum_impulse_force_conversion_error = max(
                    maximum_impulse_force_conversion_error, float(error)
                )
        for contact in contacts:
            reconstructed += contact["normal_wrench_world_n_nm"]
            key = contact["actor_path"] + " -> " + contact["filter_path"]
            if (
                key not in peak_body
                or contact["normal_force_magnitude_n"]
                > peak_body[key]["normal_force_magnitude_n"]
            ):
                peak_body[key] = {"t_s": row["t"], **contact}
            if contact["normal_force_magnitude_n"] > 0.1 and key not in first_body:
                first_body[key] = row["t"]
        maximum_wrench_arithmetic_error = max(
            maximum_wrench_arithmetic_error,
            float(np.max(np.abs(reconstructed - row["normal_wrench_world"]))),
        )
    active = [
        r for r in executions if not r["stopped"] and r.get("accepted_tcp") is not None
    ]
    tracking = np.array(
        [
            np.linalg.norm(np.asarray(r["accepted_tcp"][:3]) - r["measured_tcp"][:3])
            for r in active
        ]
    )
    ik = np.array(
        [
            np.linalg.norm(np.asarray(r["accepted_tcp"][:3]) - r["requested_tcp"][:3])
            for r in active
        ]
    )
    peak_row = active[int(np.argmax(tracking))]
    snapshots = {}
    for label, time in [
        (
            "first_over_wrench_threshold",
            guard["first_over_threshold"]["t_s"]
            if guard["first_over_threshold"]
            else None,
        ),
        ("maximum_wrench_deviation", guard["maximum_force_deviation"]["t_s"]),
        ("stop", stop_t),
    ]:
        if time is not None:
            e = min(relevant, key=lambda r: abs(r["t"] - time))
            snapshots[label] = wrist_snapshot(
                wrist_by_time[round(e["wrist_capture_t"], 9)]
            )
    gel_before_stop = [r for r in gel if r["t"] <= end_t + 1e-9]
    pad_peaks = {}
    for row in gel_before_stop:
        for side, pad in zip(["left", "right"], row["per_pad"]):
            for contact in pad["per_filter_contacts"]:
                key = side + " -> " + contact["filter_path"]
                if (
                    key not in pad_peaks
                    or contact["normal_force_magnitude_n"]
                    > pad_peaks[key]["normal_force_n"]
                ):
                    pad_peaks[key] = {
                        "t_s": row["t"],
                        "normal_force_n": contact["normal_force_magnitude_n"],
                        "gel_force_n": contact["gel_compression_n"],
                        "record": contact,
                        "pad_rejection_budgets_n": pad["ignored_by_reason_n"],
                    }
    closest_gel = max(
        (r for r in gel_before_stop if r["t"] <= end_t), key=lambda r: r["t"]
    )
    return {
        "case_id": case,
        "directory": str(folder),
        "completed_exit_code": entry["exit_code"],
        "run_duration_s": run["duration_s"],
        "execution_end_s": end_t,
        "input_sha256": {
            name: sha(folder / name)
            for name in [
                "case.json",
                "initialization.json",
                "run.json",
                "effective_config.json",
                "sim_trace.npz",
                "execution_trace.jsonl",
                "planner_trace.json",
                "delivered_plans.jsonl",
                "wrist_contact_trace.jsonl",
                "gel_contact_trace.json",
                "policy_info.json",
            ]
        },
        "metrics": metrics,
        "first_safety_stop": stop,
        "wrench_subguard_reconstruction": guard,
        "wrench_subguard_stop_matches_log": True,
        "input_audit": {
            "causal_exact_cache_match": True,
            "cache_max_error": wrist_cache_error,
            "maximum_wrench_arithmetic_error_n_nm": maximum_wrench_arithmetic_error,
            "maximum_impulse_force_conversion_error_n": maximum_impulse_force_conversion_error,
            "initial_wrist_bias_n_nm": run["initial_recorded_wrist_bias"],
            "initial_wrist_force_norm_n": float(
                np.linalg.norm(run["initial_recorded_wrist_bias"][:3])
            ),
            "wrist_actor_paths": run["wrist_proxy_metadata"]["actor_paths"],
            "wrist_environment_paths": run["wrist_proxy_metadata"]["environment_paths"],
        },
        "tracking": {
            "maximum_accepted_requested_position_error_m": float(ik.max()),
            "maximum_accepted_measured_position_error_m": float(tracking.max()),
            "first_accepted_measured_error_gt_20mm_s": next(
                (r["t"] for r, v in zip(active, tracking) if v > 0.02), None
            ),
            "maximum_tracking_error_row": {
                k: peak_row.get(k)
                for k in [
                    "t",
                    "requested_tcp",
                    "accepted_tcp",
                    "measured_tcp",
                    "measured_qd",
                    "target_q",
                    "measured_q",
                ]
            },
            "all_reported_ik_rejects": max(
                r["diagnostics"].get("ik_rejects_total", 0) for r in executions
            ),
        },
        "wrist_snapshots": snapshots,
        "per_body_peak_contact_before_stop": list(peak_body.values()),
        "per_body_first_force_gt_point1_n_s": first_body,
        "gel": {
            "peak_per_pad_before_stop_n": np.max(
                [r["normal_force_n"] for r in gel_before_stop], axis=0
            ).tolist(),
            "last_causal_capture_t_s": closest_gel["t"],
            "last_causal_normal_force_n": closest_gel["normal_force_n"],
            "per_body_peaks": pad_peaks,
        },
        "latch_observed": any(
            r["diagnostics"].get("grip_latch") is not None for r in executions
        ),
        "delivered_veto_action_counts": dict(
            Counter(
                r["diagnostics"].get("terminal_veto", {}).get("action", "missing")
                for r in delivered
            )
        ),
    }


def main():
    design = json.loads((ROOT / "campaign_snapshot.json").read_text())
    progress = json.loads((ROOT / "progress.json").read_text())
    hardware_path = BASE / "runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml"
    hardware = yaml.safe_load(hardware_path.read_text())
    cases = [
        f"{policy}__start_1787395928__seed{seed}"
        for policy in [
            "fta1500_nfe1_k4",
            "fta3000_nfe1_k4",
            "v5_6_nfe1_k4",
            "fta1500_nfe5_k1",
        ]
        for seed in [903101, 903102]
    ]
    for case in cases:
        assert progress["trials"][case]["status"] == "completed", case
    results = [summarize_case(case, design, progress, hardware) for case in cases]
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All eight corrected-study trials at start1787395928 only; descriptive morphology, no ranking or pooled superseded trials.",
        "campaign_snapshot_sha256": sha(ROOT / "campaign_snapshot.json"),
        "hardware_sha256": sha(hardware_path),
        "source_sha256": {
            name: sha(SOURCE / name)
            for name in [
                "tools/sim/run_waffles.py",
                "tools/sim/gripper_wrist.py",
                "tools/sim/gel_contact.py",
                "phantom/deploy/safety.py",
                "phantom/sim/policy_metrics.py",
            ]
        },
        "trials": results,
        "limits": [
            "Read-only completed-trial audit; no new inference, simulation, source changes, or tuning.",
            "Only housing and two pad rigid bodies versus table/mat/bin/packet are included in the wrist proxy; no proximal-arm or self-contact attribution.",
            "Contact normals/points and impulse-to-force arithmetic establish modeled actor identity, not physical UR3 wrist calibration. Friction, inertia, gravity, current-estimation transfer and pose-varying real bias are omitted.",
            "Wrench guard uses calm-only rolling EMA, limits60N/15Nm and0.3s debounce, not raw bias-added wrench norm.",
            "Peak/body/gel contact summaries end at the safety stop, excluding the subsequent2s observation tail; object-state metrics retain the complete trace.",
            "First additional closure uses command > initial measured closure+2/255 and the prior15Hz pad-midpoint-to-OBB surface distance; not fingertip clearance or first increase after reopening.",
            "Absolute closure threshold0.25 is exceeded at startup; frozen reach boolean is not an unambiguous approach assessment here.",
            "Safety failures in this conditional model are not infrastructure failures or proven real-world failures. No teacher ranking from this single start.",
        ],
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
