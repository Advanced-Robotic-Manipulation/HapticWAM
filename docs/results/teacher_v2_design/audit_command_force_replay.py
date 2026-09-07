#!/usr/bin/env python3
"""Read-only sampled mechanics comparison of identical recorded commands."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integral(y, t):
    return float(np.sum(np.diff(t) * (y[:-1] + y[1:]) / 2))


def interp(t, source_t, x):
    x = np.asarray(x)
    flat = x.reshape(len(x), -1)
    return np.column_stack([np.interp(t, source_t, v) for v in flat.T]).reshape(
        (len(t),) + x.shape[1:]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prefix", default="teacher_commands_full_dt")
    args = parser.parse_args()
    original = np.load(args.reference / "sim_trace.npz")
    execution_path = args.reference / "execution_trace.jsonl"
    executions = [
        json.loads(line)
        for line in execution_path.read_text().splitlines()
        if line.strip()
    ]
    commands_t = np.array([r["t"] for r in executions])
    commands_q = np.array([r["target_q"] for r in executions])
    fingers = np.array([r["target_finger_q"] for r in executions])
    results, loaded, cfgs = [], {}, {}
    for ms in (4, 2, 1):
        path = args.root / f"{args.prefix}{ms}ms"
        run = json.loads((path / "run.json").read_text())
        config = json.loads((path / "effective_config.json").read_text())
        initial = json.loads((path / "initialization.json").read_text())
        cmd = json.loads((path / "command_replay.json").read_text())
        arrays = np.load(path / "sim_trace.npz")
        t, force = arrays["t"], arrays["packet_robot_normal_force"]
        assert run["mode"] == "command_replay"
        assert cmd["sha256"] == sha(execution_path)
        assert np.isfinite(force).all() and (force >= 0).all()
        expected = np.searchsorted(commands_t, t + 1e-10, side="right") - 1
        q = np.array(
            [
                commands_q[k] if k >= 0 else initial["configured_robot_q"]
                for k in expected
            ]
        )
        gel = json.loads((path / "gel_contact_trace.json").read_text())
        deepest, max_budget, max_gel = None, 0, 0
        raw_contact_peaks = {}
        for row in gel:
            for side, pad in zip(("left", "right"), row["per_pad"]):
                max_gel = max(max_gel, pad["normal_force_n"])
                max_budget = max(
                    max_budget,
                    abs(
                        pad["observed_filtered_normal_force_n"]
                        - pad["accepted_contact_normal_magnitude_n"]
                        - pad["ignored_contact_normal_force_n"]
                    ),
                )
                for body in pad["per_filter_contacts"]:
                    label = body["filter_path"]
                    raw_contact_peaks[label] = max(
                        raw_contact_peaks.get(label, 0),
                        body["normal_force_magnitude_n"],
                    )
                    for i, separation in enumerate(body["separation_by_contact_m"]):
                        if (
                            deepest is None
                            or separation < deepest["signed_separation_m"]
                        ):
                            deepest = {
                                "t_s": row["t"],
                                "side": side,
                                "body": label,
                                "signed_separation_m": separation,
                                "normal_force_n": body["normal_force_by_contact_n"][i],
                                "point_pad_m": body["contact_points_pad_m"][i],
                                "normal_pad": body["contact_normals_pad"][i],
                            }
        peak = int(np.argmax(force))
        results.append(
            {
                "dt_s": ms / 1000,
                "directory": str(path),
                "mode": run["mode"],
                "rows": len(t),
                "first_t_s": float(t[0]),
                "last_t_s": float(t[-1]),
                "maximum_sample_gap_s": float(np.diff(t).max()),
                "command_input_sha256": cmd["sha256"],
                "command_last_t_s": cmd["last_command_s"],
                "covers_all_commands_and_2s_tail": bool(
                    t[-1] >= cmd["last_command_s"] + 2 - ms / 1000 - 1e-9
                ),
                "command_target_q_max_error_rad": float(
                    abs(arrays["target_q"] - q).max()
                ),
                "recorded_finger_targets_min_max_m": [
                    float(fingers.min()),
                    float(fingers.max()),
                ],
                "finger_target_validation_limit": "Exact input file hashed; target_finger_q is not separately logged in sim_trace. Its gripper columns are measured closure and estimated OBJ status, not target closure. dt4 measured finger/OBJ state equality is checked against the original run.",
                "force_peak_n": float(force[peak]),
                "force_peak_t_s": float(t[peak]),
                "sampled_normal_impulse_ns": integral(force, t),
                "sampled_time_mean_normal_force_n": integral(force, t) / (t[-1] - t[0]),
                "force_peak_pose_m": arrays["waffle_position"][peak].tolist(),
                "final_packet_m": arrays["waffle_position"][-1].tolist(),
                "max_qd_rad_s": float(abs(arrays["qd"]).max()),
                "max_target_q_error_rad": float(
                    abs(arrays["target_q"] - arrays["q"]).max()
                ),
                "max_gel_force_n": max_gel,
                "max_gel_accounting_error_n": max_budget,
                "per_filter_raw_peak_n": raw_contact_peaks,
                "deepest_contact": deepest,
                "initial_q_error_rad": initial["robot_initial_max_joint_error_rad"],
                "initial_qd_rad_s": initial["robot_initial_max_joint_speed_rad_s"],
                "input_sha256": {
                    p.name: sha(p)
                    for p in path.iterdir()
                    if p.name
                    in (
                        "run.json",
                        "sim_trace.npz",
                        "gel_contact_trace.json",
                        "effective_config.json",
                        "initialization.json",
                        "command_replay.json",
                    )
                },
            }
        )
        loaded[ms] = arrays
        cfgs[ms] = config
    invariance = {}
    for ms in (2, 1):
        config = json.loads(json.dumps(cfgs[ms]))
        config["physics"]["dt"] = cfgs[4]["physics"]["dt"]
        invariance[str(ms)] = config == cfgs[4]
    four = loaded[4]
    original_index = {round(float(t), 8): i for i, t in enumerate(original["t"])}
    pairs = [
        (i, original_index[round(float(t), 8)])
        for i, t in enumerate(four["t"])
        if round(float(t), 8) in original_index
    ]
    ix, iy = np.array(pairs).T
    same_time = {}
    for key in sorted(set(original.files) & set(four.files)):
        x, y = four[key][ix], original[key][iy]
        finite = np.isfinite(x) & np.isfinite(y)
        same_time[key] = {
            "bitwise_equal": bool(np.array_equal(x, y, equal_nan=True)),
            "max_difference": float(abs(x[finite] - y[finite]).max())
            if finite.any()
            else None,
        }
    two, one = loaded[2], loaded[1]
    common_t = two["t"][(two["t"] >= one["t"][0]) & (two["t"] <= one["t"][-1])]
    pose_delta = two["waffle_position"][: len(common_t)] - interp(
        common_t, one["t"], one["waffle_position"]
    )
    dt2_result, dt1_result = results[1], results[2]
    convergence = {
        "dt2_vs_dt1_peak_relative_difference": abs(
            dt2_result["force_peak_n"] - dt1_result["force_peak_n"]
        )
        / dt1_result["force_peak_n"],
        "dt2_vs_dt1_sampled_impulse_relative_difference": abs(
            dt2_result["sampled_normal_impulse_ns"]
            - dt1_result["sampled_normal_impulse_ns"]
        )
        / dt1_result["sampled_normal_impulse_ns"],
        "dt2_vs_dt1_max_packet_translation_difference_m_on_dt2_grid": float(
            np.linalg.norm(pose_delta, axis=1).max()
        ),
        "comparison_interpolation": "linear on saved scene-rate positions; no physical high-rate force integration available",
        "thresholds_proposed_before_results": {
            "relative_peak": 0.10,
            "relative_impulse": 0.05,
            "max_pose_difference_m": 0.002,
        },
    }
    convergence["passes_sampled_diagnostic_thresholds"] = (
        convergence["dt2_vs_dt1_peak_relative_difference"] <= 0.10
        and convergence["dt2_vs_dt1_sampled_impulse_relative_difference"] <= 0.05
        and convergence["dt2_vs_dt1_max_packet_translation_difference_m_on_dt2_grid"]
        <= 0.002
    )
    out = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_case": str(args.reference),
        "source_trace_sha256": sha(args.reference / "sim_trace.npz"),
        "results": results,
        "only_dt_changed_in_effective_scene_config": invariance,
        "dt4_original_same_time_prefix": {
            "matched_rows": len(pairs),
            "last_matched_time_s": float(four["t"][ix[-1]]),
            "original_rows": len(original["t"]),
            "replay_rows": len(four["t"]),
            "array_comparison": same_time,
        },
        "convergence": convergence,
        "conclusion": "High arm-driven packet/toe load persists and increases with reduced timestep; mapper correction does not eliminate this physical load. Keep force and safety realism explicitly unvalidated.",
        "limitations": [
            "Sampled 15Hz force peaks/impulses can miss subframe transients; this is a bounded numerical sensitivity diagnostic, not solver or material calibration.",
            "Recorded target replay does not re-run policy, asynchronous timing, or safety logic; no trial success score is assigned.",
            "Force magnitudes, packet deformation and contact offsets require physical measurements. No force clipping, safety-limit relaxation or contact disabling is justified by this diagnostic.",
        ],
    }
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
