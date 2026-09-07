#!/usr/bin/env python3
"""Summarize saved read-only evidence; no simulator or model imports."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
# data/ -> behavior_debug/ -> results/
RESULTS = HERE.parent.parent
DESIGN = RESULTS / "teacher_v2_design"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


r = read(HERE / "demonstration_audit.json")
w = read(HERE / "training_window_audit.json")
payload = read(DESIGN / "teacher_payloads.json")["checkpoints"][0]
initial_path = RESULTS / "teacher_pick_place_v1/sept4_policy_initial_state.json"
initial = read(initial_path)
full = r["full"]
tcp = np.asarray([x["initial_tcp"] for x in full])
ref = np.asarray(initial["tcp"])
rot = (
    (Rotation.from_rotvec(tcp[:, 3:]) * Rotation.from_rotvec(ref[3:]).inv()).magnitude()
    * 180
    / np.pi
)
summary = {
    "schema_version": 1,
    "counts": r["counts"],
    "input_sha256": {
        str(p.relative_to(RESULTS)): sha(p)
        for p in [
            HERE / "demonstration_audit.json",
            HERE / "training_window_audit.json",
            DESIGN / "teacher_payloads.json",
            DESIGN / "cohort_candidate.json",
            initial_path,
        ]
    },
    "checkpoint_current_bytes_match_prior_cpu_payload": r["checkpoint"]["sha256"]
    == payload["sha256"],
    "checkpoint": payload,
    "historical_training_manifest_status": "Not established. Saved train split but no episode list/root/code revision in inspected payload. Current incomplete mirrors are not evidence of corrupt historical optimizer input.",
    "current_initial_state": initial,
    "start_difference_from_sept4": {
        "position_distance_mm": (
            1000 * np.linalg.norm(tcp[:, :3] - ref[:3], axis=1)
        ).tolist(),
        "orientation_geodesic_deg": rot.tolist(),
    },
    "full_native_wrist_limit_audit": {
        "measured_samples": sum(x["samples"] for x in full),
        "episodes_cross_468mm": sum(
            x["wrist_radius_over_468mm_samples"] > 0 for x in full
        ),
        "max_radius_m": max(x["wrist_radius_m"]["max"] for x in full),
        "minimum_margin_m": 0.468 - max(x["wrist_radius_m"]["max"] for x in full),
    },
    "thin_excluded_trajectories": sum(
        any(
            g["initialized_chunks"] < g["expected_chunks"]
            for g in x["storage"].values()
        )
        for x in r["thin"]
    ),
    "native_tactile_episode_median_ranges": {},
    "per_episode": [],
}
for phase in [
    "initial_0_1s",
    "approach",
    "loaded_pre_lift",
    "lift_carry",
    "after_release",
]:
    summary["native_tactile_episode_median_ranges"][phase] = {}
    for side in ["left", "right"]:
        vals = {}
        for key in [
            "area_mm2",
            "mask_frac",
            "slip",
            "depth_abs_p95_mm",
            "depth_abs_max_mm",
            "wrench_sdk",
        ]:
            items = [
                x["phases"][phase]["tactile"][side][key]
                for x in full
                if x["phases"].get(phase)
            ]
            v = np.asarray([x["p50"] for x in items if x])
            vals[key] = {
                "min_episode_median": v.min(axis=0).tolist(),
                "max_episode_median": v.max(axis=0).tolist(),
            }
        summary["native_tactile_episode_median_ranges"][phase][side] = vals
for x in full:
    tr = x["events"]["gripper_release"]["native_samples"]
    row = {
        "episode": x["episode"],
        "metadata_success": x["success_metadata"],
        "duration_s": x["duration_s"],
        "max_wrist_radius_mm": x["wrist_radius_m"]["max"] * 1000,
        "max_joint_speed_rad_s": x["max_measured_joint_speed_rad_s"],
        "initial_closure": x["initial_gripper"],
        "bilateral_load_proxy_s": x["events"]["bilateral_load"]["t_s"],
        "lift30mm_proxy_s": x["events"]["tcp_lift_30mm"]["t_s"],
        "release_proxy_s": x["events"]["gripper_release"]["t_s"],
        "carry_peak_tcp_z_m": x["phases"]["lift_carry"]["tcp_pose"]["max"][2],
        "release_tcp": tr["arm_tcp_pose"]["value"],
        "release_q": tr["arm_q"]["value"],
        "release_gripper": tr["gripper"]["value"][0],
    }
    ww = next(y for y in w["episodes"] if y["episode"] == x["episode"])
    row["current_loader_valid_anchor_range_s"] = ww["loader_anchor_valid_range_s"]
    tt = x["trajectory_10hz"]
    i = np.searchsorted(tt["t_s"], ww["loader_anchor_valid_range_s"][0])
    row["approx_first_valid_anchor_tcp_delta_mm"] = (
        1000 * (np.asarray(tt["tcp"][i][:3]) - np.asarray(x["initial_tcp"][:3]))
    ).tolist()
    row["anchor_delta_sampling_note"] = (
        "10Hz retained trajectory; first sample on/after formula-derived earliest anchor"
    )
    summary["per_episode"].append(row)
(HERE / "summary.json").write_text(
    json.dumps(summary, indent=2, allow_nan=False) + "\n"
)
with (HERE / "demonstrations.csv").open("w") as f:
    rows = [
        {
            k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
            for k, v in row.items()
        }
        for row in summary["per_episode"]
    ]
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("Wrote summary.json and demonstrations.csv; full episodes:", len(full))
