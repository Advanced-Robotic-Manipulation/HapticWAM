#!/usr/bin/env python3
"""Read-only CPU audit; run on compute3 and capture stdout locally as JSON."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v3"
REFERENCE = BASE / "runs/teacher_pick_place_v1/campaign"
STAGES = (
    "reach",
    "closure",
    "acquisition",
    "lift",
    "carry",
    "release_in_bin",
    "full_task",
    "first_drop",
)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def radius(q):
    a2, a3, d4 = 0.24365, 0.21325, 0.11235
    return float(np.sqrt(a2 * a2 + a3 * a3 + 2 * a2 * a3 * np.cos(q[2]) + d4 * d4))


def angle(a, b):
    return float(
        np.rad2deg(
            (Rotation.from_rotvec(a).inv() * Rotation.from_rotvec(b)).magnitude()
        )
    )


def at_scene(z, t):
    i = int(np.argmin(np.abs(z["t"] - t)))
    return {
        "scene_t_s": float(z["t"][i]),
        "tcp_xyz_rotvec": z["tcp"][i].tolist(),
        "closure": float(z["gripper"][i, 0]),
        "packet_center_m": z["waffle_position"][i].tolist(),
        "packet_orientation_wxyz": z["waffle_orientation_wxyz"][i].tolist(),
        "packet_force_per_pad_n": z["pad_packet_normal_force"][i].tolist(),
        "wrist_radius_m": radius(z["q"][i]),
    }


def audit(root, case, label):
    folder = root / "rollouts" / case
    score_path = root / "analysis/trials" / (case + ".json")
    score = read(score_path)["metrics"]
    assert score["valid_for_scoring"], (label, score["invalid_reasons"])
    cfg = read(folder / "effective_config.json")
    release = read(root / "runtime/placement_release.json")
    lo, hi = np.asarray(release["tcp_min_m"]), np.asarray(release["tcp_max_m"])
    ex = rows(folder / "execution_trace.jsonl")
    stop = next(r for r in ex if r["stopped"])
    live = [r for r in ex if r["t"] <= stop["t"]]
    z = np.load(folder / "sim_trace.npz")
    stage = {
        k: None
        if score["event_times_s"][k] is None
        else at_scene(z, score["event_times_s"][k])
        for k in STAGES
    }
    closure_t = score["event_times_s"]["closure"]
    preclosure = None
    if closure_t is not None:
        i = int(np.flatnonzero(z["t"] < closure_t)[-1])
        rot = Rotation.from_quat(z["waffle_orientation_wxyz"][i][[1, 2, 3, 0]])
        local = rot.inv().apply(
            z["pad_position"][i].mean(axis=0) - z["waffle_position"][i]
        )
        distance = np.linalg.norm(
            np.maximum(np.abs(local) - np.asarray(cfg["waffle"]["size"]) / 2, 0)
        )
        preclosure = {
            "sample_t_s": float(z["t"][i]),
            "pad_midpoint_to_packet_obb_m": float(distance),
            "closure": float(z["gripper"][i, 0]),
        }
    first_latch = next(
        (r["t"] for r in live if r["diagnostics"].get("grip_latch") is not None), None
    )
    post_latch = (
        [] if first_latch is None else [r for r in live if r["t"] >= first_latch]
    )
    deliveries = rows(folder / "delivered_plans.jsonl")
    by_id = {p["diagnostics"]["terminal_veto"]["replan_index"]: p for p in deliveries}
    raw = read(folder / "planner_trace.json")
    raw_by_id = {p["replan_id"]: p for p in raw}
    played, unique_played = [], set()
    for r in post_latch:
        if r["stopped"] or r["active_replan_id"] not in by_id:
            continue
        p = by_id[r["active_replan_id"]]
        idx = int(
            np.clip(
                r["diagnostics"]["play_time_s"] * 10,
                0,
                min(len(p["actions"]), 10) - 1e-6,
            )
        )
        veto = p["diagnostics"]["terminal_veto"]
        eligible = veto["action"] not in (
            "recovery_open",
            "recovery_tactile",
            "retry_cap",
        )
        if veto["action"] == "close_masked":
            eligible = idx in veto.get("placement_release_passthrough_indices", [])
        key = (r["active_replan_id"], idx)
        unique_played.add(key)
        played.append(
            {
                "t_s": r["t"],
                "key": key,
                "eligible": eligible,
                "raw_closure": raw_by_id[key[0]]["actions"][idx][6],
                "delivered_closure": p["actions"][idx][6],
                "window": r["diagnostics"]["placement_release"]["window_active"],
            }
        )
    raw_post = (
        []
        if first_latch is None
        else [p for p in raw if first_latch <= p["t_created"] <= stop["t"]]
    )
    raw_open = [
        (p["replan_id"], i)
        for p in raw_post
        for i, a in enumerate(p["actions"])
        if a[6] <= 0.45
    ]
    eligible_open = [
        p for p in played if p["eligible"] and p["delivered_closure"] <= 0.45
    ]
    first_open = None if not eligible_open else eligible_open[0]["t_s"]
    first_window = next(
        (
            r["t"]
            for r in post_latch
            if r["diagnostics"]["placement_release"]["window_active"]
        ),
        None,
    )
    stop_pose = np.asarray(stop["measured_tcp"])
    prior = min(live, key=lambda r: abs(r["t"] - (stop["t"] - 0.5)))
    stop_sample = at_scene(z, stop["t"])
    stop_i = int(np.argmin(np.abs(z["t"] - stop["t"])))
    causal_i = int(np.flatnonzero(z["t"] < stop["t"])[-1])
    causal_sample = at_scene(z, float(z["t"][causal_i]))
    bilateral = (z["pad_packet_normal_force"] > 0.1).all(axis=1)
    loaded_i = np.flatnonzero(bilateral & (z["t"] < stop["t"]))
    last_contact_t = None if not len(loaded_i) else float(z["t"][loaded_i[-1]])
    last_contact_row = (
        None
        if last_contact_t is None
        else min(live, key=lambda r: abs(r["t"] - last_contact_t))
    )
    gel = [r for r in read(folder / "gel_contact_trace.json") if r["t"] <= stop["t"]]
    gf = np.asarray([r["normal_force_n"] for r in gel])
    packet_delta_z = float(z["waffle_position"][stop_i, 2] - z["waffle_position"][0, 2])
    pre_stop_force = np.asarray(stop_sample["packet_force_per_pad_n"])
    last_lift = score["event_times_s"]["lift"]
    after_lift = z["t"] >= (last_lift if last_lift is not None else 0)
    before_stop = z["t"] <= stop["t"]
    phases = Counter(r["diagnostics"]["placement_release"]["phase"] for r in live)
    out = {
        "label": label,
        "folder": str(folder),
        "score_outcomes": score["outcomes"],
        "score_events_s": score["event_times_s"],
        "reach_before_absolute_closure": preclosure,
        "additional_closure_diagnostic": score.get("reach_diagnostic"),
        "stage_samples": stage,
        "object_max_rise_m": score["object"]["max_lift_m"],
        "first_latch_s": first_latch,
        "gel_force_peak_per_pad_n": gf.max(axis=0).tolist(),
        "max_concurrent_weaker_gel_force_n": float(gf.min(axis=1).max()),
        "last_pre_stop_bilateral_contact": None
        if last_contact_row is None
        else {
            "scene": at_scene(z, last_contact_t),
            "executor_t_s": last_contact_row["t"],
            "commanded_closure": last_contact_row["gripper_command"],
            "latch": last_contact_row["diagnostics"]["grip_latch"],
            "packet_center_rise_m": float(
                z["waffle_position"][loaded_i[-1], 2] - z["waffle_position"][0, 2]
            ),
        },
        "release": {
            "config": release,
            "phase_counts": dict(phases),
            "at_stop": stop["diagnostics"]["placement_release"],
            "first_window_s": first_window,
            "first_eligible_played_opening_s": first_open,
            "eligible_played_opening_executor_samples_after_first_latch": len(
                eligible_open
            ),
            "eligible_played_opening_unique_rows_after_first_latch": len(
                {p["key"] for p in eligible_open}
            ),
            "eligible_played_opening_inside_window_samples": sum(
                p["window"] for p in eligible_open
            ),
            "raw_post_latch_predicted_opening_rows": len(raw_open),
            "raw_post_latch_opening_rows_played": sum(
                p in unique_played for p in raw_open
            ),
            "raw_post_latch_opening_rows_unplayed": sum(
                p not in unique_played for p in raw_open
            ),
            "post_latch_played_raw_openings_including_plans_created_before_latch": len(
                {p["key"] for p in played if p["raw_closure"] <= 0.45}
            ),
            "minimum_original_eligible_played_closure": min(
                (p["delivered_closure"] for p in played if p["eligible"]), default=None
            ),
        },
        "stop": {
            "t_s": stop["t"],
            "safety_events": stop["diagnostics"]["safety_events"],
            "measured_tcp_xyz_rotvec": stop_pose.tolist(),
            "requested_tcp_xyz_rotvec": stop["requested_tcp"],
            "tracking_error_m": float(
                np.linalg.norm(stop_pose[:3] - np.asarray(stop["requested_tcp"][:3]))
            ),
            "tracking_rotation_error_deg": angle(
                stop_pose[3:], stop["requested_tcp"][3:]
            ),
            "wrist_radius_m": radius(stop["measured_q"]),
            "wrist_radius_limit_m": 0.468,
            "release_below_xyz_m": np.maximum(lo - stop_pose[:3], 0).tolist(),
            "release_above_xyz_m": np.maximum(stop_pose[:3] - hi, 0).tolist(),
            "prior_sample_t_s": prior["t"],
            "last_half_second_measured_delta_xyz_m": (
                stop_pose[:3] - np.asarray(prior["measured_tcp"][:3])
            ).tolist(),
            "last_half_second_requested_delta_xyz_m": (
                np.asarray(stop["requested_tcp"][:3])
                - np.asarray(prior["requested_tcp"][:3])
            ).tolist(),
            "last_half_second_orientation_change_deg": angle(
                prior["measured_tcp"][3:], stop_pose[3:]
            ),
            "latch": stop["diagnostics"]["grip_latch"],
            "measured_closure": stop["measured_gripper"][0],
            "commanded_closure": stop["gripper_command"],
            "scene_sample": stop_sample,
            "last_strictly_pre_stop_scene_sample": causal_sample,
            "packet_center_rise_m": packet_delta_z,
            "physical_bilateral_and_raised_at_nearest_sample": bool(
                np.all(pre_stop_force > 0.1) and packet_delta_z >= 0.03
            ),
            "drop_relative_to_stop_s": None
            if score["event_times_s"]["first_drop"] is None
            else score["event_times_s"]["first_drop"] - stop["t"],
            "gel_last_causal_t_s": gel[-1]["t"],
            "gel_last_causal_force_n": gf[-1].tolist(),
        },
        "maximum_tcp_z_after_lift_before_stop_m": float(
            z["tcp"][after_lift & before_stop, 2].max()
        ),
        "input_sha256": {
            name: sha(folder / name)
            for name in (
                "sim_trace.npz",
                "execution_trace.jsonl",
                "planner_trace.json",
                "delivered_plans.jsonl",
                "effective_config.json",
                "gel_contact_trace.json",
            )
        },
        "score_sha256": sha(score_path),
    }
    # Compact progress-normalized samples retain no full observation recordings.
    out["carry_relative_samples"] = (
        []
        if score["event_times_s"]["carry"] is None
        else [
            at_scene(z, score["event_times_s"]["carry"] + dt)
            for dt in (0, 0.5, 1, 1.5, 2)
            if score["event_times_s"]["carry"] + dt <= stop["t"]
        ]
    )
    return out, z, live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["reproduction", "components/baseline", "components/finish_only"],
    )
    args = parser.parse_args()
    ref, _ref_z, ref_live = audit(
        REFERENCE, "teacher__placement_xm10_ym10mm__seed4242", "original_success"
    )
    cases = []
    for group in args.groups:
        root = ROOT / group
        progress = read(root / "progress.json")
        assert progress["status"] == "all_trials_completed"
        assert all(
            v["status"] == "completed" and v["exit_code"] == 0
            for v in progress["trials"].values()
        )
        for case in sorted(progress["trials"]):
            out, _z, _live = audit(root, case, group + "/" + case)
            out["same_stage_delta_from_original"] = {}
            for stage in STAGES:
                a, b = out["stage_samples"][stage], ref["stage_samples"][stage]
                if a is not None and b is not None:
                    out["same_stage_delta_from_original"][stage] = {
                        "tcp_delta_xyz_m": (
                            np.asarray(a["tcp_xyz_rotvec"][:3])
                            - b["tcp_xyz_rotvec"][:3]
                        ).tolist(),
                        "orientation_difference_deg": angle(
                            a["tcp_xyz_rotvec"][3:], b["tcp_xyz_rotvec"][3:]
                        ),
                        "closure_delta": a["closure"] - b["closure"],
                    }
            # Match progress toward the bin in Y on the successful carry-to-release leg.
            ref_leg = [
                r
                for r in ref_live
                if ref["score_events_s"]["carry"]
                <= r["t"]
                <= ref["score_events_s"]["release_in_bin"]
            ]
            same_y = min(
                ref_leg,
                key=lambda r: abs(
                    r["measured_tcp"][1] - out["stop"]["measured_tcp_xyz_rotvec"][1]
                ),
            )
            pose = out["stop"]["measured_tcp_xyz_rotvec"]
            out["original_at_nearest_transport_y"] = {
                "reference_t_s": same_y["t"],
                "reference_pose_xyz_rotvec": same_y["measured_tcp"],
                "y_mismatch_m": abs(same_y["measured_tcp"][1] - pose[1]),
                "case_minus_reference_xyz_m": (
                    np.asarray(pose[:3]) - same_y["measured_tcp"][:3]
                ).tolist(),
                "orientation_difference_deg": angle(
                    pose[3:], same_y["measured_tcp"][3:]
                ),
                "reference_wrist_radius_m": radius(same_y["measured_q"]),
                "reference_closure": same_y["measured_gripper"][0],
            }
            cases.append(out)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "Original successful reference plus completed explicitly listed v3 diagnostic groups; no model ranking or score mutations.",
                "groups": args.groups,
                "definitions": {
                    "preclosure_reach": "Last 15Hz pad-midpoint-to-packet OBB distance before frozen absolute closure>=.25 event; metric is geometric proxy, not fingertip clearance.",
                    "stage": "Nearest 15Hz sample to original frozen score event, up to .034s from event; sustained stage gates unchanged.",
                    "stop": "Exact 125Hz execution first safety stop; object evidence nearest 15Hz sample can fall just after stop.",
                    "played_opening": "Delivered gripper ZOH row at recorded play_time, original-policy eligibility from historical terminal-veto diagnostics; stopped rows excluded. Raw proposal tails counted separately.",
                    "nearest_y": "Original carry-to-release trajectory nearest measured transport Y; progress diagnostic, not full state matching or counterfactual.",
                    "limits": "Rendering and inference timing differ. No controlled causal attribution from cross-run deltas; sensor/force magnitudes and geometry remain simulation hypotheses.",
                },
                "reference": ref,
                "cases": cases,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
