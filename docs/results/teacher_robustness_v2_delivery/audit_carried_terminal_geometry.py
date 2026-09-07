#!/usr/bin/env python3
"""Read-only CPU geometry/release audit of the two corrected confirmation carries.

Run on compute3 with the project venv. Prints JSON; writes no trial/source files.
IK examples are geometric existence checks, not motion or collision validation.
"""

import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.dont_write_bytecode = True
BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
SOURCE = BASE / "source_teacher_v2_delivery"
ROOT = BASE / "runs/teacher_robustness_v2_delivery/confirmation"
sys.path.insert(0, str(SOURCE))
from phantom.sim.kinematics import dh_frames, forward_kinematics, inverse_kinematics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    design = json.loads((ROOT / "campaign_snapshot.json").read_text())
    # Runtime release specification is pinned in the campaign, not inferred.
    release = design["adapter_profile"]["placement_release"]
    lo, hi = np.asarray(release["tcp_min_m"]), np.asarray(release["tcp_max_m"])
    hardware_path = BASE / "runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml"
    hardware = yaml.safe_load(hardware_path.read_text())
    a2, a3, d4 = hardware["safety"]["ur_dh_a2_a3_d4_m"]
    limit = hardware["safety"]["wrist_extension_stop_m"]
    rate = hardware["control"]["action_rate_hz"]
    max_play = design["inference_settings"]["max_play"]

    def radius(q):
        return float(np.sqrt(a2 * a2 + a3 * a3 + 2 * a2 * a3 * np.cos(q[2]) + d4 * d4))

    results = []
    for case in (
        "v5_6_nfe1_k4__start_1787396060__seed903201",
        "fta3000_nfe1_k4__start_1787396314__seed903201",
    ):
        folder = ROOT / "rollouts" / case
        executions = read_rows(folder / "execution_trace.jsonl")
        stop = next(row for row in executions if row["stopped"])
        assert stop["diagnostics"]["safety_events"] == ["wrist_extension"]
        rows = [row for row in executions if row["t"] <= stop["t"]]
        cfg = json.loads((folder / "effective_config.json").read_text())
        z = np.load(folder / "sim_trace.npz")
        plans = read_rows(folder / "delivered_plans.jsonl")
        plans_by_id = {
            row["diagnostics"]["terminal_veto"]["replan_index"]: row for row in plans
        }
        q = np.asarray(stop["measured_q"])
        pose = np.asarray(stop["measured_tcp"])
        frames = dh_frames(q)
        frame_radius = float(np.linalg.norm(frames[4, :3, 3] - frames[1, :3, 3]))
        assert abs(frame_radius - radius(q)) < 1e-12
        sample = int(np.argmin(np.abs(z["t"] - stop["t"])))
        object_center = z["waffle_position"][sample]
        object_rot = Rotation.from_quat(
            z["waffle_orientation_wxyz"][sample][[1, 2, 3, 0]]
        )
        corners = object_center + object_rot.apply(
            np.array(list(itertools.product((-1, 1), repeat=3)))
            * np.asarray(cfg["waffle"]["size"])
            / 2
        )
        bin_center = np.asarray(cfg["bin"]["center"])
        bin_size = np.asarray(cfg["bin"]["size"])
        wall = cfg["bin"]["wall"]
        bin_xy_lo = bin_center[:2] - bin_size[:2] / 2 + wall
        bin_xy_hi = bin_center[:2] + bin_size[:2] / 2 - wall
        rim_z = float(bin_center[2] + bin_size[2])
        held = [row for row in rows if row["diagnostics"].get("grip_latch") is not None]
        played = []
        for row in held:
            if row["stopped"] or row["active_replan_id"] not in plans_by_id:
                continue
            plan = plans_by_id[row["active_replan_id"]]
            actions = plan["actions"]
            cap = len(actions) if max_play is None else min(len(actions), max_play)
            index = int(
                np.clip(row["diagnostics"]["play_time_s"] * rate, 0, cap - 1e-6)
            )
            veto = plan["diagnostics"]["terminal_veto"]
            original = veto["action"] not in (
                "recovery_open",
                "recovery_tactile",
                "retry_cap",
            )
            if veto["action"] == "close_masked":
                original = index in veto.get(
                    "placement_release_passthrough_indices", []
                )
            played.append(
                {
                    "t_s": row["t"],
                    "delivered_pre_latch_closure": actions[index][6],
                    "original_policy_eligible": original,
                    "release_window_active": row["diagnostics"]["placement_release"][
                        "window_active"
                    ],
                }
            )
        inside = lambda xyz: bool(np.all(xyz >= lo) and np.all(xyz <= hi))
        distances = [
            float(
                np.linalg.norm(
                    np.clip(row["measured_tcp"][:3], lo, hi) - row["measured_tcp"][:3]
                )
            )
            for row in held
        ]
        examples = []
        # Preserve the measured gripper orientation. One closest release point,
        # one over the physical bin centre at the declared release ceiling.
        targets = {
            "closest_release_volume_point": np.clip(pose[:3], lo, hi),
            "above_bin_center_at_release_ceiling": np.r_[bin_center[:2], hi[2]],
        }
        for name, xyz in targets.items():
            target = np.r_[xyz, pose[3:]]
            ik = inverse_kinematics(
                target,
                q,
                max_joint_delta_rad=10,
                max_iterations=500,
                damping=0.001,
                position_tolerance_m=1e-6,
                rotation_tolerance_rad=1e-6,
            )
            predicted_corners = corners + xyz - pose[:3]
            examples.append(
                {
                    "name": name,
                    "target_tcp_pose_xyz_rotvec": target.tolist(),
                    "inside_declared_release_volume": inside(xyz),
                    "nominal_ik_converged": ik.success,
                    "q_rad": ik.q.tolist(),
                    "position_residual_m": ik.position_error_m,
                    "rotation_residual_rad": ik.rotation_error_rad,
                    "elbow_guard_radius_m": radius(ik.q),
                    "elbow_guard_margin_m": limit - radius(ik.q),
                    "packet_xy_contained_if_relative_grasp_frozen": bool(
                        np.all(predicted_corners[:, :2] >= bin_xy_lo)
                        and np.all(predicted_corners[:, :2] <= bin_xy_hi)
                    ),
                    "packet_lowest_corner_above_rim_if_relative_grasp_frozen_m": float(
                        predicted_corners[:, 2].min() - rim_z
                    ),
                    "limits": "Nominal kinematics only; no collision, branch transition, joint-speed, grip-retention, trajectory or opening-clearance validation. Relative object geometry is sampled up to 1/15 s from the executor stop.",
                }
            )
        finite_played = [p for p in played if p["original_policy_eligible"]]
        results.append(
            {
                "case_id": case,
                "first_stop_s": stop["t"],
                "measured_tcp_pose_xyz_rotvec": pose.tolist(),
                "requested_tcp_pose_xyz_rotvec": stop["requested_tcp"],
                "q_rad": q.tolist(),
                "elbow_q_rad": float(q[2]),
                "elbow_guard_radius_m": radius(q),
                "guard_limit_m": limit,
                "guard_excess_m": radius(q) - limit,
                "dh_wrist_center_frame4_xyz_m": frames[4, :3, 3].tolist(),
                "dh_shoulder_frame1_xyz_m": frames[1, :3, 3].tolist(),
                "radius_matches_dh_frame4_to_frame1_m": frame_radius,
                "nominal_fk_to_logged_tcp_position_error_m": float(
                    np.linalg.norm(forward_kinematics(q)[:3, 3] - pose[:3])
                ),
                "tcp_outside_release_below_xyz_m": np.maximum(
                    lo - pose[:3], 0
                ).tolist(),
                "tcp_outside_release_above_xyz_m": np.maximum(
                    pose[:3] - hi, 0
                ).tolist(),
                "first_latch_s": held[0]["t"],
                "post_latch_inside_release_executor_samples": sum(
                    inside(np.asarray(row["measured_tcp"][:3])) for row in held
                ),
                "post_latch_release_window_active_samples": sum(
                    row["diagnostics"]["placement_release"]["window_active"]
                    for row in held
                ),
                "minimum_post_latch_distance_to_release_volume_m": min(distances),
                "release_phase_counts_post_latch": dict(
                    Counter(
                        row["diagnostics"]["placement_release"]["phase"] for row in held
                    )
                ),
                "release_at_stop": stop["diagnostics"]["placement_release"],
                "measured_closure_at_stop": stop["measured_gripper"][0],
                "commanded_closure_at_stop": stop["gripper_command"],
                "latch_at_stop": stop["diagnostics"]["grip_latch"],
                "eligible_played_pre_latch_closure_min_after_latch": min(
                    (p["delivered_pre_latch_closure"] for p in finite_played),
                    default=None,
                ),
                "eligible_played_opening_samples_after_latch": sum(
                    p["delivered_pre_latch_closure"] <= release["open_command_max"]
                    for p in finite_played
                ),
                "eligible_opening_inside_window_samples_after_latch": sum(
                    p["release_window_active"]
                    and p["delivered_pre_latch_closure"] <= release["open_command_max"]
                    for p in finite_played
                ),
                "opening_reconstruction": "Executor _pose_at zero-order gripper index at logged play_time, delivered actions matched by terminal_veto replan_index; original-policy eligibility follows release_controller. No claim about unused proposal tails.",
                "object_pose_sample_s": float(z["t"][sample]),
                "object_center_at_nearest_scene_sample_m": object_center.tolist(),
                "object_corner_min_m": corners.min(axis=0).tolist(),
                "object_corner_max_m": corners.max(axis=0).tolist(),
                "physical_bin_inner_xy_min_m": bin_xy_lo.tolist(),
                "physical_bin_inner_xy_max_m": bin_xy_hi.tolist(),
                "physical_bin_rim_z_m": rim_z,
                "object_xy_contained_at_stop": bool(
                    np.all(corners[:, :2] >= bin_xy_lo)
                    and np.all(corners[:, :2] <= bin_xy_hi)
                ),
                "nominal_fixed_orientation_ik_examples": examples,
                "input_sha256": {
                    name: sha(folder / name)
                    for name in (
                        "execution_trace.jsonl",
                        "delivered_plans.jsonl",
                        "sim_trace.npz",
                        "effective_config.json",
                    )
                },
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "release_config": release,
                "scope": "Read-only post-study diagnosis; does not alter safety, scores, source, scene or controller. Kinematic existence is insufficient for a safe release trajectory.",
                "guard_parameters_a2_a3_d4_m": [a2, a3, d4],
                "cases": results,
                "source_sha256": {
                    name: sha(SOURCE / name)
                    for name in (
                        "phantom/sim/kinematics.py",
                        "phantom/deploy/safety.py",
                        "phantom/deploy/executor.py",
                        "phantom/deploy/release_controller.py",
                        "phantom/sim/policy_adapter.py",
                    )
                },
                "campaign_sha256": sha(ROOT / "campaign_snapshot.json"),
                "hardware_sha256": sha(hardware_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
