#!/usr/bin/env python3
"""CPU-only native-start geometry and observed settling audit; no policy scoring."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gripper_clearance(state, cfg, fk):
    """Exact vertical bounds of configured extruded ellipses and housing box.

    A positive separation from the highest environment top proves separation
    for these shapes, regardless of XY. It says nothing about the arm links.
    """
    g = cfg["gripper"]
    tool = fk(state["q"], np.zeros(6))
    rotation = tool[:3, :3] @ Rotation.from_euler("z", g.get("yaw", 0)).as_matrix()
    gap = (
        g["stroke"] / 2 * (1 - np.clip(state["gripper"] / g["pad_touch_command"], 0, 1))
    )
    bodies = [("housing", [0, 0, 0.047], g["housing_size"], "box")]
    for side, sign in (("left", 1), ("right", -1)):
        center = np.array([sign * (gap + g["pad_size"][0] / 2), 0, g["pad_center_z"]])
        for name, offset, size in (
            ("pad", [0, 0, 0], g["pad_size"]),
            ("backing", [sign * 0.008, 0, 0], [0.006, 0.030, 0.060]),
            ("linkage", [sign * 0.012, 0, -0.051], [0.008, 0.016, 0.060]),
        ):
            bodies.append(
                (
                    side + "_" + name,
                    center + offset,
                    size,
                    g.get("contact_shape", "box"),
                )
            )
    top = {
        "packet": cfg["waffle"]["center"][2] + cfg["waffle"]["size"][2] / 2,
        "bin": cfg["bin"]["center"][2] + cfg["bin"]["size"][2],
        "table": cfg["table"]["top_z"],
        "mat": cfg["mat"]["center"][2] + cfg["mat"]["size"][2] / 2,
    }
    result = []
    for name, center, size, shape in bodies:
        z = (tool[:3, 3] + rotation @ center)[2]
        half = np.array(size) / 2
        extent = (
            abs(rotation[2, 0]) * half[0]
            + np.hypot(rotation[2, 1] * half[1], rotation[2, 2] * half[2])
            if shape == "ellipse"
            else abs(rotation[2]) @ half
        )
        lower = float(z - extent)
        result.append(
            {
                "body": name,
                "minimum_z_m": lower,
                "vertical_clearance_lower_bound_m": {
                    key: lower - value for key, value in top.items()
                },
            }
        )
    return {
        "modeled_inner_aperture_m": float(gap * 2),
        "components": result,
        "minimum_environment_vertical_clearance_m": min(
            min(r["vertical_clearance_lower_bound_m"].values()) for r in result
        ),
    }


def observed(folder, state, canonical, thresholds):
    init_path = folder / "initialization.json"
    if not init_path.exists():
        return {
            "status": "not_completed_or_missing_initialization",
            "directory": str(folder),
        }
    init = json.loads(init_path.read_text())
    support = init.get("packet_support_contacts")
    required_finite = np.r_[
        init["settled_robot_q"],
        init["settled_robot_qd"],
        init["settled_packet_center_m"],
        init["packet_velocity_m_s"],
    ]
    values = {
        "all_finite": bool(np.isfinite(required_finite).all()),
        "requested_native_q_exact": bool(
            np.array_equal(init["configured_robot_q"], state["q"])
        ),
        "initial_state_sha_matches": init["policy_initial_state_sha256"]
        == state["_input_sha256"],
        "canonical_packet_exact": init["configured_packet_center_m"] == canonical,
        "packet_displacement_m": init["settling_displacement_m"],
        "packet_speed_m_s": float(np.linalg.norm(init["packet_velocity_m_s"])),
        "q_error_max_rad": init["robot_initial_max_joint_error_rad"],
        "qd_max_rad_s": init["robot_initial_max_joint_speed_rad_s"],
        "settling_duration_s": init["policy_robot_settling_s"],
        "packet_robot_normal_force_n": support["packet_robot_normal_force"]
        if support
        else None,
        "packet_bin_normal_force_n": support["packet_bin_normal_force"]
        if support
        else None,
        "packet_support_contacts": support,
    }
    reasons = []
    for key in (
        "all_finite",
        "requested_native_q_exact",
        "initial_state_sha_matches",
        "canonical_packet_exact",
    ):
        if not values[key]:
            reasons.append(key)
    for key, threshold in (
        ("packet_displacement_m", thresholds["packet_displacement_max_m"]),
        ("q_error_max_rad", thresholds["joint_error_max_rad"]),
        ("qd_max_rad_s", thresholds["joint_speed_max_rad_s"]),
    ):
        if not np.isfinite(values[key]) or values[key] > threshold:
            reasons.append(key)
    if support is None:
        reasons.append("support_evidence_missing")
    elif values["packet_robot_normal_force_n"] > thresholds["packet_robot_force_max_n"]:
        reasons.append("packet_robot_contact_at_start")
    settle_path = folder / "robot_settling.npz"
    if settle_path.exists():
        samples = np.load(settle_path)["samples"]
        tail = samples[samples[:, 0] >= samples[-1, 0] - 0.25]
        values["settling_trace"] = {
            "rows": len(samples),
            "all_finite": bool(np.isfinite(samples).all()),
            "maximum_q_speed_entire_settle_rad_s": float(abs(samples[:, 7:]).max()),
            "maximum_q_speed_last_0_25s_rad_s": float(abs(tail[:, 7:]).max()),
            "maximum_q_error_last_0_25s_rad": float(
                abs(tail[:, 1:7] - state["q"]).max()
            ),
            "tail_metrics_are_diagnostic_only": True,
        }
    else:
        reasons.append("settling_trace_missing")
    # save-stage-only deliberately returns before run.json and sim_trace.npz.
    values["no_rollout_trace_expected_for_save_stage_only"] = True
    return {
        "status": "passed_declared_observable_start_checks"
        if not reasons
        else "failed_or_incomplete_start_checks",
        "failure_reasons": reasons,
        "directory": str(folder),
        "observed": values,
        "input_sha256": {
            p.name: sha(p)
            for p in (init_path, settle_path, folder / "effective_config.json")
            if p.exists()
        },
        "whole_arm_environment_and_self_collision_clearance_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from phantom.sim.kinematics import forward_kinematics

    cfgpath = args.source / "configs/sim/waffles_pick_place.json"
    cfg = json.loads(cfgpath.read_text())
    thresholds = {
        "joint_error_max_rad": 0.02,
        "joint_speed_max_rad_s": 0.05,
        "packet_displacement_max_m": 0.002,
        "packet_robot_force_max_n": 0.1,
        "gripper_geometry_clearance_min_m": 0.002,
    }
    records = []
    for path in sorted(
        (args.source / "configs/sim/initial_states").glob("waffles_aug22_*.json")
    ):
        state = json.loads(path.read_text())
        state["_input_sha256"] = sha(path)
        geometry = gripper_clearance(state, cfg, forward_kinematics)
        pose = forward_kinematics(state["q"])
        native = state["measured_tcp_pose"]
        record = {
            "initial_state_file": str(path),
            "initial_state_sha256": state["_input_sha256"],
            "episode": state["provenance"]["episode"],
            "measured_q": state["q"],
            "measured_gripper": state["gripper"],
            "measured_tcp_m_rad": native,
            "native_field_equality": {
                "q": state["q"] == state["source_samples"]["arm_q"]["value"],
                "gripper": state["gripper"]
                == state["source_samples"]["gripper"]["value"][0],
                "wrist_ft": state["wrist_ft"]
                == state["source_samples"]["arm_ft"]["value"],
            },
            "nominal_fk_error_m": float(np.linalg.norm(pose[:3, 3] - native[:3])),
            "nominal_fk_rotation_error_rad": float(
                (
                    Rotation.from_matrix(pose[:3, :3]).inv()
                    * Rotation.from_rotvec(native[3:])
                ).magnitude()
            ),
            "gripper_geometry": geometry,
            "gripper_clearance_pass": geometry[
                "minimum_environment_vertical_clearance_m"
            ]
            >= thresholds["gripper_geometry_clearance_min_m"],
        }
        if args.runs:
            record["simulation_preflight"] = observed(
                args.runs / ("start_" + path.stem),
                state,
                cfg["waffle"]["center"],
                thresholds,
            )
        records.append(record)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(args.source),
                "canonical_scene_sha256": sha(cfgpath),
                "thresholds": thresholds,
                "threshold_status": "q/qd limits already enforced by runtime; tighter packet/clearance checks proposed before policy outcome inspection, requiring parent freeze decision",
                "records": records,
                "limitations": [
                    "Gripper vertical-clearance proof covers configured pad, backing, linkage and housing only, not UR3 self-collision or all arm/environment pairs.",
                    "The original renderer warmup assigns q repeatedly; the subsequent two-second free-hold equilibrium, unchanged packet and no robot-packet support must be checked independently.",
                    "Initialization telemetry samples packet only after settling; missing transient packet/contact history prevents proof of zero earlier collision.",
                    "Measured q/TCP use native UR controller order and rotvec; factory-calibrated vs nominal FK differences are diagnostics, not arbitrary joint-angle exclusions.",
                    "A failed start is an explicit geometry/initialization exclusion, not a policy failure or reason to substitute a later measured pose.",
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
