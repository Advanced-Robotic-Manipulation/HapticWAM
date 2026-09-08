#!/usr/bin/env python3
"""Evaluate recorded-motion waffle trials from physical traces and real evidence.

This evaluator never runs Isaac or a hardware SDK. It evaluates failures too,
and does not certify an unproven object attachment/pose-control history. Native
tactile signal magnitudes are sensor units; only PhysX force outputs use N.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phantom.sim.geometry import bin_geometry  # noqa: E402
COMPLETE = "ep_waffles_1787395928_000"


def sustained_onset(t, condition, hold=0.20, after=0):
    start = None
    for stamp, yes in zip(t, condition):
        if stamp < after or not yes:
            start = None
        elif start is None:
            start = float(stamp)
        elif stamp - start >= hold - 1e-9:
            return start
    return None


def longest_gap(t, condition, start, end):
    ids = np.flatnonzero((t >= start) & (t <= end))
    longest = 0.0
    begin = None
    if not len(ids):
        return None
    for i in ids:
        if not condition[i] and begin is None:
            begin = float(t[i])
        if condition[i] and begin is not None:
            longest = max(longest, float(t[i]) - begin)
            begin = None
    if begin is not None:
        longest = max(longest, float(t[ids[-1]]) - begin)
    return longest


def corners_world(position, quaternion_wxyz, size):
    vertices = (
        np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        * np.asarray(size)
        / 2
    )
    rotations = Rotation.from_quat(
        np.asarray(quaternion_wxyz)[:, [1, 2, 3, 0]]
    ).as_matrix()
    return (
        np.einsum("nij,kj->nki", rotations, vertices) + np.asarray(position)[:, None, :]
    )


def projected_points(xyz, camera):
    inverse = np.linalg.inv(np.asarray(camera["world_from_cv"], float))
    points = np.asarray(xyz) @ inverse[:3, :3].T + inverse[:3, 3]
    uv = np.c_[
        camera["fx"] * points[:, 0] / points[:, 2] + camera["cx"],
        camera["fy"] * points[:, 1] / points[:, 2] + camera["cy"],
    ]
    return uv, points[:, 2] > 0


def image_alignment(trace, cfg, tracks, pose_fit, events=None):
    """Compare one fixed surface landmark projected from physical object pose.

    The landmark is anchored to the real initial visible color centroid. It is
    an approximate proxy: the centroid of a visible printed region can shift
    under viewpoint/occlusion even without physical slip. No 3D ground-truth
    object trajectory is implied by this monocular construction.
    """
    if pose_fit is None or "waffle_orientation_wxyz" not in trace:
        return {
            "status": "unavailable",
            "reason": "need conditional initial object pose and simulated orientation",
        }
    valid = [
        r
        for r in tracks
        if r.get("visibility") == "visible" and r.get("x_px") not in (None, "")
    ]
    if not valid:
        return {"status": "unavailable", "reason": "no unoccluded real wrapper tracks"}
    first = tracks[0]
    camera = pose_fit["camera"]
    transform = np.asarray(camera["world_from_cv"], float)
    ray = transform[:3, :3] @ np.array(
        [
            (float(first["x_px"]) - camera["cx"]) / camera["fx"],
            (float(first["y_px"]) - camera["cy"]) / camera["fy"],
            1,
        ]
    )
    center = np.asarray(pose_fit["conditional_initial_packet_center_m"])
    z = center[2] + np.asarray(pose_fit["assumed_packet_size_m"])[2] / 2
    initial_world = transform[:3, 3] + ray * ((z - transform[2, 3]) / ray[2])
    local = (
        Rotation.from_euler("z", pose_fit["conditional_initial_packet_yaw_rad"])
        .inv()
        .apply(initial_world - center)
    )
    times = np.asarray([float(r["camera_t_s"]) for r in valid])
    mask = (times >= trace["t"][0]) & (times <= trace["t"][-1])
    times = times[mask]
    real = np.asarray([[float(r["x_px"]), float(r["y_px"])] for r in valid])[mask]
    if len(times) < 2:
        return {"status": "unavailable", "reason": "no common image time support"}
    positions = np.stack(
        [
            np.interp(times, trace["t"], trace["waffle_position"][:, i])
            for i in range(3)
        ],
        1,
    )
    rotations = Slerp(
        trace["t"],
        Rotation.from_quat(trace["waffle_orientation_wxyz"][:, [1, 2, 3, 0]]),
    )(times)
    predicted, in_front = projected_points(
        positions + rotations.apply(local), cfg["camera"]
    )
    error = np.linalg.norm(predicted - real, axis=1)
    error = error[in_front]
    if not len(error):
        return {"status": "unavailable", "reason": "projected landmark behind camera"}
    delta = (predicted - real)[in_front]
    scored_t = times[in_front]
    phases = {}
    if events is not None:
        contact = events.get("dual_pad_contact_above_2_sensor_units_s")
        lift = events.get("lift_20mm_above_min_s")
        release = events.get("release_closure_drop_0p15_s")
        bounds = {
            "before_contact": (0, contact),
            "carry": (lift, release or float(times[-1])),
            "after_release": (release, float(times[-1]) + 0.001),
        }
        for name, (start, end) in bounds.items():
            if start is None or end is None:
                continue
            selected = (scored_t >= start) & (scored_t < end)
            if selected.any():
                phases[name] = {
                    "frames": int(selected.sum()),
                    "rmse_px": float(np.sqrt(np.mean(error[selected] ** 2))),
                    "mean_predicted_minus_real_xy_px": delta[selected].mean(0).tolist(),
                }
    return {
        "status": "measured_proxy",
        "mapping": "real initial colored-region centroid backprojected to assumed object top, fixed in object frame, then projected from simulated pose",
        "local_landmark_m": local.tolist(),
        "frames_scored": len(error),
        "real_visible_frames": len(valid),
        "coverage_fraction": len(error) / len(valid),
        "excluded_nonvisible_frames": len(tracks) - len(valid),
        "rmse_px": float(np.sqrt(np.mean(error**2))),
        "median_px": float(np.median(error)),
        "p95_px": float(np.percentile(error, 95)),
        "maximum_px": float(error.max()),
        "mean_predicted_minus_real_xy_px": delta.mean(0).tolist(),
        "phase_errors": phases,
        "trajectory": [
            {
                "t_s": float(stamp),
                "real_xy_px": actual.tolist(),
                "projected_xy_px": simulated.tolist(),
                "error_px": float(err),
            }
            for stamp, actual, simulated, err in zip(
                scored_t, real[in_front], predicted[in_front], error
            )
        ],
        "scored_time_range_s": [float(times.min()), float(times.max())],
        "limitation": "This is a conditional projected wrapper-landmark proxy, not photometric accuracy or calibrated 3D object-pose error.",
    }


def evaluate_arrays(trace, run, cfg, real, timeline, tracks=None, pose_fit=None):
    t = np.asarray(trace["t"], float)
    if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(
            "simulation timestamps must be finite, strictly increasing, and contain at least three rows"
        )
    required = ("q", "target_q", "tcp", "waffle_position", "gripper")
    for key in required:
        if (
            key not in trace
            or len(trace[key]) != len(t)
            or not np.isfinite(trace[key]).all()
        ):
            raise ValueError(f"missing, misaligned or nonfinite trace stream: {key}")
    ev = timeline["telemetry_events"]
    complete = bool(timeline["visual_adjudication"]["complete_pick_place"])
    reference_contact = ev.get(
        "dual_pad_contact_above_2_sensor_units_s", ev.get("dual_pad_contact_over_2N_s")
    )
    reference_lift = ev["lift_20mm_above_min_s"]
    reference_release = ev["release_closure_drop_0p15_s"]
    carry_start = (
        (reference_lift + 0.25) if reference_lift is not None else float(t[-1]) + 1
    )
    carry_end = (
        (reference_release - 0.25)
        if reference_release is not None
        else float(real["t"][-1]) - 0.15
    )
    carry = (t >= carry_start) & (t <= carry_end)
    gates = {}
    metrics = {}
    intervals = np.diff(t)
    metrics["trace_sampling"] = {
        "rows": len(t),
        "median_interval_s": float(np.median(intervals)),
        "maximum_interval_s": float(intervals.max()),
        "event_onset_sampling_uncertainty_s": float(intervals.max()),
        "limitation": "Contact and release boundaries are sampled; an onset may precede the first positive row by up to one interval. Subsample transients are not certified.",
    }

    def gate(name, value, criterion):
        gates[name] = {
            "pass": None if value is None else bool(value),
            "criterion": criterion,
        }

    coverage = bool(t[0] <= 0.1 and t[-1] >= float(real["t"][-1]) - 0.1)
    gate(
        "recording_time_coverage",
        coverage,
        "cover the real recording through its last frame within0.1s",
    )
    expected_name = Path(timeline["episode"]).name
    gate(
        "same_reference_episode",
        Path(run.get("episode", "")).name == expected_name,
        "run and reference episode names match",
    )
    provenance = run.get("object_dynamics")
    free = (
        None
        if provenance is None
        else (
            provenance.get("rigid_body_dynamic") is True
            and provenance.get("kinematic") is False
            and provenance.get("attachments") == []
            and provenance.get("pose_writes_after_initialization") == 0
        )
    )
    gate(
        "free_dynamic_object_provenance",
        free,
        "audited dynamic free body, no attachment and no post-initialization pose writes",
    )
    gate(
        "dynamic_articulation_mode",
        run.get("mode") in ("dynamics", "policy"),
        "articulation must be driven by physics; kinematic replay/contact-probe do not validate recorded pick/place",
    )
    if "physics_t" in trace:
        skew = float(np.max(np.abs(trace["physics_t"] - t)))
        metrics["physics_clock_max_error_s"] = skew
        gate(
            "physics_clock_alignment",
            skew <= 0.001,
            "physical simulation clock follows trace time within1ms",
        )
    else:
        gate("physics_clock_alignment", None, "physics_t is required")
    drive_error = np.abs(np.asarray(trace["q"]) - np.asarray(trace["target_q"]))
    reference_joint_time = real.get("native_arm_q_t", real["t"])
    reference_joint_values = real.get("native_arm_q", real["q"])
    reference_q = np.stack(
        [
            np.interp(t, reference_joint_time, reference_joint_values[:, j])
            for j in range(6)
        ],
        1,
    )
    metrics["actual_drive_tracking"] = {
        "reference_source": "native_arm_q"
        if "native_arm_q" in real
        else "replay_grid_q",
        "p95_absolute_joint_error_rad": float(np.percentile(drive_error, 95)),
        "max_absolute_joint_error_rad": float(drive_error.max()),
        "recorded_joint_rmse_rad": float(
            np.sqrt(np.mean((trace["q"] - reference_q) ** 2))
        ),
    }
    gate(
        "actual_drive_tracking",
        np.percentile(drive_error, 95) <= 0.05 and drive_error.max() <= 0.20,
        "actual-vs-command joint error p95<=.05rad and max<=.20rad",
    )
    if run.get("mode") == "dynamics":
        gate(
            "recorded_joint_motion",
            metrics["actual_drive_tracking"]["recorded_joint_rmse_rad"] <= 0.05,
            "actual recorded-motion joint RMSE<=.05rad",
        )
    normals = trace.get("pad_packet_normal_force")
    contact = None
    if (
        normals is not None
        and np.asarray(normals).shape == (len(t), 2)
        and np.isfinite(normals).all()
    ):
        bilateral = np.all(np.asarray(normals) > 0.05, axis=1)
        contact = sustained_onset(t, bilateral, 0.20)
        fraction = float(np.mean(bilateral[carry])) if carry.any() else 0.0
        gap = longest_gap(t, bilateral, carry_start, carry_end)
        metrics["packet_contact"] = {
            "source": "PhysX normal forces filtered between packet and each pad body; N",
            "onset_s": contact,
            "reference_onset_s": reference_contact,
            "reference_force_units": "native sensor units; no Newton comparison",
            "bilateral_carry_fraction": fraction,
            "longest_bilateral_gap_s": gap,
            "carry_interval_s": [carry_start, carry_end],
            "peak_per_pad_N": np.max(normals, axis=0).tolist(),
            "surface_limit": "pad-body filtering can include backing/linkage colliders; it does not independently identify a gel face",
        }
        gate(
            "timed_bilateral_packet_contact",
            contact is not None and abs(contact - reference_contact) <= 0.5,
            "both packet-filtered pad normal forces>.05N for.2s; onset within.5s of real sensor contact",
        )
        gate(
            "sustained_bilateral_carry_contact",
            fraction >= 0.85 and gap is not None and gap <= 0.30,
            "bilateral contact on>=85% of carry frames, no sampled gap>.30s",
        )
    else:
        normals = None
        gate(
            "timed_bilateral_packet_contact",
            None,
            "requires packet-filtered normal forces; net table/gripper forces are insufficient",
        )
        gate(
            "sustained_bilateral_carry_contact",
            None,
            "requires packet-filtered normal forces",
        )
    position = np.asarray(trace["waffle_position"])
    initial = np.median(position[t <= min(1, t[-1])], axis=0)
    lift_onset = sustained_onset(t, position[:, 2] > initial[2] + 0.02, 0.15)
    max_lift = float(np.max(position[:, 2] - initial[2]))
    metrics["physical_object_lift"] = {
        "onset_20mm_s": lift_onset,
        "reference_tcp_lift_onset_s": reference_lift,
        "maximum_object_lift_m": max_lift,
        "reference_object_3d_lift_m": None,
        "reference_limit": "real calibrated 3D object pose is unavailable; onset comparison uses measured TCP lift plus visual lift interval",
    }
    gate(
        "physical_lift",
        lift_onset is not None
        and abs(lift_onset - reference_lift) <= 0.6
        and max_lift >= 0.10,
        "free packet rises>=10cm;20mm onset within.6s of measured/visually corroborated lift",
    )
    if carry.sum() >= 3:
        tcp_rotation = Rotation.from_rotvec(np.asarray(trace["tcp"])[carry, 3:])
        relative = tcp_rotation.inv().apply(
            position[carry] - np.asarray(trace["tcp"])[carry, :3]
        )
        anchor = np.median(relative[: min(5, len(relative))], axis=0)
        slip = np.linalg.norm(relative - anchor, axis=1)
        metrics["retention"] = {
            "tcp_frame_translation_slip_p95_m": float(np.percentile(slip, 95)),
            "tcp_frame_translation_slip_max_m": float(slip.max()),
            "carry_frames": int(carry.sum()),
        }
        gate(
            "packet_retention",
            np.percentile(slip, 95) <= 0.020 and slip.max() <= 0.050,
            "packet stays within20mm(p95)/50mm(max) of captured TCP-frame position during carry",
        )
    else:
        gate("packet_retention", False, "at least three carry samples required")
    if complete:
        after = t > reference_contact
        closure = np.asarray(trace["gripper"])[:, 0]
        running = np.maximum.accumulate(np.where(after, closure, 0))
        release = sustained_onset(
            t, after & (closure < running - 0.15), 0.1, reference_contact
        )
        unloaded = (
            sustained_onset(
                t,
                np.all(np.asarray(normals) < 0.05, axis=1),
                0.25,
                (contact + 0.2) if contact is not None else reference_release - 1,
            )
            if normals is not None
            else None
        )
        metrics["release"] = {
            "closure_drop_onset_s": release,
            "reference_release_s": reference_release,
            "packet_filtered_both_pads_unloaded_s": unloaded,
        }
        gate(
            "timed_release",
            release is not None
            and abs(release - reference_release) <= 0.5
            and unloaded is not None
            and abs(unloaded - reference_release) <= 1,
            "closure decreases>=.15 within.5s of real release and both packet contacts unload within1s",
        )
        final_start = max(reference_release + 1, float(real["t"][-1]) - 2)
        final = t >= final_start
        if "waffle_orientation_wxyz" in trace and final.sum() >= 10:
            corners = corners_world(
                position[final],
                trace["waffle_orientation_wxyz"][final],
                cfg["waffle"]["size"],
            )
            geometry = bin_geometry(cfg["bin"])
            lower, upper = geometry.interior_bounds
            bin_corners = geometry.to_interior_frame(corners)
            low, high = lower[:2], upper[:2]
            inside_xy = np.all(
                (bin_corners[:, :, :2] >= low - 0.005)
                & (bin_corners[:, :, :2] <= high + 0.005),
                axis=(1, 2),
            )
            inside_z = (corners[:, :, 2].min(1) >= lower[2] - 0.005) & (
                corners[:, :, 2].max(1) <= upper[2] + 0.01
            )
            contained = inside_xy & inside_z
            displacement = np.linalg.norm(
                position[final] - np.median(position[final], axis=0), axis=1
            )
            speed = np.linalg.norm(
                np.gradient(position[final], t[final], axis=0), axis=1
            )
            rotations = Rotation.from_quat(
                trace["waffle_orientation_wxyz"][final][:, [1, 2, 3, 0]]
            )
            angular = (rotations[1:] * rotations[:-1].inv()).magnitude() / np.diff(
                t[final]
            )
            metrics["final_bin"] = {
                "window_s": [float(t[final][0]), float(t[final][-1])],
                "contained_fraction": float(np.mean(contained)),
                "minimum_oriented_corner_z_m": float(corners[:, :, 2].min()),
                "position_excursion_max_m": float(displacement.max()),
                "speed_p95_m_s": float(np.percentile(speed, 95)),
                "angular_speed_p95_rad_s": float(np.percentile(angular, 95)),
            }
            gate(
                "final_oriented_bin_containment",
                contained.all() and t[final][-1] - t[final][0] >= 1.8,
                "all oriented packet corners stay inside bin throughout final>=1.8s;5mm spatial tolerance",
            )
            gate(
                "post_release_settling",
                displacement.max() <= 0.01
                and np.percentile(speed, 95) <= 0.025
                and np.percentile(angular, 95) <= 0.30,
                "final position excursion<=10mm, p95 speed<=25mm/s, p95 angular speed<=.30rad/s",
            )
        else:
            gate(
                "final_oriented_bin_containment",
                None,
                "requires object orientations and>=10 final-window samples",
            )
            gate(
                "post_release_settling",
                None,
                "requires sufficient post-release samples",
            )
    metrics["image_alignment"] = (
        image_alignment(trace, cfg, tracks, pose_fit, ev)
        if tracks
        else {"status": "unavailable"}
    )
    image = metrics["image_alignment"]
    if image.get("status") == "measured_proxy":
        gate(
            "projected_wrapper_trajectory",
            image["coverage_fraction"] >= 0.95
            and image["rmse_px"] <= 12
            and image["p95_px"] <= 20,
            "conditional visible-wrapper landmark proxy:>=95% coverage,RMSE<=12px,p95<=20px; exclude flagged occlusion/blur",
        )
    else:
        gate(
            "projected_wrapper_trajectory",
            None,
            "requires comparable real wrapper tracks and initial pose mapping",
        )
    failed = [k for k, v in gates.items() if v["pass"] is False]
    unknown = [k for k, v in gates.items() if v["pass"] is None]
    physical_failed = [
        name for name in failed if name != "projected_wrapper_trajectory"
    ]
    physical_unknown = [
        name for name in unknown if name != "projected_wrapper_trajectory"
    ]
    # Completing the physical task and reproducing its timing are distinct.
    # Keep every fidelity failure above; this outcome view never overrides it.
    outcome_names = [
        "recording_time_coverage",
        "same_reference_episode",
        "free_dynamic_object_provenance",
        "dynamic_articulation_mode",
        "physics_clock_alignment",
        "actual_drive_tracking",
        "sustained_bilateral_carry_contact",
        "packet_retention",
    ]
    if complete:
        outcome_names.extend(
            ["final_oriented_bin_containment", "post_release_settling"]
        )
    outcome_gates = {name: gates[name]["pass"] for name in outcome_names}
    outcome_gates["packet_lift_at_least_10cm"] = bool(
        lift_onset is not None and max_lift >= 0.10
    )
    if complete:
        outcome_gates["release_and_contact_unloading_observed"] = bool(
            release is not None and unloaded is not None
        )
    outcome_verdict = (
        "fail"
        if any(value is False for value in outcome_gates.values())
        else "unverified"
        if any(value is None for value in outcome_gates.values())
        else "achieved_in_simulation"
    )
    return {
        "schema_version": 1,
        "task": "full_recorded_pick_place" if complete else "recorded_pick_lift_only",
        "verdict": "fail"
        if failed
        else "unverified"
        if unknown
        else "pass_within_declared_tolerances",
        "physical_verdict": "fail"
        if physical_failed
        else "unverified"
        if physical_unknown
        else "pass_within_declared_tolerances",
        "task_outcome_verdict": outcome_verdict,
        "task_outcome_gates": outcome_gates,
        "image_proxy_verdict": "pass"
        if gates["projected_wrapper_trajectory"]["pass"] is True
        else "fail"
        if gates["projected_wrapper_trajectory"]["pass"] is False
        else "unverified",
        "failed_gates": failed,
        "unverified_gates": unknown,
        "gates": gates,
        "metrics": metrics,
        "reference_events": ev,
        "reference_visual_adjudication": timeline["visual_adjudication"],
        "limitations": [
            "A partial pick/lift recording cannot certify placement.",
            "Thresholds are explicit engineering acceptance proposals, not validated sim-to-real transfer bounds.",
            "Real sensor magnitudes and simulated Newton-valued contact forces are not directly calibrated here.",
            "Free-body provenance is audited run metadata; trace arrays alone cannot disprove hidden attachment/teleport code.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument(
        "--evidence-root",
        type=Path,
        default=ROOT / "artifacts/isaac_waffles/evidence/pick_place_analysis",
    )
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    run = json.loads((args.run / "run.json").read_text())
    reference = (
        args.reference
        or ROOT / "artifacts/isaac_waffles/evidence/fit" / Path(run["episode"]).name
    )
    evidence = args.evidence_root / reference.name
    trace = dict(np.load(args.run / "sim_trace.npz", allow_pickle=False))
    real = dict(np.load(reference / "replay.npz", allow_pickle=False))
    cfg = json.loads((args.run / "effective_config.json").read_text())
    timeline = json.loads((evidence / "timeline.json").read_text())
    tracks = list(csv.DictReader((evidence / "image_tracks.csv").open()))
    pose_name = (
        "teleop_initial_packet_pose.json"
        if reference.name == COMPLETE
        else "deployment_initial_packet_pose.json"
    )
    pose_path = args.evidence_root / pose_name
    pose = (
        json.loads(pose_path.read_text())
        if reference.name in (COMPLETE, "ep_teacher_waffles_1788535016_005")
        and pose_path.exists()
        else None
    )
    result = evaluate_arrays(trace, run, cfg, real, timeline, tracks, pose)
    result.update(run=str(args.run), reference=str(reference))
    destination = args.out or args.run / "pick_place_evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "failed": result["failed_gates"],
                "unverified": result["unverified_gates"],
                "output": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
