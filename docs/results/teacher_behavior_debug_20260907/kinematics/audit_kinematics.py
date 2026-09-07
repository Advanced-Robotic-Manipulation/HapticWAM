#!/usr/bin/env python3
"""Read-only CPU teacher trajectory/IK audit; no physics or policy execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
KINEMATICS_SHA = "120c6caadfeeb9112c1bbb292e76c0f07c76ef4326059756c372ad340ebf73b6"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s]


def radius(q):
    q = np.asarray(q)
    return np.sqrt(
        0.24365**2 + 0.21325**2 + 0.11235**2 + 2 * 0.24365 * 0.21325 * np.cos(q[..., 2])
    )


def angle(a, b):
    return float(
        np.rad2deg(
            (Rotation.from_rotvec(a) * Rotation.from_rotvec(b).inv()).magnitude()
        )
    )


def path_geometry(start_q, endpoint, ik, fk, *, same_orientation):
    """Offline Cartesian homotopy; proves only this nominal IK path, no contacts."""
    q = np.asarray(start_q).copy()
    start = fk(q)
    target = np.asarray(endpoint).copy()
    if same_orientation:
        target[3:] = start[3:]
    rotations = Slerp([0.0, 1.0], Rotation.from_rotvec([start[3:], target[3:]]))
    n = max(40, int(np.ceil(np.linalg.norm(start[:3] - target[:3]) / 0.005)))
    records = []
    for u in np.linspace(0, 1, n + 1)[1:]:
        pose = np.r_[
            start[:3] + u * (target[:3] - start[:3]), rotations([u]).as_rotvec()[0]
        ]
        result = ik(pose, q, max_joint_delta_rad=0.35)
        records.append(
            {
                "fraction": float(u),
                "IK_success": bool(result.success),
                "reason": result.reason,
                "wrist_radius_m": float(radius(result.q)),
                "q2_rad": float(result.q[2]),
                "joint_step_max_rad": float(np.max(abs(result.q - q))),
                "position_error_m": result.position_error_m,
                "rotation_error_rad": result.rotation_error_rad,
            }
        )
        if not result.success:
            break
        q = result.q
    complete = len(records) == n and all(x["IK_success"] for x in records)
    return {
        "complete": complete,
        "steps": len(records),
        "planned_steps": n,
        "start_pose": start.tolist(),
        "endpoint_pose": target.tolist(),
        "max_radius_m": max(x["wrist_radius_m"] for x in records),
        "inside_measured_radius_limit_all_steps": all(
            x["wrist_radius_m"] <= 0.468 for x in records
        ),
        "max_raw_joint_step_rad": max(x["joint_step_max_rad"] for x in records),
        "final_q": q.tolist(),
        "last_step": records[-1],
        "limitation": "Nominal local IK at geometric fractions, not timed commands. No collision, payload, gripper, safety hitbox or hardware path certification.",
    }


def analyze(group, root, seed):
    from phantom.sim.kinematics import geometric_jacobian

    name = f"fta1500_nfe1_k4__successful_anchor__seed{seed}"
    folder = root / "rollouts" / name
    score_path = root / "analysis/trials" / f"{name}.json"
    score = read(score_path)
    assert score["metrics"]["valid_for_scoring"], (
        name,
        score["metrics"]["invalid_reasons"],
    )
    for f, digest in score["input_sha256"].items():
        assert sha(folder / f) == digest, (name, f)
    cfg, info = (
        read(folder / "effective_config.json"),
        read(folder / "policy_info.json"),
    )
    all_e = rows(folder / "execution_trace.jsonl")
    stop = next((r for r in all_e if r["stopped"]), None)
    finish = next((r for r in all_e if r["diagnostics"].get("completed_reason")), None)
    end_t = stop["t"] if stop else finish["t"] if finish else all_e[-1]["t"]
    e = [r for r in all_e if r["t"] <= end_t]
    t = np.array([r["t"] for r in e])
    tcp = np.array([r["measured_tcp"] for r in e])
    req = np.array([r["requested_tcp"] for r in e])
    q = np.array([r["measured_q"] for r in e])
    qd = np.array([r["measured_qd"] for r in e])
    cq = np.array([r["target_q"] for r in e])
    wr = radius(q)
    cwr = radius(cq)
    metrics = score["metrics"]
    times = metrics["event_times_s"]
    release = info["placement_release"]
    low = np.array(release["tcp_min_m"])
    high = np.array(release["tcp_max_m"])
    boxcenter = (low + high) / 2
    inside = np.logical_and(tcp[:, :3] >= low, tcp[:, :3] <= high).all(axis=1)
    with np.load(folder / "sim_trace.npz") as tr:
        st = tr["t"].copy()
        obj = tr["waffle_position"].copy()
        force = (
            tr["pad_packet_normal_force"].copy()
            if "pad_packet_normal_force" in tr
            else tr["pad_packet_force"].copy()
        )

    def snap(i):
        j = int(np.argmin(abs(st - t[i])))
        return {
            "t_s": float(t[i]),
            "tcp": tcp[i].tolist(),
            "requested_tcp": req[i].tolist(),
            "q": q[i].tolist(),
            "target_q": cq[i].tolist(),
            "wrist_radius_m": float(wr[i]),
            "command_wrist_radius_m": float(cwr[i]),
            "tracking_position_error_m": float(np.linalg.norm(req[i, :3] - tcp[i, :3])),
            "tracking_rotation_error_deg": angle(req[i, 3:], tcp[i, 3:]),
            "tool_z_axis_base": Rotation.from_rotvec(tcp[i, 3:])
            .as_matrix()[:, 2]
            .tolist(),
            "measured_closure": float(e[i]["measured_gripper"][0]),
            "object_center_m": obj[j].tolist(),
            "pad_packet_normal_force_n": force[j].tolist(),
            "release_volume_xyz_distance_m": np.maximum(
                np.maximum(low - tcp[i, :3], tcp[i, :3] - high), 0
            ).tolist(),
        }

    stages = {
        k: snap(int(np.argmin(abs(t - v))))
        for k, v in times.items()
        if v is not None and v <= end_t
    }
    for key, thresh in [("first_radius_440mm", 0.440), ("first_radius_460mm", 0.460)]:
        hits = np.flatnonzero((wr >= thresh) & (t >= times["acquisition"]))
        stages[key] = snap(int(hits[0])) if len(hits) else None
    for dt in [4, 2, 1, 0]:
        stages[f"terminal_minus_{dt}s"] = snap(int(np.argmin(abs(t - (end_t - dt)))))
    lift_t = times["lift"] if times["lift"] is not None else times["acquisition"]
    active_ids = sorted(
        {
            r["active_replan_id"]
            for r in e
            if r["t"] >= end_t - 2 and r.get("active_replan_id") is not None
        }
    )
    plans = read(folder / "planner_trace.json")
    raw = []
    for plan in plans:
        if plan["replan_id"] not in active_ids:
            continue
        a = np.asarray(plan["actions"])
        head = a[:10].sum(axis=0)
        p = np.asarray(plan["t0_pose"])
        toward = boxcenter[:2] - p[:2]
        cosine = (
            np.dot(head[:2], toward)
            / (np.linalg.norm(head[:2]) * np.linalg.norm(toward))
            if np.linalg.norm(head[:2]) * np.linalg.norm(toward) > 0
            else None
        )
        raw.append(
            {
                "id": plan["replan_id"],
                "capture_t": plan["t"],
                "status": plan["status"],
                "capture_tcp": p.tolist(),
                "raw_first10_delta_xyz_m": head[:3].tolist(),
                "raw_first10_additive_rotation_vector_delta_rad": head[3:6].tolist(),
                "raw_first10_endpoint_SO3_change_deg": angle(p[3:] + head[3:6], p[3:]),
                "raw_xy_cosine_toward_release_volume_center": float(cosine)
                if cosine is not None
                else None,
                "native_latency_s": plan["latency_s"],
                "scope": "Raw first10 proposal steps of plans actually active within terminal2s. Includes predicted steps that may not have played; do not identify this sum with executed motion.",
            }
        )
    in_lift = np.flatnonzero(t >= lift_t)
    last_i = int(np.argmin(abs(t - (end_t - 2))))
    delta = tcp[-1, :3] - tcp[last_i, :3]
    translation_radial = rotation_radial = 0.0
    for j in range(last_i, len(t) - 1):
        middle = (q[j] + q[j + 1]) / 2
        jac = geometric_jacobian(middle)
        gradient = np.zeros(6)
        gradient[2] = -0.24365 * 0.21325 * np.sin(middle[2]) / radius(middle)
        task_gradient = gradient @ np.linalg.pinv(jac, rcond=1e-8)
        displacement = jac @ (q[j + 1] - q[j])
        translation_radial += task_gradient[:3] @ displacement[:3]
        rotation_radial += task_gradient[3:] @ displacement[3:]
    errors = np.linalg.norm(req[:, :3] - tcp[:, :3], axis=1)
    rotational = (
        Rotation.from_rotvec(req[:, 3:]) * Rotation.from_rotvec(tcp[:, 3:]).inv()
    ).magnitude()
    result = {
        "group": group,
        "seed": seed,
        "directory": str(folder),
        "score_sha256": sha(score_path),
        "input_sha256": score["input_sha256"],
        "additional_input_sha256": {
            "delivered_plans.jsonl": sha(folder / "delivered_plans.jsonl")
        },
        "outcomes": metrics["outcomes"],
        "event_times_s": times,
        "physical_object_metrics": metrics["object"],
        "stop_t_s": None if stop is None else stop["t"],
        "stop_events": [] if stop is None else stop["diagnostics"].get("safety_events"),
        "finish_t_s": None if finish is None else finish["t"],
        "release_volume": release,
        "scene_bin": cfg["bin"],
        "first_measured_inside_release_volume_t_s": float(t[np.flatnonzero(inside)[0]])
        if inside.any()
        else None,
        "stages": stages,
        "raw_active_terminal_plans": raw,
        "summary": {
            "max_tcp_z_m": float(tcp[:, 2].max()),
            "max_tcp_z_t_s": float(t[np.argmax(tcp[:, 2])]),
            "max_wrist_radius_m": float(wr.max()),
            "min_abs_elbow_rad": float(abs(q[:, 2]).min()),
            "max_abs_qd_rad_s": float(abs(qd).max()),
            "max_raw_measured_joint_step_rad": float(abs(np.diff(q, axis=0)).max()),
            "ik_reject_count": sum(r.get("ik_success") is False for r in e),
            "tracking_max_position_m": float(errors.max()),
            "tracking_p95_position_m": float(np.quantile(errors, 0.95)),
            "tracking_max_rotation_deg": float(np.rad2deg(rotational.max())),
            "terminal2s_actual_delta_xyz_m": delta.tolist(),
            "terminal2s_actual_SO3_change_deg": angle(tcp[-1, 3:], tcp[last_i, 3:]),
            "terminal2s_wrist_radius_change_m": float(wr[-1] - wr[last_i]),
            "terminal2s_differential_translation_contribution_m": float(
                translation_radial
            ),
            "terminal2s_differential_rotation_contribution_m": float(rotation_radial),
            "radial_decomposition_definition": "Midpoint nominal Jacobian decomposition of measured joint increments into TCP translation/angular twist. Contributions sum approximately to actual radius change; coordinate-based differential attribution, not a causal policy intervention.",
            "lift_or_acq_to_terminal_tcp_delta_xyz_m": (
                tcp[-1, :3] - tcp[in_lift[0], :3]
            ).tolist(),
            "lift_or_acq_to_terminal_SO3_change_deg": angle(
                tcp[-1, 3:], tcp[in_lift[0], 3:]
            ),
            "raw_proposal_rotation_convention": "Additive UR rotation-vector coordinates; not Euler angles or world angular velocities.",
        },
    }
    # Compact samples are for a descriptive figure only, not a new scoring grid.
    sample = np.arange(0, len(t), max(1, round(0.1 / np.median(np.diff(t)))))
    plot = {
        "t": t[sample].tolist(),
        "tcp": tcp[sample].tolist(),
        "radius": wr[sample].tolist(),
        "tracking": errors[sample].tolist(),
        "q2": q[sample, 2].tolist(),
    }
    return result, plot, e


def plot_results(results, plots, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for r, p in zip(results, plots):
        x = np.array(p["tcp"])
        t = np.array(p["t"])
        lift = r["event_times_s"]["lift"]
        color = (
            "tab:green"
            if r["outcomes"]["full_task"]
            else "tab:orange"
            if r["outcomes"]["lifted"]
            else ".65"
        )
        label = (
            f"{r['group'].split('_')[0]} {r['seed']}"
            if r["outcomes"]["full_task"]
            else None
        )
        axes[0, 0].plot(
            x[:, 1], x[:, 2], color=color, alpha=0.9 if label else 0.5, label=label
        )
        axes[0, 1].plot(x[:, 0], x[:, 1], color=color, alpha=0.9 if label else 0.5)
        if lift is not None:
            axes[1, 0].plot(
                t - lift, p["radius"], color=color, alpha=0.9 if label else 0.5
            )
            axes[1, 1].plot(
                t - lift,
                1000 * np.array(p["tracking"]),
                color=color,
                alpha=0.9 if label else 0.5,
            )
    release = results[0]["release_volume"]
    lo = release["tcp_min_m"]
    hi = release["tcp_max_m"]
    from matplotlib.patches import Rectangle

    axes[0, 0].add_patch(
        Rectangle(
            (lo[1], lo[2]),
            hi[1] - lo[1],
            hi[2] - lo[2],
            fill=False,
            color="tab:blue",
            linestyle="--",
        )
    )
    axes[0, 1].add_patch(
        Rectangle(
            (lo[0], lo[1]),
            hi[0] - lo[0],
            hi[1] - lo[1],
            fill=False,
            color="tab:blue",
            linestyle="--",
        )
    )
    axes[0, 0].set(
        xlabel="TCP base Y (m)",
        ylabel="TCP base Z (m)",
        title="Side view; dashed = release volume projection",
    )
    axes[0, 1].set(
        xlabel="TCP base X (m)",
        ylabel="TCP base Y (m)",
        title="Top view; green = strict placement",
    )
    axes[1, 0].axhline(0.468, color="red", linestyle="--")
    axes[1, 0].set(
        xlabel="Time from sustained lift (s)",
        ylabel="Wrist radius (m)",
        title="Only cases with a sustained lift",
    )
    axes[1, 1].set(
        xlabel="Time from sustained lift (s)",
        ylabel="Requested–measured TCP error (mm)",
        title="Pose tracking; no collision certification",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Saved teacher trajectories: V5 reserved12 + V7 K4 development2\nOrange: lift without placement; grey: no sustained lift; timing differs",
        fontsize=12,
    )
    fig.savefig(out / "trajectory_comparison.png", dpi=160)
    fig.savefig(out / "trajectory_comparison.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--plot", action="store_true", help="Also render figures; requires matplotlib"
    )
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Render existing numeric outputs without rereading trials or solving IK",
    )
    args = ap.parse_args()
    os.nice(19)
    if args.plot_only:
        plot_results(
            read(args.out / "kinematics_audit.json")["cases"],
            read(args.out / "plot_samples.json"),
            args.out,
        )
        return
    if args.out.exists():
        raise FileExistsError("Preserve existing audit; choose a new directory")
    source = args.base / "source_teacher_anchor_minimal_v5"
    assert sha(source / "phantom/sim/kinematics.py") == KINEMATICS_SHA
    sys.path.insert(0, str(source))
    from phantom.sim.kinematics import forward_pose, inverse_kinematics

    results = []
    plots = []
    raw = []
    for group, root, seeds in [
        (
            "v5_confirmation",
            args.base / "runs/teacher_success_anchor_v5/confirmation",
            range(904501, 904513),
        ),
        (
            "v7_rpc_k4",
            args.base / "runs/teacher_success_anchor_v7/diagnostic",
            [904301, 904302],
        ),
    ]:
        for seed in seeds:
            r, p, e = analyze(group, root, seed)
            results.append(r)
            plots.append(p)
            raw.append(e)
            print(
                json.dumps(
                    {
                        "case": f"{group}/{seed}",
                        "stop": r["stop_events"],
                        "full": r["outcomes"]["full_task"],
                    }
                ),
                flush=True,
            )
    successes = [i for i, r in enumerate(results) if r["outcomes"]["full_task"]]
    assert len(successes) == 2
    reference = results[successes[0]]
    endpoint = reference["stages"]["release_in_bin"]["tcp"]
    for r in results:
        start = r["stages"]["terminal_minus_2s"]
        r["nominal_box_feasibility"] = {
            "reference_case": f"{reference['group']}/{reference['seed']}",
            "reference_pose_event": "release_in_bin (observed successful robot TCP)",
            "constant_own_orientation": path_geometry(
                start["q"],
                endpoint,
                inverse_kinematics,
                forward_pose,
                same_orientation=True,
            ),
            "interpolate_to_success_orientation": path_geometry(
                start["q"],
                endpoint,
                inverse_kinematics,
                forward_pose,
                same_orientation=False,
            ),
        }
    args.out.mkdir(parents=True)
    output = {
        "scope": "14 immutable recorded teacher trials; descriptive kinematics and CPU nominal IK only. V5 confirmation12 and V7 development2 remain separate; no score change, inference, simulator or hardware.",
        "kinematics_source": str(source / "phantom/sim/kinematics.py"),
        "kinematics_sha256": KINEMATICS_SHA,
        "definitions": {
            "orientation": "UR axis-angle rotation vector; actual changes use SO(3) geodesic degrees.",
            "radius": "Nominal UR3 shoulder-to-DH4 wrist-center radius from measured elbow q2. Configured stop .468m.",
            "targets": "requested_tcp is executor request; target_q is accepted drive target; measured fields are physics feedback.",
            "stage_time": "Nearest125Hz executor sample to frozen15Hz physical event; no thresholds respecified.",
        },
        "cases": results,
    }
    (args.out / "kinematics_audit.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    (args.out / "plot_samples.json").write_text(
        json.dumps(plots, separators=(",", ":")) + "\n"
    )
    if args.plot:
        plot_results(results, plots, args.out)
    print(json.dumps({"complete": True, "cases": len(results), "out": str(args.out)}))


if __name__ == "__main__":
    main()
