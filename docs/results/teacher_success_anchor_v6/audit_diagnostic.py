#!/usr/bin/env python3
"""CPU audit of completed v6 limiter diagnostics and their two old baselines.

No waiting, simulation, inference or source mutation. Outputs go outside raw
case folders. A readiness failure exits before any per-trial analysis is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
PINS = {
    "tools/sim/analyze_policy_campaign.py": "d47b4fa18fb093cbfd2aa751be908e53209a861aba9a1a49d4d71974aeaf4abc",
    "phantom/sim/policy_metrics.py": "b8460fc530acd6f11ea166f3fbff0ddc2ee74f4c73d956752abab0435975969f",
}
SEEDS = (904301, 904302)
ELBOW_MIN, SPEED_MAX, MEASURED_STOP = 0.40, 1.0, 0.468


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def radius(q):
    """Nominal UR3 wrist-center distance calculated from raw elbow joint."""
    q = np.asarray(q, dtype=float)
    return np.sqrt(
        0.24365**2 + 0.21325**2 + 0.11235**2 + 2 * 0.24365 * 0.21325 * np.cos(q[..., 2])
    )


def command_summary(execution, initial, enabled):
    errors = []
    for name in ("configured_robot_q", "settled_robot_q"):
        q = np.asarray(initial[name], float)
        if q.shape != (6,) or not np.isfinite(q).all():
            errors.append("invalid_initial_" + name)
        elif not -np.pi <= q[2] <= np.pi:
            errors.append("startup_elbow_outside_principal_branch:" + name)
    times = np.asarray([r["t"] for r in execution])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        errors.append("invalid_execution_clock")
    first_stop = next((r for r in execution if r["stopped"]), None)
    live = [r for r in execution if first_stop is None or r["t"] <= first_stop["t"]]
    commands = [r for r in live if not r["stopped"]]
    measured_q = np.asarray([r["measured_q"] for r in live], float)
    measured_qd = np.asarray([r["measured_qd"] for r in live], float)
    target_q = np.asarray([r["target_q"] for r in commands], float)
    if any(
        a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all()
        for a in (measured_q, measured_qd, target_q)
    ):
        raise ValueError("Missing or nonfinite six-joint command/measurement arrays")
    if not -np.pi <= measured_q[0, 2] <= np.pi:
        errors.append(
            "startup_elbow_outside_principal_branch:first_execution_measurement"
        )
    if np.any(abs(target_q[:, 2]) > np.pi):
        errors.append("command_elbow_left_principal_branch")
    limit = float(radius([0, 0, ELBOW_MIN]))
    step_speeds, branch_steps, newly_outside, escapes = [], [], [], []
    for previous, current in pairwise(execution):
        # Switching a safety hold to measured q is not a streamed servo motion.
        if previous["stopped"] or current["stopped"]:
            continue
        before, after = (
            np.asarray(previous["target_q"]),
            np.asarray(current["target_q"]),
        )
        elapsed = current["t"] - previous["t"]
        step_speeds.append(float(np.max(abs(after - before)) / elapsed))
        branch_steps.append(float(np.max(abs(after - before))))
        if float(radius(after)) > limit + 1e-9:
            if float(radius(after)) < float(radius(before)) - 1e-12:
                escapes.append(current["t"])
            else:
                newly_outside.append(current["t"])
    if enabled and step_speeds and max(step_speeds) > SPEED_MAX + 1e-8:
        errors.append("enabled_command_joint_speed_exceeded")
    if enabled and newly_outside:
        errors.append("enabled_command_outside_margin_without_inward_escape")
    metadata = [r.get("servo_reach_limiter") for r in commands]
    if enabled and any(not isinstance(item, dict) for item in metadata):
        errors.append("missing_limiter_per_command_metadata")
    if not enabled and any(item is not None for item in metadata):
        errors.append("undeclared_baseline_limiter_metadata")
    records = [item for item in metadata if isinstance(item, dict)]
    stop_events = (
        [] if first_stop is None else first_stop["diagnostics"].get("safety_events", [])
    )
    stop_reason = None if first_stop is None else first_stop.get("stop_reason")
    first_finish = next(
        (r for r in live if r["diagnostics"].get("completed_reason")), None
    )
    if stop_reason == "servo_limiter_stall":
        category = "executed_limiter_stall"
    elif first_stop is not None:
        category = (
            "measured_safety_stop" if stop_events else "other_executed_controller_stop"
        )
    elif first_finish is not None:
        category = "execution_FINISH_observed"
    else:
        category = "horizon_without_execution_FINISH"
    r_measured = radius(measured_q)
    qd_index = np.unravel_index(np.argmax(abs(measured_qd)), measured_qd.shape)
    tracking = np.linalg.norm(
        np.asarray([r["requested_tcp"][:3] for r in commands])
        - np.asarray([r["measured_tcp"][:3] for r in commands]),
        axis=1,
    )
    return {
        "audit_errors": errors,
        "commanded": {
            "rows_before_first_stop": len(commands),
            "elbow_abs_min_rad": float(abs(target_q[:, 2]).min()),
            "wrist_radius_max_m_from_target_q": float(radius(target_q).max()),
            "minimum_elbow_margin_rad": float(abs(target_q[:, 2]).min() - ELBOW_MIN),
            "minimum_wrist_radius_margin_m": float(limit - radius(target_q).max()),
            "maximum_consecutive_step_speed_rad_s": max(step_speeds, default=None),
            "maximum_consecutive_raw_joint_step_rad": max(branch_steps, default=None),
            "outside_command_envelope_rows": int(
                np.count_nonzero(radius(target_q) > limit + 1e-9)
            ),
            "out_of_envelope_inward_escape_rows": len(escapes),
            "outside_without_inward_escape_rows": len(newly_outside),
            "first_outside_without_escape_s": next(iter(newly_outside), None),
            "speed_definition": "Max absolute delta of consecutive submitted target_q divided by execution elapsed seconds; first command and safety-hold transition excluded, held unchanged targets contribute zero.",
        },
        "measured_before_first_stop_inclusive": {
            "wrist_radius_max_m_computed_from_measured_q": float(r_measured.max()),
            "wrist_radius_max_t_s": live[int(np.argmax(r_measured))]["t"],
            "wrist_stop_margin_at_peak_m": float(MEASURED_STOP - r_measured.max()),
            "maximum_abs_joint_velocity_rad_s": float(abs(measured_qd).max()),
            "maximum_joint_velocity_t_s": live[qd_index[0]]["t"],
            "maximum_joint_velocity_joint_index": int(qd_index[1]),
            "maximum_requested_vs_measured_tcp_position_error_m": float(tracking.max()),
            "interpretation": "Joint q and qd are measured PhysX articulation state. Wrist radius is computed nominal geometry, not a separate measured wrist-center sensor.",
        },
        "limiter": {
            "declared_enabled": enabled,
            "mode_counts": dict(Counter(item.get("mode") for item in records)),
            "violation_counts": dict(
                Counter(item["violation"] for item in records if item.get("violation"))
            ),
            "rejected_rows": sum(r.get("ik_success") is False for r in commands),
            "rejected_reason_counts": dict(
                Counter(
                    r.get("ik_reason") for r in commands if r.get("ik_success") is False
                )
            ),
            "maximum_consecutive_rejects_reported": max(
                (item.get("consecutive_rejects", 0) for item in records), default=0
            ),
            "maximum_ik_calls_reported": max(
                (item.get("ik_calls", 0) for item in records), default=0
            ),
            "first_intervention_s": next(
                (
                    r["t"]
                    for r in commands
                    if (r.get("servo_reach_limiter") or {}).get("mode")
                    in ("step", "slide", "hold")
                ),
                None,
            ),
        },
        "termination": {
            "category": category,
            "first_stop_reason": stop_reason,
            "first_stop_t_s": None if first_stop is None else first_stop["t"],
            "first_stop_safety_events": stop_events,
            "first_stop_measured_tcp": None
            if first_stop is None
            else first_stop["measured_tcp"],
            "first_stop_requested_tcp": None
            if first_stop is None
            else first_stop["requested_tcp"],
            "first_stop_gripper_command": None
            if first_stop is None
            else first_stop["gripper_command"],
            "first_execution_FINISH_t_s": None
            if first_finish is None
            else first_finish["t"],
            "FINISH_is_not_physical_task_success": True,
        },
    }


def ensure_completed(root):
    progress = read(root / "progress.json")
    for seed in SEEDS:
        name = f"teacher__fixed_anchor__seed{seed}"
        item = progress.get("trials", {}).get(name, {})
        if item.get("status") != "completed" or item.get("exit_code") != 0:
            raise RuntimeError(
                f"Not ready: {root}/{name} is not a completed successful process. No watcher started."
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=BASE / "runs/teacher_success_anchor_v6/diagnostic"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASE / "runs/teacher_success_anchor_v3/minimal_profile_diagnostic",
    )
    parser.add_argument(
        "--driver", type=Path, default=BASE / "source_teacher_anchor_driver_v6"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.nice(19)
    output = args.out.resolve()
    if output == args.driver.resolve() or args.driver.resolve() in output.parents:
        raise ValueError("Audit output must not be inside the frozen source")
    for group in (args.root, args.baseline):
        ensure_completed(group)
        if (
            output in (group.resolve(), group.resolve() / "rollouts")
            or group.resolve() / "rollouts" in output.parents
        ):
            raise ValueError("Audit output must be separate from raw trials")
    for name, expected in PINS.items():
        if sha(args.driver / name) != expected:
            raise ValueError("Pinned audit/scorer source changed: " + name)
    if args.out.exists():
        raise FileExistsError(
            "Choose a new audit output; existing results are preserved"
        )
    sys.path.insert(0, str(args.driver))
    from tools.sim.analyze_policy_campaign import load_design, load_trials

    result = {
        "scope": "Two completed development-seed v6 diagnostics and prior minimal baselines; no ranking, retuning or controlled causal estimate.",
        "analyzer_sha256": PINS,
        "cases": [],
        "caveats": [
            "Same sampling seeds do not imply matched rendered inputs or native inference-delivery timing.",
            "Command bounds and measured motion are reported separately; limiter does not guarantee measured stopping distance.",
            "Physical outcomes come only from frozen strict free-body/support metrics, never completion flags.",
            "This helper does not identify colliding actors or certify physical sensor calibration.",
        ],
    }
    args.out.mkdir(parents=True)
    for label, group in (
        ("minimal_baseline", args.baseline),
        ("limiter_v6", args.root),
    ):
        design, design_sha = load_design(group / "campaign_snapshot.json")
        enabled = design.get("adapter_profile", {}).get("servo_reach_limiter", False)
        if enabled != (label == "limiter_v6"):
            raise ValueError("Group limiter declaration disagrees with comparison role")
        for seed in SEEDS:
            name = f"teacher__fixed_anchor__seed{seed}"
            folder = group / "rollouts" / name
            scores = load_trials(folder, design, design_sha, args.out / label)
            if len(scores) != 1:
                raise ValueError("Expected exactly one independently scored trial")
            record = next(iter(scores.values()))
            initial, run, info = (
                read(folder / filename)
                for filename in ("initialization.json", "run.json", "policy_info.json")
            )
            execution = rows(folder / "execution_trace.jsonl")
            summary = command_summary(execution, initial, enabled)
            with np.load(folder / "sim_trace.npz") as trace:
                scene = {
                    "first_t_s": float(trace["t"][0]),
                    "last_t_s": float(trace["t"][-1]),
                    "maximum_measured_abs_qd_rad_s_including_tail": float(
                        abs(trace["qd"]).max()
                    ),
                    "maximum_computed_wrist_radius_m_including_tail": float(
                        radius(trace["q"]).max()
                    ),
                    "maximum_clock_disagreement_s": float(
                        abs(trace["t"] - trace["physics_t"]).max()
                    ),
                }
            original_score_path = group / "analysis/trials" / (name + ".json")
            original_score_agrees = None
            if original_score_path.exists():
                old_metrics = read(original_score_path)["metrics"]
                original_score_agrees = all(
                    old_metrics[key] == record["metrics"][key]
                    for key in ("outcomes", "event_times_s")
                )
                if not original_score_agrees:
                    summary["audit_errors"].append("original_physical_score_disagrees")
            result["cases"].append(
                {
                    "group": label,
                    "case_id": name,
                    "sampling_seed": seed,
                    "raw_directory": str(folder),
                    "campaign_sha256": design_sha,
                    "score_input_sha256": record["input_sha256"],
                    "initialization_sha256": sha(folder / "initialization.json"),
                    "independent_score_path": record["score_path"],
                    "original_outcomes_and_events_agree": original_score_agrees,
                    "configured_and_settled_initialization": initial,
                    "limiter_metadata": info.get("servo_reach_limiter"),
                    "scene_clock_and_tail": scene,
                    "reported_policy_completed_reason": run.get(
                        "policy_completed_reason"
                    ),
                    "physical_metrics": record["metrics"],
                    **summary,
                }
            )
    errors = [error for item in result["cases"] for error in item["audit_errors"]]
    result["status"] = (
        "audit_pass"
        if not errors
        and all(
            item["physical_metrics"]["valid_for_scoring"] for item in result["cases"]
        )
        else "audit_errors"
    )
    (args.out / "diagnostic_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# V6 limiter diagnostic comparison",
        "",
        "Descriptive development comparison; native timing differs and physical success is scored independently of FINISH.",
        "",
        "| Group / seed | Acquired / lift / carry / full | Termination | Limiter modes | Max command speed | Max measured radius |",
        "|---|---|---|---|---:|---:|",
    ]
    for item in result["cases"]:
        outcomes = item["physical_metrics"]["outcomes"]
        stages = "/".join(
            str(int(outcomes[key]))
            for key in ("acquired", "lifted", "carried", "full_task")
        )
        lines.append(
            f"| {item['group']} / {item['sampling_seed']} | {stages} | {item['termination']['category']} | {item['limiter']['mode_counts']} | {item['commanded']['maximum_consecutive_step_speed_rad_s']:.6f} rad/s | {item['measured_before_first_stop_inclusive']['wrist_radius_max_m_computed_from_measured_q']:.6f} m |"
        )
    lines += [
        "",
        f"Audit status: **{result['status']}**. Details, raw hashes, strict metrics, startup branch checks and distinct commanded/measured fields are in [diagnostic_audit.json](diagnostic_audit.json).",
        "",
        *result["caveats"],
    ]
    (args.out / "diagnostic_audit.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(result["cases"]),
                "out": str(args.out),
            }
        )
    )
    if result["status"] != "audit_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
