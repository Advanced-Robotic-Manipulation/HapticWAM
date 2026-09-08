#!/usr/bin/env python3
"""Independent CPU audit of the six completed V8 paired development trials.

Read-only raw/frozen inputs; consumes authoritative scores without rescoring.
Does not wait or launch any process. All completion checks precede trace reads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
SEEDS = {904301, 904302}
CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def sha(path):
    path = Path(path)
    if path not in CACHE:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        CACHE[path] = h.hexdigest()
    return CACHE[path]


def demand(test, message):
    if not test:
        raise ValueError(message)


def flag(errors, test, message):
    if not test and message not in errors:
        errors.append(message)


def radius(q, dh):
    q = np.asarray(q)
    a2, a3, d4 = dh
    return np.sqrt(a2 * a2 + a3 * a3 + d4 * d4 + 2 * a2 * a3 * np.cos(q[..., 2]))


def discover(root):
    """All six must finish and have authoritative score files before analysis."""
    launch = read(root / "launch_plan.json")
    plans = launch["plans"]
    if len(plans) != 3:
        raise RuntimeError("Not ready: expected three manifest-defined arms")
    complete = read(root / "complete.json")
    if complete.get("trials") != 6:
        raise RuntimeError("Not ready: six-trial completion marker absent")
    groups = []
    for plan in plans:
        output = Path(plan["output"])
        progress = read(output / "progress.json")
        summary = read(output / "analysis/summary.json")
        entries = progress.get("trials", {})
        if (
            progress.get("status") != "all_trials_completed"
            or len(entries) != 2
            or any(v.get("status") != "completed" for v in entries.values())
            or len(summary.get("trials", [])) != 2
        ):
            raise RuntimeError(f"Not ready: {output} is not completely analyzed")
        scores = []
        for index in summary["trials"]:
            p = Path(index["score_path"])
            if not p.is_file():
                raise RuntimeError(f"Not ready: authoritative score {p}")
            score = read(p)
            scores.append((p, score))
        if {s["sampling_seed"] for _, s in scores} != SEEDS:
            raise RuntimeError("Unexpected or incomplete paired seed identities")
        groups.append((plan, output, progress, summary, scores))
    return launch, groups


def verify_frozen(group, plan):
    frozen_path = group / "frozen_inputs.json"
    frozen = read(frozen_path)
    demand(frozen == plan["inputs"], f"Launch/frozen input manifest differs: {group}")
    checked = {}
    for root_key, values_key in [
        ("source_root", "source_sha256"),
        ("external_controller_source_root", "external_controller_source_sha256"),
        ("live_repository", "live_core_sha256"),
        ("prepared_episode", "episode_sha256"),
    ]:
        for relative, digest in frozen.get(values_key, {}).items():
            p = Path(frozen[root_key]) / relative
            demand(sha(p) == digest, f"Frozen file changed: {p}")
            checked[str(p)] = digest
    for path_key, digest_key in [
        ("hardware_config", "hardware_sha256"),
        ("robot_usd", "robot_usd_sha256"),
    ]:
        if frozen.get(path_key):
            p = Path(frozen[path_key])
            demand(sha(p) == frozen[digest_key], f"Frozen input changed: {p}")
            checked[str(p)] = frozen[digest_key]
    for mapping in ["adapter_profile_inputs", "condition_initial_state_inputs"]:
        for entry in frozen.get(mapping, {}).values():
            if isinstance(entry, dict) and entry.get("path") and entry.get("sha256"):
                demand(
                    sha(entry["path"]) == entry["sha256"],
                    f"Profile input changed: {entry['path']}",
                )
                checked[entry["path"]] = entry["sha256"]
    return frozen, {"manifest_sha256": sha(frozen_path), "checked_files": checked}


def load_fk(source):
    path = source / "phantom/sim/kinematics.py"
    name = "v8_pinned_nominal_kinematics"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.forward_pose


def plan_after_hold(onset, plans, deliveries, execution):
    candidates = [
        p
        for p in plans
        if float(p.get("diagnostics", {}).get("sim_observation_capture_t", p["t"]))
        > onset + 1e-9
    ]
    if not candidates:
        return None
    p = min(candidates, key=lambda p: p["t"])
    captured = p.get("diagnostics", {}).get("sim_observation_capture_t", p["t"])
    delivered = next(
        (d for d in deliveries if abs(d["captured_snapshot_t"] - captured) < 1e-8), None
    )
    played = [
        r
        for r in execution
        if not r["stopped"] and r.get("active_replan_id") == p["replan_id"]
    ]
    moving = [
        r
        for r in played
        if r.get("ik_success") is True and r.get("ik_reason") != "constraint_hold"
    ]
    return {
        "replan_id": p["replan_id"],
        "capture_t_s": captured,
        "request_t_s": p.get("diagnostics", {}).get("sim_inference_request_t", p["t"]),
        "status": p["status"],
        "planned_first_action_t_s": p.get("first_action_t"),
        "delivery_t_s": None if delivered is None else delivered["delivery_t"],
        "activated": False if delivered is None else delivered["activated"],
        "first_played_execution_t_s": None if not played else played[0]["t"],
        "first_nonheld_accepted_execution_t_s": None if not moving else moving[0]["t"],
        "scope": "Observation timestamp is after hold onset, not proof of settled motion. Played acceptance is separate from nonzero physical progress.",
    }


def hold_audit(execution, plans, deliveries, budget_s, progress_rad):
    errors = []
    expected_start = None
    anchor = None
    timed_out = False
    starts = []
    held_rows = []
    states = []
    for i, r in enumerate(execution):
        meta = r.get("servo_reach_limiter") or {}
        state = meta.get("constraint_hold")
        if state is None:
            continue
        states.append((r, state))
        flag(errors, budget_s is not None, "undeclared_hold_telemetry")
        if budget_s is None:
            continue
        q = np.asarray(r["target_q"])
        held = bool(state["held"])
        if expected_start is None and held:
            expected_start = r["t"]
            anchor = q.copy()
            starts.append({"start_t_s": expected_start, "anchor_q": q.tolist()})
        if expected_start is not None:
            elapsed = r["t"] - expected_start
            if elapsed >= budget_s:
                timed_out = True
            # A timeout candidate was never applied to target_q. Its proposed
            # displacement cannot be independently recovered from that field.
            progress = float(max(abs(q - anchor)))
            if not timed_out and not held and progress >= progress_rad:
                expected_start = anchor = None
            flag(
                errors,
                bool(state["active"]) == (expected_start is not None),
                "hold_active_state_disagrees",
            )
            flag(
                errors,
                bool(state["timed_out"]) == timed_out,
                "hold_timeout_state_disagrees",
            )
            flag(
                errors,
                state["started_at_s"] == expected_start,
                "hold_deadline_restarted_without_progress",
            )
            wanted_elapsed = 0 if expected_start is None else r["t"] - expected_start
            flag(
                errors,
                abs(state["elapsed_s"] - wanted_elapsed) < 1e-8,
                "hold_elapsed_clock_disagrees",
            )
            if not timed_out:
                flag(
                    errors,
                    abs(state["progress_rad"] - progress) < 1e-8,
                    "hold_progress_disagrees",
                )
        if expected_start is None:
            flag(
                errors,
                not state["active"] and state["started_at_s"] is None,
                "unstarted_hold_reported_active",
            )
        if held and not state["timed_out"]:
            held_rows.append(r)
            flag(
                errors,
                r.get("ik_success") is True and r.get("ik_reason") == "constraint_hold",
                "streamed_hold_misclassified",
            )
            flag(
                errors, r.get("accepted_tcp") is not None, "held_pose_feedback_missing"
            )
            if i:
                flag(
                    errors,
                    np.array_equal(q, np.asarray(execution[i - 1]["target_q"])),
                    "held_joint_target_changed",
                )
                previous = execution[i - 1].get("servo_reach_limiter") or {}
                if previous.get("consecutive_rejects") is not None:
                    flag(
                        errors,
                        meta.get("consecutive_rejects")
                        == previous["consecutive_rejects"],
                        "held_stream_changed_fault_count",
                    )
        if state["timed_out"]:
            flag(
                errors,
                r.get("ik_success") is None
                and r.get("ik_reason") == "servo_constraint_hold_timeout",
                "timeout_classified_as_ik_failure",
            )
    for record in starts:
        record["first_post_onset_plan"] = plan_after_hold(
            record["start_t_s"], plans, deliveries, execution
        )
    first = next(
        (
            r
            for r in execution
            if r.get("ik_reason") in ("limiter_hold", "constraint_hold")
        ),
        None,
    )
    stops = [r for r in execution if r["stopped"]]
    stop = stops[0] if stops else None
    timeout = next((r for r, s in states if s["timed_out"]), None)
    if timeout is not None:
        flag(
            errors,
            stop is not None
            and stop.get("stop_reason") == "servo_constraint_hold_timeout",
            "timeout_missing_ordinary_stop",
        )
        if stop is not None:
            flag(
                errors,
                stop["gripper_command"] == timeout["gripper_command"],
                "timeout_changed_gripper_command",
            )
    return {
        "audit_errors": errors,
        "declared_timeout_s": budget_s,
        "progress_threshold_rad": progress_rad,
        "telemetry_rows": len(states),
        "streamed_hold_rows": len(held_rows),
        "max_elapsed_s": max((s["elapsed_s"] for _, s in states), default=0),
        "budget_sessions": starts,
        "first_any_hold_t_s": None if first is None else first["t"],
        "first_post_any_hold_plan": None
        if first is None
        else plan_after_hold(first["t"], plans, deliveries, execution),
        "first_hold_gripper_command": None
        if first is None
        else first["gripper_command"],
        "last_streamed_hold_gripper_command": None
        if not held_rows
        else held_rows[-1]["gripper_command"],
        "first_hold_latch": None
        if first is None
        else first["diagnostics"].get("grip_latch"),
        "last_hold_latch": None
        if not held_rows
        else held_rows[-1]["diagnostics"].get("grip_latch"),
        "timeout_t_s": None if timeout is None else timeout["t"],
        "stop_gripper_command": None if stop is None else stop["gripper_command"],
        "scope": "Hold budget reproduced independently from accepted joint targets; no same-timeout-tick candidate-motion assertion. Grip values are reported, not assumed constant when policy changes them.",
    }


def analyze_case(label, folder, score, design, fk):
    errors = []
    flag(errors, score["metrics"]["valid_for_scoring"], "authoritative_score_invalid")
    demand(
        sha(folder / "case.json") == score["case_sha256"], "Case metadata bytes changed"
    )
    for f, digest in score["input_sha256"].items():
        demand(sha(folder / f) == digest, f"Saved score input changed: {folder / f}")
    demand(
        score["metrics"]["thresholds"] == design["thresholds"],
        "Physical thresholds differ from frozen design",
    )
    demand(
        score["case"]["scoring_thresholds"] == design["thresholds"],
        "Case thresholds differ from design",
    )
    execution = rows(folder / "execution_trace.jsonl")
    plans = read(folder / "planner_trace.json")
    deliveries = rows(folder / "delivered_plans.jsonl")
    info, run = read(folder / "policy_info.json"), read(folder / "run.json")
    initial = read(folder / "initialization.json")
    safety = info["hardware_effective"]["safety"]
    dh = safety["ur_dh_a2_a3_d4_m"]
    profile = design.get("adapter_profile", {})
    enabled = profile.get("servo_reach_limiter", False)
    hold_s = profile.get("servo_constraint_hold_s")
    metadata = info.get("servo_reach_limiter") or {}
    flag(
        errors,
        info["hardware_effective"] == design["runtime_hardware"]["effective_model"],
        "effective_hardware_differs",
    )
    if hold_s is not None:
        flag(
            errors,
            metadata.get("constraint_hold_s")
            == hold_s
            == safety.get("servo_constraint_hold_s"),
            "hold_profile_mismatch",
        )
    flag(errors, bool(metadata) == enabled, "limiter_profile_mismatch")
    flag(
        errors,
        safety.get("servo_constraint_hold_s") == hold_s,
        "hardware_hold_profile_mismatch",
    )
    if enabled:
        run_metadata = {
            k: v
            for k, v in (run.get("servo_reach_limiter") or {}).items()
            if k != "final_consecutive_rejects"
        }
        flag(errors, run_metadata == metadata, "run_and_policy_limiter_metadata_differ")
        for key, expected in [
            ("elbow_min_rad", safety["elbow_min_rad"]),
            ("joint_speed_max_rad_s", safety["servo_joint_speed_max_rad_s"]),
            ("measured_wrist_extension_stop_m", safety["wrist_extension_stop_m"]),
        ]:
            flag(
                errors,
                metadata.get(key) == expected,
                "limiter_hardware_" + key + "_differs",
            )
    stop = next((r for r in execution if r["stopped"]), None)
    active = [r for r in execution if not r["stopped"]]
    t = np.asarray([r["t"] for r in execution])
    flag(
        errors,
        np.isfinite(t).all() and (np.diff(t) > 0).all(),
        "invalid_execution_clock",
    )
    q = np.asarray([r["measured_q"] for r in execution])
    qd = np.asarray([r["measured_qd"] for r in execution])
    cq = np.asarray([r["target_q"] for r in active])
    flag(
        errors,
        all(
            x.ndim == 2 and x.shape[1] == 6 and np.isfinite(x).all()
            for x in [q, qd, cq]
        ),
        "invalid_joint_state",
    )
    flag(
        errors,
        abs(q[0, 2]) <= np.pi and np.all(abs(cq[:, 2]) <= np.pi),
        "nonprincipal_elbow_branch",
    )
    for key in ["configured_robot_q", "settled_robot_q"]:
        flag(errors, abs(initial[key][2]) <= np.pi, "nonprincipal_initial_" + key)
    command_steps = []
    feedback_errors = []
    stopped_feedback = []
    hold_motion = []
    for i, r in enumerate(execution):
        if r.get("accepted_tcp") is not None and r["stopped"]:
            # The frozen runner's explicit safety_hold reports measured TCP and
            # commands measured q, rather than reporting nominal FK. Preserve
            # both distinctions; do not relax the active command equality gate.
            stopped_feedback.append(
                {
                    "t_s": r["t"],
                    "nominal_fk_component_error": float(
                        max(abs(fk(r["target_q"]) - r["accepted_tcp"]))
                    ),
                    "target_equals_measured_q": bool(
                        np.array_equal(r["target_q"], r["measured_q"])
                    ),
                    "feedback_equals_measured_tcp": bool(
                        np.array_equal(r["accepted_tcp"], r["measured_tcp"])
                    ),
                }
            )
            flag(
                errors,
                r.get("ik_reason") == "safety_hold"
                and stopped_feedback[-1]["target_equals_measured_q"]
                and stopped_feedback[-1]["feedback_equals_measured_tcp"],
                "stopped_feedback_is_not_measured_hold",
            )
        elif r.get("accepted_tcp") is not None:
            feedback_errors.append(
                float(max(abs(fk(r["target_q"]) - r["accepted_tcp"])))
            )
        if i and not r["stopped"] and not execution[i - 1]["stopped"]:
            d = np.asarray(r["target_q"]) - execution[i - 1]["target_q"]
            command_steps.append(float(max(abs(d)) / (r["t"] - execution[i - 1]["t"])))
            if r.get("ik_reason") == "constraint_hold":
                hold_motion.append(float(max(abs(d))))
    flag(
        errors,
        max(feedback_errors, default=0) < 1e-10,
        "accepted_pose_is_not_target_fk",
    )
    if enabled:
        flag(
            errors,
            np.min(abs(cq[:, 2])) >= safety["elbow_min_rad"] - 1e-10,
            "command_elbow_limit_violated",
        )
        flag(
            errors,
            max(command_steps, default=0)
            <= safety["servo_joint_speed_max_rad_s"] + 1e-8,
            "command_step_speed_violated",
        )
    flag(errors, max(hold_motion, default=0) == 0, "held_commands_move")
    hold = hold_audit(
        execution,
        plans,
        deliveries,
        hold_s,
        metadata.get("constraint_hold_progress_rad", 0.001),
    )
    errors.extend(hold["audit_errors"])
    with np.load(folder / "sim_trace.npz") as tr:
        for key in [
            "t",
            "physics_t",
            "packet_robot_normal_force",
            "packet_bin_normal_force",
        ]:
            flag(errors, key in tr, "missing_trace_" + key)
        st = tr["t"].copy()
        clock_error = float(max(abs(st - tr["physics_t"])))
        flag(
            errors,
            clock_error <= design["thresholds"]["clock_tolerance_s"],
            "physics_clock_disagrees",
        )
        support = {
            "packet_robot_force_peak_n": float(np.max(tr["packet_robot_normal_force"])),
            "packet_bin_force_peak_n": float(np.max(tr["packet_bin_normal_force"])),
            "last_robot_force_n": float(tr["packet_robot_normal_force"][-1]),
            "last_bin_force_n": float(tr["packet_bin_normal_force"][-1]),
            "placement_support": score["metrics"].get("placement_support"),
        }
    horizon = design["horizon_s"]
    dt = design["nominal_scene"]["physics"]["dt"]
    expected_end = (
        horizon
        if stop is None
        else min(horizon, stop["t"] + design["post_stop_observation_s"])
    )
    flag(
        errors,
        abs(run["duration_s"] - expected_end) <= 2 * dt + 1e-8,
        "unexpected_horizon_or_stop_tail",
    )
    flag(
        errors,
        abs(float(st[-1]) - run["duration_s"])
        <= design["thresholds"]["max_sample_gap_s"],
        "trace_ends_early",
    )
    finish = next(
        (r for r in execution if r["diagnostics"].get("completed_reason")), None
    )
    physical = score["metrics"]
    category = (
        ("horizon_FINISH" if finish else "horizon_no_FINISH")
        if stop is None
        else (
            "constraint_hold_timeout"
            if stop["stop_reason"] == "servo_constraint_hold_timeout"
            else "legacy_limiter_stall"
            if stop["stop_reason"] == "servo_limiter_stall"
            else "measured_safety_stop"
            if stop["diagnostics"].get("safety_events")
            else "other_controller_stop"
        )
    )
    return {
        "group": label,
        "seed": score["sampling_seed"],
        "case_id": folder.name,
        "audit_errors": sorted(set(errors)),
        "raw_directory": str(folder),
        "score_inputs_sha256": score["input_sha256"],
        "extra_inputs_sha256": {
            f: sha(folder / f) for f in ["delivered_plans.jsonl", "initialization.json"]
        },
        "physical": {
            k: physical[k]
            for k in [
                "valid_for_scoring",
                "invalid_reasons",
                "outcomes",
                "event_times_s",
                "object",
                "placement_support",
            ]
        },
        "termination": {
            "category": category,
            "stop_reason": None if stop is None else stop["stop_reason"],
            "first_stop_t_s": None if stop is None else stop["t"],
            "safety_events": []
            if stop is None
            else stop["diagnostics"].get("safety_events", []),
            "first_FINISH_t_s": None if finish is None else finish["t"],
            "duration_s": run["duration_s"],
            "expected_end_s": expected_end,
            "horizon_s": horizon,
            "trace_first_last_s": [float(st[0]), float(st[-1])],
            "max_physics_clock_error_s": clock_error,
        },
        "command": {
            "enabled": enabled,
            "hold_s": hold_s,
            "metadata": metadata,
            "min_abs_elbow_rad": float(np.min(abs(cq[:, 2]))),
            "max_wrist_radius_m": float(radius(cq, dh).max()),
            "max_step_speed_rad_s": max(command_steps, default=None),
            "max_accepted_tcp_fk_component_error": max(feedback_errors, default=None),
            "stopped_feedback": stopped_feedback,
            "max_held_joint_motion_rad": max(hold_motion, default=0),
            "ik_failure_counts": dict(
                Counter(
                    r.get("ik_reason") for r in active if r.get("ik_success") is False
                )
            ),
        },
        "measured": {
            "max_wrist_radius_m": float(radius(q, dh).max()),
            "wrist_stop_m": safety["wrist_extension_stop_m"],
            "max_abs_qd_rad_s": float(abs(qd).max()),
            "max_requested_tcp_position_error_m": float(
                max(
                    np.linalg.norm(
                        np.asarray(r["requested_tcp"][:3]) - r["measured_tcp"][:3]
                    )
                    for r in active
                )
            ),
            "scope": "Execution observations through first stop; radius is nominal geometry computed from measured q.",
        },
        "support": support,
        "holds": hold,
    }


def markdown(result):
    out = [
        "# V8 paired carry-hold audit",
        "",
        "Six contemporaneous development trials, two seeds per setting. Physical outcomes come from the unchanged authoritative strict scorer. No historical pooling or causal estimate from unmatched inference timing.",
        "",
        "| Setting / seed | Acquire / lift / carry / full / drop | Stop | End | Max measured radius | Holds / max elapsed |",
        "|---|---|---|---:|---:|---|",
    ]
    for c in result["cases"]:
        o = c["physical"]["outcomes"]
        h = c["holds"]
        t = c["termination"]
        stages = "/".join(
            str(int(o[k]))
            for k in ["acquired", "lifted", "carried", "full_task", "dropped"]
        )
        out.append(
            f"| {c['group']} / {c['seed']} | {stages} | {t['stop_reason'] or 'none'} | {t['duration_s']:.3f} s | {1000 * c['measured']['max_wrist_radius_m']:.2f} mm | {h['streamed_hold_rows']} / {h['max_elapsed_s']:.3f} s |"
        )
    out += [
        "",
        f"Audit status: **{result['status']}**. [Numerical audit](completed_audit.json) includes frozen hashes, fixed thresholds, physical support, command/FK checks, independent hold-budget replay and post-hold plan delivery/activation.",
        "",
        "FINISH is not physical success. A hold timeout is an executed controller outcome, not an infrastructure failure. Constraint and joint limits bound submitted commands; they do not certify measured stopping distance, contact-force realism or hardware safety. Fresh capture/delivery timestamps show whether a response could play; they do not establish that it recovers the task.",
    ]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=BASE / "runs/teacher_carry_hotfix_v8/paired"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.nice(19)
    demand(not args.out.exists(), "Choose a new output directory")
    _launch, groups = discover(args.root)
    results = []
    inventories = []
    designs = []
    for plan, group, progress, summary, scores in groups:
        design_path = group / "campaign_snapshot.json"
        design = read(design_path)
        demand(sha(design_path) == plan["campaign_sha256"], "Campaign snapshot changed")
        demand(
            progress["campaign_sha256"] == plan["campaign_sha256"],
            "Progress campaign differs",
        )
        demand(
            summary["campaign_sha256"] == plan["campaign_sha256"],
            "Analysis campaign differs",
        )
        frozen, inventory = verify_frozen(group, plan)
        source = Path(frozen["source_root"])
        demand(
            source.resolve() not in args.out.resolve().parents,
            "Output inside source forbidden",
        )
        demand(
            group.resolve() / "rollouts" not in args.out.resolve().parents,
            "Output inside raw rollouts forbidden",
        )
        fk = load_fk(source)
        inventory.update(group=plan["id"], campaign_sha256=plan["campaign_sha256"])
        inventories.append(inventory)
        designs.append(design)
        for score_path, score in scores:
            folder = Path(score["directory"])
            demand(
                folder.parent.resolve() == (group / "rollouts").resolve(),
                "Score points outside current group",
            )
            demand(
                score["case"]["campaign_sha256"] == plan["campaign_sha256"],
                "Wrong case campaign",
            )
            policy = next(
                p for p in design["policies"] if p["id"] == score["policy_id"]
            )
            demand(
                score["case"]["checkpoint_sha256"] == policy["checkpoint_sha256"],
                "Wrong scored checkpoint",
            )
            ckpt = Path(frozen["live_repository"]) / policy["checkpoint"]
            demand(sha(ckpt) == policy["checkpoint_sha256"], "Checkpoint bytes changed")
            result = analyze_case(plan["id"], folder, score, design, fk)
            result.update(score_path=str(score_path), score_sha256=sha(score_path))
            results.append(result)
    for other in designs[1:]:
        for k in [
            "nominal_scene",
            "conditions",
            "policies",
            "inference_settings",
            "thresholds",
            "sampling_seeds",
        ]:
            demand(designs[0][k] == other[k], f"Undeclared matched difference: {k}")
    result = {
        "status": "audit_pass"
        if all(not c["audit_errors"] for c in results)
        else "audit_errors",
        "scope": "Six contemporaneous V8 completed trials; consume saved scores, no rescoring, inference or simulation.",
        "audit_revision": "v2: active command FK equality remains 1e-10; stopped safety_hold separately requires exact measured q/TCP equality. Original v1 findings retained beside this report.",
        "launch_sha256": sha(args.root / "launch_plan.json"),
        "completion_sha256": sha(args.root / "complete.json"),
        "helper_sha256": sha(__file__),
        "thresholds": designs[0]["thresholds"],
        "source_checks": inventories,
        "cases": results,
    }
    args.out.mkdir(parents=True)
    (args.out / "completed_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (args.out / "completed_audit.md").write_text(markdown(result))
    print(
        json.dumps(
            {"status": result["status"], "cases": len(results), "output": str(args.out)}
        )
    )
    if result["status"] != "audit_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
