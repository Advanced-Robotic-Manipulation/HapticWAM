#!/usr/bin/env python3
"""Audit four completed V9 trials; no inference, physics or raw-input writes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
V8_AUDIT_SHA = "2219a5258f5914ae391b6181195affb50c918187efcd38e1ce8fb40c842e1cd2"
SEEDS = {904301, 904302}
BLOCKS = [("control_dwell200", 0.2), ("treatment_dwell100", 0.1)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def scalar_stats(values):
    x = np.asarray(values, dtype=float)
    require(np.isfinite(x).all(), "Nonfinite latency")
    return {
        "n": len(x),
        "mean_s": float(np.mean(x)) if len(x) else None,
        "median_s": float(np.median(x)) if len(x) else None,
        "min_s": float(np.min(x)) if len(x) else None,
        "max_s": float(np.max(x)) if len(x) else None,
    }


def discover(root):
    launch = read(root / "launch_plan.json")
    complete = read(root / "complete.json")
    require(
        complete.get("status") == "four_valid_trials_completed"
        and complete.get("trials") == 4,
        "Not ready: four valid trials have not completed",
    )
    plans = launch["plans"]
    require([p["id"] for p in plans] == [b[0] for b in BLOCKS], "Wrong block order")
    cases = []
    for plan, (label, dwell) in zip(plans, BLOCKS, strict=True):
        output = Path(plan["output"])
        progress = read(output / "progress.json")
        summary = read(output / "analysis/summary.json")
        require(progress["status"] == "all_trials_completed", "Incomplete controller")
        require(
            len(progress["trials"]) == 2
            and all(x["status"] == "completed" for x in progress["trials"].values()),
            "Incomplete block denominator",
        )
        require(
            summary["status"] == "complete" and len(summary["trials"]) == 2,
            "Incomplete primary scoring",
        )
        require(not summary["missing_trial_keys"], "Missing primary score")
        design = read(output / "campaign_snapshot.json")
        digest = sha(output / "campaign_snapshot.json")
        require(
            digest
            == plan["campaign_sha256"]
            == progress["campaign_sha256"]
            == summary["campaign_sha256"],
            "Campaign hash differs across launch/progress/score",
        )
        require(set(design["sampling_seeds"]) == SEEDS, "Wrong seeds")
        require(design["planned_counts"]["total"] == 2, "Wrong planned count")
        require(
            design["adapter_profile"]["placement_release"]["opening_hold_s"] == dwell,
            "Wrong declared dwell",
        )
        entries = []
        for index in summary["trials"]:
            score_path = Path(index["score_path"])
            score = read(score_path)
            require(score["metrics"]["valid_for_scoring"], "Invalid trial")
            folder = Path(score["directory"])
            require(
                folder.parent.resolve() == (output / "rollouts").resolve(),
                "Wrong raw directory",
            )
            entries.append((score_path, score, folder))
        require(
            {s[1]["sampling_seed"] for s in entries} == SEEDS, "Incomplete paired seeds"
        )
        cases.append((plan, output, design, entries, label, dwell))
    return launch, complete, cases


def design_diff(designs):
    administrative = {
        "campaign_id",
        "frozen_at_utc",
        "freeze_authorization",
        "sampling_seeds",
        "planned_counts",
        "execution_order",
        "comparability_requirements",
        "analysis",
        "execution_phases",
        "development_split_note",
    }
    baseline = designs[0]
    candidate = copy.deepcopy(designs[1])
    candidate["adapter_profile"]["placement_release"]["opening_hold_s"] = 0.2
    require(set(candidate) == set(baseline), "Undeclared design fields")
    for key in baseline:
        if key not in administrative:
            require(
                candidate[key] == baseline[key],
                "Undeclared behavior difference: " + key,
            )
    require(baseline["horizon_s"] == 60, "Changed full horizon")
    require(baseline["post_stop_observation_s"] == 2, "Changed stop tail")
    require(
        baseline["adapter_profile"]["servo_constraint_hold_s"] == 2.5,
        "Changed arm hold budget",
    )
    return {
        "only_behavior_difference": "adapter_profile.placement_release.opening_hold_s",
        "control_s": 0.2,
        "treatment_s": 0.1,
        "administrative_fields_excluded_from_behavior_comparison": sorted(
            administrative
        ),
    }


def opening_spells(records):
    groups, group = [], []
    for r in records:
        if r["eligible"] and r["window_active"] and r["delivered_grip"] <= 0.45:
            group.append(r)
        elif group:
            groups.append(group)
            group = []
    if group:
        groups.append(group)
    return [
        {
            "first_t_s": g[0]["t_s"],
            "last_t_s": g[-1]["t_s"],
            "elapsed_s": g[-1]["t_s"] - g[0]["t_s"],
            "ticks": len(g),
            "replan_ids": sorted({r["replan_id"] for r in g}),
            "raw_grip_min_max": [
                min(r["raw_grip"] for r in g),
                max(r["raw_grip"] for r in g),
            ],
            "delivered_grip_min_max": [
                min(r["delivered_grip"] for r in g),
                max(r["delivered_grip"] for r in g),
            ],
            "command_grip_min_max": [
                min(r["command_grip"] for r in g),
                max(r["command_grip"] for r in g),
            ],
        }
        for g in groups
    ]


def release_audit(folder, design, source, common):
    execution = common.rows(folder / "execution_trace.jsonl")
    plans = read(folder / "planner_trace.json")
    deliveries = common.rows(folder / "delivered_plans.jsonl")
    info, run = read(folder / "policy_info.json"), read(folder / "run.json")
    config = info["placement_release"]
    require(
        config == design["adapter_profile"]["placement_release"],
        "Actual release config differs",
    )
    if "placement_release" in run:
        # run.json can include final controller state; compare configuration
        # only where that field is a configuration object.
        if "opening_hold_s" in run["placement_release"]:
            require(run["placement_release"] == config, "Run release config differs")
    release_path = source / "phantom/sim/release_controller.py"
    frozen = module(release_path, "v9_frozen_release_gate")
    gate = frozen.PlacementReleaseController(
        frozen.PlacementReleaseConfig.from_dict(config)
    )
    pd = {p["replan_id"]: p for p in plans}
    dd = {d["captured_snapshot_t"]: d for d in deliveries}
    rate = info["hardware_effective"]["control"]["action_rate_hz"]
    cap = info["max_play_steps"]
    records, eligibility_errors = [], []
    previous_latch = previous_grip = None
    predicted_commit = None
    gate_state_errors = []
    for r in execution:
        if r["stopped"]:
            continue
        p = pd.get(r.get("active_replan_id"))
        if p is None:
            continue
        capture = p["diagnostics"]["sim_observation_capture_t"]
        require(capture == p["t"], "Capture time identity changed")
        d = dd[capture]
        index = int(
            np.clip(
                r["diagnostics"]["play_time_s"] * rate,
                0,
                min(len(p["actions"]), cap) - 1e-6,
            )
        )
        veto = d["diagnostics"].get("terminal_veto", {})
        action = veto.get("action")
        original = action not in ("recovery_open", "recovery_tactile", "retry_cap")
        if action == "close_masked":
            original = index in veto.get("placement_release_passthrough_indices", [])
        diag = r["diagnostics"]
        release = diag["placement_release"]
        record = {
            "t_s": r["t"],
            "replan_id": p["replan_id"],
            "index": index,
            "raw_grip": p["actions"][index][6],
            "delivered_grip": d["actions"][index][6],
            "command_grip": r["gripper_command"],
            "observed_latch": diag.get("grip_latch"),
            "measured_grip": r["measured_gripper"][0],
            "measured_tcp": r["measured_tcp"],
            "eligible": bool(original and not diag["stale_plan_hold"]),
            "window_active": release["window_active"],
            "release": release,
        }
        records.append(record)
        if record["eligible"] and record["raw_grip"] != record["delivered_grip"]:
            eligibility_errors.append({"t_s": r["t"], "veto_action": action})
        if predicted_commit is None:
            gate.note_latch(previous_latch)
            # The gate holding/unarmed branches do not access pad loads. End
            # this independent replay at first commit; no invented tactile.
            gate.update(
                r["t"],
                tcp=r["measured_tcp"],
                policy_grip=record["delivered_grip"],
                measured_grip=record["measured_grip"],
                pad_loads={},
                eligible=record["eligible"],
                accepted_grip=previous_grip,
            )
            gate.note_latch(record["observed_latch"])
            if (
                gate.phase != release["phase"]
                or gate.opening_since != release["opening_since_s"]
            ):
                gate_state_errors.append(
                    {
                        "t_s": r["t"],
                        "replay_phase": gate.phase,
                        "actual_phase": release["phase"],
                        "replay_opening_since_s": gate.opening_since,
                        "actual_opening_since_s": release["opening_since_s"],
                    }
                )
            predicted_commit = gate.committed_at
        previous_latch = record["observed_latch"]
        previous_grip = record["command_grip"]
    commit = next(
        (r for r in records if r["release"]["committed_at_s"] is not None), None
    )
    actual_commit = None if commit is None else commit["release"]["committed_at_s"]
    require(
        not eligibility_errors, "Release eligibility claimed changed policy closure"
    )
    require(not gate_state_errors, "Precommit release gate state does not replay")
    require(predicted_commit == actual_commit, "Gate commit timestamp does not replay")
    latched = next((r for r in records if r["observed_latch"] is not None), None)
    opened = next(
        (
            r
            for r in records
            if latched
            and r["t_s"] > latched["t_s"]
            and r["command_grip"] <= config["open_command_max"]
        ),
        None,
    )
    measured_opened = next(
        (
            r
            for r in records
            if opened
            and r["t_s"] >= opened["t_s"]
            and r["measured_grip"] <= config["open_command_max"]
        ),
        None,
    )
    finished = next(
        (r for r in records if r["release"]["finished_at_s"] is not None), None
    )
    selected_open = [
        r
        for r in records
        if r["eligible"]
        and r["window_active"]
        and r["delivered_grip"] <= config["open_command_max"]
    ]
    changed = [r for r in selected_open if r["command_grip"] != r["delivered_grip"]]
    plan_release_tails = []
    for p in plans:
        rr = [r for r in records if r["replan_id"] == p["replan_id"]]
        if not rr or not any(r["window_active"] for r in rr):
            continue
        indices = [
            i for i, a in enumerate(p["actions"]) if a[6] <= config["open_command_max"]
        ]
        if indices:
            played = {r["index"] for r in rr}
            plan_release_tails.append(
                {
                    "replan_id": p["replan_id"],
                    "capture_t_s": p["t"],
                    "raw_opening_indices": indices,
                    "played_indices": sorted(played),
                    "unplayed_opening_indices": sorted(set(indices) - played),
                }
            )
    with np.load(folder / "sim_trace.npz") as tr:
        times = tr["t"]
        contact_snapshots = []
        events = {
            "policy_permission": actual_commit,
            "command_open": None if opened is None else opened["t_s"],
            "measured_open": None
            if measured_opened is None
            else measured_opened["t_s"],
            "controller_FINISH": None if finished is None else finished["t_s"],
            "last_trace": float(times[-1]),
        }
        for label, at in events.items():
            if at is None:
                continue
            i = int(np.argmin(abs(times - at)))
            contact_snapshots.append(
                {
                    "event": label,
                    "event_t_s": at,
                    "nearest_trace_t_s": float(times[i]),
                    "sample_offset_s": float(times[i] - at),
                    "object_xyz_m": tr["waffle_position"][i].tolist(),
                    "robot_contact_n": float(tr["packet_robot_normal_force"][i]),
                    "bin_contact_n": float(tr["packet_bin_normal_force"][i]),
                    "measured_gripper": tr["gripper"][i].tolist(),
                }
            )
        support = {
            "bin_contact_peak_n": float(tr["packet_bin_normal_force"].max()),
            "last_robot_contact_n": float(tr["packet_robot_normal_force"][-1]),
            "last_bin_contact_n": float(tr["packet_bin_normal_force"][-1]),
            "contact_snapshots": contact_snapshots,
        }
    return {
        "actual_config": config,
        "release_controller_sha256": sha(release_path),
        "independent_precommit_gate_replay": "exact phase/opening clock/commit match",
        "first_loaded_latch_t_s": None if latched is None else latched["t_s"],
        "first_commit_t_s": actual_commit,
        "first_command_open_after_latch_t_s": None if opened is None else opened["t_s"],
        "first_measured_open_after_command_t_s": None
        if measured_opened is None
        else measured_opened["t_s"],
        "controller_FINISH_t_s": None if finished is None else finished["t_s"],
        "first_commit_record": commit,
        "eligible_opening_spells": opening_spells(records),
        "eligible_opening_ticks": len(selected_open),
        "eligible_opening_ticks_command_differs": len(changed),
        "plans_with_opening_and_active_release_window": plan_release_tails,
        "last_release_diagnostics": execution[-1]["diagnostics"]["placement_release"],
        "support": support,
        "inference_latency": {
            key: scalar_stats(
                [p["diagnostics"][field] for p in plans if field in p["diagnostics"]]
            )
            for key, field in [
                ("native", "sim_native_inference_latency_s"),
                ("client_rpc", "sim_policy_replan_wall_time_s"),
                ("effective_delivery", "sim_effective_delivery_delay_s"),
            ]
        },
        "veto_actions": dict(
            Counter(
                d["diagnostics"].get("terminal_veto", {}).get("action")
                for d in deliveries
            )
        ),
        "scope": "Original actions are the selected K4 candidate before veto. Independent gate replay ends at first commit; physical outcomes use saved strict scores and support telemetry, never the gate FINISH flag.",
    }


def markdown(result):
    def fmt(value):
        return "—" if value is None else f"{value:.3f} s"

    out = [
        "# V9 release dwell audit",
        "",
        "Four trials on two reused development seeds. One measured arm start and one waffle pose; actual RPC delivery timing can differ across repeats.",
        "",
        "| Dwell / seed | Acquire / lift / carry / place / drop | Release permission | Command open | Stop | End |",
        "|---|---|---:|---:|---|---:|",
    ]
    for c in result["cases"]:
        o, r, t = c["physical"]["outcomes"], c["release"], c["termination"]
        stages = "/".join(
            str(int(o[k]))
            for k in ["acquired", "lifted", "carried", "full_task", "dropped"]
        )
        out.append(
            f"| {r['actual_config']['opening_hold_s']:.1f} s / {c['seed']} | {stages} | {fmt(r['first_commit_t_s'])} | {fmt(r['first_command_open_after_latch_t_s'])} | {', '.join(t['safety_events']) or t['stop_reason'] or 'none'} | {t['duration_s']:.3f} s |"
        )
    out += [
        "",
        f"Audit: **{result['status']}**. [Numerical audit](audit_v9.json) pins raw and frozen inputs and reports matched per-seed outcomes, actual latency, accepted command bounds, hold budgets, release proposals, latch transitions and physical robot/bin support.",
        "",
        "No historical pooling or reliable-winner inference. A release permission or controller FINISH alone is not placement. Physical task success and any subsequent safety stop are reported separately.",
    ]
    return "\n".join(out) + "\n"


def pre_intervention_comparison(control, treatment, common):
    """Locate differences before the release dwell can alter execution.

    This does not replay physics or infer the specific cause of runtime
    variation. Only diagnostics, not physical scoring thresholds, use the
    explicitly reported numerical tolerances below.
    """
    raw = [Path(c["raw_directory"]) for c in (control, treatment)]
    records = [common.rows(p / "execution_trace.jsonl") for p in raw]
    maps = [{round(r["t"], 9): r for r in rr} for rr in records]
    common_times = sorted(set(maps[0]) & set(maps[1]))
    commits = [c["release"]["first_commit_t_s"] for c in (control, treatment)]
    cutoff = min((t for t in commits if t is not None), default=None)
    starts = [
        g["first_t_s"]
        for c in (control, treatment)
        for g in c["release"]["eligible_opening_spells"]
    ]
    earliest_open = min(starts, default=None)
    before = [t for t in common_times if cutoff is None or t < cutoff - 1e-9]
    definitions = {
        "measured_q_rad": ("measured_q", 1e-6, False),
        "target_q_rad": ("target_q", 1e-6, False),
        "measured_tcp_position_m": ("measured_tcp", 0.001, True),
        "gripper_command": ("gripper_command", 0.001, False),
    }
    differences = {}
    for name, (field, tolerance, xyz_only) in definitions.items():
        first = None
        maximum = 0.0
        for t in before:
            a = np.asarray(maps[0][t][field])
            b = np.asarray(maps[1][t][field])
            if xyz_only:
                a, b = a[:3], b[:3]
            delta = float(np.max(abs(a - b)))
            maximum = max(maximum, delta)
            if first is None and delta > tolerance:
                first = {"t_s": t, "max_abs_difference": delta}
        differences[name] = {
            "tolerance": tolerance,
            "first_difference_before_either_commit": first,
            "maximum_before_either_commit": maximum,
        }
    deliveries = [common.rows(p / "delivered_plans.jsonl") for p in raw]
    initial_error = float(
        np.max(
            abs(
                np.asarray(control["initialization"]["settled_robot_q"])
                - treatment["initialization"]["settled_robot_q"]
            )
        )
    )
    return {
        "first_recorded_commit_either_run_s": cutoff,
        "first_eligible_opening_either_run_s": earliest_open,
        "earliest_possible_dwell100_commit_from_first_opening_s": None
        if earliest_open is None
        else earliest_open + 0.1,
        "matched_execution_ticks_before_either_commit": len(before),
        "initial_settled_joint_difference_rad": initial_error,
        "first_delivery_s_control_treatment": [
            d[0]["delivery_t"] if d else None for d in deliveries
        ],
        "differences": differences,
        "interpretation": "Differences before either recorded release commit cannot be attributed to the release dwell changing robot/gripper execution. They expose repeatability variation in real-time delivery, inference, observations or physics; this audit does not isolate which source caused it. The reported tolerances are diagnostic only and do not alter task scoring.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=BASE / "runs/teacher_release_dwell_v9/paired"
    )
    parser.add_argument(
        "--shared-audit",
        type=Path,
        default=BASE
        / "runs/teacher_carry_hotfix_v8/review_tools/audit_completed_v2.py",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--v8-frozen-inputs",
        type=Path,
        default=BASE
        / "runs/teacher_carry_hotfix_v8/paired/bounded_hold/frozen_inputs.json",
    )
    args = parser.parse_args()
    os.nice(19)
    require(not args.out.exists(), "Use a new output directory")
    require(
        sha(args.shared_audit) == V8_AUDIT_SHA,
        "Shared independent command-audit code changed",
    )
    common = module(args.shared_audit, "v9_shared_command_audit")
    _launch, _complete, groups = discover(args.root)
    difference = design_diff([g[2] for g in groups])
    v8_inputs = read(args.v8_frozen_inputs)
    output_cases, inventories = [], []
    for plan, output, design, entries, label, dwell in groups:
        frozen, inventory = common.verify_frozen(output, plan)
        require(
            set(frozen) == set(v8_inputs), "Frozen manifest field set differs from V8"
        )
        for key in frozen:
            if key != "campaign_sha256":
                require(
                    frozen[key] == v8_inputs[key],
                    "Source/input differs from V8: " + key,
                )
        source = Path(frozen["source_root"])
        require(
            source.resolve() not in args.out.resolve().parents,
            "Output inside frozen source",
        )
        require(
            (output / "rollouts").resolve() not in args.out.resolve().parents,
            "Output inside raw traces",
        )
        fk = common.load_fk(source)
        inventories.append(
            dict(inventory, group=label, campaign_sha256=plan["campaign_sha256"])
        )
        for score_path, score, folder in entries:
            policy = next(
                p for p in design["policies"] if p["id"] == score["policy_id"]
            )
            require(
                score["case"]["campaign_sha256"] == plan["campaign_sha256"],
                "Case campaign differs",
            )
            require(
                score["case"]["checkpoint_sha256"] == policy["checkpoint_sha256"],
                "Case checkpoint differs",
            )
            checkpoint = Path(frozen["live_repository"]) / policy["checkpoint"]
            require(
                sha(checkpoint) == policy["checkpoint_sha256"],
                "Checkpoint bytes changed",
            )
            c = common.analyze_case(label, folder, score, design, fk)
            c["initialization"] = read(folder / "initialization.json")
            c.update(
                score_path=str(score_path),
                score_sha256=sha(score_path),
                opening_hold_s=dwell,
                release=release_audit(folder, design, source, common),
            )
            output_cases.append(c)
    paired = []
    first_initial = output_cases[0]["initialization"]
    for c in output_cases[1:]:
        for key in (
            "configured_robot_q",
            "configured_packet_center_m",
            "policy_initial_state",
            "policy_initial_state_sha256",
        ):
            require(
                c["initialization"][key] == first_initial[key],
                "Unmatched initial configuration: " + key,
            )
    for seed in sorted(SEEDS):
        control = next(
            c for c in output_cases if c["seed"] == seed and c["opening_hold_s"] == 0.2
        )
        treatment = next(
            c for c in output_cases if c["seed"] == seed and c["opening_hold_s"] == 0.1
        )
        paired.append(
            {
                "seed": seed,
                "control_outcomes": control["physical"]["outcomes"],
                "treatment_outcomes": treatment["physical"]["outcomes"],
                "full_task_difference": int(
                    treatment["physical"]["outcomes"]["full_task"]
                )
                - int(control["physical"]["outcomes"]["full_task"]),
                "control_latency": control["release"]["inference_latency"],
                "treatment_latency": treatment["release"]["inference_latency"],
                "pre_intervention": pre_intervention_comparison(
                    control, treatment, common
                ),
            }
        )
    result = {
        "status": "audit_pass"
        if all(not c["audit_errors"] for c in output_cases)
        else "audit_errors",
        "scope": "Four V9 trials only; no rescoring, retries, physics, inference or hardware actions. AABB order required by unchanged frozen driver; reused development seeds and measured RPC timing preclude a reliable-winner inference.",
        "helper_sha256": sha(__file__),
        "shared_audit_sha256": sha(args.shared_audit),
        "launch_sha256": sha(args.root / "launch_plan.json"),
        "completion_sha256": sha(args.root / "complete.json"),
        "unchanged_v8_frozen_inputs_sha256": sha(args.v8_frozen_inputs),
        "design_comparison": difference,
        "source_checks": inventories,
        "thresholds": groups[0][2]["thresholds"],
        "cases": output_cases,
        "paired": paired,
    }
    args.out.mkdir(parents=True)
    (args.out / "audit_v9.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (args.out / "audit_v9.md").write_text(markdown(result))
    print(
        json.dumps(
            {"status": result["status"], "cases": len(output_cases), "paired": paired}
        )
    )
    if result["status"] != "audit_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
