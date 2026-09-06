#!/usr/bin/env python3
"""Frozen teacher screen selection and paired confirmation; CPU-only raw audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.sim.analyze_policy_campaign import (
    load_design,
    load_trials,
    paired_cluster_interval,
    planned_keys,
    read_rows,
)

TIERS = ("clean_place", "strict_full_place", "lifted", "acquired", "safety_stop")
FORCE_EVENTS = {"wrench_limit", "tactile_fz", "tactile_depth"}
SAFETY_EVENTS = FORCE_EVENTS | {
    "protective_stop",
    "wrist_extension",
    "joint_speed",
    "hitbox_exit",
    "arm_stale",
    "camera_scene_stale",
    "top_workspace",
    "workspace_exit",
}
LIMITATIONS = [
    "Operational simulator selection gates, not hardware success probabilities.",
    "Discovery ranking is selection-biased; only reserved-start confirmation can declare a clear simulator winner.",
    "Bootstrap resamples measured starts with both matched seeds intact; six start clusters give limited precision.",
    "A zero or degenerate interval does not establish equivalence; no selection from representative videos.",
    "Contact geometry, wrist/tactile proxies, support telemetry and camera estimates remain conditional on the frozen simulator.",
    "NFE1/K4 versus NFE5/K1 changes two recipe factors; it is not an isolated NFE effect.",
]


def latency_summary(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
    }


def trial_evidence(record, run, trace, execution, planners, thresholds):
    """Combine independent object truth with controller diagnostics, never invert it."""
    metrics = record.get("metrics") or {}
    outcomes, obj = metrics.get("outcomes", {}), metrics.get("object", {})
    support, control = metrics.get("placement_support", {}), metrics.get("control", {})
    stopped = [r for r in execution if r.get("stopped") is True]
    first_stop = stopped[0] if stopped else None
    reason = run.get("policy_stop_reason") or control.get("stop_reason")
    if not reason and first_stop:
        reason = first_stop.get("stop_reason")
    actual_stop = bool(reason or stopped)
    terminal_events = sorted(
        set((first_stop or {}).get("diagnostics", {}).get("safety_events", []))
    )
    all_events = sorted(
        {
            e
            for r in execution
            for e in r.get("diagnostics", {}).get("safety_events", [])
        }
    )
    reasons = list(metrics.get("invalid_reasons", []))
    if (
        actual_stop
        and reason in ("safety_stop", "protective_stop")
        and not terminal_events
    ):
        reasons.append("terminal_safety_kind_missing")
    event_set = set(terminal_events) | ({reason} if reason else set())
    safety_stop = actual_stop and (
        reason in ("safety_stop", "protective_stop")
        or bool(event_set & SAFETY_EVENTS)
        or any(e.startswith("tactile_") and e.endswith("_stale") for e in event_set)
    )
    final = {}
    for key in ("packet_robot_normal_force", "packet_bin_normal_force"):
        a = np.asarray(trace.get(key, []), float)
        value = (
            float(a[-1]) if a.ndim == 1 and len(a) and np.isfinite(a).all() else None
        )
        final[key] = value
        if value is None or value < 0:
            reasons.append(f"missing_or_invalid_final_{key}")
    final_supported = (
        final["packet_robot_normal_force"] is not None
        and final["packet_bin_normal_force"] is not None
        and final["packet_robot_normal_force"]
        <= thresholds["support_robot_force_max_n"]
        and final["packet_bin_normal_force"] > thresholds["support_bin_force_min_n"]
    )
    completed = run.get("policy_completed_reason")
    completion_rows = [
        r for r in execution if r.get("diagnostics", {}).get("completed_reason")
    ]
    execution_completed = (
        completion_rows[-1]["diagnostics"]["completed_reason"]
        if completion_rows
        else None
    )
    if completed != execution_completed:
        reasons.append("run_execution_completion_reason_mismatch")
    strict = bool(
        outcomes.get("full_task")
        and support.get("verified_placement") is True
        and support.get("required") is True
    )
    clean = bool(
        strict
        and obj.get("final_inside_bin") is True
        and obj.get("final_contacts_unloaded") is True
        and final_supported
        and completed == "placement_release_finished"
        and not actual_stop
    )
    valid = (
        record.get("status") == "scored"
        and metrics.get("valid_for_scoring") is True
        and not reasons
    )
    diagnostics = [p.get("diagnostics", {}) for p in planners]
    stages = [
        ("strict_full_place", strict),
        ("released_in_bin", outcomes.get("released_in_bin")),
        ("carried", outcomes.get("carried")),
        ("lifted", outcomes.get("lifted")),
        ("acquired", outcomes.get("acquired")),
        ("reach_before_closure", outcomes.get("reach_before_closure")),
    ]
    return {
        "policy_id": record["policy_id"],
        "condition_id": record["condition_id"],
        "sampling_seed": record["sampling_seed"],
        "directory": record["directory"],
        "valid_for_selection": valid,
        "invalid_reasons": reasons,
        "clean_place": clean,
        "strict_full_place": strict,
        "furthest_physical_stage": next(
            (name for name, achieved in stages if achieved), "none"
        ),
        **{
            k: outcomes.get(k)
            for k in (
                "reach_before_closure",
                "acquired",
                "lifted",
                "carried",
                "released_in_bin",
                "dropped",
            )
        },
        "reach_error_before_first_closing_motion_m": obj.get(
            "reach_error_before_first_closing_motion_m"
        ),
        "event_times_s": metrics.get("event_times_s", {}),
        "final_inside_bin": obj.get("final_inside_bin"),
        "final_contacts_unloaded": obj.get("final_contacts_unloaded"),
        "final_bin_supported_robot_unloaded": bool(final_supported),
        "final_support_forces_n": final,
        "actual_stop": actual_stop,
        "stop_reason": reason,
        "stop_time_s": first_stop.get("t") if first_stop else None,
        "terminal_safety_events": terminal_events,
        "logged_safety_events_including_nonterminal": all_events,
        "safety_stop": bool(safety_stop),
        "wrench_limit_stop": bool(actual_stop and "wrench_limit" in event_set),
        "tactile_force_limit_stop": bool(
            actual_stop and event_set & {"tactile_fz", "tactile_depth"}
        ),
        "completed_reason": completed,
        "completion_hold_rows": sum(
            bool(r.get("diagnostics", {}).get("completion_hold")) for r in execution
        ),
        "completion_time_s": (
            completion_rows[0]["diagnostics"].get("completed_at_s")
            if completion_rows
            else None
        ),
        "ik_rejects": control.get("ik_rejects"),
        "hold_duration_s": control.get("hold_duration_s"),
        "stale_plan_hold_rows": control.get("stale_plan_hold_rows"),
        "native_inference_latency_s": latency_summary(
            [d.get("sim_native_inference_latency_s") for d in diagnostics]
        ),
        "delivery_latency_s": latency_summary(
            [d.get("sim_effective_inference_latency_s") for d in diagnostics]
        ),
        "inference_wall_time_s": latency_summary(
            [p.get("inference_wall_time_s") for p in planners]
        ),
        "activation_delay_s": control.get("activation_delay_s"),
        "initialization": run.get("policy_initialization", run.get("initialization")),
        "collision_coverage": metrics.get("collisions", {}).get("observability"),
        "pad_packet_force_peak_n": obj.get("pad_packet_normal_force_peak_n"),
    }


def evaluate_selection(design, trials, stage):
    """Pure selection over fully audited rows; missing/invalid pairs block scoring."""
    count = (4, 4) if stage == "screen" else (2, 6)
    if stage not in ("screen", "confirmation"):
        raise ValueError("Unknown stage")
    if (len(design["policies"]), len(design["conditions"])) != count or len(
        design["sampling_seeds"]
    ) != 2:
        raise ValueError(
            f"{stage} requires {count[0]} teachers x {count[1]} starts x 2 seeds"
        )
    if any(p.get("architecture") != "teacher" for p in design["policies"]):
        raise ValueError("Teacher-only design required")
    if design["thresholds"].get("require_support_verified_release") is not True:
        raise ValueError("Strict support-verified physical release must be frozen")
    expected = set(planned_keys(design))
    by_key = {}
    for row in trials:
        key = (row["policy_id"], row["condition_id"], int(row["sampling_seed"]))
        if key in by_key or key not in expected:
            raise ValueError(f"Duplicate or unplanned trial: {key}")
        by_key[key] = row
    missing = sorted(expected - set(by_key))
    invalid = [
        list(k) for k, r in by_key.items() if r["valid_for_selection"] is not True
    ]
    result = {
        "schema_version": 1,
        "stage": stage,
        "expected_trials": len(expected),
        "available_trials": len(trials),
        "missing_trial_keys": [list(k) for k in missing],
        "invalid_trial_keys": invalid,
        "selected_ids": [],
        "clear_simulator_winner": None,
        "ranking_tiers": list(TIERS) + ["frozen_candidate_order"],
        "limitations": LIMITATIONS,
    }
    if missing or invalid:
        result["status"] = "incomplete_or_invalid_matched_stage"
        return result
    totals = []
    for order, policy in enumerate(design["policies"]):
        rows = [r for r in trials if r["policy_id"] == policy["id"]]
        total = {
            "policy_id": policy["id"],
            "candidate_order": order,
            "trials": len(rows),
        }
        for field in TIERS + (
            "carried",
            "released_in_bin",
            "dropped",
            "actual_stop",
            "wrench_limit_stop",
            "tactile_force_limit_stop",
        ):
            total[field] = sum(bool(r[field]) for r in rows)
        total["ranking_key"] = [-total[k] for k in TIERS[:-1]] + [
            total["safety_stop"],
            order,
        ]
        totals.append(total)
    totals.sort(key=lambda r: r["ranking_key"])
    result.update(status="complete_valid_matched_stage", ranking=totals)
    if stage == "screen":
        result["selected_ids"] = [r["policy_id"] for r in totals[:2]]
        result["conclusion"] = (
            "screen_top_two_for_reserved_confirmation; no winner claim"
        )
        return result
    a, b = totals
    matrices = []
    for policy in (a["policy_id"], b["policy_id"]):
        matrices.append(
            [
                [
                    int(by_key[policy, c["id"], int(s)]["clean_place"])
                    for s in design["sampling_seeds"]
                ]
                for c in design["conditions"]
            ]
        )
    estimate = paired_cluster_interval(
        *matrices, confidence=0.95, replicates=10000, seed=20260906
    )
    gates = {
        "clean_place_at_least_8_of_12": a["clean_place"] >= 8,
        "paired_95pct_lower_bound_positive": estimate["interval"][0] > 0,
        "no_increase_in_drops": a["dropped"] <= b["dropped"],
        "no_increase_in_wrench_limit_stops": a["wrench_limit_stop"]
        <= b["wrench_limit_stop"],
        "no_increase_in_tactile_force_limit_stops": a["tactile_force_limit_stop"]
        <= b["tactile_force_limit_stop"],
    }
    result.update(
        comparison={
            "policy_a": a["policy_id"],
            "policy_b": b["policy_id"],
            "outcome": "clean_place",
            "estimate": estimate,
        },
        confirmation_gates=gates,
    )
    if all(gates.values()):
        result["clear_simulator_winner"] = a["policy_id"]
        result["conclusion"] = "clear_simulator_winner_under_frozen_operational_gates"
    else:
        result["conclusion"] = (
            "inconclusive; pickup_candidate_only"
            if any(r["lifted"] for r in totals)
            else "inconclusive; no_sustained_pickup_candidate"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "confirmation"), required=True)
    args = parser.parse_args()
    design, digest = load_design(args.campaign)
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_trials(args.runs, design, digest, args.out / "raw_analysis")
    rows = []
    for record in records.values():
        folder = Path(record["directory"])
        if record.get("metrics") is None:
            rows.append(
                {
                    **record,
                    "valid_for_selection": False,
                    "invalid_reasons": record.get(
                        "missing_inputs", ["missing metrics"]
                    ),
                }
            )
            continue
        with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
            row = trial_evidence(
                record,
                json.loads((folder / "run.json").read_text()),
                trace,
                read_rows(folder / "execution_trace.jsonl"),
                read_rows(folder / "planner_trace.json"),
                design["thresholds"],
            )
        rows.append(row)
    result = evaluate_selection(design, rows, args.stage)
    for row in rows:
        row["study_stage"] = args.stage
    result.update(
        campaign_sha256=digest,
        campaign=str(args.campaign),
        runs=str(args.runs),
        trials=rows,
    )
    (args.out / "selection.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    columns = sorted({k for r in rows for k in r if k not in ("case", "metrics")})
    with (args.out / "trials.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                    if k in columns
                }
            )
    lines = [
        f"# Teacher {args.stage} selection",
        "",
        f"Status: **{result['status']}**.",
        "",
        f"Valid matched coverage: {len(rows) - len(result['invalid_trial_keys'])}/{result['expected_trials']}; missing {len(result['missing_trial_keys'])}, invalid {len(result['invalid_trial_keys'])}.",
        "",
        "Clean placement requires strict physical task success, final bin containment, unloaded robot/pads, positive bin support, controller completion and no actual stop. Controller completion alone earns nothing.",
        "",
        "| Candidate | Clean | Strict place | Lift | Acquire | Safety stops | Drops |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("ranking", []):
        lines.append(
            "| "
            + " | ".join(
                str(row[k])
                for k in (
                    "policy_id",
                    "clean_place",
                    "strict_full_place",
                    "lifted",
                    "acquired",
                    "safety_stop",
                    "dropped",
                )
            )
            + " |"
        )
    lines += [
        "",
        f"Conclusion: {result.get('conclusion', 'Selection withheld pending complete valid matching')}.",
        "",
        f"Selected screen IDs: {result['selected_ids']}; clear simulator winner: {result['clear_simulator_winner']}.",
        "",
        "See [full selection and gates](selection.json) and [per-trial CSV](trials.csv). Native inference, effective delivery and activation latencies are distinct. Logged safety events include nonterminal clamps; terminal causes are separate.",
        "",
    ]
    lines += ["- " + text for text in LIMITATIONS]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "selected_ids", "clear_simulator_winner")
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "complete_valid_matched_stage" else 2


if __name__ == "__main__":
    raise SystemExit(main())
