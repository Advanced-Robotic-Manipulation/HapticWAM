#!/usr/bin/env python3
"""Audit physical-placement selection at the frozen v1 anchor; no GPU execution."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_analysis = importlib.import_module("tools.sim.analyze_policy_campaign")
load_design, load_trials = _analysis.load_design, _analysis.load_trials
planned_keys, read_rows = _analysis.planned_keys, _analysis.read_rows
trial_evidence = importlib.import_module(
    "tools.sim.select_teacher_candidate"
).trial_evidence
_design = importlib.import_module("tools.sim.teacher_anchor_design")
PROTOCOL, digest = _design.PROTOCOL, _design.digest
load_protocol, validate_design = _design.load_protocol, _design.validate_design

DESCENDING = (
    "strict_full_place",
    "final_supported_placement",
    "clean_place",
    "lifted",
    "acquired",
)
COUNTS = DESCENDING + (
    "carried",
    "released_in_bin",
    "dropped",
    "actual_stop",
    "safety_stop",
    "pre_placement_safety_stop",
    "post_placement_safety_stop",
    "wrench_limit_stop",
    "tactile_force_limit_stop",
    "pre_placement_force_limit_stop",
)
LIMITATIONS = [
    "Prospective confirmation applies to one selected fixed physical simulator anchor, not varied arm starts or hardware success probabilities.",
    "Screen ranking is selection-biased; only the twelve disjoint paired confirmation seeds enter winner tests.",
    "A degenerate zero interval does not demonstrate equivalence. Small binary samples also require the predeclared exact discordant-pair test.",
    "Physical completion persists if a controller stop occurs later. Clean FINISH is secondary and is absent from the unchanged legacy-v1 controller.",
    "Wrench/tactile force-limit stops include tactile depth. Proxy force calibration, camera, rigid packet mechanics, native timing and shared GPU variation remain limitations.",
    "NFE5/K4 versus NFE1/K4 changes NFE only; its extra native latency remains part of the recipe's measured performance.",
]


def anchor_trial_evidence(*args, **kwargs):
    row = trial_evidence(*args, **kwargs)
    full_t = row["event_times_s"].get("full_task")
    stop_t = row["stop_time_s"]
    if row["strict_full_place"] and (full_t is None or not np.isfinite(full_t)):
        row["invalid_reasons"].append("strict_placement_time_missing")
    if row["actual_stop"] and (stop_t is None or not np.isfinite(stop_t)):
        row["invalid_reasons"].append("actual_stop_time_missing")
    row["valid_for_selection"] &= not row["invalid_reasons"]
    row["final_supported_placement"] = bool(
        row["final_inside_bin"] is True
        and row["final_contacts_unloaded"] is True
        and row["final_bin_supported_robot_unloaded"] is True
    )
    before = full_t is None or (stop_t is not None and stop_t < full_t)
    row["pre_placement_safety_stop"] = bool(row["safety_stop"] and before)
    row["post_placement_safety_stop"] = bool(row["safety_stop"] and not before)
    row["pre_placement_force_limit_stop"] = bool(
        row["pre_placement_safety_stop"]
        and (row["wrench_limit_stop"] or row["tactile_force_limit_stop"])
    )
    return row


def paired_seed_statistics(a, b, rule):
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if (
        a.shape != (12,)
        or b.shape != a.shape
        or not np.isin(a, [0, 1]).all()
        or not np.isin(b, [0, 1]).all()
    ):
        raise ValueError("Exactly twelve matched binary sampling-seed pairs required")
    differences = a - b
    rng = np.random.default_rng(rule["bootstrap_seed"])
    indexes = rng.integers(0, len(a), size=(rule["bootstrap_replicates"], len(a)))
    samples = differences[indexes].mean(axis=1)
    alpha = (1 - rule["confidence_level"]) / 2
    bounds = np.quantile(samples, [alpha, 1 - alpha])
    n10, n01 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n = n10 + n01
    exact_p = (
        min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n10, n01) + 1)) / 2**n)
        if n
        else 1.0
    )
    return {
        "difference_a_minus_b": float(differences.mean()),
        "interval": bounds.tolist(),
        "confidence_level": rule["confidence_level"],
        "bootstrap_replicates": rule["bootstrap_replicates"],
        "bootstrap_seed": rule["bootstrap_seed"],
        "paired_seed_count": len(a),
        "bootstrap_unit": "Matched sampling-seed pair at ONE fixed physical state; no arm-start resampling",
        "discordant_a_only": n10,
        "discordant_b_only": n01,
        "exact_two_sided_mcnemar_p": exact_p,
        "degenerate_interval": bool(bounds[0] == bounds[1]),
        "interpretation": "Descriptive conditional simulator uncertainty. A degenerate interval does not establish equivalence.",
    }


def evaluate_selection(protocol, design, trials, stage):
    validate_design(protocol, design, stage)
    expected, by_key = set(planned_keys(design)), {}
    for row in trials:
        key = (row["policy_id"], row["condition_id"], int(row["sampling_seed"]))
        if key in by_key or key not in expected:
            raise ValueError(f"Duplicate or unplanned trial: {key}")
        by_key[key] = row
    missing = sorted(expected - set(by_key))
    invalid = [
        list(k) for k, r in by_key.items() if r.get("valid_for_selection") is not True
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
        "ranking_tiers": list(DESCENDING)
        + ["pre_placement_safety_stop ascending", "frozen_candidate_order"],
        "primary_outcome": "Strict support-verified physical full_task; later stops are separate",
        "limitations": LIMITATIONS,
    }
    if missing or invalid:
        result["status"] = "incomplete_or_invalid_matched_stage"
        return result
    frozen_order = {p["id"]: i for i, p in enumerate(protocol["candidates"])}
    totals = []
    for policy in design["policies"]:
        rows = [r for r in trials if r["policy_id"] == policy["id"]]
        if any(any(type(r.get(k)) is not bool for k in COUNTS) for r in rows):
            raise ValueError(
                "Complete selection rows require explicit boolean outcome/stop fields"
            )
        total = {
            "policy_id": policy["id"],
            "trials": len(rows),
            "candidate_order": frozen_order[policy["id"]],
        }
        total.update({field: sum(r[field] for r in rows) for field in COUNTS})
        total["ranking_key"] = [-total[k] for k in DESCENDING] + [
            total["pre_placement_safety_stop"],
            total["candidate_order"],
        ]
        totals.append(total)
    totals.sort(key=lambda row: row["ranking_key"])
    result.update(status="complete_valid_matched_stage", ranking=totals)
    if stage == "screen":
        result.update(
            selected_ids=[r["policy_id"] for r in totals[:2]],
            conclusion="screen_top_two_for_reserved_confirmation; no winner claim",
        )
        return result
    leader, comparator = totals
    condition = design["conditions"][0]["id"]
    vectors = [
        [
            int(by_key[p["policy_id"], condition, int(seed)]["strict_full_place"])
            for seed in design["sampling_seeds"]
        ]
        for p in (leader, comparator)
    ]
    rule = protocol["confirmation_rule"]
    estimate = paired_seed_statistics(*vectors, rule)
    gates = {
        "physical_place_at_least_8_of_12": leader["strict_full_place"]
        >= rule["minimum_physical_full_tasks"],
        "paired_95pct_lower_bound_positive": estimate["interval"][0] > 0,
        "exact_two_sided_p_le_0_05": estimate["exact_two_sided_mcnemar_p"]
        <= rule["small_sample_guard"]["alpha"],
        "no_increase_in_drops": leader["dropped"] <= comparator["dropped"],
        "no_increase_in_pre_placement_force_limit_stops": leader[
            "pre_placement_force_limit_stop"
        ]
        <= comparator["pre_placement_force_limit_stop"],
    }
    winner = leader["policy_id"] if all(gates.values()) else None
    result.update(
        paired_comparison={
            "leader": leader["policy_id"],
            "comparator": comparator["policy_id"],
            "sampling_seeds": design["sampling_seeds"],
            "estimate": estimate,
        },
        winner_gates=gates,
        clear_simulator_winner=winner,
        conclusion=(
            "clear_physical_placement_winner_at_fixed_simulator_anchor"
            if winner
            else "conditional_ranked_leader_without_validated_winner"
            if leader["strict_full_place"]
            else "pickup_candidate_only_no_placement_winner"
            if leader["lifted"]
            else "inconclusive_no_placement_winner"
        ),
    )
    return result


def collect_trials(runs, design, design_sha, out):
    records = load_trials(runs, design, design_sha, out)
    rows = []
    for record in records.values():
        folder = Path(record["directory"])
        if record.get("metrics") is None:
            rows.append(
                {
                    "policy_id": record["policy_id"],
                    "condition_id": record["condition_id"],
                    "sampling_seed": record["sampling_seed"],
                    "directory": str(folder),
                    "valid_for_selection": False,
                    "invalid_reasons": record.get(
                        "missing_inputs", ["missing metrics"]
                    ),
                }
            )
            continue
        with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
            row = anchor_trial_evidence(
                record,
                json.loads((folder / "run.json").read_text()),
                trace,
                read_rows(folder / "execution_trace.jsonl"),
                read_rows(folder / "planner_trace.json"),
                design["thresholds"],
            )
        status_path = folder / "run_status.json"
        if (
            not status_path.exists()
            or json.loads(status_path.read_text()).get("status") != "completed"
        ):
            row["valid_for_selection"] = False
            row["invalid_reasons"].append("controller_trial_not_completed")
        row["score_path"] = record["score_path"]
        row["score_sha256"] = digest(record["score_path"])
        row["input_sha256"] = record.get("input_sha256", {})
        row["case_sha256"] = record["case_sha256"]
        rows.append(row)
    return rows


def write_report(out, result):
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    rows = result["trials"]
    columns = sorted({k for r in rows for k in r})
    with (out / "trials.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            }
            for row in rows
        )
    lines = [
        f"# Fixed-anchor teacher {result['stage']}",
        "",
        f"Status: **{result['status']}**.",
        "",
        f"Expected {result['expected_trials']}; available {result['available_trials']}; missing {len(result['missing_trial_keys'])}; invalid {len(result['invalid_trial_keys'])}.",
        "",
        "Physical placement is primary. A later stop does not erase completed placement. Final support and clean FINISH are separate diagnostics.",
        "",
        "| Teacher/recipe | Physical place | Final support | Clean | Lift | Acquire | Pre-place safety | Later safety | Drops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result.get("ranking", []):
        lines.append(
            "| "
            + " | ".join(
                str(r[k])
                for k in (
                    "policy_id",
                    "strict_full_place",
                    "final_supported_placement",
                    "clean_place",
                    "lifted",
                    "acquired",
                    "pre_placement_safety_stop",
                    "post_placement_safety_stop",
                    "dropped",
                )
            )
            + " |"
        )
    lines += [
        "",
        f"Conclusion: {result.get('conclusion', 'Selection withheld')}. Clear simulator winner: {result['clear_simulator_winner']}.",
        "",
        "[All rows and gates](selection.json) · [Per-trial CSV](trials.csv). Native inference, effective delivery and activation latency remain separate. Tactile force-limit stops include depth; logged nonterminal clamps do not become terminal causes.",
        "",
    ]
    lines += ["- " + text for text in LIMITATIONS]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "confirmation"), required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    design, sha = load_design(args.campaign)
    validate_design(protocol, design, args.stage)
    rows = collect_trials(args.runs, design, sha, args.out / "raw_analysis")
    result = evaluate_selection(protocol, design, rows, args.stage)
    result.update(
        campaign_sha256=sha,
        protocol_sha256=digest(args.protocol),
        campaign=str(args.campaign),
        runs=str(args.runs),
        trials=rows,
    )
    write_report(args.out, result)
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
