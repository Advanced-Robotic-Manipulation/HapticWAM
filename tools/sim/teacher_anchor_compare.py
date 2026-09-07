#!/usr/bin/env python3
"""Parameterized fixed-anchor teacher comparison; CPU-only, frozen protocol inputs."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_analysis = importlib.import_module("tools.sim.analyze_policy_campaign")
load_design, load_trials = _analysis.load_design, _analysis.load_trials
planned_keys, read_rows = _analysis.planned_keys, _analysis.read_rows
_anchor = importlib.import_module("tools.sim.select_teacher_anchor")
anchor_trial_evidence, paired_seed_statistics = (
    _anchor.anchor_trial_evidence,
    _anchor.paired_seed_statistics,
)
DESCENDING, COUNTS = _anchor.DESCENDING, _anchor.COUNTS
_design = importlib.import_module("tools.sim.teacher_anchor_design")
digest, candidate_policy, phases = (
    _design.digest,
    _design.candidate_policy,
    _design.phases,
)
LIMITATIONS = [
    "Development-informed profile amendment; only prospective model seeds enter selection and reserved confirmation.",
    "One fixed physical simulator state, not unseen arm-start or hardware success probability.",
    "Degenerate intervals do not establish equivalence; the exact paired small-sample guard remains required.",
    "Physical placement is primary and survives later stops. FINISH is secondary and never supplies object-state success.",
    "Historical request-time veto and pad-only wrist proxy are intentional minimal-profile limitations; this is not identical to current native deployment.",
    "Native inference latency and shared-compute/render variation remain part of the measured closed-loop behavior.",
]
SHARED = (
    "nominal_scene",
    "prepared_episode",
    "horizon_s",
    "delivery_latency_s",
    "adapter_profile",
    "thresholds",
    "runtime_hardware",
    "inference_settings",
    "initialization_requirements",
    "post_stop_observation_s",
    "early_termination",
    "runtime_contract",
    "execution_admission",
    "conditions",
)


def read_checked(spec):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = REPO / path
    if digest(path) != spec["sha256"]:
        raise ValueError(f"Frozen input hash changed: {path}")
    return json.loads(path.read_text())


def validate_design(protocol, protocol_sha, design, stage):
    if (
        not protocol.get("status", "").startswith("frozen")
        or design.get("status") != "frozen"
    ):
        raise ValueError("Frozen protocol and campaign required")
    if (
        design.get("protocol", {}).get("sha256") != protocol_sha
        or design["protocol"].get("stage") != stage
    ):
        raise ValueError("Protocol hash or stage mismatch")
    common = protocol["common_inputs"]
    expected = read_checked(protocol["legacy_screen_template"])
    manifest = read_checked(design["runtime_contract"]["source_manifest"])
    if (
        manifest["output_source"] != common["source"]
        or manifest["base_unchanged_after_assembly"] is not True
        or set(manifest["changed_files"])
        != set(protocol["execution_admission"]["allowed_changed_runtime_files"])
    ):
        raise ValueError("Unapproved minimal runtime assembly")
    expected.update(
        adapter_profile=common["comparison_adapter_profile"],
        execution_admission=protocol["execution_admission"],
    )
    expected["runtime_contract"]["source"] = common["source"]
    expected["runtime_contract"]["source_manifest"] = copy.deepcopy(
        design["runtime_contract"]["source_manifest"]
    )
    for row in expected["runtime_contract"]["source_and_input_hashes"]:
        if row["group"] == "source_sha256":
            relative = str(Path(row["path"]).relative_to(manifest["base_source"]))
            row["path"] = str(Path(common["source"]) / relative)
            row["sha256"] = manifest["source_sha256"][relative]
    if any(design.get(k) != expected[k] for k in SHARED):
        raise ValueError("Frozen physical/profile/hardware inputs changed")
    candidates = {p["id"]: candidate_policy(p) for p in protocol["candidates"]}
    ids = [p["id"] for p in design["policies"]]
    declared = protocol["stages"]["teacher_screen" if stage == "screen" else stage]
    if (
        len(ids) != (4 if stage == "screen" else 2)
        or len(set(ids)) != len(ids)
        or not set(ids) <= set(candidates)
        or design["policies"] != [candidates[p] for p in ids]
    ):
        raise ValueError("Candidate or inference recipe mismatch")
    if stage == "screen" and ids != declared["candidates"]:
        raise ValueError("Screen candidate order changed")
    if design["sampling_seeds"] != declared["sampling_seeds"]:
        raise ValueError("Prospective/reserved seeds changed")
    if design["execution_phases"] != phases(protocol, stage, ids):
        raise ValueError("Execution order changed")
    n = 6 if stage == "screen" else 12
    if design["planned_counts"] != {
        "conditions": 1,
        "seeds_per_condition": n,
        "per_policy": n,
        "primary": 24,
        "secondary": 0,
        "total": 24,
    }:
        raise ValueError("Matched grid counts changed")


def legacy_completion_evidence(record, run, trace, execution, planners, thresholds):
    """Read old FINISH telemetry without adding fields to any source file."""
    effective = run
    source = "run_and_execution"
    reasons = {r.get("diagnostics", {}).get("completed_reason") for r in execution}
    reasons.discard(None)
    if "policy_completed_reason" not in run:
        source = "execution_only" if reasons else "no_completion_recorded"
        effective = dict(run)
        effective["policy_completed_reason"] = (
            next(iter(reasons)) if len(reasons) == 1 else None
        )
    row = anchor_trial_evidence(
        record, effective, trace, execution, planners, thresholds
    )
    row["completion_reason_source"] = source
    row["raw_run_has_completion_field"] = "policy_completed_reason" in run
    if len(reasons) > 1:
        row["valid_for_selection"] = False
        row["invalid_reasons"].append("contradictory_execution_completion_reasons")
    return row


def collect(runs, design, sha, out):
    rows = []
    for record in load_trials(runs, design, sha, out).values():
        folder = Path(record["directory"])
        if record.get("metrics") is None:
            rows.append(
                {
                    **{
                        k: record[k]
                        for k in (
                            "policy_id",
                            "condition_id",
                            "sampling_seed",
                            "directory",
                        )
                    },
                    "valid_for_selection": False,
                    "invalid_reasons": record.get(
                        "missing_inputs", ["missing metrics"]
                    ),
                }
            )
            continue
        with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
            row = legacy_completion_evidence(
                record,
                json.loads((folder / "run.json").read_text()),
                trace,
                read_rows(folder / "execution_trace.jsonl"),
                read_rows(folder / "planner_trace.json"),
                design["thresholds"],
            )
        status = folder / "run_status.json"
        if (
            not status.exists()
            or json.loads(status.read_text()).get("status") != "completed"
        ):
            row["valid_for_selection"] = False
            row["invalid_reasons"].append("controller_trial_not_completed")
        row.update(
            input_sha256=record.get("input_sha256", {}),
            case_sha256=record["case_sha256"],
            score_sha256=digest(record["score_path"]),
        )
        rows.append(row)
    return rows


def evaluate_selection(protocol, design, trials, stage):
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


def derive_confirmation(
    protocol, protocol_sha, screen, screen_sha, selection, selection_sha
):
    validate_design(protocol, protocol_sha, screen, "screen")
    if (
        selection.get("campaign_sha256") != screen_sha
        or selection.get("stage") != "screen"
        or selection.get("status") != "complete_valid_matched_stage"
    ):
        raise ValueError("Complete valid hash-linked screen selection required")
    recomputed = evaluate_selection(
        protocol, screen, selection.get("trials", []), "screen"
    )
    if any(
        recomputed.get(k) != selection.get(k)
        for k in ("status", "selected_ids", "ranking")
    ):
        raise ValueError("Selected IDs/ranking disagree with frozen screen rows")
    ids = selection["selected_ids"]
    policies = {p["id"]: p for p in screen["policies"]}
    result = copy.deepcopy(screen)
    result.update(
        campaign_id=protocol["protocol_id"] + "_confirmation",
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        sampling_seeds=list(protocol["stages"]["confirmation"]["sampling_seeds"]),
        policies=[policies[p] for p in ids],
        primary_policy_order=list(ids),
        planned_counts={
            "conditions": 1,
            "seeds_per_condition": 12,
            "per_policy": 12,
            "primary": 24,
            "secondary": 0,
            "total": 24,
        },
        execution_phases=phases(protocol, "confirmation", ids),
        protocol={**screen["protocol"], "stage": "confirmation"},
        confirmation_provenance={
            "protocol_sha256": protocol_sha,
            "screen_campaign_sha256": screen_sha,
            "screen_selection_sha256": selection_sha,
            "selected_ids": ids,
        },
    )
    validate_design(protocol, protocol_sha, result, "confirmation")
    return result


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
                for k, v in r.items()
            }
            for r in rows
        )
    lines = [
        f"# Teacher {result['stage']} comparison",
        "",
        f"Status: **{result['status']}**.",
        "",
        f"Expected {result['expected_trials']}; missing {len(result['missing_trial_keys'])}; invalid {len(result['invalid_trial_keys'])}.",
        "",
        "| Candidate | Physical place | Final support | Clean FINISH | Lift | Acquire | Pre-place safety | Later safety | Drops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result.get("ranking", []):
        lines.append(
            "| "
            + " | ".join(
                str(r[k])
                for k in [
                    "policy_id",
                    "strict_full_place",
                    "final_supported_placement",
                    "clean_place",
                    "lifted",
                    "acquired",
                    "pre_placement_safety_stop",
                    "post_placement_safety_stop",
                    "dropped",
                ]
            )
            + " |"
        )
    lines += [
        "",
        f"Conclusion: {result.get('conclusion', 'Selection withheld')}; clear simulator winner: {result['clear_simulator_winner']}.",
        "",
        "[All gates and rows](selection.json) · [CSV](trials.csv). Completion may come from consistent execution diagnostics when the legacy run header omits it. Controller flags never supply physical object success.",
        "",
    ]
    lines += ["- " + t for t in LIMITATIONS]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("score", "confirm"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "confirmation"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    protocol_sha = digest(args.protocol)
    design, sha = load_design(args.campaign)
    if args.operation == "score":
        if args.runs is None or args.stage is None:
            parser.error("score requires --runs and --stage")
        validate_design(protocol, protocol_sha, design, args.stage)
        rows = collect(args.runs, design, sha, args.out / "raw_analysis")
        result = evaluate_selection(protocol, design, rows, args.stage)
        result.update(
            protocol_sha256=protocol_sha,
            campaign_sha256=sha,
            campaign=str(args.campaign),
            runs=str(args.runs.absolute()),
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
    if args.selection is None:
        parser.error("confirm requires --selection")
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite confirmation")
    selection = json.loads(args.selection.read_text())
    result = derive_confirmation(
        protocol, protocol_sha, design, sha, selection, digest(args.selection)
    )
    with tempfile.TemporaryDirectory(prefix="teacher_confirmation_audit_") as temp:
        fresh = collect(Path(selection["runs"]), design, sha, Path(temp))
    recomputed = evaluate_selection(protocol, design, fresh, "screen")
    if any(
        recomputed.get(k) != selection.get(k)
        for k in ("status", "selected_ids", "ranking")
    ):
        raise ValueError("Raw rescore differs from selected screen")
    before = {(r["policy_id"], r["sampling_seed"]): r for r in selection["trials"]}
    if any(
        r.get("input_sha256")
        != before[r["policy_id"], r["sampling_seed"]].get("input_sha256")
        or r.get("case_sha256")
        != before[r["policy_id"], r["sampling_seed"]].get("case_sha256")
        for r in fresh
    ):
        raise ValueError("Raw screen input hashes changed")
    result["confirmation_provenance"].update(
        screen_campaign_path=str(args.campaign.absolute()),
        screen_selection_path=str(args.selection.absolute()),
        raw_screen_rescore="all24 cases match frozen selection and input hashes",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "path": str(args.out),
                "sha256": digest(args.out),
                "selected_ids": selection["selected_ids"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
