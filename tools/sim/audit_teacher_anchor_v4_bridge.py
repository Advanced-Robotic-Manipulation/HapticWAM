#!/usr/bin/env python3
"""CPU audit of both development bridge trials; never substitute them for model scores."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_design = importlib.import_module("tools.sim.teacher_anchor_v4_design")
PROTOCOL, PROTOCOL_SHA = _design.PROTOCOL, _design.PROTOCOL_SHA
load_protocol, checked_json = _design.load_protocol, _design.checked_json
collect_trials = importlib.import_module(
    "tools.sim.select_teacher_anchor_v4"
).collect_trials
planned_keys = importlib.import_module("tools.sim.analyze_policy_campaign").planned_keys


def evaluate_bridge(protocol, design, rows):
    expected = set(planned_keys(design))
    by_key = {}
    for row in rows:
        key = (row["policy_id"], row["condition_id"], int(row["sampling_seed"]))
        if key in by_key or key not in expected:
            raise ValueError(
                "Duplicate or unplanned bridge case; preserve attempt provenance"
            )
        by_key[key] = row
    if len(expected) != 2 or set(design["sampling_seeds"]) != {904301, 904302}:
        raise ValueError("Bridge must use exactly the two frozen development seeds")
    missing = sorted(expected - set(by_key))
    tolerance = (
        protocol["common_inputs"]["fixed_scene"]["physics"]["dt"]
        + protocol["common_inputs"]["thresholds"]["clock_tolerance_s"]
    )
    trials = []
    for key, row in sorted(by_key.items()):
        end = row.get("raw_trace_end_s")
        duration = row.get("reported_duration_s")
        gates = {
            "valid_complete_policy_trace": row.get("valid_for_selection") is True,
            "strict_support_verified_physical_place": row.get("strict_full_place")
            is True,
            "final_bin_supported_and_robot_unloaded": row.get(
                "final_supported_placement"
            )
            is True,
            "finished_controller": row.get("completed_reason")
            == "placement_release_finished",
            "no_actual_stop": row.get("actual_stop") is False,
            "full_physics_horizon": bool(
                end is not None and np.isfinite(end) and abs(end - 60) <= tolerance
            ),
            "duration_matches_raw_trace": bool(
                end is not None
                and duration is not None
                and np.isfinite(duration)
                and abs(duration - end) <= tolerance
            ),
        }
        trials.append(
            {**row, "bridge_gates": gates, "bridge_passed": all(gates.values())}
        )
    passed = (
        not missing and len(trials) == 2 and all(r["bridge_passed"] for r in trials)
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "blocked_incomplete_or_failed_bridge",
        "protocol_sha256": PROTOCOL_SHA,
        "bridge_campaign_sha256": protocol["bridge"]["campaign_sha256"],
        "expected_trials": 2,
        "available_trials": len(rows),
        "missing_trial_keys": [list(k) for k in missing],
        "trials": trials,
        "horizon_s": 60,
        "horizon_tolerance_s": tolerance,
        "decision": "open_prospective_fixed_profile_screen"
        if passed
        else "stop_at_diagnosis_no_fallback_no_seed_search",
        "interpretation": "Both reused development seeds are required to validate this common corrected profile. They remain excluded from model ranking and do not estimate prospective success probability.",
    }


def audit_bridge(protocol, runs, out):
    design = checked_json(
        protocol["bridge"]["campaign_path"], protocol["bridge"]["campaign_sha256"]
    )
    rows = collect_trials(
        runs, design, protocol["bridge"]["campaign_sha256"], out / "raw_analysis"
    )
    for row in rows:
        row.pop(
            "score_path", None
        )  # Temporary output locations are not evidence identity.
        if row.get("valid_for_selection") is not True:
            continue
        folder = Path(row["directory"])
        run = json.loads((folder / "run.json").read_text())
        with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
            row["raw_trace_start_s"] = float(trace["t"][0])
            row["raw_trace_end_s"] = float(trace["t"][-1])
        row["reported_duration_s"] = run.get("duration_s")
        row["run_mode"] = run.get("mode")
        if row["run_mode"] != "policy":
            row["valid_for_selection"] = False
            row["invalid_reasons"].append("bridge_must_be_actual_policy")
    result = evaluate_bridge(protocol, design, rows)
    result["runs"] = str(Path(runs).absolute())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="New immutable audit JSON path"
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite an existing bridge audit")
    result = audit_bridge(
        load_protocol(args.protocol), args.runs, args.out.parent / "bridge_audit_inputs"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ["status", "decision", "available_trials"]}, indent=2
        )
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
