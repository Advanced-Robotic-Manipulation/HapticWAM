#!/usr/bin/env python3
"""Score one zoo trial with the unchanged independent object-state scorer; CPU only.

Writes ``trial_result.json`` with the ordinal stage (0 no grab, 1 grab, 2 pick,
3 in box still gripped, 4 placed), the drop flag, stop reason/time, latency
summary and a latency-confound flag. Nothing here relaxes a threshold.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

STAGES = ("no_grab", "grab", "pick", "in_box_gripped", "placed")


def read_rows(path):
    text = Path(path).read_text()
    if path.suffix == ".json":
        value = json.loads(text)
        return value if isinstance(value, list) else value.get("rows", value)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def stage_of(outcomes, obj):
    if outcomes.get("full_task") or outcomes.get("released_in_bin"):
        return 4
    if outcomes.get("carried") or outcomes.get("lifted"):
        in_box = obj.get("grasped_over_bin_any") or obj.get("grasped_inside_bin_any") or (
            obj.get("final_over_bin") and obj.get("final_bilateral_contact"))
        return 3 if in_box else 2
    if outcomes.get("acquired"):
        return 1
    return 0


def latency_summary(planner_rows, isaac_log, reference_p95=None):
    lat = [r["latency_s"] for r in planner_rows if r.get("latency_s") is not None]
    rpc = [(r.get("diagnostics") or {}).get("sim_policy_replan_wall_time_s") for r in planner_rows]
    rpc = [x for x in rpc if x is not None]
    # the first call carries server warm-up; it is excluded from the gate
    if len(rpc) > 1:
        rpc = rpc[1:]
    if len(lat) > 1:
        lat = lat[1:]
    statuses = {}
    for r in planner_rows:
        statuses[r.get("status")] = statuses.get(r.get("status"), 0) + 1
    capped = 0
    if isaac_log.exists():
        capped = len(re.findall("insufficient_capped_lead", isaac_log.read_text(errors="replace")))
    values = rpc or lat
    summary = {"replans": len(planner_rows), "statuses": statuses, "capped_lead_rejections": capped,
               "rpc_wall_mean_s": float(np.mean(values)) if values else None,
               "rpc_wall_p95_s": float(np.percentile(values, 95)) if values else None,
               "rpc_wall_max_s": float(np.max(values)) if values else None}
    # gate: any observation-epoch admission rejection, or p95 above 1.5x the
    # recipe's single-lane reference (fixed 0.5 s when no reference exists)
    limit = 1.5 * reference_p95 if reference_p95 else 0.5
    summary["gate_p95_limit_s"] = limit
    summary["latency_confounded"] = bool(capped > 0 or (summary["rpc_wall_p95_s"] or 0) > limit)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--latency-reference-p95", type=float, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime))
    from phantom.sim.policy_metrics import evaluate_policy_trace

    folder = args.trial
    out = args.output or folder / "trial_result.json"
    result = {"trial": folder.name, "status": "missing_inputs"}
    required = ["run.json", "sim_trace.npz", "effective_config.json", "execution_trace.jsonl", "planner_trace.json"]
    missing = [n for n in required if not (folder / n).exists()]
    integrity = folder / "native_mechanics_failure.json"
    if missing == ["run.json"] and integrity.exists():
        # the every-step mechanics monitor aborted the rollout before run.json was written;
        # the partial trace is scored with run=None (scorer marks it invalid, flags still
        # observed) so the stage reached *before* the integrity stop is reported separately
        run = None
    elif missing:
        result["missing"] = missing
        run_status = folder / "run_status.json"
        if run_status.exists():
            result["run_status"] = json.loads(run_status.read_text())
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return
    else:
        run = json.loads((folder / "run.json").read_text())
    effective = json.loads((folder / "effective_config.json").read_text())
    thresholds = json.loads(args.thresholds.read_text())
    planner = read_rows(folder / "planner_trace.json")
    with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
        metrics = evaluate_policy_trace(trace, effective, thresholds, run=run,
                                        execution_trace=read_rows(folder / "execution_trace.jsonl"),
                                        planner_trace=planner)
    outcomes = metrics["outcomes"]
    obj = metrics.get("object", {})
    stage = stage_of(outcomes, obj)
    events = (run or {}).get("events") or []
    stop_reason = (run or {}).get("policy_stop_reason")
    safety = [e for e in events if e.get("event") in ("policy_stop", "controller_stop", "no_progress_timeout")]
    if run is None:
        failure = json.loads(integrity.read_text())
        diag = failure.get("diagnostic", {})
        result["integrity_stop"] = {"phase": failure.get("phase"), "t_s": failure.get("t_s"), "gates": diag.get("gates"),
                                    "joint_limit_violation_per_joint_rad": {k: v for k, v in
                                                                           (diag.get("joint_limit_violation_per_joint_rad") or {}).items() if v},
                                    "coupling_max_abs_rad": diag.get("coupling_max_abs_rad")}
        stop_reason = "integrity_stop_native_mechanics"
    status = "integrity_stopped" if run is None else ("scored" if metrics.get("valid_for_scoring") else "invalid")
    result.update(status=status,
                  invalid_reasons=metrics.get("invalid_reasons", []),
                  stage=stage, stage_name=STAGES[stage], outcomes=outcomes, dropped=bool(outcomes.get("dropped")),
                  event_times_s=metrics.get("event_times_s"), stop_reason=stop_reason,
                  stop_events=safety[:3], duration_s=(run or {}).get("duration_s", (result.get("integrity_stop") or {}).get("t_s")),
                  object=obj, control={k: metrics.get("control", {}).get(k) for k in
                                        ("stop_reason", "ik_rejects", "replans", "plans_activated", "effective_latency_s")},
                  latency=latency_summary(planner, folder / "isaac.log", args.latency_reference_p95),
                  reach_error_m=obj.get("reach_error_before_first_closing_motion_m"),
                  min_pad_object_m=obj.get("minimum_pad_midpoint_to_object_m"))
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result[k] for k in ("trial", "status", "stage_name", "stop_reason", "duration_s")},
                     default=str), "latency_confounded=", result["latency"]["latency_confounded"])


if __name__ == "__main__":
    main()
