"""Read-only completed first-start diagnostic. No ranking before all32screen cases."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phantom.sim.policy_metrics import _rotations, evaluate_policy_trace
from tools.sim.analyze_policy_campaign import (
    condition_scene,
    load_design,
    read_rows,
    runtime_audit,
    scene_mismatches,
)
from tools.sim.select_teacher_candidate import trial_evidence

ROOT = Path(
    "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_robustness_v2/screen"
)
EXPECTED = "8445fae218b18a455162b19664e89973707ddfbd83aca88e46acc8d9345f0c98"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


design, sha = load_design(ROOT / "campaign_snapshot.json")
assert sha == EXPECTED
condition = design["conditions"][0]
expected = [
    (p, int(seed)) for p in design["policies"] for seed in design["sampling_seeds"]
]
folders, statuses = [], []
for policy, seed in expected:
    folder = ROOT / "rollouts" / f"{policy['id']}__{condition['id']}__seed{seed}"
    status_path = folder / "run_status.json"
    status = (
        json.loads(status_path.read_text()).get("status")
        if status_path.exists()
        else "missing"
    )
    folders.append(folder)
    statuses.append(status)
result = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "method": __doc__,
    "campaign_sha256": sha,
    "condition_id": condition["id"],
    "expected_trials": 8,
    "completed": statuses.count("completed"),
    "statuses": dict(zip([p.name for p in folders], statuses)),
    "selection_permitted": False,
    "limitation": "One recorded start only. No candidate selection, winner inference, tuning or source mutation; all32matched screen cases required.",
}
if any(s != "completed" for s in statuses):
    result["status"] = "waiting_for_complete_first_start"
    print(json.dumps(result, indent=2))
    raise SystemExit(0)
initials, rows = [], []
for (policy, seed), folder in zip(expected, folders):
    case = json.loads((folder / "case.json").read_text())
    run = json.loads((folder / "run.json").read_text())
    info = json.loads((folder / "policy_info.json").read_text())
    server = json.loads((folder / "server_ready.json").read_text())
    config = json.loads((folder / "effective_config.json").read_text())
    execution = read_rows(folder / "execution_trace.jsonl")
    plans = read_rows(folder / "planner_trace.json")
    with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
        metrics = evaluate_policy_trace(
            trace,
            config,
            design["thresholds"],
            run=run,
            execution_trace=execution,
            planner_trace=plans,
        )
        errors = runtime_audit(
            design,
            policy,
            condition,
            info,
            server,
            run,
            trace["t"],
            metrics["control"]["stop_reason"],
        )
        if scene_mismatches(condition_scene(design, condition), config):
            errors.append("scene_mismatch")
        if case["campaign_sha256"] != sha:
            errors.append("case_campaign_hash_mismatch")
        if case["checkpoint_sha256"] != policy["checkpoint_sha256"]:
            errors.append("case_checkpoint_hash_mismatch")
        if case["server_ready_sha256"] != digest(folder / "server_ready.json"):
            errors.append("server_snapshot_hash_mismatch")
        record = {
            "policy_id": policy["id"],
            "condition_id": condition["id"],
            "sampling_seed": seed,
            "directory": str(folder),
            "metrics": metrics,
            "status": "scored"
            if metrics["valid_for_scoring"] and not errors
            else "invalid",
        }
        row = trial_evidence(record, run, trace, execution, plans, design["thresholds"])
        row["runtime_errors"] = errors
        row["physical_invalid_reasons"] = metrics["invalid_reasons"]
        row["reach_diagnostic"] = metrics.get("reach_diagnostic")
        row["initial_actual_closure"] = float(
            np.asarray(trace["gripper"])[0].reshape(-1)[0]
        )
        row["initial_q"] = np.asarray(trace["q"])[0].tolist()
        row["initial_tcp"] = np.asarray(trace["tcp"])[0].tolist()
        row["initial_packet_center"] = np.asarray(trace["waffle_position"])[0].tolist()
        rotation = _rotations(np.asarray(trace["waffle_orientation_wxyz"][:1]))[0]
        delta = (
            np.asarray(trace["pad_position"])[0].mean(axis=0)
            - np.asarray(trace["waffle_position"])[0]
        )
        local_midpoint = rotation.T @ delta
        row["initial_pad_midpoint_to_object_m"] = float(
            np.linalg.norm(
                np.maximum(
                    np.abs(local_midpoint) - np.asarray(config["waffle"]["size"]) / 2, 0
                )
            )
        )
        row["max_lift_m"] = metrics["object"].get("max_lift_m")
        row["minimum_pad_midpoint_to_object_m"] = metrics["object"].get(
            "minimum_pad_midpoint_to_object_m"
        )
        row["trace_end_s"] = float(trace["t"][-1])
        row["max_clock_error_s"] = float(
            np.max(np.abs(trace["t"] - trace["physics_t"]))
        )
        row["effective_settings"] = info.get("effective")
        row["server_weights"] = server.get("weights")
        initials.append(
            {
                key: row[key]
                for key in (
                    "initial_q",
                    "initial_tcp",
                    "initial_packet_center",
                    "initial_actual_closure",
                )
            }
        )
        rows.append(row)
result.update(
    status="complete_first_start_diagnostic",
    trials=rows,
    valid_for_selection_metadata=sum(r["valid_for_selection"] for r in rows),
    all_source_runtime_and_score_checks_pass=all(
        r["valid_for_selection"] and not r["runtime_errors"] for r in rows
    ),
    initial_state_max_range={
        key: float(np.ptp(np.asarray([r[key] for r in initials]), axis=0).max())
        for key in initials[0]
    },
)
print(json.dumps(result, indent=2, allow_nan=False))
