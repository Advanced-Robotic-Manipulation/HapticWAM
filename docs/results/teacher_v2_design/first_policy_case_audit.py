"""Read-only first completed v2 case audit. Run by stdin from frozen source."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phantom.sim.policy_metrics import evaluate_policy_trace
from tools.sim.analyze_policy_campaign import (
    condition_scene,
    load_design,
    read_rows,
    runtime_audit,
    scene_mismatches,
)
from tools.sim.select_teacher_candidate import trial_evidence

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_robustness_v2/screen"
SOURCE = BASE / "source_teacher_v2"
EXPECTED_DESIGN = "8445fae218b18a455162b19664e89973707ddfbd83aca88e46acc8d9345f0c98"
EXPECTED_PARENT = "9226c8c2779116616cb33fe4970fecd1fe0e5204b1ae4fd6ce614be7252b5170"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


result = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "method": __doc__,
    "campaign_root": str(ROOT),
    "status": "waiting_for_first_completed_case",
}
completed = []
for path in sorted((ROOT / "rollouts").glob("*/run_status.json")):
    status = json.loads(path.read_text())
    if status.get("status") == "completed":
        completed.append(path.parent)
if not completed:
    print(json.dumps(result, indent=2))
    raise SystemExit(0)
folder = completed[0]
design, design_sha = load_design(ROOT / "campaign_snapshot.json")
assert design_sha == EXPECTED_DESIGN
assert digest(SOURCE / "configs/sim/teacher_v2_protocol.json") == EXPECTED_PARENT
assert design["protocol"]["sha256"] == EXPECTED_PARENT
case = json.loads((folder / "case.json").read_text())
policy = next(p for p in design["policies"] if p["id"] == case["policy_id"])
condition = next(c for c in design["conditions"] if c["id"] == case["condition_id"])
run = json.loads((folder / "run.json").read_text())
config = json.loads((folder / "effective_config.json").read_text())
info = json.loads((folder / "policy_info.json").read_text())
server = json.loads((folder / "server_ready.json").read_text())
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
    runtime_errors = runtime_audit(
        design,
        policy,
        condition,
        info,
        server,
        run,
        trace["t"],
        metrics["control"]["stop_reason"],
    )
    record = {
        "policy_id": case["policy_id"],
        "condition_id": case["condition_id"],
        "sampling_seed": case["sampling_seed"],
        "directory": str(folder),
        "status": "scored"
        if metrics["valid_for_scoring"] and not runtime_errors
        else "invalid",
        "metrics": metrics,
    }
    selection = trial_evidence(
        record, run, trace, execution, plans, design["thresholds"]
    )
    clock = {
        "samples": len(trace["t"]),
        "start_s": float(trace["t"][0]),
        "end_s": float(trace["t"][-1]),
        "max_physics_vs_t_error_s": float(
            np.max(np.abs(trace["physics_t"] - trace["t"]))
        ),
    }
checks = {
    "campaign_hash": case["campaign_sha256"] == design_sha,
    "checkpoint_hash_case": case["checkpoint_sha256"] == policy["checkpoint_sha256"],
    "checkpoint_hash_file": digest(policy["checkpoint"]) == policy["checkpoint_sha256"],
    "server_ready_hash": case["server_ready_sha256"]
    == digest(folder / "server_ready.json"),
    "scene_matches": not scene_mismatches(condition_scene(design, condition), config),
    "runtime_audit": not runtime_errors,
    "physical_metrics_valid": metrics["valid_for_scoring"],
    "selector_metadata_valid": selection["valid_for_selection"],
}
sources = json.loads((ROOT / "frozen_inputs.json").read_text())
source_mismatches = []
count = 0
for key, base in [
    ("source_sha256", Path(sources["source_root"])),
    ("live_core_sha256", Path(sources["live_repository"])),
    ("episode_sha256", Path(sources["prepared_episode"])),
]:
    for relative, sha in sources[key].items():
        count += 1
        path = base / relative
        if not path.is_file() or digest(path) != sha:
            source_mismatches.append(str(path))
for spec in (condition["initial_state"], design["adapter_profile"]["tactile_baseline"]):
    count += 1
    if digest(spec["path"]) != spec["sha256"]:
        source_mismatches.append(spec["path"])
checks["source_and_initial_inputs"] = not source_mismatches
result.update(
    status="passed" if all(checks.values()) else "failed",
    case_id=folder.name,
    completed_cases_at_capture=len(completed),
    checks=checks,
    source_files_checked=count,
    source_mismatches=source_mismatches,
    runtime_errors=runtime_errors,
    metric_invalid_reasons=metrics["invalid_reasons"],
    clock=clock,
    effective_settings=info.get("effective"),
    server_weights=server.get("weights"),
    initial_state_provenance=info.get("policy_initial_state_provenance"),
    baseline_provenance=info.get("tactile_baseline_provenance"),
    physical_outcomes=metrics["outcomes"],
    selection_evidence=selection,
)
print(json.dumps(result, indent=2, allow_nan=False))
