"""Read-only first completed corrected-delivery v2 case audit. Run by stdin from frozen source."""

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
ROOT = BASE / "runs/teacher_robustness_v2_delivery/screen"
SOURCE = BASE / "source_teacher_v2_delivery"
EXPECTED_DESIGN = "5ea36cfb5404b6e20461a251beb1173c9e0ee809e279dfca4733e6acc502bda0"
EXPECTED_PARENT = "5cba45bfb944d621a2b5ffc2850366aba83c60b0c95f013b3593d37fc5ff8b70"


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
assert digest(SOURCE / design["protocol"]["path"]) == EXPECTED_PARENT
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
wrist_rows = read_rows(folder / "wrist_contact_trace.jsonl")
wrist_t = np.asarray([r["t"] for r in wrist_rows])
wrist_values = np.asarray([r["wrist_ft"] for r in wrist_rows])
initial = json.loads(Path(condition["initial_state"]["path"]).read_text())
bias = np.asarray(initial["wrist_ft"])
wrist_error = {
    "impulse_to_force": 0.0,
    "actor_wrench": 0.0,
    "total_wrench": 0.0,
    "bias_addition": 0.0,
}
actor_names = {"gripper_housing", "left_pad", "right_pad"}
for row in wrist_rows:
    assert np.array_equal(row["recorded_bias"], bias)
    assert row["physics_dt_s"] == config["physics"]["dt"]
    assert row["api_dt_argument"] == 1.0
    assert {a["actor_path"].rsplit("/", 1)[-1] for a in row["per_actor"]} == actor_names
    total = np.zeros(6)
    for actor in row["per_actor"]:
        value = np.zeros(6)
        for contact in actor["contacts"]:
            scalar = (
                np.asarray(contact["normal_impulse_signed_ns"]) / row["physics_dt_s"]
            )
            wrist_error["impulse_to_force"] = max(
                wrist_error["impulse_to_force"],
                float(abs(scalar - contact["normal_force_signed_n"]).max()),
            )
            forces = scalar[:, None] * np.asarray(contact["normals_world"])
            value[:3] += forces.sum(axis=0)
            value[3:] += np.cross(
                np.asarray(contact["points_world_m"]) - np.asarray(row["tcp_pose"][:3]),
                forces,
            ).sum(axis=0)
        wrist_error["actor_wrench"] = max(
            wrist_error["actor_wrench"],
            float(abs(value - actor["normal_wrench_world"]).max()),
        )
        total += value
    wrist_error["total_wrench"] = max(
        wrist_error["total_wrench"],
        float(abs(total - row["normal_wrench_world"]).max()),
    )
    wrist_error["bias_addition"] = max(
        wrist_error["bias_addition"], float(abs(total + bias - row["wrist_ft"]).max())
    )
with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
    idx = np.searchsorted(wrist_t, trace["t"] + 1e-10, side="right") - 1
    assert (idx >= 0).all()
    wrist_error["frame_cache_value"] = float(
        abs(trace["wrist_ft"] - wrist_values[idx]).max()
    )
    wrist_error["frame_cache_t"] = float(
        abs(trace["wrist_capture_t"] - wrist_t[idx]).max()
    )
    wrist_error["125hz_clock_s"] = float(
        abs(wrist_t[1:] - (wrist_t[1] + np.arange(len(wrist_t) - 1) / 125)).max()
    )
wrist_audit = {
    "samples": len(wrist_rows),
    "first_executor_sample_t_s": float(wrist_t[1]),
    "clock_semantics": "Initial t0 sample, then 125Hz executor samples after the first rendered camera becomes available.",
    "start_s": float(wrist_t[0]),
    "end_s": float(wrist_t[-1]),
    "maximum_arithmetic_error": wrist_error,
    "metadata": info.get("wrist_proxy_metadata"),
    "peak_contact_force_norm_n": float(
        np.linalg.norm(wrist_values[:, :3] - bias[:3], axis=1).max()
    ),
    "trace_sha256": digest(folder / "wrist_contact_trace.jsonl"),
}
checks = {
    "current_delivery_veto": info.get("terminal_veto_feedback_source")
    == "current_delivery",
    "gripper_wrist_metadata": run.get("wrist_model")
    == info.get("wrist_model")
    == "gripper_contact_proxy"
    and run.get("wrist_proxy_metadata") == info.get("wrist_proxy_metadata")
    and info.get("wrist_sampling_rate_hz") == 125,
    "gripper_wrist_arithmetic_and_cache": max(wrist_error.values()) < 1e-9,
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
    wrist_audit=wrist_audit,
    effective_settings=info.get("effective"),
    server_weights=server.get("weights"),
    initial_state_provenance=info.get("policy_initial_state_provenance"),
    baseline_provenance=info.get("tactile_baseline_provenance"),
    physical_outcomes=metrics["outcomes"],
    selection_evidence=selection,
)
print(json.dumps(result, indent=2, allow_nan=False))
