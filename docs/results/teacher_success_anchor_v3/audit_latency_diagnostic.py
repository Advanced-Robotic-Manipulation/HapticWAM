"""Read completed adapter-delay diagnostic; stdout only, no runtime mutations."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
SOURCE = BASE / "source_teacher_anchor_latency_v3"
ROOT = BASE / "runs/teacher_success_anchor_v3/latency_schedule_seed4242"
NEW = ROOT / "rollout"
OLD = (
    BASE
    / "runs/teacher_pick_place_v1/campaign/rollouts/teacher__placement_xm10_ym10mm__seed4242"
)
sys.path.insert(0, str(SOURCE))
from phantom.sim.policy_metrics import evaluate_policy_trace


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a = np.array(values, float)
    return (
        {
            "count": len(a),
            "min": float(a.min()),
            "mean": float(a.mean()),
            "max": float(a.max()),
        }
        if len(a)
        else {"count": 0}
    )


def data(path):
    return {
        "plans": read(path / "planner_trace.json"),
        "execution": rows(path / "execution_trace.jsonl"),
        "delivered": rows(path / "delivered_plans.jsonl"),
        "run": read(path / "run.json"),
        "config": read(path / "effective_config.json"),
    }


def timing(value):
    result = []
    plans = {row["replan_id"]: row for row in value["plans"]}
    for p in value["plans"]:
        t = p["t"]
        e = next((row for row in reversed(value["execution"]) if row["t"] <= t), {})
        prev_id = e.get("active_replan_id")
        prev = plans.get(prev_id)
        diag = p.get("diagnostics", {})
        native = diag.get("sim_native_inference_latency_s", p["latency_s"])
        delivered = next(
            (row for row in value["delivered"] if row["captured_snapshot_t"] == t), None
        )
        previous_delivered = next(
            (
                row
                for row in value["delivered"]
                if prev and row["captured_snapshot_t"] == prev["t"]
            ),
            None,
        )
        result.append(
            {
                "replan_id": p["replan_id"],
                "request_t_s": t,
                "latency_s": p["latency_s"],
                "native_latency_s": native,
                "rpc_wall_s": p.get("inference_wall_time_s"),
                "action_start_s": p["action_times"][0],
                "activated_at_s": p.get("activated_at"),
                "status": p["status"],
                "k_pick": diag.get("k_pick"),
                "k_rejected": diag.get("k_rejected"),
                "previous_active_plan_id": prev_id,
                "cpk_offset_requested": round(prev["latency_s"]) if prev else None,
                "selection_reference_offset_10hz": int(
                    np.clip(round((t + native - prev["action_times"][0]) * 10), 0, 15)
                )
                if prev
                else None,
                "previous_cpk_invalidated": previous_delivered.get("diagnostics", {})
                .get("terminal_veto", {})
                .get("cpk_invalidated")
                if previous_delivered
                else None,
                "veto_action": delivered.get("diagnostics", {})
                .get("terminal_veto", {})
                .get("action")
                if delivered
                else None,
            }
        )
    return result


old, new = data(OLD), data(NEW)
z = np.load(NEW / "sim_trace.npz")
design = read(BASE / "runs/teacher_pick_place_v1/campaign/campaign_snapshot.json")
score = evaluate_policy_trace(
    z,
    new["config"],
    design["thresholds"],
    run=new["run"],
    execution_trace=new["execution"],
    planner_trace=new["plans"],
)
schedule = read(NEW / "anchor_latency_schedule.json")
a, b = timing(old), timing(new)
assert len(b) == len(schedule["used"])
comparisons = []
for x, y, s, op, np_ in zip(a, b, schedule["used"], old["plans"], new["plans"]):
    assert x["replan_id"] == y["replan_id"] == s["replan_id"]
    delta = np.asarray(np_["actions"]) - np.asarray(op["actions"])
    comparisons.append(
        {
            "replan_id": x["replan_id"],
            "request_time_delta_s": y["request_t_s"] - x["request_t_s"],
            "latency_delta_s": y["latency_s"] - x["latency_s"],
            "action_grid_exact": op["action_times"] == np_["action_times"],
            "activation_delta_s": None
            if y["activated_at_s"] is None or x["activated_at_s"] is None
            else y["activated_at_s"] - x["activated_at_s"],
            "schedule_used_matches_original": s["latency_s"] == x["latency_s"]
            and s["actual_request_t_s"] == y["request_t_s"]
            and s["recorded_request_t_s"] == x["request_t_s"],
            "k_pick_equal": x["k_pick"] == y["k_pick"],
            "cpk_offset_requested_equal": x["cpk_offset_requested"]
            == y["cpk_offset_requested"],
            "selection_reference_offset_equal": x["selection_reference_offset_10hz"]
            == y["selection_reference_offset_10hz"],
            "raw_actions_equal": bool(np.array_equal(op["actions"], np_["actions"])),
            "head10_xyz_endpoint_delta_mm": float(
                np.linalg.norm(delta[:10, :3].sum(0)) * 1000
            ),
            "all_closure_rmse": float(np.sqrt(np.mean(delta[:, 6] ** 2))),
            "old": x,
            "diagnostic": y,
        }
    )
first_difference = lambda key: next((r for r in comparisons if not r[key]), None)
stops = [e for e in new["execution"] if e.get("stopped")]
managed = read(ROOT / "managed.json")
obsa = np.load(OLD / "observations/0000.npz")
obsb = np.load(NEW / "observations/0000.npz")
observation_diff = {}
for key in sorted(set(obsa.files) | set(obsb.files)):
    if key not in obsa or key not in obsb:
        observation_diff[key] = {"missing": True}
        continue
    x, y = obsa[key], obsb[key]
    delta = x.astype(float) - y.astype(float)
    observation_diff[key] = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "bitwise_equal": x.shape == y.shape
        and x.dtype == y.dtype
        and x.tobytes() == y.tobytes(),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "max_abs": float(np.abs(delta).max()),
    }
server_state_candidates = list((ROOT / "server").glob("*.json"))
server_state = {p.name: read(p) for p in server_state_candidates}
release_events = [
    {"t": r["t"], "release": r.get("diagnostics", {}).get("placement_release")}
    for r in new["execution"]
    if r.get("diagnostics", {}).get("placement_release", {}).get("event")
]
result = {
    "scope": "Independent CPU analysis of completed historical adapter-delay diagnostic. No source/run changes, GPU or inference. Excluded from all model-screen/confirmation denominators.",
    "paths": {"original": str(OLD), "diagnostic": str(NEW), "source": str(SOURCE)},
    "scorer_sha256": sha(SOURCE / "phantom/sim/policy_metrics.py"),
    "score": score,
    "original_score": evaluate_policy_trace(
        np.load(OLD / "sim_trace.npz"),
        old["config"],
        design["thresholds"],
        run=old["run"],
        execution_trace=old["execution"],
        planner_trace=old["plans"],
    ),
    "run_summary": {
        "requested_horizon_s": managed["declared_horizon_s"],
        "actual_final_sample_s": float(z["t"][-1]),
        "first_stop": stops[0] if stops else None,
        "post_stop_observation_s": float(z["t"][-1] - stops[0]["t"]) if stops else None,
        "release_events": release_events,
    },
    "schedule": {
        "source_sha256_matches": schedule["sha256"] == sha(OLD / "planner_trace.json"),
        "available_rows": schedule["rows"],
        "consumed_rows": len(schedule["used"]),
        "unused_rows": schedule["rows"] - len(schedule["used"]),
        "all_consumed_match_source": all(
            r["schedule_used_matches_original"] for r in comparisons
        ),
        "all_request_times_match": all(
            r["request_time_delta_s"] == 0 for r in comparisons
        ),
        "all_action_grids_match": all(r["action_grid_exact"] for r in comparisons),
        "all_matched_activations_match": all(
            r["activation_delta_s"] in (0, None) for r in comparisons
        ),
        "observed_native_latency_s": stats([r["native_latency_s"] for r in b]),
        "replayed_adapter_latency_s": stats([r["latency_s"] for r in b]),
        "request_time_error_s": stats([r["request_time_delta_s"] for r in comparisons]),
        "report": schedule,
        "managed_schedule_is_prelaunch_template": managed["schedule"]["used"] == [],
        "limitation": "Only adapter delivery/action-grid timing is fixed. Native K4 selection uses newly measured current latency. All model observations and native outputs remain live. Requested CPK offsets do not guarantee package survival after veto.",
    },
    "selection": {
        "first_k_difference": first_difference("k_pick_equal"),
        "first_requested_cpk_offset_difference": first_difference(
            "cpk_offset_requested_equal"
        ),
        "first_selection_reference_difference": first_difference(
            "selection_reference_offset_equal"
        ),
        "first_raw_action_difference": first_difference("raw_actions_equal"),
        "rows": comparisons,
    },
    "first_observation_comparison": observation_diff,
    "identical_inputs": {
        "effective_config": new["config"] == old["config"],
        "initialization": read(NEW / "initialization.json")
        == read(OLD / "initialization.json"),
        "managed_input_hashes": managed["inputs"],
    },
    "lifecycle": {
        "status": managed["status"],
        "rollout_exit_code": managed["rollout_exit_code"],
        "cleanup": managed["cleanup"],
        "owned_server_pid": managed["owned_server_pid"],
        "owned_isaac_pid": managed["owned_isaac_pid"],
        "owned_server_process_exists": Path(
            "/proc", str(managed["owned_server_pid"])
        ).exists(),
        "owned_isaac_process_exists": Path(
            "/proc", str(managed["owned_isaac_pid"])
        ).exists(),
        "saved_server_statuses": {
            name: {
                k: d.get(k)
                for k in ["status", "pid", "checkpoint_sha256", "weights", "effective"]
            }
            for name, d in server_state.items()
        },
    },
    "server_identities": managed["server_ready"],
    "input_sha256": {
        str(p): sha(p)
        for p in [
            OLD / "planner_trace.json",
            OLD / "execution_trace.jsonl",
            OLD / "observations/0000.npz",
            NEW / "planner_trace.json",
            NEW / "execution_trace.jsonl",
            NEW / "delivered_plans.jsonl",
            NEW / "sim_trace.npz",
            NEW / "run.json",
            NEW / "effective_config.json",
            NEW / "policy_info.json",
            NEW / "anchor_latency_schedule.json",
            NEW / "observations/0000.npz",
            ROOT / "managed.json",
        ]
    },
}
print(json.dumps(result, indent=2, allow_nan=False))
