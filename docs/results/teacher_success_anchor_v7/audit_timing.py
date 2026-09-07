#!/usr/bin/env python3
"""Read-only four-case timing-contract and completion audit; never runs inference."""

import hashlib
import json
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v7"
CAMPAIGN = ROOT / "diagnostic/campaign_snapshot.json"
CAMPAIGN_SHA = "e68ca523862e805538b9f150c8394363091aa9d9dac6bd950b6215fc148d739f"


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summary(values):
    a = np.asarray(values, float)
    if not len(a):
        return {"n": 0}
    return {
        "n": len(a),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.9)),
        "max": float(a.max()),
    }


def case(policy, condition, seed):
    name = f"{policy['id']}__{condition['id']}__seed{seed}"
    folder = ROOT / "diagnostic/rollouts" / name
    score_path = ROOT / "diagnostic/analysis/trials" / (name + ".json")
    if not score_path.exists() or not (folder / "run_status.json").exists():
        return {"case_id": name, "audit_status": "pending", "calls": []}
    status = read(folder / "run_status.json")
    if status["status"] != "completed":
        return {"case_id": name, "audit_status": "pending", "calls": []}
    score = read(score_path)
    for file, digest in score["input_sha256"].items():
        assert sha(folder / file) == digest, (name, file)
    info = read(folder / "policy_info.json")
    plans = read(folder / "planner_trace.json")
    execution = lines(folder / "execution_trace.jsonl")
    cmd = status["command"]
    assert cmd[cmd.index("--policy-delivery-clock") + 1] == "rpc_wall"
    assert "--servo-reach-limiter" not in cmd and "--policy-latency" not in cmd
    assert info["policy_delivery_clock"] == "rpc_wall"
    server = read(folder / "server_ready.json")
    expected_k = policy["inference_settings"]["k_seeds"]
    assert info["effective"]["k_seeds"] == expected_k
    assert server["effective"]["k_seeds"] == expected_k
    assert info["ckpt_sha"] == policy["checkpoint_sha256"]
    assert server["checkpoint_sha256"] == policy["checkpoint_sha256"]
    assert server["weights"] == "EMA"
    for setting_name, config_name in (
        ("nfe", "nfe"),
        ("guidance", "guidance"),
        ("parity_fixes", "parity"),
        ("persistent_noise", "persistent_noise"),
        ("task_text", "task_text"),
    ):
        assert (
            info["effective"][setting_name] == policy["inference_settings"][config_name]
        )
        assert (
            server["effective"][setting_name]
            == policy["inference_settings"][config_name]
        )
    calls = []
    errors = []
    for p in plans:
        if "actions" not in p:
            errors.append(
                {
                    k: p.get(k)
                    for k in ("replan_id", "status", "error_type", "error_category")
                }
            )
            continue
        d = p["diagnostics"]
        t, native = float(p["t"]), float(p["latency_s"])
        wall = float(d["sim_policy_replan_wall_time_s"])
        outer = float(p["inference_wall_time_s"])
        added = float(d["sim_inference_delay_add_s"])
        delivery = float(d["sim_effective_delivery_delay_s"])
        values = [t, native, wall, outer, added, delivery]
        assert np.isfinite(values).all() and min(native, wall, added) >= 0
        assert d["sim_policy_delivery_clock"] == "rpc_wall"
        assert outer + 1e-9 >= wall and wall + 1e-9 >= native
        assert abs(native - d["sim_native_inference_latency_s"]) < 1e-9
        assert abs(added - condition["inference_delay_add_s"]) < 1e-9
        assert abs(delivery - wall - added) < 1e-9
        assert abs(d["sim_policy_delivery_base_s"] - wall) < 1e-9
        assert abs(d["sim_rpc_minus_native_latency_s"] - wall + native) < 1e-9
        assert abs(p["t_created"] - t) < 1e-9
        assert abs(d["sim_inference_request_t"] - t) < 1e-9
        grid = t + native + np.arange(len(p["actions"])) / 10
        assert np.allclose(p["action_times"], grid, atol=1e-8, rtol=0)
        activated = p.get("activated_at")
        if activated is not None:
            assert np.isfinite(activated) and activated + 1e-8 >= t + delivery
        calls.append(
            {
                "replan_id": p["replan_id"],
                "t_s": t,
                "status": p["status"],
                "native_s": native,
                "client_wall_s": wall,
                "outer_runner_s": outer,
                "client_minus_native_s": wall - native,
                "outer_minus_client_s": outer - wall,
                "action_grid_origin_s": float(grid[0]),
                "scheduled_delivery_s": t + delivery,
                "activated_at_s": activated,
                "activation_minus_scheduled_s": None
                if activated is None
                else activated - t - delivery,
                "elapsed_native_head_at_activation_s": None
                if activated is None
                else activated - grid[0],
                "k_seeds": d.get("k_seeds"),
                "k_pick": d.get("k_pick"),
            }
        )
        # Native policy omits K-selection diagnostics for its K1 fast path.
        assert d.get("k_seeds", 1) == expected_k
    complete = [r for r in execution if r["diagnostics"].get("completed_reason")]
    stops = [r for r in execution if r["stopped"]]
    first_stop = (
        None
        if not stops
        else {
            "t_s": stops[0]["t"],
            "reason": stops[0]["stop_reason"],
            "safety_events": stops[0]["diagnostics"].get("safety_events"),
        }
    )
    completion = None
    if complete:
        first = complete[0]
        target = np.asarray(first["requested_tcp"])
        completion = {
            "first_t_s": first["t"],
            "last_t_s": complete[-1]["t"],
            "rows": len(complete),
            "reason": first["diagnostics"]["completed_reason"],
            "release_diagnostics": first["diagnostics"]["placement_release"],
            "held_target_is_achieved_tcp": bool(
                np.array_equal(target, first["measured_tcp"])
            ),
            "requested_target_max_change": float(
                np.max(
                    np.abs(np.asarray([r["requested_tcp"] for r in complete]) - target)
                )
            ),
            "stop_after_completion": any(r["stopped"] for r in complete),
            "replans_after_completion": sum(p["t"] > first["t"] for p in plans),
        }
    durations = {"cap_dwell_excluding_stale_stop_finish_s": 0.0, "stale_hold_s": 0.0}
    for row, after in pairwise(execution):
        if row["stopped"] or row["diagnostics"].get("completed_reason"):
            continue
        dt = after["t"] - row["t"]
        assert dt > 0
        d = row["diagnostics"]
        if d.get("stale_plan_hold"):
            durations["stale_hold_s"] += dt
        elif row.get("active_replan_id") is not None and d["play_time_s"] >= 1.0 - 1e-6:
            durations["cap_dwell_excluding_stale_stop_finish_s"] += dt
    with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
        physical_end = {
            "last_t_s": float(trace["t"][-1]),
            "final_packet_position_m": trace["waffle_position"][-1].tolist(),
            "final_packet_robot_normal_n": float(
                trace["packet_robot_normal_force"][-1]
            ),
            "final_packet_bin_normal_n": float(trace["packet_bin_normal_force"][-1]),
        }
        if complete:
            mask = trace["t"] >= complete[0]["t"]
            physical_end["post_finish_packet_robot_normal_max_n"] = float(
                trace["packet_robot_normal_force"][mask].max()
            )
    return {
        "case_id": name,
        "audit_status": "checked",
        "run_exit_code": status.get("exit_code"),
        "score_valid": score["metrics"]["valid_for_scoring"],
        "all_returned_plan_clock_contracts_pass": True,
        "inference_errors": errors,
        "score_sha256": sha(score_path),
        "score_input_sha256": score["input_sha256"],
        "primary_outcomes": score["metrics"]["outcomes"],
        "event_times_s": score["metrics"]["event_times_s"],
        "placement_support": score["metrics"]["placement_support"],
        "control": score["metrics"]["control"],
        "first_stop": first_stop,
        "completion": completion,
        "physical_end": physical_end,
        "trace_sampling": score["metrics"]["trace_sampling"],
        "hold_duration_diagnostics": durations,
        "calls": calls,
        "timing_summary_s": {
            field: summary([r[field] for r in calls])
            for field in (
                "native_s",
                "client_wall_s",
                "outer_runner_s",
                "client_minus_native_s",
                "outer_minus_client_s",
            )
        },
    }


def main():
    assert sha(CAMPAIGN) == CAMPAIGN_SHA
    design = read(CAMPAIGN)
    cases = [
        case(p, c, s)
        for p in design["policies"]
        for c in design["conditions"]
        for s in design["sampling_seeds"]
    ]
    assert len(cases) == len({c["case_id"] for c in cases}) == 4
    pending = [r["case_id"] for r in cases if r["audit_status"] != "checked"]
    aggregate = {}
    for policy in design["policies"]:
        calls = [
            r
            for c in cases
            if c["case_id"].startswith(policy["id"] + "__")
            for r in c["calls"]
        ]
        aggregate[policy["id"]] = {
            field: summary([r[field] for r in calls])
            for field in (
                "native_s",
                "client_wall_s",
                "outer_runner_s",
                "client_minus_native_s",
                "outer_minus_client_s",
            )
        }
    print(
        json.dumps(
            {
                "scope": "Read-only development timing and completion audit. All four planned cases retained; no model-ranking confirmation or causal attribution of outcome to timing.",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "complete" if not pending else "partial",
                "pending_cases": pending,
                "campaign_sha256": CAMPAIGN_SHA,
                "cases": cases,
                "aggregate_timing_s": aggregate,
                "durations_definition": "Forward measured executor intervals, no extrapolation of last row. Cap dwell excludes stale, stopped and FINISH rows; it is separate from existing held-command metrics.",
                "timing_definition": "Client policy-call wall is measured directly. Outer runner additionally includes snapshot/callback/adapter work. Client-minus-native includes remaining policy/client work, not pure network transport.",
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
