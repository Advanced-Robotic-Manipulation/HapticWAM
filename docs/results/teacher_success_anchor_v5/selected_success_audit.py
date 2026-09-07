#!/usr/bin/env python3
"""Read-only CPU audit of a preselected held-out illustration, not model selection."""

import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v5"
CASE = "fta1500_nfe1_k4__successful_anchor__seed904510"
RUN = ROOT / "confirmation/rollouts" / CASE
SOURCE = BASE / "source_teacher_anchor_minimal_v5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    execution = lines(RUN / "execution_trace.jsonl")
    delivered = lines(RUN / "delivered_plans.jsonl")
    plans = read(RUN / "planner_trace.json")
    info = read(RUN / "policy_info.json")
    score_path = ROOT / "confirmation/analysis/trials" / f"{CASE}.json"
    score = read(score_path)
    for name, digest in score["input_sha256"].items():
        assert sha(RUN / name) == digest
    completed = [r for r in execution if r["diagnostics"].get("completed_reason")]
    first = completed[0]
    release = first["diagnostics"]["placement_release"]
    events = {}
    for event, time in (
        ("opening", release["opening_since_s"]),
        ("committed", release["committed_at_s"]),
        ("unloaded_open", release["unloaded_since_s"]),
        ("finished", release["finished_at_s"]),
    ):
        row = min(execution, key=lambda r: abs(r["t"] - time))
        assert abs(row["t"] - time) < 1e-8
        events[event] = {
            key: row[key]
            for key in (
                "t",
                "measured_tcp",
                "measured_gripper",
                "gripper_command",
                "active_replan_id",
                "diagnostics",
            )
        }
    opening = [
        r
        for r in execution
        if release["opening_since_s"] <= r["t"] <= release["committed_at_s"]
    ]
    ids = {r["active_replan_id"] for r in opening}
    assert len(ids) == 1
    index = ids.pop()
    current = next(
        p
        for p in delivered
        if p["diagnostics"]["terminal_veto"]["replan_index"] == index
    )
    actions = np.asarray(current["actions"])
    assert np.array_equal(actions, np.asarray(plans[index]["actions"]))
    grip_steps = [min(int(r["diagnostics"]["play_time_s"] * 10), 9) for r in opening]
    teacher_open = actions[grip_steps, 6]
    hold = np.asarray([r["requested_tcp"] for r in completed])
    measured = np.asarray([r["measured_tcp"] for r in completed])
    config = info["placement_release"]
    finish_index = first["active_replan_id"]
    finish_plan = next(
        p
        for p in delivered
        if p["diagnostics"]["terminal_veto"]["replan_index"] == finish_index
    )
    opening_tcp = np.asarray([r["measured_tcp"][:3] for r in opening])
    opening_inside = np.logical_and(
        opening_tcp >= config["tcp_min_m"], opening_tcp <= config["tcp_max_m"]
    ).all()
    dwell = [
        r for r in execution if release["unloaded_since_s"] <= r["t"] <= first["t"]
    ]
    with np.load(RUN / "policy_tactile.npz", allow_pickle=False) as z:
        gel_t = z["t"]
        loads = z["gel_normal_force"]
        load_window = loads[
            (gel_t >= release["unloaded_since_s"] - 0.6) & (gel_t <= first["t"])
        ]
        gel_summary = {
            "sample_count": len(load_window),
            "gel_normal_force_max_n_by_pad": np.max(load_window, axis=0).tolist(),
        }
    with np.load(RUN / "sim_trace.npz", allow_pickle=False) as z:
        times = z["t"]
        after = times >= first["t"]
        physical = {
            "trace_t_s": [float(times[0]), float(times[-1])],
            "scene_samples": len(times),
            "peak_packet_pad_normal_n": float(z["pad_packet_normal_force"].max()),
            "packet_robot_normal_max_after_finish_n": float(
                z["packet_robot_normal_force"][after].max()
            ),
            "packet_bin_normal_final_n": z["packet_bin_normal_force"][-1].tolist(),
            "packet_final_position_m": z["waffle_position"][-1].tolist(),
            "packet_post_finish_max_displacement_m": float(
                np.linalg.norm(
                    z["waffle_position"][after] - z["waffle_position"][after][0], axis=1
                ).max()
            ),
        }
    result = {
        "case_id": CASE,
        "scope": "Selected held-out illustration; all24 confirmation trials remain the denominator. Read-only source and telemetry checks; no model selection or rescoring.",
        "source": str(SOURCE),
        "source_sha256": {
            name: sha(SOURCE / name)
            for name in (
                "phantom/sim/policy_adapter.py",
                "phantom/sim/release_controller.py",
                "tools/sim/deployment_filters.py",
            )
        },
        "input_sha256": {
            name: sha(RUN / name)
            for name in (
                "execution_trace.jsonl",
                "delivered_plans.jsonl",
                "planner_trace.json",
                "policy_tactile.npz",
                "sim_trace.npz",
                "policy_info.json",
                "run.json",
            )
        },
        "score_sha256": sha(score_path),
        "score_input_hashes_match": True,
        "primary_metrics": score["metrics"],
        "release_config": config,
        "events": events,
        "opening_evidence": {
            "unmodified_teacher_delivered_chunk_equals_original": True,
            "active_plan_index": index,
            "execution_rows": len(opening),
            "teacher_grip_sample_range": [
                float(teacher_open.min()),
                float(teacher_open.max()),
            ],
            "all_teacher_grip_samples_at_or_below_open_threshold": bool(
                np.all(teacher_open <= config["open_command_max"])
            ),
            "all_measured_tcp_inside_release_volume": bool(opening_inside),
            "sampling": "Frozen _pose_at: min(int(play_time_s*10),9), max_play_steps10. No pose/grip interpolation is invented.",
        },
        "unloaded_open_dwell": {
            "duration_s": first["t"] - release["unloaded_since_s"],
            "measured_grip_max": max(r["measured_gripper"][0] for r in dwell),
            "accepted_open_at_finish": first["gripper_command"],
            "saved_gel_window": gel_summary,
            "scope": "Saved gel normal force is an uncalibrated proxy; it is not a calibrated physical pressure measurement.",
        },
        "last_plan_before_finish": {
            "index": finish_index,
            "delivery_t_s": finish_plan["delivery_t"],
            "raw_teacher_grip": [a[6] for a in plans[finish_index]["actions"]],
            "delivered_grip": [a[6] for a in finish_plan["actions"]],
            "terminal_veto": finish_plan["diagnostics"]["terminal_veto"],
            "interpretation": "Teacher already commanded opening and physical placement was confirmed before this plan. Native historical recovery_open increases opening to .232 before the final accepted-open gate; final aperture is not attributed solely to the teacher.",
        },
        "finish_hold": {
            "first_t_s": first["t"],
            "last_t_s": completed[-1]["t"],
            "rows": len(completed),
            "all_not_stopped": all(not r["stopped"] for r in execution),
            "all_post_finish_completion_hold": all(
                r["diagnostics"]["completion_hold"] for r in completed
            ),
            "hold_target_equals_first_finish_measured_tcp": bool(
                np.array_equal(hold[0], first["measured_tcp"])
            ),
            "requested_tcp_max_change": float(np.abs(hold - hold[0]).max()),
            "measured_tcp_max_translation_error_m": float(
                np.linalg.norm(measured[:, :3] - hold[:, :3], axis=1).max()
            ),
            "post_finish_replan_count": sum(p["t"] > first["t"] for p in plans),
            "controller_completion_is_object_success": False,
        },
        "physical_summary": physical,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
