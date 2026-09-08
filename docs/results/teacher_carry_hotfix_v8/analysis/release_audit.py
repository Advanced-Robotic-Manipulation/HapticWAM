#!/usr/bin/env python3
"""Read-only V8 release audit and observed-input permission counterfactual.

This invokes the frozen release gate only up to first commit. It never changes
recorded commands, scores an alternate physical outcome, or launches physics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location("frozen_v8_release", path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def spells(records, predicate):
    groups, current = [], []
    for row in records:
        if predicate(row):
            current.append(row)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [
        {
            "first_t_s": g[0]["t_s"],
            "last_t_s": g[-1]["t_s"],
            "elapsed_s": g[-1]["t_s"] - g[0]["t_s"],
            "ticks": len(g),
            "sample_duration_s": sum(x["command_dt_s"] for x in g),
            "replan_ids": sorted({x["replan_id"] for x in g}),
            "indices": sorted({x["action_index"] for x in g}),
            "raw_grip_min_max": [
                min(x["raw_grip"] for x in g),
                max(x["raw_grip"] for x in g),
            ],
            "delivered_grip_min_max": [
                min(x["delivered_grip"] for x in g),
                max(x["delivered_grip"] for x in g),
            ],
            "command_grip_min_max": [
                min(x["command_grip"] for x in g),
                max(x["command_grip"] for x in g),
            ],
        }
        for g in groups
    ]


def permission_replay(records, config, frozen_module, timeout):
    gate_config = dict(config, opening_hold_s=timeout)
    gate = frozen_module.PlacementReleaseController(
        frozen_module.PlacementReleaseConfig.from_dict(gate_config)
    )
    previous_latch = None
    previous_grip = None
    mismatches = []
    for r in records:
        gate.note_latch(previous_latch)
        # Until the first commit the holding/unarmed branches do not read
        # pad_loads or finish_permitted. No synthetic tactile is inferred.
        gate.update(
            r["t_s"],
            tcp=r["measured_tcp"],
            policy_grip=r["delivered_grip"],
            measured_grip=r["measured_grip"],
            pad_loads={},
            eligible=r["eligible"],
            accepted_grip=previous_grip,
        )
        gate.note_latch(r["observed_latch"])
        if timeout == config["opening_hold_s"]:
            actual = r["release_diagnostics"]
            if (
                gate.phase != actual["phase"]
                or gate.opening_since != actual["opening_since_s"]
            ):
                mismatches.append(
                    {
                        "t_s": r["t_s"],
                        "replay_phase": gate.phase,
                        "observed_phase": actual["phase"],
                        "replay_opening_since_s": gate.opening_since,
                        "observed_opening_since_s": actual["opening_since_s"],
                    }
                )
        if gate.committed_at is not None:
            return {
                "opening_hold_s": timeout,
                "first_commit_t_s": gate.committed_at,
                "opening_since_s": gate.opening_since,
                "policy_grip_at_commit": r["delivered_grip"],
                "replan_id": r["replan_id"],
                "action_index": r["action_index"],
                "observed_baseline_gripper_command": r["command_grip"],
                "baseline_state_mismatches": mismatches,
            }
        previous_latch = r["observed_latch"]
        previous_grip = r["command_grip"]
    return {
        "opening_hold_s": timeout,
        "first_commit_t_s": None,
        "baseline_state_mismatches": mismatches,
    }


def case(folder, score, frozen_module):
    execution = rows(folder / "execution_trace.jsonl")
    plans = read(folder / "planner_trace.json")
    deliveries = rows(folder / "delivered_plans.jsonl")
    info = read(folder / "policy_info.json")
    config = info["placement_release"]
    pd = {p["replan_id"]: p for p in plans}
    dd = {d["captured_snapshot_t"]: d for d in deliveries}
    rate = info["hardware_effective"]["control"]["action_rate_hz"]
    play_cap = info["max_play_steps"]
    records = []
    for r in execution:
        if r["stopped"]:
            continue
        p = pd.get(r.get("active_replan_id"))
        if p is None:
            continue
        capture = p["diagnostics"]["sim_observation_capture_t"]
        assert capture == p["t"]
        d = dd[capture]
        cap = min(len(p["actions"]), play_cap)
        index = int(np.clip(r["diagnostics"]["play_time_s"] * rate, 0, cap - 1e-6))
        veto = d["diagnostics"].get("terminal_veto", {})
        action = veto.get("action")
        original = action not in ("recovery_open", "recovery_tactile", "retry_cap")
        if action == "close_masked":
            original = index in veto.get("placement_release_passthrough_indices", [])
        diag = r["diagnostics"]
        records.append(
            {
                "t_s": r["t"],
                "command_dt_s": r["command_dt"],
                "replan_id": p["replan_id"],
                "action_index": index,
                "raw_grip": p["actions"][index][6],
                "delivered_grip": d["actions"][index][6],
                "command_grip": r["gripper_command"],
                "measured_grip": r["measured_gripper"][0],
                "measured_tcp": r["measured_tcp"],
                "observed_latch": diag.get("grip_latch"),
                "eligible": bool(original and not diag["stale_plan_hold"]),
                "window_active": diag["placement_release"]["window_active"],
                "release_diagnostics": diag["placement_release"],
            }
        )
    replays = [
        permission_replay(records, config, frozen_module, dwell) for dwell in (0.2, 0.1)
    ]
    assert not replays[0]["baseline_state_mismatches"], (
        "Observed baseline gate did not replay exactly"
    )
    opening = spells(
        records,
        lambda x: (
            x["raw_grip"] <= config["open_command_max"]
            and x["window_active"]
            and x["eligible"]
        ),
    )
    plan_reviews = []
    for p in plans:
        selected = [r for r in records if r["replan_id"] == p["replan_id"]]
        if not selected or not any(r["window_active"] for r in selected):
            continue
        opening_indices = [
            i for i, a in enumerate(p["actions"]) if a[6] <= config["open_command_max"]
        ]
        if not opening_indices:
            continue
        played = sorted({r["action_index"] for r in selected})
        plan_reviews.append(
            {
                "replan_id": p["replan_id"],
                "capture_t_s": p["t"],
                "delivery_t_s": dd[p["t"]]["delivery_t"],
                "raw_opening_indices": opening_indices,
                "played_indices": played,
                "opening_indices_never_played": sorted(
                    set(opening_indices) - set(played)
                ),
                "raw_gripper_chunk": [a[6] for a in p["actions"]],
            }
        )
    with np.load(folder / "sim_trace.npz") as tr:
        t = tr["t"]
        snapshots = []
        for at in [20, 24, 26, 30, 40, 50, 60]:
            if at > float(t[-1]) + 0.008:
                continue
            i = int(np.argmin(abs(t - at)))
            snapshots.append(
                {
                    "t_s": float(t[i]),
                    "object_xyz_m": tr["waffle_position"][i].tolist(),
                    "robot_contact_n": float(tr["packet_robot_normal_force"][i]),
                    "bin_contact_n": float(tr["packet_bin_normal_force"][i]),
                    "measured_gripper": tr["gripper"][i].tolist(),
                }
            )
        support = {
            "bin_contact_peak_n": float(tr["packet_bin_normal_force"].max()),
            "last_robot_contact_n": float(tr["packet_robot_normal_force"][-1]),
            "snapshots": snapshots,
        }
    return {
        "group": folder.parent.parent.name,
        "seed": score["sampling_seed"],
        "raw_directory": str(folder),
        "physical_outcomes": score["metrics"]["outcomes"],
        "config": config,
        "raw_sha256": {
            f: sha(folder / f)
            for f in [
                "execution_trace.jsonl",
                "planner_trace.json",
                "delivered_plans.jsonl",
                "policy_info.json",
                "sim_trace.npz",
            ]
        },
        "eligible_original_opening_spells_inside_release_volume": opening,
        "permission_replay": replays,
        "plans_with_opening_and_active_release_window": plan_reviews,
        "last_release_diagnostics": execution[-1]["diagnostics"]["placement_release"],
        "support": support,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "Use a new output file"
    completion = read(args.root / "complete.json")
    assert completion["status"] == "six_valid_trials_completed"
    plans = read(args.root / "launch_plan.json")["plans"]
    results = []
    for plan in plans:
        group = Path(plan["output"])
        frozen_path = (
            Path(plan["inputs"]["source_root"]) / "phantom/sim/release_controller.py"
        )
        gate = module(frozen_path)
        for index in read(group / "analysis/summary.json")["trials"]:
            score = read(index["score_path"])
            assert score["metrics"]["valid_for_scoring"]
            folder = Path(score["directory"])
            for f, digest in score["input_sha256"].items():
                assert sha(folder / f) == digest, f"Changed scored input: {f}"
            result = case(folder, score, gate)
            result.update(
                score_sha256=sha(index["score_path"]),
                release_controller_path=str(frozen_path),
                release_controller_sha256=sha(frozen_path),
            )
            results.append(result)
    output = {
        "status": "passed",
        "scope": "Read-only observed-input release audit. Counterfactual replay stops at first permission commit; changed arm/gripper physics, tactile observations, subsequent policy replans, retention, placement and safety are NOT inferred.",
        "helper_sha256": sha(__file__),
        "launch_sha256": sha(args.root / "launch_plan.json"),
        "completion_sha256": sha(args.root / "complete.json"),
        "notes": [
            "Original proposal is the selected K4 candidate before terminal veto; unselected candidate arrays were not recorded.",
            "The gate holding branch reads t, TCP, policy gripper and eligibility. Pad loads are not accessed before first commit; the counterfactual does not fabricate tactile observations.",
            "0.200s baseline phase and opening_since values match every replayed observed tick; 0.100s is an unexecuted configuration hypothesis.",
        ],
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": output["status"],
                "cases": len(results),
                "permission_commits": [
                    {
                        "group": c["group"],
                        "seed": c["seed"],
                        "replays": c["permission_replay"],
                    }
                    for c in results
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
