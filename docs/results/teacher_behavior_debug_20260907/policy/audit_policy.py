#!/usr/bin/env python3
"""CPU-only audit of saved teacher proposals, delivery transforms and playback.

Reads completed V5 confirmation ftA1500/12 and V7 diagnostic/4. Prints JSON;
never imports inference, mutates recordings, or rescales primary scores.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    a = np.asarray(values, float)
    return (
        {
            "n": len(a),
            "min": float(a.min()),
            "median": float(np.median(a)),
            "max": float(a.max()),
            "sum": float(a.sum()),
        }
        if len(a)
        else {"n": 0}
    )


def integral(actions, play_s, rate=10.0, cap=10):
    u = float(np.clip(play_s * rate, 0, cap - 1e-6))
    weights = np.clip(u - np.arange(len(actions)), 0, 1)
    return weights @ actions[:, :6]


def phase(t, events):
    if events.get("lift") is not None and t >= events["lift"]:
        return "after_lift"
    if events.get("acquisition") is not None and t >= events["acquisition"]:
        return "after_acquisition_before_lift"
    return "before_acquisition"


def audit_case(stage, folder, score_path):
    status = read(folder / "run_status.json")
    score = read(score_path)
    assert status["status"] == "completed" and status["exit_code"] == 0
    assert score["metrics"]["valid_for_scoring"]
    hashes = {f: sha(folder / f) for f in score["input_sha256"]}
    assert hashes == score["input_sha256"]
    for f in ("delivered_plans.jsonl", "run_status.json"):
        hashes[f] = sha(folder / f)
    plans = read(folder / "planner_trace.json")
    delivery = rows(folder / "delivered_plans.jsonl")
    execution = rows(folder / "execution_trace.jsonl")
    info, server = read(folder / "policy_info.json"), read(folder / "server_ready.json")
    assert (
        info["ckpt_sha"]
        == "67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e"
    )
    assert info["effective"]["nfe"] == 1 and info["effective"]["persistent_noise"]
    assert info["terminal_veto"]["implementation"] == "fd4a032"
    hw = info["hardware_effective"]
    rate = hw["control"]["action_rate_hz"]
    cap = info["max_play_steps"]
    assert rate == 10 and cap == 10
    by_t = {p["t"]: p for p in plans if "actions" in p}
    by_id = {p["replan_id"]: p for p in by_t.values()}
    delivered_by_id = {}
    plan_records = []
    previous_accepted = None
    for p in plans:
        if "actions" not in p:
            continue
        d = next((d for d in delivery if d["captured_snapshot_t"] == p["t"]), None)
        a = np.asarray(p["actions"], float)
        diag = p["diagnostics"]
        record = {
            "replan_id": p["replan_id"],
            "request_s": p["t"],
            "native_latency_s": p["latency_s"],
            "status": p["status"],
            "head9_delta_mm": (a[:9, :3].sum(0) * 1000).tolist(),
            "head3_delta_mm": (a[:3, :3].sum(0) * 1000).tolist(),
            "head10_delta_mm": (a[:10, :3].sum(0) * 1000).tolist(),
            "full16_delta_mm": (a[:, :3].sum(0) * 1000).tolist(),
            "head10_grip_range": [float(a[:10, 6].min()), float(a[:10, 6].max())],
            "head10_grip_values": a[:10, 6].tolist(),
            "sigma_max": float(np.max(p["sigma"])),
            "selected_p_contact": 1 - p["p_evt"][0],
            "k_pick": diag.get("k_pick", 0),
            "k_rejected": diag.get("k_rejected", 0),
            "all_k_head_descent_mm": diag.get("head_dz_mm"),
            # Native source uses previous accepted Plan.latency, not delivered
            # wall time or elapsed played action index. Actual CPK tensor/token
            # is not serialized, so this index is explicitly reconstructed.
            "reconstructed_prev_cpk_index": None
            if previous_accepted is None
            else round(previous_accepted["latency_s"] / 1.0),
            "reconstructed_prev_cpk_invalidated": None
            if previous_accepted is None
            else delivered_by_id[previous_accepted["replan_id"]]["diagnostics"][
                "terminal_veto"
            ]["cpk_invalidated"],
        }
        if previous_accepted is not None:
            assert (
                delivered_by_id[previous_accepted["replan_id"]]["delivery_t"] <= p["t"]
            )
            offset = round(
                (p["t"] + p["latency_s"] - previous_accepted["action_times"][0]) * rate
            )
            clipped = int(np.clip(offset, 0, 15))
            reference = np.asarray(
                delivered_by_id[previous_accepted["replan_id"]]["actions"]
            )[clipped:, :6]
            m = min(len(reference), len(a))
            difference = a[:m, :6] - reference[:m]
            squared = (difference**2).sum(0)
            record["selected_continuity_diagnostic"] = {
                "unclipped_offset": offset,
                "clipped_offset": clipped,
                "true_future_overlap": p["t"] + p["latency_s"]
                <= previous_accepted["action_times"][-1],
                "translation_squared_m2": float(squared[:3].sum()),
                "rotation_squared_rad2": float(squared[3:].sum()),
                "rotation_fraction_of_mixed_unit_sum": float(
                    squared[3:].sum() / squared.sum()
                )
                if squared.sum()
                else None,
                "limitation": "Selected candidate only; nonselected full action arrays were not saved. This is not a reconstructed candidate ranking.",
            }
        if d is not None:
            assert by_t[d["captured_snapshot_t"]] is p
            b = np.asarray(d["actions"], float)
            v = d["diagnostics"]["terminal_veto"]
            delivered_by_id[p["replan_id"]] = d
            record.update(
                delivery_s=d["delivery_t"],
                activated=d["activated"],
                veto=v,
                pose_changed=bool(np.any(a[:, :6] != b[:, :6])),
                grip_changed_samples=int(np.sum(a[:, 6] != b[:, 6])),
                pose_delta_max_abs=float(np.max(abs(a[:, :6] - b[:, :6]))),
                full16_upward_removed_mm=float(
                    (np.maximum(a[:, 2], 0).sum() - np.maximum(b[:, 2], 0).sum()) * 1000
                ),
                full16_lateral_change_mm=float(
                    np.linalg.norm((a[:, :2] - b[:, :2]).sum(0)) * 1000
                ),
                delivered_head10_delta_mm=(b[:10, :3].sum(0) * 1000).tolist(),
            )
            if d["activated"]:
                previous_accepted = p
        plan_records.append(record)

    events = score["metrics"]["event_times_s"]
    phase_stats = defaultdict(lambda: defaultdict(float))
    errors, clip_offsets, veto_offsets, grip_errors = [], [], [], []
    activity, milestones = {}, []
    current, previous, last_cmd = None, None, None
    last_release, last_latched = None, False
    for index, row in enumerate(execution):
        t, diag = row["t"], row["diagnostics"]
        dt = row["command_dt"]
        duration = execution[index + 1]["t"] - t if index + 1 < len(execution) else 0
        pid = row["active_replan_id"]
        finished = bool(diag.get("completed_reason"))
        stopped = row["stopped"]
        release = diag.get("placement_release", {})
        latched = diag.get("grip_latch") is not None
        if release.get("phase") != last_release or latched != last_latched:
            milestones.append(
                {
                    "t_s": t,
                    "replan_id": pid,
                    "release": release,
                    "grip_latch": diag.get("grip_latch"),
                    "measured_tcp": row["measured_tcp"],
                    "measured_grip": row["measured_gripper"][0],
                    "commanded_grip": row["gripper_command"],
                }
            )
            last_release, last_latched = release.get("phase"), latched
        if diag["plan_activated"]:
            p, d = by_id[pid], delivered_by_id[pid]
            a, b = np.asarray(p["actions"]), np.asarray(d["actions"])
            elapsed = max(0, t - p["action_times"][0])
            assert last_cmd is not None
            previous = current
            current = {
                "id": pid,
                "a": a,
                "b": b,
                "swap": t,
                "anchor_b": last_cmd - integral(b, elapsed),
                "anchor_a": last_cmd - integral(a, elapsed),
                "play": elapsed,
                "sigma": p["sigma"],
            }
            activity[pid] = {
                "first_s": t,
                "last_s": t,
                "initial_elapsed_s": elapsed,
                "last_play_s": elapsed,
                "played_up_raw_mm": 0.0,
                "played_up_delivered_mm": 0.0,
                "played_down_raw_mm": 0.0,
                "played_xy_raw_mm": 0.0,
                "played_raw_grip_min": 1.0,
                "played_raw_grip_max": 0.0,
                "played_delivered_grip_min": 1.0,
                "played_delivered_grip_max": 0.0,
                "played_command_grip_min": 1.0,
                "played_command_grip_max": 0.0,
                "played_measured_grip_min": 1.0,
                "played_measured_grip_max": 0.0,
            }
        if current is not None and not finished and not stopped:
            assert current["id"] == pid
            cat = phase(t, events)
            counter = phase_stats[cat]
            if index:
                command_delta = (
                    np.asarray(row["requested_tcp"][:3])
                    - execution[index - 1]["requested_tcp"][:3]
                )
                counter["requested_up_mm"] += float(max(command_delta[2], 0) * 1000)
                counter["requested_down_mm"] += float(max(-command_delta[2], 0) * 1000)
                counter["requested_xy_path_mm"] += float(
                    np.linalg.norm(command_delta[:2]) * 1000
                )
            play, old_play = diag["play_time_s"], current["play"]
            old_u, new_u = np.clip(np.array([old_play, play]) * rate, 0, cap - 1e-6)
            weights = np.clip(new_u - np.arange(16), 0, 1) - np.clip(
                old_u - np.arange(16), 0, 1
            )
            for label, vals in (
                ("played_up_raw_mm", np.maximum(current["a"][:, 2], 0)),
                ("played_up_delivered_mm", np.maximum(current["b"][:, 2], 0)),
                ("played_down_raw_mm", np.maximum(-current["a"][:, 2], 0)),
                ("played_xy_raw_mm", np.linalg.norm(current["a"][:, :2], axis=1)),
            ):
                value = float(weights @ vals * 1000)
                counter[label] += value
                activity[pid][label] += value
            current["play"] = play
            activity[pid].update(last_s=t, last_play_s=play)
            pre = current["anchor_b"] + integral(current["b"], play)
            raw_pre = current["anchor_a"] + integral(current["a"], play)
            if (
                previous is not None
                and t - current["swap"] < hw["control"]["chunk_blend_s"]
            ):
                gov = hw["safety"]["governor"]
                factor = 1 - np.clip(
                    (max(previous["sigma"]) - gov["sigma_lo"])
                    / (gov["sigma_hi"] - gov["sigma_lo"]),
                    0,
                    1,
                ) * (1 - gov["min_scale"])
                previous["play"] += dt * factor
                beta = (t - current["swap"]) / hw["control"]["chunk_blend_s"]
                pre = beta * pre + (1 - beta) * (
                    previous["anchor_b"] + integral(previous["b"], previous["play"])
                )
                raw_pre = beta * raw_pre + (1 - beta) * (
                    previous["anchor_a"] + integral(previous["a"], previous["play"])
                )
            else:
                previous = None
            if diag["stale_plan_hold"]:
                counter["stale_hold_s"] += duration
            else:
                xyz = pre[:3].copy()
                if "workspace_clamp" in diag["safety_events"]:
                    xyz = np.clip(
                        xyz,
                        *np.asarray([hw["safety"]["workspace_m"][k] for k in "xyz"]).T,
                    )
                delta = xyz - last_cmd[:3]
                length, limit = (
                    np.linalg.norm(delta),
                    hw["arm"]["limits"]["tcp_speed_m_s"] * dt,
                )
                if length > limit:
                    xyz = last_cmd[:3] + delta * limit / length
                    counter["translation_rate_limited_s"] += duration
                err = float(np.linalg.norm(xyz - row["requested_tcp"][:3]))
                errors.append(err)
                clip_offsets.append(float(np.linalg.norm(pre[:3] - xyz)))
                veto_offsets.append(float(np.linalg.norm(raw_pre[:3] - pre[:3])))
            k = int(np.clip(play * rate, 0, cap - 1e-6))
            raw_g = float(current["a"][k, 6])
            delivered_g = float(current["b"][k, 6])
            for label, value in (
                ("raw", raw_g),
                ("delivered", delivered_g),
                ("command", row["gripper_command"]),
                ("measured", row["measured_gripper"][0]),
            ):
                activity[pid][f"played_{label}_grip_min"] = min(
                    activity[pid][f"played_{label}_grip_min"], value
                )
                activity[pid][f"played_{label}_grip_max"] = max(
                    activity[pid][f"played_{label}_grip_max"], value
                )
            expected_g = float(np.clip(delivered_g, 0, hw["gripper"]["max_close_cmd"]))
            if latched:
                expected_g = max(expected_g, diag["grip_latch"])
            grip_errors.append(abs(expected_g - row["gripper_command"]))
            counter["active_s"] += duration
            counter["raw_open_s"] += duration * (raw_g <= 0.45)
            counter["raw_open_masked_s"] += duration * (
                raw_g <= 0.45 and delivered_g > 0.45
            )
            counter["raw_open_not_commanded_s"] += duration * (
                raw_g <= 0.45 and row["gripper_command"] > 0.45
            )
            counter["latch_holds_above_delivered_s"] += duration * (
                row["gripper_command"] > delivered_g + 2 / 255 and latched
            )
            counter["veto_grip_difference_s"] += duration * (
                abs(raw_g - delivered_g) > 2 / 255
            )
            counter["raw_close_blocked_s"] += duration * (
                raw_g > 0.45 and row["gripper_command"] <= 0.45
            )
            counter["raw_open_in_release_window_s"] += duration * (
                raw_g <= 0.45 and release.get("window_active", False)
            )
            counter["cap_dwell_s"] += duration * (play * rate >= cap - 1e-5)
        if row.get("accepted_tcp") is not None and row.get("ik_success"):
            last_cmd = np.asarray(row["accepted_tcp"], float)
        elif last_cmd is None:
            last_cmd = np.asarray(row["requested_tcp"], float)
    # Every non-stop, pre-FINISH target must match the independently rebuilt
    # delivered-action integration, blend and translation rate clamp.
    assert max(errors, default=0) < 1e-7, (folder.name, max(errors))
    assert max(grip_errors, default=0) < 1e-7, (folder.name, max(grip_errors))
    for record in plan_records:
        if record["replan_id"] in activity:
            record["playback"] = activity[record["replan_id"]]
    return {
        "stage": stage,
        "case_id": folder.name,
        "folder": str(folder),
        "input_sha256": hashes,
        "score_sha256": sha(score_path),
        "settings": info["effective"],
        "delivery_clock": info.get("policy_delivery_clock", "native"),
        "source_sha256": server["inference_source_sha256"],
        "outcomes": score["metrics"]["outcomes"],
        "event_times_s": events,
        "object_metrics": score["metrics"]["object"],
        "stop_reason": score["metrics"]["control"]["stop_reason"],
        "first_stop": next(
            (
                {
                    "t_s": r["t"],
                    "events": r["diagnostics"]["safety_events"],
                    "measured_tcp": r["measured_tcp"],
                    "commanded_grip": r["gripper_command"],
                }
                for r in execution
                if r["stopped"]
            ),
            None,
        ),
        "plans": len(plans),
        "delivered": len(delivery),
        "pose_changed_plans": sum(bool(p.get("pose_changed")) for p in plan_records),
        "veto_counts": dict(
            Counter(d["diagnostics"]["terminal_veto"]["action"] for d in delivery)
        ),
        "k_picks": dict(Counter(p["k_pick"] for p in plan_records)),
        "k_rejected_calls": sum(p["k_rejected"] > 0 for p in plan_records),
        "phase_totals": dict(phase_stats),
        "milestones": milestones,
        "translation_reconstruction_error_m": stats(errors),
        "gripper_reconstruction_error": stats(grip_errors),
        "translation_preclamp_to_command_offset_m": stats(clip_offsets),
        "raw_to_delivered_rebased_translation_offset_m": stats(veto_offsets),
        "played_steps_per_plan": stats(
            [min(cap, a["last_play_s"] * rate) for a in activity.values()]
        ),
        "plan_records": plan_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/home/physicalai/phantom-icra-2027/sim/waffles"),
    )
    args = parser.parse_args()
    result = []
    for version, stage, count in (("v5", "confirmation", 12), ("v7", "diagnostic", 4)):
        root = args.base / f"runs/teacher_success_anchor_{version}" / stage
        folders = sorted(p for p in (root / "rollouts").glob("fta1500_*") if p.is_dir())
        assert len(folders) == count
        for folder in folders:
            result.append(
                audit_case(
                    version, folder, root / "analysis/trials" / (folder.name + ".json")
                )
            )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "analysis": "Post-experiment descriptive audit; no new inference, rescoring or causal intervention.",
                "trials": result,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
