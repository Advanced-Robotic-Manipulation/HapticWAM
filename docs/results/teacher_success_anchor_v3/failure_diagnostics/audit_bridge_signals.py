#!/usr/bin/env python3
"""Completed combined-profile signal/guard diagnostics; CPU read-only."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v3"
SOURCE = BASE / "source_teacher_v2_delivery"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def radius(q):
    a, b, d = 0.24365, 0.21325, 0.11235
    return float(np.sqrt(a * a + b * b + 2 * a * b * np.cos(q[2]) + d * d))


def compact_execution(row):
    return {
        k: row.get(k)
        for k in (
            "t",
            "measured_tcp",
            "requested_tcp",
            "measured_gripper",
            "gripper_command",
            "measured_wrist_ft",
            "target_q",
            "active_replan_id",
            "diagnostics",
        )
    }


def pair_contacts(row):
    return [
        {
            "actor": actor["actor_path"].split("/")[-1],
            "filter": pair["filter_path"],
            "normal_force_n": pair["normal_force_magnitude_n"],
            "points_world_m": pair["points_world_m"],
            "signed_separation_m": pair["signed_separation_m"],
        }
        for actor in row["per_actor"]
        for pair in actor["contacts"]
        if pair["normal_force_magnitude_n"] > 0.1
    ]


def main():
    hardware_path = BASE / "runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml"
    hw = yaml.safe_load(hardware_path.read_text())
    outputs = []
    for seed in (904301, 904302):
        name = f"teacher__fixed_anchor__seed{seed}"
        path = ROOT / "corrected_profile_bridge/rollouts" / name
        gelpath = ROOT / "components/gel_v2_only/rollouts" / name
        ex, gx = (
            rows(path / "execution_trace.jsonl"),
            rows(gelpath / "execution_trace.jsonl"),
        )
        stop = next(r for r in ex if r["stopped"])
        live = [r for r in ex if r["t"] <= stop["t"]]
        raw = read(path / "planner_trace.json")
        raw_by_id = {p["replan_id"]: p for p in raw}
        delivered = rows(path / "delivered_plans.jsonl")
        rewrites = []
        for delivery in delivered:
            veto = delivery["diagnostics"]["terminal_veto"]
            p = raw_by_id[veto["replan_index"]]
            diff = np.asarray(delivery["actions"]) - np.asarray(p["actions"])
            if np.any(diff != 0):
                rewrites.append(
                    {
                        "replan_id": p["replan_id"],
                        "capture_t_s": p["t"],
                        "delivery_t_s": delivery["delivery_t"],
                        "veto": veto,
                        "changed_channels": np.flatnonzero(
                            (diff != 0).any(axis=0)
                        ).tolist(),
                        "max_absolute_delta_per_channel": np.abs(diff)
                        .max(axis=0)
                        .tolist(),
                        "original_closure_min_max": [
                            min(a[6] for a in p["actions"]),
                            max(a[6] for a in p["actions"]),
                        ],
                    }
                )
        wrist = [
            r for r in rows(path / "wrist_contact_trace.jsonl") if r["t"] <= stop["t"]
        ]
        stop_wrist = wrist[-1]
        contact_summary = {}
        for row in wrist:
            for pair in pair_contacts(row):
                key = pair["actor"] + " -> " + pair["filter"]
                summary = contact_summary.setdefault(
                    key,
                    {"first_over_point1N_s": row["t"], "maximum_normal_force_n": 0.0},
                )
                summary["maximum_normal_force_n"] = max(
                    summary["maximum_normal_force_n"], pair["normal_force_n"]
                )
        gel = [r for r in read(path / "gel_contact_trace.json") if r["t"] <= stop["t"]]
        first_loaded_command_outside = next(
            (
                r
                for r in live
                if not r["stopped"]
                and r.get("target_q") is not None
                and radius(r["target_q"]) > 0.468
            ),
            None,
        )
        source_paths = (
            "phantom/deploy/safety.py",
            "phantom/sim/policy_adapter.py",
            "phantom/sim/kinematics.py",
            "tools/sim/run_waffles.py",
        )
        info, gelinfo = (
            read(path / "policy_info.json"),
            read(gelpath / "policy_info.json"),
        )
        init, gelinit = (
            read(path / "initialization.json"),
            read(gelpath / "initialization.json"),
        )
        profile_keys = (
            "wrist_model",
            "wrist_sampling_rate_hz",
            "gel_contact_coverage",
            "terminal_veto",
            "terminal_veto_feedback_source",
            "placement_release",
            "max_play_steps",
            "policy_latency_override_s",
            "initial_recorded_wrist_bias",
        )
        profile = {
            k: {"bridge": info.get(k), "gel_only": gelinfo.get(k)}
            for k in profile_keys
            if info.get(k) != gelinfo.get(k)
        }
        initial_a, initial_b = (
            np.load(gelpath / "observations/0000.npz"),
            np.load(path / "observations/0000.npz"),
        )
        startup = {
            k: {
                "bitwise_equal": initial_a[k].dtype == initial_b[k].dtype
                and initial_a[k].tobytes() == initial_b[k].tobytes(),
                "rms_delta": float(
                    np.sqrt(
                        np.mean(
                            (initial_a[k].astype(float) - initial_b[k].astype(float))
                            ** 2
                        )
                    )
                ),
            }
            for k in initial_a.files
        }
        gelplans = read(gelpath / "planner_trace.json")
        matches = {}
        events = read(
            ROOT / "corrected_profile_bridge/analysis/trials" / (name + ".json")
        )["metrics"]["event_times_s"]
        gel_events = read(
            ROOT / "components/gel_v2_only/analysis/trials" / (name + ".json")
        )["metrics"]["event_times_s"]
        for stage in ("acquisition", "lift", "carry"):
            if events[stage] is not None and gel_events[stage] is not None:
                a = min(live, key=lambda r: abs(r["t"] - events[stage]))
                b = min(gx, key=lambda r: abs(r["t"] - gel_events[stage]))
                diff = np.asarray(a["measured_tcp"]) - b["measured_tcp"]
                matches[stage] = {
                    "bridge_t_s": a["t"],
                    "gel_t_s": b["t"],
                    "bridge_minus_gel_xyz_m": diff[:3].tolist(),
                    "orientation_difference_deg": float(
                        np.rad2deg(
                            (
                                Rotation.from_rotvec(a["measured_tcp"][3:]).inv()
                                * Rotation.from_rotvec(b["measured_tcp"][3:])
                            ).magnitude()
                        )
                    ),
                    "bridge_wrist_ft": a["measured_wrist_ft"],
                    "gel_wrist_ft": b["measured_wrist_ft"],
                    "bridge_grip_command": a["gripper_command"],
                    "gel_grip_command": b["gripper_command"],
                }
        transport = [
            r
            for r in gx
            if gel_events["carry"] <= r["t"] <= gel_events["release_in_bin"]
        ]
        same_y = min(
            transport, key=lambda r: abs(r["measured_tcp"][1] - stop["measured_tcp"][1])
        )
        outputs.append(
            {
                "seed": seed,
                "folder": str(path),
                "profile_differences_from_gel_only": profile,
                "first_observation_comparison_to_gel_only": startup,
                "scene_config_bytes_equal": sha(path / "effective_config.json")
                == sha(gelpath / "effective_config.json"),
                "initialization_equal": read(path / "initialization.json")
                == read(gelpath / "initialization.json"),
                "initialization_differences": {
                    key: {"bridge": init.get(key), "gel_only": gelinit.get(key)}
                    for key in init.keys() | gelinit.keys()
                    if init.get(key) != gelinit.get(key)
                },
                "first_plan_capture_and_activation_s": {
                    "bridge": [raw[0]["t"], raw[0]["activated_at"]],
                    "gel_only": [gelplans[0]["t"], gelplans[0]["activated_at"]],
                },
                "delivered_plan_count": len(delivered),
                "veto_actions": dict(
                    Counter(
                        p["diagnostics"]["terminal_veto"]["action"] for p in delivered
                    )
                ),
                "rewritten_plan_count": len(rewrites),
                "rewrites": rewrites,
                "first_stop": compact_execution(stop),
                "stop_wrist_normal_wrench_world": stop_wrist["normal_wrench_world"],
                "stop_contacts": pair_contacts(stop_wrist),
                "contact_pair_summary": contact_summary,
                "last_gel_t_s": gel[-1]["t"],
                "last_gel_per_filter": [
                    [
                        {
                            k: p.get(k)
                            for k in (
                                "filter_path",
                                "normal_force_magnitude_n",
                                "gel_compression_n",
                                "ignored_normal_force_n",
                                "coverage",
                            )
                        }
                        for p in pad["per_filter_contacts"]
                    ]
                    for pad in gel[-1]["per_pad"]
                ],
                "first_accepted_command_above_measured_wrist_limit": None
                if first_loaded_command_outside is None
                else {
                    "t_s": first_loaded_command_outside["t"],
                    "target_q": first_loaded_command_outside["target_q"],
                    "target_radius_m": radius(first_loaded_command_outside["target_q"]),
                    "measured_radius_m": radius(
                        first_loaded_command_outside["measured_q"]
                    ),
                    "lead_before_measured_stop_s": stop["t"]
                    - first_loaded_command_outside["t"],
                },
                "stage_matched_gel_comparisons": matches,
                "gel_at_nearest_transport_y": {
                    "t_s": same_y["t"],
                    "measured_tcp": same_y["measured_tcp"],
                    "y_mismatch_m": abs(
                        same_y["measured_tcp"][1] - stop["measured_tcp"][1]
                    ),
                    "wrist_radius_m": radius(same_y["measured_q"]),
                },
                "input_sha256": {
                    f: sha(path / f)
                    for f in (
                        "sim_trace.npz",
                        "execution_trace.jsonl",
                        "planner_trace.json",
                        "delivered_plans.jsonl",
                        "wrist_contact_trace.jsonl",
                        "gel_contact_trace.json",
                        "policy_info.json",
                        "effective_config.json",
                        "initialization.json",
                        "observations/0000.npz",
                    )
                },
                "gel_reference_input_sha256": {
                    f: sha(gelpath / f)
                    for f in (
                        "effective_config.json",
                        "initialization.json",
                        "observations/0000.npz",
                        "execution_trace.jsonl",
                        "planner_trace.json",
                        "policy_info.json",
                    )
                },
                "source_sha256": {f: sha(SOURCE / f) for f in source_paths},
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "Read-only completed combined bridge versus successful gel-only trials; profiles and live timing differ; no causal attribution or new policy score.",
                "hardware_guard": {
                    "wrist_extension_stop_m": hw["safety"]["wrist_extension_stop_m"],
                    "ur_dh_a2_a3_d4_m": hw["safety"]["ur_dh_a2_a3_d4_m"],
                    "reach_clamp_m": hw["safety"]["reach_clamp_m"],
                    "hardware_sha256": sha(hardware_path),
                },
                "cases": outputs,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
