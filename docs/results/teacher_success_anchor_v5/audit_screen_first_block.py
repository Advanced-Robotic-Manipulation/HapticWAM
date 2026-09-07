#!/usr/bin/env python3
"""Audit each completed first-block trial once with the frozen external analyzer.

CPU only. Writes solely screen/live_audit; stops after four cases or 15 minutes.
No policy ranking, raw edits, simulator or model invocation.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v5/screen"
DRIVER = BASE / "source_teacher_anchor_driver_v5"
ANALYZER_SHA = "1c9ff5d13f7e98f60346cd9c27eb834f4720011623dd988ab20c03e2b14c0245"
OUT = ROOT / "live_audit/first_block"
sys.path.insert(0, str(DRIVER))
from tools.sim.analyze_policy_campaign import load_design, load_trials, policy_settings


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"Frozen input changed: {path}: {actual} != {expected}")
    return actual


def array_hash(array):
    a = np.ascontiguousarray(array)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256_array_bytes": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def main():
    os.nice(19)
    frozen = read(ROOT / "frozen_inputs.json")
    check_hash(DRIVER / "tools/sim/analyze_policy_campaign.py", ANALYZER_SHA)
    design, design_sha = load_design(ROOT / "campaign_snapshot.json")
    assert design_sha == frozen["campaign_sha256"]
    initial = design["adapter_profile"]["initial_state"]
    baseline = design["adapter_profile"]["tactile_baseline"]
    profile_sha = {
        "initial_state": check_hash(Path(initial["path"]), initial["sha256"]),
        "tactile_baseline": check_hash(Path(baseline["path"]), baseline["sha256"]),
        "hardware": check_hash(
            Path(frozen["hardware_config"]), frozen["hardware_sha256"]
        ),
        "robot_usd": check_hash(Path(frozen["robot_usd"]), frozen["robot_usd_sha256"]),
    }
    counts = {}
    for root_key, hashes_key in (
        ("source_root", "source_sha256"),
        ("external_controller_source_root", "external_controller_source_sha256"),
        ("live_repository", "live_core_sha256"),
        ("prepared_episode", "episode_sha256"),
    ):
        for name, digest in frozen[hashes_key].items():
            check_hash(Path(frozen[root_key]) / name, digest)
        counts[hashes_key] = len(frozen[hashes_key])
    checkpoint_hashes = {}
    for policy in design["policies"]:
        p = Path(policy["checkpoint"])
        if str(p) not in checkpoint_hashes:
            checkpoint_hashes[str(p)] = check_hash(p, policy["checkpoint_sha256"])
    block = design["execution_phases"][0]
    assert block["sampling_seeds"] == [904401] and len(block["policy_ids"]) == 4
    names = [
        f"{policy}__{block['condition_ids'][0]}__seed904401"
        for policy in block["policy_ids"]
    ]
    policies = {p["id"]: p for p in design["policies"]}
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "status": "waiting_for_first_block",
        "pid": os.getpid(),
        "campaign_sha256": design_sha,
        "frozen_inputs_sha256": sha(ROOT / "frozen_inputs.json"),
        "external_analyzer_path": str(DRIVER / "tools/sim/analyze_policy_campaign.py"),
        "external_analyzer_sha256": ANALYZER_SHA,
        "frozen_hash_counts_verified": counts,
        "profile_input_sha256": profile_sha,
        "checkpoint_file_hashes_verified": checkpoint_hashes,
        "expected_cases": names,
        "cases": [],
        "scope": "Four candidates at shared first-block seed904401; integrity/recipe/time coverage only, no model ranking.",
    }
    deadline = time.monotonic() + 15 * 60
    completed = set()
    summary = OUT / "progress_audit.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "preflight": "pass",
                "frozen_files": sum(counts.values()),
                "first_block": names,
            }
        ),
        flush=True,
    )
    while len(completed) < 4 and time.monotonic() < deadline:
        progress = read(ROOT / "progress.json")
        for name in names:
            if name in completed:
                continue
            trial = progress.get("trials", {}).get(name, {})
            if trial.get("status") != "completed":
                continue
            if trial.get("exit_code") != 0:
                raise RuntimeError(f"Completed trial failed infrastructure: {name}")
            folder = ROOT / "rollouts" / name
            records = load_trials(folder, design, design_sha, OUT)
            assert len(records) == 1
            record = next(iter(records.values()))
            if (
                record["status"] != "scored"
                or not record["metrics"]["valid_for_scoring"]
            ):
                raise RuntimeError(
                    f"External analyzer rejected {name}: {record.get('metrics')}"
                )
            case = read(folder / "case.json")
            server, info, run = (
                read(folder / "server_ready.json"),
                read(folder / "policy_info.json"),
                read(folder / "run.json"),
            )
            recipe = policy_settings(design, policies[case["policy_id"]])
            execution = [
                json.loads(line)
                for line in (folder / "execution_trace.jsonl").read_text().splitlines()
            ]
            stops = [r for r in execution if r["stopped"]]
            finished = [
                r for r in execution if r["diagnostics"].get("completed_reason")
            ]
            plans = read(folder / "planner_trace.json")
            plan_recipes = sorted(
                {
                    (p["diagnostics"].get("nfe"), p["diagnostics"].get("k_seeds"))
                    for p in plans
                }
            )
            assert all(x == (recipe["nfe"], recipe["k_seeds"]) for x in plan_recipes)
            initial_json = read(folder / "initialization.json")
            with np.load(folder / "observations/0000.npz") as z:
                initial_arrays = {k: array_hash(z[k]) for k in z.files if k != "rgb"}
            with np.load(folder / "sim_trace.npz") as z:
                first_t, last_t = float(z["t"][0]), float(z["t"][-1])
                clocks = {
                    "first_trace_t_s": first_t,
                    "last_trace_t_s": last_t,
                    "run_reported_duration_s": run["duration_s"],
                    "configured_horizon_s": design["horizon_s"],
                    "trace_rows": len(z["t"]),
                    "scene_frames": len(z["frame_t"]),
                    "finite_q_tcp_object": bool(
                        all(
                            np.isfinite(z[k]).all()
                            for k in ("q", "qd", "tcp", "waffle_position")
                        )
                    ),
                    "max_abs_reported_minus_physics_time_s": float(
                        np.max(abs(z["t"] - z["physics_t"]))
                    )
                    if "physics_t" in z
                    else None,
                }
            if stops:
                coverage = "recorded_safety_termination_with_observation_tail"
            elif finished:
                coverage = (
                    "execution_FINISH_with_declared_observation_tail_or_full_horizon"
                )
            else:
                coverage = "full_horizon"
                assert (
                    abs(last_t - design["horizon_s"])
                    <= design["thresholds"]["max_sample_gap_s"]
                )
            row = {
                "case_id": name,
                "policy_id": case["policy_id"],
                "sampling_seed": case["sampling_seed"],
                "status": "pass",
                "external_strict_score_valid": True,
                "invalid_reasons": record["metrics"]["invalid_reasons"],
                "independent_score_path": record["score_path"],
                "independent_score_sha256": sha(Path(record["score_path"])),
                "checkpoint_sha256": server["checkpoint_sha256"],
                "actual_weights": server["weights"],
                "actual_system": server["system"],
                "expected_recipe": recipe,
                "server_effective": server["effective"],
                "policy_effective": info["effective"],
                "every_raw_plan_nfe_k": plan_recipes,
                "scene_config_sha256": sha(folder / "effective_config.json"),
                "initialization_sha256": sha(folder / "initialization.json"),
                "configured_and_settled_initialization": initial_json,
                "first_non_rgb_observation_arrays": initial_arrays,
                "initial_state_provenance": info["policy_initial_state_provenance"],
                "tactile_baseline_provenance": info["tactile_baseline_provenance"],
                "runtime_profile": {
                    k: info.get(k)
                    for k in (
                        "tactile_model",
                        "gel_contact_coverage",
                        "wrist_model",
                        "terminal_veto",
                        "terminal_veto_feedback_source",
                        "placement_release",
                        "max_play_steps",
                        "policy_latency_override_s",
                    )
                },
                "coverage": coverage,
                "clocks": clocks,
                "first_stop_t_s": None if not stops else stops[0]["t"],
                "first_safety_events": []
                if not stops
                else stops[0]["diagnostics"]["safety_events"],
                "first_FINISH_execution_t_s": None
                if not finished
                else finished[0]["t"],
                "full_task": record["metrics"]["outcomes"]["full_task"],
                "full_task_t_s": record["metrics"]["event_times_s"]["full_task"],
                "score_input_sha256": record["input_sha256"],
            }
            if result["cases"]:
                ref = result["cases"][0]
                row["comparison_to_first_case"] = {
                    "scene_bytes_identical": row["scene_config_sha256"]
                    == ref["scene_config_sha256"],
                    "initialization_bytes_identical": row["initialization_sha256"]
                    == ref["initialization_sha256"],
                    "non_rgb_arrays_identical": row["first_non_rgb_observation_arrays"]
                    == ref["first_non_rgb_observation_arrays"],
                    "initial_state_hash_identical": row["initial_state_provenance"][
                        "sha256"
                    ]
                    == ref["initial_state_provenance"]["sha256"],
                    "tactile_baseline_hash_identical": row[
                        "tactile_baseline_provenance"
                    ]["sha256"]
                    == ref["tactile_baseline_provenance"]["sha256"],
                }
                assert all(row["comparison_to_first_case"].values()), row[
                    "comparison_to_first_case"
                ]
            result["cases"].append(row)
            completed.add(name)
            result["status"] = (
                "first_block_complete"
                if len(completed) == 4
                else "waiting_for_first_block"
            )
            summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "new_completed_audit": name,
                        "count": len(completed),
                        "recipe_verified": [recipe["nfe"], recipe["k_seeds"]],
                        "strict_valid": True,
                        "coverage": coverage,
                    }
                ),
                flush=True,
            )
        if len(completed) < 4:
            time.sleep(30)
    if len(completed) < 4:
        result["status"] = "bounded_wait_expired_partial_audit"
        summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "audited": len(completed),
                "summary": str(summary),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
