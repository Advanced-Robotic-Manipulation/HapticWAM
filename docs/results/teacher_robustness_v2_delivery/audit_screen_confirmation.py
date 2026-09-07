#!/usr/bin/env python3
"""CPU read-only screen recomputation and reserved-confirmation derivation audit.

Run by stdin from source_teacher_v2_delivery using the live Python environment.
This does not read or score confirmation outcomes.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phantom.sim.policy_metrics import evaluate_policy_trace
from tools.sim.analyze_policy_campaign import (
    condition_scene,
    read_rows,
    runtime_audit,
    scene_mismatches,
)
from tools.sim.freeze_teacher_confirmation import (
    derive_confirmation,
    shared_configuration_sha256,
)
from tools.sim.select_teacher_candidate import evaluate_selection, trial_evidence

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
SOURCE = BASE / "source_teacher_v2_delivery"
ROOT = BASE / "runs/teacher_robustness_v2_delivery"
PARENT_SHA = "5cba45bfb944d621a2b5ffc2850366aba83c60b0c95f013b3593d37fc5ff8b70"
SCREEN_SHA = "5ea36cfb5404b6e20461a251beb1173c9e0ee809e279dfca4733e6acc502bda0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parent_path = SOURCE / "configs/sim/teacher_v2_delivery_protocol.json"
    screen_path = SOURCE / "configs/sim/teacher_v2_delivery_screen.json"
    selection_path = ROOT / "screen/selection/selection.json"
    confirm_path = ROOT / "teacher_v2_delivery_confirmation.json"
    parent, screen, selection, confirm = map(
        read, (parent_path, screen_path, selection_path, confirm_path)
    )
    assert digest(parent_path) == PARENT_SHA
    assert digest(screen_path) == SCREEN_SHA
    assert digest(ROOT / "screen/campaign_snapshot.json") == SCREEN_SHA
    assert selection["campaign_sha256"] == SCREEN_SHA
    assert selection["status"] == "complete_valid_matched_stage"
    assert selection["available_trials"] == selection["expected_trials"] == 32
    assert not selection["missing_trial_keys"] and not selection["invalid_trial_keys"]
    assert read(ROOT / "screen/progress.json")["status"] == "all_trials_completed"
    expected = {
        (p["id"], c["id"], seed)
        for p in screen["policies"]
        for c in screen["conditions"]
        for seed in screen["sampling_seeds"]
    }
    actual = {
        (r["policy_id"], r["condition_id"], r["sampling_seed"])
        for r in selection["trials"]
    }
    assert expected == actual and len(selection["trials"]) == 32
    recomputed_rows, case_audits = [], []
    fields = (
        "valid_for_selection",
        "clean_place",
        "strict_full_place",
        "acquired",
        "lifted",
        "carried",
        "released_in_bin",
        "dropped",
        "safety_stop",
        "actual_stop",
        "stop_reason",
        "stop_time_s",
        "wrench_limit_stop",
        "tactile_force_limit_stop",
        "completed_reason",
        "completion_time_s",
        "event_times_s",
        "final_inside_bin",
        "final_contacts_unloaded",
        "final_bin_supported_robot_unloaded",
        "final_support_forces_n",
    )
    for saved in selection["trials"]:
        folder = Path(saved["directory"])
        assert folder.is_relative_to(ROOT / "screen/rollouts")
        assert read(folder / "run_status.json")["status"] == "completed"
        policy = next(p for p in screen["policies"] if p["id"] == saved["policy_id"])
        condition = next(
            c for c in screen["conditions"] if c["id"] == saved["condition_id"]
        )
        case, run, config, info, server = map(
            read,
            (
                folder / f
                for f in (
                    "case.json",
                    "run.json",
                    "effective_config.json",
                    "policy_info.json",
                    "server_ready.json",
                )
            ),
        )
        assert case["campaign_sha256"] == SCREEN_SHA
        assert case["checkpoint_sha256"] == policy["checkpoint_sha256"]
        assert case["server_ready_sha256"] == digest(folder / "server_ready.json")
        assert not scene_mismatches(condition_scene(screen, condition), config)
        execution, plans = (
            read_rows(folder / "execution_trace.jsonl"),
            read_rows(folder / "planner_trace.json"),
        )
        with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
            metrics = evaluate_policy_trace(
                trace,
                config,
                screen["thresholds"],
                run=run,
                execution_trace=execution,
                planner_trace=plans,
            )
            errors = runtime_audit(
                screen,
                policy,
                condition,
                info,
                server,
                run,
                trace["t"],
                metrics["control"]["stop_reason"],
            )
            assert metrics["valid_for_scoring"] and not errors
            record = {
                "policy_id": saved["policy_id"],
                "condition_id": saved["condition_id"],
                "sampling_seed": saved["sampling_seed"],
                "directory": str(folder),
                "status": "scored",
                "metrics": metrics,
            }
            row = trial_evidence(
                record, run, trace, execution, plans, screen["thresholds"]
            )
        mismatches = [key for key in fields if row.get(key) != saved.get(key)]
        assert not mismatches, (folder.name, mismatches)
        recomputed_rows.append(row)
        case_audits.append(
            {
                "case_id": folder.name,
                "recomputed_fields_match": True,
                "runtime_valid": True,
                "sha256": {
                    f: digest(folder / f)
                    for f in (
                        "sim_trace.npz",
                        "execution_trace.jsonl",
                        "planner_trace.json",
                        "run.json",
                        "effective_config.json",
                        "case.json",
                    )
                },
            }
        )
    rank = evaluate_selection(screen, recomputed_rows, "screen")
    assert rank["ranking"] == selection["ranking"]
    assert rank["selected_ids"] == selection["selected_ids"]
    derived = derive_confirmation(
        parent,
        screen,
        selection,
        protocol_sha=PARENT_SHA,
        screen_sha=SCREEN_SHA,
        selection_sha=digest(selection_path),
    )
    # The recorded derivation time is metadata; every other field must match.
    derived["frozen_at_utc"] = confirm["frozen_at_utc"]
    assert derived == confirm
    assert digest(confirm_path) == digest(ROOT / "confirmation/campaign_snapshot.json")
    shared = shared_configuration_sha256(confirm)
    assert (
        shared
        == shared_configuration_sha256(screen)
        == parent["shared_configuration_sha256"]
    )
    initial_hashes = {}
    for condition in screen["conditions"] + confirm["conditions"]:
        spec = condition["initial_state"]
        assert digest(Path(spec["path"])) == spec["sha256"]
        initial_hashes[condition["id"]] = spec["sha256"]
    baseline = confirm["adapter_profile"]["tactile_baseline"]
    assert digest(Path(baseline["path"])) == baseline["sha256"]
    assert confirm["sampling_seeds"] == [903201, 903202]
    assert {c["id"] for c in confirm["conditions"]}.isdisjoint(
        c["id"] for c in screen["conditions"]
    )
    output = {
        "status": "passed",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All32 completed-screen raw scores, selector table and derived24-trial confirmation configuration; no confirmation outcomes read or scored.",
        "hashes": {
            "parent": PARENT_SHA,
            "screen": SCREEN_SHA,
            "selection": digest(selection_path),
            "confirmation": digest(confirm_path),
            "shared_configuration": shared,
            "baseline": baseline["sha256"],
        },
        "recomputed_screen_cases": len(recomputed_rows),
        "recomputed_fields": list(fields),
        "ranking": rank["ranking"],
        "selected_ids": rank["selected_ids"],
        "screen_total_acquired": sum(r["acquired"] for r in recomputed_rows),
        "screen_total_lifted": sum(r["lifted"] for r in recomputed_rows),
        "screen_total_strict_place": sum(
            r["strict_full_place"] for r in recomputed_rows
        ),
        "confirmation_conditions": [c["id"] for c in confirm["conditions"]],
        "confirmation_sampling_seeds": confirm["sampling_seeds"],
        "confirmation_planned_counts": confirm["planned_counts"],
        "confirmation_execution_phases": confirm["execution_phases"],
        "initial_state_hashes": initial_hashes,
        "case_audits": case_audits,
        "label_notes": [
            "tactile_force_limit_stop includes tactile_fz OR tactile_depth; label tactile force/depth stops.",
            "Safety stops exclude the separate non-safety veto_retry_cap controller stop.",
            "Zero lift/place counts or degenerate intervals do not establish equivalence or hardware success probabilities.",
            "Screen ranking is selection-biased; selected configurations have no sustained pickup demonstrated here.",
        ],
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
