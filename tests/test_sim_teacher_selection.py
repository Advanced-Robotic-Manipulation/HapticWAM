import copy

import numpy as np
import pytest

from tools.sim.select_teacher_candidate import evaluate_selection, trial_evidence


def design(stage="screen"):
    npolicy, nstarts = (4, 4) if stage == "screen" else (2, 6)
    return {
        "policies": [
            {"id": f"teacher{i}", "architecture": "teacher"} for i in range(npolicy)
        ],
        "conditions": [{"id": f"start{i}"} for i in range(nstarts)],
        "sampling_seeds": [71, 73],
        "thresholds": {
            "require_support_verified_release": True,
            "support_robot_force_max_n": 0.1,
            "support_bin_force_min_n": 0.1,
        },
    }


def grid(d):
    return [
        {
            "policy_id": p["id"],
            "condition_id": c["id"],
            "sampling_seed": s,
            "valid_for_selection": True,
            "clean_place": False,
            "strict_full_place": False,
            "lifted": False,
            "acquired": False,
            "carried": False,
            "released_in_bin": False,
            "safety_stop": False,
            "actual_stop": False,
            "dropped": False,
            "wrench_limit_stop": False,
            "tactile_force_limit_stop": False,
        }
        for p in d["policies"]
        for c in d["conditions"]
        for s in d["sampling_seeds"]
    ]


def physical_trial(full=True, completion=True, stop=None, terminal_events=None):
    record = {
        "policy_id": "teacher0",
        "condition_id": "start0",
        "sampling_seed": 71,
        "directory": "/synthetic",
        "status": "scored",
        "metrics": {
            "valid_for_scoring": True,
            "invalid_reasons": [],
            "outcomes": {"full_task": full, "acquired": full, "lifted": full},
            "object": {"final_inside_bin": True, "final_contacts_unloaded": True},
            "placement_support": {"required": True, "verified_placement": full},
            "control": {},
        },
    }
    reason = "placement_release_finished" if completion else None
    run = {"policy_completed_reason": reason, "policy_stop_reason": stop}
    execution = [
        {
            "t": 10.0,
            "stopped": False,
            "diagnostics": {
                "completed_reason": reason,
                "completed_at_s": 10.0 if completion else None,
                "safety_events": ["workspace_clamp"],
            },
        }
    ]
    if stop:
        execution.append(
            {
                "t": 12.0,
                "stopped": True,
                "stop_reason": stop,
                "diagnostics": {
                    "completed_reason": reason,
                    "safety_events": terminal_events or [],
                },
            }
        )
    trace = {
        "packet_robot_normal_force": np.array([0.0, 0.0]),
        "packet_bin_normal_force": np.array([0.35, 0.35]),
    }
    return record, run, trace, execution


def test_controller_completion_without_physical_task_never_earns_clean_place():
    inputs = physical_trial(full=False)
    row = trial_evidence(*inputs, [], design()["thresholds"])
    assert row["valid_for_selection"] is True
    assert row["completed_reason"] == "placement_release_finished"
    assert row["strict_full_place"] is False
    assert row["clean_place"] is False


def test_final_robot_support_or_lost_bin_support_disqualifies_clean_placement():
    for robot_force, bin_force in ((0.11, 0.35), (0.0, 0.1)):
        inputs = physical_trial()
        inputs[2]["packet_robot_normal_force"][-1] = robot_force
        inputs[2]["packet_bin_normal_force"][-1] = bin_force
        row = trial_evidence(*inputs, [], design()["thresholds"])
        assert row["strict_full_place"] is True
        assert row["clean_place"] is False


def test_later_safety_stop_preserves_strict_task_but_prevents_clean_place():
    row = trial_evidence(
        *physical_trial(stop="safety_stop", terminal_events=["wrench_limit"]),
        [],
        design()["thresholds"],
    )
    assert row["strict_full_place"] is True
    assert row["clean_place"] is False
    assert row["wrench_limit_stop"] is True
    assert row["terminal_safety_events"] == ["wrench_limit"]
    assert row["stop_time_s"] == 12.0


def test_nonterminal_workspace_clamp_is_not_a_safety_stop_and_latencies_stay_distinct():
    plans = [
        {
            "latency_s": 0.4,
            "inference_wall_time_s": 0.8,
            "diagnostics": {
                "sim_native_inference_latency_s": 0.7,
                "sim_effective_inference_latency_s": 0.4,
            },
        }
    ]
    row = trial_evidence(*physical_trial(), plans, design()["thresholds"])
    assert row["clean_place"] is True
    assert row["safety_stop"] is False
    assert row["native_inference_latency_s"]["mean"] == 0.7
    assert row["delivery_latency_s"]["mean"] == 0.4
    assert row["inference_wall_time_s"]["mean"] == 0.8


def test_missing_final_support_and_ambiguous_terminal_cause_are_invalid():
    inputs = physical_trial(stop="safety_stop")
    del inputs[2]["packet_bin_normal_force"]
    row = trial_evidence(*inputs, [], design()["thresholds"])
    assert row["valid_for_selection"] is False
    assert "terminal_safety_kind_missing" in row["invalid_reasons"]
    assert "missing_or_invalid_final_packet_bin_normal_force" in row["invalid_reasons"]


def test_screen_requires_complete_valid_matched_grid():
    d = design()
    rows = grid(d)
    result = evaluate_selection(d, rows[:-1], "screen")
    assert result["selected_ids"] == []
    assert len(result["missing_trial_keys"]) == 1
    rows[-1]["valid_for_selection"] = False
    assert evaluate_selection(d, rows, "screen")["selected_ids"] == []
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_selection(d, rows + [rows[0]], "screen")


def test_screen_lexicographic_physical_tiers_and_frozen_order():
    d = design()
    rows = grid(d)
    # One clean placement beats eight non-clean strict placements.
    for r in rows:
        if r["policy_id"] == "teacher0":
            r["strict_full_place"] = True
        if r["policy_id"] == "teacher1":
            r["lifted"] = True
    rows[-1]["clean_place"] = rows[-1]["strict_full_place"] = True
    result = evaluate_selection(d, rows, "screen")
    assert result["selected_ids"] == ["teacher3", "teacher0"]
    tied = evaluate_selection(d, grid(d), "screen")
    assert tied["selected_ids"] == ["teacher0", "teacher1"]
    assert tied["clear_simulator_winner"] is None


def test_confirmation_requires_8_of_12_and_positive_cluster_interval():
    d = design("confirmation")
    rows = grid(d)
    for r in rows:
        if r["policy_id"] == "teacher0" and int(r["condition_id"][-1]) < 4:
            r["clean_place"] = r["strict_full_place"] = r["lifted"] = True
    result = evaluate_selection(d, rows, "confirmation")
    assert result["clear_simulator_winner"] == "teacher0"
    assert result["comparison"]["estimate"]["configuration_clusters"] == 6
    assert result["comparison"]["estimate"]["seeds_per_cluster"] == 2
    rows[0]["clean_place"] = False
    result = evaluate_selection(d, rows, "confirmation")
    assert result["clear_simulator_winner"] is None
    assert result["confirmation_gates"]["clean_place_at_least_8_of_12"] is False


@pytest.mark.parametrize(
    "field", ["dropped", "wrench_limit_stop", "tactile_force_limit_stop"]
)
def test_confirmation_rejects_increased_drop_or_force_stop_even_with_success_advantage(
    field,
):
    d = design("confirmation")
    rows = grid(d)
    for r in rows:
        if r["policy_id"] == "teacher0":
            r["clean_place"] = r["strict_full_place"] = r["lifted"] = True
    rows[0]["clean_place"] = False
    rows[0][field] = True
    result = evaluate_selection(d, rows, "confirmation")
    assert result["clear_simulator_winner"] is None
    assert result["confirmation_gates"]["paired_95pct_lower_bound_positive"] is True


def test_seed_disagreements_are_not_independent_start_clusters():
    d = design("confirmation")
    rows = grid(d)
    for r in rows:
        r["clean_place"] = (r["sampling_seed"] == 71) == (r["policy_id"] == "teacher0")
    result = evaluate_selection(d, rows, "confirmation")
    assert result["comparison"]["estimate"]["interval"] == [0.0, 0.0]
    assert result["clear_simulator_winner"] is None


def test_teacher_only_and_strict_support_protocol_enforced():
    d = design()
    altered = copy.deepcopy(d)
    altered["policies"][0]["architecture"] = "student"
    with pytest.raises(ValueError, match="Teacher-only"):
        evaluate_selection(altered, grid(d), "screen")
    altered = copy.deepcopy(d)
    altered["thresholds"]["require_support_verified_release"] = False
    with pytest.raises(ValueError, match="support-verified"):
        evaluate_selection(altered, grid(d), "screen")
