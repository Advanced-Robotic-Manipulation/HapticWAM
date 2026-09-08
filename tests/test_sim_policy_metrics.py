"""Policy outcomes must follow physical object events, not executor labels."""

import json
from copy import deepcopy

import numpy as np
import pytest

from phantom.sim.policy_metrics import evaluate_policy_trace


@pytest.fixture
def successful_trial():
    t = np.arange(0, 8.001, 0.05)
    n = len(t)
    position = np.c_[
        np.zeros(n),
        np.interp(t, [0, 3, 4, 8], [0, 0, 0.3, 0.3]),
        np.interp(t, [0, 1.5, 2, 5, 6, 8], [0.04, 0.04, 0.14, 0.14, 0.035, 0.035]),
    ]
    contact = np.repeat((((t >= 1) & (t < 5)) * 2.0)[:, None], 2, axis=1)
    trace = {
        "t": t,
        "physics_t": t.copy(),
        "q": np.zeros((n, 6)),
        "tcp": np.c_[position, np.zeros((n, 3))],
        "gripper": np.c_[np.where((t >= 1) & (t < 5), 0.6, 0.0), np.full(n, 3)],
        "waffle_position": position,
        "waffle_orientation_wxyz": np.tile([1.0, 0, 0, 0], (n, 1)),
        "pad_position": position[:, None] + np.array([[-0.025, 0, 0], [0.025, 0, 0]]),
        "pad_packet_normal_force": contact,
        "pad_force": np.zeros((n, 2, 3)),
        "pad_packet_force": np.zeros((n, 2, 3)),
    }
    config = {
        "waffle": {"size": [0.08, 0.03, 0.02]},
        "bin": {"center": [0, 0.3, 0.02], "size": [0.3, 0.2, 0.15], "wall": 0.005},
    }
    run = {
        "mode": "policy",
        "object_dynamics": {
            "rigid_body_dynamic": True,
            "kinematic": False,
            "attachments": [],
            "pose_writes_after_initialization": 0,
        },
    }
    return trace, config, run


def test_complete_task_is_ordered_physical_contact_lift_carry_release(successful_trial):
    trace, config, run = successful_trial
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["valid_for_scoring"]
    assert all(value for name, value in result["outcomes"].items() if name != "dropped")
    assert not result["outcomes"]["dropped"]
    times = result["event_times_s"]
    assert (
        times["reach"]
        < times["closure"]
        <= times["acquisition"]
        < times["lift"]
        < times["carry"]
        < times["release_in_bin"]
        < times["full_task"]
    )
    assert result["object"]["max_lift_m"] == pytest.approx(0.1)
    assert result["object"]["final_inside_bin"]
    json.dumps(result, allow_nan=False)


def test_scoring_is_independent_of_recorded_joint_trajectory_and_absolute_time(
    successful_trial,
):
    trace, config, run = successful_trial
    baseline = evaluate_policy_trace(trace, config, run=run)
    changed = deepcopy(trace)
    changed["q"] = np.random.default_rng(7).normal(size=trace["q"].shape)
    changed["target_q"] = np.full_like(
        trace["q"], np.nan
    )  # No reference tracking gate.
    changed["t"] += 123
    changed["physics_t"] += 123
    evaluated = evaluate_policy_trace(changed, config, run=run)
    assert evaluated["outcomes"] == baseline["outcomes"]
    assert evaluated["valid_for_scoring"]
    for key, onset in baseline["event_times_s"].items():
        if onset is not None:
            assert evaluated["event_times_s"][key] == pytest.approx(onset + 123)


def test_tcp_lift_tactile_floor_load_and_lift_complete_flag_do_not_award_success(
    successful_trial,
):
    trace, config, run = successful_trial
    trace["waffle_position"][:] = trace["waffle_position"][0]
    trace["pad_packet_normal_force"][:] = 0
    trace["pad_force"][:, :, 2] = 20
    trace["gripper"][:, 1] = 2
    run["policy_stop_reason"] = "lift_complete"
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["valid_for_scoring"]
    assert not result["outcomes"]["acquired"]
    assert not result["outcomes"]["lifted"]
    assert not result["outcomes"]["full_task"]
    assert result["control"]["stop_reason"] == "lift_complete"
    assert result["collisions"]["pad_environment_force_peak_n"] == 20


def test_one_frame_lift_spike_cannot_satisfy_elapsed_time_gate(successful_trial):
    trace, config, run = successful_trial
    trace["waffle_position"][:, 2] = 0.04
    trace["waffle_position"][50, 2] = 0.3
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["outcomes"]["acquired"]
    assert not result["outcomes"]["lifted"]
    assert not result["outcomes"]["full_task"]


def test_lift_and_carry_from_separate_failed_grasps_cannot_be_spliced(successful_trial):
    trace, config, run = successful_trial
    t = trace["t"]
    trace["pad_packet_normal_force"][(t >= 2.7) & (t < 3.1)] = 0
    trace["waffle_position"][t >= 2.7, 2] = 0.04
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["outcomes"]["lifted"]
    assert not result["outcomes"]["carried"]
    assert not result["outcomes"]["full_task"]


def test_short_contact_sampling_gap_is_tolerated_without_losing_retained_carry(
    successful_trial,
):
    trace, config, run = successful_trial
    trace["pad_packet_normal_force"][70] = 0
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["outcomes"]["full_task"]


def test_bin_containment_uses_all_rotated_corners_not_object_center(successful_trial):
    trace, config, run = successful_trial
    trace["waffle_position"][:, 1] *= 0.38 / 0.3
    trace["waffle_orientation_wxyz"][:] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["outcomes"]["carried"]
    assert not result["object"]["final_inside_bin"]
    assert not result["outcomes"]["released_in_bin"]


def test_explicit_bin_geometry_preserves_equivalent_legacy_scoring(successful_trial):
    trace, config, run = successful_trial
    baseline = evaluate_policy_trace(trace, config, run=run)
    config["bin"] = {
        "geometry_model": "rectangular_envelope",
        "center": [0, 0.3, 0.02],
        "outer_size": [0.3, 0.2, 0.15],
        "opening_size": [0.29, 0.19],
        "floor_thickness": 0.005,
    }
    assert evaluate_policy_trace(trace, config, run=run) == baseline


@pytest.mark.parametrize("final_x,inside", [(0.13, True), (0.16, False)])
def test_measured_opening_not_outside_envelope_determines_placement(
    successful_trial, final_x, inside
):
    trace, config, run = successful_trial
    config["bin"] = {
        "geometry_model": "rectangular_envelope",
        "center": [0, 0.3, 0.02],
        "outer_size": [0.4, 0.3, 0.19],
        "opening_size": [0.36, 0.26],
        "floor_thickness": 0.005,
    }
    trace["waffle_position"][:, 0] = np.interp(
        trace["t"], [0, 3, 4, 8], [0, 0, final_x, final_x]
    )
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["valid_for_scoring"]
    assert result["object"]["final_inside_bin"] == inside
    assert result["outcomes"]["full_task"] == inside
    assert result["outcomes"]["lifted"]


@pytest.mark.parametrize(
    "floor,final_z,inside",
    [
        (0.005, 0.035, True),
        (0.020, 0.035, False),
        (0.005, 0.2, True),
        (0.005, 0.205, False),
    ],
)
def test_measured_floor_and_rim_height_independently_bound_all_packet_corners(
    successful_trial, floor, final_z, inside
):
    trace, config, run = successful_trial
    config["bin"] = {
        "geometry_model": "rectangular_envelope",
        "center": [0, 0.3, 0.02],
        "outer_size": [0.4, 0.3, 0.19],
        "opening_size": [0.36, 0.26],
        "floor_thickness": floor,
    }
    trace["waffle_position"][trace["t"] >= 6, 2] = final_z
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["valid_for_scoring"]
    assert result["object"]["final_inside_bin"] == inside


def test_release_above_bin_without_settling_is_not_complete_placement(successful_trial):
    trace, config, run = successful_trial
    trace["waffle_position"][trace["t"] >= 5, 2] = 0.25
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["outcomes"]["carried"]
    assert not result["outcomes"]["full_task"]


def test_post_lift_release_and_fall_outside_bin_is_a_drop(successful_trial):
    trace, config, run = successful_trial
    trace["waffle_position"][:, 1] = 0
    result = evaluate_policy_trace(trace, config, run=run)
    assert result["outcomes"]["lifted"]
    assert result["outcomes"]["dropped"]
    assert result["event_times_s"]["first_drop"] > 5
    assert not result["outcomes"]["full_task"]


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_time",
        "clock_skew",
        "nonfinite_pose",
        "quaternion",
        "sparse_time",
        "missing_contact",
    ],
)
def test_invalid_physical_traces_never_earn_success(successful_trial, failure):
    trace, config, run = successful_trial
    if failure == "duplicate_time":
        trace["t"][3] = trace["t"][2]
    elif failure == "clock_skew":
        trace["physics_t"] += 0.01
    elif failure == "nonfinite_pose":
        trace["waffle_position"][2, 0] = np.nan
    elif failure == "quaternion":
        trace["waffle_orientation_wxyz"][2] = 0
    elif failure == "sparse_time":
        trace["t"][3:] += 1
        trace["physics_t"] = trace["t"].copy()
    else:
        del trace["pad_packet_normal_force"]
    result = evaluate_policy_trace(trace, config, run=run)
    assert not result["valid_for_scoring"]
    assert result["invalid_reasons"]
    assert not any(result["outcomes"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        {"attachments": ["fixed_joint"]},
        {"pose_writes_after_initialization": 1},
        {"kinematic": True},
    ],
)
def test_object_provenance_excludes_attached_or_teleported_success(
    successful_trial, mutation
):
    trace, config, run = successful_trial
    run["object_dynamics"].update(mutation)
    result = evaluate_policy_trace(trace, config, run=run)
    assert not result["valid_for_scoring"]
    assert "free_body_provenance_missing_or_failed" in result["invalid_reasons"]


def test_control_rejections_holds_latencies_and_post_success_stop_are_separate(
    successful_trial,
):
    trace, config, run = successful_trial
    run["policy_stop_reason"] = "safety_stop"
    execution = [
        {"t": 1.0, "ik_success": False, "ik_reason": "branch_guard"},
        {"t": 1.2, "ik_success": True, "diagnostics": {"stale_plan_hold": True}},
        {"t": 1.5, "ik_success": True},
    ]
    planners = [
        {
            "t_created": 0.5,
            "latency_s": 0.2,
            "inference_wall_time_s": 0.15,
            "activated_at": 0.75,
        }
    ]
    result = evaluate_policy_trace(
        trace, config, run=run, execution_trace=execution, planner_trace=planners
    )
    assert result["valid_for_scoring"] and result["outcomes"]["full_task"]
    control = result["control"]
    assert control["ik_rejects"] == 1
    assert control["ik_reject_reasons"] == {"branch_guard": 1}
    assert control["hold_duration_s"] == pytest.approx(0.5)
    assert control["effective_latency_s"]["mean"] == 0.2
    assert control["activation_delay_s"]["mean"] == 0.25
    assert control["stop_reason"] == "safety_stop"


def test_reach_proxy_is_diagnostic_and_not_a_success_requirement(successful_trial):
    trace, config, run = successful_trial
    trace["pad_position"][trace["t"] < 1, :, 2] += 0.1
    result = evaluate_policy_trace(trace, config, run=run)
    assert not result["outcomes"]["reach_before_closure"]
    assert result["outcomes"]["full_task"]


def test_partial_initial_closure_has_independent_delta_onset_reach_error(
    successful_trial,
):
    trace, config, run = successful_trial
    trace["gripper"][trace["t"] < 1, 0] = 0.427
    execution = [
        {"t": 0.0, "gripper_command": 0.427},
        {"t": 0.5, "gripper_command": 0.430},  # Below hardware deadband.
        {"t": 1.02, "gripper_command": 0.440},
    ]
    trace["pad_position"][20, :, 2] += 0.03  # OBB half-height is 10 mm.
    result = evaluate_policy_trace(trace, config, run=run, execution_trace=execution)
    diagnostic = result["reach_diagnostic"]
    assert diagnostic["absolute_threshold_already_exceeded_at_start"]
    assert diagnostic["first_closing_motion_s"] == 1.02
    assert diagnostic["sample_time_s"] == 1.0
    assert diagnostic["reach_error_before_first_closing_motion_m"] == pytest.approx(
        0.02
    )
    assert not result["outcomes"]["reach_before_closure"]
    assert result["outcomes"]["full_task"]


def add_support_telemetry(trace):
    trace["packet_robot_normal_force"] = trace["pad_packet_normal_force"].sum(axis=1)
    trace["packet_bin_normal_force"] = np.where(trace["t"] >= 6, 0.35, 0.0)


def test_strict_placement_requires_free_packet_supported_by_bin(successful_trial):
    trace, config, run = successful_trial
    add_support_telemetry(trace)
    result = evaluate_policy_trace(
        trace, config, {"require_support_verified_release": True}, run=run
    )
    assert result["valid_for_scoring"] and result["outcomes"]["full_task"]
    support = result["placement_support"]
    assert support["required"] and support["verified_placement"]
    assert support["confirmation_time_s"] == result["event_times_s"]["full_task"]
    assert support["confirmation_time_s"] >= 6.5
    json.dumps(result, allow_nan=False)


def test_legacy_metric_preserved_but_strict_missing_support_is_invalid(
    successful_trial,
):
    trace, config, run = successful_trial
    legacy = evaluate_policy_trace(trace, config, run=run)
    assert legacy["outcomes"]["full_task"] and legacy["valid_for_scoring"]
    assert legacy["placement_support"]["verified_placement"] is None
    strict = evaluate_policy_trace(
        trace, config, {"require_support_verified_release": True}, run=run
    )
    assert not strict["valid_for_scoring"]
    assert not any(strict["outcomes"].values())
    assert (
        "missing_misaligned_or_nonfinite:packet_robot_normal_force"
        in strict["invalid_reasons"]
    )


@pytest.mark.parametrize(
    "failure",
    ["robot_support", "no_bin_support", "fleeting_support", "interrupted_settle"],
)
def test_strict_support_prevents_false_release_positives(successful_trial, failure):
    trace, config, run = successful_trial
    add_support_telemetry(trace)
    if failure == "robot_support":
        # Pads unloaded, but another robot body still supports the packet.
        trace["packet_robot_normal_force"][trace["t"] >= 5] = 0.2
    elif failure == "no_bin_support":
        trace["packet_bin_normal_force"][:] = 0
    elif failure == "fleeting_support":
        trace["packet_bin_normal_force"][:] = 0
        trace["packet_bin_normal_force"][130] = 0.35
    else:
        trace["packet_bin_normal_force"][::4] = 0
    strict = evaluate_policy_trace(
        trace, config, {"require_support_verified_release": True}, run=run
    )
    assert strict["valid_for_scoring"] and strict["outcomes"]["carried"]
    assert not strict["outcomes"]["released_in_bin"]
    assert not strict["outcomes"]["full_task"]
    assert strict["placement_support"]["verified_placement"] is False
    legacy = evaluate_policy_trace(trace, config, run=run)
    assert legacy["outcomes"]["full_task"]  # Historical gate is unchanged.


@pytest.mark.parametrize("mutation", ["negative", "nan", "shape", "missing_bin"])
def test_strict_support_invalid_data_cannot_earn_success(successful_trial, mutation):
    trace, config, run = successful_trial
    add_support_telemetry(trace)
    if mutation == "negative":
        trace["packet_robot_normal_force"][0] = -1
    elif mutation == "nan":
        trace["packet_bin_normal_force"][0] = np.nan
    elif mutation == "shape":
        trace["packet_robot_normal_force"] = np.zeros((len(trace["t"]), 2))
    else:
        del trace["packet_bin_normal_force"]
    result = evaluate_policy_trace(
        trace, config, {"require_support_verified_release": True}, run=run
    )
    assert not result["valid_for_scoring"]
    assert not any(result["outcomes"].values())


def test_support_thresholds_and_flag_type_are_explicit(successful_trial):
    trace, config, run = successful_trial
    add_support_telemetry(trace)
    trace["packet_robot_normal_force"][trace["t"] >= 5] = 0.1
    trace["packet_bin_normal_force"][trace["t"] >= 6] = 0.1
    thresholds = {"require_support_verified_release": True}
    assert not evaluate_policy_trace(trace, config, thresholds, run=run)["outcomes"][
        "full_task"
    ]
    trace["packet_bin_normal_force"][trace["t"] >= 6] = 0.100001
    assert evaluate_policy_trace(trace, config, thresholds, run=run)["outcomes"][
        "full_task"
    ]
    with pytest.raises(TypeError, match="must be boolean"):
        evaluate_policy_trace(
            trace, config, {"require_support_verified_release": "false"}, run=run
        )
