"""Explicit native controller profile and independent gripper playback parity."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from phantom_test_utils import make_small_hw
from test_native_minimal_v5 import RELEASE, VETO, native_planner, state
from test_sim_policy_adapter import POSE, Policy, observe, plan, tick
from test_sim_terminal_veto import proposal

from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.minimal_v5 import planner_class
from phantom.deploy.release_controller import restore_policy_openings
from phantom.sim.policy_adapter import SimulationPolicyAdapter
from tools.sim.deployment_filters import TerminalVetoFilter


def make_adapter(*, profile="minimal_v5", pose_cap=10, grip_cap=10):
    hw = make_small_hw(safety={"wrist_extension_stop_m": None, "reach_clamp_m": None})
    veto = TerminalVetoFilter(
        hw, VETO, implementation="fd4a032", controller_profile=profile
    )
    return SimulationPolicyAdapter(
        hw, Policy(), mode="teacher", max_play_steps=pose_cap,
        grip_play_steps=grip_cap, plan_filter=veto, release_config=RELEASE,
        controller_profile=profile,
    )


def request(t=0.0, *, grip=0.2, z=0.2):
    ur = np.zeros(26)
    ur[12:18] = POSE
    ur[14] = z
    ur[-2] = grip
    return SimpleNamespace(t=t, ur_state=ur)


def test_explicit_profile_uses_request_feedback_when_delivery_pose_changed():
    historical = make_adapter()
    default = make_adapter(profile=None)
    captured = request(z=0.2, grip=0.2)
    low = POSE.copy()
    low[2] = 0.08
    for ad in (historical, default):
        observe(ad, 0.1, pose=low, grip=0.3)
    old = historical.plan_filter(proposal(), captured, historical)
    live = default.plan_filter(proposal(), captured, default)
    assert old.diag["terminal_veto"]["action"] == "close_masked"
    assert old.diag["terminal_veto"]["feedback_source"] == "request_snapshot_historical"
    assert old.diag["terminal_veto"]["feedback_gripper"] == 0.2
    assert old.diag["terminal_veto"]["controller_profile"] == "minimal_v5"
    assert live.diag["terminal_veto"]["action"] == "close_allowed"
    assert live.diag["terminal_veto"]["feedback_source"] == "current_delivery"
    assert default.controller_profile_metadata is None
    assert historical.controller_profile_metadata["veto_implementation"] == "fd4a032"


@pytest.mark.parametrize("executed_close", [False, True])
def test_profile_matches_native_decisions_cpk_and_executed_close_history(executed_close):
    ad = make_adapter()
    native_type = planner_class("minimal_v5", native_planner())
    native = native_type.__new__(native_type)
    native.veto = SimpleNamespace(**VETO)
    native.executor = ad
    native_state = state()
    cases = [(0.0, 0.1, 0.2, 0.2), (0.1, 0.99, 0.6, 0.2), (0.2, 0.99, 0.6, 0.08)]
    for n, (t, p_none, grip, z) in enumerate(cases):
        if n == 1 and executed_close:
            ad._grip_hist.append((0.05, 0.6))
        observe(ad, t, grip=grip)
        snap = request(t, grip=grip, z=z)
        raw = proposal(p_none=p_none)
        native_plan = deepcopy(raw)
        native_record = native._apply_veto(native_plan, snap.ur_state[12:18], grip, native_state, n)
        restore_policy_openings(native_plan, raw.actions, native_record, ad)
        actual = ad.plan_filter(raw, snap, ad)
        np.testing.assert_array_equal(actual.actions, native_plan.actions)
        assert actual.diag["terminal_veto"]["action"] == native_record["action"]
        assert actual.cpk == native_plan.cpk
        assert actual._cpk_token == native_plan._cpk_token
        assert ad.plan_filter.state == native_state
        if n == 1:
            assert (native_record["action"] == "recovery_open") == executed_close


@pytest.mark.parametrize("pose_cap,grip_cap", [(16, 10), (10, 16), (10, 10), (None, 10), (10, None), (None, None)])
def test_separate_playback_caps_match_native_pose_and_gripper(pose_cap, grip_cap):
    ad = make_adapter(pose_cap=pose_cap, grip_cap=grip_cap)
    native = ChunkExecutor.__new__(ChunkExecutor)
    native.hw = ad.hw
    native.max_play_steps, native.grip_play_steps = pose_cap, grip_cap
    proposed = plan()
    proposed.actions[:, 6] = np.linspace(0.1, 0.9, 16)
    for t in (0, 0.51, 0.99, 1.0, 1.31, 2.0):
        target, grip = ad._pose_at(proposed, t)
        native_target, native_grip = native._pose_at(proposed, t)
        np.testing.assert_array_equal(target, native_target)
        assert grip == native_grip


def test_capped_gripper_history_and_release_eligibility_ignore_unplayed_tail():
    hw = make_small_hw(safety={"wrist_extension_stop_m": None, "reach_clamp_m": None})
    ad = SimulationPolicyAdapter(hw, Policy(), max_play_steps=16, grip_play_steps=10)
    observe(ad)
    proposed = plan(grip=0.6, delta=0.0001)
    proposed.actions[10:, 6] = 0.2
    assert ad.submit(proposed, 0)
    for i in range(34):
        t = i * 0.04
        observe(ad, t)
        command = tick(ad, t)
        assert command.gripper == pytest.approx(0.6)
    assert all(grip == pytest.approx(0.6) for _, grip in ad.entered_grip_after(-1))
    np.testing.assert_array_equal(ad.gripper_cmd_at([0.1, 0.99, 1.31]), [0.6] * 3)
    ad._plan.diag = {"terminal_veto": {
        "action": "close_masked", "placement_release_passthrough_indices": [10],
    }}
    assert not ad._original_policy_grip()
    ad._plan.diag["terminal_veto"]["placement_release_passthrough_indices"] = [9]
    assert ad._original_policy_grip()


def test_profile_rejects_missing_release_and_conflicting_veto():
    hw = make_small_hw()
    with pytest.raises(ValueError, match="explicit fd4a032"):
        TerminalVetoFilter(hw, VETO, controller_profile="minimal_v5")
    veto = TerminalVetoFilter(hw, VETO, implementation="fd4a032", controller_profile="minimal_v5")
    with pytest.raises(ValueError, match="explicit release bounds"):
        SimulationPolicyAdapter(hw, Policy(), mode="teacher", plan_filter=veto, controller_profile="minimal_v5")
    with pytest.raises(ValueError, match="profiles must agree"):
        SimulationPolicyAdapter(hw, Policy(), mode="teacher", plan_filter=veto, release_config=RELEASE)
