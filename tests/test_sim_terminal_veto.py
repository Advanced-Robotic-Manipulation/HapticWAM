"""Native veto rules, version differences, and delayed simulator delivery."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from phantom_test_utils import make_small_hw

from phantom.sim.policy_adapter import SimulationPolicyAdapter
from tools.sim.deployment_filters import TerminalVetoFilter


class Ring:
    def __init__(self, fz=0.0):
        self.values = np.zeros((1, 6))
        self.values[:, 2] = fz

    def latest(self, n):
        values = self.values[-n:]
        return np.arange(len(values), dtype=float) / 8, {"wrench": values}


class Feedback:
    def __init__(self):
        self._last_observe_t = 0.0
        self.rings = {"tactile_left": Ring(), "tactile_right": Ring()}
        self.history = []
        self.latch_clears = 0
        self.stopped_reason = None

    def entered_grip_after(self, t):
        return [(s, g) for s, g in self.history if s > t]

    def clear_grip_latch(self):
        self.latch_clears += 1

    def request_stop(self, reason):
        self.stopped_reason = reason


def hardware():
    return SimpleNamespace(
        gripper=SimpleNamespace(max_close_cmd=0.8),
        arm=SimpleNamespace(dof=6),
        ur_state_dim=26,
        tactile=SimpleNamespace(
            rate_hz=8,
            sensors=[SimpleNamespace(name="left"), SimpleNamespace(name="right")],
        ),
    )


def snapshot(t=0.0, grip=0.2, z=0.2):
    ur = np.zeros(26)
    ur[14] = z
    ur[-2] = grip
    return SimpleNamespace(t=t, ur_state=ur)


def proposal(*, grip=0.65, p_none=0.99):
    actions = np.zeros((4, 7))
    actions[:, 2] = [0.02, -0.05, 0.01, 0.05]
    actions[:, 6] = grip
    return SimpleNamespace(
        actions=actions,
        p_evt=np.array([p_none, 1 - p_none, 0.0, 0.0, 0.0]),
        cpk="local-contact-package",
        _cpk_token="remote-contact-package",
        diag={"original_diagnostic": {"picked": 2}},
    )


def apply(veto, feedback, *, t, grip=0.2, z=0.2, p_none=0.99, command=0.65):
    feedback._last_observe_t = t
    return veto(proposal(grip=command, p_none=p_none), snapshot(t, grip, z), feedback)


def action(result):
    return result.diag["terminal_veto"]["action"]


@pytest.mark.parametrize("implementation", ["live", "fd4a032"])
def test_mask_holds_measured_aperture_and_preserves_raw_proposal(implementation):
    veto = TerminalVetoFilter(hardware(), implementation=implementation)
    feedback = Feedback()
    raw = proposal()
    original = deepcopy(raw)
    filtered = veto(raw, snapshot(grip=0.29), feedback)
    assert action(filtered) == "close_masked"
    np.testing.assert_array_equal(filtered.actions[:, 6], 0.29)
    np.testing.assert_array_equal(filtered.actions[:, :6], original.actions[:, :6])
    np.testing.assert_array_equal(raw.actions, original.actions)
    assert raw.diag == original.diag
    assert raw._cpk_token == original._cpk_token
    assert filtered.cpk is None and filtered._cpk_token is None
    assert filtered.diag["terminal_veto"]["cpk_invalidated"]
    assert filtered.diag["terminal_veto"]["implementation"] == implementation


def test_proposed_tail_close_does_not_arm_recovery_without_executed_transition():
    veto, feedback = TerminalVetoFilter(hardware()), Feedback()
    assert action(apply(veto, feedback, t=0, p_none=0.1)) == "close_allowed"
    feedback.history = [(0.05, 0.2), (0.1, 0.3)]
    assert action(apply(veto, feedback, t=0.2)) == "close_masked"
    assert veto.state["retries"] == 0
    assert feedback.latch_clears == 0


@pytest.mark.parametrize("implementation", ["live", "fd4a032"])
def test_executed_close_recovery_and_versioned_latch_behavior(implementation):
    veto = TerminalVetoFilter(
        hardware(), {"open_aperture": 0.232}, implementation=implementation
    )
    feedback = Feedback()
    apply(veto, feedback, t=0, p_none=0.1)
    feedback.history = [(0.1, 0.6)]
    result = apply(veto, feedback, t=0.2, grip=0.6)
    assert action(result) == "recovery_open"
    np.testing.assert_array_equal(result.actions[:, 6], 0.232)
    assert np.max(np.cumsum(result.actions[:, 2])) <= 0.0
    assert result._cpk_token is None
    assert feedback.latch_clears == (1 if implementation == "live" else 0)
    assert veto.state["retries"] == 1


def test_late_p_none_does_not_rearm_an_ongoing_hold():
    veto, feedback = TerminalVetoFilter(hardware()), Feedback()
    apply(veto, feedback, t=0, p_none=0.1)
    feedback.history = [(0.1, 0.6)]
    apply(veto, feedback, t=0.2, grip=0.6, p_none=0.1)
    feedback.history += [(0.3, 0.6)]
    apply(veto, feedback, t=0.4, grip=0.6, p_none=0.1)
    feedback.history += [(0.5, 0.6)]
    result = apply(veto, feedback, t=0.6, grip=0.6)
    assert action(result) == "close_masked"
    assert veto.state["closed_idx"] is None
    assert veto.state["retries"] == 0


@pytest.mark.parametrize("implementation", ["live", "fd4a032"])
def test_floor_only_close_suppresses_model_only_recovery(implementation):
    veto = TerminalVetoFilter(
        hardware(), {"z_ref": 0.0415, "z_margin": 0.0615}, implementation=implementation
    )
    feedback = Feedback()
    result = apply(veto, feedback, t=0, z=0.103)
    assert action(result) == "close_allowed"
    assert result.diag["terminal_veto"]["at_floor"]
    feedback.history = [(0.1, 0.6)]
    result = apply(veto, feedback, t=0.2, z=0.15, grip=0.6)
    assert action(result) == "recovery_skipped_floor_close"
    assert result._cpk_token is not None


@pytest.mark.parametrize(
    ("implementation", "expected"),
    [("live", "recovery_skipped_loaded"), ("fd4a032", "recovery_open")],
)
def test_loaded_pad_exclusion_exists_only_in_current_native_version(
    implementation, expected
):
    veto = TerminalVetoFilter(hardware(), implementation=implementation)
    feedback = Feedback()
    for ring in feedback.rings.values():
        ring.values[:, 2] = 1.5
    apply(veto, feedback, t=0, p_none=0.1)
    feedback.history = [(0.1, 0.6)]
    for ring in feedback.rings.values():
        ring.values[:, 2] = 5.0
    result = apply(veto, feedback, t=0.2, grip=0.6)
    assert action(result) == expected
    if implementation == "live":
        assert result.diag["terminal_veto"]["pad_load"] == {"left": 3.5, "right": 3.5}


def test_tactile_recovery_waits_for_rise_and_persistence_on_delivery_clock():
    veto, feedback = TerminalVetoFilter(hardware()), Feedback()
    apply(veto, feedback, t=0, grip=0.6, p_none=0.1)
    # A long close without a rise is not a phantom grasp.
    assert action(apply(veto, feedback, t=4, grip=0.6, p_none=0.1)) == "none"
    apply(veto, feedback, t=4.1, grip=0.6, z=0.24, p_none=0.1)
    assert action(apply(veto, feedback, t=4.39, grip=0.6, z=0.24, p_none=0.1)) == "none"
    feedback._last_observe_t = 4.41
    # Request time is old; the native persistence clock is delivery time.
    result = veto(proposal(p_none=0.1), snapshot(4.2, 0.6, 0.24), feedback)
    assert action(result) == "recovery_tactile"
    assert result.diag["terminal_veto"]["snapshot_t_s"] == 4.2
    assert result.diag["terminal_veto"]["applied_at_s"] == 4.41
    assert np.max(np.cumsum(result.actions[:, 2])) <= 0.0
    assert feedback.latch_clears == 1


def test_historical_recovery_requires_no_tactile_ring_and_has_no_tactile_branch():
    veto = TerminalVetoFilter(hardware(), implementation="fd4a032")
    feedback = Feedback()
    feedback.rings = {}
    apply(veto, feedback, t=0, grip=0.6, p_none=0.1)
    apply(veto, feedback, t=4, grip=0.6, z=0.24, p_none=0.1)
    assert action(apply(veto, feedback, t=5, grip=0.6, z=0.24, p_none=0.1)) == "none"
    with pytest.raises(NotImplementedError, match="each pad wrench"):
        TerminalVetoFilter(hardware())(proposal(), snapshot(5), feedback)


def test_retry_cap_stops_without_mutating_proposal_or_claiming_cpk_rewrite():
    veto = TerminalVetoFilter(hardware(), {"max_retries": 0})
    feedback = Feedback()
    apply(veto, feedback, t=0, p_none=0.1)
    feedback.history = [(0.1, 0.6)]
    result = apply(veto, feedback, t=0.2, grip=0.6)
    assert action(result) == "retry_cap"
    assert feedback.stopped_reason == "veto_retry_cap"
    assert result.diag["terminal_veto"]["stop_reason"] == "veto_retry_cap"
    assert not result.diag["terminal_veto"]["cpk_invalidated"]
    assert result._cpk_token is not None
    veto.reset()
    assert veto.state["retries"] == 0 and veto.state["g_min"] is None
    assert veto.replan_index == 0 and veto.last_record is None


def test_trailing_load_matches_native_sample_count_and_explicit_wrench_baseline():
    baseline = {name: [0, 0, 1, 0, 0, 0] for name in ("left", "right")}
    veto = TerminalVetoFilter(hardware(), wrench_base=baseline)
    feedback = Feedback()
    for ring in feedback.rings.values():
        ring.values = np.zeros((9, 6))
        ring.values[:, 2] = [100, 6, 1, 1, 1, 1, 1, 1, 1]
    result = apply(veto, feedback, t=1, p_none=0.1)
    assert action(result) == "close_allowed"
    assert veto._pad_loads(veto.state, 1.0) == {"left": 5.0, "right": 5.0}


def test_delayed_adapter_filter_gets_captured_snapshot_and_current_feedback_then_stops():
    hw = make_small_hw(safety={"wrist_extension_stop_m": None, "reach_clamp_m": None})
    tcp = np.array([-0.3, -0.12, 0.25, 0, np.pi, 0])
    native = proposal(grip=0.2, p_none=0.1)
    native.actions = np.zeros((16, 7))
    native.actions[:, 6] = 0.2
    native.t0_pose = tcp.copy()
    native.t_created = 0.0
    native.action_times = 0.2 + np.arange(16) / hw.control.action_rate_hz
    native.sigma = np.zeros(3)
    native.gate = 0.0
    native.latency_s = 0.2
    calls, deliveries = [], []

    def filter_plan(plan, captured, adapter):
        calls.append((captured.t, captured.ur_state[-2], adapter._last_observe_t))
        plan = deepcopy(plan)
        plan.actions[:, 6] = 0.35
        adapter.request_stop("veto_retry_cap")
        return plan

    ad = SimulationPolicyAdapter(
        hw,
        SimpleNamespace(replan=lambda *_: native),
        mode="student",
        plan_filter=filter_plan,
        delivered_plan_callback=lambda *args: deliveries.append(args),
    )

    def observe(t, grip):
        ad.observe(
            t,
            rgb=np.zeros((12, 16, 3), dtype=np.uint8),
            q=np.array([0, -1.4, 1.5, -1.7, 1.4, 0]),
            qd=np.zeros(6),
            tcp_pose=tcp,
            tcp_speed=np.zeros(6),
            gripper_state=np.array([grip, 0]),
            wrist_ft=np.zeros(6),
        )

    observe(0, 0.2)
    raw = ad.replan(t=0)
    assert calls == []
    observe(0.2, 0.4)
    command = ad.step(0.2)
    assert calls == [(0.0, pytest.approx(0.2), 0.2)]
    assert command.stopped and command.reason == "veto_retry_cap"
    assert ad._plan is None  # A stopped filter must not activate a plan.
    assert len(deliveries) == 1 and deliveries[0][-1] is False
    np.testing.assert_array_equal(raw.actions[:, 6], 0.2)
    np.testing.assert_array_equal(native.actions[:, 6], 0.2)
    np.testing.assert_array_equal(deliveries[0][0].actions[:, 6], 0.35)


@pytest.mark.parametrize(("load", "expected_command"), [(3.0, 0.63), (2.5, 0.55)])
def test_loaded_latch_preserves_preload_through_measured_aperture_veto(
    load, expected_command
):
    """A mask must not remove loaded drive preload once native latch has armed."""
    hw = make_small_hw(
        safety={
            "wrist_extension_stop_m": None,
            "reach_clamp_m": None,
            "grip_latch_fz_n": 2.5,
            "workspace_m": {"x": [-0.7, 0.15], "y": [-0.5, 0.3], "z": [0.03, 0.8]},
        }
    )
    tcp = np.array([-0.3, -0.12, 0.13, 0, np.pi, 0])
    calls = []

    def replan(obs, _previous, _tcp):
        first = not calls
        calls.append(obs)
        plan = proposal(grip=0.63 if first else 0.53, p_none=0.1 if first else 0.7)
        plan.actions = np.zeros((16, 7))
        plan.actions[:, 6] = 0.63 if first else 0.53
        plan.t0_pose = tcp.copy()
        plan.t_created = obs.t
        plan.action_times = obs.t + np.arange(16) / hw.control.action_rate_hz
        plan.sigma = np.zeros(3)
        plan.gate = 0.0
        plan.latency_s = 0.0
        return plan

    veto = TerminalVetoFilter(
        hw, {"z_ref": 0.0415, "z_margin": 0.0615}, implementation="fd4a032"
    )
    ad = SimulationPolicyAdapter(hw, SimpleNamespace(replan=replan), plan_filter=veto)

    def observe(t, grip, normal):
        ad.observe(
            t,
            rgb=np.zeros((12, 16, 3), np.uint8),
            q=np.array([0, -1.4, 1.5, -1.7, 1.4, 0]),
            qd=np.zeros(6),
            tcp_pose=tcp,
            tcp_speed=np.zeros(6),
            gripper_state=np.array([grip, 2 if normal else 3]),
            wrist_ft=np.zeros(6),
            tactile={
                s.name: {"wrench": np.array([0, 0, -normal, 0, 0, 0])}
                for s in hw.tactile.sensors
            },
        )

    def step(t):
        command = ad.step(t)
        assert not command.stopped
        ad.report_execution(t, accepted=True, gripper_command=command.gripper)
        return command.gripper

    observe(0, 0.2, 0)
    ad.replan(t=0)
    assert step(0) == pytest.approx(0.63)
    # Object blocks closure at.55 while the accepted drive target remains.63.
    observe(0.008, 0.55, load)
    assert step(0.008) == pytest.approx(0.63)
    ad.replan(t=0.008)
    observe(0.016, 0.55, 0)
    step(0.016)
    assert veto.last_record["action"] == "close_masked"
    assert veto.last_record["cpk_invalidated"]
    assert ad._plan.actions[0, 6] == pytest.approx(0.55)
    # Beyond the blend and contact window, the latch remains until an explicit
    # release. Equality at2.5N is deliberately insufficient to arm native >2.5.
    observe(1.0, 0.55, 0)
    assert step(1.0) == pytest.approx(expected_command)
    assert not any(ad.safety.contact_load.values())
    assert ad._grip_hist[-1][1] == pytest.approx(expected_command)
    assert ad.gripper_cmd_at(np.array([1.0]))[0] == pytest.approx(expected_command)
