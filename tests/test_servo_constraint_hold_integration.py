"""Real driver/executor/adapter code with fake RTDE and a controlled CPU clock."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_sim_policy_adapter import adapter, observe, plan
from test_ur_servo_limiter import FakeCtrl, make_arm

from phantom.deploy.executor import ChunkExecutor
from phantom.drivers.real import ur as ur_module
from phantom.drivers.servo_hold import ConstraintHoldBudget, ServoHoldTimeout
from phantom.drivers.servo_limiter import ServoLimits, select_servo_step
from phantom.sim.kinematics import forward_pose
from tools.sim.run_waffles import report_servo_limiter_execution

Q = np.array([0.1, -1.5, 0.4, 0.5, 1.2, -3.0])
POSE = forward_pose(Q)
LIMITS = ServoLimits(elbow_min_rad=0.4, joint_speed_max_rad_s=1.0)


class ConstraintCtrl(FakeCtrl):
    """Finite in-branch IK answers that cannot satisfy the elbow constraint."""

    def __init__(self):
        super().__init__()
        self.mode = "hold"
        self.send_ok = True
        self.queries = 0

    def getInverseKinematics(self, pose, qnear=None):
        self.queries += 1
        q = np.asarray(qnear, float).copy()
        if self.mode == "empty":
            return []
        if self.mode == "branch":
            q[4] += 1.0
        elif self.mode == "hold":
            q[2] -= 0.02
        elif self.mode != "stationary":
            raise AssertionError(self.mode)
        return q.tolist()

    def servoJ(self, q, *args):
        if not self.send_ok:
            return False
        return super().servoJ(q, *args)


def native_arm(monkeypatch, timeout=0.5):
    clock = [0.0]
    monkeypatch.setattr(
        ur_module, "time", SimpleNamespace(perf_counter=lambda: clock[0])
    )
    arm = make_arm(
        {
            "elbow_min_rad": 0.4,
            "servo_joint_speed_max_rad_s": 1.0,
            "servo_constraint_hold_s": timeout,
        }
    )
    arm._ctrl = ConstraintCtrl()
    arm._last_qsol = Q.tolist()
    arm._last_cmd_pose = POSE.copy()
    return arm, clock


def target():
    p = POSE.copy()
    p[0] += 0.002
    return p


def test_native_verified_hold_streams_past_25_ticks_then_typed_timeout(monkeypatch):
    arm, clock = native_arm(monkeypatch)
    for tick in range(50):
        clock[0] = tick * 0.008
        result = arm.servo_l(target(), 0.008, 0.1, 300)
        assert result.sent and result.reason == "constraint_hold"
        np.testing.assert_array_equal(result.pose, POSE)
        np.testing.assert_array_equal(arm._ctrl.streamed[-1], Q)
        assert arm._ik_rejects == arm._ik_rejects_total == 0
        assert arm.constraint_hold_last["started_at_s"] == 0
        assert arm.constraint_hold_last["all_ik_on_branch"]
        assert arm.constraint_hold_last["all_ik_valid"]
    clock[0] = 0.5
    with pytest.raises(ServoHoldTimeout) as error:
        arm.servo_l(target(), 0.008, 0.1, 300)
    assert error.value.reason == "servo_constraint_hold_timeout"
    assert len(arm._ctrl.streamed) == 50  # Timeout did not claim a new send.
    assert arm.constraint_hold_last["timed_out"]


def test_native_accepted_stationary_commands_do_not_clear_hold_deadline(monkeypatch):
    arm, clock = native_arm(monkeypatch)
    arm.servo_l(target(), 0.008, 0.1, 300)
    arm._ctrl.mode = "stationary"
    nearby = POSE.copy()
    nearby[0] += 1e-6  # A tolerance-level IK acceptance with identical joints.
    for tick in range(1, 50):
        clock[0] = tick * 0.008
        result = arm.servo_l(nearby, 0.008, 0.1, 300)
        assert result.sent and result.reason == "sent"
        np.testing.assert_array_equal(result.pose, POSE)
        assert arm.constraint_hold_last["active"]
        assert arm.constraint_hold_last["started_at_s"] == 0
    clock[0] = 0.5
    with pytest.raises(ServoHoldTimeout):
        arm.servo_l(POSE, 0.008, 0.1, 300)


def test_native_ik_fault_streak_survives_verified_hold_streaming(monkeypatch):
    arm, clock = native_arm(monkeypatch, timeout=5)
    for fault in range(1, 25):
        arm._ctrl.mode = "empty" if fault % 2 else "branch"
        clock[0] += 0.008
        rejected = arm.servo_l(target(), 0.008, 0.1, 300)
        assert not rejected.sent
        assert arm._ik_rejects == fault
        arm._ctrl.mode = "hold"
        clock[0] += 0.008
        held = arm.servo_l(target(), 0.008, 0.1, 300)
        assert held.sent and held.reason == "constraint_hold"
        assert arm._ik_rejects == fault
    arm._ctrl.mode = "branch"
    clock[0] += 0.008
    with pytest.raises(RuntimeError, match="25 consecutive") as error:
        arm.servo_l(target(), 0.008, 0.1, 300)
    assert not isinstance(error.value, ServoHoldTimeout)


def test_native_servoj_false_remains_a_fault_and_never_updates_anchor(monkeypatch):
    arm, _ = native_arm(monkeypatch)
    arm._ctrl.send_ok = False
    with pytest.raises(RuntimeError, match="servoJ rejected") as error:
        arm.servo_l(target(), 0.008, 0.1, 300)
    assert not isinstance(error.value, ServoHoldTimeout)
    assert not arm._ctrl.streamed
    np.testing.assert_array_equal(arm._last_qsol, Q)
    np.testing.assert_array_equal(arm._last_cmd_pose, POSE)


def held_selection():
    ctrl = ConstraintCtrl()
    result = select_servo_step(
        target(), POSE, Q, 0.008, ctrl.getInverseKinematics, LIMITS
    )
    assert result.reason == "limiter_hold" and not result.accepted
    assert result.all_ik_valid and result.all_ik_on_branch
    return result


def sim_tick(ad, t, selection, budget, rejects=0, qd=None):
    observe(ad, t, pose=POSE, q=Q, qd=qd, grip=0.63)
    command = ad.step(t)
    telemetry = {}
    achieved, rejects = report_servo_limiter_execution(
        ad,
        t,
        selection,
        command.gripper,
        rejects,
        hold_budget=budget,
        qref=Q,
        limits=LIMITS,
        dt=0.008,
        telemetry=telemetry,
    )
    return command, achieved, rejects, telemetry


class DelayedPlan:
    def replan(self, obs, prev, tcp):
        return plan(obs.t, tcp=tcp, latency=0.2, delta=0.002, grip=0.63)


def test_sim_hold_truthful_fk_waits_for_fresh_plan_and_times_out_preserving_grip():
    ad, budget, selection = adapter(), ConstraintHoldBudget(0.5), held_selection()
    ad.policy = DelayedPlan()
    for tick in range(50):
        t = tick * 0.008
        command, achieved, rejects, telemetry = sim_tick(ad, t, selection, budget)
        if tick == 1:
            ad.replan(t=t)
        assert not command.stopped and rejects == 0
        np.testing.assert_array_equal(achieved, forward_pose(Q))
        np.testing.assert_array_equal(ad._last_cmd, achieved)
        assert ad.ik_rejects == ad.ik_rejects_total == 0
        assert telemetry["started_at_s"] == 0
        if tick == 26:
            assert command.diagnostics["plan_activated"]  # 0.208s, past 25 ticks.
            assert ad._plan is not None and ad.stopped_reason is None
    _, achieved, rejects, telemetry = sim_tick(ad, 0.5, selection, budget)
    assert achieved is None and rejects == 0 and telemetry["timed_out"]
    assert ad.stopped_reason == "servo_constraint_hold_timeout"
    assert ad.ik_rejects == ad.ik_rejects_total == 0
    assert ad._awaiting_feedback is None
    observe(ad, 0.508, pose=POSE, q=Q, grip=0.63)
    stopped = ad.step(0.508)
    assert stopped.stopped and stopped.reason == "servo_constraint_hold_timeout"
    assert stopped.gripper == pytest.approx(0.63)
    np.testing.assert_array_equal(stopped.tcp_pose, POSE)
    ad.report_execution(0.508, accepted=True, tcp_pose=POSE, gripper_command=0.63)


def test_sim_verified_holds_do_not_erase_prior_real_ik_rejections():
    ad, budget = adapter(), ConstraintHoldBudget(1)
    bad = select_servo_step(target(), POSE, Q, 0.008, lambda *_: [], LIMITS)
    _, achieved, rejects, _ = sim_tick(ad, 0.0, bad, budget)
    assert achieved is None and rejects == ad.ik_rejects == 1
    for tick in range(1, 35):
        _, achieved, rejects, _ = sim_tick(
            ad, tick * 0.008, held_selection(), budget, rejects
        )
        assert achieved is not None and rejects == ad.ik_rejects == 1
    assert ad.ik_rejects_total == 1 and ad.stopped_reason is None


def test_sim_measured_safety_preempts_a_live_constraint_hold():
    ad, budget, selection = adapter(), ConstraintHoldBudget(1), held_selection()
    sim_tick(ad, 0.0, selection, budget)
    observe(ad, 0.008, pose=POSE, q=Q, qd=np.ones(6) * 3, grip=0.63)
    stopped = ad.step(0.008)
    assert stopped.stopped and stopped.reason == "safety_stop"
    assert "joint_speed" in stopped.diagnostics["safety_events"]
    # Runner sees command.stopped before reporting limiter selection or sends.
    ad.report_execution(
        0.008, accepted=True, tcp_pose=POSE, gripper_command=stopped.gripper
    )
    assert ad._pending is None and ad.stopped_reason == "safety_stop"


def test_intermediate_ik_branch_fault_cannot_enter_the_verified_hold_path():
    calls = 0

    def solve(_pose, qref):
        nonlocal calls
        calls += 1
        q = np.asarray(qref).copy()
        q[2] -= 0.02
        if calls > 1:
            q[4] += 1
        return q

    result = select_servo_step(target(), POSE, Q, 0.008, solve, LIMITS)
    assert result.reason == "limiter_hold" and result.all_ik_valid
    assert not result.all_ik_on_branch
    ad = adapter()
    _, achieved, rejects, telemetry = sim_tick(ad, 0.0, result, ConstraintHoldBudget(1))
    assert achieved is None and rejects == ad.ik_rejects == 1
    assert not telemetry


def test_native_executor_typed_timeout_is_ordinary_stop_and_does_not_open_grip():
    ad = adapter()
    moves = []
    ex = ChunkExecutor(
        ad.hw,
        SimpleNamespace(),
        SimpleNamespace(move=lambda *args: moves.append(args)),
        ad.safety,
    )
    ex._grip_latch = ex._grip_target = 0.63

    def timeout():
        raise ServoHoldTimeout()

    ex._run = timeout
    ex._run_guarded()
    assert ex.stopped_reason == "servo_constraint_hold_timeout"
    assert ex.crash_text is None and ex._stop.is_set()
    assert ex._grip_target is None and not moves
    assert ex.halt_state["reason"] == "servo_constraint_hold_timeout"
