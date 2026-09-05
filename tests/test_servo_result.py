"""Issue #7 (review 09-05): the executor anchors on what the arm RECEIVED.
Real URArm + fake RTDE responses; real ChunkExecutor + a stub arm."""
from __future__ import annotations

import math
import threading
import time
import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.drivers.base import ServoResult
from phantom.drivers.real.ur import URArm
from tests.test_ur_servo_limiter import (A2, A3, D1, FakeCtrl, FakeRecv, LIMITS,
                                         make_arm, planar_ik, pose_at_reach, prime)


class _Ctrl(FakeCtrl):
    def __init__(self):
        super().__init__()
        self.mode = "ok"          # ok | empty | nan | short

    def getInverseKinematics(self, pose, qnear=None):
        if self.mode == "empty":
            return []
        if self.mode == "nan":
            return [float("nan")] * 6
        if self.mode == "short":
            return [0.1, 0.2]
        return super().getInverseKinematics(pose, qnear)


def _arm(limits=None):
    a = make_arm(limits)
    a._ctrl = _Ctrl()
    return a


def test_sent_result_carries_the_streamed_pose():
    a = _arm(None)
    p0 = pose_at_reach(0.40)
    r = a.servo_l(p0, 0.008, 0.1, 300)
    assert r.sent and r.reason == "sent" and np.allclose(r.pose, p0)
    assert len(a._ctrl.streamed) == 1


@pytest.mark.parametrize("mode", ["empty", "nan", "short"])
def test_invalid_ik_never_reaches_servoj_and_reports_a_hold(mode):
    a = _arm(None)
    prime(a, pose_at_reach(0.40))
    a._ctrl.mode = mode
    r = a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)
    assert r.sent is False and r.reason == "ik_invalid"
    assert len(a._ctrl.streamed) == 1, "the invalid solve must not be streamed"
    assert np.allclose(r.pose, pose_at_reach(0.40)), "held pose = last streamed"


def test_branch_reject_reports_a_hold_and_bounded_holds_raise():
    a = _arm(None)
    prime(a, pose_at_reach(0.40))
    a._ctrl.flip = True
    r = a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)
    assert r.sent is False and r.reason == "ik_branch"
    with pytest.raises(RuntimeError):
        for _ in range(URArm.IK_REJECT_LIMIT):
            a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)


def r_of(elbow: float) -> float:
    """Shoulder->TCP distance of the FAKE planar arm at a given elbow angle
    (no d4 offset: emin=0.40 rad is r=0.4479 m, full extension 0.4569 m)."""
    return math.sqrt(A2 * A2 + A3 * A3 + 2 * A2 * A3 * math.cos(elbow))


def test_limiter_shortened_step_is_reported_as_the_streamed_pose():
    a = _arm(LIMITS)
    prime(a, pose_at_reach(r_of(0.43)))            # just inside the elbow limit
    far = pose_at_reach(r_of(0.36))                # ~3 mm step across it
    r = a.servo_l(far, 0.008, 0.1, 300)
    assert r.sent, r
    assert np.linalg.norm(r.pose[:3] - far[:3]) > 1e-4, "must be shortened"
    assert np.allclose(a._ctrl.streamed[-1], planar_ik(r.pose)), \
        "the reported pose is exactly what was streamed"
    assert abs(a._ctrl.streamed[-1][2]) >= a.hw.safety.elbow_min_rad - 1e-9


def test_infeasible_anchor_escapes_instead_of_deadlocking():
    """Review item 2: parked BELOW elbow_min_rad (the 0.468 m stop allows
    it), rate-limited retract commands must move the arm out, not hold
    forever until the 25-reject crash."""
    a = _arm(LIMITS)
    emin = a.hw.safety.elbow_min_rad
    start = pose_at_reach(r_of(emin - 0.10))       # elbow 0.30 rad < 0.40
    prime(a, start)
    assert abs(planar_ik(start)[2]) < emin
    sent = 0
    for _ in range(50):
        # the executor rate-limits from the STREAMED anchor, so each tick's
        # target is 1.5 mm inward of what the arm last received
        cur = a._last_cmd_pose if a._last_cmd_pose is not None else start
        r_cur = float(np.linalg.norm(cur[:3] - np.array([0.0, 0.0, D1])))
        res = a.servo_l(pose_at_reach(r_cur - 0.0015), 0.008, 0.1, 300)
        sent += int(res.sent)
    assert sent > 0, "no escape from the infeasible anchor"
    assert abs(a._ctrl.streamed[-1][2]) > (emin - 0.10) + 0.02, "elbow must have moved out"
    assert a._ik_rejects < URArm.IK_REJECT_LIMIT


def test_boundary_slide_makes_tangential_progress():
    """Review item 4: a sustained request pushing OUT and SIDEWAYS at the
    boundary must move the TCP along the reach sphere, not stall."""
    a = _arm(LIMITS)
    emin = a.hw.safety.elbow_min_rad
    start = pose_at_reach(r_of(emin + 0.005))
    prime(a, start)
    for _ in range(20):
        cur = a._last_cmd_pose.copy()
        tgt = cur.copy()
        tgt[0] -= 0.002                           # further out (x<0 half)
        tgt[1] += 0.001                           # and sideways
        a.servo_l(tgt, 0.008, 0.1, 300)
    lateral = abs(a._last_cmd_pose[1] - start[1])
    assert lateral > 2e-3, f"only {lateral*1e3:.3f} mm of tangential progress in 20 ticks"
    assert abs(a._ctrl.streamed[-1][2]) >= emin - 1e-9, "never streamed past the elbow limit"


# ---------------------------------------------------------------- executor
class _StubArm:
    """servo_l holds on request; get_state gives a fixed pose."""
    def __init__(self, hold_ticks):
        self.hold_ticks = hold_ticks
        self.calls = 0
        self.streamed = []

    def servo_l(self, pose, dt, lookahead, gain):
        self.calls += 1
        if self.calls in self.hold_ticks:
            return ServoResult(False, self.streamed[-1] if self.streamed else None, "ik_branch")
        self.streamed.append(np.asarray(pose, float).copy())
        return ServoResult(True, pose, "sent")

    def get_state(self):
        from phantom.drivers.base import ArmState
        p = self.streamed[-1] if self.streamed else np.zeros(6)
        return ArmState(t_host=0.0, seq=0, t_rtde=0.0, q=np.zeros(6), qd=np.full(6, 0.3),
                        tcp_pose=p, tcp_speed=np.zeros(6), ft=np.zeros(6),
                        protective_stop=False, robot_mode=7)

    def stop(self, *a, **k):
        pass

    def servo_stop(self):
        pass


class _PassSafety:
    def __init__(self):
        from phantom.deploy.safety import SafetyAction, SafetyVerdict
        self._ok = SafetyVerdict(action=SafetyAction.OK)
        self.log_events = []

    def check(self, t, target):
        return self._ok

    def clamp_target(self, target):
        return target


def _plan(t0_pose, H=10, step=0.001):
    from phantom.inference.policy import Plan
    acts = np.zeros((H, 7))
    acts[:, 0] = step                      # +1 mm in x per action step
    return Plan(t_created=time.time(), t0_pose=np.asarray(t0_pose, float),
                actions=acts, action_times=time.time() + np.arange(H) / 10.0,
                sigma=np.zeros(3), gate=0.5, p_evt=np.zeros(5), cpk=None,
                latency_s=0.01)


def test_executor_run_anchors_on_the_streamed_pose():
    """Issue #7 through the REAL ChunkExecutor._run loop with a driver that
    holds some ticks: while the driver holds, the executor's anchor
    (`last_cmd`) must stay at the last STREAMED pose, and at the end it
    must equal the last pose the arm received."""
    from phantom.deploy.executor import ChunkExecutor
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    hw = hw.model_copy(update={"safety": hw.safety.model_copy(
        update={"stale_plan_timeout_s": 10.0})})
    holds = set(range(4, 9))                                 # ticks 4-8 held
    arm = _StubArm(hold_ticks=holds)
    ex = ChunkExecutor(hw, arm=arm, gripper=None, safety=_PassSafety())
    seen_during_hold = []

    real_servo = arm.servo_l
    def servo_spy(pose, dt, lookahead, gain):
        res = real_servo(pose, dt, lookahead, gain)
        if not res.sent:
            seen_during_hold.append(ex.last_cmd().copy())
        return res
    arm.servo_l = servo_spy

    p0 = np.zeros(6)
    with ex._lock:
        ex._plan = _plan(p0)
        ex._swap_t = time.perf_counter()
        ex._play_time = 0.0
    th = threading.Thread(target=ex._run, daemon=True)
    th.start()
    time.sleep(0.25)
    ex._stop.set()
    th.join(1.0)

    assert arm.calls >= 9 and len(arm.streamed) >= 4
    last_streamed_before_hold = arm.streamed[2]              # tick 3 = 3rd streamed
    for anchor in seen_during_hold:
        assert np.allclose(anchor, last_streamed_before_hold), \
            "anchor advanced while the driver was holding"
    assert np.allclose(ex.last_cmd(), arm.streamed[-1]), \
        "final anchor must be the last pose the arm received"
    # and the proposals kept advancing (the plan is still being played)
    assert arm.streamed[-1][0] > last_streamed_before_hold[0]


def test_halt_snapshots_arm_state_before_teardown():
    from phantom.deploy.executor import ChunkExecutor
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    arm = _StubArm(hold_ticks=set())
    arm.servo_l(np.array([0.1, 0.2, 0.3, 0, 0, 0]), 0.008, 0.1, 300)
    ex = ChunkExecutor(hw, arm=arm, gripper=None, safety=None)
    ex._halt("wrist_extension")
    assert ex.halt_state["reason"] == "wrist_extension"
    assert ex.halt_state["qd_max"] == pytest.approx(0.3)
    assert ex.halt_state["tcp_pose"][:3] == pytest.approx([0.1, 0.2, 0.3])


def test_sustained_limiter_hold_ends_the_episode_not_the_budget():
    """Verify 09-05 #3: with the limiter ON, an unreachable target for 25
    ticks must raise (executor crash net), not freeze silently."""
    a = _arm(LIMITS)
    prime(a, pose_at_reach(0.40))
    a._ctrl.mode = "empty"
    with pytest.raises(RuntimeError):
        for _ in range(URArm.IK_REJECT_LIMIT + 1):
            a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)
    assert a._ik_rejects_total >= URArm.IK_REJECT_LIMIT


def test_elbow_escape_never_streams_a_joint_whip():
    """Verify 09-05 #2: the infeasible-anchor escape still obeys the
    per-tick joint-speed rule."""
    a = _arm(LIMITS)
    emin = a.hw.safety.elbow_min_rad
    vmax = a.hw.safety.servo_joint_speed_max_rad_s
    start = pose_at_reach(r_of(emin - 0.10))
    prime(a, start)
    r = a.servo_l(pose_at_reach(r_of(emin + 0.20)), 0.008, 0.1, 300)   # 0.3 rad in one tick
    if r.sent:
        q_prev = planar_ik(start)
        dq = max(abs(x - y) for x, y in zip(a._ctrl.streamed[-1], q_prev))
        assert dq / 0.008 <= vmax + 1e-9, f"streamed {dq/0.008:.2f} rad/s > {vmax}"
