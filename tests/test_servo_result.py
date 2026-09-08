"""Issue #7 (review 09-05): the executor anchors on what the arm RECEIVED.
Real URArm + fake RTDE responses; real ChunkExecutor + a stub arm."""
from __future__ import annotations

import math
import sys
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


def test_sustained_limiter_hold_ends_the_episode_not_the_budget(monkeypatch):
    """Verify 09-05 #3 + rig 09-07: with the limiter ON, an unreachable
    target is a HOLD that survives well past 25 ticks (a replan is ~0.85 s =
    100+ ticks) and ends the episode only after HOLD_BUDGET_S of continuous
    holding — never a silent freeze."""
    a = _arm(LIMITS)
    prime(a, pose_at_reach(0.40))
    a._ctrl.mode = "empty"
    clock = [1000.0]
    monkeypatch.setattr("phantom.drivers.real.ur.time.monotonic", lambda: clock[0])
    for _ in range(URArm.IK_REJECT_LIMIT * 4):        # 100 ticks, 0.8 s: still holding
        clock[0] += 0.008
        r = a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)
        assert r.sent is False
    assert a._ik_rejects_total >= URArm.IK_REJECT_LIMIT
    clock[0] += URArm.HOLD_BUDGET_S
    with pytest.raises(RuntimeError, match="held for"):
        a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)


def test_no_solution_hold_outlives_a_replan_then_times_out(monkeypatch):
    """Rig 09-07 regression: 12/43 episodes died as 'servo cannot stream'
    after 25 no-solution ticks (0.2 s) while the arm stood at the reach
    boundary — shorter than one replan, so the planner never got to fix the
    target. ik_invalid must hold through several replans, then time out."""
    a = _arm()
    prime(a, pose_at_reach(0.40))
    a._ctrl.mode = "empty"
    clock = [50.0]
    monkeypatch.setattr("phantom.drivers.real.ur.time.monotonic", lambda: clock[0])
    for _ in range(300):                               # 2.4 s of holding
        clock[0] += 0.008
        r = a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)
        assert r.sent is False and r.reason == "ik_invalid"
    assert a._last_cmd_pose is not None and a._ik_rejects == 300
    clock[0] += URArm.HOLD_BUDGET_S
    with pytest.raises(RuntimeError, match="held for"):
        a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)


def test_a_sent_tick_resets_the_hold_budget(monkeypatch):
    a = _arm()
    prime(a, pose_at_reach(0.40))
    clock = [50.0]
    monkeypatch.setattr("phantom.drivers.real.ur.time.monotonic", lambda: clock[0])
    a._ctrl.mode = "empty"
    for _ in range(100):
        clock[0] += 0.008
        a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)
    a._ctrl.mode = "ok"
    clock[0] += 0.008
    assert a.servo_l(pose_at_reach(0.40), 0.008, 0.1, 300).sent is True
    assert a._hold_since is None and a._ik_rejects == 0
    a._ctrl.mode = "empty"
    clock[0] += URArm.HOLD_BUDGET_S - 0.5              # a fresh hold, budget restarted
    r = a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)
    assert r.sent is False


def test_branch_rejects_still_abort_after_25_ticks():
    """A branch flip is a fault, not a hold: the 25-tick net stays."""
    a = _arm()
    prime(a, pose_at_reach(0.40))
    a._ctrl.flip = True
    with pytest.raises(RuntimeError, match="cannot stream"):
        for _ in range(URArm.IK_REJECT_LIMIT):
            a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)


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


def test_grip_latch_reads_and_clears_are_serialized_by_the_executor_lock():
    """Ultrareview 09-05: the planner's clear and the executor's read-modify-
    write of `_grip_latch` must be mutually exclusive under `_lock` (a
    timing stress test cannot show the single-bytecode race; the lock
    discipline can be shown deterministically)."""
    from phantom.deploy.executor import ChunkExecutor
    hw = load_hardware("configs/hardware.nuc.mock.yaml")

    class _Safety:
        contact_load = {"left": 9.0, "right": 9.0}

    ex = ChunkExecutor(hw, arm=_StubArm(set()), gripper=None, safety=_Safety())
    assert ex._latched_grip(0.6) == 0.6 and ex._grip_latch == 0.6
    done = threading.Event()

    def clear():
        ex.clear_grip_latch(); done.set()

    ex._lock.acquire()                    # the executor holds the lock mid-update
    try:
        threading.Thread(target=clear, daemon=True).start()
        assert not done.wait(0.2), "clear_grip_latch bypassed the executor lock"
        assert ex._grip_latch == 0.6
    finally:
        ex._lock.release()
    assert done.wait(1.0) and ex._grip_latch is None
    # and the reader takes the same lock
    ex._latched_grip(0.7)
    assert ex._grip_latch == 0.7
    ex._lock.acquire()
    got = []
    threading.Thread(target=lambda: got.append(ex._latched_grip(0.9)), daemon=True).start()
    time.sleep(0.2); assert got == [], "_latched_grip bypassed the executor lock"
    ex._lock.release(); time.sleep(0.2)
    assert got == [0.9] and ex._grip_latch == 0.9


def test_control_loss_is_diagnosed_and_kept_for_stop_json(monkeypatch):
    """Rig 09-07: 15 episodes ended as 'servoJ rejected' with no robot-side
    cause recorded. The driver must snapshot safety bits, modes, force and
    the commanded-vs-actual joint gap into `control_loss_last` (runtime folds
    it into stop_state) and name the cause in the exception."""
    a = _arm()
    prime(a, pose_at_reach(0.40))

    class DeadCtrl(_Ctrl):
        def servoJ(self, q, v, a_, t, lookahead, gain):
            return False

    class Recv(FakeRecv):
        def isProtectiveStopped(self): return True
        def getRobotMode(self): return 7
        def getSafetyMode(self): return 3
        def getSafetyStatusBits(self): return (1 << 2) | (1 << 8)   # protective_stopped + violation
        def getActualQd(self): return [0.0, 0.0, 0.29, 0.0, 0.0, 0.0]
        def getActualTCPForce(self): return [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]

    a._ctrl = DeadCtrl(); a._recv = Recv(pose_at_reach(0.40))
    monkeypatch.setitem(sys.modules, "dashboard_client", None)   # import fails -> dashboard_error
    with pytest.raises(RuntimeError, match="protective_stopped.*violation"):
        a.servo_l(pose_at_reach(0.401), 0.008, 0.1, 300)
    d = a.control_loss_last
    assert d["protective_stop"] is True and d["safety_mode"] == 3
    assert d["safety_status_names"] == ["protective_stopped", "violation"]
    assert d["tcp_force"][:2] == [3.0, 4.0] and "|F|=5.0N" in d["summary"]
    assert "q_cmd_minus_actual_max_rad" in d and "dashboard_error" in d


def test_typed_driver_faults_name_their_stop_reason():
    """Rig 09-07: a hold timeout, a branch fault and a UR protective stop all
    landed as `executor_crash` in stop.json. The executor's crash net must keep
    the fault's own name, and run_deploy must still treat all of them as
    'control script suspect' for the stage-0 reconnect."""
    from phantom.deploy.executor import ChunkExecutor
    from phantom.drivers.base import ControlLost, ServoBranchFault, ServoHoldTimeout
    from phantom.scripts.run_deploy import _CONTROL_DEAD_REASONS
    for exc, want in ((ServoHoldTimeout("x"), "servo_hold_timeout"),
                      (ServoBranchFault("x"), "servo_branch_fault"),
                      (ControlLost("x"), "control_lost"),
                      (RuntimeError("x"), "executor_crash")):
        ex = ChunkExecutor.__new__(ChunkExecutor)
        seen = []
        ex._run = lambda e=exc: (_ for _ in ()).throw(e)
        ex._halt = lambda r: seen.append(("halt", r))
        ex._set_reason = lambda r: seen.append(("reason", r))
        ex._run_guarded()
        assert seen == [("halt", want), ("reason", want)], (exc, seen)
        assert want in _CONTROL_DEAD_REASONS
        assert ex.crash_text
