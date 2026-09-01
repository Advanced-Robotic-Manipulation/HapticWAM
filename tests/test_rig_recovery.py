"""Rig-session recovery regressions (audit 2026-08-26).

Four defects that each cost a whole rig session and that no mock rehearsal
could surface:

1. run_deploy never rebuilt the RTDE control script after a protective stop,
   so every later episode homed-failed and recorded zero motion.
2. URArm._servo_active was sticky — one failed servoStop (or any
   reconnect_control) refused move_l, i.e. start-pose homing, forever.
3. One failed RealSense pipeline rebuild killed the scene camera for the whole
   session and nothing detected the frozen RGB stream.
4. SafetyMonitor logged + retained one event per executor TICK while a
   condition persisted (~15k warnings from inside the 2 ms servo loop).
"""

from __future__ import annotations

import inspect
import logging
import sys
import time
import types
import uuid
from contextlib import contextmanager

import numpy as np
import pytest

from phantom.deploy.safety import SafetyAction, SafetyMonitor, camera_stale_s
from phantom.drivers.mock.ur import MockArm
from phantom.drivers.real.ur import URArm
from phantom.recording.ringbuffer import SharedRingBuffer
from phantom_test_utils import make_hw, make_small_hw


# ---------------------------------------------------------------------------
# 2. URArm servo guard
# ---------------------------------------------------------------------------

class _DeadScriptCtrl:
    """Control interface whose SOCKET is up but whose control script died (a UR
    protective stop): ur_rtde raises out of servoStop in that state."""

    def __init__(self, connected: bool = True):
        self._connected = connected
        self.moves = []

    def isConnected(self):
        return self._connected

    def servoStop(self):
        raise RuntimeError("Robot is disconnected")

    def moveL(self, pose, speed, accel, async_):
        self.moves.append(list(pose))
        return True

    def servoJ(self, *a):
        return False

    def stopScript(self):
        pass

    def disconnect(self):
        pass


class _RecvStub:
    def __init__(self, runtime_state: int):
        self.runtime_state = runtime_state

    def getRuntimeState(self):
        return self.runtime_state


def _arm_with(ctrl, runtime_state):
    arm = URArm(make_hw())
    arm._ctrl = ctrl
    arm._recv = _RecvStub(runtime_state)
    arm._servo_active = True
    return arm


def test_failed_servo_stop_clears_guard_when_script_is_dead():
    """The servo session died with the control script — the guard MUST clear,
    otherwise move_l (start-pose homing) is refused for the rest of the
    session and the operator hand-jogs every episode into an OOD start."""
    ctrl = _DeadScriptCtrl()
    arm = _arm_with(ctrl, URArm._RT_STOPPED)
    arm.servo_stop()                       # does not raise: servo IS stopped
    assert arm._servo_active is False
    arm.move_l(np.zeros(6), 0.1, 0.3)      # reaches the controller now
    assert len(ctrl.moves) == 1


def test_failed_servo_stop_clears_guard_when_control_stream_lost():
    """Same, via the other death path: _require_ctrl raises because the socket
    dropped."""
    arm = _arm_with(_DeadScriptCtrl(connected=False), URArm._RT_STOPPED)
    arm.servo_stop()
    assert arm._servo_active is False


def test_failed_servo_stop_keeps_guard_while_script_still_running():
    """The one case the guard must survive: servoStop failed against a script
    that is still PLAYING, so a servo stream may genuinely still be live."""
    arm = _arm_with(_DeadScriptCtrl(), URArm._RT_PLAYING)
    with pytest.raises(RuntimeError, match="Robot is disconnected"):
        arm.servo_stop()
    assert arm._servo_active is True
    with pytest.raises(RuntimeError, match="servo mode may be active"):
        arm.move_l(np.zeros(6), 0.1, 0.3)


def test_reconnect_control_clears_servo_guard(monkeypatch):
    """A brand-new control script cannot have a live servo session; leaving the
    guard set blocked a moveL that is now perfectly legal."""
    monkeypatch.setattr(URArm, "_reconnect_settle_s", 0.0)
    monkeypatch.setattr(URArm, "_connect_control", lambda self: None)
    arm = _arm_with(_DeadScriptCtrl(), URArm._RT_STOPPED)
    arm.reconnect_control()
    assert arm._servo_active is False


def test_mock_arm_matches_real_control_session_api():
    """Mock parity: the deploy recovery path must be rehearsable without a
    robot (MockArm had no reconnect_control at all)."""
    arm = MockArm(make_small_hw())
    arm.connect(control=True)
    arm.servo_l(np.zeros(6), 0.008, 0.1, 300)
    assert arm._servo_active is True
    assert arm.is_ready_for_control() == (True, "")
    assert arm.program_running() is True
    arm.reconnect_control()
    assert arm._servo_active is False and arm.n_reconnects == 1
    arm.trigger_protective_stop(True)
    ok, why = arm.is_ready_for_control()
    assert ok is False and "protective" in why
    assert arm.program_running() is False


# ---------------------------------------------------------------------------
# 1. run_deploy protective-stop recovery
# ---------------------------------------------------------------------------

class _RecoveringArm:
    """Arm whose control script comes back after `ready_after` operator
    prompts, and whose script starts iff `script_starts`."""

    def __init__(self, ready_after=0, script_starts=True):
        self.ready_after = ready_after
        self.script_starts = script_starts
        self.prompts = 0
        self.n_reconnects = 0
        self._running = False

    def is_ready_for_control(self):
        if self.ready_after > 0:
            return False, "robot is STILL protective-stopped"
        return True, ""

    def reconnect_control(self):
        self.n_reconnects += 1
        self._running = self.script_starts

    def program_running(self):
        return self._running


def _run_deploy(monkeypatch, arm, timeout_s=0.05):
    from phantom.scripts import run_deploy as RD
    monkeypatch.setattr(RD, "_SCRIPT_START_TIMEOUT_S", timeout_s)

    def _input(_prompt=""):
        arm.prompts += 1
        arm.ready_after -= 1
        return ""

    monkeypatch.setattr("builtins.input", _input)
    return RD


def test_recover_control_blocks_on_operator_then_rebuilds(monkeypatch):
    arm = _RecoveringArm(ready_after=2)
    RD = _run_deploy(monkeypatch, arm)
    assert RD.recover_control(arm, "protective_stop") is True
    assert arm.prompts == 2          # blocked until the pendant was cleared
    assert arm.n_reconnects == 1


def test_recover_control_fails_loudly_when_script_never_starts(monkeypatch):
    """isConnected() is only the socket — a rebuilt interface whose script did
    not come up must be reported here, not by a rejected servoJ mid-episode."""
    arm = _RecoveringArm(script_starts=False)
    RD = _run_deploy(monkeypatch, arm)
    assert RD.recover_control(arm, "executor_crash") is False
    assert arm.n_reconnects == RD._RECONNECT_TRIES


def test_run_deploy_recovers_before_homing_the_next_episode():
    """Stage 0 (recover) must precede stage 1 (homing move), and the previous
    episode's stop reason must be what arms it."""
    from phantom.scripts import run_deploy as RD
    src = inspect.getsource(RD.main)
    assert "prev_reason = res.stopped_reason" in src
    assert src.index("recover_control(") < src.index("sp.move_to_start")
    assert "protective_stop" in RD._CONTROL_DEAD_REASONS
    assert "executor_crash" in RD._CONTROL_DEAD_REASONS


# ---------------------------------------------------------------------------
# 3. RealSense pipeline rebuild
# ---------------------------------------------------------------------------

class _FakeRS:
    """pyrealsense2 stand-in whose device can be BUSY (rebuild fails) and/or
    STALLED (wait_for_frames times out) — the USB-stall state the rebuild was
    written for."""

    def __init__(self, hw):
        self.busy = False
        self.stalled = False
        self.starts = 0
        self.hwc = hw.cameras.scene.color.hwc
        outer = self

        class _Cfg:
            def enable_device(self, s):
                pass

            def enable_stream(self, *a, **k):
                pass

        class _Frame:
            def get_data(self):
                return np.zeros(outer.hwc, dtype=np.uint8)

        class _Frames:
            def get_color_frame(self):
                return _Frame()

            def get_timestamp(self):
                return 1000.0

        class _Pipe:
            def __init__(self):
                self.started = False

            def start(self, cfg):
                outer.starts += 1
                if outer.busy:
                    raise RuntimeError("Device or resource busy")
                self.started = True

            def stop(self):
                if not self.started:
                    raise RuntimeError("stop() cannot be called before start()")
                self.started = False

            def wait_for_frames(self):
                if not self.started:
                    raise RuntimeError("wait_for_frames() cannot be called "
                                       "before start()")
                if outer.stalled:
                    raise RuntimeError("Frame didn't arrive within 5000")
                return _Frames()

        e = types.SimpleNamespace
        self.module = types.ModuleType("pyrealsense2")
        self.module.stream = e(color=0, depth=1)
        self.module.format = e(rgb8=0, z16=1)
        self.module.config = _Cfg
        self.module.pipeline = _Pipe


@contextmanager
def _fake_camera(monkeypatch):
    import phantom.drivers.real.realsense as R
    hw = make_small_hw()
    rs = _FakeRS(hw)
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs.module)
    monkeypatch.setattr(R, "_REBUILD_BACKOFF_S", 0.0)   # no waiting in tests
    monkeypatch.setattr(R, "_REBUILD_BACKOFF_MAX_S", 0.0)
    yield R.RealSenseCamera(hw.cameras.scene, "scene"), rs, R


def _read_ignoring(cam, n):
    for _ in range(n):
        try:
            cam.read()
        except RuntimeError:
            pass


def test_failed_pipeline_start_leaves_no_unstarted_pipeline(monkeypatch):
    """A pipeline published before start() succeeded is what made every later
    read() blow up inside disconnect() and strand _pipe at None forever."""
    with _fake_camera(monkeypatch) as (cam, rs, _R):
        rs.busy = True
        with pytest.raises(RuntimeError, match="busy"):
            cam.connect()
        assert cam._pipe is None
        assert cam.healthy is False


def test_camera_recovers_after_rebuilds_that_hit_a_busy_device(monkeypatch):
    """The pre-fix driver made exactly TWO pipeline.start() attempts and then
    served frozen frames forever. The rebuild must retry until the device frees
    up — and a real frame is what proves recovery."""
    with _fake_camera(monkeypatch) as (cam, rs, _R):
        cam.connect()
        assert cam.healthy and rs.starts == 1
        rs.stalled = rs.busy = True             # USB stall + device busy
        _read_ignoring(cam, 5)                  # 3 timeouts, then 3 rebuild tries
        assert rs.starts > 2, "rebuild gave up after the old two attempts"
        assert cam.healthy is False             # loud, not frozen
        assert cam.last_frame_age() == float("inf")
        rs.stalled = rs.busy = False            # link comes back
        frame = cam.read()
        assert frame.color.shape == rs.hwc
        assert cam.healthy is True
        assert cam._rebuilds == 0               # counter re-armed by a real frame
        assert cam.last_frame_age() < 1.0


def test_camera_declared_dead_after_bounded_rebuild_attempts(monkeypatch):
    """A permanently wedged camera must end in a loud error every read, never
    in a silently frozen ring."""
    with _fake_camera(monkeypatch) as (cam, rs, R):
        cam.connect()
        rs.stalled = rs.busy = True
        _read_ignoring(cam, 40)
        assert rs.starts == 1 + R._REBUILD_MAX_ATTEMPTS   # bounded
        assert cam.healthy is False
        with pytest.raises(RuntimeError, match="is dead"):
            cam.read()


# ---------------------------------------------------------------------------
# 3c/4. SafetyMonitor: camera staleness + per-condition logging
# ---------------------------------------------------------------------------

@contextmanager
def _rings(hw):
    uid = uuid.uuid4().hex[:8]
    h, w, c = hw.cameras.scene.color.hwc
    rings = {
        "arm": SharedRingBuffer(f"r_arm_{uid}", 16, {
            "ft": ((6,), "float64"), "protective_stop": ((), "uint8")}, create=True),
        "camera_scene": SharedRingBuffer(f"r_cam_{uid}", 4, {
            "color": ((h, w, c), "uint8")}, create=True),
    }
    try:
        rings["arm"].push(time.perf_counter(), ft=np.zeros(6),
                          protective_stop=np.uint8(0))
        yield rings
    finally:
        for ring in rings.values():
            ring.close()


def _push_cam(rings, hw, ts):
    rings["camera_scene"].push(ts, color=np.zeros(hw.cameras.scene.color.hwc,
                                                 dtype=np.uint8))


def _push_arm(rings, ts):
    """The arm ring has its own freshness guard (arm_stale_s) — a test that
    simulates seconds of executor ticks must keep the arm stream alive too, or
    it is testing `arm_stale` instead of whatever it meant to test."""
    rings["arm"].push(ts, ft=np.zeros(6), protective_stop=np.uint8(0))


IN_BOX = np.array([0.0, -0.45, 0.25, 0.0, 3.14, 0.0])
# outside the workspace box (y above its -0.2 ceiling) but INSIDE the
# reach_clamp_m radius, so exactly ONE condition (workspace_clamp) fires
OUT_OF_BOX = np.array([0.0, -0.1, 0.25, 0.0, 3.14, 0.0])


def test_stale_scene_camera_stops_the_episode():
    """A wedged RealSense froze the ring and the policy kept replanning on the
    pre-stall frame with the arm still driving — nothing looked at it."""
    hw = make_small_hw()
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        t0 = time.perf_counter()
        _push_cam(rings, hw, t0)
        assert mon.check(t0, IN_BOX).action == SafetyAction.OK
        # keep the ARM stream alive: this test is about the camera alone
        t1 = t0 + camera_stale_s(hw) + 0.01
        _push_arm(rings, t1)
        v = mon.check(t1, IN_BOX)
        assert v.action == SafetyAction.STOP_EPISODE
        assert [e.kind for e in v.events] == ["camera_scene_stale"]


def test_sustained_clamp_is_one_event_and_one_log_line(caplog):
    """~15k retained events and ~15k stderr writes from inside the 2 ms servo
    loop, for a single clamp condition."""
    hw = make_small_hw()
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        t0 = time.perf_counter()
        _push_cam(rings, hw, t0)
        with caplog.at_level(logging.WARNING, logger="phantom.deploy.safety"):
            for k in range(400):                      # 0.8 s at 500 Hz
                _push_cam(rings, hw, t0 + k * 0.002)
                _push_arm(rings, t0 + k * 0.002)
                v = mon.check(t0 + k * 0.002, OUT_OF_BOX)
                assert v.action == SafetyAction.CLAMP  # clamping every tick
        clamp_logs = [r for r in caplog.records if "workspace_clamp" in r.message]
        assert len(clamp_logs) == 1, "one condition must log once, not per tick"
        assert len(mon.log_events) == 1
        assert mon.log_events[0].count == 400          # ticks are counted
        assert mon.dropped_events == 0


def test_condition_clearing_re_arms_the_rising_edge():
    hw = make_small_hw()
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        t0 = time.perf_counter()
        for k, target in enumerate([OUT_OF_BOX, OUT_OF_BOX, IN_BOX, OUT_OF_BOX]):
            _push_cam(rings, hw, t0 + k * 0.002)
            mon.check(t0 + k * 0.002, target)
        assert [e.kind for e in mon.log_events] == ["workspace_clamp"] * 2
        assert [e.count for e in mon.log_events] == [2, 1]


def test_log_events_are_bounded(monkeypatch):
    import phantom.deploy.safety as S
    monkeypatch.setattr(S, "_MAX_LOG_EVENTS", 3)
    hw = make_small_hw()
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        t0 = time.perf_counter()
        for k in range(20):                      # clamp / clear / clamp / ...
            _push_cam(rings, hw, t0 + k * 0.002)
            mon.check(t0 + k * 0.002, OUT_OF_BOX if k % 2 == 0 else IN_BOX)
        assert len(mon.log_events) == 3
        assert mon.dropped_events == 7


# ---------------------------------------------------------------------------
# 3c. planner-side freshness (the policy must never see a frozen frame)
# ---------------------------------------------------------------------------

class _AgeRing:
    """Ring stub whose newest timestamp is `age` seconds old."""

    def __init__(self, fields, age=0.0, n=2):
        self.fields, self.age, self.n = fields, age, n

    def latest(self, k=1):
        m = min(k, self.n)
        ts = np.array([time.perf_counter() - self.age] * m)
        return ts, {f: np.stack([np.asarray(v)] * m) for f, v in self.fields.items()}

    def latest_ts(self):
        return time.perf_counter() - self.age


def _snapshot_rings(hw, cam_age):
    dof = hw.arm.dof
    return {
        "camera_scene": _AgeRing({"color": np.zeros(hw.cameras.scene.color.hwc,
                                                    np.uint8)}, age=cam_age, n=1),
        "arm": _AgeRing({"q": np.zeros(dof), "qd": np.zeros(dof),
                         "tcp_pose": np.zeros(6), "tcp_speed": np.zeros(6),
                         "ft": np.zeros(6)}),
        "gripper": _AgeRing({"state": np.array([0.4, 3.0])}, n=1),
    }


def test_snapshot_rejects_a_stale_scene_frame():
    from phantom.deploy.planner import SnapshotBuilder
    hw = make_small_hw()
    sb = SnapshotBuilder(hw, types.SimpleNamespace(
        rings=_snapshot_rings(hw, cam_age=camera_stale_s(hw) + 0.5)), "vision_only")
    with pytest.raises(AssertionError, match="camera_scene stale"):
        sb.build()
    sb = SnapshotBuilder(hw, types.SimpleNamespace(
        rings=_snapshot_rings(hw, cam_age=0.0)), "vision_only")
    assert sb.build().rgb.shape == hw.cameras.scene.color.hwc
