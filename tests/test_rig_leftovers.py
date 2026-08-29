"""Rig-session leftovers from the 2026-08-27 delta review.

Three deploy-path defects that each survive a mock dry run and each cost real
rig time:

1. The new stale-ring hard requirements raised out of `SnapshotBuilder.build()`
   MID-EPISODE, escaped `DeploymentRuntime.run_episode`'s bare try/finally and
   reached run_deploy as an rc-1 traceback: no operator label prompt and an
   empty episode directory. The ring warm-up budget (10 s) was also shorter
   than the RealSense driver's own first pipeline rebuild (~17 s), so a camera
   that was healing itself normally aborted the episode.
2. `RealSenseCamera` marks itself dead once its bounded rebuild gives up, but
   nothing read that: `SensorSession.all_alive()` only looks at thread
   liveness, so a permanently wedged camera became an endless 10 Hz retry
   against a ring nobody writes instead of the session-fatal `worker_died`.
3. A swallowed `servo_stop failed` left `URArm._servo_active` set, and
   run_deploy's `_CONTROL_DEAD_REASONS` omitted `safety_stop`, so the next
   episode's homing `move_l` was refused and downgraded to a log line for the
   rest of the campaign.
"""

from __future__ import annotations

import logging
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.deploy import runtime as RT
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.planner import (SnapshotBuilder, StaleStreamError,
                                    snapshot_stop_reason)
from phantom.deploy.runtime import (DeploymentRuntime, _wait_rings_warm,
                                    fatal_reason, warmup_timeout_s)
from phantom.drivers.real.realsense import first_frame_budget_s
from phantom.recording.workers import ThreadPoller, make_camera_poller
from phantom_test_utils import make_small_hw


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _CrawlPolicy:
    """Constant crawl, exactly like the existing mock dry-run policy; `hook` is
    called at the top of every replan so a test can wedge a device mid-episode."""

    def __init__(self, hw, hook=None):
        self.hw = hw
        self.hook = hook
        self.n = 0

    def reset_episode(self):
        self.n = 0

    def replan(self, snap, prev_plan, tcp_pose):
        from phantom.inference.policy import Plan
        hw = self.hw
        self.n += 1
        if self.hook is not None:
            self.hook(self.n)
        H, A = hw.control.chunk_horizon, hw.control.action_dim
        actions = np.zeros((H, A))
        actions[:, 0] = 5e-4
        actions[:, 6] = 0.3
        now = time.perf_counter()
        return Plan(t_created=now,
                    t0_pose=np.asarray(tcp_pose, dtype=np.float64).copy(),
                    actions=actions,
                    action_times=now + 0.05 + np.arange(H) / hw.control.action_rate_hz,
                    sigma=np.zeros(4), gate=1.0, p_evt=np.zeros(5), cpk=None)


class _RaisingRing:
    """Ring stub that never has a sample (the shape _wait_rings_warm blocks on)."""

    def latest(self, k=1):
        return np.zeros(0), {}

    def latest_ts(self):
        return None


def _never_warm_builder(hw):
    return SnapshotBuilder(hw, types.SimpleNamespace(
        rings={"camera_scene": _RaisingRing()}), "vision_only")


# ===========================================================================
# 1. the staleness raise must not escape run_episode; warm-up must cover a
#    RealSense pipeline rebuild
# ===========================================================================

def test_stale_stream_error_carries_a_stop_reason():
    e = StaleStreamError("camera_scene_stale", "camera_scene stale by 9.9s")
    assert isinstance(e, AssertionError)        # _wait_rings_warm still sees it
    assert snapshot_stop_reason(e) == "camera_scene_stale"
    # anything else still ends the episode cleanly, just generically
    assert snapshot_stop_reason(AssertionError("boom")) == "snapshot_invalid"


def test_snapshot_staleness_reasons_are_the_stop_reasons():
    """The camera/arm/gripper hard requirements must each name the stop reason
    the episode will end with, not just a message."""
    hw = make_small_hw()
    t = time.perf_counter()

    class _Ring:
        def __init__(self, ts, fields):
            self.ts, self.fields = np.asarray(ts, dtype=np.float64), fields

        def latest(self, k=1):
            m = min(k, len(self.ts))
            sl = slice(len(self.ts) - m, len(self.ts))
            return self.ts[sl].copy(), {f: np.stack([np.asarray(v)] * m)
                                        for f, v in self.fields.items()}

    cam = {"color": np.zeros(hw.cameras.scene.color.hwc, np.uint8)}
    arm = {"q": np.zeros(hw.arm.dof), "qd": np.zeros(hw.arm.dof),
           "tcp_pose": np.zeros(6), "tcp_speed": np.zeros(6), "ft": np.zeros(6)}

    def _build(cam_t, arm_t):
        rings = {"camera_scene": _Ring([cam_t], cam),
                 "arm": _Ring(np.linspace(arm_t - 0.05, arm_t, 8), arm),
                 "gripper": _Ring([t], {"state": np.array([0.4, 3.0])})}
        return SnapshotBuilder(hw, types.SimpleNamespace(rings=rings),
                               "vision_only").build()

    with pytest.raises(StaleStreamError) as ei:
        _build(t - 60.0, t)
    assert ei.value.reason == "camera_scene_stale"
    with pytest.raises(StaleStreamError) as ei:
        _build(t, t - 60.0)
    assert ei.value.reason == "arm_stale"


def test_stale_camera_mid_episode_ends_the_episode_through_the_normal_path(
        tmp_path, monkeypatch):
    """THE regression: a ring that goes stale after the episode is under way
    used to propagate out of run_episode. The operator then lost the label
    prompt and the episode directory was left empty — on the rig it surfaced as
    an rc-1 traceback from run_deploy."""
    hw = make_small_hw()
    real_build = SnapshotBuilder.build
    calls = {"n": 0}

    def build(self):
        calls["n"] += 1
        if calls["n"] > 2:          # 1 = warm-up, 2 = first replan
            raise StaleStreamError(
                "camera_scene_stale",
                "camera_scene stale by 9.90s (>0.50s) — the scene pipeline is "
                "wedged")
        return real_build(self)

    monkeypatch.setattr(SnapshotBuilder, "build", build)
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="vision_only",
                           out_root=tmp_path) as rt:
        res = rt.run_episode(task="whiteboard", max_replans=10)

    assert res.stopped_reason == "camera_scene_stale"    # clean stop, no raise
    assert res.fatal_reason is None
    assert res.episode_path is not None                  # recorder closed
    assert res.trace_path is not None and res.trace_path.exists()
    assert res.n_replans == 1                            # the one good replan


def test_warmup_failure_is_a_clean_stop_reason_too(tmp_path, monkeypatch):
    """A ring that never warms must not crash the process either: the episode
    ends, the directory is closed and the operator can still label it."""
    hw = make_small_hw()

    def build(self):
        raise StaleStreamError("arm_stale", "no arm samples yet")

    monkeypatch.setattr(SnapshotBuilder, "build", build)
    monkeypatch.setattr(RT, "warmup_timeout_s", lambda: 0.2)
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="vision_only",
                           out_root=tmp_path) as rt:
        res = rt.run_episode(task="whiteboard", max_replans=3)
    assert res.stopped_reason == "arm_stale"
    assert res.fatal_reason is None
    assert res.n_replans == 0


def test_warmup_budget_covers_one_realsense_pipeline_rebuild():
    """10 s aborted at two thirds of the driver's FIRST rebuild: 3 blocking
    wait_for_frames() timeouts at the 5 s pyrealsense2 default, then the
    rebuild itself."""
    assert first_frame_budget_s() >= 15.0          # 3 x 5 s of timeouts alone
    assert warmup_timeout_s() >= 20.0
    assert warmup_timeout_s() > first_frame_budget_s()


def test_wait_rings_warm_says_what_it_is_waiting_for(caplog):
    hw = make_small_hw()
    with caplog.at_level(logging.INFO, logger="phantom.deploy.runtime"):
        with pytest.raises(AssertionError):
            _wait_rings_warm(_never_warm_builder(hw), timeout_s=0.15)
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "no camera frames yet" in msgs           # names the missing stream
    assert "did not warm" in msgs


# ===========================================================================
# 2. a camera that declares itself dead must become a DEAD WORKER
# ===========================================================================

class _FakeCam:
    """Camera stub with the driver health contract: `healthy` False covers the
    recoverable rebuild window, `dead_reason` only the unrecoverable end."""

    def __init__(self):
        self.reads = 0
        self.wedged = False
        self.dead_reason: str | None = None
        self.cfg = SimpleNamespace(fps=30.0)

    @property
    def healthy(self):
        return not self.wedged and self.dead_reason is None

    def read(self):
        self.reads += 1
        if self.wedged or self.dead_reason:
            raise RuntimeError("camera stalled")
        return SimpleNamespace(t_host=time.perf_counter(), color=np.zeros((2, 2, 3),
                                                                         np.uint8))


class _NullRing:
    def push(self, t, **kw):
        pass


def _wait_until(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_camera_poller_exits_when_the_driver_declares_itself_dead():
    """The retry loop is unbounded by design; a device that will never produce
    another sample must end the WORKER so all_alive() reports it."""
    cam = _FakeCam()
    poller = make_camera_poller(cam, "scene", 30.0, _NullRing())
    poller.start()
    try:
        assert _wait_until(lambda: cam.reads > 0)
        cam.dead_reason = "6 consecutive pipeline rebuilds failed"
        assert _wait_until(lambda: not poller.is_alive())
        assert poller.dead_reason == "6 consecutive pipeline rebuilds failed"
    finally:
        poller.stop()


def test_camera_poller_survives_a_recoverable_rebuild():
    """`healthy` is False during the driver's own bounded rebuild, which
    usually SUCCEEDS (a USB stall on the DmTac hub). Killing the worker there
    would end a rig session over a stall the driver was about to heal — the
    poller must key off `dead_reason`, not `healthy`."""
    cam = _FakeCam()
    poller = make_camera_poller(cam, "scene", 60.0, _NullRing())
    poller.start()
    try:
        cam.wedged = True                       # healthy False, dead_reason None
        assert not cam.healthy
        time.sleep(0.3)
        assert poller.is_alive(), "a recoverable stall must not kill the worker"
        cam.wedged = False
        n = cam.reads
        assert _wait_until(lambda: cam.reads > n)   # still polling after recovery
    finally:
        poller.stop()


def test_poller_without_a_death_probe_is_unchanged():
    """Tactile/arm/gripper pollers pass no dead_fn: their retry loop must keep
    its old unbounded behaviour."""
    calls = {"n": 0}

    def poll():
        calls["n"] += 1
        raise RuntimeError("nope")

    poller = ThreadPoller("x", 100.0, poll, _NullRing())
    poller.start()
    try:
        assert _wait_until(lambda: calls["n"] >= 3)
        assert poller.is_alive() and poller.dead_reason is None
    finally:
        poller.stop()


def test_mock_camera_matches_the_driver_health_contract():
    """Parity: the mock must expose the same two-level health signal, or the
    dead-worker path can only ever be exercised on real hardware."""
    from phantom.drivers.base import Camera
    from phantom.drivers.mock.realsense import MockCamera
    from phantom.drivers.real.realsense import RealSenseCamera
    for cls in (Camera, MockCamera, RealSenseCamera):
        assert isinstance(cls.dead_reason, property)
        assert isinstance(cls.healthy, property)

    hw = make_small_hw()
    cam = MockCamera(hw.cameras.scene, "scene")
    cam.connect()
    assert cam.healthy and cam.dead_reason is None
    cam.read()
    cam._dead_reason = "wedged"
    assert cam.healthy is False and cam.dead_reason == "wedged"
    with pytest.raises(RuntimeError, match="is dead"):
        cam.read()


def test_a_dead_scene_camera_is_session_fatal(tmp_path):
    """End to end over the REAL SensorSession on mock drivers: wedging the
    camera mid-episode must stop the episode as `worker_died` and mark the
    whole deployment session fatal (run_deploy exit code 5), not spin at
    10 Hz forever."""
    hw = make_small_hw()
    state: dict = {}

    def hook(n):
        if n == 1:
            state["cam"].__dict__["_dead_reason"] = "bounded rebuild gave up"
        # give the poller a moment to notice and exit
        _wait_until(lambda: not state["rt"].session.all_alive(), timeout=3.0)

    pol = _CrawlPolicy(hw, hook=hook)
    with DeploymentRuntime(hw, pol, mode="vision_only", out_root=tmp_path) as rt:
        state["rt"] = rt
        state["cam"] = rt.rig.cameras["scene"]
        res = rt.run_episode(task="whiteboard", max_replans=6)

    assert res.stopped_reason == "worker_died"
    assert res.fatal_reason == "worker_died"       # run_deploy returns 5
    assert res.episode_path is not None            # still labelled + saved


# ===========================================================================
# 3. a failed servo_stop must arm the control-script rebuild
# ===========================================================================

def _executor(hw, arm):
    return ChunkExecutor(hw, arm, SimpleNamespace(), SimpleNamespace())


class _CleanThread:
    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


def _stopped(hw, arm):
    ex = _executor(hw, arm)
    ex._thread = ex._grip_thread = _CleanThread()
    ex.stop()
    return ex


def test_failed_servo_stop_becomes_a_control_dead_stop_reason():
    """URArm re-raises from servo_stop() ONLY when the control script is still
    playing — i.e. `_servo_active` stays True and the next episode's homing
    move_l is refused. Swallowed as a log line, that downgraded the rest of the
    campaign to hand-jogged OOD starts."""
    from phantom.scripts.run_deploy import _CONTROL_DEAD_REASONS
    hw = make_small_hw()

    def boom():
        raise RuntimeError("servoStop failed while the script is running")

    ex = _stopped(hw, SimpleNamespace(servo_stop=boom))
    assert ex.stopped_reason == "servo_stop_failed"
    assert ex.stopped_reason in _CONTROL_DEAD_REASONS   # -> recover_control()
    assert fatal_reason(ex) is None                     # recoverable, not fatal


def test_a_clean_servo_stop_leaves_the_episode_reason_alone():
    hw = make_small_hw()
    ex = _stopped(hw, SimpleNamespace(servo_stop=lambda: None))
    assert ex.stopped_reason is None


def test_failed_servo_stop_never_overwrites_the_real_stop_reason():
    """First writer wins: the operator must still see WHY the episode ended."""
    hw = make_small_hw()

    def boom():
        raise RuntimeError("servoStop failed")

    ex = _executor(hw, SimpleNamespace(servo_stop=boom))
    ex._thread = ex._grip_thread = _CleanThread()
    ex.request_stop("protective_stop")
    ex.stop()
    assert ex.stopped_reason == "protective_stop"


def test_control_dead_reasons_cover_the_swallowed_stop_paths():
    """`safety_stop` calls URArm.stop(), whose stopL failure is swallowed, and
    one of the conditions that raises it is `arm_stale` (a dead RTDE stream).
    `worker_died` is session-fatal before the arm is reused, but the predicate
    must still answer True."""
    from phantom.scripts.run_deploy import _CONTROL_DEAD_REASONS
    for r in ("protective_stop", "executor_crash", "motion_stall",
              "servo_stop_failed", "safety_stop", "worker_died"):
        assert r in _CONTROL_DEAD_REASONS
    # a clean episode must NOT trigger a control rebuild
    assert None not in _CONTROL_DEAD_REASONS
    assert "camera_scene_stale" not in _CONTROL_DEAD_REASONS


# ---------------------------------------------------------------------------
# 3b. a homing move_l failure right after a control rebuild refuses the episode
# ---------------------------------------------------------------------------

class _StubArm:
    def __init__(self):
        self.n_reconnects = 0

    def is_ready_for_control(self):
        return True, ""

    def reconnect_control(self):
        self.n_reconnects += 1

    def program_running(self):
        return True

    def get_state(self):
        return SimpleNamespace(tcp_pose=np.zeros(6))


class _StubRuntime:
    """Stands in for DeploymentRuntime inside run_deploy.main()."""

    results: list = []
    last: "_StubRuntime | None" = None

    def __init__(self, hw, policy, mode, out_root, **kw):
        type(self).last = self
        self.kw = kw                    # deploy levers (parity_fixes, veto)
        self.rig = SimpleNamespace(
            arm=_StubArm(),
            gripper=SimpleNamespace(get_state=lambda: SimpleNamespace(
                position=0.0, obj=3.0)))
        self.recorder = SimpleNamespace(relabel=lambda *a, **k: None)
        self.episodes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run_episode(self, **kw):
        res = self.results[min(self.episodes, len(self.results) - 1)]
        self.episodes += 1
        return res


def _main_with_stubs(monkeypatch, tmp_path, *, homing_raises, results, argv=()):
    """Drive run_deploy.main() over a real-arm campaign with every device and
    the whole model stack stubbed out."""
    from phantom.deploy import start_pose as sp
    from phantom.scripts import run_deploy as RD

    hw = make_small_hw(mode={"drivers": "real"})
    monkeypatch.setattr(RD, "load_hardware", lambda *a, **k: hw)
    monkeypatch.setattr(RD, "load_paths", lambda *a, **k: SimpleNamespace(
        validate=lambda **k: None, episodes_root=lambda: tmp_path))
    monkeypatch.setattr(RD, "build_policy", lambda *a, **k: SimpleNamespace(
        nfe=1, guidance=1.0, rf=SimpleNamespace()))
    _StubRuntime.results = results
    monkeypatch.setattr(RD, "DeploymentRuntime", _StubRuntime)
    monkeypatch.setattr(RD.time, "sleep", lambda *_: None)     # AE settle
    monkeypatch.setattr("builtins.input", lambda *_: "")

    stats = object()
    monkeypatch.setattr(sp, "load_start_stats", lambda: {"grasp": stats})
    monkeypatch.setattr(sp, "start_sigma_report",
                        lambda *a, **k: (np.zeros(7), "in distribution"))

    def _move(*a, **k):
        if homing_raises:
            raise RuntimeError("move_l while servo mode may be active; "
                               "call servo_stop() first")

    monkeypatch.setattr(sp, "move_to_start", _move)
    return RD.main(["--system", "vision_only", "--task", "grasp", "--tiny",
                    "--episodes", "2", *argv])


def _result(reason):
    from phantom.deploy.runtime import EpisodeResult
    return EpisodeResult(episode_path=None, stopped_reason=reason, n_replans=1,
                         safety_events=0, trace_path=None, fatal_reason=None)


def test_homing_failure_after_a_control_rebuild_refuses_the_episode(
        monkeypatch, tmp_path, caplog):
    """Rebuilt the script, verified it playing, and move_l STILL failed: the
    arm is not controllable from this process. Continuing is what produced a
    campaign of hand-jogged OOD starts."""
    with caplog.at_level(logging.ERROR, logger="run_deploy"):
        rc = _main_with_stubs(monkeypatch, tmp_path, homing_raises=True,
                              results=[_result("safety_stop")])
    assert rc == 4
    assert any("not controllable" in r.getMessage() for r in caplog.records)


def test_homing_failure_with_no_preceding_recovery_stays_tolerant(
        monkeypatch, tmp_path):
    """Unchanged for the ordinary case: jog by hand, the start gate re-checks
    before anything runs."""
    rc = _main_with_stubs(monkeypatch, tmp_path, homing_raises=True,
                          results=[_result(None)])
    assert rc == 0
    assert _StubRuntime.last.rig.arm.n_reconnects == 0   # nothing to recover


def test_a_safety_stop_now_arms_the_control_rebuild(monkeypatch, tmp_path):
    """Episode 1 ends with `safety_stop` -> episode 2 must rebuild the RTDE
    control script BEFORE homing (it did not before 2026-08-27)."""
    rc = _main_with_stubs(monkeypatch, tmp_path, homing_raises=False,
                          results=[_result("safety_stop"), _result(None)])
    assert rc == 0
    assert _StubRuntime.last.rig.arm.n_reconnects == 1


def test_a_clean_episode_does_not_rebuild_the_control_script(monkeypatch, tmp_path):
    rc = _main_with_stubs(monkeypatch, tmp_path, homing_raises=False,
                          results=[_result(None), _result(None)])
    assert rc == 0
    assert _StubRuntime.last.rig.arm.n_reconnects == 0
