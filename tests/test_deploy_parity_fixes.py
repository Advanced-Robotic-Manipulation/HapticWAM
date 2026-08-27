"""Deploy-path parity + fail-closed regressions (Codex static review 2026-08-27).

Five defects on the real-robot path, none of which a mock dry run surfaces:

1. The arm ring had no freshness guard: a dead RTDE-receive worker froze
   tcp_pose, tcp_speed, the wrist F/T window and protective_stop at once while
   the policy kept replanning and the executor kept servoing.
2. A timed-out executor/gripper join only logged; run_deploy then started the
   next episode with a stale worker still owning the Robotiq socket.
3. `--ema` was opt-in at deploy while run_eval always loads EMA — the rig and
   the eval numbers came from different artifacts of one checkpoint.
4. The planner advanced its prev_chunk/prev_cpk feedback state even when the
   executor REJECTED the plan, conditioning the next replan on motion that
   never happened.
5. The deploy wrist window was the last `window_len` ROWS of the arm ring — a
   sample count, not the `window_s` duration WindowSampler.sample() resamples
   onto with np.linspace + np.interp.
"""

from __future__ import annotations

import inspect
import logging
import time
import types
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.planner import PlannerLoop, SnapshotBuilder
from phantom.deploy.runtime import DeploymentRuntime, fatal_reason
from phantom.deploy.safety import (SafetyAction, SafetyMonitor, arm_stale_s,
                                   camera_stale_s)
from phantom.recording.ringbuffer import SharedRingBuffer
from phantom_test_utils import make_small_hw

IN_BOX = np.array([0.0, -0.45, 0.25, 0.0, 3.14, 0.0])


# ---------------------------------------------------------------------------
# ring stubs
# ---------------------------------------------------------------------------

class _ArmRing:
    """Arm ring stub with EXPLICIT per-sample timestamps (the whole point of
    items 1 and 5 is that deploy must read the ring by time, not by row)."""

    def __init__(self, ts, ft=None, dof=6):
        self.ts = np.asarray(ts, dtype=np.float64)
        n = len(self.ts)
        self.ft = np.zeros((n, 6)) if ft is None else np.asarray(ft, dtype=np.float64)
        self.dof = dof

    def latest(self, k=1):
        m = min(k, len(self.ts))
        sl = slice(len(self.ts) - m, len(self.ts))
        return self.ts[sl].copy(), {
            "q": np.zeros((m, self.dof)), "qd": np.zeros((m, self.dof)),
            "tcp_pose": np.zeros((m, 6)), "tcp_speed": np.zeros((m, 6)),
            "ft": self.ft[sl].copy(),
        }

    def latest_ts(self):
        return float(self.ts[-1])


class _FixedRing:
    """Ring stub whose newest row is `age` seconds old."""

    def __init__(self, fields, age=0.0, n=1):
        self.fields, self.age, self.n = fields, age, n

    def latest(self, k=1):
        m = min(k, self.n)
        ts = np.array([time.perf_counter() - self.age] * m)
        return ts, {f: np.stack([np.asarray(v)] * m) for f, v in self.fields.items()}

    def latest_ts(self):
        return time.perf_counter() - self.age


def _snapshot_rings(hw, arm_ring):
    return {
        "camera_scene": _FixedRing({"color": np.zeros(hw.cameras.scene.color.hwc,
                                                      np.uint8)}),
        "arm": arm_ring,
        "gripper": _FixedRing({"state": np.array([0.4, 3.0])}),
    }


def _builder(hw, arm_ring):
    return SnapshotBuilder(hw, types.SimpleNamespace(
        rings=_snapshot_rings(hw, arm_ring)), "vision_only")


# ---------------------------------------------------------------------------
# 1. arm-ring freshness (SafetyMonitor + SnapshotBuilder + session liveness)
# ---------------------------------------------------------------------------

@contextmanager
def _rings(hw):
    uid = uuid.uuid4().hex[:8]
    h, w, c = hw.cameras.scene.color.hwc
    rings = {
        "arm": SharedRingBuffer(f"p_arm_{uid}", 16, {
            "ft": ((6,), "float64"), "protective_stop": ((), "uint8")}, create=True),
        "camera_scene": SharedRingBuffer(f"p_cam_{uid}", 4, {
            "color": ((h, w, c), "uint8")}, create=True),
    }
    try:
        yield rings
    finally:
        for ring in rings.values():
            ring.close()


def _push(rings, hw, t_arm, t_cam):
    rings["arm"].push(t_arm, ft=np.zeros(6), protective_stop=np.uint8(0))
    rings["camera_scene"].push(t_cam, color=np.zeros(hw.cameras.scene.color.hwc,
                                                     dtype=np.uint8))


def test_arm_stale_threshold_is_derived_from_the_rtde_rate():
    hw = make_small_hw()
    assert arm_stale_s(hw) == max(0.5, 50.0 / hw.arm.rtde_receive_hz)
    # the rate-derived branch (only reachable below 100 Hz, which the shipped
    # e-series config forbids — 125/250/500 only)
    slow = SimpleNamespace(arm=SimpleNamespace(rtde_receive_hz=20.0))
    assert arm_stale_s(slow) == pytest.approx(2.5)      # 50 samples at 20 Hz


def test_stale_arm_ring_stops_the_episode():
    """A dead RTDE-receive worker leaves the ring frozen: tcp_pose, tcp_speed,
    the F/T window and protective_stop all go stale together and the executor
    keeps servoing off it."""
    hw = make_small_hw()
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        t0 = time.perf_counter()
        _push(rings, hw, t0, t0)
        assert mon.check(t0, IN_BOX).action == SafetyAction.OK
        # camera still fresh, arm frozen -> the arm alone must stop the episode
        t1 = t0 + arm_stale_s(hw) + 0.01
        rings["camera_scene"].push(t1, color=np.zeros(hw.cameras.scene.color.hwc,
                                                      dtype=np.uint8))
        v = mon.check(t1, IN_BOX)
        assert v.action == SafetyAction.STOP_EPISODE
        assert [e.kind for e in v.events] == ["arm_stale"]


def test_recovered_requires_a_fresh_arm_stream():
    """Hysteresis resume must not fire off a frozen arm sample (teleop would
    resume with no live protective-stop or wrench signal at all)."""
    hw = make_small_hw()
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        now = time.perf_counter()
        _push(rings, hw, now - arm_stale_s(hw) - 0.5, now)
        assert not mon.recovered()
        _push(rings, hw, now, now)
        assert mon.recovered()


def test_snapshot_rejects_a_stale_arm_ring():
    hw = make_small_hw()
    t = time.perf_counter()
    stale = _ArmRing(np.linspace(t - arm_stale_s(hw) - 1.0,
                                 t - arm_stale_s(hw) - 0.5, 8))
    with pytest.raises(AssertionError, match="arm ring stale"):
        _builder(hw, stale).build()
    fresh = _ArmRing(np.linspace(t - 0.3, t, 8))
    assert _builder(hw, fresh).build().ur_state.shape == (2 * hw.arm.dof + 14,)


def test_snapshot_empty_arm_ring_is_a_warm_up_assertion():
    """_wait_rings_warm swallows AssertionError/IndexError while the rings fill
    — an empty arm ring must land in that path, not raise something else."""
    from phantom.deploy.runtime import _wait_rings_warm
    hw = make_small_hw()
    with pytest.raises(AssertionError, match="no arm samples yet"):
        _builder(hw, _ArmRing(np.zeros(0))).build()
    with pytest.raises(AssertionError):
        _wait_rings_warm(_builder(hw, _ArmRing(np.zeros(0))), timeout_s=0.1)


class _StubExecutor:
    def __init__(self, accepts=None):
        self.stopped_reason = None
        self.accepts = accepts
        self.submitted = []

    def last_cmd(self):
        return None                       # keeps the stall watchdog out of it

    def request_stop(self, reason):
        if self.stopped_reason is None:
            self.stopped_reason = reason

    def submit(self, plan):
        ok = True if self.accepts is None else self.accepts[len(self.submitted)
                                                            % len(self.accepts)]
        self.submitted.append(plan)
        return ok


class _StubSnapshots:
    def __init__(self, hw):
        self.hw = hw
        self.ur = np.zeros(2 * hw.arm.dof + 12)

    def build(self):
        return SimpleNamespace(t=time.perf_counter(), ur_state=self.ur)


def _plan(tag):
    return SimpleNamespace(latency_s=0.01, gate=0.5, sigma=np.zeros(3),
                           p_evt=np.array([1.0, 0, 0, 0, 0]),
                           actions=np.zeros((4, 7)), diag={"tag": tag}, tag=tag)


class _TaggingPolicy:
    """Records the prev_plan it was handed on each replan."""

    def __init__(self):
        self.seen = []
        self.n = 0

    def replan(self, snap, prev_plan, tcp_pose):
        self.seen.append(None if prev_plan is None else prev_plan.tag)
        self.n += 1
        return _plan(self.n - 1)


def test_planner_stops_the_episode_when_a_sensor_worker_dies():
    hw = make_small_hw()
    calls = {"n": 0}

    def all_alive():
        calls["n"] += 1
        return calls["n"] <= 2           # dies before the 3rd replan

    ex = _StubExecutor()
    loop = PlannerLoop(hw, _TaggingPolicy(), _StubSnapshots(hw), ex,
                       session=SimpleNamespace(all_alive=all_alive))
    loop.run(max_replans=10)
    assert ex.stopped_reason == "worker_died"
    assert len(loop.trace) == 2          # nothing planned after the death


def test_planner_without_a_session_is_unchanged():
    hw = make_small_hw()
    ex = _StubExecutor()
    loop = PlannerLoop(hw, _TaggingPolicy(), _StubSnapshots(hw), ex)
    loop.run(max_replans=3)
    assert ex.stopped_reason is None and len(loop.trace) == 3


def test_run_episode_refuses_to_start_with_a_dead_worker():
    """Fail closed BEFORE the recorder opens a zarr or the arm is driven."""
    rt = object.__new__(DeploymentRuntime)
    rt.hw = make_small_hw()
    rt.session = SimpleNamespace(all_alive=lambda: False, rings={})
    rt.recorder = object()
    res = rt.run_episode(task="whiteboard")
    assert res.episode_path is None
    assert res.stopped_reason == "worker_died"
    assert res.fatal_reason == "worker_died"


# ---------------------------------------------------------------------------
# 2. a timed-out executor/gripper join ends the SESSION
# ---------------------------------------------------------------------------

class _Zombie:
    """Thread that never exits (a gripper worker wedged in a socket
    transaction) — join() returns, is_alive() stays True."""

    def __init__(self):
        self.joins = []

    def join(self, timeout=None):
        self.joins.append(timeout)

    def is_alive(self):
        return True


class _CleanThread(_Zombie):
    def is_alive(self):
        return False


def _executor(hw):
    return ChunkExecutor(hw, SimpleNamespace(servo_stop=lambda: None),
                         SimpleNamespace(), SimpleNamespace())


def test_join_timeout_is_fatal_for_the_deployment_session():
    hw = load_hardware(None)
    ex = _executor(hw)
    ex._thread, ex._grip_thread = _CleanThread(), _Zombie()
    ex.stop()
    assert ex.join_failed is True
    assert ex.stopped_reason == "executor_crash"     # existing signal preserved
    assert fatal_reason(ex) == "executor_join_timeout"


def test_clean_stop_is_not_fatal():
    hw = load_hardware(None)
    ex = _executor(hw)
    ex._thread, ex._grip_thread = _CleanThread(), _CleanThread()
    ex.stop()
    assert ex.join_failed is False
    assert fatal_reason(ex) is None


def test_stop_still_joins_with_the_full_budgets():
    """The joins must stay CONCLUSIVE (2 s executor / 6 s gripper — the gripper
    worker can be inside two 2 s socket timeouts); surfacing the failure must
    not have shortened them."""
    hw = load_hardware(None)
    ex = _executor(hw)
    ex._thread, ex._grip_thread = _CleanThread(), _CleanThread()
    ex.stop()
    assert ex._thread.joins == [2.0]
    assert ex._grip_thread.joins == [6.0]


def test_a_plain_safety_stop_is_not_session_fatal():
    """recover_control() handles these between episodes — only unfixable
    conditions may end the process."""
    ex = SimpleNamespace(join_failed=False, stopped_reason="protective_stop")
    assert fatal_reason(ex) is None
    assert fatal_reason(SimpleNamespace(join_failed=False,
                                        stopped_reason="worker_died")) == "worker_died"


def test_run_deploy_exits_instead_of_starting_another_episode():
    from phantom.scripts import run_deploy as RD
    src = inspect.getsource(RD.main)
    assert "res.fatal_reason" in src and "return 5" in src
    # the fatal check must come AFTER the label prompt: the crashed episode is
    # exactly the one worth labelling before the process dies
    assert src.index("outcome? [s]uccess") < src.rindex("res.fatal_reason")


# ---------------------------------------------------------------------------
# 3. EMA is the deploy default
# ---------------------------------------------------------------------------

def _parse(*argv):
    from phantom.scripts.run_deploy import build_parser
    return build_parser().parse_args(["--system", "teacher", "--task", "x", *argv])


def test_ema_is_the_default_and_the_old_flag_still_parses():
    assert _parse().ema is True                 # bare launch == what eval loads
    assert _parse("--ema").ema is True          # existing GO scripts
    assert _parse("--no-ema").ema is False
    assert _parse("--raw").ema is False
    with pytest.raises(SystemExit):
        _parse("--ema", "--no-ema")


@pytest.fixture
def _stub_build(monkeypatch):
    from phantom.scripts import run_deploy as RD
    cap: dict = {}
    monkeypatch.setattr(RD.torch, "load", lambda *a, **k: {"ema": {"w": 1}})
    monkeypatch.setattr(RD, "build_model",
                        lambda *a, **k: SimpleNamespace(rf=SimpleNamespace(net=object())))

    def _load(path, model, *, hw, load_ema=False, payload=None, **kw):
        cap["load_ema"] = load_ema
        return payload or {}

    monkeypatch.setattr(RD.C, "load_phantom_checkpoint", _load)
    monkeypatch.setattr(RD, "PhantomPolicy", lambda *a, **k: SimpleNamespace())
    return cap


def _pol_args(**kw):
    base = dict(system="teacher", ckpt="teacher_003000.pt", tiny=True,
                device="cpu", nfe=None, drop_video=False, text="", task="t")
    base.update(kw)
    return SimpleNamespace(**base)


def test_build_policy_loads_ema_by_default(_stub_build, caplog):
    from phantom.scripts import run_deploy as RD
    with caplog.at_level(logging.INFO, logger="run_deploy"):
        RD.build_policy(_pol_args(), make_small_hw(), None)
    assert _stub_build["load_ema"] is True
    # the operator must be able to read which artifact went to the rig
    assert any("EMA" in r.getMessage() for r in caplog.records)


def test_build_policy_honours_the_raw_escape_hatch(_stub_build):
    from phantom.scripts import run_deploy as RD
    RD.build_policy(_pol_args(ema=False), make_small_hw(), None)
    assert _stub_build["load_ema"] is False


# ---------------------------------------------------------------------------
# 4. feedback state advances only on ACCEPTED plans
# ---------------------------------------------------------------------------

def test_prev_plan_advances_only_when_the_executor_accepted_it():
    """A plan rejected for not covering replan_min_lead_s is never commanded;
    feeding it back as prev_chunk conditions the next replan on motion that
    never happened."""
    hw = make_small_hw()
    ex = _StubExecutor(accepts=[True, False, False, True])
    pol = _TaggingPolicy()
    loop = PlannerLoop(hw, pol, _StubSnapshots(hw), ex)
    loop.run(max_replans=5)
    # plans 0 and 3 accepted; 1, 2 and 4 rejected -> prev stays at the last
    # accepted plan's tag
    assert pol.seen == [None, 0, 0, 0, 3]
    assert [t["accepted"] for t in loop.trace] == [True, False, False, True, True]


def test_all_accepted_keeps_the_previous_behaviour():
    hw = make_small_hw()
    pol = _TaggingPolicy()
    loop = PlannerLoop(hw, pol, _StubSnapshots(hw), _StubExecutor())
    loop.run(max_replans=4)
    assert pol.seen == [None, 0, 1, 2]


def test_the_executed_history_upgrade_is_marked_deferred():
    """The larger change (prev_chunk = executed action grid, as
    WindowSampler.sample() builds it) is deliberately deferred — the pointer
    must survive so it is not re-found as a fresh bug."""
    src = inspect.getsource(PlannerLoop.run)
    assert "DEFERRED" in src and "windows.py" in src


# ---------------------------------------------------------------------------
# 5. wrist F/T window = a DURATION on the training interp grid
# ---------------------------------------------------------------------------

def _ramp_ring(hw, t_end, n, span_s):
    """Arm ring whose ft channels encode each sample's own age: ft[i, k] =
    k + (ts[i] - t_end). A correct window is then the interp GRID itself."""
    ts = np.linspace(t_end - span_s, t_end, n)
    ft = (ts - t_end)[:, None] + np.arange(6)[None, :]
    return _ArmRing(ts, ft)


def test_wrist_window_is_the_training_linspace_interp_grid():
    hw = make_small_hw()                  # window_s 0.1 s
    L, W = hw.wrist_ft.window_len, hw.wrist_ft.window_s
    t_end = time.perf_counter()
    # arm ring spans 3.5x the window at a rate unrelated to wrist_ft.rate_hz —
    # the row-slice version returned the last L rows (0.35 s of history here)
    snap = _builder(hw, _ramp_ring(hw, t_end, n=4 * L, span_s=3.5 * W)).build()
    expected = np.linspace(-W, 0.0, L)[:, None] + np.arange(6)[None, :]
    assert snap.wrist_window.shape == (L, 6)
    assert snap.wrist_window.dtype == np.float32
    assert np.allclose(snap.wrist_window, expected, atol=1e-5)


def test_wrist_window_spans_window_s_at_any_ring_rate():
    """The pre-fix window was a ROW COUNT: on a cb3 (arm ring at 125 Hz,
    wrist_ft.rate_hz 500 -> window_len 4x the rows the window holds) it fed the
    WristTCN four times the trained duration."""
    hw = make_small_hw()
    L, W = hw.wrist_ft.window_len, hw.wrist_ft.window_s
    t_end = time.perf_counter()
    slow = _builder(hw, _ramp_ring(hw, t_end, n=L // 4, span_s=4 * W)).build()
    # oldest grid point is exactly window_s back, whatever the ring rate is
    assert slow.wrist_window[0, 0] == pytest.approx(-W, abs=1e-5)
    assert slow.wrist_window[-1, 0] == pytest.approx(0.0, abs=1e-5)


def test_wrist_window_warm_up_pads_with_the_oldest_sample():
    """Fewer samples than the window spans: np.interp clamps to ft[0] below the
    first timestamp — the same leading repeat-pad the row-slice version did, so
    _wait_rings_warm still sees a buildable snapshot."""
    hw = make_small_hw()
    L = hw.wrist_ft.window_len
    t_end = time.perf_counter()
    one = _ArmRing(np.array([t_end]), np.arange(6, dtype=np.float64)[None, :])
    win = _builder(hw, one).build().wrist_window
    assert win.shape == (L, 6) and win.dtype == np.float32
    assert np.allclose(win, np.arange(6)[None, :])


def test_wrist_window_anchors_on_the_newest_arm_sample_not_now():
    """Anchoring at t_now would flat-extrapolate the window's TAIL by the arm
    sample's age (np.interp clamps above the last timestamp)."""
    hw = make_small_hw()
    L, W = hw.wrist_ft.window_len, hw.wrist_ft.window_s
    t_end = time.perf_counter() - 0.2          # newest sample is 0.2 s old
    snap = _builder(hw, _ramp_ring(hw, t_end, n=4 * L, span_s=3.5 * W)).build()
    expected = np.linspace(-W, 0.0, L)[:, None] + np.arange(6)[None, :]
    assert np.allclose(snap.wrist_window, expected, atol=1e-5)
    # a t_now anchor would repeat the newest value across the last 0.2 s
    assert not np.allclose(snap.wrist_window[-1], snap.wrist_window[-2])


# ---------------------------------------------------------------------------
# mock parity: the dry run must still complete two back-to-back episodes
# ---------------------------------------------------------------------------

class _DryRunPolicy:
    """Constant crawl; records the wrist window it was handed."""

    def __init__(self, hw):
        self.hw = hw
        self.windows = []

    def reset_episode(self):
        pass

    def replan(self, snap, prev_plan, tcp_pose):
        from phantom.inference.policy import Plan
        hw = self.hw
        self.windows.append(snap.wrist_window.shape)
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


def test_mock_dry_run_still_completes_two_episodes(tmp_path):
    """End-to-end over the REAL SensorSession + ring buffers on mock drivers:
    the new arm-freshness asserts and the resampled wrist window must not trip
    on the mock arm poller, and a clean episode must stay non-fatal so the next
    one starts."""
    hw = make_small_hw()
    pol = _DryRunPolicy(hw)
    with DeploymentRuntime(hw, pol, mode="vision_only", out_root=tmp_path) as rt:
        first = rt.run_episode(task="whiteboard", max_replans=3)
        second = rt.run_episode(task="whiteboard", max_replans=2)
    assert (first.stopped_reason, first.fatal_reason) == (None, None)
    assert (second.stopped_reason, second.fatal_reason) == (None, None)
    assert (first.n_replans, second.n_replans) == (3, 2)
    assert set(pol.windows) == {(hw.wrist_ft.window_len, 6)}


def test_camera_guard_still_independent_of_the_arm_guard():
    hw = make_small_hw()
    t = time.perf_counter()
    rings = _snapshot_rings(hw, _ArmRing(np.linspace(t - 0.05, t, 8)))
    rings["camera_scene"] = _FixedRing(
        {"color": np.zeros(hw.cameras.scene.color.hwc, np.uint8)},
        age=camera_stale_s(hw) + 0.5)
    sb = SnapshotBuilder(hw, types.SimpleNamespace(rings=rings), "vision_only")
    with pytest.raises(AssertionError, match="camera_scene stale"):
        sb.build()
