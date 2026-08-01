"""collect/session.py — the main loop: episode lifecycle, safeguard flow,
absolute action logging, lite stream filter. Everything stubbed (no hardware,
no processes) — the loop runs in a thread and is driven via panel buttons."""

import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from phantom_test_utils import make_small_hw
from phantom.data.schema import STREAM_ACTIONS_ABS, tactile_stream

from phantom.data_collect.config import CollectConfig, SafeguardConfig
from phantom.data_collect.panel import CollectPanel, CollectPanelState
from phantom.data_collect.safeguard import ArmGuard, TactileSafeguard
from phantom.data_collect.session import CollectApp, lite_stream_filter
from phantom.data_collect.teleop import TrackPhase
from test_data_collect_safeguard import FakeRing, _rings as tactile_rings


# ---------------------------------------------------------------------- stubs
class StubEcho:
    def __init__(self):
        self.gripper = 0.3

    def poll(self):
        return types.SimpleNamespace(gripper=self.gripper, buttons={})


class StubStreamer:
    def __init__(self):
        self.phase = TrackPhase.TRACK
        self.in_workspace_hold = False
        self.error = None
        self.holds = 0
        self.resumes = 0
        self.last_cmd = np.arange(6, dtype=np.float64)

    def hold(self):
        self.holds += 1

    def resume(self):
        self.resumes += 1


class StubPilot:
    def __init__(self):
        self.error = None
        self.suspended = False
        self.last_sent = 0.3
        self.opens = 0
        self.releases = 0
        self.targets = []

    def set_target(self, v):
        self.targets.append(v)

    def open_now(self):
        self.opens += 1
        self.suspended = True

    def release(self):
        self.releases += 1
        self.suspended = False


class StubRecorder:
    def __init__(self):
        self.started = []
        self.stopped = []          # (success, notes, abort)
        self.relabels = []         # (path, success, discard)
        self.actions = []          # (stream, action)
        self.recording = False

    def start(self, meta, name):
        self.started.append(name)
        self.recording = True
        return Path(name)

    def stop(self, *, success=None, notes="", abort=False):
        if not self.recording:
            return None
        self.recording = False
        self.stopped.append((success, notes, abort))
        return Path(self.started[-1]) if self.started else None

    def relabel(self, path, *, success=None, discard=False, notes=""):
        self.relabels.append((Path(path), success, discard))

    def record_action(self, t, action, stream="actions"):
        self.actions.append((stream, np.asarray(action)))


class StubSession:
    def __init__(self, rings):
        self.rings = rings
        self.alive = True

    def all_alive(self):
        return self.alive


# ---------------------------------------------------------------------- rig
@pytest.fixture
def env():
    hw = make_small_hw(control={"action_rate_hz": 50.0})
    cc = CollectConfig.model_validate({
        "ports": {"dmtac_left": 0, "dmtac_right": 1},
        "storage": {"external_drive": "Z:/nowhere"}})
    state = CollectPanelState()
    panel = CollectPanel(state)      # NOT started: queue + state work offline
    app = CollectApp(cc, hw, panel)

    rings = tactile_rings(hw, force=0.0)
    arm = FakeRing()
    arm.push(time.perf_counter(), ft=np.zeros(6), protective_stop=np.uint8(0),
             q=np.zeros(6), tcp_pose=np.zeros(6), tcp_speed=np.zeros(6))
    rings["arm"] = arm

    echo, streamer, pilot = StubEcho(), StubStreamer(), StubPilot()
    recorder = StubRecorder()
    session = StubSession(rings)
    safeguard = TactileSafeguard(
        hw, rings, SafeguardConfig(force_limit_n=4.0),
        on_trip=lambda info: app._trip_now(streamer, pilot, info))
    guard = ArmGuard(hw, rings)
    rig = types.SimpleNamespace(arm=types.SimpleNamespace(zero_ft=lambda: None))

    s = types.SimpleNamespace(task="demo", text="", operator="op",
                              mode="full", target_episodes=0)
    thread = threading.Thread(
        target=app._loop,
        args=(s, rig, session, recorder, echo, streamer, pilot, guard,
              safeguard, None),
        daemon=True)
    thread.start()
    yield types.SimpleNamespace(hw=hw, app=app, panel=panel, state=state,
                                echo=echo, streamer=streamer, pilot=pilot,
                                recorder=recorder, safeguard=safeguard,
                                session=session, rings=rings, thread=thread)
    panel.push_button("quit")
    thread.join(3.0)
    assert not thread.is_alive()


def _settle(pred, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------- tests
def test_episode_lifecycle_with_absolute_actions(env):
    env.panel.push_button("start_stop")
    assert _settle(lambda: env.recorder.recording)
    assert env.recorder.started[0].startswith("ep_demo_")
    assert _settle(lambda: env.state.snapshot()["episode_phase"] == "recording")
    # absolute actions flow while recording: [q_target(6) | gripper(1)]
    assert _settle(lambda: len(env.recorder.actions) >= 3)
    stream, action = env.recorder.actions[0]
    assert stream == STREAM_ACTIONS_ABS
    assert action.shape == (7,)
    assert np.allclose(action[:6], env.streamer.last_cmd)   # absolute q target
    assert action[6] == pytest.approx(env.pilot.last_sent)  # absolute gripper
    # STOP -> episode finalized WITHOUT a verdict, now awaiting one
    env.panel.push_button("start_stop")
    assert _settle(lambda: not env.recorder.recording)
    assert env.recorder.stopped[-1][0] is None              # no verdict yet
    assert _settle(
        lambda: env.state.snapshot()["episode_phase"] == "awaiting_verdict")
    assert env.state.snapshot()["episodes"] == []           # not banked yet
    # verdict applied AFTER the stop
    env.panel.push_button("success")
    assert _settle(lambda: env.recorder.relabels != [])
    assert env.recorder.relabels[-1][1] is True             # success verdict
    assert _settle(lambda: env.state.snapshot()["episodes"] != [])
    assert env.state.snapshot()["episodes"][-1]["outcome"] == "success"
    assert _settle(lambda: env.state.snapshot()["episode_phase"] == "idle")


def test_episode_fail_verdict_after_stop(env):
    env.panel.push_button("start_stop")
    assert _settle(lambda: env.recorder.recording)
    env.panel.push_button("start_stop")
    assert _settle(
        lambda: env.state.snapshot()["episode_phase"] == "awaiting_verdict")
    env.panel.push_button("fail")
    assert _settle(lambda: env.recorder.relabels != [])
    assert env.recorder.relabels[-1][1] is False            # fail verdict
    assert env.state.snapshot()["episodes"][-1]["outcome"] == "fail"
    assert _settle(lambda: env.state.snapshot()["episode_phase"] == "idle")


def test_episode_discard_verdict_after_stop(env):
    env.panel.push_button("start_stop")
    assert _settle(lambda: env.recorder.recording)
    env.panel.push_button("start_stop")
    assert _settle(
        lambda: env.state.snapshot()["episode_phase"] == "awaiting_verdict")
    env.panel.push_button("abort")            # discard the stopped episode
    assert _settle(lambda: env.recorder.relabels != [])
    assert env.recorder.relabels[-1][2] is True             # discard=True
    assert env.state.snapshot()["episodes"][-1]["outcome"] == "discarded"
    assert _settle(lambda: env.state.snapshot()["episode_phase"] == "idle")


def test_gripper_not_fed_by_the_loop(env):
    """The pilot pulls the leader at device rate — the 10 Hz session loop
    must NOT be in the gripper path (review finding: +100 ms latency)."""
    env.echo.gripper = 0.66
    time.sleep(0.2)
    assert env.pilot.targets == []


def test_safeguard_trip_flow(env):
    env.panel.push_button("start_stop")
    assert _settle(lambda: env.recorder.recording)
    # hot sensor -> trip (the safeguard thread is simulated by check_once)
    name = env.hw.tactile.sensors[0].name
    ring = env.rings[f"tactile_{name}"]
    fields = ring.data["fields_ds"][0]
    ring.push(time.perf_counter(), fields_ds=fields,
              wrench=np.array([9, 0, 0, 0, 0, 0], np.float32),
              area=np.float32(0))
    env.safeguard.check_once()
    # time-critical part ran on the caller thread
    assert env.streamer.holds == 1 and env.pilot.opens == 1
    # loop bookkeeping: episode saved as failure, panel warns
    assert _settle(lambda: not env.recorder.recording)
    success, notes, abort = env.recorder.stopped[-1]
    assert success is False and "safeguard" in notes and not abort
    snap = env.state.snapshot()
    assert snap["safeguard_tripped"] and "safeguard" in snap["safeguard_msg"].lower()
    assert snap["episodes"][-1]["outcome"] == "safeguard"

    # while tripped: start_stop is blocked
    n_started = len(env.recorder.started)
    env.panel.push_button("start_stop")
    time.sleep(0.2)
    assert len(env.recorder.started) == n_started

    # cool the sensor, operator clicks RESUME — everything continues
    ring.push(time.perf_counter(), fields_ds=fields,
              wrench=np.zeros(6, np.float32), area=np.float32(0))
    env.panel.push_button("resume_safeguard")
    assert _settle(lambda: env.safeguard.tripped is None)
    assert _settle(lambda: env.pilot.releases == 1 and env.streamer.resumes == 1)
    assert _settle(lambda: not env.state.snapshot()["safeguard_tripped"])
    env.panel.push_button("start_stop")
    assert _settle(lambda: len(env.recorder.started) == n_started + 1)


def test_abort_discards(env):
    env.panel.push_button("start_stop")
    assert _settle(lambda: env.recorder.recording)
    env.panel.push_button("abort")
    assert _settle(lambda: not env.recorder.recording)
    assert env.recorder.stopped[-1][2] is True              # abort=True
    assert env.state.snapshot()["episodes"][-1]["outcome"] == "discarded"


def test_arm_guard_pstop_holds_and_autoresumes(env):
    env.panel.push_button("start_stop")
    assert _settle(lambda: env.recorder.recording)
    env.rings["arm"].push(time.perf_counter(), ft=np.zeros(6),
                          protective_stop=np.uint8(1), q=np.zeros(6),
                          tcp_pose=np.zeros(6), tcp_speed=np.zeros(6))
    assert _settle(lambda: env.state.snapshot()["arm_hold"] == "pstop")
    assert env.streamer.holds >= 1
    assert not env.recorder.recording
    assert "arm_guard" in env.recorder.stopped[-1][1]
    # pendant cleared -> auto-resume (reference behaviour for ARM events)
    env.rings["arm"].push(time.perf_counter(), ft=np.zeros(6),
                          protective_stop=np.uint8(0), q=np.zeros(6),
                          tcp_pose=np.zeros(6), tcp_speed=np.zeros(6))
    assert _settle(lambda: env.state.snapshot()["arm_hold"] == "")
    assert env.streamer.resumes >= 1


def test_disable_while_tripped_releases_motion(env):
    """Review finding: disabling the safeguard cleared the latch but left the
    streamer held and the pilot suspended — rig frozen with no way out."""
    env.panel.push_button("start_stop")
    assert _settle(lambda: env.recorder.recording)
    name = env.hw.tactile.sensors[0].name
    ring = env.rings[f"tactile_{name}"]
    fields = ring.data["fields_ds"][0]
    ring.push(time.perf_counter(), fields_ds=fields,
              wrench=np.array([9, 0, 0, 0, 0, 0], np.float32),
              area=np.float32(0))
    env.safeguard.check_once()
    assert _settle(lambda: env.state.snapshot()["safeguard_tripped"])
    env.safeguard.configure(enabled=False)       # panel toggle
    assert _settle(lambda: not env.state.snapshot()["safeguard_tripped"])
    assert _settle(lambda: env.pilot.releases >= 1 and env.streamer.resumes >= 1)


def test_worker_death_mid_episode_fails_the_episode(env):
    """Review finding: a dead worker mid-episode must not let a truncated
    episode be finalized as a normal save."""
    env.panel.push_button("start_stop")
    assert _settle(lambda: env.recorder.recording)
    env.session.alive = False
    assert _settle(lambda: not env.thread.is_alive())     # session fails loudly
    success, notes, abort = env.recorder.stopped[-1]
    assert success is False and "worker_died" in notes
    assert env.state.snapshot()["episodes"][-1]["outcome"] == "worker_died"


# ---------------------------------------------------------------------- misc
def test_lite_stream_filter():
    keep = ["arm_q", "arm_ft", "gripper", "camera_scene_color",
            STREAM_ACTIONS_ABS,
            tactile_stream("left", "wrench"), tactile_stream("right", "area")]
    drop = [tactile_stream("left", "fields_ds"),
            tactile_stream("right", "keyframes"),
            tactile_stream("left", "infer_img"),
            tactile_stream("right", "raw_img")]
    assert all(lite_stream_filter(s) for s in keep)
    assert not any(lite_stream_filter(s) for s in drop)
