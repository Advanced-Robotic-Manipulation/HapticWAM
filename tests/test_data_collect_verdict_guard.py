"""Two label-safety regressions in the collection tool (rig session, 2026-08).

1. An episode still awaiting a success/fail verdict was shipped as an ordinary
   full-weight demo when the session ended: "Stop episode" finalizes the take
   on disk with success=None, and every exit path (quit, arm fault) left it
   that way — is_failure_demo() is False for success=None, so WindowSampler
   gave the rejected take action_weight 1.0.
2. Panel buttons pressed while no session was running stayed in the queue and
   fired on the first tick of the NEXT session (a phantom episode recorded
   during the engage glide, or an instant quit of a freshly brought-up rig).
"""

import threading
import time
import types

import numpy as np
import pytest

from phantom_test_utils import make_small_hw
from phantom.data.episode_store import list_episodes
from phantom.data.schema import EpisodeMeta
from phantom.data_collect.config import CollectConfig, SafeguardConfig
from phantom.data_collect.panel import (CollectPanel, CollectPanelState,
                                        CollectRunner)
from phantom.data_collect.safeguard import ArmGuard, TactileSafeguard
from phantom.data_collect.session import CollectApp

from test_data_collect_safeguard import FakeRing, _rings as tactile_rings
from test_data_collect_session import (StubEcho, StubPilot, StubRecorder,
                                       StubSession, StubStreamer, _settle)


# --------------------------------------------------------------------- rig
@pytest.fixture
def env():
    """The session loop on stubs, same wiring as test_data_collect_session —
    but the teardown never forces a quit, so a test can observe the loop
    REFUSING to end while an episode is unjudged."""
    hw = make_small_hw(control={"action_rate_hz": 50.0})
    cc = CollectConfig.model_validate({
        "ports": {"dmtac_left": 0, "dmtac_right": 1},
        "storage": {"external_drive": "Z:/nowhere"}})
    state = CollectPanelState()
    panel = CollectPanel(state)          # NOT started: queue + state offline
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
    yield types.SimpleNamespace(panel=panel, state=state, recorder=recorder,
                                streamer=streamer, thread=thread)
    for _ in range(3):                   # confirm-to-force, whatever the state
        if not thread.is_alive():
            break
        panel.push_button("quit")
        thread.join(1.0)
    assert not thread.is_alive()


def _stop_with_pending(env):
    """Record one episode and stop it -> finalized on disk, no verdict yet."""
    env.panel.push_button("start_stop")
    assert _settle(lambda: env.recorder.recording)
    env.panel.push_button("start_stop")
    assert _settle(
        lambda: env.state.snapshot()["episode_phase"] == "awaiting_verdict")
    assert env.recorder.stopped[-1][0] is None          # success=None on disk


# ----------------------------------------------- 1. no verdict, no shipping
def test_quit_is_refused_while_an_episode_awaits_a_verdict(env):
    """The bug: 'End session (auto-offload)' next to a pending verdict ended
    the session and offloaded the unjudged take as a normal demo."""
    _stop_with_pending(env)
    env.panel.push_button("quit")
    time.sleep(0.4)
    assert env.thread.is_alive(), "session ended with an unjudged episode"
    snap = env.state.snapshot()
    assert "no verdict" in snap["error"]
    assert snap["episode_phase"] == "awaiting_verdict"   # verdict still offered
    assert snap["phase"] == "running"                    # not stuck 'stopping'
    assert env.recorder.relabels == []                   # nothing filed yet


def test_verdict_after_a_refused_quit_clears_the_banner_and_ends(env):
    _stop_with_pending(env)
    env.panel.push_button("quit")
    assert _settle(lambda: "no verdict" in env.state.snapshot()["error"])
    env.panel.push_button("fail")
    assert _settle(lambda: env.recorder.relabels != [])
    assert env.recorder.relabels[-1][1] is False
    assert _settle(lambda: env.state.snapshot()["error"] == "")
    # the refusal is not sticky: the next End session ends the session
    env.panel.push_button("quit")
    env.thread.join(3.0)
    assert not env.thread.is_alive()


def test_confirmed_quit_files_the_episode_as_unlabeled(env):
    """Second press wins — the session is never strandable on the rig — but
    the take is filed as NOT training-ready, never as a plain demo."""
    _stop_with_pending(env)
    env.panel.push_button("quit")
    assert _settle(lambda: "no verdict" in env.state.snapshot()["error"])
    env.panel.push_button("quit")
    env.thread.join(3.0)
    assert not env.thread.is_alive()
    kw = env.recorder.relabel_kw[-1]
    assert kw["success"] is None            # nobody judged it: do not invent one
    assert kw["status"] == "aborted"        # skipped by list_episodes/uploader
    assert kw["tags"] == ["unlabeled"]
    assert not kw["discard"]                # the data stays on disk
    ep = env.state.snapshot()["episodes"][-1]
    assert ep["outcome"] == "unjudged" and ep["tags"] == ["unlabeled"]


def test_verdict_and_quit_in_the_same_tick_keeps_the_verdict(env):
    """Both cards are on screen at once: a Success click immediately followed
    by End session lands in ONE pop. The quit branch runs first — it must not
    throw the operator's judgment away."""
    _stop_with_pending(env)
    env.panel.push_button("success")
    env.panel.push_button("quit")            # same 100 ms pop as the verdict
    env.thread.join(3.0)
    assert not env.thread.is_alive()         # the quit is honoured, one tick late
    assert env.recorder.relabels[-1][1] is True          # success verdict kept
    assert env.recorder.relabel_kw[-1]["status"] is None  # not filed unlabeled
    assert env.state.snapshot()["episodes"][-1]["outcome"] == "success"


def test_arm_fault_files_a_pending_episode_as_unlabeled(env):
    """A motion-thread death cannot wait for a verdict; the pending take must
    still not reach the auto-offload as a full-weight demo."""
    _stop_with_pending(env)
    env.streamer.error = "servoJ rejected"
    assert _settle(lambda: not env.thread.is_alive())
    kw = env.recorder.relabel_kw[-1]
    assert kw["status"] == "aborted" and kw["tags"] == ["unlabeled"]
    assert kw["success"] is None
    assert env.state.snapshot()["episodes"][-1]["outcome"] == "unjudged"


def test_unlabeled_episode_is_not_training_ready(tmp_path):
    """End to end on a real meta.json: what relabel() writes must actually
    drop out of list_episodes (the sampler's only source of episodes)."""
    from phantom.recording.recorder import EpisodeRecorder
    r = EpisodeRecorder.__new__(EpisodeRecorder)
    r.out_root = tmp_path
    for name, kw in (("ep_judged", dict(success=True)),
                     ("ep_unjudged", dict(status="aborted",
                                          tags=["unlabeled"],
                                          notes="no operator verdict"))):
        (tmp_path / name).mkdir()
        EpisodeMeta(task="t", status="finalized").save(
            tmp_path / name / "meta.json")
        r.relabel(tmp_path / name, **kw)

    listed = [p.name for p in list_episodes(tmp_path)]
    assert listed == ["ep_judged"]
    meta = EpisodeMeta.load(tmp_path / "ep_unjudged" / "meta.json")
    assert meta.status == "aborted" and meta.tags == ["unlabeled"]
    assert meta.success is None
    assert (tmp_path / "ep_unjudged").exists()   # data kept, just not listed


# --------------------------------------------- 2. stale buttons never leak
class _FakeApp:
    """Just enough app for CollectRunner: run_session records what the
    session's FIRST pop_buttons() would have seen."""

    def __init__(self, panel):
        self.panel = panel
        self.cc = types.SimpleNamespace(default_mode="full")
        self.started = threading.Event()
        self.release = threading.Event()
        self.first_pop = None

    def run_session(self, s):
        self.first_pop = self.panel.pop_buttons()
        self.started.set()
        self.release.wait(3.0)

    def offload(self, staging=None):
        pass


@pytest.fixture
def runner_env():
    panel = CollectPanel(CollectPanelState())
    app = _FakeApp(panel)
    runner = CollectRunner(app, panel)
    panel.runner = runner
    yield types.SimpleNamespace(panel=panel, app=app, runner=runner)
    app.release.set()
    if runner._thread is not None:
        runner._thread.join(3.0)


def test_buttons_pressed_with_no_session_are_dropped(runner_env):
    """The bug: POST /api/cmd queued unconditionally, and the queue's only
    consumer is the next session's loop."""
    for btn in ("start_stop", "quit", "abort"):
        assert runner_env.panel.push_button(btn) is False
    assert runner_env.panel.pop_buttons() == {}


def test_queue_is_drained_at_session_start(runner_env):
    """Belt and braces: whatever slipped into the queue before this session
    (during a previous session's teardown) must not fire on tick one."""
    runner_env.panel._cmds.put("start_stop")     # bypass the door check
    assert runner_env.runner.start_session({"task": "demo"}) is None
    assert runner_env.app.started.wait(3.0)
    assert runner_env.app.first_pop == {}, "stale button leaked into a session"


def test_buttons_are_accepted_while_a_session_runs(runner_env):
    assert runner_env.runner.start_session({"task": "demo"}) is None
    assert runner_env.app.started.wait(3.0)
    assert runner_env.panel.push_button("start_stop") is True
    assert runner_env.panel.pop_buttons() == {"start_stop": True}
