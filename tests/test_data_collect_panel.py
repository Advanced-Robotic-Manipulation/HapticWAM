"""collect/panel.py — endpoints, button whitelist, safeguard POST, SSE shape."""

import json
import threading
import time
import types
import urllib.request

import pytest

from phantom.data_collect.panel import (BUTTONS, CollectPanel,
                                        CollectPanelState, CollectRunner)


def _wait(pred, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


class StubSafeguard:
    def __init__(self):
        self.enabled = True
        self.force_limit_n = 4.0
        self.depth_limit = 0.5
        self.calls = []

    def configure(self, **kw):
        self.calls.append(kw)
        for k, v in kw.items():
            setattr(self, k, v)
        if self.force_limit_n <= 0:
            raise ValueError("force_limit_n must be > 0")


class StubRunner:
    def __init__(self):
        self.started = []
        self.stopped = 0
        self.offloads = 0

    def start_session(self, cfg):
        self.started.append(cfg)
        return None if cfg.get("task") else "task name: letters/digits/_-"

    def stop_session(self):
        self.stopped += 1

    def start_offload(self):
        self.offloads += 1
        return None


@pytest.fixture
def panel():
    sg = StubSafeguard()
    runner = StubRunner()
    p = CollectPanel(CollectPanelState(), host="127.0.0.1", port=0,
                     runner=runner, safeguard_provider=lambda: sg,
                     setup_provider=lambda: {"rig_name": "test"})
    p.start()
    yield p, sg, runner
    p.stop()


def _get(p, path):
    with urllib.request.urlopen(f"{p.url}{path}", timeout=5) as r:
        return json.loads(r.read())


def _post(p, path, body):
    req = urllib.request.Request(
        f"{p.url}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_status_and_setup(panel):
    p, _, _ = panel
    s = _get(p, "/api/status")
    assert s["phase"] == "setup" and "safeguard_enabled" in s
    assert _get(p, "/api/setup")["rig_name"] == "test"


def test_buttons_whitelisted_and_queued(panel):
    p, _, _ = panel
    for b in BUTTONS:
        code, r = _post(p, "/api/cmd", {"button": b})
        assert code == 200 and r["ok"]
    assert set(p.pop_buttons()) == set(BUTTONS)
    code, r = _post(p, "/api/cmd", {"button": "rm_rf_slash"})
    assert code == 400 and "unknown" in r["error"]
    assert p.pop_buttons() == {}


def test_safeguard_post_configures_live_session(panel):
    p, sg, _ = panel
    code, r = _post(p, "/api/safeguard",
                    {"enabled": False, "force_limit_n": 2.5})
    assert code == 200 and r["live"]
    assert sg.calls == [{"enabled": False, "force_limit_n": 2.5}]
    assert p.state.snapshot()["safeguard_force_limit_n"] == 2.5
    # invalid value -> 400, not a crash
    code, r = _post(p, "/api/safeguard", {"force_limit_n": "abc"})
    assert code == 400
    code, r = _post(p, "/api/safeguard", {"force_limit_n": -3})
    assert code == 400


def test_safeguard_post_without_session_stashes_values(panel):
    p, _, _ = panel
    p.safeguard_provider = lambda: None
    code, r = _post(p, "/api/safeguard", {"force_limit_n": 7.0})
    assert code == 200 and not r["live"]
    assert p.state.snapshot()["safeguard_force_limit_n"] == 7.0


def test_session_and_offload_endpoints(panel):
    p, _, runner = panel
    code, _ = _post(p, "/api/session/start", {"task": "demo", "mode": "lite"})
    assert code == 200 and runner.started[-1]["task"] == "demo"
    code, r = _post(p, "/api/session/start", {"task": ""})
    assert code == 409 and "task" in r["error"]
    code, _ = _post(p, "/api/session/stop", {})
    assert code == 200 and runner.stopped == 1
    code, _ = _post(p, "/api/offload", {})
    assert code == 200 and runner.offloads == 1


def test_page_serves(panel):
    p, _, _ = panel
    with urllib.request.urlopen(f"{p.url}/", timeout=5) as r:
        html = r.read().decode()
    assert "SAFEGUARD" in html and "RESUME COLLECTION" in html
    assert "rerun" in html and "Offload" in html
    # the session button's new busy state (spinner + live step line)
    assert "busyBtn" in html and "busyDetail" in html and "spinner" in html


def test_busy_detail_in_status(panel):
    p, _, _ = panel
    s = _get(p, "/api/status")
    assert "busy_detail" in s and s["busy_detail"] == ""


def test_page_has_sse_reconnect_resilience(panel):
    p, _, _ = panel
    with urllib.request.urlopen(f"{p.url}/", timeout=5) as r:
        html = r.read().decode()
    # auto-reconnect with backoff + full resync that drops stale pending state
    assert "es.onopen" in html and "setTimeout(listen,backoff)" in html
    assert "backoff=Math.min(backoff*2,5000)" in html
    assert "pending='';epPending='';resync()" in html
    # disconnected indicator + button lockout
    assert 'id="connPill"' in html and "server reconnecting" in html
    assert "body.disconnected button" in html
    # optimistic-spinner timeout safety
    assert "Date.now()-pendingT>10000" in html


def test_episode_phase_in_status_defaults_idle(panel):
    p, _, _ = panel
    s = _get(p, "/api/status")
    assert s["episode_phase"] == "idle" and s["episode_detail"] == ""


def test_episode_buttons_state_machine_markup(panel):
    p, _, _ = panel
    with urllib.request.urlopen(f"{p.url}/", timeout=5) as r:
        html = r.read().decode()
    # every visibility group the render() state machine toggles must exist
    for el in ("epStartBtn", "epRecCtl", "epStopBtn", "epBusyBtn",
               "epVerdict", "epDetail"):
        assert f'id="{el}"' in html, el
    # English labels only
    assert "Start episode" in html and "Awaiting" not in html
    assert "awaiting_verdict" in html   # the state-machine JS


def test_viewer_mode_in_status_defaults_web(panel):
    p, _, _ = panel
    assert _get(p, "/api/status")["viewer_mode"] == "web"


def test_viewer_mode_native_is_seeded_and_snapshotted():
    p = CollectPanel(CollectPanelState(viewer_mode="native"))
    assert p.state.snapshot()["viewer_mode"] == "native"


def test_page_has_controls_only_layout(panel):
    p, _, _ = panel
    with urllib.request.urlopen(f"{p.url}/", timeout=5) as r:
        html = r.read().decode()
    # native mode strips the viewer via the controls-only body class; the
    # width-driven card grid keeps the panel usable in a small window
    assert "controls-only" in html and "viewer_mode" in html
    # cards pack densely via CSS multi-column masonry (no ragged grid tails)
    assert "column-width" in html and "break-inside:avoid" in html
    # the embedded-iframe path is retained as the web-mode fallback
    assert 'id="rerunCard"' in html and 'id="rerun"' in html


class _FakeApp:
    """Stand-in for CollectApp: drives busy_detail/phase the way the real
    run_session does, so CollectRunner's phase ownership can be tested with no
    hardware."""

    def __init__(self, state, *, run_body):
        self.cc = types.SimpleNamespace(default_mode="full")
        self._state = state
        self._run_body = run_body

    def run_session(self, s):
        self._run_body(self._state)


def test_runner_busy_detail_bringup_then_running_then_cleared():
    state = CollectPanelState()
    panel = CollectPanel(state)      # not started: state + queue work offline
    seen = {}
    release = threading.Event()

    def body(st):
        st.update(busy_detail="connecting arm")
        snap = st.snapshot()
        seen["detail"] = snap["busy_detail"]
        seen["phase"] = snap["phase"]          # phase stays "starting" here
        st.update(phase="running", busy_detail="")
        seen["running_detail"] = st.snapshot()["busy_detail"]
        release.wait(3.0)                      # hold the "session" open

    runner = CollectRunner(_FakeApp(state, run_body=body), panel)
    assert runner.start_session({"task": "demo"}) is None
    assert _wait(lambda: state.snapshot()["phase"] == "running")
    assert seen["detail"] == "connecting arm"
    assert seen["phase"] == "starting"         # bring-up runs under "starting"
    assert state.snapshot()["busy_detail"] == ""   # cleared once running
    release.set()
    assert _wait(lambda: not runner.is_running())
    assert state.snapshot()["phase"] == "setup"
    assert state.snapshot()["busy_detail"] == ""   # cleared on clean finish


def test_runner_busy_detail_kept_on_error():
    state = CollectPanelState()
    panel = CollectPanel(state)

    def body(st):
        st.update(busy_detail="sensors + camera (open + warmup)")
        raise RuntimeError("dmtac open failed")

    runner = CollectRunner(_FakeApp(state, run_body=body), panel)
    assert runner.start_session({"task": "demo"}) is None
    assert _wait(lambda: not runner.is_running())
    snap = state.snapshot()
    # the last step reached is preserved so the operator sees where it died
    assert snap["busy_detail"] == "sensors + camera (open + warmup)"
    assert snap["error"] == "dmtac open failed"
    assert snap["phase"] == "setup"
