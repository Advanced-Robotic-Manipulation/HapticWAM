"""Web control panel: HTTP API, SSE snapshot, command queue, page integrity.
Stdlib-only — no rerun needed."""

import json
import urllib.request

import pytest

from phantom.viz.panel import _BUTTONS, _PAGE, PanelServer, PanelState


@pytest.fixture
def server():
    state = PanelState(task="unit", operator="ci", teleop="echo")
    srv = PanelServer(state, host="127.0.0.1", port=0)   # 0 = ephemeral port
    srv.start()
    srv.port = srv._httpd.server_address[1]
    yield srv
    srv.stop()


def _get(srv, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}{path}", timeout=5) as r:
        return r.status, r.read()


def _post(srv, path, obj):
    req = urllib.request.Request(
        f"http://127.0.0.1:{srv.port}{path}", data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_served(server):
    code, body = _get(server, "/")
    assert code == 200
    html = body.decode()
    assert "PHANTOM" in html and "EventSource" in html and "/api/cmd" in html
    # every loop button is wired in the page ("quit" goes through
    # /api/session/stop -> SessionRunner pushes it into the loop)
    for btn in _BUTTONS:
        if btn != "quit":
            assert btn in _PAGE
    assert "/api/session/stop" in _PAGE and "/api/session/start" in _PAGE


def test_status_snapshot(server):
    server.state.update(recording=True, episode="ep_x", wrench_N=3.5, ep_count=2)
    code, body = _get(server, "/api/status")
    assert code == 200
    s = json.loads(body)
    assert s["recording"] is True and s["episode"] == "ep_x"
    assert s["wrench_N"] == 3.5 and s["ep_count"] == 2
    assert s["task"] == "unit" and "ep_seconds" in s
    assert "_lock" not in s


def test_command_queue_roundtrip(server):
    for btn in ("start_stop", "zero_ft"):
        code, resp = _post(server, "/api/cmd", {"button": btn})
        assert code == 200 and resp["ok"]
    assert server.pop_buttons() == {"start_stop": True, "zero_ft": True}
    assert server.pop_buttons() == {}                    # drained


def test_unknown_button_rejected(server):
    code, resp = _post(server, "/api/cmd", {"button": "rm_rf"})
    assert code == 400
    assert server.pop_buttons() == {}


def test_404(server):
    with pytest.raises(urllib.error.HTTPError):
        _get(server, "/nope")


def test_state_thread_safety_smoke():
    import threading
    st = PanelState()
    def spam():
        for i in range(500):
            st.update(wrench_N=float(i), recording=bool(i % 2))
            st.snapshot()
    ts = [threading.Thread(target=spam) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert isinstance(st.snapshot()["wrench_N"], float)


# ---------------------------------------------------------------------------
# v2: phases, episode log, setup payload, session lifecycle endpoints
# ---------------------------------------------------------------------------

class _StubRunner:
    def __init__(self):
        self.started = None
        self.stopped = False
        self.err = None
    def start_session(self, cfg):
        self.started = cfg
        return self.err
    def stop_session(self):
        self.stopped = True


def test_episode_log_and_reset():
    st = PanelState()
    st.add_episode(name="ep_a", outcome="success", tags=["x"])
    st.add_episode(name="ep_b", outcome="discarded")
    s = st.snapshot()
    assert [e["outcome"] for e in s["episodes"]] == ["success", "discarded"]
    st.reset_session(task="new", phase="starting")
    s = st.snapshot()
    assert s["episodes"] == [] and s["task"] == "new" and s["phase"] == "starting"


def test_setup_endpoint(server):
    server.setup_provider = lambda: {"rig_name": "test-rig", "mode": "mock"}
    code, body = _get(server, "/api/setup")
    assert code == 200 and json.loads(body)["rig_name"] == "test-rig"


def test_session_start_delegates(server):
    runner = _StubRunner()
    server.runner = runner
    code, resp = _post(server, "/api/session/start",
                       {"task": "t1", "teleop": "none", "target_episodes": 3})
    assert code == 200 and resp["ok"]
    assert runner.started["task"] == "t1"
    code, resp = _post(server, "/api/session/stop", {})
    assert code == 200 and runner.stopped


def test_session_start_error_propagates(server):
    runner = _StubRunner()
    runner.err = "сессия уже запущена"
    server.runner = runner
    code, resp = _post(server, "/api/session/start", {"task": "t"})
    assert code == 409 and "запущена" in resp["error"]


def test_session_start_without_runner_409(server):
    server.runner = None
    code, resp = _post(server, "/api/session/start", {"task": "t"})
    assert code == 409


def test_runner_validation():
    from phantom.viz.session import SessionRunner
    from phantom_test_utils import make_hw
    hw = make_hw()
    state = PanelState()
    srv = PanelServer(state, host="127.0.0.1", port=0)
    r = SessionRunner(hw, srv)
    assert r.start_session({"task": "плохое имя"}) is not None      # non-ascii
    assert r.start_session({"task": "ok", "teleop": "warp"}) is not None
    # echo configured in the repo yaml -> passes validation (don't actually run)


def test_setup_payload_shape():
    from phantom.viz.session import setup_payload
    from phantom_test_utils import make_hw
    p = setup_payload(make_hw())
    assert p["mode"] == "mock" and p["echo_configured"] is True
    assert isinstance(p["sensors"], list) and len(p["sensors"]) == 2
    assert "disk_free_gb" in p and "out_root" in p
