"""CollectRerun child-process isolation (the libarrow-segfault containment).

The parent-side proxy must guarantee: a dead/crashing child never affects the
parent, events drop cleanly under backpressure (never block the record loop),
stop() terminates the child, and the restart cap is honored (then run without
viz). No real rerun server is needed — the parent never imports rerun, and the
child logic is exercised through a fake process/context or a lightweight fake
child target (NO rerun) spawned as a real OS process for the isolation proof.
"""

from __future__ import annotations

import os
import queue
import signal
import sys
import time

import pytest

from phantom_test_utils import make_small_hw
from phantom.recording.workers import build_session_rings
from phantom.data_collect.rerun_view import CollectRerun


# ===========================================================================
# a real, lightweight child target (spawned as a real process, NO rerun)
# ===========================================================================

def _fake_child_ok(*args):
    # positional layout of _rerun_child_main: first six fixed, queues last --
    # *args keeps this fake valid when middle config args are added
    (hw_yaml, ring_specs, mode, rate_hz, web_port, ws_port) = args[:6]
    event_q, result_q, stop_evt = args[-3:]
    """Mimics _rerun_child_main WITHOUT importing rerun: reports a URL, then
    drains events until stopped / orphaned. Used to prove real-process
    isolation without needing a rerun server."""
    result_q.put(("url", f"http://localhost:{web_port}/?fake"))
    while not stop_evt.is_set():
        if os.getppid() == 1:
            break
        try:
            item = event_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break


# ===========================================================================
# fake in-process context/process — deterministic parent-logic tests
# ===========================================================================

class _FakeQueue:
    """queue.Queue with the .close() the proxy expects + bounded put_nowait."""
    def __init__(self, maxsize=0):
        self._q = queue.Queue(maxsize=maxsize)

    def put(self, item, *a, **k):
        self._q.put(item, *a, **k)

    def put_nowait(self, item):
        self._q.put_nowait(item)

    def get(self, *a, **k):
        return self._q.get(*a, **k)

    def qsize(self):
        return self._q.qsize()

    def close(self):
        pass


class _FakeProc:
    """In-process stand-in: start() runs the target's URL handshake only, then
    the test flips liveness to simulate a crash / normal exit."""
    def __init__(self, target, args, daemon=None, name=None, url="http://fake"):
        self.target, self.args = target, args
        self.name, self.daemon = name, daemon
        self.pid = 4242
        self.exitcode = None
        self._alive = True
        self._url = url

    def start(self):
        # emulate the child's server-up handshake: args[-2] is result_q
        result_q = self.args[-2]
        if self._url is not None:
            result_q.put(("url", self._url))

    def die(self, exitcode=-signal.SIGSEGV):
        self._alive = False
        self.exitcode = exitcode

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.die(-signal.SIGTERM)

    def kill(self):
        self.die(-signal.SIGKILL)


class _FakeCtx:
    """Fake spawn context capturing every Process it hands out."""
    def __init__(self, url="http://fake"):
        self.url = url
        self.procs: list[_FakeProc] = []
        self.event_qmax: int | None = None

    def Queue(self, maxsize=0):
        if maxsize:
            self.event_qmax = maxsize
        return _FakeQueue(maxsize=maxsize)

    def Event(self):
        import threading
        return threading.Event()

    def Process(self, target, args, daemon=None, name=None):
        p = _FakeProc(target, args, daemon=daemon, name=name, url=self.url)
        self.procs.append(p)
        return p


@pytest.fixture
def proxy():
    hw = make_small_hw()
    rings = build_session_rings(hw, session_id=f"testviz{time.time_ns()}")
    v = CollectRerun(hw, rings, mode="lite", rate_hz=10.0)
    yield v, hw, rings
    try:
        v.stop()
    except Exception:
        pass
    for r in rings.values():
        r.close()


def _use_fake_ctx(proxy_obj, url="http://fake"):
    ctx = _FakeCtx(url=url)
    proxy_obj._ctx = ctx
    return ctx


# ===========================================================================
# tests
# ===========================================================================

def test_start_reports_viewer_url(proxy):
    v, hw, rings = proxy
    _use_fake_ctx(v, url="http://localhost:9091/?x")
    v.start()
    assert v.viewer_url == "http://localhost:9091/?x"
    assert v.is_alive()


def test_start_timeout_disables_cleanly(proxy):
    v, hw, rings = proxy
    ctx = _use_fake_ctx(v, url=None)      # child never reports a URL
    v._START_TIMEOUT = 0.3
    v.start()
    assert v.viewer_url is None           # degraded, but no exception
    assert not v.is_alive()               # child was killed


def test_events_drop_when_queue_full(proxy):
    v, hw, rings = proxy
    _use_fake_ctx(v)
    v._EVENT_QUEUE_MAX = 3                 # tiny queue, nobody drains it
    v.start()
    for i in range(100):                   # far more than capacity
        v.log_event(f"e{i}")               # must never raise / block
    assert v._event_q.qsize() <= 3         # excess silently dropped


def test_child_crash_does_not_affect_parent_and_restarts(proxy):
    v, hw, rings = proxy
    ctx = _use_fake_ctx(v)
    v.start()
    assert v.is_alive()
    ctx.procs[-1].die()                    # simulate a libarrow segfault
    v.log_event("post-crash")             # detected here; must not raise
    assert v.is_alive()                    # a fresh child was spawned
    assert v._restarts == 1
    assert len(ctx.procs) == 2


def test_restart_cap_then_runs_without_viz(proxy):
    v, hw, rings = proxy
    ctx = _use_fake_ctx(v)
    v.start()
    for _ in range(5):                     # keep killing the child
        ctx.procs[-1].die()
        v.log_event("kill")
    assert v._disabled                     # gave up after the cap
    assert v.viewer_url is None
    assert v._restarts == CollectRerun._MAX_RESTARTS
    # once disabled, further events are cheap no-ops (no restart storms)
    n = len(ctx.procs)
    v.log_event("ignored")
    assert len(ctx.procs) == n


def test_stop_is_idempotent_and_safe_when_never_started(proxy):
    v, hw, rings = proxy
    v.stop()                               # never started
    v.stop()                               # twice
    assert not v.is_alive()


# ---------------------------------------------------------------------------
# real-process isolation: a real spawned child, killed with SIGKILL, must not
# take the parent down; stop() must terminate it.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="uses posix signals")
def test_real_child_kill_does_not_kill_parent(proxy):
    v, hw, rings = proxy
    v._child_target = _fake_child_ok       # real spawn, but no rerun
    v.start()
    assert v.viewer_url and v.viewer_url.endswith("?fake")
    child_pid = v._proc.pid
    assert v.is_alive()

    os.kill(child_pid, signal.SIGKILL)     # hard crash, like the segfault
    for _ in range(50):                    # let the OS reap it
        if not v._proc.is_alive():
            break
        time.sleep(0.05)
    assert not v._proc.is_alive()

    # parent is fine and recovers: next event triggers a clean restart
    v.log_event("after real kill")
    assert v.is_alive()
    assert v._proc.pid != child_pid
    assert v._restarts == 1

    v.stop()                               # must terminate the live child
    assert not v.is_alive()
