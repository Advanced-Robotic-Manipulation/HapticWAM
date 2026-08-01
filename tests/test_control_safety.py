"""Thread-safety guards that prevent the on-rig rtde_control / libarrow segfaults:

1. rtde_control.so use-after-free — servo_j (teleop streamer thread) racing
   disconnect (session teardown), AND the reconnect-after-protective-stop path
   (double reconnect / stacking a second control script on one robot).
2. libarrow.so — the rerun record-loop event log must never block behind a
   backpressured viz tick.
"""

import sys
import threading
import time
import types

import numpy as np
import pytest

from phantom_test_utils import make_hw
from phantom.drivers.real.ur import URArm
from phantom.viz.rerun_logger import RerunLogger


@pytest.fixture(autouse=True)
def _reset_live_ctrl_count():
    """The single-controller invariant is a process-wide class counter — keep
    tests independent."""
    URArm._live_ctrl_count = 0
    yield
    URArm._live_ctrl_count = 0


# ---------------------------------------------------------------------------
# 1. UR control-interface serialization
# ---------------------------------------------------------------------------

class _FakeCtrl:
    """Records the peak number of threads inside ANY control call at once."""
    def __init__(self):
        self._n = 0
        self.max_concurrent = 0
        self._guard = threading.Lock()
        self.disconnected = False

    def _busy(self):
        with self._guard:
            self._n += 1
            self.max_concurrent = max(self.max_concurrent, self._n)
        time.sleep(0.002)
        with self._guard:
            self._n -= 1

    def isConnected(self):
        return not self.disconnected

    def servoJ(self, *a):
        self._busy()
        return True

    def servoStop(self):
        self._busy()

    def stopScript(self):
        self._busy()

    def setTcp(self, *a):
        pass

    def setPayload(self, *a):
        pass

    def disconnect(self):
        self._busy()
        self.disconnected = True


def test_servo_j_and_disconnect_never_run_concurrently():
    """The lock must serialize servo_j against disconnect — a concurrent pair
    is the rtde_control use-after-free segfault."""
    arm = URArm(make_hw())
    ctrl = _FakeCtrl()
    arm._ctrl = ctrl
    arm._recv = None                       # disconnect() skips the recv side

    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            try:
                arm.servo_j(np.zeros(6), 0.008, 0.1, 300)
            except RuntimeError:
                return                     # after disconnect: clean raise, not a crash

    t = threading.Thread(target=hammer)
    t.start()
    time.sleep(0.02)
    for _ in range(20):
        arm.servo_j(np.zeros(6), 0.008, 0.1, 300)  # main thread also servos
    arm.disconnect()
    stop.set()
    t.join(2.0)
    assert not t.is_alive()
    assert ctrl.max_concurrent == 1        # never two control calls at once
    assert arm._ctrl is None               # disconnected cleanly


def test_servo_j_after_disconnect_raises_not_segfaults():
    """A servo_j that loses the race and runs after disconnect must raise a
    Python error (caught upstream), never touch a freed C++ object."""
    arm = URArm(make_hw())
    arm._ctrl = _FakeCtrl()
    arm._recv = None
    arm.disconnect()
    try:
        arm.servo_j(np.zeros(6), 0.008, 0.1, 300)
        assert False, "expected RuntimeError after disconnect"
    except RuntimeError:
        pass


def test_reconnect_always_rebuilds_even_if_socket_connected():
    """Blind teardown -> rebuild, even when isConnected() is True: the socket
    staying up says nothing about the control script (early-returning on it
    caused the 125 Hz reconnect/reject loop, and probing the script can
    segfault -- see the comment in reconnect_control)."""
    arm = URArm(make_hw())
    ctrl = _FakeCtrl()                       # isConnected() -> True
    arm._ctrl = ctrl
    built = []
    arm._connect_control = lambda: built.append(1)
    arm.reconnect_control()
    assert built == [1]                      # exactly one rebuild
    assert ctrl.disconnected                 # old interface torn down first


def test_reconnect_rebuilds_and_drops_old_reference():
    """A dead interface is torn down (ref dropped first) and replaced."""
    arm = URArm(make_hw())
    old = _FakeCtrl()
    old.disconnected = True                  # isConnected() -> False
    arm._ctrl = old
    new = _FakeCtrl()
    arm._connect_control = lambda: setattr(arm, "_ctrl", new)
    arm.reconnect_control()
    assert arm._ctrl is new


def test_connect_control_releases_half_built_on_setup_failure(monkeypatch):
    """If setTcp/setPayload fail after the interface is built, the half-built
    C++ object must be released (else it lingers as a 2nd controller) and
    self._ctrl must stay None."""
    arm = URArm(make_hw())
    built = _FakeCtrl()
    built.setTcp = lambda *a: (_ for _ in ()).throw(RuntimeError("setup fail"))
    fake_mod = types.SimpleNamespace(RTDEControlInterface=lambda *a, **k: built)
    monkeypatch.setitem(sys.modules, "rtde_control", fake_mod)
    with pytest.raises(RuntimeError, match="setup fail"):
        arm._connect_control()
    assert built.disconnected                # released
    assert arm._ctrl is None                 # never published


def test_connect_control_refuses_second_interface(monkeypatch):
    """The single-controller invariant: opening a 2nd control interface while
    one is already live must RAISE (converting the two-controllers segfault into
    a loud error), not proceed."""
    arm = URArm(make_hw())
    URArm._live_ctrl_count = 1                # pretend a controller is already live
    fake_mod = types.SimpleNamespace(RTDEControlInterface=lambda *a, **k: _FakeCtrl())
    monkeypatch.setitem(sys.modules, "rtde_control", fake_mod)
    with pytest.raises(RuntimeError, match="SECOND RTDE control interface"):
        arm._connect_control()
    assert arm._ctrl is None
    assert URArm._live_ctrl_count == 1       # unchanged (never reserved a 2nd)


def test_connect_and_teardown_balance_the_controller_slot(monkeypatch):
    """connect reserves the single slot; teardown frees it, so sequential
    sessions never trip the invariant."""
    arm = URArm(make_hw())
    monkeypatch.setitem(
        sys.modules, "rtde_control",
        types.SimpleNamespace(RTDEControlInterface=lambda *a, **k: _FakeCtrl()))
    arm._connect_control()
    assert URArm._live_ctrl_count == 1
    arm._recv = None
    arm.disconnect()
    assert URArm._live_ctrl_count == 0       # slot freed -> next session can connect
    assert arm._ctrl is None


def test_reconnect_and_servo_never_concurrent(monkeypatch):
    """reconnect_control (rebuilds the interface) must serialize against servo_j
    — a rebuild racing an in-flight servoJ is the reconnect-path use-after-free."""
    arm = URArm(make_hw())
    monkeypatch.setattr(URArm, "_reconnect_settle_s", 0.0)  # no settle in the unit test
    ctrl = _FakeCtrl()
    arm._ctrl = ctrl
    # reconnect swaps in a fresh interface each time (under the same lock)
    def fake_connect():
        arm._ctrl = _FakeCtrl()
    arm._connect_control = fake_connect
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            try:
                arm.servo_j(np.zeros(6), 0.008, 0.1, 300)
            except RuntimeError:
                pass
    t = threading.Thread(target=hammer)
    t.start()
    time.sleep(0.02)
    for _ in range(15):
        arm._ctrl.disconnected = True        # force the rebuild branch
        arm.reconnect_control()
    stop.set()
    t.join(2.0)
    assert not t.is_alive()
    assert arm._ctrl.max_concurrent <= 1     # servoJ and reconnect never overlap


# ---------------------------------------------------------------------------
# 2. rerun event logging must not block under backpressure
# ---------------------------------------------------------------------------

class _FakeRR:
    def __init__(self):
        self.calls = 0

    def TextLog(self, text, level="INFO"):
        return (text, level)

    def log(self, path, payload):
        self.calls += 1


def test_log_event_drops_instead_of_blocking_under_contention():
    """If a backpressured tick is holding _rr_lock, log_event (called from the
    record loop) must return quickly and drop, not stall the session."""
    viz = RerunLogger(make_hw(), {})
    viz._rr = _FakeRR()
    # simulate a tick stuck inside rr.log holding the process-wide lock
    RerunLogger._rr_lock.acquire()
    try:
        t0 = time.perf_counter()
        viz.log_event("blocked event")
        dt = time.perf_counter() - t0
        assert dt < 0.5                    # bounded by the acquire timeout, not 5 s
        assert viz._rr.calls == 0          # dropped, not logged
    finally:
        RerunLogger._rr_lock.release()
    # lock free again: the event logs normally
    viz.log_event("ok event")
    assert viz._rr.calls == 1
