"""Regression tests for the three collection fixes (2026-07-27):
discard really deletes, the kept-episode counter decrements, and a UR3
protective stop does not kill the streamer/session."""

import numpy as np
import pytest

from phantom.data.schema import EpisodeMeta
from phantom.data_collect.teleop import DirectServoStreamer


# --------------------------------------------------------------- discard
class _StubWriter:
    def __init__(self, path):
        self.path = path
        self.meta = EpisodeMeta(task="t")

    def abort(self):
        self.meta.status = "aborted"


def _recorder(tmp_path):
    """A recorder with just enough wiring to exercise the discard paths."""
    from phantom.recording.recorder import EpisodeRecorder
    r = EpisodeRecorder.__new__(EpisodeRecorder)
    r.out_root = tmp_path
    return r


def _make_episode(tmp_path, name="ep_x"):
    d = tmp_path / name
    (d / "arm_q.zarr").mkdir(parents=True)
    (d / "arm_q.zarr" / "blob").write_bytes(b"x" * 4096)
    EpisodeMeta(task="t", status="finalized").save(d / "meta.json")
    return d


def test_relabel_discard_deletes_the_episode(tmp_path):
    r = _recorder(tmp_path)
    ep = _make_episode(tmp_path)
    assert ep.exists()
    r.relabel(ep, discard=True)
    assert not ep.exists(), "discard must DELETE, not just mark aborted"


def test_relabel_success_still_only_rewrites_meta(tmp_path):
    r = _recorder(tmp_path)
    ep = _make_episode(tmp_path)
    r.relabel(ep, success=True)
    assert ep.exists()
    assert EpisodeMeta.load(ep / "meta.json").success is True


def test_delete_is_confined_to_the_staging_root(tmp_path):
    r = _recorder(tmp_path / "staging")
    (tmp_path / "staging").mkdir()
    outside = _make_episode(tmp_path, "ep_outside")
    assert r._delete_episode(outside) is False
    assert outside.exists(), "must refuse to delete outside out_root"
    assert r._delete_episode(tmp_path / "staging") is False


def _abortable_recorder(tmp_path, small_hw, ep):
    import threading
    import types
    r = _recorder(tmp_path)
    r.hw = small_hw
    r.clock = None
    r.stream_filter = None
    r.session = types.SimpleNamespace(rings={})   # nothing to drain
    r._writer = _StubWriter(ep)
    r._thread = None
    r._stop = threading.Event()
    r._cursors = {}
    r._action_bufs = {}
    r._action_lock = threading.Lock()
    r._bytes_written = 0
    r._t_started = 0.0
    return r


def test_stop_abort_keeps_partial_data(tmp_path, small_hw):
    """abort marks the episode and KEEPS it on disk (arm faults / quits must
    never destroy data — the panel says 'saved (aborted)' and means it)."""
    ep = _make_episode(tmp_path)
    r = _abortable_recorder(tmp_path, small_hw, ep)
    assert r.stop(abort=True) == ep
    assert ep.exists(), "aborted episode must stay on disk"


def test_stop_abort_delete_flag_removes(tmp_path, small_hw):
    """physical removal only on the explicit delete flag."""
    ep = _make_episode(tmp_path)
    r = _abortable_recorder(tmp_path, small_hw, ep)
    assert r.stop(abort=True, delete=True) is None
    assert not ep.exists(), "delete=True must remove the tree"


# ------------------------------------------------- protective-stop survival
@pytest.mark.parametrize("msg", [
    "servoJ rejected - the RTDE control script is not running",
    "RTDE control stream lost (robot rebooted / protective stop)",
    "control interface not connected (connect(control=True))",
])
def test_control_faults_are_recoverable(msg):
    assert DirectServoStreamer._is_ctl_fault(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", [
    "a stream worker died", "shared memory segment vanished",
    "leader serial port closed",
])
def test_other_faults_still_kill_the_streamer(msg):
    assert DirectServoStreamer._is_ctl_fault(RuntimeError(msg)) is False


class _PstopArm:
    """servo_j fails like a UR whose control script died with a pstop."""

    def __init__(self):
        self.calls = 0

    def servo_j(self, q, dt, lookahead, gain):
        self.calls += 1
        raise RuntimeError("servoJ rejected - the RTDE control script is not "
                           "running (clear the pendant popup)")


def test_servo_swallows_a_pstop_instead_of_raising(small_hw):
    s = DirectServoStreamer.__new__(DirectServoStreamer)
    s.hw = small_hw
    s.arm = _PstopArm()
    s._lock = __import__("threading").Lock()
    s.ctl_lock = __import__("threading").Lock()
    from phantom.data_collect.teleop import TrackPhase
    s._phase = TrackPhase.TRACK
    s._last_cmd = None
    s._ctl_lost = False
    # must NOT raise: the session loop treats a streamer exception as fatal
    assert s._servo(np.zeros(small_hw.arm.dof), 0.008) is False
    assert s._ctl_lost is True, "the loop needs this flag to reconnect"
