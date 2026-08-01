"""collect/teleop.py — device-rate command path: engage, track, hold, resume."""

import time

import numpy as np

from phantom_test_utils import make_small_hw
from phantom.drivers.mock.ur import MockArm

from phantom.data_collect.config import TeleopTuning
from phantom.data_collect.teleop import DirectServoStreamer, TrackPhase


class FakeLeader:
    def __init__(self, q=None):
        self.q = q

    def latest_q_target(self):
        return None if self.q is None else np.asarray(self.q, dtype=np.float64)


class StampedLeader:
    """Leader exposing the drop-out-aware API (q, velocity, timestamp)."""
    def __init__(self, q=None, v=None, t=0.0):
        self.q = None if q is None else np.asarray(q, dtype=np.float64)
        self.v = np.zeros(6) if v is None else np.asarray(v, dtype=np.float64)
        self.t = t

    def latest_q_target(self):
        return None if self.q is None else self.q.copy()

    def latest_target_stamped(self):
        if self.q is None:
            return None
        return self.q.copy(), self.v.copy(), self.t


def _hw():
    # keep the mock arm allowed to move fast enough for the test targets
    return make_small_hw(arm={"limits": {"joint_speed_rad_s": 4.0}})


def _mk(hw, leader):
    arm = MockArm(hw)
    arm.connect(control=True)
    streamer = DirectServoStreamer(hw, arm, leader, TeleopTuning(
        v_max_rad_s=3.0, a_max_rad_s2=10.0,
        engage_v_max_rad_s=1.5, engage_eps_rad=0.05))
    return arm, streamer


def _settle(pred, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_engage_then_track_reaches_leader_target():
    hw = _hw()
    leader = FakeLeader()
    arm, s = _mk(hw, leader)
    s.start()
    try:
        assert s.phase is TrackPhase.ENGAGE      # no leader yet — no commands
        q0 = arm.get_state().q
        target = q0 + 0.3
        leader.q = target
        assert _settle(lambda: s.phase is TrackPhase.TRACK)
        assert _settle(lambda: np.allclose(arm.get_state().q, target, atol=0.06))
        assert s.error is None
        # the recorded absolute action is the commanded joint target
        assert np.allclose(s.last_cmd, target, atol=0.06)
    finally:
        s.stop()


def test_track_follows_moving_target():
    hw = _hw()
    leader = FakeLeader()
    arm, s = _mk(hw, leader)
    q0 = arm.get_state().q
    leader.q = q0.copy()
    s.start()
    try:
        assert _settle(lambda: s.phase is TrackPhase.TRACK)
        for step in (0.1, 0.2, 0.3):
            leader.q = q0 + step
            assert _settle(lambda: np.allclose(arm.get_state().q, q0 + step,
                                               atol=0.06))
    finally:
        s.stop()


def test_hold_freezes_until_resume():
    hw = _hw()
    leader = FakeLeader()
    arm, s = _mk(hw, leader)
    q0 = arm.get_state().q
    leader.q = q0.copy()
    s.start()
    try:
        assert _settle(lambda: s.phase is TrackPhase.TRACK)
        s.hold()                                   # safeguard trip
        assert s.phase is TrackPhase.HOLD
        frozen = arm.get_state().q.copy()
        leader.q = q0 + 0.5                        # operator keeps moving the exo
        time.sleep(0.15)
        assert np.allclose(arm.get_state().q, frozen, atol=0.02)   # arm did not follow
        s.resume()                                 # panel Resume: slow re-engage
        assert _settle(lambda: s.phase is TrackPhase.TRACK)
        assert _settle(lambda: np.allclose(arm.get_state().q, q0 + 0.5, atol=0.06))
    finally:
        s.stop()


def test_dead_leader_is_safe():
    hw = _hw()
    arm, s = _mk(hw, FakeLeader(None))
    s.start()
    try:
        time.sleep(0.1)
        assert s.last_cmd is None                  # never commanded anything
        assert s.error is None
    finally:
        s.stop()


def test_hold_during_engage_is_not_stomped():
    """Review finding (critical): the ENGAGE->TRACK transition used to
    overwrite a concurrent safeguard hold() — the arm kept tracking after a
    latched trip. All transitions are now compare-and-set: HOLD must win."""
    hw = _hw()
    leader = FakeLeader()
    arm, s = _mk(hw, leader)
    q0 = arm.get_state().q
    leader.q = q0 + 1.2          # long engage (engage_v_max 1.5 -> ~0.8 s)
    s.start()
    try:
        assert _settle(lambda: s.phase is TrackPhase.ENGAGE and
                       s.last_cmd is not None)
        s.hold()                 # safeguard trips mid-engage
        # give the streamer many cycles to (wrongly) leave HOLD
        for _ in range(30):
            assert s.phase is TrackPhase.HOLD
            time.sleep(0.01)
        frozen = arm.get_state().q.copy()
        time.sleep(0.2)
        assert np.allclose(arm.get_state().q, frozen, atol=0.02)
    finally:
        s.stop()


def test_leader_target_extrapolates_across_dropout():
    """A stalled leader (timestamp not advancing) must keep the target GLIDING
    at its last velocity, not freeze — the fix for stall-then-jump. Past the
    extrapolation cap it holds (no runaway)."""
    hw = _hw()
    ld = StampedLeader(q=np.full(6, 0.1), v=np.full(6, 2.0), t=5.0)
    arm, s = _mk(hw, ld)                                   # thread NOT started
    period = 1.0 / 125.0
    assert np.allclose(s._leader_target(5.0, period), 0.1)  # fresh -> verbatim
    prev = 0.1
    for k in range(1, 6):                                   # device stalled
        tgt = s._leader_target(5.0 + k * period, period)
        assert tgt[0] > prev                               # still moving forward
        prev = tgt[0]
    held_a = s._leader_target(5.0 + 0.20, period)[0]        # past cap (0.12 s)
    held_b = s._leader_target(5.0 + 0.30, period)[0]
    assert held_a == held_b                                # capped, no runaway
    arm.disconnect()


def test_leader_target_reengages_after_long_gap():
    """A gap longer than stale_reengage_s must flag a glide-back (ENGAGE), so a
    long drop-out doesn't end in an accel-clamped lurch to the accumulated pose."""
    hw = _hw()
    ld = StampedLeader(q=np.zeros(6), v=np.zeros(6), t=5.0)
    arm, s = _mk(hw, ld)
    period = 1.0 / 125.0
    s._leader_target(5.0, period)                          # first fresh sample
    ld.q = np.full(6, 0.4)
    ld.t = 6.0                                             # 1.0 s gap > 0.4 s
    s._leader_target(6.0, period)
    assert s._stale_reengage is True
    arm.disconnect()


def test_streamer_tracks_stamped_leader_end_to_end():
    """The stamped-leader path drives the arm the same as the plain one."""
    hw = _hw()
    ld = StampedLeader(q=None)
    arm, s = _mk(hw, ld)
    q0 = arm.get_state().q
    ld.q = q0.copy()
    ld.t = 1.0
    s.start()
    try:
        assert _settle(lambda: s.phase is TrackPhase.TRACK)
        for i, step in enumerate((0.1, 0.2, 0.3), start=2):
            ld.q = q0 + step
            ld.t = float(i)                                # fresh timestamps
            assert _settle(lambda: np.allclose(arm.get_state().q, q0 + step,
                                               atol=0.06))
        assert s.error is None
    finally:
        s.stop()


def test_double_start_does_not_spawn_second_thread():
    """A duplicate start() must not launch a second streamer thread — two
    threads would run two RTDE control interfaces on one robot (the double-
    engage that preceded the reconnect-path segfault)."""
    hw = _hw()
    leader = FakeLeader()
    arm, s = _mk(hw, leader)
    leader.q = arm.get_state().q.copy()
    s.start()
    try:
        first = s._thread
        s.start()                              # duplicate — must be ignored
        assert s._thread is first              # same thread object, not replaced
        assert s.is_running()
    finally:
        s.stop()


def test_resume_is_noop_unless_held():
    """resume() must not disturb ENGAGE/TRACK (arm-guard vs tactile-latch
    interplay: a stray resume cannot skip the engage glide)."""
    hw = _hw()
    leader = FakeLeader()
    arm, s = _mk(hw, leader)
    leader.q = arm.get_state().q.copy()
    s.start()
    try:
        assert _settle(lambda: s.phase is TrackPhase.TRACK)
        s.resume()                                 # not held: no-op
        assert s.phase is TrackPhase.TRACK
    finally:
        s.stop()
