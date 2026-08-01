"""collect/gripper.py — continuous pilot: mailbox, deadband, open-priority."""

import time

from phantom_test_utils import make_small_hw

from phantom.data_collect.config import GripperTuning
from phantom.data_collect.gripper import GripperPilot


class FakeGripper:
    def __init__(self):
        self.moves = []          # (position, speed, force)

    def move(self, position, speed, force):
        self.moves.append((float(position), float(speed), float(force)))


def _pilot(hw, **tuning):
    g = FakeGripper()
    p = GripperPilot(hw, g, GripperTuning(**{"rate_hz": 200.0, **tuning}))
    return p, g


def _settle(pred, timeout=1.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_continuous_positions_reach_the_gripper():
    hw = make_small_hw()
    p, g = _pilot(hw)
    p.start()
    try:
        for target in (0.2, 0.5, 0.83):
            p.set_target(target)
            assert _settle(lambda: g.moves and abs(g.moves[-1][0] - target) < 1e-9)
        # proportional, not binary: three distinct intermediate positions
        assert len({round(m[0], 2) for m in g.moves}) == 3
        # snappy speed, pad-safe force from the hardware config
        assert g.moves[-1][1] == 1.0
        assert g.moves[-1][2] == hw.gripper.default_force
    finally:
        p.stop()


def test_deadband_suppresses_noop_commands():
    hw = make_small_hw()
    p, g = _pilot(hw, deadband=0.05)
    p.start()
    try:
        p.set_target(0.5)
        assert _settle(lambda: len(g.moves) == 1)
        for wiggle in (0.51, 0.49, 0.52, 0.48):   # all inside the deadband
            p.set_target(wiggle)
            time.sleep(0.03)
        assert len(g.moves) == 1
        p.set_target(0.6)                          # outside — goes through
        assert _settle(lambda: len(g.moves) == 2)
    finally:
        p.stop()


def test_open_now_overrides_and_suspends():
    hw = make_small_hw()
    p, g = _pilot(hw)
    p.start()
    try:
        p.set_target(0.7)
        assert _settle(lambda: g.moves and g.moves[-1][0] == 0.7)
        p.open_now()
        # full open at max speed lands even though the leader still says 0.7
        assert _settle(lambda: g.moves[-1][0] == 0.0 and g.moves[-1][1] == 1.0)
        assert p.suspended
        n = len(g.moves)
        p.set_target(0.9)                          # leader squeezes: ignored
        time.sleep(0.05)
        assert len(g.moves) == n
        p.release()                                # panel Resume
        assert not p.suspended
        p.set_target(0.9)
        assert _settle(lambda: g.moves[-1][0] == 0.9)
    finally:
        p.stop()


def test_last_sent_reflects_actual_commands():
    hw = make_small_hw()
    p, g = _pilot(hw)
    p.start()
    try:
        assert p.last_sent is None
        p.set_target(0.4)
        assert _settle(lambda: p.last_sent == 0.4)
    finally:
        p.stop()


class FakeLeader:
    def __init__(self, v=0.0):
        self.v = v

    def latest_gripper01(self):
        return self.v


def test_leader_pull_and_fresh_value_after_release():
    """Review finding (critical): after open_now()+release() the pilot must
    follow the FRESH leader value, never replay the pre-trip squeeze."""
    hw = make_small_hw()
    g = FakeGripper()
    leader = FakeLeader(0.8)                       # operator squeezing
    p = GripperPilot(hw, g, GripperTuning(rate_hz=200.0), leader=leader)
    p.start()
    try:
        assert _settle(lambda: g.moves and g.moves[-1][0] == 0.8)
        p.open_now()                               # safeguard trip
        assert _settle(lambda: g.moves[-1][0] == 0.0)
        leader.v = 0.1                             # operator relaxed meanwhile
        p.release()                                # panel Resume
        # follows the CURRENT leader value — not the stale 0.8 squeeze
        assert _settle(lambda: g.moves[-1][0] == 0.1)
        assert 0.8 not in [m[0] for m in g.moves[2:]]
    finally:
        p.stop()
