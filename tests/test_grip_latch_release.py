"""Placement release of the aperture latch (rig 09-08 seed 110: carried to
the box, asked to open for 5 s, latch never let go)."""
from __future__ import annotations

import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.deploy.executor import ChunkExecutor
from tests.test_servo_result import _StubArm


class _Ring:
    def __init__(self):
        self.z = 0.30

    def latest(self, n):
        return np.array([1.0]), {"tcp_pose": np.array([[-0.38, 0.1, self.z, 0, 0, 0]])}


class _Safety:
    def __init__(self):
        self.contact_load = {"left": 9.0, "right": 9.0}
        self.ring = _Ring()
        self.rings = {"arm": self.ring}


def _ex(monkeypatch, **cfg):
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    for k, v in cfg.items():
        object.__setattr__(hw.safety, k, v)
    ex = ChunkExecutor(hw, arm=_StubArm(set()), gripper=None, safety=_Safety())
    clock = [100.0]
    monkeypatch.setattr("phantom.deploy.executor.time.perf_counter", lambda: clock[0])
    return ex, clock


def _tick(ex, clock, grip, z, dt=0.008):
    ex.safety.ring.z = z
    clock[0] += dt
    return ex._latched_grip(grip)


def test_release_low_after_a_carry(monkeypatch):
    ex, clock = _ex(monkeypatch)
    assert _tick(ex, clock, 0.64, 0.07) == 0.64 and ex._grip_latch == 0.64     # grasp latches
    for _ in range(50):
        assert _tick(ex, clock, 0.64, 0.30) == 0.64                            # carry high
    for _ in range(30):
        assert _tick(ex, clock, 0.35, 0.10) == 0.64                            # 0.24 s: still held
    for _ in range(40):
        g = _tick(ex, clock, 0.35, 0.10)                                       # past 0.5 s
    assert g == 0.35 and ex._grip_latch is None
    # pads still loaded right after the release: no instant re-latch
    assert _tick(ex, clock, 0.33, 0.10) == 0.33 and ex._grip_latch is None
    # once unloaded, a new grasp latches again
    ex.safety.contact_load = {"left": 0.0, "right": 0.0}
    _tick(ex, clock, 0.10, 0.10)
    ex.safety.contact_load = {"left": 9.0, "right": 9.0}
    assert _tick(ex, clock, 0.60, 0.07) == 0.60 and ex._grip_latch == 0.60


def test_no_release_high_or_at_the_grasp_site_or_with_a_small_drop(monkeypatch):
    ex, clock = _ex(monkeypatch)
    _tick(ex, clock, 0.64, 0.07)
    for _ in range(200):                                     # open request while HIGH (09-04 drops)
        assert _tick(ex, clock, 0.30, 0.30) == 0.64
    ex2, clock2 = _ex(monkeypatch)
    _tick(ex2, clock2, 0.64, 0.07)
    for _ in range(200):                                     # open request at the grasp site, never lifted
        assert _tick(ex2, clock2, 0.30, 0.07) == 0.64
    ex3, clock3 = _ex(monkeypatch)
    _tick(ex3, clock3, 0.64, 0.07)
    for _ in range(50):
        _tick(ex3, clock3, 0.64, 0.30)
    for _ in range(200):                                     # low after a carry but only 0.10 below the latch
        assert _tick(ex3, clock3, 0.55, 0.10) == 0.64


def test_release_request_must_be_continuous(monkeypatch):
    ex, clock = _ex(monkeypatch)
    _tick(ex, clock, 0.64, 0.07)
    for _ in range(50):
        _tick(ex, clock, 0.64, 0.30)
    for _ in range(3):
        for _ in range(40):                                  # 0.32 s open ...
            assert _tick(ex, clock, 0.35, 0.10) == 0.64
        assert _tick(ex, clock, 0.64, 0.10) == 0.64          # ... then a close: streak resets
    assert ex._grip_latch == 0.64


def test_release_disabled_keeps_the_old_latch(monkeypatch):
    ex, clock = _ex(monkeypatch, grip_latch_release_drop=0.0)
    _tick(ex, clock, 0.64, 0.07)
    for _ in range(50):
        _tick(ex, clock, 0.64, 0.30)
    for _ in range(200):
        assert _tick(ex, clock, 0.20, 0.10) == 0.64


def test_no_measured_z_means_no_release(monkeypatch):
    ex, clock = _ex(monkeypatch)
    ex.safety.rings = {}
    _tick(ex, clock, 0.64, 0.07)
    for _ in range(200):
        assert _tick(ex, clock, 0.20, 0.10) == 0.64


def test_gripper_channel_is_capped_while_the_pose_plays_the_tail(monkeypatch):
    """09-08 seed 115: 16 played steps executed each chunk tail's opening
    once per replan. The pose may play to 16; the gripper stops at
    grip_play_steps."""
    from phantom.inference.policy import Plan
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    ex = ChunkExecutor(hw, arm=_StubArm(set()), gripper=None, safety=_Safety(),
                       max_play_steps=16, grip_play_steps=10)
    H = 16
    acts = np.zeros((H, 7)); acts[:, 2] = 0.01                 # rise 1 cm per step
    acts[:, 6] = 0.6; acts[12:, 6] = 0.1                       # tail says OPEN
    plan = Plan.__new__(Plan); plan.actions = acts; plan.t0_pose = np.zeros(6)
    rate = hw.control.action_rate_hz
    tgt9, g9 = ex._pose_at(plan, 9.0 / rate)
    tgt15, g15 = ex._pose_at(plan, 15.5 / rate)
    assert g9 == 0.6 and g15 == 0.6                            # tail opening never reaches the fingers
    assert tgt15[2] > tgt9[2] > 0                              # pose keeps playing into the tail
    ex2 = ChunkExecutor(hw, arm=_StubArm(set()), gripper=None, safety=_Safety(),
                        max_play_steps=16, grip_play_steps=None)
    assert ex2._pose_at(plan, 15.5 / rate)[1] == 0.1          # uncapped: tail opening plays


def test_per_episode_reset_clears_control_loss_and_limiter_counters(monkeypatch):
    """The arm driver is shared across the episodes of one process: the
    control-loss snapshot (09-10) AND the servo-limiter hit/hold counters
    (09-11: every later stop.json carried the launch-to-date totals) must be
    cleared when an episode starts."""
    ex, _clock = _ex(monkeypatch)
    ex.arm.control_loss_last = {"summary": "stale"}
    ex.arm.limiter_last = {"stale": 1}
    ex.arm._limiter_hits = 7
    ex.arm._limiter_holds = 3
    ex._reset_arm_episode_state()
    assert ex.arm.control_loss_last == {} and ex.arm.limiter_last == {}
    assert ex.arm._limiter_hits == 0 and ex.arm._limiter_holds == 0
