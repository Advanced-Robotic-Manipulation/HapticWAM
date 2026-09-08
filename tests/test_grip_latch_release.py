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
