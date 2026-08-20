"""Start-pose homing + OOD gate + stall watchdog (postmortem 2026-08-20)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.deploy import start_pose as sp
from phantom.deploy.planner import PlannerLoop


def _hw():
    return load_hardware(None)


def test_start_stats_load_all_tasks():
    stats = sp.load_start_stats()
    assert set(stats) == {"Carton", "egg", "waffles", "whiteboard"}
    for t, st in stats.items():
        assert st.n == 180
        assert st.tcp_mean.shape == (6,) and st.tcp_std.shape == (6,)
        assert np.all(st.tcp_std > 0)
        assert 0.0 <= st.gripper_mean <= 1.0
    # spot-check against the generation run (full 790-ep dataset, 2026-08-20)
    w = stats["waffles"]
    assert np.allclose(w.tcp_mean[:3], [-0.3745, -0.2825, 0.3340], atol=1e-4)


def test_sample_start_pose_within_one_sigma():
    st = sp.load_start_stats()["waffles"]
    rng = np.random.default_rng(0)
    for _ in range(200):
        tcp, grip = sp.sample_start_pose(st, rng)
        assert np.all(np.abs(tcp - st.tcp_mean) <= st.tcp_std + 1e-12)
        assert 0.0 <= grip <= 1.0


def test_sigma_report_flags_postmortem_start():
    """The actual episode-1 start from the failed session must gate out."""
    st = sp.load_start_stats()["waffles"]
    bad = np.array([-0.3751, -0.1957, 0.3063, -1.12, -1.89, 1.52])
    sig, table = sp.start_sigma_report(st, bad, gripper_pos=0.45)
    assert float(sig[1]) > 2.5          # y was the killer axis
    assert float(np.max(sig)) > 2.5
    assert "sigma" in table
    good = st.tcp_mean.copy()
    sig2, _ = sp.start_sigma_report(st, good)
    assert float(np.max(sig2)) < 1e-9


def test_wait_gripper_settled():
    class G:
        def __init__(self, objs):
            self.objs = list(objs)
        def get_state(self):
            return SimpleNamespace(obj=self.objs.pop(0) if len(self.objs) > 1
                                   else self.objs[0])
    assert sp.wait_gripper_settled(G([0.0, 0.0, 3.0]), timeout_s=2.0)
    assert not sp.wait_gripper_settled(G([0.0]), timeout_s=0.3)


def test_move_to_start_mock_rig():
    hw = _hw()
    from phantom.drivers.mock.ur import MockArm
    from phantom.drivers.mock.robotiq import MockGripper
    arm = MockArm(hw); arm.connect(control=True)
    grip = MockGripper(hw.gripper); grip.connect(); grip.activate()
    st = sp.load_start_stats()["waffles"]
    tcp_t, grip_t = sp.move_to_start(arm, grip, hw, st,
                                     rng=np.random.default_rng(1))
    got = arm.get_state().tcp_pose
    assert np.allclose(got, tcp_t, atol=1e-9)
    sig, _ = sp.start_sigma_report(st, got)
    assert float(np.max(sig)) <= 1.0 + 1e-9


class _StubExecutor:
    """Commanded pose advances every call; stopped_reason writable."""
    def __init__(self):
        self.stopped_reason = None
        self._cmd = np.zeros(6)
    def last_cmd(self):
        self._cmd = self._cmd + np.array([0.01, 0, 0, 0, 0, 0])
        return self._cmd.copy()
    def submit(self, plan):
        return True


class _StubSnapshots:
    """TCP frozen at zero — the arm 'is not moving'."""
    def __init__(self, hw):
        self.hw = hw
        dof = hw.arm.dof
        self.ur = np.zeros(2 * dof + 12)
    def build(self):
        return SimpleNamespace(t=time.perf_counter(), ur_state=self.ur)


class _StubPolicy:
    def replan(self, snap, prev_plan, tcp_pose):
        H = 16
        return SimpleNamespace(latency_s=0.01, gate=0.0,
                               p_evt=np.array([1.0, 0, 0, 0, 0]),
                               sigma=np.zeros(3), actions=np.zeros((H, 7)),
                               action_times=np.array([time.perf_counter() + 10]),
                               t_created=time.perf_counter(),
                               t0_pose=np.zeros(6))


def test_stall_watchdog_stops_episode():
    hw = _hw()
    loop = PlannerLoop(hw, _StubPolicy(), _StubSnapshots(hw), _StubExecutor())
    loop.run(max_replans=10)
    assert loop.executor.stopped_reason == "motion_stall"
    # commanded 10mm/window vs 0 actual -> strikes at replans 1,2 -> <=4 replans
    assert len(loop.trace) <= 4


def test_no_stall_when_arm_follows():
    hw = _hw()

    class FollowingSnapshots(_StubSnapshots):
        def __init__(self, hw, ex):
            super().__init__(hw)
            self.ex = ex
        def build(self):
            dof = self.hw.arm.dof
            cmd = self.ex._cmd
            self.ur = self.ur.copy()
            self.ur[2 * dof:2 * dof + 6] = cmd   # tcp tracks commanded exactly
            return SimpleNamespace(t=time.perf_counter(), ur_state=self.ur)

    ex = _StubExecutor()
    loop = PlannerLoop(hw, _StubPolicy(), FollowingSnapshots(hw, ex), ex)
    loop.run(max_replans=6)
    assert ex.stopped_reason is None
    assert len(loop.trace) == 6
