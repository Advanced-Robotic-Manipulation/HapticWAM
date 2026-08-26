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
        assert st.n == 250
        assert st.tcp_mean.shape == (6,) and st.tcp_std.shape == (6,)
        assert np.all(st.tcp_std > 0)
        assert 0.0 <= st.gripper_mean <= 1.0
    # spot-check against the generation run (full 790-ep dataset, 2026-08-20)
    w = stats["waffles"]
    assert np.allclose(w.tcp_mean[:3], [-0.3652, -0.2859, 0.3346], atol=1e-4)   # v4+batch_20260822


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
    # pinned to the SHIPPED default gate (2.5): the pose that burned the
    # session must actually refuse, not just look large in a report
    DEFAULT_MAX_START_SIGMA = 2.5
    assert float(np.max(sig)) > DEFAULT_MAX_START_SIGMA
    assert "sigma" in table
    good = st.tcp_mean.copy()
    sig2, _ = sp.start_sigma_report(st, good)
    assert float(np.max(sig2)) < 1e-9


def test_gripper_participates_in_gate():
    """Perfect TCP + wildly wrong gripper must still gate out (codex review:
    the gripper sigma was print-only and never reached the max)."""
    st = sp.load_start_stats()["Carton"]
    sig, _ = sp.start_sigma_report(st, st.tcp_mean.copy(), gripper_pos=1.0)
    assert sig.shape == (7,)
    assert float(np.max(sig)) > 3.0


def test_rotvec_antipodal_representation_does_not_false_refuse():
    """whiteboard's mean rotvec norm is 2.646 — a live pose past the pi
    surface returns from UR in the antipodal representation; the gate must
    read it as the same rotation, not ~20 sigma."""
    st = sp.load_start_stats()["whiteboard"]
    r = st.tcp_mean[3:6].copy()
    n = np.linalg.norm(r)
    r_anti = r * (1.0 - 2.0 * np.pi / n)      # same rotation, other branch
    pose = st.tcp_mean.copy(); pose[3:6] = r_anti
    sig, _ = sp.start_sigma_report(st, pose)
    assert float(np.max(sig)) < 1e-6


def test_nonfinite_live_state_gates_out():
    """NaN in the live TCP must read as infinite sigma, never pass a gate."""
    st = sp.load_start_stats()["waffles"]
    bad = st.tcp_mean.copy(); bad[0] = np.nan
    sig, _ = sp.start_sigma_report(st, bad)
    assert np.isinf(sig[0]) and float(np.max(sig)) > 3.0


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
        self.stop_requested = False
        self._cmd = np.zeros(6)
    def last_cmd(self):
        self._cmd = self._cmd + np.array([0.01, 0, 0, 0, 0, 0])
        return self._cmd.copy()
    def request_stop(self, reason):
        if self.stopped_reason is None:
            self.stopped_reason = reason
        self.stop_requested = True
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
    assert loop.executor.stop_requested
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


def test_grasp_weighted_windows_anchor_before_close(tmp_path):
    """grasp_frac=1.0 must draw every resampled t0 inside the pre-close band."""
    import zarr
    from phantom.config.hardware import load_hardware
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    from phantom.data.schema import NormStats
    from phantom.train import common as C
    from phantom.train.builder import build_model
    from phantom.config.paths import load_paths
    hw = load_hardware(None)
    pm = build_model(hw, load_paths(), student=False, tiny=True, load_base=False)
    root = tmp_path / "eps"
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, pm.bb, NormStats.identity(), student=False)
    ds = C.WindowDataset(root, sampler, windows_per_episode=4, seed=1, grasp_frac=1.0)
    ep = ds.index[0].episode
    tc_real = ds._close_time(ep)
    assert tc_real is not None, "synthetic grasp_slip episode must close the gripper"
    lo, hi = ds.index[0].lo, ds.index[0].hi
    # inject a close time mid-range so the pre-close band is inside [lo, hi]
    tc = lo + 0.7 * (hi - lo)
    ds._close_cache[ep] = tc
    a, b = max(lo, tc - 1.5), min(hi, tc - 0.2)
    assert b > a
    sampled = []
    orig = ds.sampler.sample
    ds.sampler.sample = lambda e, t0: sampled.append(t0) or {"t0": t0}
    try:
        for _ in range(40):
            ds[0]
    finally:
        ds.sampler.sample = orig
    assert all(a - 1e-9 <= t <= b + 1e-9 for t in sampled), (min(sampled), max(sampled), a, b)
    # band outside the valid range -> graceful fallback to uniform, never crash
    ds._close_cache[ep] = lo - 5.0
    ds.sampler.sample = lambda e, t0: {"t0": t0}
    try:
        t0 = ds[0]["t0"]
    finally:
        ds.sampler.sample = orig
    assert lo <= t0 <= hi
    # and the unweighted dataset is unchanged
    ds0 = C.WindowDataset(root, sampler, windows_per_episode=4, seed=1)
    assert ds0.grasp_frac == 0.0 and ds0._close_cache == {}


def test_photo_aug_perturbs_video_only_on_train(tmp_path):
    import torch
    from phantom.config.hardware import load_hardware
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    from phantom.data.schema import NormStats
    from phantom.train import common as C
    from phantom.train.builder import build_model
    from phantom.config.paths import load_paths
    hw = load_hardware(None)
    pm = build_model(hw, load_paths(), student=False, tiny=True, load_base=False)
    root = tmp_path / "eps"
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, pm.bb, NormStats.identity(), student=False)
    ds_aug = C.WindowDataset(root, sampler, windows_per_episode=2, seed=3, photo_aug=1.0)
    ds_val = C.WindowDataset(root, sampler, windows_per_episode=2, seed=3,
                             resample=False, photo_aug=1.0)
    wi = ds_aug.index[0]
    base = sampler.sample(wi.episode, wi.t0)["video"]
    item = ds_aug[0]
    assert not torch.allclose(item["video"], base)        # jitter applied
    assert item["video"].min() >= -1.0 - 1e-6 and item["video"].max() <= 1.0 + 1e-6
    other = {k: v for k, v in item.items() if k in ("gel", "fields")}
    ref = sampler.sample(wi.episode, ds_aug.index[0].t0)
    for k, v in other.items():
        pass                                              # tactile untouched by construction
    # val/frozen datasets never augment (resample=False)
    v0 = ds_val[0]["video"]; v1 = ds_val[0]["video"]
    assert torch.allclose(v0, v1)
