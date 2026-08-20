"""Regression tests for the 2026-08-14 deploy root-cause fixes.

Pins the three defects behind the first rig sessions' timid dithering:
(1) the deploy path never polled the gripper — snapshot gripper dims were a
    frozen zeros(2) substitute that collapsed sampled action magnitude ~3x;
(2) plan swap re-anchored playback to a pose measured one inference latency
    earlier AND restarted at index 0, rewinding ~all of each cycle's advance;
(3) first-replan prev_chunk used raw zeros in NORMALIZED space (= per-dim
    demo-mean offset; -1.5 sigma on the gripper channel).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.deploy.executor import ChunkExecutor
from phantom.inference.policy import Plan


def _hw():
    return load_hardware(None)          # configs/hardware.yaml (mock rig)


def _plan(hw, t_created, dstep=0.005, latency=0.0):
    """Production semantics: action_times[0] = t_created + inference latency
    (phantom/inference/policy.py builds exactly this). The old +10 s fudge
    decoupled the two time origins and HID the u0-anchor bug the executor
    carried until 2026-08-20."""
    H = hw.control.chunk_horizon
    A = hw.control.action_dim
    actions = np.zeros((H, A))
    actions[:, 0] = dstep               # constant +x crawl
    actions[:, 6] = 0.5
    rate = hw.control.action_rate_hz
    return Plan(t_created=t_created,
                t0_pose=np.array([.1, .2, .3, 0, 0, 0], dtype=np.float64),
                actions=actions,
                action_times=t_created + latency + np.arange(H) / rate,
                sigma=np.zeros(4), gate=1.0, p_evt=np.zeros(5), cpk=None)


class _StubGripper:
    def __init__(self):
        self.n = 0

    def get_state(self):
        self.n += 1
        return SimpleNamespace(t_host=time.perf_counter(),
                               position=0.42, obj=3.0)


class _StubRing:
    def __init__(self):
        self.rows = []

    def push(self, ts, **values):
        self.rows.append((ts, values))


def test_submit_fresh_plan_plays_full_chunk():
    """A plan submitted right after inference (now == action_times[0]) must
    play from the START of its chunk. The pre-2026-08-20 anchor used
    t_created, silently discarding latency*rate actions of every chunk (~14
    of 16 at the rig's 1.4 s replans — the postmortem's 35 actions / 17
    replans)."""
    hw = _hw()
    ex = ChunkExecutor(hw, arm=None, gripper=None, safety=None)
    ex._last_cmd = np.array([.15, .25, .35, .01, .02, .03])
    latency = 1.4
    plan = _plan(hw, t_created=time.perf_counter() - latency, latency=latency)
    assert ex.submit(plan)
    assert ex._play_time == pytest.approx(0.0, abs=0.02)
    # continuity: the plan's pose at the seeded play time IS the last command
    target, _ = ex._pose_at(plan, ex._play_time)
    np.testing.assert_allclose(target, ex._last_cmd, atol=1e-9)
    # advancing play time moves the target FORWARD from _last_cmd
    rate = hw.control.action_rate_hz
    t2, _ = ex._pose_at(plan, ex._play_time + 2.0 / rate)
    assert t2[0] > target[0]


def test_submit_stale_plan_skips_elapsed_head():
    """If submit happens 0.45 s after the plan's execution grid began,
    playback resumes at the step whose time it actually is."""
    hw = _hw()
    ex = ChunkExecutor(hw, arm=None, gripper=None, safety=None)
    ex._last_cmd = np.array([.15, .25, .35, .01, .02, .03])
    elapsed = 0.45
    plan = _plan(hw, t_created=time.perf_counter() - elapsed, latency=0.0)
    assert ex.submit(plan)
    rate = hw.control.action_rate_hz
    assert ex._play_time == pytest.approx(elapsed, abs=0.02)
    target, _ = ex._pose_at(plan, ex._play_time)
    np.testing.assert_allclose(target, ex._last_cmd, atol=1e-9)
    # already-elapsed steps are never reported as executed
    assert ex._last_action_k == int(elapsed * rate) - 1


def test_submit_without_last_cmd_keeps_measured_anchor():
    hw = _hw()
    ex = ChunkExecutor(hw, arm=None, gripper=None, safety=None)
    plan = _plan(hw, t_created=time.perf_counter() - 0.3, latency=0.0)
    t0 = plan.t0_pose.copy()
    assert ex.submit(plan)
    np.testing.assert_allclose(plan.t0_pose, t0)   # first plan: anchor untouched
    assert ex._play_time == pytest.approx(0.3, abs=0.02)


def test_second_submit_anchors_to_commanded_not_snapshot_pose():
    """The old behavior: plan 2 rewound to a pose measured before plan 1's
    motion. Now plan 2 must continue from the commanded pose."""
    hw = _hw()
    ex = ChunkExecutor(hw, arm=None, gripper=None, safety=None)
    p1 = _plan(hw, t_created=time.perf_counter() - 0.5)
    assert ex.submit(p1)
    # pretend the executor advanced and commanded up to here:
    ex._last_cmd = np.array([.2, .2, .3, 0, 0, 0])
    stale_snapshot_pose = np.array([.1, .2, .3, 0, 0, 0])   # measured pre-inference
    p2 = _plan(hw, t_created=time.perf_counter() - 0.5)
    p2.t0_pose = stale_snapshot_pose
    assert ex.submit(p2)
    target, _ = ex._pose_at(p2, ex._play_time)
    np.testing.assert_allclose(target, ex._last_cmd, atol=1e-9)
    # and NOT the stale measured pose
    assert not np.allclose(target, stale_snapshot_pose)


def test_executor_polls_gripper_into_ring_while_idle():
    hw = _hw()
    ring = _StubRing()
    ex = ChunkExecutor(hw, arm=None, gripper=_StubGripper(), safety=None,
                       gripper_ring=ring)
    ex.start()
    time.sleep(0.15)
    ex.stop()
    assert len(ring.rows) >= 2, "gripper state must flow with no plan active"
    ts, values = ring.rows[0]
    np.testing.assert_allclose(values["state"], [0.42, 3.0])


def test_snapshot_hard_requires_gripper_state():
    """The zeros(2) substitute is gone: an empty gripper ring must raise."""
    from phantom.deploy.planner import SnapshotBuilder
    hw = _hw()

    class _Ring:
        def __init__(self, n, fields):
            self.n, self.fields = n, fields

        def latest(self, k=1):
            if self.n == 0:
                return np.array([]), {f: np.array([]) for f in self.fields}
            ts = np.array([time.perf_counter()] * min(k, self.n))
            return ts, {f: np.stack([np.asarray(v)] * len(ts))
                        for f, v in self.fields.items()}

    dof = hw.arm.dof
    rings = {
        "camera_scene": _Ring(1, {"color": np.zeros((32, 32, 3), np.uint8)}),
        "arm": _Ring(2, {"q": np.zeros(dof), "qd": np.zeros(dof),
                         "tcp_pose": np.zeros(6), "tcp_speed": np.zeros(6),
                         "ft": np.zeros(6)}),
        "gripper": _Ring(0, {"state": np.zeros(2)}),
    }
    sb = SnapshotBuilder(hw, SimpleNamespace(rings=rings), "vision_only")
    with pytest.raises(AssertionError, match="gripper"):
        sb.build()
    rings["gripper"] = _Ring(1, {"state": np.array([0.4, 3.0])})
    snap = sb.build()
    np.testing.assert_allclose(snap.ur_state[-2:], [0.4, 3.0])


def test_first_replan_prev_chunk_is_normalized_zero_action():
    """Raw zeros in normalized space decode to the demo-mean action (gripper
    -1.5 sigma). 'No previous chunk' must be a normalized PHYSICAL zero."""
    from phantom.data.schema import NormStats
    hw = _hw()
    A = hw.control.action_dim
    mean = np.full(A, 0.4, dtype=np.float32)
    std = np.full(A, 0.2, dtype=np.float32)
    norm = NormStats(mean={"action": mean}, std={"action": std})
    expect = norm.normalize("action", np.zeros((hw.control.chunk_horizon, A),
                                               dtype=np.float32))
    np.testing.assert_allclose(expect, -2.0)   # (0-0.4)/0.2
    # the policy builds exactly this when prev_plan is None (checked at the
    # source level to avoid constructing the full model here)
    import inspect
    import re
    from phantom.inference import policy as P
    src = inspect.getsource(P.PhantomPolicy._batch_from_obs)
    assert re.search(r'normalize\(\s*"action",\s*np\.zeros', src), \
        "prev_chunk fallback must be a NORMALIZED zero action"
