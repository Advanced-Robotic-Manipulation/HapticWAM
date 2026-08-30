"""tools/replay_rig.py — offline replay of recorded deploy episodes.

Two things have to hold for the replay to be worth anything (REVIEW_SYNTHESIS
P1 / GATE G0):

1. the snapshot it rebuilds from the zarr streams at a replan's time must be
   the snapshot `SnapshotBuilder.build()` actually handed the policy on the
   rig (ur_state + wrist window at least), and
2. the tool must produce one finite row per ACCEPTED replan.

Both are checked against a real mock-driver deploy episode produced through
the same `DeploymentRuntime` path tests/test_deploy_parity_fixes.py uses, so
the trace, the zarr streams and the clock offset are the genuine artifacts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from phantom.config.hardware import HardwareConfig
from phantom.config.paths import load_paths
from phantom.data.schema import STREAM_ARM_TCP_POSE, STREAM_GRIPPER
from phantom_test_utils import make_small_hw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))


def _cosmos_available() -> bool:
    try:
        from phantom.backbone import loader as bl
        bl.setup_cosmos(load_paths())
        return True
    except Exception:
        return False


class _RecordingPolicy:
    """Constant crawl (the mock dry-run policy), but it KEEPS every snapshot
    it was handed and burns ~one rig replan period so the streams advance."""

    def __init__(self, hw):
        self.hw = hw
        self.snaps: list = []

    def reset_episode(self):
        pass

    def replan(self, snap, prev_plan, tcp_pose):
        from phantom.inference.policy import Plan
        hw = self.hw
        time.sleep(0.2)
        self.snaps.append(snap)
        H, A = hw.control.chunk_horizon, hw.control.action_dim
        actions = np.zeros((H, A))
        actions[:, 0], actions[:, 2] = 5e-4, -1e-3
        actions[:, 6] = 0.3
        now = time.perf_counter()
        return Plan(t_created=now, t0_pose=np.asarray(tcp_pose, np.float64).copy(),
                    actions=actions,
                    action_times=now + 0.05 + np.arange(H) / hw.control.action_rate_hz,
                    sigma=np.zeros(4), gate=1.0, p_evt=np.zeros(5), cpk=None,
                    diag={"nfe": 5, "guidance": 1.0})


@pytest.fixture(scope="module")
def mock_deploy_episode(tmp_path_factory):
    """A real recorded deploy episode (zarr streams + planner_trace.json)."""
    from phantom.deploy.runtime import DeploymentRuntime
    hw = make_small_hw()
    pol = _RecordingPolicy(hw)
    out = tmp_path_factory.mktemp("deploy")
    with DeploymentRuntime(hw, pol, mode="teacher", out_root=out) as rt:
        res = rt.run_episode(task="whiteboard", max_replans=5)
    # `replan_cap` since 2026-08-30: the loop's own caps are named now
    assert res.stopped_reason in (None, "replan_cap")
    assert res.episode_path is not None
    hw_yaml = out / "hardware.small.yaml"
    hw_yaml.write_text(yaml.safe_dump(hw.model_dump(mode="json")))
    return hw, res.episode_path, pol.snaps, hw_yaml


# ---------------------------------------------------------------------------
# 1. snapshot parity with SnapshotBuilder
# ---------------------------------------------------------------------------

def test_rebuilt_snapshot_matches_the_snapshot_builder(mock_deploy_episode):
    from replay_rig import RigEpisode
    hw, ep_path, deploy_snaps, _ = mock_deploy_episode
    ep = RigEpisode(ep_path, hw)
    assert len(ep.trace) == len(deploy_snaps)

    ts_arm = ep.ts(STREAM_ARM_TCP_POSE)
    poses = np.asarray(ep.reader.data(STREAM_ARM_TCP_POSE)[:], np.float64)
    ts_grip = ep.ts(STREAM_GRIPPER)
    dof = hw.arm.dof
    compared = 0
    for i, dsnap in enumerate(deploy_snaps):
        t = ep.t_master(i)
        if t - hw.wrist_ft.window_s < ts_arm[0]:
            continue          # the ring reached back before recording started
        # the row deploy read: an EXACT match of the recorded tcp_pose row
        hit = np.flatnonzero(
            (poses.astype(np.float32) == dsnap.ur_state[2 * dof:2 * dof + 6]).all(1))
        assert len(hit), f"replan {i}: deploy's tcp_pose is not a recorded row"
        # selection rule parity: deploy stamps snap.t BEFORE reading the rings,
        # so its row may be at most one sample newer than the last row <= t
        i_replay = ep.last_leq(STREAM_ARM_TCP_POSE, t)
        assert np.min(np.abs(hit - i_replay)) <= 1, \
            f"replan {i}: replay picked row {i_replay}, deploy used {hit}"
        # rebuilt AT that row's timestamp the snapshot must match exactly
        i_deploy = int(hit[np.argmin(np.abs(hit - i_replay))])
        snap, _ = ep.snapshot(float(ts_arm[i_deploy]), None, teacher=True)
        assert np.array_equal(snap.ur_state[:4 * dof], dsnap.ur_state[:4 * dof])
        assert np.allclose(snap.wrist_window, dsnap.wrist_window, atol=1e-5)
        assert snap.wrist_window.shape == (hw.wrist_ft.window_len, 6)
        # the gripper row is read after the arm row on the rig — within one
        assert abs(ep.last_leq(STREAM_GRIPPER, float(ts_arm[i_deploy]))
                   - ep.last_leq(STREAM_GRIPPER, t)) <= 1
        assert snap.fields.shape[0] == len(hw.tactile.sensors)
        assert snap.gel is not None and snap.contact_state is not None
        compared += 1
    assert compared >= 2, "no replan had a fully covered wrist window"


def test_measured_prev_chunk_is_the_measured_tcp_delta(mock_deploy_episode):
    """--prev-chunk measured must reproduce training's action semantics:
    chained pose deltas of the MEASURED TCP on the 10 Hz grid ending at t."""
    from replay_rig import RigEpisode
    hw, ep_path, _, _ = mock_deploy_episode
    ep = RigEpisode(ep_path, hw)
    t = ep.t_master(len(ep.trace) - 1)
    prev = ep.measured_prev_chunk(t)
    assert prev.shape == (hw.control.chunk_horizon, hw.control.action_dim)
    assert np.isfinite(prev).all()
    poses = np.asarray(ep.reader.data(STREAM_ARM_TCP_POSE)[:], np.float64)
    span = float(poses[ep.nearest(STREAM_ARM_TCP_POSE, t - 0.1), 2]
                 - poses[ep.nearest(STREAM_ARM_TCP_POSE, t - 1.6), 2])
    assert prev[:, :3].sum(0)[2] == pytest.approx(span, abs=2e-4)


# ---------------------------------------------------------------------------
# 2. end-to-end CLI on a random tiny backbone (CPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _cosmos_available(), reason="cosmos repo not importable")
def test_replay_rig_cli_tiny_writes_one_row_per_accepted_replan(
        mock_deploy_episode, tmp_path, monkeypatch):
    import replay_rig
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    out = tmp_path / "replay.json"
    monkeypatch.setattr(sys, "argv", [
        "replay_rig.py", "--tiny", "--hardware", str(hw_yaml),
        "--episodes", str(ep_path), "--seeds", "2", "--nfe", "1",
        "--persistent-noise", "--prev-chunk", "proposal",
        "--prev-cpk", "chained", "--out", str(out)])
    assert replay_rig.main() == 0

    payload = json.loads(out.read_text())
    trace = json.loads((ep_path / "planner_trace.json").read_text())
    n_accepted = sum(1 for r in trace if r["accepted"])
    (epr,) = payload["episodes"]
    assert epr["n_replans"] == n_accepted
    assert [r["replan"] for r in epr["rows"]] == \
        [i for i, r in enumerate(trace) if r["accepted"]]
    for r in epr["rows"]:
        for k in ("head_dz", "tail_dz", "chunk_dz", "grip_max", "close_step",
                  "head_dz_std", "head_dz_err", "trace_head_dz"):
            assert np.isfinite(r[k]), f"{k} is not finite"
        assert isinstance(r["trace_in_spread"], bool)
        assert 0.0 <= r["close_step"] <= hw.control.chunk_horizon
    # the mock policy commanded a flat -1 mm/step descent -> -9 mm over the head
    assert epr["rows"][0]["trace_head_dz"] == pytest.approx(-9.0, abs=1e-6)
    assert 0.0 <= payload["summary"]["trace_in_spread"] <= 1.0


@pytest.mark.skipif(not _cosmos_available(), reason="cosmos repo not importable")
def test_prev_chunk_swaps_change_the_conditioning(mock_deploy_episode, tmp_path):
    """E3's swap must actually reach the batch: `zeros` is the normalized-zero
    first-replan path, `measured` the training-parity chunk."""
    import replay_rig
    from phantom.inference.policy import Plan
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    ep = replay_rig.RigEpisode(ep_path, hw)
    args = replay_rig.argparse.Namespace(tiny=True, ckpt=None, nfe=1, guidance=1.0,
                                         seeds=1, persistent_noise=False)
    policy = replay_rig.build_policy(args, hw)
    t = ep.t_master(len(ep.trace) - 1)
    snap, _ = ep.snapshot(t, None, teacher=True)
    zeros = policy._batch_from_obs(snap, None)["prev_chunk"]
    measured = policy._batch_from_obs(snap, Plan(
        t_created=t, t0_pose=np.zeros(6), actions=ep.measured_prev_chunk(t),
        action_times=np.zeros(1), sigma=np.zeros(1), gate=0.0, p_evt=np.zeros(1),
        cpk=None))["prev_chunk"]
    assert zeros.shape == measured.shape
    assert not torch.allclose(zeros, measured)
