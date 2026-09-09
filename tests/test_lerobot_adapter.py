"""The pi0.5 / LeRobot deploy adapter.

Everything here runs against a STUB policy object — no `lerobot` import, no
checkpoint, no GPU. What is pinned is the half we own: the observation
mapping, the delta/absolute action decoding, horizon padding, the Plan fields
`phantom/deploy/` reads, and the fact that the resulting Plan travels the REAL
policy-server wire and is accepted by the REAL `ChunkExecutor`.
"""
from __future__ import annotations

import threading
import time
from multiprocessing.connection import Listener

import numpy as np
import pytest
import torch

from phantom.config.hardware import load_hardware
from phantom.data import derived as dv
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.governor import SpeedGovernor
from phantom.inference.lerobot_policy import (DEFAULT_IMAGE_KEY,
                                              DEFAULT_STATE_KEY, LeRobotPolicy)
from phantom.inference.policy import ObsSnapshot
from phantom.inference.remote import (PHANTOM_AUTHKEY, PolicyServer,
                                      RemotePolicy)


def _hw():
    return load_hardware(None)          # configs/hardware.yaml (mock rig)


TCP = np.array([0.35, -0.12, 0.28, 0.10, -2.90, 0.05])
GRIP = 0.37


def _snap(hw, t=None, chunk_rows=None):
    """A student-shaped snapshot: scene frame + ur_state, no tactile."""
    ur = np.zeros(hw.ur_state_dim, np.float32)
    dof = hw.arm.dof
    ur[2 * dof:2 * dof + 6] = TCP
    ur[-2] = GRIP
    rng = np.random.default_rng(0)
    return ObsSnapshot(t=time.perf_counter() if t is None else t,
                       rgb=rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
                       wrist_window=np.zeros((hw.wrist_ft.window_len, 6), np.float32),
                       ur_state=ur)


class _StubChunkPolicy:
    """Chunk-native LeRobot policy: `predict_action_chunk` + `reset`."""

    def __init__(self, chunk: np.ndarray):
        self.chunk = np.asarray(chunk, dtype=np.float32)
        self.resets = 0
        self.seen: list[dict] = []

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch):
        self.seen.append(batch)
        return torch.from_numpy(self.chunk)[None]      # (1, T, A)


class _StubStepPolicy:
    """Older surface: only `select_action`, one action per call."""

    def __init__(self, chunk: np.ndarray):
        self.chunk = np.asarray(chunk, dtype=np.float32)
        self.i = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.i = 0

    def select_action(self, batch):
        a = self.chunk[min(self.i, len(self.chunk) - 1)]
        self.i += 1
        return torch.from_numpy(a)


def _adapter(hw, chunk, **kw):
    return LeRobotPolicy(_StubChunkPolicy(chunk), hw, task_text="pick up the egg",
                         device="cpu", **kw)


def _delta_chunk(H, dz=-0.004, grip=0.6):
    a = np.zeros((H, 7), np.float32)
    a[:, 0] = 0.003
    a[:, 2] = dz
    a[:, 4] = 0.001
    a[:, 6] = grip
    return a


# ---------------------------------------------------------------- observation
def test_observation_matches_the_exporter_contract():
    hw = _hw()
    ad = _adapter(hw, _delta_chunk(hw.control.chunk_horizon))
    batch = ad.observation(_snap(hw))

    state = batch[DEFAULT_STATE_KEY]
    assert state.shape == (1, 7) and state.dtype == np.float32
    # [tcp x,y,z, rx,ry,rz (rotvec, base frame), gripper aperture] — the same
    # ur_state slices PlannerLoop.run reads for `tcp_pose` and `grip_now`
    np.testing.assert_allclose(state[0, :6], TCP, rtol=0, atol=1e-6)
    assert state[0, 6] == pytest.approx(GRIP, abs=1e-6)

    img = batch[DEFAULT_IMAGE_KEY]
    assert img.shape == (1, 3, 224, 224) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert batch["task"] == ["pick up the egg"]


def test_task_text_is_live_and_reconfigurable():
    hw = _hw()
    ad = _adapter(hw, _delta_chunk(hw.control.chunk_horizon))
    ad.task_text = "place the waffle"                  # what `configure` does
    assert ad.observation(_snap(hw))["task"] == ["place the waffle"]
    ad.use_task_key = False
    assert "task" not in ad._to_torch(ad.observation(_snap(hw)))


def test_short_ur_state_is_refused_not_silently_sliced():
    hw = _hw()
    ad = _adapter(hw, _delta_chunk(hw.control.chunk_horizon))
    snap = _snap(hw)
    snap.ur_state = snap.ur_state[:8]
    with pytest.raises(ValueError, match="ur_state"):
        ad.observation(snap)


# --------------------------------------------------------------------- plan
def test_delta_chunk_gives_a_plan_the_executor_plays():
    hw = _hw()
    H, rate = hw.control.chunk_horizon, hw.control.action_rate_hz
    chunk = _delta_chunk(H)
    ad = _adapter(hw, chunk)
    snap = _snap(hw)
    plan = ad.replan(snap, None, TCP)

    assert plan.actions.shape == (H, 7)
    np.testing.assert_allclose(plan.actions, chunk, rtol=0, atol=1e-6)
    np.testing.assert_allclose(plan.t0_pose, TCP)
    assert plan.action_times.shape == (H,)
    # action_times[0] = obs.t + inference latency; the grid is 1/rate apart
    assert plan.action_times[0] == pytest.approx(snap.t + plan.latency_s, abs=1e-6)
    np.testing.assert_allclose(np.diff(plan.action_times), 1.0 / rate, atol=1e-9)
    assert plan.cpk is None
    # the planner logs these with .tolist() every replan — they must be arrays
    assert plan.p_evt.tolist() == [0.0] * 5 and plan.sigma.tolist() == [0.0] * H
    assert plan.gate == 0.0

    # ... and the REAL executor accepts it and plays cumulative deltas
    ex = ChunkExecutor(hw, arm=None, gripper=None, safety=None)
    assert ex.submit(plan) is True
    cum = np.cumsum(plan.actions[:, :6], axis=0)
    for k in (0, 5, H - 1):
        target, grip = ex._pose_at(plan, (k + 1 - 1e-9) / rate)
        np.testing.assert_allclose(target, plan.t0_pose + cum[k], atol=1e-6)
        assert grip == pytest.approx(chunk[k, 6], abs=1e-6)
    # exact endpoint: the last commanded pose is t0_pose + the full cumsum
    end, _ = ex._pose_at(plan, (H - 1e-9) / rate)
    np.testing.assert_allclose(end, TCP + chunk[:, :6].sum(axis=0), atol=1e-6)


def test_absolute_chunk_is_differenced_and_replays_the_same_poses():
    """`--action-space absolute`: the executor's cumsum of our deltas must
    reproduce the absolute poses the policy predicted, rotvec guard included."""
    hw = _hw()
    H, rate = hw.control.chunk_horizon, hw.control.action_rate_hz
    poses = np.zeros((H, 7), np.float32)
    for k in range(H):
        poses[k, :3] = TCP[:3] + np.array([0.004 * (k + 1), 0, -0.003 * (k + 1)])
        poses[k, 3:6] = TCP[3:] + np.array([0, 0.002 * (k + 1), 0])
        poses[k, 6] = 0.2 + 0.02 * k
    ad = _adapter(hw, poses, action_space="absolute")
    plan = ad.replan(_snap(hw), None, TCP)

    ex = ChunkExecutor(hw, arm=None, gripper=None, safety=None)
    assert ex.submit(plan)
    for k in range(H):
        target, grip = ex._pose_at(plan, (k + 1 - 1e-9) / rate)
        np.testing.assert_allclose(target, poses[k, :6], atol=1e-6)
        assert grip == pytest.approx(poses[k, 6], abs=1e-6)
    # first delta is measured from the MEASURED pose, not from poses[0]
    np.testing.assert_allclose(plan.actions[0, :6], dv.pose_delta(TCP, poses[0, :6]),
                               atol=1e-6)


def test_absolute_rotvec_wrap_uses_pose_delta_not_raw_subtraction():
    """A rotvec that flips sign between two representations of the SAME
    orientation must produce a small delta (the executor cumsums in rotvec
    space; a raw subtraction would command a ~2*pi whip)."""
    hw = _hw()
    tcp = np.array([0.3, 0.0, 0.3, 0.0, 3.10, 0.0])
    flipped = np.array([0.3, 0.0, 0.3, 0.0, -3.10, 0.0])   # ~same orientation
    poses = np.tile(np.concatenate([flipped, [0.5]]), (hw.control.chunk_horizon, 1))
    ad = _adapter(hw, poses.astype(np.float32), action_space="absolute")
    plan = ad.replan(_snap(hw), None, tcp)
    assert np.linalg.norm(plan.actions[0, 3:6]) < 0.2, plan.actions[0]
    np.testing.assert_allclose(plan.actions[1:, :6], 0.0, atol=1e-6)


# ------------------------------------------------------------------ horizon
def test_long_chunk_is_truncated_to_the_deploy_horizon():
    """pi05's native chunk (50) is far longer than our 16-step grid."""
    hw = _hw()
    H = hw.control.chunk_horizon
    chunk = _delta_chunk(50)
    chunk[:, 1] = np.arange(50) * 1e-3            # make each row identifiable
    plan = _adapter(hw, chunk).replan(_snap(hw), None, TCP)
    assert plan.actions.shape == (H, 7)
    np.testing.assert_allclose(plan.actions, chunk[:H], atol=1e-6)
    assert plan.diag["chunk_steps"] == 50 and plan.diag["padded_steps"] == 0


def test_short_chunk_pads_by_holding_pose_and_aperture():
    hw = _hw()
    H = hw.control.chunk_horizon
    chunk = _delta_chunk(4, grip=0.8)
    plan = _adapter(hw, chunk).replan(_snap(hw), None, TCP)
    assert plan.actions.shape == (H, 7)
    np.testing.assert_allclose(plan.actions[:4], chunk, atol=1e-6)
    # padded rows: zero POSE delta (hold), last commanded aperture
    np.testing.assert_allclose(plan.actions[4:, :6], 0.0, atol=1e-12)
    np.testing.assert_allclose(plan.actions[4:, 6], 0.8, atol=1e-6)
    assert plan.diag["padded_steps"] == H - 4


def test_repeat_padding_extends_the_last_row_when_asked():
    hw = _hw()
    H = hw.control.chunk_horizon
    chunk = _delta_chunk(4, grip=0.8)
    plan = _adapter(hw, chunk, pad_mode="repeat").replan(_snap(hw), None, TCP)
    np.testing.assert_allclose(plan.actions[4:], np.repeat(chunk[-1:], H - 4, 0),
                               atol=1e-6)


def test_empty_and_non_finite_chunks_raise():
    hw = _hw()
    with pytest.raises(ValueError, match="empty"):
        _adapter(hw, np.zeros((0, 7), np.float32)).replan(_snap(hw), None, TCP)
    bad = _delta_chunk(hw.control.chunk_horizon)
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _adapter(hw, bad).replan(_snap(hw), None, TCP)
    with pytest.raises(ValueError, match="action dims"):
        _adapter(hw, np.zeros((4, 6), np.float32)).replan(_snap(hw), None, TCP)


def test_select_action_fallback_reproduces_the_chunk():
    """A policy exposing only `select_action` is driven H times."""
    hw = _hw()
    H = hw.control.chunk_horizon
    chunk = _delta_chunk(H)
    chunk[:, 1] = np.arange(H) * 1e-3
    ad = LeRobotPolicy(_StubStepPolicy(chunk), hw, device="cpu")
    plan = ad.replan(_snap(hw), None, TCP)
    np.testing.assert_allclose(plan.actions, chunk, atol=1e-6)


# ---------------------------------------------------------- deploy semantics
def test_zero_sigma_keeps_the_speed_governor_at_full_scale():
    hw = _hw()
    plan = _adapter(hw, _delta_chunk(hw.control.chunk_horizon)) \
        .replan(_snap(hw), None, TCP)
    gov = SpeedGovernor(hw.safety.governor)
    assert gov.scale_profile(plan.sigma) == pytest.approx(1.0)


def test_unsupported_deploy_flags_are_refused_not_ignored():
    hw = _hw()
    ad = _adapter(hw, _delta_chunk(hw.control.chunk_horizon))
    ad.assert_deploy_flags(terminal_veto=False, parity_fixes=False, k_seeds=1)
    for kw, needle in ((dict(terminal_veto=True), "ACC head"),
                       (dict(parity_fixes=True), "prev_chunk"),
                       (dict(drop_video=True), "only observation"),
                       (dict(k_seeds=4), "one chunk per replan")):
        with pytest.raises(RuntimeError, match=needle):
            ad.assert_deploy_flags(**kw)


def test_reset_episode_clears_the_queue_and_a_seed_reaches_the_rng():
    hw = _hw()
    ad = _adapter(hw, _delta_chunk(hw.control.chunk_horizon))
    ad.reset_episode()
    assert ad.policy.resets == 1
    # exactly what PolicyServer._handle_owned does for ("reset_episode", seed)
    ad.rf._gen = torch.Generator().manual_seed(1234)
    ad.rf.reset_episode_noise()
    assert ad.policy.resets == 2
    assert torch.initial_seed() == 1234


# ------------------------------------------------------------------- server
@pytest.fixture
def live_server():
    hw = _hw()
    policy = _adapter(hw, _delta_chunk(hw.control.chunk_horizon))
    srv = PolicyServer(policy, ckpt="pi05_ckpt_dir", ckpt_sha="0badcafe1234")
    listener = Listener(("127.0.0.1", 0), authkey=PHANTOM_AUTHKEY)
    port = listener.address[1]
    threading.Thread(target=srv.serve_forever,
                     kwargs={"port": port, "listener": listener},
                     daemon=True).start()
    time.sleep(0.05)
    yield hw, srv, port
    try:
        listener.close()
    except Exception:
        pass


def test_info_and_replan_round_trip_over_the_real_wire(live_server):
    hw, srv, port = live_server
    # PICK.sh's probe: ckpt + idle/busy + sha, answered without owning anything
    info = RemotePolicy.probe(("127.0.0.1", port))
    assert info["ckpt"] == "pi05_ckpt_dir" and info["ckpt_sha"] == "0badcafe1234"
    assert info["busy"] is False and info["wrench_baseline_rows"] == 0

    client = RemotePolicy(("127.0.0.1", port), {"task_text": "pick up the egg",
                                                "guidance": 1.0})
    assert client.info["busy"] is True
    assert srv.policy.task_text == "pick up the egg"
    assert client.wrench_baseline_rows == 0

    client.remote_reset(7)
    assert srv.policy.policy.resets >= 1

    snap = _snap(hw)
    plan = client.replan(snap, None, TCP)
    assert plan.actions.shape == (hw.control.chunk_horizon, 7)
    assert plan.cpk is None and plan.diag["policy"] == "lerobot"
    ex = ChunkExecutor(hw, arm=None, gripper=None, safety=None)
    assert ex.submit(plan) is True
    # a second replan with the first plan as prev_plan (token round-trip)
    plan2 = client.replan(_snap(hw), plan, TCP)
    assert ex.submit(plan2) is True
    client.close()
    time.sleep(0.1)
    assert RemotePolicy.probe(("127.0.0.1", port))["busy"] is False


# ------------------------------------------------------------------ digests
def test_dir_digest_tracks_the_weight_files(tmp_path):
    from phantom.scripts.lerobot_server import dir_digest
    d = tmp_path / "pi05"
    d.mkdir()
    (d / "config.json").write_text('{"type": "pi05"}')
    (d / "model.safetensors").write_bytes(b"weights-v1")
    first = dir_digest(str(d))
    assert first and len(first) == 12
    assert dir_digest(str(d)) == first                   # stable
    (d / "model.safetensors").write_bytes(b"weights-v2")
    assert dir_digest(str(d)) != first                   # content-sensitive
    assert dir_digest(str(tmp_path / "nope")) is None
