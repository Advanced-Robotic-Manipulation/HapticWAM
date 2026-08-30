"""Small fixes from the 2026-08-28 review, batch 2026-08-29.

1  §1.11 guidance — the CFG null branch's ACC inputs must differ from the
   conditional branch's ONLY in the observations (PhantomRF.align_guidance_acc).
2  Shared-memory ring names must not collide between two sessions that start in
   the same wall-clock second (workers.new_session_id).
3  P9 — a deploy rollout must carry the BASE hardware config hash, not the hash
   of the run-time safety overrides, or train_teacher's CONFIG DRIFT gate
   refuses every rollout.
4  P9 — rollout `actions.zarr` must be re-derived from the MEASURED TCP on the
   action grid (tools/rederive_rollout_actions.py), keeping the executor's
   proposal stream as `actions_plan.zarr`.
5  The post-episode verdict prompt must be able to record damage.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.data.derived import pose_delta
from phantom.data.episode_store import EpisodeReader, EpisodeWriter
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ACTIONS_PLAN,
                                 STREAM_ARM_TCP_POSE, STREAM_GRIPPER,
                                 EpisodeMeta)
from phantom.recording import workers as W
from phantom.recording.recorder import EpisodeRecorder
from phantom.scripts import run_deploy as RD
from phantom_test_utils import make_small_hw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rederive_rollout_actions as RRA  # noqa: E402


# ===========================================================================
# 1 — guidance: the null branch's ACC inputs
# ===========================================================================

def _acc(summary, wrist=None):
    torch = pytest.importorskip("torch")
    from phantom.model.acc import AccInputs
    return AccInputs(
        wrist_feat_B_D=torch.zeros(2, 4) if wrist is None else wrist,
        intent_B_H_A=torch.zeros(2, 3, 7),
        prev_cpk_summary_B_S=summary,
        react_score_B=None)


def test_align_guidance_acc_shares_the_conditional_summary():
    """CFG's v_obs - v_null is only 'the effect of the observations' if the two
    branches agree on everything else. The prev-contact summary is INTENT (the
    previous replan's own output), like prev_chunk, which _null_obs_batch
    already leaves untouched."""
    torch = pytest.importorskip("torch")
    cond = _acc(torch.full((2, 5), 3.0))
    null = _acc(torch.full((2, 5), -1.0))
    out = RD_align(cond, null)
    assert out is null                                  # mutated in place
    assert torch.equal(null.prev_cpk_summary_B_S, cond.prev_cpk_summary_B_S)


def test_align_guidance_acc_leaves_the_observation_inputs_alone():
    """It must NOT null-align the observations — those are the whole point of
    the guidance direction."""
    torch = pytest.importorskip("torch")
    cond = _acc(torch.ones(2, 5), wrist=torch.ones(2, 4))
    null = _acc(torch.zeros(2, 5), wrist=torch.zeros(2, 4))
    RD_align(cond, null)
    assert torch.equal(null.wrist_feat_B_D, torch.zeros(2, 4))
    assert not torch.equal(null.wrist_feat_B_D, cond.wrist_feat_B_D)


def test_align_guidance_acc_is_a_noop_without_guidance():
    torch = pytest.importorskip("torch")
    assert RD_align(_acc(torch.zeros(2, 5)), None) is None


def RD_align(cond, null):
    from phantom.model.rf import PhantomRectifiedFlow
    return PhantomRectifiedFlow.align_guidance_acc(cond, null)


# ===========================================================================
# 2 — shared-memory ring name collisions
# ===========================================================================

def test_session_ids_in_the_same_second_are_distinct(monkeypatch):
    """The exact collision: two pytest runs (or two rig relaunches) starting in
    the same wall-clock second used to derive the same shm segment names."""
    monkeypatch.setattr(W.time, "time", lambda: 1_700_000_123.4)
    ids = {W.new_session_id() for _ in range(64)}
    assert len(ids) == 64
    # seconds stay the leading component (segments stay greppable in /dev/shm)
    assert all(i.startswith("123-") for i in ids)


def test_two_sessions_in_the_same_second_get_distinct_ring_names(monkeypatch):
    monkeypatch.setattr(W.time, "time", lambda: 1_700_000_000.4)
    hw = make_small_hw()
    a = b = None
    try:
        a = W.build_session_rings(hw, W.new_session_id())
        b = W.build_session_rings(hw, W.new_session_id())   # used to raise
        names_a = {r.name for r in a.values()}
        names_b = {r.name for r in b.values()}
        assert names_a and not (names_a & names_b)
    finally:
        for rings in (a, b):
            for r in (rings or {}).values():
                r.close()


def test_the_old_scheme_really_did_collide():
    """Guards the test above: identical session ids MUST still raise, otherwise
    the uniqueness test proves nothing."""
    hw = make_small_hw()
    fixed = f"collide{time.time_ns()}"
    first = W.build_session_rings(hw, fixed)
    try:
        with pytest.raises(FileExistsError):
            W.build_session_rings(hw, fixed)
    finally:
        for r in first.values():
            r.close()


# ===========================================================================
# 3 — deploy episodes carry the BASE hardware config hash
# ===========================================================================

class _CrawlPolicy:
    """Constant crawl (same shape as the mock dry-run policy)."""

    def __init__(self, hw):
        self.hw = hw

    def reset_episode(self):
        pass

    def replan(self, snap, prev_plan, tcp_pose):
        from phantom.inference.policy import Plan
        hw = self.hw
        H, A = hw.control.chunk_horizon, hw.control.action_dim
        actions = np.zeros((H, A))
        actions[:, 0] = 5e-4
        actions[:, 6] = 0.3
        now = time.perf_counter()
        return Plan(t_created=now,
                    t0_pose=np.asarray(tcp_pose, dtype=np.float64).copy(),
                    actions=actions,
                    action_times=now + 0.05 + np.arange(H) / hw.control.action_rate_hz,
                    sigma=np.zeros(4), gate=1.0, p_evt=np.zeros(5), cpk=None)


@pytest.fixture(scope="module")
def overridden_rollout(tmp_path_factory):
    """A mock deploy episode recorded under z-floor + hitbox + speed-cap
    overrides, exactly as run_deploy applies them."""
    from phantom.deploy.runtime import DeploymentRuntime
    from phantom.deploy.safety import (apply_hitbox, apply_tcp_speed_limit,
                                       apply_z_floor)
    base = make_small_hw()
    lo = np.array([-0.4, -0.4, 0.05])
    hi = np.array([0.4, 0.4, 0.5])
    hw = apply_hitbox(base, lo, hi, 0.03)
    hw = apply_z_floor(hw, 0.10)
    hw = apply_tcp_speed_limit(hw, 0.05)
    assert hw.config_hash() != base.config_hash(), \
        "the overrides must actually change the hash, or this test is vacuous"
    overrides = {
        "hitbox_m": {"x": list(hw.safety.hitbox_m.x), "y": list(hw.safety.hitbox_m.y),
                     "z": list(hw.safety.hitbox_m.z), "margin_m": 0.03},
        "z_floor_m": float(hw.safety.workspace_m.z[0]),
        "tcp_speed_m_s": float(hw.arm.limits.tcp_speed_m_s),
    }
    out = tmp_path_factory.mktemp("deploy_overridden")
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="student", out_root=out,
                           base_hw=base, deploy_overrides=overrides) as rt:
        res = rt.run_episode(task="whiteboard", max_replans=2,
                             policy_name="student")
    assert res.episode_path is not None
    return base, hw, Path(res.episode_path)


def test_rollout_carries_the_base_config_hash(overridden_rollout):
    """Without this, hw.config_hash() is the OVERRIDDEN config's and every
    rollout trips train_teacher's CONFIG DRIFT gate against the demos."""
    base, hw, ep = overridden_rollout
    meta = EpisodeMeta.load(ep / "meta.json")
    assert meta.config_hash == base.config_hash()
    assert meta.config_hash != hw.config_hash()


def test_rollout_records_the_safety_overrides(overridden_rollout):
    base, hw, ep = overridden_rollout
    meta = EpisodeMeta.load(ep / "meta.json")
    assert meta.deploy_overrides["z_floor_m"] == pytest.approx(0.10)
    assert meta.deploy_overrides["tcp_speed_m_s"] == pytest.approx(0.05)
    assert meta.deploy_overrides["hitbox_m"]["margin_m"] == pytest.approx(0.03)


def test_window_sampler_sees_no_config_drift(overridden_rollout):
    """The end the fix is for: the drift counter train_teacher hard-errors on."""
    pytest.importorskip("torch")
    from phantom.data.schema import NormStats
    from phantom.data.windows import WindowSampler
    base, hw, ep = overridden_rollout
    sampler = WindowSampler(base, _BB(), NormStats.identity(), student=False)
    sampler._ep(ep)
    assert sampler.n_config_drift == 0


class _BB(SimpleNamespace):
    """Minimal backbone stand-in: _ep() only reads meta, never the backbone."""


def test_a_recorder_that_presets_no_hash_still_gets_stamped(tmp_path):
    """Teleop and everything else keep the old behaviour."""
    hw = make_small_hw()
    meta = EpisodeMeta(task="egg")
    EpisodeWriter(tmp_path / "ep_x", hw, meta)
    assert meta.config_hash == hw.config_hash()


# ===========================================================================
# 4 — rollout actions.zarr re-derivation
# ===========================================================================

def _traj(t: np.ndarray) -> np.ndarray:
    """A smooth 6-D TCP path: 40 mm/s in x, a slow descent in z, tiny wrist
    rotation. Deliberately NOT axis-aligned so a dropped rotvec term shows."""
    p = np.zeros((len(t), 6))
    p[:, 0] = 0.20 + 0.040 * t
    p[:, 1] = -0.10 + 0.010 * np.sin(0.7 * t)
    p[:, 2] = 0.30 - 0.020 * t
    p[:, 3] = 2.20 + 0.01 * t
    p[:, 4] = 0.05 * np.cos(0.5 * t)
    p[:, 5] = -0.30
    return p


@pytest.fixture
def synthetic_rollout(tmp_path):
    """A deploy episode whose `actions` stream is an executor PROPOSAL at
    governor-warped times, with measured TCP + gripper streams underneath."""
    hw = make_small_hw()
    ep = tmp_path / "ep_student_whiteboard_1_000"
    meta = EpisodeMeta(task="whiteboard", policy="student", success=True,
                       status="finalized")
    w = EpisodeWriter(ep, hw, meta)

    t_arm = np.arange(0.0, 6.0, 1.0 / 125.0)          # measured at RTDE rate
    w.append(STREAM_ARM_TCP_POSE, t_arm, _traj(t_arm).astype(np.float32))
    t_g = np.arange(0.0, 6.0, 1.0 / 20.0)             # measured gripper
    grip = np.zeros((len(t_g), 2), dtype=np.float32)
    grip[:, 0] = np.clip(0.05 * t_g, 0.0, 0.9)
    grip[:, 1] = 3.0
    w.append(STREAM_GRIPPER, t_g, grip)

    # the proposal: warped cadence (7 Hz, not 10) and values that are NOT the
    # measured motion — the whole point of re-deriving
    t_plan = np.arange(0.3, 5.7, 1.0 / 7.0)
    plan = np.zeros((len(t_plan), 7), dtype=np.float32)
    plan[:, 0] = -0.026                               # the rig's under-commit
    plan[:, 6] = 0.42
    w.append(STREAM_ACTIONS, t_plan, plan)
    w.finalize(success=True)
    return hw, ep, t_plan, plan


def test_rederived_actions_integrate_to_the_measured_tcp(synthetic_rollout):
    hw, ep, t_plan, plan = synthetic_rollout
    assert RRA.rederive_actions(ep, hw) == "rederived"
    r = EpisodeReader(ep)
    ts = r.ts(STREAM_ACTIONS)
    act = np.asarray(r.data(STREAM_ACTIONS)[:], dtype=np.float64)

    rate = hw.control.action_rate_hz
    assert np.allclose(np.diff(ts), 1.0 / rate, atol=1e-9)   # on the 10 Hz grid
    # cumsum from the pose one grid step BEFORE the first action reproduces the
    # measured trajectory (within the 8 ms measurement quantisation x 40 mm/s)
    start = _traj(np.array([ts[0] - 1.0 / rate]))[0, :3]
    recon = start + np.cumsum(act[:, :3], axis=0)
    assert np.abs(recon - _traj(ts)[:, :3]).max() < 1e-3     # 1 mm

    # rotation is a real delta too, not zeros
    assert np.abs(act[:, 3:6]).max() > 0
    # gripper column is the COMMANDED aperture (validation 2026-08-30 F20):
    # a demo records grip_cmd, and the measured position is the grasp OUTCOME
    assert np.allclose(act[:, 6], plan[0, 6])
    assert np.abs(act[:, 6] - np.clip(0.05 * ts, 0.0, 0.9)).max() > 0.01
    # the POSE channels are emphatically not the proposal any more
    assert np.abs(act[:, 0] - plan[0, 0]).min() > 1e-4


def test_the_proposal_stream_is_preserved(synthetic_rollout):
    hw, ep, t_plan, plan = synthetic_rollout
    RRA.rederive_actions(ep, hw)
    r = EpisodeReader(ep)
    assert r.has(STREAM_ACTIONS_PLAN)
    assert np.allclose(r.ts(STREAM_ACTIONS_PLAN), t_plan)
    assert np.allclose(np.asarray(r.data(STREAM_ACTIONS_PLAN)[:]), plan)
    assert RRA.REDERIVED_TAG in EpisodeMeta.load(ep / "meta.json").tags


def test_rederivation_is_idempotent(synthetic_rollout):
    hw, ep, t_plan, plan = synthetic_rollout
    assert RRA.rederive_actions(ep, hw) == "rederived"
    first = np.asarray(EpisodeReader(ep).data(STREAM_ACTIONS)[:]).copy()
    assert RRA.rederive_actions(ep, hw) == "skipped"
    assert RRA.rederive_actions(ep, hw) == "skipped"
    again = np.asarray(EpisodeReader(ep).data(STREAM_ACTIONS)[:])
    assert np.array_equal(first, again)
    assert EpisodeMeta.load(ep / "meta.json").tags.count(RRA.REDERIVED_TAG) == 1


def test_rederived_rows_match_the_teleop_derivation(synthetic_rollout):
    """session.py records pose_delta(prev_measured, measured) at each tick —
    row for row, this must be the same function of the same samples."""
    hw, ep, t_plan, plan = synthetic_rollout
    r0 = EpisodeReader(ep)
    tcp_ts = r0.ts(STREAM_ARM_TCP_POSE)
    tcp = np.asarray(r0.data(STREAM_ARM_TCP_POSE)[:], dtype=np.float64)
    grip_ts = r0.ts(STREAM_GRIPPER)
    RRA.rederive_actions(ep, hw)
    act = np.asarray(EpisodeReader(ep).data(STREAM_ACTIONS)[:], dtype=np.float64)
    grid = RRA.action_grid(tcp_ts, grip_ts, hw.control.action_rate_hz)
    idx = RRA._latest_at_or_before(tcp_ts, grid)
    want = np.stack([pose_delta(tcp[a], tcp[b])
                     for a, b in zip(idx[:-1], idx[1:])])
    assert np.array_equal(act[:, :6].astype(np.float32), want)


def test_teleop_episodes_are_not_rollouts(synthetic_rollout):
    hw, ep, _, _ = synthetic_rollout
    assert RRA.is_rollout(EpisodeMeta(task="egg", policy="student"))
    assert not RRA.is_rollout(EpisodeMeta(task="egg", policy="teleop"))
    assert not RRA.is_rollout(EpisodeMeta(task="egg"))


# ===========================================================================
# 5 — post-episode damage verdict
# ===========================================================================

@pytest.fixture
def labelled(tmp_path):
    ep = tmp_path / "ep_student_egg_1_000"
    ep.mkdir()
    EpisodeMeta(task="egg", policy="student", status="aborted",
                tags=["unlabeled"]).save(ep / "meta.json")
    rec = SimpleNamespace(out_root=tmp_path)
    rec.relabel = lambda *a, **kw: EpisodeRecorder.relabel(rec, *a, **kw)
    return rec, ep


@pytest.mark.parametrize("ans,code,damage", [
    ("s", "s", False), ("sd", "s", True),
    ("f", "f", False), ("fd", "f", True),
    ("c", "c", False), ("cd", "c", True),
    ("FD knocked the pack off", "f", True),
    ("f d", "f", False),                 # 'd' as a note word is NOT the flag
    ("success", "s", False),             # the old lenient first-char behaviour
    ("", "", False), ("nope", "", False),
])
def test_parse_verdict(ans, code, damage):
    assert RD.parse_verdict(ans) == (code, damage)


@pytest.mark.parametrize("ans,success", [("sd", True), ("fd broke the egg", False)])
def test_damage_is_written_onto_the_episode(labelled, ans, success):
    recorder, ep = labelled
    assert RD.label_episode(recorder, ep, ans) == ans[0]
    m = EpisodeMeta.load(ep / "meta.json")
    assert m.damage is True and "damaged" in m.tags
    assert m.success is success and m.status == "finalized"
    assert "DAMAGE" in m.notes


def test_no_damage_flag_leaves_the_field_false(labelled):
    recorder, ep = labelled
    RD.label_episode(recorder, ep, "s clean run")
    m = EpisodeMeta.load(ep / "meta.json")
    assert m.damage is False and "damaged" not in m.tags


def test_contaminated_can_also_be_damaged(labelled):
    recorder, ep = labelled
    assert RD.label_episode(recorder, ep, "cd bumped and cracked") == "c"
    m = EpisodeMeta.load(ep / "meta.json")
    assert m.damage is True and m.success is None
    assert "contaminated" in m.tags and "damaged" in m.tags


def test_damage_field_is_backward_compatible(tmp_path):
    """Old meta.json files have no `damage` key at all."""
    ep = tmp_path / "ep_old"
    ep.mkdir()
    (ep / "meta.json").write_text('{"task": "egg", "success": true}')
    m = EpisodeMeta.load(ep / "meta.json")
    assert m.damage is False
    assert "damage" in m.to_dict()


def test_the_prompt_advertises_the_flag():
    assert "d" in RD.VERDICT_PROMPT and "DAMAGE" in RD.VERDICT_PROMPT
