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
        assert snap.wrist_window.shape == (hw.wrist_ft.window_len, 6)
        # The wrist window is anchored ONE READ EARLIER than ur_state:
        # SnapshotBuilder.build() takes rings["arm"].latest(self._n_arm) and
        # anchors the F/T grid at ts_a[-1], then takes a SECOND
        # rings["arm"].latest(1) for ur_state. A 125 Hz sample landing between
        # those two reads leaves ur_state one row newer than its own wrist
        # anchor, which made this assertion fail on ~1 run in 3 (F15, root-caused
        # 2026-08-30 — the flake was the race, not the rebuild). Both anchors are
        # therefore admissible; an exact match at ONE of them is still a strict
        # parity check, because the mock F/T stream is fresh noise per sample.
        anchors = [i_deploy] + ([i_deploy - 1] if i_deploy > 0 else [])
        windows = [ep.snapshot(float(ts_arm[a]), None, teacher=True)[0].wrist_window
                   for a in anchors]
        assert any(np.allclose(w, dsnap.wrist_window, atol=1e-5) for w in windows), \
            (f"replan {i}: deploy's wrist window matches neither the rebuild at "
             f"its own ur_state row {i_deploy} nor the one at {i_deploy - 1}")
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


# ---------------------------------------------------------------------------
# 3. F16 (validation 2026-08-30): --parity-fixes, per-episode seeding,
#    --seed-from-meta, actions_pre_veto, --ckpt under --tiny.
#    Every one of these goes through the tool's OWN main().
# ---------------------------------------------------------------------------

def _copy_episode(ep_path: Path, dst: Path, name: str | None = None) -> Path:
    """A copy of the recorded episode under a NEW episode directory name.

    The name matters: the per-episode seed is derived from it (that is what
    makes it independent of the --episodes list), so two copies under the same
    basename are, correctly, the same episode as far as seeding goes."""
    import shutil
    out = dst / (name or ep_path.name)
    shutil.copytree(ep_path, out)
    return out


def _set_tags(ep: Path, tags: list[str]) -> None:
    meta = json.loads((ep / "meta.json").read_text())
    meta["tags"] = tags
    (ep / "meta.json").write_text(json.dumps(meta))


def _run_cli(monkeypatch, hw_yaml, episodes, out: Path, *extra,
             seeds: str | None = "2") -> dict:
    """replay_rig.main() exactly as an operator runs it.

    `seeds=None` omits --seeds entirely, which is what lets --seed-from-meta
    take K from the episode's own `kseeds:<K>` tag."""
    import replay_rig
    argv = ["replay_rig.py", "--tiny", "--hardware", str(hw_yaml), "--episodes"]
    argv += [str(e) for e in episodes]
    if seeds is not None:
        argv += ["--seeds", seeds]
    argv += ["--nfe", "1", "--out", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert replay_rig.main() == 0
    return json.loads(out.read_text())


def _head_dz(payload: dict, k: int = 0) -> list[float]:
    return [r["head_dz"] for r in payload["episodes"][k]["rows"]]


# --- the parity construction -----------------------------------------------

def test_parity_prev_chunk_is_built_by_the_deploy_snapshot_builder(
        mock_deploy_episode, tmp_path, monkeypatch):
    """F16: the parity prev_chunk must come from
    `SnapshotBuilder.prev_chunk_from_history` — deploy's own method — not from
    a second implementation in the replay that can drift away from it."""
    from phantom.deploy import planner as P
    from replay_rig import RigEpisode
    hw, ep_path, _, _ = mock_deploy_episode
    calls = []
    orig = P.SnapshotBuilder.prev_chunk_from_history

    def spy(self, t_now, ts_a, arm, grip_now):
        calls.append((t_now, float(grip_now)))
        return orig(self, t_now, ts_a, arm, grip_now)

    monkeypatch.setattr(P.SnapshotBuilder, "prev_chunk_from_history", spy)
    ep = RigEpisode(ep_path, hw)
    t = ep.t_master(len(ep.trace) - 1)
    snap, _ = ep.snapshot(t, None, teacher=True, parity=True)
    assert calls, "the replay did not call deploy's prev_chunk_from_history"
    assert calls[-1][0] == t
    # ... and it lands on the SNAPSHOT, which is what the policy prefers
    assert snap.prev_chunk is not None
    assert snap.prev_chunk.shape == (hw.control.chunk_horizon, hw.control.action_dim)
    assert np.isfinite(snap.prev_chunk).all()
    # the gripper channel is the EXECUTED command (STREAM_ACTIONS), not the
    # measured aperture: the mock policy commanded 0.3 from the first replan,
    # and grid steps before the first executed step fall back to the measured
    # aperture (0.0 here) exactly as SnapshotBuilder does on the rig
    g = np.asarray(snap.prev_chunk[:, 6], dtype=np.float64)
    assert np.all(np.isclose(g, 0.3) | np.isclose(g, 0.0)), g
    assert np.isclose(g[-1], 0.3), "the newest grid step is an executed command"
    # without parity the snapshot carries none of it (legacy conditioning)
    plain, _ = ep.snapshot(t, None, teacher=True)
    assert plain.prev_chunk is None


def test_parity_snapshot_uses_measured_dt_and_consecutive_frame_reactive(
        mock_deploy_episode):
    """The other two SnapshotBuilder parity switches, at replan 0 where the
    legacy path has no previous replan to difference at all."""
    from replay_rig import RigEpisode
    hw, ep_path, _, _ = mock_deploy_episode
    ep = RigEpisode(ep_path, hw)
    t = ep.t_master(len(ep.trace) - 1)
    legacy, _ = ep.snapshot(t, None, teacher=True)
    parity, _ = ep.snapshot(t, None, teacher=True, parity=True)
    # legacy reactive needs a PREVIOUS REPLAN; parity differences the two
    # consecutive fields_ds frames at t, so it is defined at every replan
    assert legacy.reactive == 0.0
    assert np.isfinite(parity.reactive)
    # contact_state: slip is flow / dt, and dt differs (nominal vs measured)
    assert legacy.contact_state.shape == parity.contact_state.shape


@pytest.mark.requires_cosmos_repo
def test_parity_fixes_reaches_the_batch_and_is_recorded(mock_deploy_episode,
                                                        tmp_path, monkeypatch):
    """End to end through main(): the flag is recorded and the default
    prev_chunk source flips to `measured`; and the batch the policy is handed
    really does carry the measured/executed past instead of the proposal.

    (The sampled NUMBERS cannot be compared across two main() calls under
    --tiny: the frozen tiny backbone is random per build and the checkpoint
    format stores only the trainable state, so the comparison below is made on
    the conditioning tensor, which is deterministic.)"""
    import replay_rig
    from phantom.inference.policy import Plan
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    base = _run_cli(monkeypatch, hw_yaml, [ep_path], tmp_path / "off.json")
    par = _run_cli(monkeypatch, hw_yaml, [ep_path], tmp_path / "on.json",
                   "--parity-fixes")
    assert base["parity_fixes"] is False and base["prev_chunk"] == "proposal"
    assert par["parity_fixes"] is True and par["prev_chunk"] == "measured"
    assert par["episodes"][0]["parity_fixes"] is True
    assert par["episodes"][0]["n_replans"] == base["episodes"][0]["n_replans"]
    assert all(np.isfinite(v) for v in _head_dz(par))
    # an explicit --prev-chunk still wins (E3 sweeps)
    z = _run_cli(monkeypatch, hw_yaml, [ep_path], tmp_path / "z.json",
                 "--parity-fixes", "--prev-chunk", "zeros")
    assert z["prev_chunk"] == "zeros"

    # what actually reaches the model
    ep = replay_rig.RigEpisode(ep_path, hw)
    t = ep.t_master(len(ep.trace) - 1)
    args = replay_rig.argparse.Namespace(tiny=True, ckpt=None, nfe=1, guidance=1.0,
                                         seeds=1, persistent_noise=False,
                                         parity_fixes=True)
    policy = replay_rig.build_policy(args, hw)
    assert policy.parity_fixes is True     # ... so replan() aligns prev_cpk_step
    snap_par, _ = ep.snapshot(t, None, teacher=True, parity=True)
    snap_leg, _ = ep.snapshot(t, None, teacher=True)
    proposal = Plan(t_created=t, t0_pose=np.zeros(6),
                    actions=np.asarray(ep.trace[0]["actions"], np.float32),
                    action_times=np.zeros(1), sigma=np.zeros(1), gate=0.0,
                    p_evt=np.zeros(1), cpk=None)
    b_par = policy._batch_from_obs(snap_par, proposal)["prev_chunk"]
    b_leg = policy._batch_from_obs(snap_leg, proposal)["prev_chunk"]
    assert not torch.allclose(b_par, b_leg), \
        "--parity-fixes did not change the intent channel the model is given"
    want = policy.norm.normalize("action", snap_par.prev_chunk.astype(np.float32))
    assert torch.allclose(b_par[0], torch.from_numpy(want), atol=1e-6)


@pytest.mark.requires_cosmos_repo
def test_parity_tag_mismatch_is_warned(mock_deploy_episode, tmp_path,
                                       monkeypatch, caplog):
    """`parity:on` episodes replayed without the flag silently condition on
    the legacy intent channel — the tool must say so."""
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    ep = _copy_episode(ep_path, tmp_path / "tagged")
    _set_tags(ep, ["parity:on", "nfe5"])
    with caplog.at_level("WARNING"):
        _run_cli(monkeypatch, hw_yaml, [ep], tmp_path / "warn.json")
    assert any("parity" in r.message.lower() and "recorded" in r.message.lower()
               for r in caplog.records), caplog.text


# --- seeding ----------------------------------------------------------------

@pytest.mark.requires_cosmos_repo
def test_seeding_is_per_episode_not_per_run(mock_deploy_episode, tmp_path,
                                            monkeypatch):
    """The generator used to be seeded ONCE before the episode loop, so every
    episode after the first started wherever the previous one's replans left it
    and the numbers moved with the --episodes list and its order (GATE G0 and
    every checkpoint comparison are read off these)."""
    import replay_rig
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    # the SAME episode three times in one run: with a per-run seed the second
    # and third differ from the first; with per-episode seeding all three are
    # the same replay of the same episode
    got = _run_cli(monkeypatch, hw_yaml, [ep_path, ep_path, ep_path],
                   tmp_path / "three.json")
    assert len(got["episodes"]) == 3
    assert _head_dz(got, 0) == _head_dz(got, 1) == _head_dz(got, 2)
    seeds = {e["seed"] for e in got["episodes"]}
    assert len(seeds) == 1 and got["episodes"][0]["seed_source"] == "name"
    # a DIFFERENT episode between them must not shift it either
    other = _copy_episode(ep_path, tmp_path / "other", "ep_other_1788000000_001")
    mixed = _run_cli(monkeypatch, hw_yaml, [ep_path, other, ep_path],
                     tmp_path / "mixed.json")
    assert _head_dz(mixed, 0) == _head_dz(mixed, 2)
    assert mixed["episodes"][1]["seed"] != mixed["episodes"][0]["seed"]
    # the seed is the BASE plus a stable offset of the episode NAME — never the
    # index, so dropping an invalid episode cannot renumber the rest
    assert got["episodes"][0]["seed"] == replay_rig.episode_seed(1000, ep_path.name)
    assert replay_rig.episode_seed(1000, "ep_a") != replay_rig.episode_seed(1000, "ep_b")
    assert (replay_rig.episode_seed(7, "ep_a")
            - replay_rig.episode_seed(0, "ep_a")) == 7
    assert got["seed_base"] == 1000 and got["seed_from_meta"] is False


@pytest.mark.requires_cosmos_repo
def test_seed_from_meta_reproduces_the_recorded_draw(mock_deploy_episode,
                                                     tmp_path, monkeypatch):
    """`run_deploy` records `seed:<n>` per episode since ba61354 — the
    strongest available validity check, and no tool read it.

    Three copies of ONE recorded episode (identical streams and trace) in a
    single run: two tagged `seed:4242`, one `seed:99`. Under --seed-from-meta
    the two that share a tag must sample identically even though their
    directory names differ, and the third must not."""
    import replay_rig
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    a1 = _copy_episode(ep_path, tmp_path / "a1", "ep_seeded_1788000000_001")
    a2 = _copy_episode(ep_path, tmp_path / "a2", "ep_seeded_1788000000_002")
    b = _copy_episode(ep_path, tmp_path / "b", "ep_seeded_1788000000_003")
    _set_tags(a1, ["nfe5", "seed:4242", "parity:off"])
    _set_tags(a2, ["nfe5", "seed:4242", "parity:off"])
    _set_tags(b, ["nfe5", "seed:99", "parity:off"])

    got = _run_cli(monkeypatch, hw_yaml, [a1, a2, b], tmp_path / "m.json",
                   "--seed-from-meta")
    assert got["seed_from_meta"] is True
    assert [e["seed"] for e in got["episodes"]] == [4242, 4242, 99]
    assert {e["seed_source"] for e in got["episodes"]} == {"meta"}
    assert _head_dz(got, 0) == _head_dz(got, 1), "the recorded draw is not reproducible"
    assert _head_dz(got, 2) != _head_dz(got, 0), "the recorded seed was ignored"
    # without the flag the same two episodes draw from their NAMES instead
    plain = _run_cli(monkeypatch, hw_yaml, [a1, a2], tmp_path / "p.json")
    assert plain["episodes"][0]["seed"] != plain["episodes"][1]["seed"]
    assert _head_dz(plain, 0) != _head_dz(plain, 1)
    assert replay_rig.meta_seed(replay_rig.RigEpisode(a1, hw).meta) == 4242
    assert replay_rig.meta_parity(replay_rig.RigEpisode(a1, hw).meta) is False


@pytest.mark.requires_cosmos_repo
def test_seed_from_meta_refuses_a_pre_fix_episode(mock_deploy_episode, tmp_path,
                                                  monkeypatch):
    """`seed:none` / no tag = no recorded draw; replaying it under some other
    noise and calling it a reproduction is the failure mode to prevent."""
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    ep = _copy_episode(ep_path, tmp_path / "old")
    _set_tags(ep, ["nfe5", "seed:none"])
    with pytest.raises(SystemExit) as e:
        _run_cli(monkeypatch, hw_yaml, [ep], tmp_path / "x.json", "--seed-from-meta")
    assert "seed:" in str(e.value) and "deploy-rng" in str(e.value)


# --- the K-seed lever: kseeds:<K> + diag.k_pick ------------------------------

def _spy_on_sample(monkeypatch):
    """Record (k_seeds, batch B) of every rf.sample call."""
    from phantom.model.rf import PhantomRectifiedFlow
    calls: list[tuple[int, int]] = []
    real = PhantomRectifiedFlow.sample

    def spy(self, batch, **kw):
        b = next(int(v.shape[0]) for v in batch.values() if torch.is_tensor(v))
        calls.append((int(kw.get("k_seeds", 1)), b))
        return real(self, batch, **kw)

    monkeypatch.setattr(PhantomRectifiedFlow, "sample", spy)
    return calls


def _tag_k(ep: Path, k: int, seed: int = 4242) -> None:
    _set_tags(ep, ["nfe5", f"seed:{seed}", "parity:off", f"kseeds:{k}"])


@pytest.mark.requires_cosmos_repo
def test_seed_from_meta_takes_k_from_the_kseeds_tag_and_expands_like_deploy(
        mock_deploy_episode, tmp_path, monkeypatch):
    """`run_deploy` tags `kseeds:<K>` and deploy hands rf.sample a B=1 batch,
    expanding to K INSIDE it. The replay pre-tiled to B=K instead, so build_x0
    drew at B=K and the noise stream no longer matched the rig's — `--seeds 4`
    reproduced nothing and `--seeds 1` (what the help prescribed) reproduced
    candidate 0, which is not the chunk the selector executed."""
    import replay_rig
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    ep = _copy_episode(ep_path, tmp_path / "k3", "ep_kseeds_1788000000_001")
    _tag_k(ep, 3)
    assert replay_rig.meta_kseeds(replay_rig.RigEpisode(ep, hw).meta) == 3

    calls = _spy_on_sample(monkeypatch)
    got = _run_cli(monkeypatch, hw_yaml, [ep], tmp_path / "k.json",
                   "--seed-from-meta", seeds=None)
    e = got["episodes"][0]
    assert (e["recorded_kseeds"], e["seeds"], e["k_seeds_expansion"]) == (3, 3, True)
    assert calls and all(c == (3, 1) for c in calls), calls
    assert all(len(r["seed_head_dz"]) == 3 for r in e["rows"])

    # ... and the legacy path still pre-tiles when there is nothing to reproduce
    calls.clear()
    _run_cli(monkeypatch, hw_yaml, [ep_path], tmp_path / "plain.json")
    assert calls and all(c == (1, 2) for c in calls), calls


@pytest.mark.requires_cosmos_repo
def test_seed_from_meta_scores_the_trace_against_the_executed_k_pick(
        mock_deploy_episode, tmp_path, monkeypatch):
    """`diag.k_pick` names WHICH of the K the selector executed (k_pick != 0 in
    310/400 recorded replans), so that is the row the trace must be scored
    against. The K-spread stays — it is E2's signal, not the reproduction."""
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    ep = _copy_episode(ep_path, tmp_path / "kp", "ep_kpick_1788000000_001")
    _tag_k(ep, 3)
    trace = json.loads((ep / "planner_trace.json").read_text())
    for r in trace:
        r.setdefault("diag", {})["k_pick"] = 2
    (ep / "planner_trace.json").write_text(json.dumps(trace))

    got = _run_cli(monkeypatch, hw_yaml, [ep], tmp_path / "kp.json",
                   "--seed-from-meta", seeds=None)
    rows = got["episodes"][0]["rows"]
    assert rows
    for r in rows:
        assert r["k_pick"] == 2
        assert r["pick_head_dz"] == pytest.approx(r["seed_head_dz"][2])
        assert r["head_dz_pick_err"] == pytest.approx(
            r["trace_head_dz"] - r["seed_head_dz"][2])
        # the reproduction check proper, and the K-spread still reported
        assert r["pick_abs_err"] >= r["best_abs_err"] >= 0.0
        assert 0 <= r["best_seed"] < 3
        assert "head_dz_std" in r and len(r["seed_head_dz"]) == 3


@pytest.mark.requires_cosmos_repo
def test_seeds_that_disagree_with_the_recorded_kseeds_is_a_hard_failure(
        mock_deploy_episode, tmp_path, monkeypatch):
    """F16's rule: a flag that contradicts the recording is refused, not warned."""
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    ep = _copy_episode(ep_path, tmp_path / "bad", "ep_kmismatch_1788000000_001")
    _tag_k(ep, 3)
    with pytest.raises(SystemExit) as e:
        _run_cli(monkeypatch, hw_yaml, [ep], tmp_path / "bad.json",
                 "--seed-from-meta", seeds="2")
    assert "kseeds:3" in str(e.value) and "--seeds 2" in str(e.value)


# --- the veto's arithmetic is not a model sample ----------------------------

@pytest.mark.requires_cosmos_repo
def test_actions_pre_veto_is_what_the_trace_columns_measure(mock_deploy_episode,
                                                            tmp_path, monkeypatch):
    """On a vetoed replan `trace["actions"]` is the veto's rewrite (scripted
    aperture, zeroed z), so trace_in_spread — GATE G0 — scores against
    arithmetic. Prefer `actions_pre_veto` when the trace carries it."""
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    ep = _copy_episode(ep_path, tmp_path / "veto")
    trace = json.loads((ep / "planner_trace.json").read_text())
    acc = [i for i, r in enumerate(trace) if r.get("accepted", True)]
    pre = np.asarray(trace[acc[0]]["actions"], dtype=np.float64).copy()
    pre[:, 2] = -0.002                      # the model's own descent
    trace[acc[0]]["actions_pre_veto"] = pre.tolist()
    # the REAL shape planner._apply_veto writes: a dict on every replan while the
    # veto is on, `action` naming what it actually did.
    trace[acc[0]]["terminal_veto"] = {"action": "recovery_open"}
    trace[acc[1]]["terminal_veto"] = {"action": "close_masked"}  # rewritten, NOT recorded
    # ... and the two no-op records that must NOT count as vetoed: nothing in
    # plan.actions was touched, so these rows are perfectly comparable (G0).
    trace[acc[2]]["terminal_veto"] = {"action": "none"}
    trace[acc[3]]["terminal_veto"] = {"action": "close_allowed"}
    (ep / "planner_trace.json").write_text(json.dumps(trace))

    got = _run_cli(monkeypatch, hw_yaml, [ep], tmp_path / "v.json")
    rows = {r["replan"]: r for r in got["episodes"][0]["rows"]}
    r0 = rows[acc[0]]
    assert r0["trace_source"] == "actions_pre_veto" and r0["trace_vetoed"] is True
    assert r0["trace_comparable"] is True
    # -2 mm/step over steps 0..8 = -18 mm, i.e. the PRE-veto chunk
    assert r0["trace_head_dz"] == pytest.approx(-18.0, abs=1e-6)
    r1 = rows[acc[1]]
    assert r1["trace_source"] == "actions" and r1["trace_comparable"] is False
    for k in (acc[2], acc[3]):
        assert rows[k]["trace_vetoed"] is False, "a no-op veto record is not a veto"
        assert rows[k]["trace_comparable"] is True
    assert got["episodes"][0]["n_uncomparable_vetoed"] == 1
    # conditioning still uses the POST-veto chunk deploy carried forward
    assert got["prev_chunk"] == "proposal"


# --- --tiny --ckpt ----------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory, mock_deploy_episode):
    """A checkpoint of the tiny backbone, with NON-identity norm stats."""
    if not _cosmos_available():
        pytest.skip("cosmos repo not importable")
    from phantom.config.training import CommonTrainConfig
    from phantom.data.schema import NormStats
    from phantom.train import common as C
    from phantom.train.builder import build_model
    hw = mock_deploy_episode[0]
    pm = build_model(hw, load_paths(), student=False, tiny=True, load_base=False)
    ns = NormStats(mean={"action": np.zeros(hw.control.action_dim, np.float32),
                         "ur_state": np.zeros(hw.ur_state_dim, np.float32)},
                   std={"action": np.full(hw.control.action_dim, 3.0, np.float32),
                        "ur_state": np.ones(hw.ur_state_dim, np.float32)})
    out = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    C.save_phantom_checkpoint(out, pm.rf, hw=hw, bb=pm.bb, mc=pm.mc,
                              train_cfg=CommonTrainConfig(), step=0, norm_stats=ns)
    return out


@pytest.mark.skipif(not _cosmos_available(), reason="cosmos repo not importable")
def test_tiny_honours_the_checkpoint(mock_deploy_episode, tiny_ckpt, tmp_path,
                                     monkeypatch):
    """--tiny --ckpt used to DROP the checkpoint silently: a random backbone
    with IDENTITY norm stats, reported as if that checkpoint had been replayed.

    (A tiny checkpoint carries the trainable state — LoRA + phantom modules —
    and the norm stats; the frozen base is random under load_base=False, which
    is why this asserts the loaded contract rather than run-to-run equality.)"""
    import replay_rig
    from phantom.train.common import trainable_state_dicts
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    got = _run_cli(monkeypatch, hw_yaml, [ep_path], tmp_path / "c1.json",
                   "--ckpt", str(tiny_ckpt))
    assert got["ckpt"] == str(tiny_ckpt)
    assert all(np.isfinite(v) for v in _head_dz(got))

    args = replay_rig.argparse.Namespace(
        tiny=True, ckpt=str(tiny_ckpt), nfe=1, guidance=1.0, seeds=1,
        persistent_noise=False, parity_fixes=False)
    pol = replay_rig.build_policy(args, hw)
    # the checkpoint's own norm stats, not NormStats.identity()
    assert float(pol.norm.std["action"][0]) == pytest.approx(3.0)
    # ... and its trainable weights really are in the live model
    payload = torch.load(str(tiny_ckpt), map_location="cpu", weights_only=False)
    lora, phantom = trainable_state_dicts(pol.rf)
    saved = {**payload["lora"], **payload["phantom_modules"]}
    live = {**lora, **phantom}
    assert saved and set(saved) == set(live)
    assert all(torch.allclose(live[k].float().cpu(), v.float().cpu(), atol=1e-5)
               for k, v in saved.items())
    # without --ckpt the tiny path is the random smoke backbone, and its norm
    # stats are the identity (a pass-through, no "action" entry at all)
    plain = replay_rig.build_policy(
        replay_rig.argparse.Namespace(tiny=True, ckpt=None, nfe=1, guidance=1.0,
                                      seeds=1, persistent_noise=False,
                                      parity_fixes=False), hw)
    assert not plain.norm.std and not plain.norm.mean
    one = np.ones((1, hw.control.action_dim), dtype=np.float32)
    assert float(np.asarray(plain.norm.normalize("action", one))[0, 0]) == 1.0


@pytest.mark.skipif(not _cosmos_available(), reason="cosmos repo not importable")
def test_tiny_with_a_full_size_checkpoint_fails_loudly(mock_deploy_episode,
                                                       tmp_path, monkeypatch):
    """Honouring --ckpt must not mean silently half-loading it."""
    import replay_rig
    hw, _, _, _ = mock_deploy_episode
    bad = tmp_path / "bad.pt"
    torch.save({"configs": {"model": {}}, "model": {"nope": torch.zeros(3)},
                "norm_stats": {"mean": {}, "std": {}}}, bad)
    args = replay_rig.argparse.Namespace(
        tiny=True, ckpt=str(bad), nfe=1, guidance=1.0, seeds=1,
        persistent_noise=False, parity_fixes=False)
    with pytest.raises(SystemExit) as e:
        replay_rig.build_policy(args, hw)
    assert "tiny" in str(e.value)


# ---------------------------------------------------------------------------
# 4. tools/replay_deploy_path.py — the E0 discriminator had NO test and could
#    not run off a GPU box (hardcoded device="cuda", no --tiny)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _cosmos_available(), reason="cosmos repo not importable")
def test_replay_deploy_path_runs_on_cpu_and_reads_the_recorded_seed(
        mock_deploy_episode, tmp_path, monkeypatch):
    import importlib
    rdp = importlib.import_module("replay_deploy_path")
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    ep = _copy_episode(ep_path, tmp_path / "dp")
    _set_tags(ep, ["seed:4242"])
    out = tmp_path / "dp.json"
    monkeypatch.setattr(sys, "argv", [
        "replay_deploy_path.py", "--ckpt", "", "--hardware", str(hw_yaml),
        "--episodes", str(ep), "--nfe", "1", "--tiny", "--device", "cpu",
        "--seed-from-meta", "--out", str(out)])
    assert rdp.main() == 0
    rows = json.loads(out.read_text())[0]["rows"]
    assert rows and all(np.isfinite(r["head_dz"]) for r in rows)
    assert all(r["trace_source"] == "actions" for r in rows)
    # --deploy-rng and --seed-from-meta are the two ERAS of run_deploy seeding
    monkeypatch.setattr(sys, "argv", [
        "replay_deploy_path.py", "--ckpt", "", "--hardware", str(hw_yaml),
        "--episodes", str(ep), "--tiny", "--device", "cpu",
        "--seed-from-meta", "--deploy-rng"])
    with pytest.raises(SystemExit):
        rdp.main()


# ---------------------------------------------------------------------------
# 5. --dump-agreement (2026-09-14): the imagined-future disagreement on the
#    rig's OWN recorded states, per accepted replan. Two things have to hold:
#    the flag's absence must leave the tool byte-identical, and the JSONL must
#    carry the agreed schema with the episode's labelled outcome attached.
# ---------------------------------------------------------------------------

DUMP_KEYS = ["episode", "day", "task", "seed", "ckpt", "label", "outcome",
             "stop", "i", "t", "k_seeds", "agreement", "pick",
             "agreement_of_pick", "spread", "trace_pick", "source"]


def test_agreement_record_schema_and_pick():
    """The record builder alone (no weights, no GPU): exact key set and order,
    and `pick`/`agreement_of_pick`/`spread` read off the K distances."""
    import replay_rig
    rec = replay_rig.agreement_record(
        episode="ep_0001", day="20260913", task="whiteboard", seed=7,
        ckpt="teacher_002000.pt", label="v6_simft2k", outcome=1,
        stop="safety_stop", i=3, t=12.5, scores=[0.4, 0.1, 0.9, 0.2],
        trace_pick=2)
    assert list(rec) == DUMP_KEYS
    assert rec["k_seeds"] == 4 and rec["pick"] == 1
    assert rec["agreement_of_pick"] == pytest.approx(0.1)
    assert rec["spread"] == pytest.approx(0.8)
    assert rec["trace_pick"] == 2 and rec["source"] == "replay"
    assert json.loads(json.dumps(rec)) == rec          # JSONL-serializable
    # the nullable columns stay null rather than becoming 0/""
    rec = replay_rig.agreement_record(
        episode="ep_0002", day="20260912", task=None, seed=None, ckpt="tiny",
        label=None, outcome=None, stop=None, i=0, t=0.0, scores=[1.0, 1.0],
        trace_pick=None)
    assert rec["seed"] is None and rec["outcome"] is None
    assert rec["label"] is None and rec["stop"] is None
    assert rec["trace_pick"] is None
    assert rec["spread"] == 0.0 and rec["pick"] == 0   # ties -> lowest index


def test_agreement_context_reads_the_episode_label(mock_deploy_episode, tmp_path):
    """The episode-level half comes from phantom.eval.stats, so the arm tags
    that disambiguate two same-basename checkpoints survive into the dump."""
    import replay_rig
    hw, ep_path, _, _ = mock_deploy_episode
    ep_dir = _copy_episode(ep_path, tmp_path / "20260913", name="ep_ctx")
    _set_tags(ep_dir, ["seed:4242", "ckpt:student_002000.pt",
                       "label:stu_ftA_r2", "stop:safety_stop"])
    replay_rig._EP_STATS.clear()
    ctx = replay_rig.agreement_context(
        replay_rig.RigEpisode(ep_dir, hw), hw, "runs/x/student_002000.pt")
    replay_rig._EP_STATS.clear()
    assert ctx["episode"] == "ep_ctx" and ctx["day"] == "20260913"
    # BARE tag values, the spelling agreement_outcome.py's --jsonl merges on
    assert ctx["seed"] == 4242 and ctx["label"] == "stu_ftA_r2"
    assert ctx["ckpt"] == "student_002000.pt"          # the `ckpt:` tag value
    assert ctx["stop"] == "safety_stop"
    assert set(ctx) == {"episode", "day", "task", "seed", "ckpt", "label",
                        "outcome", "stop"}


@pytest.mark.skipif(not _cosmos_available(), reason="cosmos repo not importable")
def test_dump_agreement_writes_one_line_per_replan_and_changes_nothing(
        mock_deploy_episode, tmp_path, monkeypatch):
    """End-to-end through main(): the dump has one line per accepted replan,
    and the run WITHOUT the flag is identical to the run with it."""
    import replay_rig
    hw, ep_path, _, hw_yaml = mock_deploy_episode
    dump = tmp_path / "agree" / "dump.jsonl"
    # --tiny builds a RANDOM backbone (load_base=False), so the two runs are
    # only comparable from the same global torch seed
    torch.manual_seed(0)
    with_dump = _run_cli(monkeypatch, hw_yaml, [ep_path], tmp_path / "a.json",
                         "--dump-agreement", str(dump))
    torch.manual_seed(0)
    without = _run_cli(monkeypatch, hw_yaml, [ep_path], tmp_path / "b.json")
    # byte-identical apart from the recorded --out path itself
    assert with_dump["episodes"] == without["episodes"]

    trace = json.loads((ep_path / "planner_trace.json").read_text())
    n_accepted = sum(1 for r in trace if r["accepted"])
    lines = [json.loads(l) for l in dump.read_text().splitlines() if l.strip()]
    assert len(lines) == n_accepted
    assert [r["i"] for r in lines] == [i for i, r in enumerate(trace)
                                       if r["accepted"]]
    for r in lines:
        assert list(r) == DUMP_KEYS
        assert r["episode"] == ep_path.name and r["ckpt"] == "tiny"
        assert r["k_seeds"] == 2 and len(r["agreement"]) == 2
        assert all(np.isfinite(v) and v >= 0.0 for v in r["agreement"])
        assert r["pick"] == int(np.argmin(r["agreement"]))
        assert r["agreement_of_pick"] == pytest.approx(min(r["agreement"]))
        assert r["spread"] == pytest.approx(max(r["agreement"])
                                            - min(r["agreement"]))
        assert r["source"] == "replay"
    # a second run TRUNCATES rather than appending to the same path
    _run_cli(monkeypatch, hw_yaml, [ep_path], tmp_path / "c.json",
             "--dump-agreement", str(dump))
    assert len([l for l in dump.read_text().splitlines() if l.strip()]) == n_accepted


def test_dump_records_merge_into_agreement_outcome(tmp_path):
    """The dump is only useful if tools/rig/analysis/agreement_outcome.py can
    read it back through --jsonl: same RECORD_KEYS in the same order, bare
    `label`/`ckpt` tag values, and `source` "replay"."""
    import importlib
    import replay_rig
    ao = importlib.import_module("rig.analysis.agreement_outcome")
    assert list(ao.RECORD_KEYS) == DUMP_KEYS
    rec = replay_rig.agreement_record(
        episode="ep_0001", day="20260913", task="Carton", seed=104,
        ckpt="teacher_002000.pt", label="v6_simft2k", outcome=0,
        stop="safety_stop", i=7, t=1.5, scores=[0.4, 0.1, 0.9, 0.2],
        trace_pick=2)
    p = tmp_path / "recs.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    got, diag = ao.read_jsonl(p)
    assert diag["n"] == 1 and diag["n_skipped"] == 0
    # nothing is dropped or re-derived on the way in
    assert got[0] == rec
