"""P9 + P10 of the 2026-08-28 review (docs/review_20260828/REVIEW_SYNTHESIS.md).

P9  — unlabeled / contaminated rollouts must never reach the optimizer at
      action weight 1: the deploy no-verdict guard, is_trainable_episode(),
      and the two places that build training indexes (manifest_split and the
      rollout intake).
P10A— the `no_distill` / `vision_only` control arms must be BUILDABLE:
      train_teacher --student and PhantomModelConfig.mask_wrist (the wrist F/T
      window is the input that made every "tactile-free" arm input-identical
      to the student).
P10B— programs that load a checkpoint must build the model from the
      CHECKPOINT'S model config, and load_phantom_checkpoint must refuse the
      mismatch rather than silently re-phasing every action frame.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom.config.backbone import BackboneConfig
from phantom.config.model import PhantomModelConfig
from phantom.config.paths import load_paths
from phantom.data.schema import EpisodeMeta, is_failure_demo, is_trainable_episode
from phantom.deploy.planner import (SnapshotBuilder, TACTILE_INPUT_MODES,
                                    WRIST_MASKED_MODES)
from phantom.deploy.runtime import DeploymentRuntime
from phantom.recording.recorder import EpisodeRecorder
from phantom.scripts import run_deploy as RD
from phantom.train import common as C
from phantom_test_utils import make_hw, make_small_hw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import intake_recovery as IR  # noqa: E402


# ===========================================================================
# P9a — is_trainable_episode
# ===========================================================================

def _fin(**kw) -> EpisodeMeta:
    kw.setdefault("status", "finalized")
    return EpisodeMeta(**kw)


def test_ordinary_and_failure_demos_stay_trainable():
    assert is_trainable_episode(_fin(task="egg", success=True))
    assert is_trainable_episode(_fin(task="egg", policy="teleop", success=True))
    # a teleop demo the operator never judged is still ordinary training data:
    # only POLICY rollouts are refused for a missing verdict
    assert is_trainable_episode(_fin(task="egg", policy="teleop", success=None))
    # deliberate failure demos train (at action_weight 0, via is_failure_demo)
    fail = _fin(task="egg_fail", success=False, tags=["deliberate_failure"])
    assert is_failure_demo(fail) and is_trainable_episode(fail)


def test_contaminated_and_unlabeled_are_never_trainable():
    # the exact hole P9 names: is_failure_demo() is False for these, so
    # WindowSampler would hand them action_weight 1.0
    contaminated = _fin(task="egg", success=None, tags=["contaminated"])
    assert not is_failure_demo(contaminated)
    assert not is_trainable_episode(contaminated)
    unlabeled = EpisodeMeta(task="egg", status="aborted", tags=["unlabeled"])
    assert not is_trainable_episode(unlabeled)
    # a contaminated episode that DID get a success verdict is still excluded
    assert not is_trainable_episode(
        _fin(task="egg", success=True, tags=["batch_20260822", "contaminated"]))


def test_unjudged_policy_rollout_is_refused():
    for system in ("teacher", "student", "vision_only", "no_distill"):
        assert not is_trainable_episode(_fin(task="egg", policy=system, success=None))
    # ... but a judged one trains
    assert is_trainable_episode(_fin(task="egg", policy="student", success=True))
    assert is_trainable_episode(_fin(task="egg", policy="student", success=False))


def test_non_finalized_is_never_trainable():
    assert not is_trainable_episode(EpisodeMeta(task="egg", success=True,
                                                status="recording"))
    assert not is_trainable_episode(EpisodeMeta(task="egg", success=True,
                                                status="aborted"))


# ===========================================================================
# P9a — the deploy no-verdict guard end to end
# ===========================================================================

class _CrawlPolicy:
    """Constant crawl (same shape as the mock dry-run policy)."""

    def __init__(self, hw):
        self.hw = hw

    def reset_episode(self):
        pass

    def replan(self, snap, prev_plan, tcp_pose):
        import time
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
def mock_rollout(tmp_path_factory):
    """ONE mock deploy episode, run through the real DeploymentRuntime."""
    out = tmp_path_factory.mktemp("deploy")
    hw = make_small_hw()
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="student",
                           out_root=out) as rt:
        res = rt.run_episode(task="whiteboard", max_replans=2,
                             policy_name="student")
    assert res.episode_path is not None
    return Path(res.episode_path)


@pytest.fixture
def rollout(mock_rollout, tmp_path):
    """A private copy of that episode's meta.json + a recorder to relabel it
    with (relabel touches nothing else)."""
    import shutil
    ep = tmp_path / mock_rollout.name
    ep.mkdir()
    shutil.copy(mock_rollout / "meta.json", ep / "meta.json")
    rec = SimpleNamespace(out_root=tmp_path)
    rec.relabel = lambda *a, **kw: EpisodeRecorder.relabel(rec, *a, **kw)
    return rec, ep


def _meta(ep: Path) -> EpisodeMeta:
    return EpisodeMeta.load(ep / "meta.json")


def test_episode_with_no_verdict_is_filed_as_unlabeled(rollout):
    """THE P9 regression: run_episode used to leave status='finalized' +
    success=None, which every lister reads as an ordinary full-weight demo."""
    recorder, ep = rollout
    m = _meta(ep)
    assert m.status == "aborted" and "unlabeled" in m.tags
    assert m.success is None
    assert not is_trainable_episode(m)


def test_enter_skip_leaves_the_episode_unlabeled(rollout):
    recorder, ep = rollout
    assert RD.label_episode(recorder, ep, "") == ""
    assert RD.label_episode(recorder, ep, "   ") == ""
    m = _meta(ep)
    assert m.status == "aborted" and "unlabeled" in m.tags
    assert not is_trainable_episode(m)


@pytest.mark.parametrize("ans,success", [("s", True), ("s went fine", True),
                                         ("f", False), ("f missed the grasp", False)])
def test_a_verdict_promotes_the_episode_back(rollout, ans, success):
    recorder, ep = rollout
    assert RD.label_episode(recorder, ep, ans) == ans[0]
    m = _meta(ep)
    assert m.status == "finalized"
    assert m.success is success
    assert "unlabeled" not in m.tags
    assert is_trainable_episode(m)
    assert m.notes.startswith("operator:")


def test_contaminated_keeps_success_none_and_stays_out_of_training(rollout):
    recorder, ep = rollout
    assert RD.label_episode(recorder, ep, "c hand in frame") == "c"
    m = _meta(ep)
    assert m.success is None                 # NOT a deliberate failure demo
    assert not is_failure_demo(m)
    assert "contaminated" in m.tags and "unlabeled" not in m.tags
    assert not is_trainable_episode(m)


def test_unrecognized_verdict_does_not_promote(rollout):
    recorder, ep = rollout
    assert RD.label_episode(recorder, ep, "yes it worked") == ""
    assert not is_trainable_episode(_meta(ep))


def test_relabel_can_clear_a_tag(tmp_path):
    ep = tmp_path / "ep_0"
    ep.mkdir()
    EpisodeMeta(task="egg", status="aborted",
                tags=["unlabeled", "batch_x"]).save(ep / "meta.json")
    rec = SimpleNamespace(out_root=tmp_path)
    EpisodeRecorder.relabel(rec, ep, success=True, status="finalized",
                            remove_tags=["unlabeled"])
    m = EpisodeMeta.load(ep / "meta.json")
    assert m.tags == ["batch_x"] and m.status == "finalized" and m.success is True


# ===========================================================================
# P9b — the two training-index builders
# ===========================================================================

def _write_ep(tasks_root: Path, name: str, **meta_kw) -> Path:
    meta_kw.setdefault("task", "egg")
    meta_kw.setdefault("status", "finalized")
    ep = tasks_root / meta_kw["task"] / name
    ep.mkdir(parents=True)
    EpisodeMeta(**meta_kw).save(ep / "meta.json")
    return ep


def _write_manifest(root: Path, rows: list[dict]) -> None:
    mf = root / "manifests" / "all.jsonl"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(name: str, task: str = "egg", **kw) -> dict:
    return {"episode": name, "task": task, "split": "train",
            "path": f"tasks/{task}/{name}", **kw}


def test_manifest_split_skips_contaminated_and_unlabeled(tmp_path, caplog):
    tasks = tmp_path / "tasks"
    _write_ep(tasks, "ep_ok", success=True)
    _write_ep(tasks, "ep_dirty", success=None, tags=["contaminated"])
    _write_ep(tasks, "ep_unjudged", status="aborted", tags=["unlabeled"])
    _write_manifest(tmp_path, [_row("ep_ok"), _row("ep_dirty"), _row("ep_unjudged")])
    eps = C.manifest_split(tasks, "train")
    assert [p.name for p in eps] == ["ep_ok"]


def test_manifest_split_refuses_an_unjudged_policy_rollout(tmp_path):
    tasks = tmp_path / "tasks"
    _write_ep(tasks, "ep_ok", success=True)
    _write_ep(tasks, "ep_roll", policy="student", success=None)
    _write_manifest(tmp_path, [_row("ep_ok"), _row("ep_roll")])
    with pytest.raises(SystemExit, match="no success verdict"):
        C.manifest_split(tasks, "train")


def test_manifest_split_still_hard_errors_on_a_crashed_episode(tmp_path):
    """The pre-existing guard must survive: a 'recording' (crashed) episode is
    a broken manifest, not a quiet skip."""
    tasks = tmp_path / "tasks"
    _write_ep(tasks, "ep_crash", status="recording", success=True)
    _write_manifest(tmp_path, [_row("ep_crash")])
    with pytest.raises(SystemExit, match="non-finalized"):
        C.manifest_split(tasks, "train")


def test_manifest_split_admits_judged_rollouts_and_failure_demos(tmp_path):
    tasks = tmp_path / "tasks"
    _write_ep(tasks, "ep_roll", policy="student", success=True)
    _write_ep(tasks, "ep_fail", task="egg_fail", success=False,
              tags=["deliberate_failure"])
    _write_manifest(tmp_path, [_row("ep_roll"),
                               _row("ep_fail", task="egg_fail")])
    assert sorted(p.name for p in C.manifest_split(tasks, "train")) == \
        ["ep_fail", "ep_roll"]


def test_intake_manifest_refuses_untrainable_episodes(tmp_path, capsys):
    tasks = tmp_path / "tasks"
    _write_ep(tasks, "ep_ok", success=True)
    _write_ep(tasks, "ep_dirty", success=None, tags=["contaminated"])
    _write_ep(tasks, "ep_unjudged", status="aborted", tags=["unlabeled"])
    _write_ep(tasks, "ep_roll", policy="vision_only", success=None)
    mf = tmp_path / "manifests" / "all.jsonl"
    mf.parent.mkdir(parents=True, exist_ok=True)
    IR.manifest(tasks, mf, val_min_eps=0)
    rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    assert [r["episode"] for r in rows] == ["ep_ok"]
    out = capsys.readouterr().out
    assert "REFUSING ep_dirty" in out and "REFUSING ep_roll" in out
    assert "REFUSED 2 episodes" in out
    # the unlabeled one is status='aborted', caught by the older status skip
    assert "skipping ep_unjudged" in out


def test_window_index_refuses_an_unjudged_rollout(tmp_path):
    """The DAgger path bypasses the manifest entirely: dagger_driver hands a
    rollout root to distill_hid --extra-data -> WindowSampler.build_index."""
    from phantom.config.backbone import BackboneConfig as BB
    from phantom.data.schema import NormStats
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    hw = make_hw()
    root = tmp_path / "rollouts"
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, BB.tiny(), NormStats.identity())
    assert sampler.build_index(root, windows_per_episode=1)      # baseline

    ep = next(p for p in root.rglob("ep_*") if (p / "meta.json").exists())
    m = EpisodeMeta.load(ep / "meta.json")
    m.policy, m.success = "student", None            # an unjudged rollout
    m.save(ep / "meta.json")
    assert sampler.build_index(root, windows_per_episode=1) == []

    m.tags = ["contaminated"]
    m.policy, m.success = "teleop", True             # judged, but contaminated
    m.save(ep / "meta.json")
    assert sampler.build_index(root, windows_per_episode=1) == []


# ===========================================================================
# P10A — mask_wrist (no model build needed: the HHT proprio path is the input)
# ===========================================================================

def _hht(mask: bool):
    from phantom.model.hht.hht import HHT
    hw = make_hw()
    bb = BackboneConfig.tiny()
    mc = PhantomModelConfig(student=True, mask_wrist=mask)
    torch.manual_seed(0)
    # student layout: obs_frames never touches the VAE
    return hw, HHT(hw, bb, mc, vae=None, student=True).eval()


def _obs_batch(hw, seed: int):
    g = torch.Generator().manual_seed(seed)
    return {"wrist": torch.randn(2, hw.wrist_ft.window_len, hw.wrist_ft.dim,
                                 generator=g),
            "ur_state": torch.zeros(2, hw.ur_state_dim)}


def test_mask_wrist_makes_the_output_invariant_to_the_wrist_window():
    from phantom.model.sequence import FrameGroup
    hw, hht = _hht(mask=True)
    a, b = _obs_batch(hw, 1), _obs_batch(hw, 2)
    assert not torch.allclose(a["wrist"], b["wrist"])
    with torch.no_grad():
        pa = hht.obs_frames(a)[FrameGroup.OBS_PROPRIO]
        pb = hht.obs_frames(b)[FrameGroup.OBS_PROPRIO]
        # the ACC leading-signal branch reads the window too
        wa, wb = hht.wrist_feature(a), hht.wrist_feature(b)
    assert torch.equal(pa, pb)
    assert torch.equal(wa, wb)


def test_without_mask_wrist_the_window_still_matters():
    """Guards against 'invariant because the path is dead anyway'."""
    from phantom.model.sequence import FrameGroup
    hw, hht = _hht(mask=False)
    a, b = _obs_batch(hw, 1), _obs_batch(hw, 2)
    with torch.no_grad():
        pa = hht.obs_frames(a)[FrameGroup.OBS_PROPRIO]
        pb = hht.obs_frames(b)[FrameGroup.OBS_PROPRIO]
    assert not torch.allclose(pa, pb)


def test_mask_wrist_survives_a_checkpoint_config_roundtrip():
    mc = PhantomModelConfig(student=True, mask_wrist=True)
    assert PhantomModelConfig.from_dict(mc.to_dict()).mask_wrist is True
    # absent from an older checkpoint's config -> default False, not a crash
    d = mc.to_dict()
    d.pop("mask_wrist")
    assert PhantomModelConfig.from_dict(d).mask_wrist is False


# ---- deploy side: the snapshot for the sensor-free arms

class _Ring:
    def __init__(self, ts, fields):
        self.ts, self.fields = np.asarray(ts, dtype=np.float64), fields

    def latest(self, k=1):
        m = min(k, len(self.ts))
        sl = slice(len(self.ts) - m, len(self.ts))
        return self.ts[sl].copy(), {f: np.stack([np.asarray(v)] * m)
                                    for f, v in self.fields.items()}


def _snapshot(mode: str):
    import time
    hw = make_small_hw()
    t = time.perf_counter()
    cam = {"color": np.zeros(hw.cameras.scene.color.hwc, np.uint8)}
    arm = {"q": np.zeros(hw.arm.dof), "qd": np.zeros(hw.arm.dof),
           "tcp_pose": np.zeros(6), "tcp_speed": np.zeros(6),
           "ft": np.arange(6, dtype=np.float64) + 1.0}
    rings = {"camera_scene": _Ring([t], cam),
             "arm": _Ring(np.linspace(t - 0.05, t, 8), arm),
             "gripper": _Ring([t], {"state": np.array([0.4, 3.0])})}
    return SnapshotBuilder(hw, types.SimpleNamespace(rings=rings), mode).build()


def test_snapshot_masks_the_wrist_window_for_the_sensor_free_arms():
    assert WRIST_MASKED_MODES == ("vision_only", "drop_tactile")
    assert "vision_only" not in TACTILE_INPUT_MODES
    for mode in WRIST_MASKED_MODES:
        snap = _snapshot(mode)
        assert np.count_nonzero(snap.wrist_window) == 0, mode
    # every other mode keeps the real recorded window ('teacher' would need
    # the tactile rings this stub does not carry)
    for mode in ("student", "no_distill"):
        assert np.count_nonzero(_snapshot(mode).wrist_window), mode


# ---- CLI: the arms are actually launchable

def test_train_teacher_parser_accepts_student_and_mask_wrist():
    """P10A: without --student the `no_distill` arm named in
    docs/training_playbook.md is not runnable at all."""
    ns = _parse_train_teacher(["--tiny", "--synthetic", "--student", "--mask-wrist"])
    assert ns.student is True and ns.mask_wrist is True
    ns = _parse_train_teacher(["--tiny", "--synthetic"])
    assert ns.student is False and ns.mask_wrist is False


def test_train_teacher_passes_student_through_to_the_model_and_sampler():
    """The flag must reach PhantomModelConfig, build_model AND WindowSampler —
    the three places that hardcoded student=False."""
    import inspect
    import phantom.train.train_teacher as TT
    src = inspect.getsource(TT.main)
    assert "PhantomModelConfig(student=args.student" in src
    assert "build_model(hw, paths, student=args.student" in src
    assert "WindowSampler(hw, pm.bb, norm, student=args.student" in src
    assert "mask_wrist=args.mask_wrist" in src


def _parse_train_teacher(argv):
    """Parse train_teacher's real CLI without running the program."""
    import phantom.train.train_teacher as TT

    captured = {}

    class _Stop(Exception):
        pass

    real_parse = TT.argparse.ArgumentParser.parse_args

    def parse(self, args=None, namespace=None):
        ns = real_parse(self, args, namespace)
        captured["ns"] = ns
        raise _Stop
    TT.argparse.ArgumentParser.parse_args = parse
    try:
        with pytest.raises(_Stop):
            TT.main(argv)
    finally:
        TT.argparse.ArgumentParser.parse_args = real_parse
    return captured["ns"]


# ===========================================================================
# P10B — build from the checkpoint's model config; refuse a mismatch
# ===========================================================================

def _cosmos_available() -> bool:
    try:
        from phantom.backbone import loader as bl
        bl.setup_cosmos(load_paths())
        return True
    except Exception:
        return False


needs_cosmos = pytest.mark.skipif(not _cosmos_available(),
                                  reason="cosmos repo not importable")


@pytest.fixture(scope="module")
def tiny_time_true():
    """A tiny TEACHER trained (nominally) with the v4/v5 recipe, and the
    checkpoint it saves."""
    from phantom.config.model import AccConfig
    from phantom.config.training import TeacherTrainConfig
    from phantom.train.builder import build_model
    import tempfile
    hw = make_hw()
    mc = PhantomModelConfig(student=False, rope_time_mode="time_true",
                            acc=AccConfig(self_anticipation="two_pass"))
    pm = build_model(hw, load_paths(), student=False, tiny=True,
                     load_base=False, mc=mc)
    p = Path(tempfile.mkdtemp(prefix="p10b")) / "teacher.pt"
    C.save_phantom_checkpoint(p, pm.rf, hw=hw, bb=pm.bb, mc=mc,
                              train_cfg=TeacherTrainConfig(), step=1)
    return hw, pm, mc, p


@needs_cosmos
def test_time_true_checkpoint_refuses_an_aligned_model(tiny_time_true):
    """P10B's headline: distill_hid / relabel / finetune_hids built models
    with PhantomModelConfig() defaults (rope 'aligned', acc 'gt_noised'). The
    RoPE table is recomputed from mc at runtime, so no saved tensor mismatch
    could ever have caught it."""
    from phantom.train.builder import build_model
    hw, _pm, _mc, p = tiny_time_true
    wrong = build_model(hw, load_paths(), student=False, tiny=True,
                        load_base=False, mc=PhantomModelConfig(student=False))
    assert wrong.mc.rope_time_mode == "aligned"          # the silent default
    with pytest.raises(RuntimeError, match="model-config drift"):
        C.load_phantom_checkpoint(p, wrong.rf, hw=hw)
    err = None
    try:
        C.load_phantom_checkpoint(p, wrong.rf, hw=hw)
    except RuntimeError as e:
        err = str(e)
    assert "rope_time_mode" in err and "time_true" in err
    assert "self_anticipation" in err or "acc" in err


@needs_cosmos
def test_the_checkpoints_own_config_loads(tiny_time_true):
    hw, pm, mc, p = tiny_time_true
    payload = torch.load(str(p), map_location="cpu", weights_only=False)
    rebuilt = PhantomModelConfig.from_dict(payload["configs"]["model"])
    assert rebuilt == mc
    C.load_phantom_checkpoint(p, pm.rf, hw=hw)           # no raise


@needs_cosmos
def test_teacher_to_student_init_is_still_allowed(tiny_time_true):
    """`student` is the ONE legitimate difference — teacher->student init."""
    import dataclasses
    from phantom.train.builder import build_model
    hw, _pm, mc, p = tiny_time_true
    student = build_model(hw, load_paths(), student=True, tiny=True,
                          load_base=False,
                          mc=dataclasses.replace(mc, student=True))
    C.load_phantom_checkpoint(p, student.rf, hw=hw, allow_missing=True)


@needs_cosmos
def test_mask_wrist_is_an_ablation_not_a_config_drift(tiny_time_true):
    """vision_only deploys a trained checkpoint with the wrist input zeroed —
    that must load, or the arm is unbuildable again."""
    import dataclasses
    from phantom.train.builder import build_model
    hw, _pm, mc, p = tiny_time_true
    ablated = build_model(hw, load_paths(), student=False, tiny=True,
                          load_base=False,
                          mc=dataclasses.replace(mc, mask_wrist=True))
    C.load_phantom_checkpoint(p, ablated.rf, hw=hw)
    assert ablated.rf.hht.mask_wrist is True


@needs_cosmos
def test_the_three_programs_build_from_the_checkpoint_config():
    """distill_hid / relabel / finetune_hids must pass an `mc` to build_model
    — the whole of P10B is that they did not."""
    import inspect
    from phantom.dagger import relabel as RL
    from phantom.train import distill_hid as DH, finetune_hids as FH
    for fn in (DH.main, RL.relabel_root, FH.main):
        src = inspect.getsource(fn)
        assert "PhantomModelConfig.from_dict" in src, fn.__qualname__
        assert "mc=" in src, fn.__qualname__


# ===========================================================================
# P10A — the no_distill arm actually trains
# ===========================================================================

@needs_cosmos
def test_tiny_student_forward_and_backward(tmp_path):
    """`train_teacher --student` builds the STUDENT layout and takes a step —
    the `no_distill` control arm the paper's headline ratio needs."""
    from phantom.data.schema import NormStats
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    from phantom.train.builder import build_model
    from phantom.model.sequence import FrameGroup
    hw = make_hw(
        tactile={"field": {"h": 48, "w": 64},
                 "raw_img": {"h": 60, "w": 80, "c": 1},
                 "infer_img": {"h": 60, "w": 80, "c": 1}, "rate_hz": 30.0},
        cameras={"scene": {"color": {"h": 60, "w": 80, "c": 3}, "fps": 10.0}},
        recording={"field_ds": {"h": 24, "w": 32}, "field_ds_rate_hz": 30.0,
                   "keyframe_rate_hz": 5.0, "keyframe_ds": {"h": 24, "w": 32},
                   "infer_img_rate_hz": 10.0, "zarr_chunk_frames": 16},
        derived={"cpk_downsample": 4},
        wrist_ft={"window_s": 0.1},
    )
    mc = PhantomModelConfig(student=True, mask_wrist=True)     # vision_only
    pm = build_model(hw, load_paths(), student=True, tiny=True,
                     load_base=False, mc=mc)
    assert not pm.layout.has(FrameGroup.OBS_GEL)
    assert not pm.layout.has(FrameGroup.OBS_MECH)
    root = tmp_path / "eps"
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, pm.bb, NormStats.identity(), student=True)
    ds = C.WindowDataset(root, sampler, windows_per_episode=2)
    batch = C.collate_windows([ds[0], ds[1]])
    assert "gel" not in batch and "fields" not in batch   # student inputs only
    parts = pm.rf.training_step(batch)
    assert torch.isfinite(parts["total"])
    parts["total"].backward()
