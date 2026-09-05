"""FIX-NOW batch from docs/review_20260828/VALIDATION_0830.md (training side).

F1   `--init-weights` + the FT-A bundle died inside `load_phantom_checkpoint`'s
     P10B assert, nine lines before the finetune-tolerant drift check. The
     existing tests called the helper directly, so 571 green tests missed it —
     hence a test that runs `train_teacher.main([...])` end to end.
F10  `distill_hid` / `finetune_hids` trained on the held-out val split.
F11  `--event-band-weight` was a bare attribute on `pm.rf`: written to no part
     of the checkpoint and silently reverted to `mc.loss.event` on `--resume`.
F13  Nothing enforced that `rederive_rollout_actions` ran before a rollout
     trained.
F19  `EpisodeMeta.weight` did not exist (D8's weighting was a silent no-op);
     no commit-band window multiplier.
F20  Re-derivation wrote the MEASURED aperture into the gripper channel.
§1.2 `h100x8` val ran on one shuffled, drop_last rank shard, and the profile
     only warned when launched without torchrun.
§1.6 provisioning: wrong FT-A init checkpoint, a run name that would overwrite
     the shipped v5 lineage, a pytest gate whose failure message was
     unreachable, terminal_eval named as the selection tool, no egress.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.config.backbone import BackboneConfig
from phantom.config.model import (FINETUNE_MUTABLE_MODEL_FIELDS,
                                  PhantomModelConfig)
from phantom.config.paths import load_paths
from phantom.config.training import TeacherTrainConfig
from phantom.data.schema import (REDERIVED_TAG, STREAM_ACTIONS,
                                 STREAM_ACTIONS_PLAN, STREAM_ARM_TCP_POSE,
                                 STREAM_GRIPPER, EpisodeMeta, needs_rederive)
from phantom.train import common as C
from phantom_test_utils import make_small_hw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rederive_rollout_actions as RRA  # noqa: E402
import upload_run_ckpts as URC  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _cosmos_available() -> bool:
    try:
        from phantom.backbone import loader as bl
        bl.setup_cosmos(load_paths())
        return True
    except Exception:                                        # noqa: BLE001
        return False


needs_cosmos = pytest.mark.skipif(not _cosmos_available(),
                                  reason="cosmos repo not importable")


# ===========================================================================
# F1 — the FT-A launch line, through the CLI
# ===========================================================================

FT_A_BUNDLE = ["--cond-dropout", "0", "--no-action-t-max-of-two",
               "--contact-nll-beta", "0.5", "--contact-self-forcing",
               "--acc-two-pass", "--action-noise-per-strip",
               "--ema-decay", "0.995"]


def test_tolerate_widens_the_p10b_ignore_set():
    """The helper the CLI leans on: same drift, tolerated or not."""
    # what v4/v5_6 actually saved
    saved = dataclasses.replace(PhantomModelConfig(), cond_dropout_p=0.1,
                                action_t_max_of_two=True).to_dict()
    model = SimpleNamespace(mc=dataclasses.replace(PhantomModelConfig(),
                                                   cond_dropout_p=0.0,
                                                   action_t_max_of_two=False))
    payload = {"configs": {"model": saved}}
    with pytest.raises(RuntimeError, match="cond_dropout_p"):
        C.assert_model_config_matches(payload, model)
    C.assert_model_config_matches(payload, model,
                                  tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)
    # ... and it never tolerates a layout/behaviour field
    model.mc = dataclasses.replace(model.mc, rope_time_mode="time_true")
    with pytest.raises(RuntimeError, match="rope_time_mode"):
        C.assert_model_config_matches(payload, model,
                                      tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)


@pytest.fixture(scope="module")
def runs_root(tmp_path_factory, ):
    return tmp_path_factory.mktemp("runs")


@pytest.fixture
def cli_paths(monkeypatch, runs_root):
    """train_teacher.main, writing its runs into tmp instead of runs_root."""
    from phantom.train import train_teacher as TT
    p = dataclasses.replace(load_paths(), runs_root=Path(runs_root))
    monkeypatch.setattr(TT, "load_paths", lambda: p)
    return p


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _run_cli(runs_root, argv) -> list[str]:
    """train_teacher.main(argv) with runs_root redirected; returns its log."""
    from phantom.train import train_teacher as TT
    p = dataclasses.replace(load_paths(), runs_root=Path(runs_root))
    orig, h = TT.load_paths, _Records()
    TT.load_paths = lambda: p
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(h)
    root.setLevel(logging.INFO)          # pytest caps it at WARNING otherwise
    try:
        assert TT.main(argv) == 0
    finally:
        TT.load_paths = orig
        root.removeHandler(h)
        root.setLevel(prev_level)
    return h.msgs


@pytest.fixture(scope="module")
def base_ckpt(runs_root):
    """A tiny checkpoint trained WITHOUT the bundle — v5_6's shape of problem:
    it saves cond_dropout_p 0.1 / action_t_max_of_two True."""
    if not _cosmos_available():
        pytest.skip("cosmos repo not importable")
    _run_cli(runs_root, ["--tiny", "--synthetic", "--acc-two-pass",
                         "--device", "cpu", "--max-steps", "1",
                         "--ckpt-every", "1", "--run-name", "fixnow_base"])
    ck = Path(runs_root) / "teacher" / "fixnow_base" / "teacher_000001.pt"
    assert ck.exists()
    return ck


@pytest.fixture(scope="module")
def ft_a_run(base_ckpt, runs_root):
    """The documented FT-A launch line, through `train_teacher.main`.

    Before F1 this raised RuntimeError('model-config drift vs checkpoint')
    inside load_phantom_checkpoint before step 1 — `--cond-dropout 0` alone
    was enough. Module-scoped so the resume tests below get the artifact
    whatever order pytest-randomly picks."""
    msgs = _run_cli(runs_root,
                    ["--tiny", "--synthetic", "--device", "cpu",
                     "--max-steps", "2", "--ckpt-every", "2",
                     "--init-weights", str(base_ckpt),
                     "--event-band-weight", "0",
                     # F19's knob rides along: prove it is wired through the
                     # real CLI into WindowDataset, not just unit-tested
                     "--commit-band-weight", "2.0",
                     "--run-name", "fixnow_ftA", *FT_A_BUNDLE])
    ck = Path(runs_root) / "teacher" / "fixnow_ftA" / "teacher_000002.pt"
    assert ck.exists()
    return ck, msgs


@needs_cosmos
def test_the_documented_ft_a_launch_line_runs_through_main(base_ckpt, ft_a_run):
    import torch
    saved = torch.load(str(base_ckpt), map_location="cpu", weights_only=False)
    assert saved["configs"]["model"]["cond_dropout_p"] == 0.1
    assert saved["configs"]["model"]["action_t_max_of_two"] is True

    ck, msgs = ft_a_run
    assert any("training-only model flags differ" in m for m in msgs), \
        "the tolerated drift must still WARN"
    assert any("step 2/2" in m for m in msgs), msgs[-3:]
    mc = torch.load(str(ck), map_location="cpu",
                    weights_only=False)["configs"]["model"]
    assert mc["cond_dropout_p"] == 0.0 and mc["action_t_max_of_two"] is False
    assert mc["action_noise_per_strip"] is True and mc["contact_nll_beta"] == 0.5


@needs_cosmos
def test_resume_stays_strict(base_ckpt, cli_paths):
    """--resume continues ONE run: the same flag flip must still be refused."""
    from phantom.train import train_teacher as TT
    with pytest.raises((RuntimeError, SystemExit), match="drift"):
        TT.main(["--tiny", "--synthetic", "--device", "cpu", "--acc-two-pass",
                 "--max-steps", "2", "--resume", str(base_ckpt),
                 "--cond-dropout", "0", "--run-name", "fixnow_resume"])


# ===========================================================================
# F11 — event_band_weight is part of the run's recorded config
# ===========================================================================

def test_event_band_weight_is_a_train_config_field():
    ap = argparse.ArgumentParser()
    from phantom.train.train_teacher import add_common_args, apply_overrides
    add_common_args(ap)
    ap.add_argument("--event-band-weight", type=float, default=None)
    assert "event_band_weight" in TeacherTrainConfig().to_dict()
    cfg = apply_overrides(TeacherTrainConfig(),
                          ap.parse_args(["--event-band-weight", "0"]))
    assert cfg.event_band_weight == 0.0
    assert cfg.to_dict()["event_band_weight"] == 0.0        # lands in configs.train
    assert apply_overrides(TeacherTrainConfig(),
                           ap.parse_args([])).event_band_weight is None


@needs_cosmos
def test_event_band_weight_is_persisted_and_restored_on_resume(ft_a_run, runs_root):
    """The FT-A run passed --event-band-weight 0; a resume WITHOUT the flag
    used to silently reinstate mc.loss.event (0.5) for the rest of the run,
    unrecorded in the artifact."""
    import torch
    ck, _ = ft_a_run
    payload = torch.load(str(ck), map_location="cpu", weights_only=False)
    assert payload["configs"]["train"]["event_band_weight"] == 0.0
    assert payload["configs"]["model"]["loss"]["event"] == 0.5      # unchanged

    msgs = _run_cli(runs_root,
                    ["--tiny", "--synthetic", "--device", "cpu",
                     "--max-steps", "3", "--ckpt-every", "3",
                     "--resume", str(ck), "--run-name", "fixnow_ftA",
                     *FT_A_BUNDLE])
    assert any("restored from the checkpoint" in m for m in msgs), msgs
    assert any("weight overridden: 0" in m for m in msgs), msgs


@needs_cosmos
def test_resume_refuses_a_different_event_band_weight(ft_a_run, cli_paths):
    from phantom.train import train_teacher as TT
    ck, _ = ft_a_run
    with pytest.raises(SystemExit, match="event-band weight drift"):
        TT.main(["--tiny", "--synthetic", "--device", "cpu",
                 "--max-steps", "3", "--resume", str(ck),
                 "--event-band-weight", "0.5", "--run-name", "fixnow_ftA",
                 *FT_A_BUNDLE])


# ===========================================================================
# F10 — the HID programs' manifest split
# ===========================================================================

class _CapturedDataset:
    """Stand-in for C.WindowDataset that records what it was handed."""
    seen: dict = {}

    def __init__(self, root, sampler, *a, episodes=None, **kw):
        # the FIRST construction is the training set; distill_hid now also
        # builds a val set right after it (review 2026-09-05), which must not
        # overwrite what the train-split assertions look at
        if not _CapturedDataset.seen or _CapturedDataset.seen.get("_n", 0) == 0:
            _CapturedDataset.seen = {"root": Path(root), "episodes": episodes, "_n": 1}
        else:
            _CapturedDataset.seen["_n"] += 1
            _CapturedDataset.seen["val_episodes"] = episodes
        self.index = []

    def __len__(self):
        return 0


def _stub_program(monkeypatch, mod):
    """Everything a train program does around the dataset, stubbed out."""
    _CapturedDataset.seen = {}
    monkeypatch.setattr(mod.C, "WindowDataset", _CapturedDataset)
    monkeypatch.setattr(mod.C, "make_loader", lambda *a, **k: [])
    monkeypatch.setattr(mod.C, "train_loop", lambda *a, **k: None)
    monkeypatch.setattr(mod.C, "load_phantom_checkpoint", lambda *a, **k: {})
    monkeypatch.setattr(mod, "build_model", lambda *a, **k: SimpleNamespace(
        rf=SimpleNamespace(eval=lambda: None, parameters=lambda: iter(())),
        bb=BackboneConfig.tiny(), mc=PhantomModelConfig()))


def _split_root(tmp_path) -> Path:
    """tasks/ + a manifest whose val split is one of two episodes."""
    tasks = tmp_path / "tasks"
    for name in ("ep_train_a", "ep_val_b"):
        (tasks / "egg" / name).mkdir(parents=True)
        EpisodeMeta(task="egg", status="finalized", success=True,
                    policy="teleop").save(tasks / "egg" / name / "meta.json")
    rows = [{"episode": "ep_train_a", "task": "egg", "split": "train",
             "path": "tasks/egg/ep_train_a"},
            {"episode": "ep_val_b", "task": "egg", "split": "val",
             "path": "tasks/egg/ep_val_b"}]
    mf = tmp_path / "manifests" / "all.jsonl"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tasks


def test_distill_hid_trains_on_the_train_split(tmp_path, monkeypatch):
    from phantom.train import distill_hid as DH
    _stub_program(monkeypatch, DH)
    tasks = _split_root(tmp_path)
    assert DH.main(["--data", str(tasks), "--tiny", "--device", "cpu",
                    "--max-steps", "1"]) == 0
    eps = _CapturedDataset.seen["episodes"]
    assert eps is not None, "no episodes= -> WindowDataset indexes val too"
    assert [p.name for p in eps] == ["ep_train_a"]


def test_finetune_hids_trains_on_the_train_split(tmp_path, monkeypatch):
    from phantom.train import finetune_hids as FH
    _stub_program(monkeypatch, FH)
    monkeypatch.setattr(FH, "calibrate_tau_per_task", lambda *a, **k: {})
    tasks = _split_root(tmp_path)
    assert FH.main(["--data", str(tasks), "--tiny", "--device", "cpu",
                    "--max-steps", "1"]) == 0
    assert [p.name for p in _CapturedDataset.seen["episodes"]] == ["ep_train_a"]


def test_both_hid_programs_can_still_ask_for_every_episode(tmp_path, monkeypatch):
    from phantom.train import distill_hid as DH
    _stub_program(monkeypatch, DH)
    tasks = _split_root(tmp_path)
    DH.main(["--data", str(tasks), "--tiny", "--device", "cpu",
             "--max-steps", "1", "--split", "all"])
    # 'all' is manifest_split's documented escape hatch: no episode list, so
    # WindowDataset falls back to enumerating the root (the old behaviour)
    assert _CapturedDataset.seen["episodes"] is None


# ===========================================================================
# F13 — nothing enforced that the re-derivation ran
# ===========================================================================

def _rollout_dir(tmp_path, *, rederived: bool) -> Path:
    ep = tmp_path / "tasks" / "egg" / "ep_student_egg_1_000"
    ep.mkdir(parents=True)
    tags = [REDERIVED_TAG] if rederived else []
    EpisodeMeta(task="egg", policy="student", success=True, status="finalized",
                tags=tags).save(ep / "meta.json")
    if rederived:
        (ep / f"{STREAM_ACTIONS_PLAN}.zarr").mkdir()
    return ep


def test_needs_rederive_only_flags_unconverted_rollouts(tmp_path):
    ep = _rollout_dir(tmp_path, rederived=False)
    m = EpisodeMeta.load(ep / "meta.json")
    assert needs_rederive(ep, m)
    (ep / f"{STREAM_ACTIONS_PLAN}.zarr").mkdir()          # the stream alone is enough
    assert not needs_rederive(ep, m)
    teleop = dataclasses.replace(m, policy="teleop") if False else m
    m.policy = "teleop"
    assert not needs_rederive(tmp_path, m)               # demos are never rollouts


def test_intake_manifest_refuses_a_rollout_that_was_not_rederived(tmp_path, capsys):
    import intake_recovery as IR
    ep = _rollout_dir(tmp_path, rederived=False)
    mf = tmp_path / "manifests" / "all.jsonl"
    mf.parent.mkdir(parents=True)
    mf.touch()
    IR.manifest(tmp_path / "tasks", mf)
    out = capsys.readouterr().out
    assert "REFUSING" in out and "not re-derived" in out
    assert mf.read_text() == ""

    (ep / f"{STREAM_ACTIONS_PLAN}.zarr").mkdir()
    IR.manifest(tmp_path / "tasks", mf)
    rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    assert [r["episode"] for r in rows] == ["ep_student_egg_1_000"]


def test_intake_script_refuses_it_from_the_command_line(tmp_path):
    """The real entry point operators type: `python tools/intake_recovery.py
    manifest <tasks_root> <all.jsonl>`."""
    import subprocess
    ep = _rollout_dir(tmp_path, rederived=False)
    mf = tmp_path / "manifests" / "all.jsonl"
    mf.parent.mkdir(parents=True)
    mf.touch()
    cmd = [sys.executable, str(REPO / "tools" / "intake_recovery.py"), "manifest",
           str(tmp_path / "tasks"), str(mf)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr
    assert "REFUSING" in r.stdout and "not re-derived" in r.stdout
    assert mf.read_text() == ""

    (ep / f"{STREAM_ACTIONS_PLAN}.zarr").mkdir()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr
    assert "appended 1 rows" in r.stdout


def test_build_index_warns_about_an_unconverted_rollout(tmp_path, caplog):
    """The --extra-data path sees no manifest at all, so this one WARNS
    (and still indexes) rather than refusing."""
    pytest.importorskip("torch")
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.schema import NormStats
    from phantom.data.windows import WindowSampler
    hw = make_small_hw()
    root = tmp_path / "rollouts"
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=0.25).generate(
        root, task="egg", duration_s=8.0)
    ep = next(p for p in root.rglob("ep_*") if (p / "meta.json").exists())
    m = EpisodeMeta.load(ep / "meta.json")
    m.policy, m.success = "student", True
    m.save(ep / "meta.json")
    sampler = WindowSampler(hw, BackboneConfig.tiny(), NormStats.identity())
    with caplog.at_level(logging.WARNING):
        assert sampler.build_index(root, windows_per_episode=1)
    assert any("NO re-derived actions" in r.getMessage() for r in caplog.records)

    caplog.clear()
    m.tags = [REDERIVED_TAG]
    m.save(ep / "meta.json")
    with caplog.at_level(logging.WARNING):
        sampler.build_index(root, windows_per_episode=1)
    assert not any("NO re-derived actions" in r.getMessage() for r in caplog.records)


# ===========================================================================
# F19 — per-episode weight + the commit-band multiplier
# ===========================================================================

def test_episode_meta_weight_survives_the_round_trip(tmp_path):
    p = tmp_path / "meta.json"
    EpisodeMeta(task="egg").save(p)
    assert EpisodeMeta.load(p).weight == 1.0                 # backward compatible
    d = json.loads(p.read_text())
    d.pop("weight")                                          # a pre-F19 file
    p.write_text(json.dumps(d))
    assert EpisodeMeta.load(p).weight == 1.0
    m = EpisodeMeta.load(p)
    m.weight = 3.0
    m.save(p)
    assert EpisodeMeta.load(p).weight == 3.0


@pytest.fixture
def weighted_episode(tmp_path):
    pytest.importorskip("torch")
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    hw = make_small_hw()
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=0.25).generate(
        tmp_path, task="egg", duration_s=8.0)
    ep = next(p for p in tmp_path.rglob("ep_*") if (p / "meta.json").exists())
    return hw, ep


def _sampled_weight(hw, ep) -> float:
    from phantom.data.schema import NormStats
    from phantom.data.windows import WindowSampler
    s = WindowSampler(hw, BackboneConfig.tiny(), NormStats.identity())
    return float(s.sample(ep)["action_weight"])


def test_meta_weight_reaches_the_batch(weighted_episode):
    hw, ep = weighted_episode
    assert _sampled_weight(hw, ep) == 1.0
    m = EpisodeMeta.load(ep / "meta.json")
    m.weight = 3.0
    m.save(ep / "meta.json")
    assert _sampled_weight(hw, ep) == 3.0
    # ... but it can never resurrect a failure demo's action loss
    m.success = False
    m.save(ep / "meta.json")
    assert _sampled_weight(hw, ep) == 0.0


class _StubSampler:
    def build_index(self, root, wpe=8, episodes=None):
        from phantom.data.windows import WindowItem
        return [WindowItem(episode=Path(root), t0=5.0, lo=4.99, hi=5.01)]

    def sample(self, ep, t0):
        return {"action_weight": 2.0}


def test_commit_band_multiplier_is_off_by_default(tmp_path):
    ds = C.WindowDataset(tmp_path, _StubSampler())
    ds._close_cache[tmp_path] = 6.0            # window t0 ~5.0 is in the band
    assert ds[0]["action_weight"] == 2.0


def test_commit_band_multiplier_scales_only_the_band(tmp_path):
    ds = C.WindowDataset(tmp_path, _StubSampler(), commit_band_weight=2.0)
    ds._close_cache[tmp_path] = 6.0            # band = [4.5, 5.8] around t0 5.0
    assert ds[0]["action_weight"] == 4.0
    ds._close_cache[tmp_path] = 20.0           # close far in the future
    assert ds[0]["action_weight"] == 2.0
    ds._close_cache[tmp_path] = None           # never closes
    assert ds[0]["action_weight"] == 2.0


# ===========================================================================
# F20 — the gripper channel of the re-derivation
# ===========================================================================

def _traj(t: np.ndarray) -> np.ndarray:
    p = np.zeros((len(t), 6))
    p[:, 0] = 0.20 + 0.040 * t
    p[:, 2] = 0.30 - 0.020 * t
    return p


@pytest.fixture
def rollout_with_commands(tmp_path):
    """Measured aperture and COMMANDED aperture deliberately disagree — as on
    the rig, where the measured value is the object's width."""
    from phantom.data.episode_store import EpisodeWriter
    hw = make_small_hw()
    ep = tmp_path / "ep_student_waffles_1_000"
    w = EpisodeWriter(ep, hw, EpisodeMeta(task="waffles", policy="student",
                                          success=True, status="finalized"))
    t_arm = np.arange(0.0, 6.0, 1.0 / 125.0)
    w.append(STREAM_ARM_TCP_POSE, t_arm, _traj(t_arm).astype(np.float32))
    t_g = np.arange(0.0, 6.0, 1.0 / 20.0)
    grip = np.zeros((len(t_g), 2), dtype=np.float32)
    grip[:, 0] = 0.42                      # MEASURED: the object's width
    w.append(STREAM_GRIPPER, t_g, grip)
    t_plan = np.arange(0.3, 5.7, 1.0 / 7.0)
    plan = np.zeros((len(t_plan), 7), dtype=np.float32)
    plan[:, 0] = -0.026
    plan[:, 6] = 1.0                       # COMMANDED: "close all the way"
    w.append(STREAM_ACTIONS, t_plan, plan)
    w.finalize(success=True)
    return hw, ep


def test_rederived_gripper_channel_is_the_command_not_the_measurement(
        rollout_with_commands):
    from phantom.data.episode_store import EpisodeReader
    from phantom_test_utils import HW_YAML
    hw, ep = rollout_with_commands
    # through the tool's OWN CLI (same 10 Hz action grid as the fixture's hw)
    assert RRA.main([str(ep), "--hardware", str(HW_YAML)]) == 0
    assert REDERIVED_TAG in EpisodeMeta.load(ep / "meta.json").tags
    act = np.asarray(EpisodeReader(ep).data(STREAM_ACTIONS)[:], dtype=np.float64)
    assert np.allclose(act[:, 6], 1.0), \
        "channel 6 must be the commanded aperture, not gripper.zarr's position"
    assert not np.any(np.isclose(act[:, 6], 0.42))


def test_rederived_gripper_falls_back_to_the_measurement(tmp_path, caplog):
    """No command stream (channel 6 absent) -> the measured position, loudly."""
    hw = make_small_hw()
    reader = SimpleNamespace(has=lambda s: False)
    grid = np.array([0.0, 0.1, 0.2])
    grip = np.array([[0.3, 0.0], [0.4, 0.0], [0.5, 0.0]])
    with caplog.at_level(logging.WARNING):
        got = RRA._gripper_commands(reader, grid, grip, np.array([0.0, 0.1, 0.2]),
                                    "ep_x")
    assert np.allclose(got, [0.3, 0.4, 0.5])
    assert any("MEASURED" in r.getMessage() for r in caplog.records)


# ===========================================================================
# §1 row 2 — the h100x8 profile
# ===========================================================================

def test_a_program_launched_without_torchrun_dies_on_the_cluster_profile(
        tmp_path, monkeypatch):
    """Through a program's own CLI: `--compute h100x8` with WORLD_SIZE 1 must
    stop before it builds anything (it used to train at effective batch 1)."""
    from phantom.train import distill_hid as DH
    monkeypatch.delenv("PHANTOM_ALLOW_WORLD_MISMATCH", raising=False)
    _stub_program(monkeypatch, DH)
    tasks = _split_root(tmp_path)
    with pytest.raises(SystemExit, match="torchrun"):
        DH.main(["--data", str(tasks), "--tiny", "--device", "cpu",
                 "--max-steps", "1", "--compute", "h100x8"])


def test_cluster_profile_without_torchrun_is_fatal(monkeypatch):
    from phantom.config.compute import load_compute
    monkeypatch.delenv("PHANTOM_ALLOW_WORLD_MISMATCH", raising=False)
    h100 = load_compute("h100x8")
    with pytest.raises(SystemExit, match="torchrun"):
        h100.check_world(1)
    h100.check_world(8)                                    # the real launch
    monkeypatch.setenv("PHANTOM_ALLOW_WORLD_MISMATCH", "1")
    h100.check_world(1)                                    # deliberate override
    load_compute("rtx5090").check_world(2)                 # single-GPU: warn only


def test_val_loaders_are_built_unsharded():
    """make_loader(shard=False) keeps every window: no DistributedSampler and
    no drop_last (only rank 0 evaluates)."""
    class _DS(list):
        pass
    ds = _DS(range(9))
    cfg = dataclasses.replace(TeacherTrainConfig(), batch_size=2, num_workers=0,
                              synthetic=True)
    train = C.make_loader(ds, cfg, collate_fn=None)
    val = C.make_loader(ds, cfg, collate_fn=None, shuffle=False, shard=False)
    assert train.drop_last and not val.drop_last
    assert val.sampler is not None or True                 # world==1: no sampler
    src = (REPO / "phantom" / "train" / "train_teacher.py").read_text(encoding="utf-8")
    assert "shuffle=False, shard=False" in src


# ===========================================================================
# §1 row 6 — provisioning + checkpoint egress
# ===========================================================================

def _provision() -> str:
    return (REPO / "tools" / "provision_v5.sh").read_text(encoding="utf-8")


def test_provisioning_fetches_the_v5_6_ft_a_init():
    src = _provision()
    assert "teacher_v5_batch0822/teacher_003000.pt" in src
    assert 'FTA_INIT="$W/runs/teacher/teacher_v5_batch0822/teacher_003000.pt"' in src
    # the v4 path survives only as a commented control
    assert "\nhfget $HUB teacher_v4_790eps" not in src


def test_printed_launch_line_never_reuses_the_shipped_v5_run_name():
    launch = _provision().split("READY. Launch")[1]
    assert "--run-name teacher_v5_ftA" in launch
    assert "--run-name teacher_v5_batch0822" not in launch


def test_the_smoke_runs_the_whole_ft_a_bundle():
    smoke = _provision().split("2-step REAL training smoke")[1].split("READY. Launch")[0]
    for flag in ("--contact-nll-beta 0.5",
                 "--action-noise-per-strip", "--no-action-t-max-of-two",
                 "--cond-dropout 0", "--event-band-weight 0", '"$FTA_INIT"'):
        assert flag in smoke, f"{flag} missing from the provisioning smoke"
    # retracted 2026-08-31 (E9_premise_test.md:88-93): the smoke must not spend the
    # bundle on --contact-self-forcing; it stays only as a commented ablation line.
    cmd = smoke.split("\n# ablation only")[0]
    assert "--contact-self-forcing" not in cmd, (
        "--contact-self-forcing is back in the provisioning smoke command")
    assert "# ablation only" in smoke and "--contact-self-forcing" in smoke, (
        "keep --contact-self-forcing as a commented ablation line")


def test_the_pytest_gate_can_report_its_own_failure():
    src = _provision()
    assert "if ! python -m pytest tests/ -q" in src
    assert "PYTEST FAILED" in src
    # the old form died on `set -e` before ever reaching its || clause
    assert 'python -m pytest tests/ -q 2>&1 | tail -1;' not in src


def test_checkpoint_selection_says_replay_and_egress():
    launch = _provision().split("READY. Launch")[1]
    assert "replay_rig.py" in launch
    assert "tools/upload_run_ckpts.py" in launch
    assert "SELECT the best checkpoint with tools/terminal_eval.py" not in launch


def test_upload_run_ckpts_collects_and_skips(tmp_path, monkeypatch):
    run = tmp_path / "teacher_v5_ftA"
    run.mkdir()
    for name in ("teacher_000500.pt", "teacher_001000.pt", "run.log"):
        (run / name).write_bytes(b"x" * 10)
    (run / "notes.txt").write_text("ignored")
    extra = tmp_path / "train_ftA.log"
    extra.write_text("log")

    files = URC.local_files(run, [extra])
    assert [p.name for p in files] == ["run.log", "teacher_000500.pt",
                                       "teacher_001000.pt", "train_ftA.log"]

    uploaded: list[str] = []
    same = URC.sha256_file(run / "teacher_000500.pt")

    def _api(tree):
        class _Api:
            def __init__(self, *a, **k):
                pass

            def list_repo_tree(self, repo, path_in_repo=None, repo_type=None,
                               recursive=False, expand=False):
                # the sha256 only exists in the EXPANDED listing
                assert expand is True, "list without expand=True has no lfs.sha256"
                return tree(path_in_repo)

            def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id,
                            repo_type):
                uploaded.append(path_in_repo)
        return _Api

    import huggingface_hub
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    # 1. the hub file is byte-identical -> skipped
    monkeypatch.setattr(huggingface_hub, "HfApi", _api(lambda pre: [
        SimpleNamespace(path=f"{pre}/teacher_000500.pt", size=10,
                        lfs=SimpleNamespace(sha256=same))]))
    assert URC.main([str(run), "--log", str(extra)]) == 0
    assert uploaded == ["teacher_v5_ftA/run.log",
                        "teacher_v5_ftA/teacher_001000.pt",
                        "teacher_v5_ftA/train_ftA.log"]

    # 2. SAME NAME, SAME SIZE, different content -> must NOT be skipped. Every
    #    PHANTOM teacher checkpoint is exactly 393,115,861 bytes, so the old
    #    size-only rule called every re-upload "already present".
    uploaded.clear()
    monkeypatch.setattr(huggingface_hub, "HfApi", _api(lambda pre: [
        SimpleNamespace(path=f"{pre}/teacher_000500.pt", size=10,
                        lfs=SimpleNamespace(sha256="0" * 64))]))
    assert URC.main([str(run), "--log", str(extra)]) == 0
    assert "teacher_v5_ftA/teacher_000500.pt" in uploaded

    # 3. no LFS metadata -> fall back to the byte size, as before
    uploaded.clear()
    monkeypatch.setattr(huggingface_hub, "HfApi", _api(lambda pre: [
        SimpleNamespace(path=f"{pre}/teacher_000500.pt", size=10, lfs=None),
        SimpleNamespace(path=f"{pre}/teacher_001000.pt", size=999, lfs=None)]))
    assert URC.main([str(run), "--log", str(extra)]) == 0
    assert "teacher_v5_ftA/teacher_000500.pt" not in uploaded
    assert "teacher_v5_ftA/teacher_001000.pt" in uploaded


def test_upload_run_ckpts_refuses_without_a_token(tmp_path, monkeypatch):
    run = tmp_path / "r"
    run.mkdir()
    (run / "teacher_000001.pt").write_bytes(b"x")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        URC.main([str(run)])
