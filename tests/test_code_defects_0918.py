"""Code defects found by the 2026-09-18 distillation audit (docs/history/
CODE_DEFECTS_20260918.md). Every behavioural change is OPT-IN: each item gets
an entry-point test for the new flag AND a regression test that the DEFAULT
path is byte-identical to what produced the paper's checkpoints.

(a) traj_distill regressed the student onto `c_pack.pack(t_pred.cpk)`, a
    pack(unpack(.)) round trip that paints a full-amplitude CoP bump on every
    step the teacher imagined as no-contact (`pack` keys the amplitude on NaN;
    `unpack` never returns NaN)  ->  --traj-target {roundtrip,raw_latents}.
(b) on-policy DAgger rollouts judged unsuccessful got action_weight 0 through
    `is_failure_demo`, so round 2's rollouts grounded no actions at all
      ->  --rollout-action-weight {failure_demo,teacher_supervised}.
(c) --grasp-frac / --photo-aug / --commit-band-weight / --split reached no
    part of the student checkpoint, so a --resume that forgot them reverted
    the window recipe silently  ->  HIDConfig fields + the teacher's existing
    accept/refuse lock, with --override-recipe as the escape hatch.
(d) the lock tied a student to its teacher's absolute path  ->  the recipe
    also records sha256[:12], and a RELOCATED teacher with a matching hash is
    accepted with a warning.
(e) configs/hardware.yaml (bench, wrist_ft.rate_hz 500) is shape-incompatible
    with every rig checkpoint (nuc, 125 Hz): wrist_window_len 125 vs 31. The
    gate is real; the refusal now says which knob and which config.
(f) SequenceLayout's docstring described a T=13/11 layout that has not existed
    for a long time (docstring only).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from phantom.config.hardware import load_hardware
from phantom.config.model import AccConfig, PhantomModelConfig
from phantom.config.paths import load_paths
from phantom.config.training import HIDConfig
from phantom.data.schema import NormStats
from phantom.data.synthetic import SyntheticEpisodeGenerator
from phantom.data.windows import WindowSampler
from phantom.model.sequence import FrameGroup, SequenceLayout
from phantom.train import common as C
from phantom.train.builder import build_model
from phantom.train.distill_hid import (distill_step, reconcile_teacher_ckpt,
                                       teacher_sample_layout,
                                       traj_distill_target)
from phantom_test_utils import make_small_hw

REPO = Path(__file__).resolve().parents[1]


# ===========================================================================
# shared tiny teacher/student pair + one real batch (test_hid_round2_knobs
# pattern: real models, real window, no weights)
# ===========================================================================

@pytest.fixture(scope="module")
def pair_and_batch(tmp_path_factory):
    hw = make_small_hw()
    root = tmp_path_factory.mktemp("eps0918")
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=6.0)
    mc = PhantomModelConfig(acc=AccConfig(self_anticipation="two_pass"))
    paths = load_paths()
    teacher = build_model(hw, paths, student=False, tiny=True, load_base=False, mc=mc)
    student = build_model(hw, paths, student=True, tiny=True, load_base=False,
                          mc=dataclasses.replace(mc, student=True))
    sampler = WindowSampler(hw, teacher.bb, NormStats.identity(), student=False)
    ds = C.WindowDataset(root, sampler, windows_per_episode=1)
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=C.collate_windows)))
    teacher.rf.eval()
    return teacher, student, batch


# ===========================================================================
# (a) traj_distill target space
# ===========================================================================

@pytest.mark.requires_cosmos_repo
def test_the_pack_roundtrip_invents_a_cop_bump_on_a_no_contact_step(pair_and_batch):
    """The defect itself: a contact package whose CoP is NaN (= no contact)
    packs to an EMPTY bump channel, but once it has been through `unpack` the
    centroid is finite and `pack` paints a full-amplitude Gaussian there —
    which is exactly what `pack(t_pred.cpk)` fed traj_distill, because the
    teacher's package always comes out of `unpack`."""
    from phantom.model.ace.packing import _CH_COP
    teacher, student, _ = pair_and_batch
    packer = student.rf.c_pack
    with torch.no_grad():
        t_pred = teacher.rf.sample(C.to_device(pair_and_batch[2], "cpu"), nfe=1)
    no_contact = dataclasses.replace(
        t_pred.cpk, cop=torch.full_like(t_pred.cpk.cop, float("nan")))
    amp_direct = float(packer.pack(no_contact)[:, _CH_COP].abs().max())
    amp_roundtrip = float(
        packer.pack(packer.unpack(packer.pack(no_contact).float()))[:, _CH_COP]
        .abs().max())
    assert amp_direct == 0.0, "a NaN CoP must pack to an EMPTY bump channel"
    assert amp_roundtrip > 0.5, (
        "reproduction: the round trip paints a full-amplitude CoP bump where "
        f"the teacher imagined no contact (got {amp_roundtrip})")


@pytest.mark.requires_cosmos_repo
def test_default_traj_target_is_the_shipped_pack_roundtrip(pair_and_batch):
    """REGRESSION: the default target tensor is bit-for-bit the expression the
    paper's checkpoints were distilled against."""
    teacher, student, batch = pair_and_batch
    b = C.to_device(batch, "cpu")
    with torch.no_grad():
        t_pred = teacher.rf.sample(b, nfe=1)
    cfg = HIDConfig(tiny=True, synthetic=True)
    assert cfg.traj_target == "roundtrip"
    got = traj_distill_target(student.rf, teacher.rf, t_pred, cfg, torch.float32)
    want = student.rf.c_pack.pack(
        t_pred.cpk.detach().to(student.rf.device)).to(torch.float32)
    assert torch.equal(got, want)


@pytest.mark.requires_cosmos_repo
def test_raw_latents_target_is_the_teachers_contact_slice(pair_and_batch):
    teacher, student, batch = pair_and_batch
    b = C.to_device(batch, "cpu")
    with torch.no_grad():
        t_pred = teacher.rf.sample(b, nfe=1)
    cfg = HIDConfig(tiny=True, synthetic=True, traj_target="raw_latents")
    got = traj_distill_target(student.rf, teacher.rf, t_pred, cfg, torch.float32)
    sl = teacher.rf.layout.frame_slice(FrameGroup.CONTACT)
    want = t_pred.x_final_B_C_T_H_W[:, :, sl].detach().to(torch.float32)
    assert torch.equal(got, want)
    assert not got.requires_grad
    # same shape as the slice traj_distill compares it against ...
    sl_s = student.rf.layout.frame_slice(FrameGroup.CONTACT)
    assert got.shape[2] == sl_s.stop - sl_s.start
    # ... and genuinely a different target from the round trip
    rt = traj_distill_target(student.rf, teacher.rf, t_pred,
                             HIDConfig(tiny=True, synthetic=True), torch.float32)
    assert not torch.allclose(got, rt)


@pytest.mark.requires_cosmos_repo
def test_teacher_sample_layout_refuses_a_layout_that_does_not_fit(pair_and_batch):
    teacher, student, batch = pair_and_batch
    b = C.to_device(batch, "cpu")
    with torch.no_grad():
        t_pred = teacher.rf.sample(b, nfe=1)
    assert teacher_sample_layout(teacher.rf, t_pred) is teacher.rf.layout
    short = dataclasses.replace(
        t_pred, x_final_B_C_T_H_W=t_pred.x_final_B_C_T_H_W[:, :, :-1])
    with pytest.raises(RuntimeError, match="layout is unknown"):
        teacher_sample_layout(teacher.rf, short)


@pytest.mark.requires_cosmos_repo
def test_raw_latents_gives_a_finite_step_with_the_same_terms(pair_and_batch):
    """Opt-in: the alternative target trains — same loss terms, all finite,
    gradient reaches the student trunk. (distill_step is not seed-reproducible
    on this tree, so the two targets are compared directly above rather than
    by re-running the step.)"""
    teacher, student, batch = pair_and_batch
    base = distill_step(student.rf, teacher.rf, batch,
                        HIDConfig(tiny=True, synthetic=True), "cpu")
    student.rf.zero_grad(set_to_none=True)
    alt = distill_step(student.rf, teacher.rf, batch,
                       HIDConfig(tiny=True, synthetic=True,
                                 traj_target="raw_latents"), "cpu")
    assert set(alt) == set(base)
    assert all(torch.isfinite(v).all() for v in alt.values())
    alt["traj_distill"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in student.rf.parameters() if p.requires_grad)


@pytest.mark.requires_cosmos_repo
def test_the_target_reaches_only_the_traj_term(pair_and_batch):
    """`cpk_target` is consumed by traj_distill and nothing else, so the flag
    cannot move behavior_match / event_distill / the grounding terms."""
    import inspect
    from phantom.train import distill_hid as DH
    src = inspect.getsource(DH.distill_step)
    assert src.count("cpk_target") == 2         # the assignment + the residual
    assert "x0_pred_s[:, :, sl].float() - cpk_target.float()" in src


@pytest.mark.requires_cosmos_repo
def test_traj_target_is_refused_when_unknown(pair_and_batch):
    teacher, student, batch = pair_and_batch
    b = C.to_device(batch, "cpu")
    with torch.no_grad():
        t_pred = teacher.rf.sample(b, nfe=1)
    with pytest.raises(ValueError, match="traj_target"):
        traj_distill_target(student.rf, teacher.rf, t_pred,
                            HIDConfig(traj_target="whatever"), torch.float32)


# ===========================================================================
# (b) on-policy rollout action weight
# ===========================================================================

@pytest.fixture(scope="module")
def rollout_ep(tmp_path_factory):
    hw = make_small_hw()
    root = tmp_path_factory.mktemp("rollouts0918")
    ep = SyntheticEpisodeGenerator(hw, seed=3, rate_scale=1.0).generate(
        root, task="Carton", duration_s=6.0)
    return hw, ep


def _meta_patch(ep: Path, **fields) -> None:
    d = json.loads((ep / "meta.json").read_text())
    d.update(fields)
    (ep / "meta.json").write_text(json.dumps(d))


def _weight(hw, ep, mode: str) -> float:
    from phantom.config.backbone import BackboneConfig
    s = WindowSampler(hw, BackboneConfig(), NormStats.identity(),
                      rollout_action_weight=mode)
    lo, hi = s.valid_range(ep)
    return float(s.sample(ep, 0.5 * (lo + hi))["action_weight"])


def test_a_judged_rollout_grounds_nothing_by_default(rollout_ep):
    """REPRODUCTION + REGRESSION: `success is False` on an on-policy rollout
    routes through `is_failure_demo`, so the default weighting — the one every
    shipped student was distilled under — is 0."""
    hw, ep = rollout_ep
    _meta_patch(ep, policy="student_v6", success=False, weight=2.0,
                tags=["synthetic", "actions_rederived"], status="finalized")
    assert _weight(hw, ep, "failure_demo") == 0.0


def test_teacher_supervised_gives_a_judged_rollout_its_weight(rollout_ep):
    hw, ep = rollout_ep
    _meta_patch(ep, policy="student_v6", success=False, weight=2.0,
                tags=["synthetic", "actions_rederived"], status="finalized")
    assert _weight(hw, ep, "teacher_supervised") == 2.0


def test_teacher_supervised_still_zeroes_a_deliberate_failure_demo(rollout_ep):
    hw, ep = rollout_ep
    _meta_patch(ep, policy="student_v6", success=False, weight=2.0,
                tags=["synthetic", "actions_rederived", "deliberate_failure"])
    assert _weight(hw, ep, "teacher_supervised") == 0.0
    _meta_patch(ep, task="Carton_fail", success=True,
                tags=["synthetic", "actions_rederived"])
    assert _weight(hw, ep, "teacher_supervised") == 0.0


def test_teacher_supervised_still_zeroes_a_teleop_failure_demo(rollout_ep):
    """Not a policy rollout: a teleop demo judged failed keeps the old rule."""
    hw, ep = rollout_ep
    _meta_patch(ep, policy="teleop", task="Carton", success=False, weight=1.0,
                tags=["synthetic"])
    assert _weight(hw, ep, "teacher_supervised") == 0.0


def test_teacher_supervised_still_zeroes_a_rollout_with_proposal_actions(rollout_ep):
    """F13: without tools/rederive_rollout_actions.py the `actions` stream is
    the executor's PRE-CLAMP proposal — grounding it is the defect, not a fix."""
    hw, ep = rollout_ep
    _meta_patch(ep, policy="student_v6", task="Carton", success=False,
                weight=2.0, tags=["synthetic"])
    assert _weight(hw, ep, "teacher_supervised") == 0.0


def test_a_successful_rollout_is_unaffected_by_the_mode(rollout_ep):
    hw, ep = rollout_ep
    _meta_patch(ep, policy="student_v6", task="Carton", success=True,
                weight=1.5, tags=["synthetic", "actions_rederived"])
    assert _weight(hw, ep, "failure_demo") == 1.5
    assert _weight(hw, ep, "teacher_supervised") == 1.5


def test_an_unknown_rollout_mode_is_refused():
    from phantom.config.backbone import BackboneConfig
    with pytest.raises(ValueError, match="rollout_action_weight"):
        WindowSampler(make_small_hw(), BackboneConfig(), NormStats.identity(),
                      rollout_action_weight="teacher")


# ===========================================================================
# (a)+(b) argparse -> recipe
# ===========================================================================

def _cfg_from_argv(argv: list[str]) -> HIDConfig:
    from phantom.train import distill_hid as DH
    from phantom.train.train_teacher import apply_overrides
    ap = argparse.ArgumentParser()
    args = DH.build_parser(ap).parse_args(argv)
    return apply_overrides(HIDConfig(), args)


def test_the_new_flags_default_to_the_shipped_recipe():
    cfg = _cfg_from_argv([])
    assert cfg.traj_target == "roundtrip"
    assert cfg.rollout_action_weight == "failure_demo"
    assert (cfg.split, cfg.grasp_frac, cfg.photo_aug,
            cfg.commit_band_weight) == ("train", 0.0, 0.0, 1.0)


def test_the_new_flags_reach_the_recipe():
    cfg = _cfg_from_argv(["--traj-target", "raw_latents",
                          "--rollout-action-weight", "teacher_supervised",
                          "--grasp-frac", "0.3", "--photo-aug", "1.0",
                          "--commit-band-weight", "1.6", "--split", "all"])
    assert cfg.traj_target == "raw_latents"
    assert cfg.rollout_action_weight == "teacher_supervised"
    assert (cfg.split, cfg.grasp_frac, cfg.photo_aug,
            cfg.commit_band_weight) == ("all", 0.3, 1.0, 1.6)


def test_every_recipe_flag_is_a_config_field_so_it_lands_in_the_checkpoint():
    d = HIDConfig().to_dict()
    for k in ("traj_target", "rollout_action_weight", "split", "grasp_frac",
              "photo_aug", "commit_band_weight", "teacher_ckpt",
              "teacher_ckpt_sha12"):
        assert k in d, k


# ===========================================================================
# (c) the recipe survives --resume (real CLI, tiny synthetic student)
# ===========================================================================

TINY = ["--tiny", "--synthetic", "--device", "cpu"]
RECIPE = ["--grasp-frac", "0.3", "--photo-aug", "1.0",
          "--commit-band-weight", "1.6", "--split", "all"]


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _run_distill(runs_root, argv) -> list[str]:
    from phantom.train import distill_hid as DH
    p = dataclasses.replace(load_paths(), runs_root=Path(runs_root))
    orig, h = DH.load_paths, _Records()
    DH.load_paths = lambda: p
    root = logging.getLogger()
    prev = root.level
    root.addHandler(h)
    root.setLevel(logging.INFO)
    try:
        assert DH.main(argv) == 0
    finally:
        DH.load_paths = orig
        root.removeHandler(h)
        root.setLevel(prev)
    return h.msgs


@pytest.fixture(scope="module")
def student_runs(tmp_path_factory):
    return tmp_path_factory.mktemp("hid0918_runs")


@pytest.fixture(scope="module")
def recipe_student(student_runs):
    _run_distill(student_runs, [*TINY, "--max-steps", "1", "--ckpt-every", "1",
                                "--run-name", "cd0918_base", *RECIPE])
    ck = Path(student_runs) / "hid" / "cd0918_base_r0" / "student_000001.pt"
    assert ck.exists()
    return ck


@pytest.mark.requires_cosmos_repo
def test_the_student_checkpoint_records_the_window_recipe(recipe_student):
    """REPRODUCTION (audit (c)): `configs.train` had no record of these at all,
    so nothing could be restored."""
    saved = torch.load(str(recipe_student), map_location="cpu",
                       weights_only=False)["configs"]["train"]
    assert (saved["grasp_frac"], saved["photo_aug"],
            saved["commit_band_weight"], saved["split"]) == (0.3, 1.0, 1.6, "all")
    assert saved["traj_target"] == "roundtrip"
    assert saved["rollout_action_weight"] == "failure_demo"


@pytest.mark.requires_cosmos_repo
def test_resume_restores_the_window_recipe_the_cli_forgot(recipe_student,
                                                          student_runs):
    msgs = _run_distill(student_runs, [*TINY, "--max-steps", "2", "--ckpt-every", "2",
                                       "--run-name", "cd0918_resume",
                                       "--resume", str(recipe_student)])
    for key in ("grasp_frac", "photo_aug", "commit_band_weight", "split"):
        assert any(key in m and "restored from the checkpoint" in m
                   for m in msgs), (key, msgs)
    assert any("grasp_frac=0.30" in m and "photo_aug=1.00" in m
               and "split=all" in m for m in msgs), msgs
    out = Path(student_runs) / "hid" / "cd0918_resume_r0" / "student_000002.pt"
    got = torch.load(str(out), map_location="cpu",
                     weights_only=False)["configs"]["train"]
    assert (got["grasp_frac"], got["photo_aug"], got["commit_band_weight"],
            got["split"]) == (0.3, 1.0, 1.6, "all")


@pytest.mark.parametrize("flag,value,label", [
    ("--grasp-frac", "0.0", "grasp_frac"),
    ("--photo-aug", "0.0", "photo_aug"),
    ("--commit-band-weight", "1.0", "commit_band_weight"),
    ("--split", "train", "split"),
    ("--traj-target", "raw_latents", "traj_target"),
    ("--rollout-action-weight", "teacher_supervised", "rollout_action_weight"),
])
@pytest.mark.requires_cosmos_repo
def test_resume_refuses_a_contradicting_recipe_flag(recipe_student, student_runs,
                                                    flag, value, label):
    from phantom.train import distill_hid as DH
    p = dataclasses.replace(load_paths(), runs_root=Path(student_runs))
    orig, DH.load_paths = DH.load_paths, lambda: p
    try:
        with pytest.raises(SystemExit, match=f"{label} drift"):
            DH.main([*TINY, "--max-steps", "2", "--run-name", "cd0918_refuse",
                     "--resume", str(recipe_student), flag, value])
    finally:
        DH.load_paths = orig


@pytest.mark.requires_cosmos_repo
def test_resume_accepts_the_recipe_passed_back_verbatim(recipe_student,
                                                        student_runs):
    msgs = _run_distill(student_runs, [*TINY, "--max-steps", "2",
                                       "--run-name", "cd0918_same",
                                       "--resume", str(recipe_student), *RECIPE])
    assert not any("drift" in m for m in msgs), msgs


@pytest.mark.requires_cosmos_repo
def test_override_recipe_accepts_the_drift_and_records_it(recipe_student,
                                                          student_runs):
    msgs = _run_distill(student_runs, [*TINY, "--max-steps", "2", "--ckpt-every", "2",
                                       "--run-name", "cd0918_override",
                                       "--resume", str(recipe_student),
                                       "--override-recipe",
                                       "--traj-target", "raw_latents",
                                       "--grasp-frac", "0.0"])
    assert any("--override-recipe" in m and "traj_target" in m for m in msgs), msgs
    out = Path(student_runs) / "hid" / "cd0918_override_r0" / "student_000002.pt"
    got = torch.load(str(out), map_location="cpu",
                     weights_only=False)["configs"]["train"]
    assert got["traj_target"] == "raw_latents" and got["grasp_frac"] == 0.0


# ===========================================================================
# (d) a relocated teacher checkpoint
# ===========================================================================

def _args(**kw):
    base = dict(teacher_ckpt="", override_recipe=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_relocated_teacher_with_the_same_hash_is_accepted(caplog):
    saved = {"teacher_ckpt": "/workspace/phantom-hid/runs/teacher/v6/t.pt",
             "teacher_ckpt_sha12": "abc123abc123"}
    cfg = HIDConfig(teacher_ckpt="/mnt/hub/teacher_v6/t.pt",
                    teacher_ckpt_sha12="abc123abc123")
    with caplog.at_level(logging.WARNING):
        out = reconcile_teacher_ckpt(saved, cfg, _args(teacher_ckpt=cfg.teacher_ckpt))
    assert out["teacher_ckpt"] == cfg.teacher_ckpt
    assert saved["teacher_ckpt"] != cfg.teacher_ckpt          # input untouched
    assert "RELOCATED" in caplog.text


def test_a_different_teacher_is_left_for_the_lock_to_refuse():
    from phantom.train.train_teacher import restore_train_config_on_resume
    saved = {"teacher_ckpt": "/workspace/runs/teacher/v6/t.pt",
             "teacher_ckpt_sha12": "abc123abc123"}
    cfg = HIDConfig(teacher_ckpt="/mnt/hub/teacher_v5/t.pt",
                    teacher_ckpt_sha12="dead00beef00")
    args = _args(teacher_ckpt=cfg.teacher_ckpt)
    out = reconcile_teacher_ckpt(saved, cfg, args)
    assert out["teacher_ckpt"] == saved["teacher_ckpt"]
    with pytest.raises(SystemExit, match="teacher_ckpt drift"):
        restore_train_config_on_resume(cfg, out, args)


def test_a_checkpoint_without_the_hash_keeps_the_path_rule():
    saved = {"teacher_ckpt": "/workspace/runs/teacher/v6/t.pt"}
    cfg = HIDConfig(teacher_ckpt="/mnt/hub/teacher_v6/t.pt",
                    teacher_ckpt_sha12="abc123abc123")
    out = reconcile_teacher_ckpt(saved, cfg, _args(teacher_ckpt=cfg.teacher_ckpt))
    assert out["teacher_ckpt"] == saved["teacher_ckpt"]


def test_resuming_without_naming_the_teacher_says_so():
    saved = {"teacher_ckpt": "/workspace/runs/teacher/v6/t.pt",
             "teacher_ckpt_sha12": "abc123abc123"}
    with pytest.raises(SystemExit, match="--resume without --teacher-ckpt"):
        reconcile_teacher_ckpt(saved, HIDConfig(), _args())


def test_file_sha12_is_the_content_identity(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"phantom" * 1000)
    b.write_bytes(b"phantom" * 1000)
    c.write_bytes(b"phantom" * 999)
    assert C.file_sha12(a) == C.file_sha12(b) != C.file_sha12(c)
    assert len(C.file_sha12(a)) == 12


@pytest.mark.requires_cosmos_repo
def test_a_relocated_teacher_resumes_end_to_end_and_a_different_one_does_not(
        student_runs):
    """The whole path, through both CLIs: tiny teacher -> student distilled
    from it -> the teacher file is MOVED -> the resume is accepted on the hash,
    while a genuinely different teacher is still refused."""
    import shutil

    from phantom.train import train_teacher as TT
    runs = Path(student_runs)
    p = dataclasses.replace(load_paths(), runs_root=runs)
    orig_tt, TT.load_paths = TT.load_paths, lambda: p
    try:
        assert TT.main([*TINY, "--acc-two-pass", "--max-steps", "1",
                        "--ckpt-every", "1", "--run-name", "cd0918_teacher"]) == 0
        assert TT.main([*TINY, "--acc-two-pass", "--max-steps", "2",
                        "--ckpt-every", "2", "--run-name", "cd0918_teacher2"]) == 0
    finally:
        TT.load_paths = orig_tt
    t1 = runs / "teacher" / "cd0918_teacher" / "teacher_000001.pt"
    t2 = runs / "teacher" / "cd0918_teacher2" / "teacher_000002.pt"

    _run_distill(runs, [*TINY, "--max-steps", "1", "--ckpt-every", "1",
                        "--run-name", "cd0918_from_teacher",
                        "--teacher-ckpt", str(t1)])
    st = runs / "hid" / "cd0918_from_teacher_r0" / "student_000001.pt"
    saved = torch.load(str(st), map_location="cpu",
                       weights_only=False)["configs"]["train"]
    assert saved["teacher_ckpt_sha12"] == C.file_sha12(t1) != ""

    moved = runs / "moved_teacher.pt"
    shutil.copy2(t1, moved)
    msgs = _run_distill(runs, [*TINY, "--max-steps", "2", "--ckpt-every", "2",
                               "--run-name", "cd0918_relocated",
                               "--teacher-ckpt", str(moved), "--resume", str(st)])
    assert any("RELOCATED" in m for m in msgs), msgs

    from phantom.train import distill_hid as DH
    orig, DH.load_paths = DH.load_paths, lambda: p
    try:
        with pytest.raises(SystemExit, match="teacher_ckpt drift"):
            DH.main([*TINY, "--max-steps", "2", "--run-name", "cd0918_wrong_teacher",
                     "--teacher-ckpt", str(t2), "--resume", str(st)])
    finally:
        DH.load_paths = orig


@pytest.mark.requires_cosmos_repo
def test_the_run_records_the_hash_of_the_teacher_it_loaded(student_runs):
    """No teacher in a tiny smoke run -> empty hash; the field exists and is
    written either way."""
    _run_distill(student_runs, [*TINY, "--max-steps", "1", "--ckpt-every", "1",
                                "--run-name", "cd0918_hash"])
    got = torch.load(str(Path(student_runs) / "hid" / "cd0918_hash_r0" /
                         "student_000001.pt"), map_location="cpu",
                     weights_only=False)["configs"]["train"]
    assert got["teacher_ckpt_sha12"] == ""


# ===========================================================================
# (e) the bench hardware config cannot load a rig checkpoint
# ===========================================================================

def test_the_bench_and_rig_hardware_configs_are_shape_incompatible():
    """CONFIRMED: configs/hardware.yaml declares wrist_ft.rate_hz 500 (bench
    e-series), configs/hardware.nuc.yaml 125 (the rig, and every v5/v6
    checkpoint). window_len = round(rate_hz * window_s) is shape-relevant, so
    125 vs 31 — the checkpoint gate REFUSES, it does not warn."""
    bench = load_hardware(str(REPO / "configs" / "hardware.yaml"))
    rig = load_hardware(str(REPO / "configs" / "hardware.nuc.yaml"))
    assert bench.wrist_ft.rate_hz == 500.0 and rig.wrist_ft.rate_hz == 125.0
    assert bench.wrist_ft.window_len == 125 and rig.wrist_ft.window_len == 31
    assert (bench.shape_relevant_fields()["wrist_window_len"]
            != rig.shape_relevant_fields()["wrist_window_len"])


def test_the_shape_refusal_names_the_knob_and_the_config():
    """The gate stays exactly as strict; the message now says which hardware
    knob drifted and which config to pass."""
    hint = C._shape_drift_hint({"wrist_window_len": (31, 125)})
    assert "wrist_ft.rate_hz" in hint and "configs/hardware.nuc.yaml" in hint


@pytest.mark.requires_cosmos_repo
def test_a_rig_checkpoint_is_refused_under_the_bench_config(recipe_student):
    payload = torch.load(str(recipe_student), map_location="cpu", weights_only=False)
    payload["configs"]["hardware_shapes"] = dict(
        payload["configs"]["hardware_shapes"], wrist_window_len=31)
    bench = load_hardware(str(REPO / "configs" / "hardware.yaml"))
    student = build_model(bench, load_paths(), student=True, tiny=True,
                          load_base=False)
    with pytest.raises(AssertionError, match="wrist_ft.rate_hz"):
        C.load_phantom_checkpoint(Path(recipe_student), student.rf, hw=bench,
                                  payload=payload)


# ===========================================================================
# (f) the layout docstring
# ===========================================================================

def test_the_sequence_docstring_matches_the_layout_it_builds():
    import phantom.model.sequence as S
    from phantom.config.backbone import BackboneConfig
    hw, bb, mc = load_hardware(None), BackboneConfig(), PhantomModelConfig()
    teacher = SequenceLayout.build(bb, mc, hw, student=False)
    student = SequenceLayout.build(bb, mc, hw, student=True)
    assert (teacher.t_total, teacher.n_tokens) == (14, 4480)
    assert (student.t_total, student.n_tokens) == (12, 3840)
    doc = S.__doc__
    assert "T=14 -> 4480 tokens" in doc and "T=12 -> 3840 tokens" in doc
    assert "T=13" not in doc and "T=11" not in doc
    for slot in teacher.slots:
        line = f"{slot.group.value:<11} [{slot.t_start}:{slot.t_start + slot.t_len})"
        assert line in doc, line
