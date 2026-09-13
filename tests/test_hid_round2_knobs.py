"""Round-2 HID knobs (review 09-05): teacher imagination at the teacher's own
NFE and a supervised student sigma head. Both default to the round-1
behaviour (nfe//2, untrained head). Real tiny models, one real batch."""
from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.utils.data import DataLoader

from phantom.config.model import AccConfig, PhantomModelConfig
from phantom.config.paths import load_paths
from phantom.config.training import HIDConfig
from phantom.data.schema import NormStats
from phantom.data.synthetic import SyntheticEpisodeGenerator
from phantom.data.windows import WindowSampler
from phantom.train import common as C
from phantom.train.builder import build_model
from phantom.train.distill_hid import distill_step
from phantom_test_utils import make_small_hw


@pytest.fixture(scope="module")
def pair_and_batch(tmp_path_factory):
    hw = make_small_hw()
    root = tmp_path_factory.mktemp("eps")
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(root, task="grasp_slip",
                                                                    duration_s=6.0)
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


def _sigma_grad(student):
    return sum(float(p.grad.abs().sum()) for p in student.rf.phantom_sigma_head.parameters()
               if p.grad is not None)


def test_round1_defaults_leave_the_sigma_head_without_gradient(pair_and_batch):
    teacher, student, batch = pair_and_batch
    student.rf.zero_grad(set_to_none=True)
    parts = distill_step(student.rf, teacher.rf, batch, HIDConfig(tiny=True, synthetic=True), "cpu")
    assert "contact_nll" not in parts
    parts["total"].backward()
    assert _sigma_grad(student) == 0.0, "round-1 recipe must be reproducible: no sigma gradient"


def test_w_sigma_trains_the_student_sigma_head_and_nothing_else(pair_and_batch):
    teacher, student, batch = pair_and_batch
    student.rf.zero_grad(set_to_none=True)
    cfg = HIDConfig(tiny=True, synthetic=True, w_sigma=1.0)
    parts = distill_step(student.rf, teacher.rf, batch, cfg, "cpu")
    assert "contact_nll" in parts and "sigma_reg" in parts
    assert torch.isfinite(parts["total"])
    # the NLL alone must reach the sigma head and NOT the trunk (detached
    # x0_pred): otherwise HID turns back into supervised training on GT
    parts["contact_nll"].backward(retain_graph=True)
    assert _sigma_grad(student) > 0.0
    trunk = [p for n, p in student.rf.named_parameters()
             if p.requires_grad and "sigma_head" not in n and p.grad is not None
             and float(p.grad.abs().sum()) > 0]
    assert trunk == [], f"NLL leaked into the trunk via {len(trunk)} params"


def test_teacher_nfe_out_of_range_is_refused(pair_and_batch):
    teacher, student, batch = pair_and_batch
    with pytest.raises(ValueError):
        distill_step(student.rf, teacher.rf, batch,
                     HIDConfig(tiny=True, synthetic=True, teacher_nfe=-3), "cpu")


def test_distill_resume_restores_the_recipe():
    """The teacher's accept/refuse rule now applies to distill_hid too."""
    import inspect
    from phantom.train import distill_hid as DH
    src = inspect.getsource(DH.main)
    assert "restore_train_config_on_resume(cfg, resume_payload" in src


def test_teacher_nfe_knob_selects_the_sampling_steps(pair_and_batch, monkeypatch):
    teacher, student, batch = pair_and_batch
    seen, depth = [], [0]
    orig = teacher.rf.sample

    def spy(b, nfe=None, **k):
        # sample() recurses for the two-pass ACC anticipation; only the
        # OUTERMOST call is the teacher imagination distill_step asked for
        if depth[0] == 0:
            seen.append(nfe)
        depth[0] += 1
        try:
            return orig(b, nfe=nfe, **k)
        finally:
            depth[0] -= 1

    monkeypatch.setattr(teacher.rf, "sample", spy)
    got = []
    for tn in (0, -1, 3):
        seen.clear()
        distill_step(student.rf, teacher.rf, batch,
                     HIDConfig(tiny=True, synthetic=True, teacher_nfe=tn), "cpu")
        # the FIRST outermost sample() of a distill_step is the imagination
        # (the ACC two-pass path issues its own sample() later in the step)
        got.append(seen[0])
    assert got == [max(1, teacher.rf.mc.nfe // 2), teacher.rf.mc.nfe, 3]


def test_distill_cli_accepts_ckpt_and_eval_cadence():
    """`--ckpt-every/--eval-every` reach HIDConfig through apply_overrides and
    are clipped to --max-steps (the v6 rental runner selects the best student
    from 500-step checkpoints; before 2026-09-08 distill_hid rejected the flags)."""
    import argparse
    from phantom.train import distill_hid as DH
    from phantom.train.train_teacher import apply_overrides
    ap = argparse.ArgumentParser()
    DH.add_common_args(ap)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    args = ap.parse_args(["--max-steps", "2000", "--ckpt-every", "500", "--eval-every", "500"])
    cfg = apply_overrides(HIDConfig(), args)
    assert cfg.ckpt_every == 500 and cfg.eval_every == 500
    args = ap.parse_args(["--max-steps", "300", "--ckpt-every", "500"])
    assert apply_overrides(HIDConfig(), args).ckpt_every == 300
    # the real parser exposes both flags
    import subprocess, sys
    out = subprocess.run([sys.executable, "-m", "phantom.train.distill_hid", "--help"],
                         capture_output=True, text=True).stdout
    assert "--ckpt-every" in out and "--eval-every" in out


def test_haptic_imagination_weights_drop_their_terms(pair_and_batch):
    """--w-traj 0 --w-event 0: the imagined-contact terms leave the total; behavior_match + grounding stay."""
    teacher, student, batch = pair_and_batch
    cfg = HIDConfig(tiny=True, synthetic=True, w_traj=0.0, w_event=0.0)
    cut = distill_step(student.rf, teacher.rf, batch, cfg, "cpu")
    zero = torch.zeros(())
    expect = (cfg.w_behavior * cut["behavior_match"]
              + cfg.w_ground * (cut["action_v_mse"] + cut.get("video_v_mse", zero) + cut["wrist_mse"]))
    if "acc_gate_bce" in cut:
        expect = expect + 0.2 * cut["acc_gate_bce"] + 0.5 * cut["acc_event_ce"]
    assert torch.isfinite(cut["total"])
    assert torch.allclose(cut["total"], expect, rtol=1e-4, atol=1e-5), (cut["total"], expect)
    assert cut["traj_distill"] > 0 and cut["event_distill"] >= 0    # still computed and logged, just unweighted


def test_haptic_weight_flags_reach_the_config():
    import argparse
    from phantom.train.train_teacher import apply_overrides
    ns = argparse.Namespace(synthetic=True, tiny=True, device="cpu", run_name="", max_steps=None,
                            batch_size=None, grad_accum=None, num_workers=None, w_traj=0.0, w_event=0.0,
                            w_behavior=None, w_sigma=None, teacher_nfe=None, lr=None, lr_new_modules=None,
                            warmup_steps=None, ckpt_every=None, eval_every=None, ema_decay=None,
                            event_band_weight=None, split=None, grasp_frac=None, photo_aug=None,
                            commit_band_weight=None, wrench_baseline_rows=None)
    cfg = apply_overrides(HIDConfig(), ns)
    assert cfg.w_traj == 0.0 and cfg.w_event == 0.0 and cfg.w_behavior == 1.0
