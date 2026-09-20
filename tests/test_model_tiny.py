"""Tiny-backbone model tests (need the cosmos repo importable through the
compat shims; skipped cleanly if it is not)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom.config.paths import load_paths


def _cosmos_available() -> bool:
    try:
        from phantom.backbone import loader as bl
        bl.setup_cosmos(load_paths())
        return True
    except Exception:
        return False


pytestmark = [
    # the model-build + train-step + checkpoint core of CI's model-smoke job
    pytest.mark.model_smoke,
    pytest.mark.skipif(not _cosmos_available(),
                       reason="cosmos repo not importable"),
]


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    from phantom_test_utils import make_hw
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
    from phantom.train.builder import build_model
    paths = load_paths()
    teacher = build_model(hw, paths, student=False, tiny=True, load_base=False)
    student = build_model(hw, paths, student=True, tiny=True, load_base=False)
    # data
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    from phantom.data.schema import NormStats
    from phantom.train import common as C
    root = tmp_path_factory.mktemp("eps")
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, teacher.bb, NormStats.identity(), student=False)
    ds = C.WindowDataset(root, sampler, windows_per_episode=2)
    batch = C.collate_windows([ds[0], ds[1]])
    return hw, teacher, student, batch


def test_teacher_training_step(tiny_models):
    hw, teacher, student, batch = tiny_models
    parts = teacher.rf.training_step(batch)
    assert torch.isfinite(parts["total"])
    parts["total"].backward()
    got_grad = [n for n, p in teacher.rf.named_parameters()
                if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert any("phantom_acc" in n for n in got_grad)
    assert any("phantom_tactile_enc" in n for n in got_grad)
    assert any("lora_" in n for n in got_grad)
    teacher.rf.zero_grad(set_to_none=True)


def test_sample_shapes(tiny_models):
    hw, teacher, student, batch = tiny_models
    with torch.no_grad():
        pred = teacher.rf.sample(batch, nfe=2)
    assert pred.actions_B_H_A.shape == (2, hw.control.chunk_horizon,
                                        hw.control.action_dim)
    assert pred.governor_sigma_B_Tc.shape[1] == teacher.bb.t_video - 1
    assert pred.acc is not None and pred.acc.g.shape == (2,)
    assert torch.isfinite(pred.actions_B_H_A).all()


def test_k_seed_sample_batches_the_denoise(tiny_models):
    """P6 K-seed sampling: K chunks per call in ONE denoise, every seed on
    identical conditioning (the only difference is the noise draw), rows laid
    out as b*K + j."""
    hw, teacher, student, batch = tiny_models
    K, B = 3, batch["prev_chunk"].shape[0]
    with torch.no_grad():
        pred = teacher.rf.sample(batch, nfe=2, k_seeds=K)
    assert pred.actions_B_H_A.shape == (B * K, hw.control.chunk_horizon,
                                        hw.control.action_dim)
    assert pred.governor_sigma_B_Tc.shape[0] == B * K
    assert pred.acc.g.shape == (B * K,) and pred.cpk.batch == B * K
    assert torch.isfinite(pred.actions_B_H_A).all()
    # different noise per row => different chunks (a broadcast bug would make
    # them identical, and the selection rule would then be a no-op)
    assert not torch.allclose(pred.actions_B_H_A[0], pred.actions_B_H_A[1])


def test_prev_cpk_step_selects_a_later_package_row(tiny_models):
    """The parity prev_cpk alignment must actually reach flatten_summary."""
    hw, teacher, student, batch = tiny_models
    with torch.no_grad():
        base = teacher.rf.sample(batch, nfe=2)
        a = teacher.rf.sample(batch, nfe=2, prev_cpk=base.cpk, prev_cpk_step=0)
        b = teacher.rf.sample(batch, nfe=2, prev_cpk=base.cpk, prev_cpk_step=1)
    assert torch.isfinite(a.actions_B_H_A).all()
    assert torch.isfinite(b.actions_B_H_A).all()
    # step is clamped to the package horizon, never an index error
    with torch.no_grad():
        teacher.rf.sample(batch, nfe=2, prev_cpk=base.cpk, prev_cpk_step=999)


def _spy_build_x0(rf, monkeypatch):
    """Record every build_x0 call: whether it was the CFG null build, whether
    the ACC two-pass anticipation sample was suppressed for it, and the
    AccInputs it produced (mutated in place afterwards by align_guidance_acc)."""
    seen: list[dict] = []
    real = rf.build_x0

    def spy(batch, layout=None, *, encode_gen=True, null_video_cond=False):
        rec = {"null": null_video_cond,
               "suppressed": getattr(rf, "_in_anticipation_pass", False)}
        seen.append(rec)
        out = real(batch, layout, encode_gen=encode_gen,
                   null_video_cond=null_video_cond)
        rec["acc"] = out[2]
        return out

    monkeypatch.setattr(rf, "build_x0", spy)
    return seen


def test_guidance_null_branch_shares_the_conditional_prev_cpk(tiny_models, monkeypatch):
    """§1.11: the CFG null branch used to keep its OWN predicted prev_cpk
    summary while the conditional branch was overwritten with the true one, so
    v_obs - v_null carried an intent perturbation as well as an observation
    one and a guidance sweep measured two things at once."""
    hw, teacher, student, batch = tiny_models
    rf = teacher.rf
    with torch.no_grad():
        base = rf.sample(batch, nfe=2)
    seen = _spy_build_x0(rf, monkeypatch)
    with torch.no_grad():
        pred = rf.sample(batch, nfe=2, prev_cpk=base.cpk, prev_cpk_step=1,
                         guidance_scale=1.5)
    assert torch.isfinite(pred.actions_B_H_A).all()

    assert [r["null"] for r in seen] == [False, True]      # conditional, then null
    cond, null = seen[0]["acc"], seen[1]["acc"]
    want = rf.c_pack.flatten_summary(base.cpk.to(rf.device), step=1).to(rf.dtype)
    assert torch.equal(cond.prev_cpk_summary_B_S, want)
    assert torch.equal(null.prev_cpk_summary_B_S, want)    # THE fix
    # ...while the OBSERVATION inputs stay nulled: guidance still guides
    assert not torch.equal(null.wrist_feat_B_D, cond.wrist_feat_B_D)
    assert torch.count_nonzero(null.react_score_B) == 0
    assert torch.count_nonzero(cond.react_score_B) > 0


def test_guidance_null_branch_skips_the_anticipation_sample(tiny_models, monkeypatch):
    """The summary it would predict is discarded, so the inner two-pass sample
    (2 NFE of the full net per replan) must not run for it either."""
    hw, teacher, student, batch = tiny_models
    rf = teacher.rf
    with torch.no_grad():
        base = rf.sample(batch, nfe=2)
    seen = _spy_build_x0(rf, monkeypatch)
    with torch.no_grad():
        rf.sample(batch, nfe=2, prev_cpk=base.cpk, guidance_scale=1.5)
    assert all(r["suppressed"] for r in seen)
    # and the flag is restored, not left latched on
    assert not getattr(rf, "_in_anticipation_pass", False)


def test_guidance_without_prev_cpk_still_aligns_the_branches(tiny_models, monkeypatch):
    """Offline (terminal_eval --guidance) there is no true package: both
    branches then share the CONDITIONAL branch's own anticipation."""
    hw, teacher, student, batch = tiny_models
    rf = teacher.rf
    seen = _spy_build_x0(rf, monkeypatch)
    with torch.no_grad():
        pred = rf.sample(batch, nfe=2, guidance_scale=2.0)
    assert torch.isfinite(pred.actions_B_H_A).all()
    cond, null = seen[0]["acc"], seen[1]["acc"]
    assert torch.equal(null.prev_cpk_summary_B_S, cond.prev_cpk_summary_B_S)


def test_drop_video_sample(tiny_models):
    hw, teacher, student, batch = tiny_models
    with torch.no_grad():
        pred = teacher.rf.sample(batch, nfe=2, drop_video=True)
    assert torch.isfinite(pred.actions_B_H_A).all()


def test_student_step_and_hid(tiny_models):
    hw, teacher, student, batch = tiny_models
    parts = student.rf.training_step(batch)
    assert torch.isfinite(parts["total"])
    from phantom.config.training import HIDConfig
    from phantom.train.distill_hid import distill_step
    teacher.rf.eval()
    parts = distill_step(student.rf, teacher.rf, batch,
                         HIDConfig(tiny=True, synthetic=True), "cpu")
    assert torch.isfinite(parts["total"])
    parts["total"].backward()
    student.rf.zero_grad(set_to_none=True)


def test_zero_bias_hook_is_noop(tiny_models):
    """With lambdas=0 and no acc inputs, the wrapped attn op must equal the
    unwrapped one -> pure-video parity is preserved at init."""
    hw, teacher, student, batch = tiny_models
    net = teacher.rf.net
    assert float(net.phantom_bias_lambdas.abs().sum()) == 0.0
    assert float(net.phantom_frame_type_emb.weight.abs().sum()) == 0.0


def test_checkpoint_roundtrip_and_student_init(tiny_models, tmp_path):
    hw, teacher, student, batch = tiny_models
    from phantom.config.training import TeacherTrainConfig
    from phantom.train import common as C
    p = tmp_path / "t.pt"
    C.save_phantom_checkpoint(p, teacher.rf, hw=hw, bb=teacher.bb, mc=teacher.mc,
                              train_cfg=TeacherTrainConfig(), step=5)
    payload = C.load_phantom_checkpoint(p, teacher.rf, hw=hw)
    assert payload["step"] == 5
    # student init drops teacher-only keys, fills the shared ones
    C.load_phantom_checkpoint(p, student.rf, hw=hw, allow_missing=True)
    # shape-mismatch guard
    from phantom_test_utils import make_hw
    hw_bad = make_hw(control={"chunk_horizon": 24})
    with pytest.raises(AssertionError, match="shape-relevant"):
        C.load_phantom_checkpoint(p, teacher.rf, hw=hw_bad)
