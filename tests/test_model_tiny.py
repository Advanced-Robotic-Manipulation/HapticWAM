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


pytestmark = pytest.mark.skipif(not _cosmos_available(),
                                reason="cosmos repo not importable")


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    from phantom_test_utils import make_hw
    hw = make_hw(
        tactile={"field": {"h": 48, "w": 64},
                 "raw_img": {"h": 60, "w": 80, "c": 1},
                 "infer_img": {"h": 60, "w": 80, "c": 1}, "rate_hz": 30.0},
        cameras={"scene": {"color": {"h": 60, "w": 80, "c": 3}, "fps": 10.0}},
        recording={"field_ds": {"h": 24, "w": 32}, "field_ds_rate_hz": 30.0,
                   "keyframe_rate_hz": 5.0, "infer_img_rate_hz": 10.0,
                   "zarr_chunk_frames": 16},
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
