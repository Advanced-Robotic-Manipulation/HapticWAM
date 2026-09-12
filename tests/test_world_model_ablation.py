"""The two world-model ablation switches (docs/ARCH_EXPLAINER_0912.md §6.2).

`--loss-video <float>`  — lambda_v override, 0 = video objective skipped while
                          the VIDEO_GEN frames stay in the layout.
`--video-attend`        — CONTACT/ACTION queries attend the VIDEO_GEN keys.

The DEFAULT path must stay byte-identical: test_default_structural_mask_values
pins the shipped mask's exact entries, and test_zero_weight_is_the_only_change
pins the loss accounting.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom.config.backbone import BackboneConfig
from phantom.config.model import (FINETUNE_MUTABLE_MODEL_FIELDS,
                                  TRAIN_ONLY_MODEL_FIELDS, LossWeights,
                                  PhantomModelConfig, flat_model_config)
from phantom.model.ace import losses as L
from phantom.model.attention_bias import structural_bias_tokens
from phantom.model.sequence import FrameGroup, SequenceLayout


def _layout(hw, **mc_kw) -> SequenceLayout:
    return SequenceLayout.build(BackboneConfig.tiny(),
                                PhantomModelConfig(**mc_kw), hw)


# ===========================================================================
# (a) the default mask is unchanged
# ===========================================================================

def test_default_structural_mask_values(small_hw):
    """Regression on the SHIPPED bias: CONTACT/ACTION x VIDEO_GEN == -1e4
    exactly, every other entry exactly 0, float32."""
    lay = _layout(small_hw)
    assert lay.video_attend is False, "video_attend must default to off"
    bias = lay.structural_attn_bias()
    assert bias.dtype == np.float32

    expect = np.zeros_like(bias)
    vid = lay.frame_slice(FrameGroup.VIDEO_GEN)
    for g in (FrameGroup.CONTACT, FrameGroup.ACTION):
        expect[lay.frame_slice(g), vid] = -1e4
    assert np.array_equal(bias, expect)
    # ...and the token-level expansion the DiT actually installs
    tok = structural_bias_tokens(lay)
    n = lay.tokens_per_frame
    assert tok.shape == (1, 1, lay.n_tokens, lay.n_tokens)
    assert tok.min().item() == -1e4 and tok.max().item() == 0.0
    assert int((tok == 0).sum().item()) == int((np.repeat(np.repeat(
        expect, n, 0), n, 1) == 0).sum())


def test_default_student_and_dropvideo_masks_unchanged(small_hw):
    for student, drop_video in ((True, False), (False, True), (True, True)):
        lay = SequenceLayout.build(BackboneConfig.tiny(), PhantomModelConfig(),
                                   small_hw, student=student,
                                   drop_video=drop_video)
        bias = lay.structural_attn_bias()
        if lay.has(FrameGroup.VIDEO_GEN):
            vid = lay.frame_slice(FrameGroup.VIDEO_GEN)
            assert (bias[lay.frame_slice(FrameGroup.ACTION), vid] == -1e4).all()
        else:
            assert (bias == 0).all()


# ===========================================================================
# (b) video_attend opens the CONTACT/ACTION -> VIDEO_GEN block
# ===========================================================================

def test_video_attend_unmasks_video_keys(small_hw):
    lay = _layout(small_hw, video_attend=True)
    assert lay.video_attend is True
    bias = lay.structural_attn_bias()
    vid = lay.frame_slice(FrameGroup.VIDEO_GEN)
    for g in (FrameGroup.CONTACT, FrameGroup.ACTION):
        assert (bias[lay.frame_slice(g), vid] == 0).all(), g
    assert (bias == 0).all(), "no other structural restriction exists"
    # the tokens the DiT installs carry no -1e4 anywhere either
    tok = structural_bias_tokens(lay)
    assert torch.count_nonzero(tok).item() == 0


def test_video_attend_keeps_the_layout_identical(small_hw):
    """Only the mask changes: same frames, same tokens, same RoPE, same
    cond mask — so a video_attend run is budget-comparable to the control."""
    base, att = _layout(small_hw), _layout(small_hw, video_attend=True)
    assert base.slots == att.slots
    assert (base.t_total, base.n_tokens) == (att.t_total, att.n_tokens)
    assert np.array_equal(base.cond_mask_T(), att.cond_mask_T())
    assert np.array_equal(base.group_of_frame(), att.group_of_frame())
    for mode in ("aligned", "append", "time_true"):
        assert np.array_equal(base.rope_frame_positions(mode),
                              att.rope_frame_positions(mode))
    assert np.array_equal(base.haptic_group_token_mask(),
                          att.haptic_group_token_mask())


def test_structural_cache_is_keyed_on_video_attend(small_hw):
    """The DiT caches the token bias per layout; two layouts that differ ONLY
    in video_attend must not share a cache entry."""
    base, att = _layout(small_hw), _layout(small_hw, video_attend=True)
    k_base = (base.slots, base.drop_video, base.video_attend, "cpu")
    k_att = (att.slots, att.drop_video, att.video_attend, "cpu")
    assert k_base != k_att


# ===========================================================================
# (c) the lambda_v override reaches the total
# ===========================================================================

def _fake_parts() -> dict:
    torch.manual_seed(0)
    keys = ("action_v_mse", "contact_nll", "contact_event_mse", "event_ce",
            "wrist_mse", "sigma_reg", "video_v_mse")
    return {k: torch.rand(()) + 0.1 for k in keys}


def test_loss_video_override_flows_into_the_total():
    parts = _fake_parts()
    w_default = LossWeights()
    assert w_default.video == 0.1, "the ablation is priced against this default"
    total_default = L.total_loss(parts, w_default)

    w_half = dataclasses.replace(w_default, video=0.05)
    assert torch.allclose(L.total_loss(parts, w_half),
                          total_default - 0.05 * parts["video_v_mse"],
                          atol=1e-6)

    # lambda_v = 0: the video term is GONE from the total, and its absence
    # equals the total of a batch that had no VIDEO_GEN frames at all
    w_zero = dataclasses.replace(w_default, video=0.0)
    no_video = {k: v for k, v in parts.items() if k != "video_v_mse"}
    assert torch.allclose(L.total_loss(parts, w_zero),
                          L.total_loss(no_video, w_default), atol=1e-6)
    assert torch.allclose(L.total_loss(parts, w_zero),
                          total_default - 0.1 * parts["video_v_mse"], atol=1e-6)


def test_zero_weight_detaches_the_video_branch():
    """0 must SKIP the term, not scale it by zero: with the term skipped the
    video prediction gets no gradient at all."""
    def _grads(w):
        parts = _fake_parts()
        # the action objective always carries gradient, so the total stays
        # differentiable either way and only the video branch is in question
        parts["action_v_mse"] = parts["action_v_mse"].clone().requires_grad_(True)
        parts["video_v_mse"] = parts["video_v_mse"].clone().requires_grad_(True)
        L.total_loss(parts, w).backward()
        return parts["action_v_mse"].grad, parts["video_v_mse"].grad

    g_act, g_vid = _grads(dataclasses.replace(LossWeights(), video=0.0))
    assert g_act is not None and g_vid is None

    g_act, g_vid = _grads(LossWeights())
    assert g_act is not None and g_vid is not None


def test_cli_builds_the_ablation_model_configs():
    """The two flags, through the actual argparse of train_teacher."""
    import argparse

    from phantom.train.train_teacher import add_common_args

    def _mc(argv):
        # mirrors main()'s construction; asserted here so a rename of either
        # flag fails a test instead of silently running the control arm
        ap = argparse.ArgumentParser()
        add_common_args(ap)
        ap.add_argument("--loss-video", type=float, default=None)
        ap.add_argument("--video-attend", action="store_true")
        a = ap.parse_args(argv)
        loss = (LossWeights() if a.loss_video is None
                else dataclasses.replace(LossWeights(), video=float(a.loss_video)))
        return PhantomModelConfig(loss=loss, video_attend=a.video_attend)

    assert _mc([]) == PhantomModelConfig()
    assert _mc(["--loss-video", "0"]).loss.video == 0.0
    assert _mc(["--video-attend"]).video_attend is True
    assert _mc(["--loss-video", "0"]).video_attend is False


# ===========================================================================
# (d) both fields round-trip through a checkpoint config
# ===========================================================================

def test_checkpoint_config_round_trip():
    mc = PhantomModelConfig(video_attend=True,
                            loss=dataclasses.replace(LossWeights(), video=0.0))
    d = mc.to_dict()
    assert d["video_attend"] is True
    assert d["loss"]["video"] == 0.0
    back = PhantomModelConfig.from_dict(d)
    assert back == mc, "eval/deploy would rebuild a different model"
    assert back.video_attend is True and back.loss.video == 0.0

    # a checkpoint saved BEFORE the flag existed keeps the default
    old = PhantomModelConfig().to_dict()
    old.pop("video_attend")
    assert PhantomModelConfig.from_dict(old).video_attend is False


def test_video_attend_is_not_a_train_only_field():
    """It changes the deployed forward, so P10B must NOT ignore it..."""
    assert "video_attend" not in TRAIN_ONLY_MODEL_FIELDS
    assert "loss.video" not in TRAIN_ONLY_MODEL_FIELDS
    # ...but a fine-tune may flip both relative to the checkpoint it inits from
    assert {"video_attend", "loss.video"} <= FINETUNE_MUTABLE_MODEL_FIELDS


def test_flat_model_config_expands_only_the_loss_weights():
    flat = flat_model_config(PhantomModelConfig().to_dict())
    assert flat["loss.video"] == 0.1 and flat["loss.action"] == 1.0
    assert "loss" not in flat
    assert isinstance(flat["acc"], dict) and isinstance(flat["lora"], dict)


def test_finetune_drift_tolerates_the_two_switches_only():
    from phantom.train.train_teacher import model_config_drift

    v6 = PhantomModelConfig(cond_dropout_p=0.1, action_t_max_of_two=True)
    saved = v6.to_dict()

    run_a = dataclasses.replace(v6, loss=dataclasses.replace(v6.loss, video=0.0))
    hard, soft = model_config_drift(saved, run_a,
                                    tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)
    assert not hard and set(soft) == {"loss.video"}

    run_b = dataclasses.replace(v6, video_attend=True)
    hard, soft = model_config_drift(saved, run_b,
                                    tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)
    assert not hard and set(soft) == {"video_attend"}

    # the control arm drifts in nothing at all
    assert model_config_drift(saved, v6) == ({}, {})

    # ...and retuning any OTHER weight is still a refusal
    bad = dataclasses.replace(v6, loss=dataclasses.replace(v6.loss, action=0.5))
    hard, _ = model_config_drift(saved, bad,
                                 tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)
    assert "loss.action" in hard
    # a --resume (no tolerate) refuses the ablation switches too
    for run in (run_a, run_b):
        hard, _ = model_config_drift(saved, run)
        assert hard


def test_p10b_load_check_follows_the_same_rule():
    from phantom.train.common import assert_model_config_matches

    saved = PhantomModelConfig().to_dict()
    payload = {"configs": {"model": saved}}

    for mc in (PhantomModelConfig(video_attend=True),
               PhantomModelConfig(loss=dataclasses.replace(LossWeights(),
                                                           video=0.0))):
        model = SimpleNamespace(mc=mc)
        with pytest.raises(RuntimeError):          # deploy / resume: strict
            assert_model_config_matches(payload, model)
        assert_model_config_matches(payload, model,     # --init-weights
                                    tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)

    # a non-tolerated weight is refused even on the fine-tune path
    model = SimpleNamespace(mc=PhantomModelConfig(
        loss=dataclasses.replace(LossWeights(), contact=2.0)))
    with pytest.raises(RuntimeError, match="loss.contact"):
        assert_model_config_matches(payload, model,
                                    tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)

    # the ablation checkpoint loaded into a model built from ITS OWN config
    # (what run_deploy/terminal_eval do) raises nothing
    att = PhantomModelConfig(video_attend=True)
    assert_model_config_matches({"configs": {"model": att.to_dict()}},
                                SimpleNamespace(mc=att))


# ===========================================================================
# (e) drop_video is an ERROR for a video_attend checkpoint
# ===========================================================================

def test_config_refuses_video_attend_with_drop_video_at_inference():
    with pytest.raises(ValueError, match="video_attend"):
        PhantomModelConfig(video_attend=True, drop_video_at_inference=True)
    with pytest.raises(ValueError):
        dataclasses.replace(PhantomModelConfig(video_attend=True),
                            drop_video_at_inference=True)
    # ... and a checkpoint that somehow carried both is refused on rebuild
    d = PhantomModelConfig(video_attend=True).to_dict()
    d["drop_video_at_inference"] = True
    with pytest.raises(ValueError):
        PhantomModelConfig.from_dict(d)


def test_sample_refuses_drop_video_for_a_video_attend_model():
    """Every inference path (policy, terminal_eval, run_deploy, smoke) funnels
    through rf.sample, so the refusal lives there. Checked on a stand-in `self`
    — the guard must fire before any tensor work."""
    from phantom.model.rf import PhantomRectifiedFlow

    att = SimpleNamespace(mc=PhantomModelConfig(video_attend=True))
    with pytest.raises(ValueError, match="drop_video is not available"):
        PhantomRectifiedFlow.sample(att, {}, nfe=1, drop_video=True)

    # the default model reaches the layout rebuild instead (and then fails on
    # the empty batch, not on the guard)
    base = SimpleNamespace(mc=PhantomModelConfig())
    with pytest.raises(Exception) as ei:
        PhantomRectifiedFlow.sample(base, {}, nfe=1, drop_video=True)
    assert "drop_video is not available" not in str(ei.value)


# ===========================================================================
# end-to-end on the tiny backbone: both arms train, and the coupled one
# refuses to be sampled without its frames
# ===========================================================================

def _cosmos_available() -> bool:
    try:
        from phantom.config.paths import load_paths
        from phantom.backbone import loader as bl
        bl.setup_cosmos(load_paths())
        return True
    except Exception:                                        # noqa: BLE001
        return False


needs_cosmos = pytest.mark.skipif(not _cosmos_available(),
                                  reason="cosmos repo not importable")


@pytest.fixture(scope="module")
def tiny_ablation(tmp_path_factory):
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
    from phantom.config.paths import load_paths
    from phantom.data.schema import NormStats
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    from phantom.train import common as C
    from phantom.train.builder import build_model
    paths = load_paths()
    models = {
        "control": build_model(hw, paths, student=False, tiny=True,
                               load_base=False, mc=PhantomModelConfig()),
        "no_video_loss": build_model(
            hw, paths, student=False, tiny=True, load_base=False,
            mc=PhantomModelConfig(loss=dataclasses.replace(LossWeights(),
                                                           video=0.0))),
        "video_attend": build_model(hw, paths, student=False, tiny=True,
                                    load_base=False,
                                    mc=PhantomModelConfig(video_attend=True)),
    }
    root = tmp_path_factory.mktemp("eps_abl")
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, models["control"].bb, NormStats.identity(),
                            student=False)
    ds = C.WindowDataset(root, sampler, windows_per_episode=2)
    return models, C.collate_windows([ds[0], ds[1]])


@needs_cosmos
def test_tiny_zero_video_weight_trains_and_keeps_the_frames(tiny_ablation):
    models, batch = tiny_ablation
    pm = models["no_video_loss"]
    assert pm.rf.layout.has(FrameGroup.VIDEO_GEN), "frames must stay"
    assert pm.rf.layout.t_total == models["control"].rf.layout.t_total
    parts = pm.rf.training_step(batch)
    assert torch.isfinite(parts["total"])
    # the metric is still REPORTED (the arm is readable) but excluded from total
    assert "video_v_mse" in parts
    assert torch.allclose(parts["total"],
                          L.total_loss({k: v for k, v in parts.items()
                                        if k not in ("total", "video_v_mse")},
                                       pm.mc.loss), atol=1e-5)


@needs_cosmos
def test_tiny_video_attend_changes_the_installed_bias_only(tiny_ablation):
    models, batch = tiny_ablation
    ctl, att = models["control"].rf, models["video_attend"].rf
    dev = torch.device("cpu")
    b_ctl = ctl.net._structural(ctl.layout, dev)
    b_att = att.net._structural(att.layout, dev)
    assert b_ctl.shape == b_att.shape
    assert b_ctl.min().item() == -1e4
    assert torch.count_nonzero(b_att).item() == 0
    parts = att.training_step(batch)
    assert torch.isfinite(parts["total"])


@needs_cosmos
def test_tiny_video_attend_refuses_drop_video_but_the_control_allows_it(tiny_ablation):
    models, batch = tiny_ablation
    with torch.no_grad():
        ok = models["control"].rf.sample(batch, nfe=2, drop_video=True)
    assert torch.isfinite(ok.actions_B_H_A).all()
    with pytest.raises(ValueError, match="drop_video is not available"):
        with torch.no_grad():
            models["video_attend"].rf.sample(batch, nfe=2, drop_video=True)
    # ...and it samples normally with its frames present
    with torch.no_grad():
        pred = models["video_attend"].rf.sample(batch, nfe=2)
    assert torch.isfinite(pred.actions_B_H_A).all()


def test_terminal_eval_guards_drop_video_before_building():
    """tools/terminal_eval.py --drop-video must fail on the CHECKPOINT's
    config, not 300 windows into the loop."""
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "tools" / "terminal_eval.py"
    text = src.read_text()
    assert re.search(r"args\.drop_video and mc\.video_attend", text)
    # and it still forwards the lever to the sampler (both call sites)
    assert text.count("drop_video=args.drop_video") == 2
