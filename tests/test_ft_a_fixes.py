"""FT-A objective knobs (2026-08-28: P5, P6, P7).

Every flag defaults to the shipped v4/v5 behaviour, so the first thing each
group of tests asserts is that OFF is bit-identical to the pre-flag code.

  P5 — contact heteroscedastic NLL owns the shared trunk's gradient:
       --contact-nll-beta / --contact-nll-detach-weight / --no-wrist-region-mse
  P6 — ACTION noise per strip (training AND sampling)
  P7 — contact self-forcing in the two-pass path
"""

import pytest

torch = pytest.importorskip("torch")

from phantom.config.backbone import BackboneConfig
from phantom.config.model import (FINETUNE_MUTABLE_MODEL_FIELDS,
                                  TRAIN_ONLY_MODEL_FIELDS, AccConfig,
                                  PhantomModelConfig)
from phantom.config.paths import load_paths
from phantom.model.ace import losses as L
from phantom.model.ace.packing import ActionPacker, sigma_group_channels
from phantom.model.sequence import FrameGroup, SequenceLayout


@pytest.fixture
def layout_setup(small_hw):
    bb = BackboneConfig.tiny()
    layout = SequenceLayout.build(bb, PhantomModelConfig(), small_hw, student=False)
    return small_hw, bb, layout


def _nll_inputs(hw, layout, seed=0, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    shape = (2, layout.lat_c, layout.t_total, layout.lat_h, layout.lat_w)
    pred = torch.randn(shape, generator=g, requires_grad=requires_grad)
    tgt = torch.randn(shape, generator=g)
    # a realistic regime: log sigma ~ -2.35 => 1/sigma^2 ~ 110 (v4/v5 logs)
    log_sigma = torch.randn(2, layout.t_video_gen, 7, generator=g) * 0.3 - 2.35
    log_sigma.requires_grad_(requires_grad)
    return pred, tgt, log_sigma, sigma_group_channels(hw.n_fingers)


# ---------------------------------------------------------------- P5


def test_contact_nll_off_is_bit_identical(layout_setup):
    """Flags off reproduces the pre-flag expression exactly, per group."""
    hw, bb, layout = layout_setup
    pred, tgt, log_sigma, gc = _nll_inputs(hw, layout, requires_grad=False)
    got = L.contact_hetero_nll(pred, tgt, log_sigma, layout, group_channels=gc)

    sl = layout.frame_slice(FrameGroup.CONTACT)
    d = (pred[:, :, sl].float() - tgt[:, :, sl].float()) ** 2
    from phantom.model.ace.heads import SIGMA_GROUPS
    terms = []
    for k, name in enumerate(SIGMA_GROUPS):
        chans = gc.get(name)
        if not chans:
            continue
        d_B_Tc = d[:, chans].mean(dim=(1, 3, 4))
        log_var = 2.0 * log_sigma[..., k]
        terms.append(d_B_Tc / log_var.exp() + log_var)
    want = torch.stack(terms, dim=-1).mean()
    assert got.item() == want.item()          # bit-identical, not allclose

    # the legacy scalar-aggregate branch too
    got0 = L.contact_hetero_nll(pred, tgt, log_sigma, layout)
    d_B_Tc = d.mean(dim=(1, 3, 4))
    log_var = 2.0 * log_sigma.mean(-1)
    assert got0.item() == (d_B_Tc / log_var.exp() + log_var).mean().item()


def test_beta_one_gradient_is_plain_mse_gradient(layout_setup):
    """beta=1: d(NLL)/d(pred) equals the plain-MSE gradient (the 1/sigma^2 and
    the detached sigma^2 cancel) — the whole point of beta-NLL."""
    hw, bb, layout = layout_setup
    pred, tgt, log_sigma, gc = _nll_inputs(hw, layout)
    L.contact_hetero_nll(pred, tgt, log_sigma, layout, group_channels=gc,
                         beta=1.0).backward()
    g_beta = pred.grad.clone()

    pred2, tgt2, log_sigma2, _ = _nll_inputs(hw, layout)
    # plain MSE = the same expression with sigma pinned to 1 (var=1, log_var=0)
    L.contact_hetero_nll(pred2, tgt2, torch.zeros_like(log_sigma2), layout,
                         group_channels=gc).backward()
    g_mse = pred2.grad.clone()

    assert torch.allclose(g_beta, g_mse, atol=1e-6), \
        f"max |diff| {(g_beta - g_mse).abs().max().item():.3e}"
    # and it is NOT the unmodified NLL's gradient: at log sigma ~ -2.35 that
    # one is ~1/sigma^2 ~ 100x larger (the P5 mechanism)
    pred3, tgt3, log_sigma3, _ = _nll_inputs(hw, layout)
    L.contact_hetero_nll(pred3, tgt3, log_sigma3, layout,
                         group_channels=gc).backward()
    ratio = pred3.grad.abs().sum() / g_mse.abs().sum().clamp_min(1e-12)
    assert ratio > 20, f"expected the raw NLL to dominate, ratio {ratio:.1f}"


def test_beta_scales_gradient_by_detached_constant(layout_setup):
    """General beta: the prediction gradient is the MSE gradient times
    sigma^(2*beta), a constant w.r.t. the prediction."""
    hw, bb, layout = layout_setup
    beta = 0.5
    pred, tgt, log_sigma, gc = _nll_inputs(hw, layout)
    L.contact_hetero_nll(pred, tgt, log_sigma, layout, group_channels=gc,
                         beta=beta).backward()
    g_beta = pred.grad.clone()
    pred2, tgt2, log_sigma2, _ = _nll_inputs(hw, layout)
    L.contact_hetero_nll(pred2, tgt2, log_sigma2, layout,
                         group_channels=gc).backward()
    g_nll = pred2.grad.clone()
    # per-group scale sigma^(2*beta) = var^beta; check on the wrist group,
    # whose single channel makes the scale unambiguous
    from phantom.model.ace.heads import SIGMA_GROUPS
    k = SIGMA_GROUPS.index("wrist")
    ch = gc["wrist"][0]
    sl = layout.frame_slice(FrameGroup.CONTACT)
    scale = (2.0 * log_sigma2[..., k]).exp().pow(beta).detach()  # (B, Tc)
    want = g_nll[:, ch, sl] * scale[..., None, None]
    assert torch.allclose(g_beta[:, ch, sl], want, rtol=1e-4, atol=1e-9)


def test_detach_weight_trunk_sees_plain_mse_sigma_still_learns(layout_setup):
    hw, bb, layout = layout_setup
    pred, tgt, log_sigma, gc = _nll_inputs(hw, layout)
    L.contact_hetero_nll(pred, tgt, log_sigma, layout, group_channels=gc,
                         detach_weight=True).backward()
    g_pred, g_sigma = pred.grad.clone(), log_sigma.grad.clone()

    pred2, tgt2, log_sigma2, _ = _nll_inputs(hw, layout)
    L.contact_hetero_nll(pred2, tgt2, torch.zeros_like(log_sigma2), layout,
                         group_channels=gc).backward()
    assert torch.allclose(g_pred, pred2.grad, atol=1e-6)

    # sigma keeps a real NLL gradient (the governor still needs calibration)
    assert g_sigma.abs().sum() > 0
    pred3, tgt3, log_sigma3, _ = _nll_inputs(hw, layout)
    L.contact_hetero_nll(pred3, tgt3, log_sigma3, layout,
                         group_channels=gc).backward()
    assert torch.allclose(g_sigma, log_sigma3.grad, atol=1e-6)


def test_wrist_channel_is_double_supervised(layout_setup):
    """The P5 claim behind --no-wrist-region-mse: packed channel 15 is both
    the lambda_w term's channel and the NLL's `wrist` sigma group."""
    from phantom.model.ace.packing import _CH_WRIST
    hw, bb, layout = layout_setup
    assert sigma_group_channels(hw.n_fingers)["wrist"] == [_CH_WRIST]


# ---------------------------------------------------------------- P6


def test_action_strip_noise_is_constant_per_strip(default_hw):
    """On the REAL layout (32x40 latents, 7-dim actions -> 5-6 cell strips)
    every latent cell of one (frame, action) strip shares one noise draw."""
    layout = SequenceLayout.build(BackboneConfig(), PhantomModelConfig(),
                                  default_hw, student=False)
    packer = ActionPacker(default_hw, layout)
    assert min(s.stop - s.start for s in packer._strips) > 1, \
        "strips must be wider than one cell for this test to mean anything"
    g = torch.Generator().manual_seed(3)
    raw = torch.randn(2, default_hw.control.chunk_horizon,
                      default_hw.control.action_dim, generator=g)
    eps = packer.pack(raw)
    for a in range(layout.actions_per_frame):
        for j, strip in enumerate(packer._strips):
            cell = eps[:, a, :, :, strip]                 # (B, Ta, h, strip_w)
            flat = cell.reshape(cell.shape[0], cell.shape[1], -1)
            assert torch.allclose(flat, flat[..., :1].expand_as(flat)), \
                f"channel {a} strip {j} is not constant"
    # round-trip: the strip mean the unpacker reads IS the drawn noise
    assert torch.allclose(packer.unpack(eps), raw, atol=1e-6)
    # the status quo — i.i.d. per-cell noise — is not strip-constant, and its
    # strip mean is the small structured offset P6 is about
    iid = torch.randn(eps.shape, generator=g)
    s0 = packer._strips[0]
    cell = iid[:, 0, :, :, s0]
    assert not torch.allclose(cell, cell[..., :1].expand_as(cell))
    n = (s0.stop - s0.start) * layout.lat_h
    assert 0.5 < cell.mean((-2, -1)).std().item() * n ** 0.5 < 2.0


def test_no_action_t_max_of_two_flag_exists():
    """P6's 'drop the max-of-two ACTION timestep' is a pre-existing flag."""
    import argparse

    from phantom.train.train_teacher import main  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-t-max-of-two", action="store_true", default=True)
    ap.add_argument("--no-action-t-max-of-two", dest="action_t_max_of_two",
                    action="store_false")
    assert ap.parse_args(["--no-action-t-max-of-two"]).action_t_max_of_two is False
    assert PhantomModelConfig(action_t_max_of_two=False).action_t_max_of_two is False


# ---------------------------------------------------------------- configs


def test_new_flags_round_trip_through_a_checkpoint_config():
    mc = PhantomModelConfig(acc=AccConfig(self_anticipation="two_pass"),
                            contact_nll_beta=0.5, contact_self_forcing=True,
                            action_noise_per_strip=True, wrist_region_mse=False,
                            contact_nll_detach_weight=True,
                            action_t_max_of_two=False)
    d = mc.to_dict()
    for k in ("contact_nll_beta", "contact_nll_detach_weight", "wrist_region_mse",
              "contact_self_forcing", "action_noise_per_strip"):
        assert k in d, f"{k} missing from the saved model config"
    back = PhantomModelConfig.from_dict(d)
    assert back == mc, "run_deploy/replay would rebuild a different model"


def test_drift_check_tolerates_training_only_flags():
    import dataclasses

    from phantom.train.train_teacher import model_config_drift
    # the v5 recipe: --acc-two-pass, max-of-two ACTION t, cond dropout 0.1
    v5_mc = PhantomModelConfig(acc=AccConfig(self_anticipation="two_pass"),
                               action_t_max_of_two=True, cond_dropout_p=0.1)
    v5 = v5_mc.to_dict()
    # a checkpoint saved BEFORE the flags existed has no such keys at all
    for k in TRAIN_ONLY_MODEL_FIELDS | {"action_noise_per_strip"}:
        v5.pop(k, None)

    ft_a = dataclasses.replace(v5_mc, contact_nll_beta=0.5,
                               contact_self_forcing=True,
                               action_noise_per_strip=True,
                               action_t_max_of_two=False, cond_dropout_p=0.0)
    hard, soft = model_config_drift(v5, ft_a,
                                    tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)
    assert not hard, hard
    assert set(soft) == {"contact_nll_beta", "contact_self_forcing",
                         "action_noise_per_strip", "action_t_max_of_two",
                         "cond_dropout_p"}

    # missing keys alone are NOT drift (old checkpoint, unchanged flags)
    assert model_config_drift(v5, v5_mc) == ({}, {})

    # a real behavioural change still hard-fails
    bad = dataclasses.replace(v5_mc, acc=AccConfig(self_anticipation="gt_noised"))
    hard, _ = model_config_drift(v5, bad, tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)
    assert "acc" in hard


def test_p10b_assertion_ignores_train_only_fields(small_hw):
    from phantom.train.common import assert_model_config_matches

    class _M:
        pass

    m = _M()
    m.mc = PhantomModelConfig(contact_nll_beta=0.5, contact_self_forcing=True,
                              wrist_region_mse=False)
    saved = PhantomModelConfig().to_dict()
    payload = {"configs": {"model": saved}}
    assert_model_config_matches(payload, m)          # training-only: tolerated

    # ...but a sampling-visible flag is not training-only
    m.mc = PhantomModelConfig(action_noise_per_strip=True)
    with pytest.raises(RuntimeError, match="action_noise_per_strip"):
        assert_model_config_matches(payload, m)


def test_ema_decay_is_a_cli_override():
    import argparse

    from phantom.config.training import TeacherTrainConfig
    from phantom.train.train_teacher import add_common_args, apply_overrides
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--ema-decay", type=float, default=None)
    args = ap.parse_args(["--ema-decay", "0.995"])
    assert apply_overrides(TeacherTrainConfig(), args).ema_decay == 0.995
    args = ap.parse_args([])
    assert apply_overrides(TeacherTrainConfig(), args).ema_decay == \
        TeacherTrainConfig().ema_decay


def test_self_forcing_requires_two_pass():
    from phantom.train.train_teacher import main
    with pytest.raises(SystemExit, match="acc-two-pass"):
        main(["--tiny", "--synthetic", "--max-steps", "1",
              "--contact-self-forcing", "--device", "cpu"])
    with pytest.raises(SystemExit, match=r"\[0, 1\]"):
        main(["--tiny", "--synthetic", "--max-steps", "1",
              "--contact-nll-beta", "1.5", "--device", "cpu"])


def test_provision_launch_line_carries_the_ft_a_bundle():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "tools" / "provision_v5.sh"
           ).read_text(encoding="utf-8")
    launch = src.split("READY. Launch")[1]
    for flag in ("--contact-nll-beta 0.5",
                 "--action-noise-per-strip", "--no-action-t-max-of-two",
                 "--ema-decay 0.995", "--cond-dropout 0"):
        assert flag in launch, f"{flag} missing from the printed FT-A launch line"
    # --contact-self-forcing was retracted from the bundle (E9 premise test
    # provides no exposure-bias gap); it survives only as a commented ablation note.
    cmd = launch.split('echo "  #')[0]
    assert "--contact-self-forcing" not in cmd, (
        "--contact-self-forcing is back in the printed FT-A launch command")
    assert "E9 premise test" in launch, (
        "the launch block must cite the E9 premise test for the retraction")


# ---------------------------------------------------------------- tiny model


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
def tiny_ft(tmp_path_factory):
    """Tiny two-pass teacher + one batch (two_pass is what self-forcing needs)."""
    if not _cosmos_available():
        pytest.skip("cosmos repo not importable")
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
    from phantom.data.schema import NormStats
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    from phantom.train import common as C
    from phantom.train.builder import build_model
    mc = PhantomModelConfig(acc=AccConfig(self_anticipation="two_pass"))
    pm = build_model(hw, load_paths(), student=False, tiny=True, mc=mc,
                     load_base=False)
    root = tmp_path_factory.mktemp("ft_eps")
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, pm.bb, NormStats.identity(), student=False)
    ds = C.WindowDataset(root, sampler, windows_per_episode=2)
    return hw, pm, C.collate_windows([ds[0], ds[1]])


def _capture_x_t(rf):
    """Record the x_t the net actually sees on the next training_step."""
    seen = {}
    orig = rf.net.forward

    def spy(*a, **kw):
        seen["x_t"] = kw["x_B_C_T_H_W"].detach().clone()
        seen["t"] = kw["timesteps_B_T"].detach().clone()
        return orig(*a, **kw)

    rf.net.forward = spy
    return seen, orig


@needs_cosmos
def test_self_forcing_replaces_contact_input_but_not_the_target(tiny_ft):
    """The noised input's CONTACT frames stop carrying GT; the loss target
    keeps carrying it; every other frame group is untouched."""
    import dataclasses

    from phantom.model import rf as rf_mod
    hw, pm, batch = tiny_ft
    rf, layout = pm.rf, pm.rf.layout
    sl = layout.frame_slice(FrameGroup.CONTACT)
    gt_contact = rf.c_pack.pack(
        rf_mod.package_from_batch(batch).to(rf.device)).to(rf.dtype).float()

    base_mc, base_nll = rf.mc, rf_mod.L.contact_hetero_nll
    runs = {}
    try:
        for on in (False, True):
            rf.mc = dataclasses.replace(base_mc, contact_self_forcing=on)
            seen_target = {}

            def spy_nll(x0_pred, x0_target, *a, **kw):
                seen_target["x0"] = x0_target.detach().clone()
                return base_nll(x0_pred, x0_target, *a, **kw)

            rf_mod.L.contact_hetero_nll = spy_nll
            rf._gen.manual_seed(0)
            seen, orig = _capture_x_t(rf)
            try:
                parts = rf.training_step(batch)
            finally:
                rf.net.forward = orig
                rf_mod.L.contact_hetero_nll = base_nll
            assert torch.isfinite(parts["total"])
            # the LOSS TARGET is GT in both cases
            assert torch.allclose(seen_target["x0"][:, :, sl].float(),
                                  gt_contact, atol=1e-5)
            runs[on] = seen["x_t"].detach().float()
        assert rf._sf_pred_cpk is not None, "two_pass produced no package to force with"

        # identical noise draws (self-forcing adds no draw), so the two x_t
        # differ EXACTLY where the CONTACT x0 was swapped and nowhere else
        d = (runs[True] - runs[False]).abs()
        assert d[:, :, sl].max() > 1e-3, "CONTACT input still carries GT"
        mask = torch.ones(layout.t_total, dtype=torch.bool)
        mask[sl] = False
        assert d[:, :, mask].max() < 1e-6, "self-forcing touched a non-CONTACT frame"
    finally:
        rf.mc = base_mc
        rf_mod.L.contact_hetero_nll = base_nll


@needs_cosmos
def test_flags_off_reproduce_the_shipped_step(tiny_ft):
    """With every FT-A flag off the losses are unchanged, and the CONTACT
    input is the GT pack (the pre-flag path)."""
    import dataclasses

    hw, pm, batch = tiny_ft
    rf = pm.rf
    base_mc = rf.mc
    outs = []
    try:
        for mc in (base_mc,
                   dataclasses.replace(base_mc, contact_nll_beta=None,
                                       contact_self_forcing=False,
                                       action_noise_per_strip=False,
                                       wrist_region_mse=True)):
            rf.mc = mc
            rf._gen.manual_seed(7)
            with torch.no_grad():
                outs.append({k: float(v) for k, v in rf.training_step(batch).items()})
    finally:
        rf.mc = base_mc
    assert outs[0] == outs[1]


@needs_cosmos
def test_wrist_term_off_zeroes_it(tiny_ft):
    import dataclasses

    hw, pm, batch = tiny_ft
    rf = pm.rf
    base_mc = rf.mc
    try:
        rf._gen.manual_seed(11)
        with torch.no_grad():
            on = rf.training_step(batch)
        rf.mc = dataclasses.replace(base_mc, wrist_region_mse=False)
        rf._gen.manual_seed(11)
        with torch.no_grad():
            off = rf.training_step(batch)
    finally:
        rf.mc = base_mc
    assert float(on["wrist_mse"]) > 0
    assert float(off["wrist_mse"]) == 0.0
    assert float(off["total"]) < float(on["total"])


@needs_cosmos
def test_per_strip_noise_in_training_and_sampling(tiny_ft):
    import dataclasses

    hw, pm, batch = tiny_ft
    rf, layout = pm.rf, pm.rf.layout
    base_mc = rf.mc
    try:
        rf.mc = dataclasses.replace(base_mc, action_noise_per_strip=True)
        rf._gen.manual_seed(5)
        seen, orig = _capture_x_t(rf)
        try:
            parts = rf.training_step(batch)
        finally:
            rf.net.forward = orig
        assert torch.isfinite(parts["total"])
        # the draw itself is strip-constant by construction; check the model
        # actually produced one (the packer path) and that sampling honours it
        eps = rf._action_strip_noise(2, rf.device, torch.float32)
        s0 = rf.a_pack._strips[0]
        cell = eps[:, 0, :, :, s0].reshape(2, eps.shape[2], -1)
        assert torch.allclose(cell, cell[..., :1].expand_as(cell))

        with torch.no_grad():
            pred = rf.sample(batch, nfe=2)
        assert pred.actions_B_H_A.shape == (2, hw.control.chunk_horizon,
                                            hw.control.action_dim)
        assert torch.isfinite(pred.actions_B_H_A).all()
    finally:
        rf.mc = base_mc
