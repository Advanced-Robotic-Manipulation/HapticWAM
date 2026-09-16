"""`--null-imagination`: the DEPLOY causal probe on the imagined contact package.

The pad-free student never measures contact, so the contact package it plans
against is pure imagination. Offline, `tools/terminal_eval.py --null` already
corrupts it two ways; this is the rig-side twin, and the two must be the SAME
mechanism or the deploy cells cannot be read against the offline table:

  prev_cpk      a ZERO ContactPackage is handed to ACC as the previous replan's
                package on EVERY replan (the first included, so the two-pass
                inner anticipation is bypassed as well)
  contact_zero  the CONTACT frames are cond-PINNED to the zero package at every
                denoise step (`rf.sample(pin_contact_x0=True)` ->
                `contact_pinned_layout`, the evaluator's own helper)

What the tests below lock down, in order: the zero package the deploy path
BUILDS is field-for-field the package the evaluator ZEROES (a long-vs-float
`event` alone would silently change the ACC summary from "no information" to
"contact_none, certain"); the pin hook is inert at its default and really pins
when asked; both scripts accept the flag; the policy passes the right kwargs;
the mode reaches the episode tags and pairs as an arm.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom.config.backbone import BackboneConfig
from phantom.config.model import N_EVENTS, PhantomModelConfig
from phantom.config.paths import load_paths
from phantom.data.schema import NormStats
from phantom.eval.stats import ARM_TAG_PREFIXES, episode_arm_tags
from phantom.inference.policy import (NULL_IMAGINATION_MODES, ObsSnapshot,
                                      PhantomPolicy, Plan)
from phantom.inference.remote import CONFIGURABLE, PolicyServer
from phantom.model.ace.packing import (ContactPackage, ContactPacker,
                                       zero_package)
from phantom.model.sequence import (FrameGroup, SequenceLayout,
                                    contact_pinned_layout)
from phantom_test_utils import make_small_hw


@pytest.fixture
def packer(small_hw):
    layout = SequenceLayout.build(BackboneConfig.tiny(), PhantomModelConfig(),
                                  small_hw, student=True)
    return small_hw, layout, ContactPacker(small_hw, layout)


# ---------------------------------------------------------------------------
# 1. the zero package: deploy BUILDS what offline ZEROES
# ---------------------------------------------------------------------------

def test_built_zero_package_matches_the_offline_zeroed_prediction(packer):
    """`--null prev_cpk` offline zeroes a PREDICTED package; deploy's first
    replan has none to zero and constructs one. They must agree exactly —
    including `event` being float probs, since a long zero one-hots to class 0
    inside `flatten_summary` and would assert "contact_none" instead of
    saying nothing."""
    hw, layout, cp = packer
    Tc = layout.frame_slice(FrameGroup.CONTACT)
    Tc = Tc.stop - Tc.start
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, layout.lat_c, Tc, layout.lat_h, layout.lat_w, generator=g)
    offline = zero_package(cp.unpack(x))          # the evaluator's mechanism
    built = cp.zero_package(batch=1)              # the deploy mechanism

    for f in ("event", "d_disp", "d_fz", "mask", "cop", "slip", "wrench", "wrist"):
        a, b = getattr(offline, f), getattr(built, f)
        assert a.shape == b.shape, f
        assert a.dtype == b.dtype, f
        assert torch.equal(a, b), f
    assert built.event.shape[-1] == N_EVENTS and built.event.dtype.is_floating_point
    # ... and therefore the ACC summary itself is all zeros either way
    assert torch.equal(cp.flatten_summary(offline), cp.flatten_summary(built))
    assert torch.count_nonzero(cp.flatten_summary(built)) == 0


def test_zero_package_horizon_defaults_to_the_layout(packer):
    hw, layout, cp = packer
    sl = layout.frame_slice(FrameGroup.CONTACT)
    assert cp.zero_package().horizon == sl.stop - sl.start
    assert cp.zero_package(batch=3, horizon=2).d_disp.shape[:2] == (3, 2)


# ---------------------------------------------------------------------------
# 2. the pin hook
# ---------------------------------------------------------------------------

def test_contact_pinned_layout_adds_only_the_contact_frames(packer):
    hw, layout, cp = packer
    base = layout.cond_mask_T()
    pinned = contact_pinned_layout(layout).cond_mask_T()
    sl = layout.frame_slice(FrameGroup.CONTACT)
    assert pinned[sl].all()
    off = np.ones_like(base, dtype=bool)
    off[sl] = False
    assert (pinned[off] == base[off]).all()      # nothing else moved
    # idempotent: a doubly-wrapped layout is the same object
    once = contact_pinned_layout(layout)
    assert contact_pinned_layout(once) is once


# ---------------------------------------------------------------------------
# 3. the policy: the kwargs rf.sample actually receives
# ---------------------------------------------------------------------------

def _cpk(B=1, Tc=3, F=2, cph=3, cpw=4):
    return ContactPackage(
        event=torch.zeros(B, Tc, N_EVENTS),
        d_disp=torch.zeros(B, Tc, F, 3, cph, cpw),
        d_fz=torch.zeros(B, Tc, F, cph, cpw), mask=torch.zeros(B, Tc, F, cph, cpw),
        cop=torch.zeros(B, Tc, F, 2), slip=torch.zeros(B, Tc, F),
        wrench=torch.zeros(B, Tc, F, 6), wrist=torch.zeros(B, Tc, 6))


def _pred(K, H, A):
    from phantom.model.rf import PhantomPrediction
    return PhantomPrediction(
        actions_B_H_A=torch.zeros(K, H, A), cpk=_cpk(K),
        event_logits_B_Tc_E=torch.zeros(K, 3, N_EVENTS),
        log_sigma_B_Tc_K=torch.zeros(K, 3, 1),
        governor_sigma_B_Tc=torch.zeros(K, 3),
        acc=SimpleNamespace(g=torch.zeros(K),
                            p_evt=torch.tensor([[1.0, 0, 0, 0, 0]] * K)),
        x_final_B_C_T_H_W=torch.zeros(K, 1, 1, 1, 1))


def _snap(hw):
    return ObsSnapshot(t=time.perf_counter(), rgb=np.zeros((4, 4, 3), np.uint8),
                       wrist_window=np.zeros((hw.wrist_ft.window_len, 6), np.float32),
                       ur_state=np.zeros(2 * hw.arm.dof + 14, np.float32))


def _prev_plan(hw, latency=0.96):
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    return Plan(t_created=0.0, t0_pose=np.zeros(6), actions=np.zeros((H, A)),
                action_times=np.arange(H) / hw.control.action_rate_hz,
                sigma=np.zeros(3), gate=0.0, p_evt=np.zeros(5), cpk=_cpk(),
                latency_s=latency)


def _fake_policy(hw, sample_fn, **kw):
    """PhantomPolicy with a stub rf — enough for `_batch_from_obs` + `replan`
    without the 2B model (the real one is exercised further down)."""
    layout = SequenceLayout.build(BackboneConfig.tiny(), PhantomModelConfig(),
                                  hw, student=True)
    pol = object.__new__(PhantomPolicy)
    pol.hw = hw
    pol.bb = SimpleNamespace(res_h=16, res_w=16, frames_pix=2, t_video=3,
                             temporal_comp=8, fps=8.0)
    pol.pm = SimpleNamespace(layout=SimpleNamespace(student=True))
    pol.norm = NormStats.identity()
    pol.task_text = "t"
    pol.nfe, pol.guidance = 5, 1.0
    pol.persistent_noise, pol.drop_video = False, False
    pol.parity_fixes = kw.get("parity_fixes", False)
    pol.latent_dt = pol.bb.temporal_comp / pol.bb.fps
    pol.k_seeds = kw.get("k_seeds", 1)
    pol.close_p = kw.get("close_p", 0.5)
    pol.rf = SimpleNamespace(sample=sample_fn, device=torch.device("cpu"),
                             dtype=torch.float32,
                             c_pack=ContactPacker(hw, layout))
    pol.null_imagination = kw.get("null_imagination", "none")
    return pol


def test_default_policy_is_off_and_passes_the_plain_kwargs():
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    seen = {}

    def sample(batch, **kw):
        seen.update(kw)
        return _pred(1, H, A)

    pol = _fake_policy(hw, sample)
    assert pol.null_imagination == "none"
    prev = _prev_plan(hw)
    pol.replan(_snap(hw), prev, np.zeros(6))
    assert seen["pin_contact_x0"] is False
    # the TRUE previous package still flows: nothing is nulled
    assert seen["prev_cpk"] is prev.cpk


def test_prev_cpk_mode_passes_a_zero_package_on_every_replan():
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    seen = {}

    def sample(batch, **kw):
        seen.update(kw)
        return _pred(1, H, A)

    pol = _fake_policy(hw, sample, null_imagination="prev_cpk", parity_fixes=True)
    prev = _prev_plan(hw, latency=0.96 * (8 / 8.0))
    pol.replan(_snap(hw), prev, np.zeros(6))
    pkg = seen["prev_cpk"]
    assert pkg is not None and pkg is not prev.cpk
    for f in ("event", "d_disp", "d_fz", "mask", "cop", "slip", "wrench", "wrist"):
        assert torch.count_nonzero(getattr(pkg, f)) == 0, f
    assert pkg.event.dtype.is_floating_point and pkg.event.shape[-1] == N_EVENTS
    # the parity step index is NOT touched — the probe nulls content, not timing
    assert seen["prev_cpk_step"] == 1
    assert seen["pin_contact_x0"] is False

    # ... and the FIRST replan too (no prev_plan): a zero package is still a
    # package, so rf.sample skips the two-pass inner anticipation
    pol.replan(_snap(hw), None, np.zeros(6))
    assert seen["prev_cpk"] is not None
    assert torch.count_nonzero(seen["prev_cpk"].d_fz) == 0


def test_contact_zero_mode_sets_the_pin_and_keeps_the_real_prev_package():
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    seen = {}

    def sample(batch, **kw):
        seen.update(kw)
        return _pred(1, H, A)

    pol = _fake_policy(hw, sample, null_imagination="contact_zero")
    prev = _prev_plan(hw)
    pol.replan(_snap(hw), prev, np.zeros(6))
    assert seen["pin_contact_x0"] is True
    # contact_zero pins the FRAMES; the ACC intent channel is a separate arm
    assert seen["prev_cpk"] is prev.cpk


def test_unknown_mode_is_refused():
    hw = make_small_hw()
    pol = _fake_policy(hw, lambda batch, **kw: None)
    with pytest.raises(ValueError):
        pol.null_imagination = "contact_gt"      # offline-only: needs GT contact
    for m in NULL_IMAGINATION_MODES:
        pol.null_imagination = m
    pol.null_imagination = None
    assert pol.null_imagination == "none"


# ---------------------------------------------------------------------------
# 4. the two scripts
# ---------------------------------------------------------------------------

def test_run_deploy_argparse_accepts_the_flag():
    from phantom.scripts.run_deploy import build_parser
    ap = build_parser()
    base = ["--system", "student", "--task", "waffles"]
    # unset by default, so server mode can tell "adopt the server's" from
    # "the operator explicitly asked for none"
    assert ap.parse_args(base).null_imagination is None
    for m in NULL_IMAGINATION_MODES:
        got = ap.parse_args(base + ["--null-imagination", m])
        assert got.null_imagination == m
    with pytest.raises(SystemExit):
        ap.parse_args(base + ["--null-imagination", "contact_gt"])


def test_policy_server_argparse_accepts_the_flag(monkeypatch):
    from phantom.scripts import policy_server as ps
    monkeypatch.setattr(ps, "probe", lambda port: 0)
    monkeypatch.setattr(sys, "argv",
                        ["policy_server", "--probe", "--port", "1",
                         "--null-imagination", "prev_cpk"])
    assert ps.main() == 0
    monkeypatch.setattr(sys, "argv",
                        ["policy_server", "--probe", "--null-imagination", "nope"])
    with pytest.raises(SystemExit):
        ps.main()
    src = open(ps.__file__).read()
    assert "IMAGINATION NULL" in src        # startup banner


def test_run_deploy_tags_the_mode_and_records_it():
    src = open(__import__("phantom.scripts.run_deploy", fromlist=["x"]).__file__).read()
    assert 'f"null:{null_imagination}"' in src
    assert 'deploy_overrides["null_imagination"]' in src
    # the server owns it: adopted when unset, refused when it disagrees
    assert 'policy.info.get("null_imagination"' in src


def test_server_reports_the_mode_and_it_is_not_per_launch_configurable():
    for mode in ("none", "prev_cpk"):
        policy = SimpleNamespace(policy_kind="phantom", wrench_baseline_rows=0,
                                 null_imagination=mode)
        assert PolicyServer(policy, ckpt="x.pt").status()["null_imagination"] == mode
    # a legacy policy object without the attribute reads as off
    legacy = SimpleNamespace(policy_kind="phantom", wrench_baseline_rows=0)
    assert PolicyServer(legacy, ckpt="x.pt").status()["null_imagination"] == "none"
    # never reconfigurable mid-session: the checkpoint is warm, the probe is not
    assert "null_imagination" not in CONFIGURABLE


def test_the_tag_pairs_as_an_arm():
    """`null:prev_cpk` and `null:none` are the same checkpoint on the same
    cell, so `stats pairs` can only separate them by this tag."""
    assert "null:" in ARM_TAG_PREFIXES
    meta = {"tags": ["ckpt:student_001000.pt", "nfe1", "null:prev_cpk"]}
    assert "null:prev_cpk" in episode_arm_tags(meta)
    assert "nfe1" not in episode_arm_tags(meta)


# ---------------------------------------------------------------------------
# 5. the real sampler (needs the cosmos repo importable)
# ---------------------------------------------------------------------------

def _cosmos_available() -> bool:
    try:
        from phantom.backbone import loader as bl
        bl.setup_cosmos(load_paths())
        return True
    except Exception:
        return False


cosmos = pytest.mark.skipif(not _cosmos_available(),
                            reason="cosmos repo not importable")


@pytest.fixture(scope="module")
def tiny_rf(tmp_path_factory):
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
        wrist_ft={"window_s": 0.1})
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    from phantom.train import common as C
    from phantom.train.builder import build_model
    teacher = build_model(hw, load_paths(), student=False, tiny=True, load_base=False)
    root = tmp_path_factory.mktemp("null_eps")
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, teacher.bb, NormStats.identity(), student=False)
    ds = C.WindowDataset(root, sampler, windows_per_episode=2)
    batch = C.collate_windows([ds[0], ds[1]])
    # the DEPLOY batch: no privileged contact package, exactly what
    # `_batch_from_obs` builds and what the evaluator zeroes for every
    # --null mode but contact_gt
    for k in list(batch):
        if k == "events" or k.startswith("cpk_"):
            batch[k] = torch.zeros_like(batch[k])
    return hw, teacher, batch


def _fresh(rf, seed=7):
    rf._gen = torch.Generator().manual_seed(seed)
    rf.reset_episode_noise()


@cosmos
def test_pin_contact_x0_default_changes_nothing(tiny_rf):
    """The flag off must be the shipped path: same noise draw order, same
    tensors, same actions — bit for bit."""
    hw, teacher, batch = tiny_rf
    rf = teacher.rf
    with torch.no_grad():
        _fresh(rf)
        a = rf.sample(batch, nfe=2)
        _fresh(rf)
        b = rf.sample(batch, nfe=2, pin_contact_x0=False)
    assert torch.equal(a.actions_B_H_A, b.actions_B_H_A)
    assert torch.equal(a.x_final_B_C_T_H_W, b.x_final_B_C_T_H_W)
    assert getattr(rf, "_pin_contact_x0", False) is False   # no leaked state


@cosmos
def test_pin_contact_x0_holds_the_contact_frames_at_the_zero_package(tiny_rf):
    hw, teacher, batch = tiny_rf
    rf = teacher.rf
    from phantom.model.rf import package_from_batch
    sl = rf.layout.frame_slice(FrameGroup.CONTACT)
    x0_contact = rf.c_pack.pack(package_from_batch(batch).to(rf.device)).to(rf.dtype)
    with torch.no_grad():
        _fresh(rf)
        free = rf.sample(batch, nfe=2)
        _fresh(rf)
        pinned = rf.sample(batch, nfe=2, pin_contact_x0=True)
    got = pinned.x_final_B_C_T_H_W[:, :, sl]
    assert torch.equal(got, x0_contact)                  # held every step
    assert not torch.equal(free.x_final_B_C_T_H_W[:, :, sl], x0_contact)
    # the pin reaches the ACTION head — otherwise the probe measures nothing
    assert not torch.equal(free.actions_B_H_A, pinned.actions_B_H_A)
    assert torch.isfinite(pinned.actions_B_H_A).all()
    assert getattr(rf, "_pin_contact_x0", False) is False
