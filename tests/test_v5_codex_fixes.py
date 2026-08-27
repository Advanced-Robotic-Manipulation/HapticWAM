"""Codex review 2026-08-26 fixes: CoP NaN sentinel survives to the packer,
event band is supervised, ACC probabilities are fp32, manifest_split is a
bijection check, intake skips non-finalized takes."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from phantom_test_utils import make_hw
from phantom.config.model import PhantomModelConfig
from phantom.config.paths import load_paths
from phantom.model.ace import losses as L
from phantom.model.ace.packing import CH_EVENT, _CH_COP, ContactPacker
from phantom.model.sequence import FrameGroup, SequenceLayout
from phantom.train import common as C
from phantom.train.builder import build_model


@pytest.fixture(scope="module")
def tiny():
    hw = make_hw()
    mc = PhantomModelConfig(student=False)
    pm = build_model(hw, load_paths(), student=False, tiny=True, load_base=False, mc=mc)
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    from phantom.data.schema import NormStats
    root = Path(tempfile.mkdtemp(prefix="cdx"))
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, pm.bb, NormStats.identity(), student=False)
    ds = C.WindowDataset(root, sampler, windows_per_episode=2)
    return hw, pm, ds


def test_window_cop_keeps_nan_for_no_contact(tiny):
    hw, pm, ds = tiny
    cops = torch.cat([ds[i]["cpk_cop"].reshape(-1, 2) for i in range(len(ds))])
    # synthetic grasp_slip has free-space frames before the grasp
    assert torch.isnan(cops[:, 0]).any(), "no-contact timesteps must carry NaN CoP, not (0,0)"
    finite = cops[~torch.isnan(cops[:, 0])]
    assert finite.numel() and torch.all(finite.abs() <= 1.0)


def test_packer_nan_cop_means_no_bump(tiny):
    hw, pm, ds = tiny
    batch = C.collate_windows([ds[0]])
    from phantom.model.rf import package_from_batch
    import dataclasses
    cpk = package_from_batch(batch)
    cpk_nan = dataclasses.replace(cpk, cop=torch.full_like(cpk.cop, float("nan")))
    cpk_zero = dataclasses.replace(cpk, cop=torch.zeros_like(cpk.cop))
    x_nan = pm.rf.c_pack.pack(cpk_nan)[:, _CH_COP]
    x_zero = pm.rf.c_pack.pack(cpk_zero)[:, _CH_COP]
    assert torch.allclose(x_nan, torch.full_like(x_nan, x_nan.min())), "NaN CoP must pack flat"
    assert (x_zero - x_nan).abs().max() > 0.5, "(0,0) CoP must pack a centre bump"


def test_event_band_is_supervised(tiny):
    hw, pm, ds = tiny
    batch = C.collate_windows([ds[0], ds[1]])
    layout = pm.rf.layout
    sl = layout.frame_slice(FrameGroup.CONTACT)
    x0 = torch.zeros(2, layout.lat_c, layout.t_total, layout.lat_h, layout.lat_w)
    pred = x0.clone()
    pred[:, CH_EVENT, sl] = 1.0
    assert L.event_band_mse(pred, x0, layout, CH_EVENT).item() == pytest.approx(1.0)
    assert L.event_band_mse(x0, x0, layout, CH_EVENT).item() == 0.0
    pm.rf.train()
    parts = pm.rf.training_step(batch)
    assert "contact_event_mse" in parts and torch.isfinite(parts["contact_event_mse"])
    w = pm.mc.loss
    assert torch.isfinite(parts["total"])


def test_acc_probabilities_are_fp32_under_bf16():
    from phantom.model.acc import AccGate as ACC, AccInputs
    hw = make_hw()
    mc = PhantomModelConfig(student=False)
    acc = ACC(mc.acc, hw, wrist_dim=4, z_vis_dim=8, cpk_summary_dim=6,
              student=False).to(torch.bfloat16)
    B = 3
    inp = AccInputs(wrist_feat_B_D=torch.zeros(B, 4),
                    intent_B_H_A=torch.zeros(B, hw.control.chunk_horizon, hw.control.action_dim),
                    prev_cpk_summary_B_S=torch.zeros(B, 6),
                    react_score_B=torch.full((B,), 40.0))
    out = acc(inp, torch.zeros(B, 8))
    assert out.g.dtype == torch.float32 and out.p_evt.dtype == torch.float32
    assert out.event_logits.dtype == torch.float32
    # a huge reactive score would round to exactly 1.0 in bf16; fp32 keeps room for the clamp
    assert out.g_react.dtype == torch.float32


def _mf(tmp_path, rows):
    (tmp_path / "manifests").mkdir(exist_ok=True)
    mf = tmp_path / "manifests" / "all.jsonl"
    mf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return mf


def _ep(tmp_path, rel, status="finalized"):
    d = tmp_path / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"task": "waffles", "status": status}))


def test_manifest_split_bijection_checks(tmp_path):
    root = tmp_path / "tasks"
    _ep(tmp_path, "tasks/waffles/ep_a"); _ep(tmp_path, "tasks/waffles/ep_b")
    good = [{"episode": "ep_a", "path": "tasks/waffles/ep_a", "split": "train"},
            {"episode": "ep_b", "path": "tasks/waffles/ep_b", "split": "val"}]
    _mf(tmp_path, good)
    assert [p.name for p in C.manifest_split(root, "train")] == ["ep_a"]
    _mf(tmp_path, good + [good[0]])                      # duplicate row
    with pytest.raises(SystemExit, match="duplicate"):
        C.manifest_split(root, "train")
    _mf(tmp_path, good + [{**good[0], "split": "val"}])   # leak
    with pytest.raises(SystemExit, match="both"):
        C.manifest_split(root, "train")
    _mf(tmp_path, good + [{"episode": "ep_c", "path": "tasks/waffles/ep_c", "split": "train"}])
    with pytest.raises(SystemExit, match="missing"):
        C.manifest_split(root, "train")
    _ep(tmp_path, "tasks/waffles/ep_a", status="recording")
    _mf(tmp_path, good)
    with pytest.raises(SystemExit, match="status"):
        C.manifest_split(root, "train")


def test_intake_manifest_skips_non_finalized(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import intake_recovery as IR
    tasks = tmp_path / "tasks"
    for name, st in (("ep_ok", "finalized"), ("ep_rec", "recording")):
        d = tasks / "waffles" / name
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"episode_id": name, "task": "waffles",
                                                 "success": True, "status": st, "tags": []}))
    mf = tmp_path / "manifests" / "all.jsonl"
    mf.parent.mkdir()
    IR.manifest(tasks, mf)
    rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    assert [r["episode"] for r in rows] == ["ep_ok"]


def test_event_band_weight_override(tiny):
    hw, pm, ds = tiny
    batch = C.collate_windows([ds[0], ds[1]])
    pm.rf.train()
    p0 = pm.rf.training_step(batch)
    pm.rf.event_band_weight = 0.0
    p1 = pm.rf.training_step(batch)
    del pm.rf.event_band_weight
    w = pm.mc.loss
    # same batch, same draws are not guaranteed; check the accounting identity instead
    assert torch.isfinite(p1["total"])
    expect = p1["total"] + w.event * p1["contact_event_mse"]
    assert torch.allclose(L.total_loss(p1, w), expect, atol=1e-5)
