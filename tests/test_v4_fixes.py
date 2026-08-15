"""Regression tests for the 2026-08-14 v4 readiness-audit fixes.

The headline one: rope_time_mode="time_true" must actually REACH the phantom
rope. phantom_dit dispatched on `== "aligned"`, so the new mode silently fell
through to the backbone's standard sequential rope (the untested "append"
geometry) while the checkpoint recorded time_true — a provably inert fix.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from phantom_test_utils import make_hw
from phantom.config.model import PhantomModelConfig
from phantom.config.paths import load_paths
from phantom.model.sequence import FrameGroup, SequenceLayout
from phantom.train.builder import build_model


def test_time_true_positions_fractional_ordered_in_range():
    from phantom.config.backbone import BackboneConfig
    hw = make_hw()
    layout = SequenceLayout.build(BackboneConfig(), PhantomModelConfig(), hw)
    pos = layout.rope_frame_positions("time_true")
    apos = pos[layout.frame_slice(FrameGroup.ACTION)]
    assert np.all(np.diff(apos) > 0), f"action positions alias: {apos}"
    assert apos.max() <= layout.t_video_gen + 1e-9
    assert pos.dtype == np.float64
    # aligned keeps exact v3 integer values + dtype (checkpoint compat)
    assert layout.rope_frame_positions("aligned").dtype == np.int64


@pytest.fixture(scope="module")
def tiny_time_true():
    hw = make_hw()
    mc = PhantomModelConfig(student=False, rope_time_mode="time_true",
                            cond_dropout_p=1.0)
    pm = build_model(hw, load_paths(), student=False, tiny=True,
                     load_base=False, mc=mc)

    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.data.windows import WindowSampler
    from phantom.data.schema import NormStats
    from phantom.train import common as C
    root = Path(tempfile.mkdtemp(prefix="v4fx"))
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    sampler = WindowSampler(hw, pm.bb, NormStats.identity(), student=False)
    ds = C.WindowDataset(root, sampler, windows_per_episode=2)
    return pm, C.collate_windows([ds[0]])


def test_time_true_reaches_phantom_rope(tiny_time_true, monkeypatch):
    """Only 'append' may use the backbone's standard sequential rope."""
    pm, batch = tiny_time_true
    calls = {"n": 0}
    orig = type(pm.rf.net)._phantom_rope

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(type(pm.rf.net), "_phantom_rope", spy)
    pm.rf.training_step({k: (v.clone() if torch.is_tensor(v) else v)
                         for k, v in batch.items()})
    assert calls["n"] >= 1, (
        "rope_time_mode='time_true' fell through to the standard rope — the "
        "phantom_dit dispatch must route every mode except 'append' through "
        "_phantom_rope")


def test_cond_dropout_gated_on_training(tiny_time_true, monkeypatch):
    """cond_dropout_p=1.0: dropout must fire in train and NEVER in eval —
    otherwise every val_* metric is contaminated by unconditional windows."""
    pm, batch = tiny_time_true
    calls = {"n": 0}
    orig = type(pm.rf)._null_obs_batch

    def spy(b):
        calls["n"] += 1
        return orig(b)

    monkeypatch.setattr(type(pm.rf), "_null_obs_batch", staticmethod(spy))

    def clone():
        return {k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    pm.rf.train()
    pm.rf.training_step(clone())
    assert calls["n"] == 1, "p=1.0 in train mode must null the obs"
    pm.rf.eval()
    pm.rf.training_step(clone())
    pm.rf.train()
    assert calls["n"] == 1, "eval mode must NEVER apply conditioning dropout"
