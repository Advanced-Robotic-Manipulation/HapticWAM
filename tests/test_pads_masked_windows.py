"""Pads-masked (simulated) episodes: events/gate from `contact_gt`, tactile losses off.

The exporter (tools/sim/export_expert_episode.py) tags a sim episode `pads_masked` and
writes `contact_gt`; WindowSampler must then take the ACC events and gate from that
stream instead of the idle pad rows and mark the window contact_weight 0, and the loss
helpers must drop such samples from the tactile reconstruction terms without NaNs.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import load_hardware
from phantom.config.model import EVENT_IDX
from phantom.data.episode_store import EpisodeWriter
from phantom.data.schema import (REDERIVED_TAG, STREAM_ACTIONS, STREAM_ARM_FT, STREAM_ARM_Q,
                                 STREAM_ARM_QD, STREAM_ARM_TCP_POSE, STREAM_ARM_TCP_SPEED,
                                 STREAM_CAMERA_SCENE, STREAM_GRIPPER, EpisodeMeta, NormStats,
                                 tactile_stream)
from phantom.data.windows import PADS_MASKED_TAG, STREAM_CONTACT_GT, WindowSampler
from phantom.model.ace import losses as L


def _write_episode(root: Path, hw, *, masked: bool, contact_from: float, contact_to: float,
                   duration: float = 12.0) -> Path:
    tags = [SIM := "sim", PADS_MASKED_TAG, REDERIVED_TAG] if masked else ["full"]
    meta = EpisodeMeta(task="waffles", text="waffles", operator="t", tags=tags,
                       policy="scripted_expert" if masked else "teleop", success=True)
    ep = root / ("ep_sim_waffles_0" if masked else "ep_waffles_0")
    w = EpisodeWriter(ep, hw, meta)
    t = np.arange(0.0, duration, 1 / 125.0)
    w.append(STREAM_ARM_Q, t, np.zeros((len(t), 6)))
    w.append(STREAM_ARM_QD, t, np.zeros((len(t), 6)))
    w.append(STREAM_ARM_TCP_POSE, t, np.zeros((len(t), 6)))
    w.append(STREAM_ARM_TCP_SPEED, t, np.zeros((len(t), 6)))
    w.append(STREAM_ARM_FT, t, np.zeros((len(t), 6)))
    w.append(STREAM_GRIPPER, t, np.zeros((len(t), 2), np.float32))
    ta = np.arange(0.1, duration, 0.1)
    w.append(STREAM_ACTIONS, ta, np.zeros((len(ta), 7), np.float32))
    tc = np.arange(0.0, duration, 1 / 15.0)
    w.append(STREAM_CAMERA_SCENE, tc, np.zeros((len(tc), 48, 64, 3), np.uint8))
    tf = np.arange(0.0, duration, 1 / 8.0)
    tk = np.arange(0.0, duration, 1 / 3.0)
    for s in hw.tactile.sensors:
        w.append(tactile_stream(s.name, "fields_ds"), tf, np.zeros((len(tf), 72, 96, 8), np.float16))
        w.append(tactile_stream(s.name, "infer_img"), tf, np.zeros((len(tf), 288, 384), np.uint8))
        w.append(tactile_stream(s.name, "keyframes"), tk, np.zeros((len(tk), 144, 192, 8), np.float16))
        w.append(tactile_stream(s.name, "wrench"), tf, np.zeros((len(tf), 6), np.float32))
        w.append(tactile_stream(s.name, "area"), tf, np.zeros(len(tf), np.float32))
    if masked:
        tg = np.arange(0.0, duration, 1 / 15.0)
        c = ((tg >= contact_from) & (tg < contact_to)).astype(np.float32)
        w.append(STREAM_CONTACT_GT, tg, np.column_stack([c, 5 * c, 5 * c]).astype(np.float32))
    w.finalize(success=True)
    return ep


@pytest.fixture(scope="module")
def sampler():
    hw = load_hardware("configs/hardware.nuc.yaml", quiet=True)
    return hw, WindowSampler(hw, BackboneConfig(), NormStats.identity(), wrench_baseline_rows=8)


def test_masked_episode_takes_events_and_gate_from_contact_gt(tmp_path, sampler):
    hw, ws = sampler
    ep = _write_episode(tmp_path, hw, masked=True, contact_from=6.0, contact_to=11.5)
    lo, hi = ws.valid_range(ep)
    assert lo < 5.9 < hi
    # window anchored just before the onset: gate positive, an onset on the latent grid
    w = ws.sample(ep, t0=5.9)
    assert w["contact_weight"] == 0.0
    assert float(w["gate_label"]) == 1.0
    ev = w["events"].numpy().tolist()
    assert EVENT_IDX["onset"] in ev and EVENT_IDX["slip"] not in ev
    # anchored inside the held contact: hold everywhere, no onset
    w = ws.sample(ep, t0=7.0)
    assert all(e == EVENT_IDX["hold"] for e in w["events"].numpy().tolist())
    # anchored long before contact: nothing, gate off
    w = ws.sample(ep, t0=2.0)
    assert float(w["gate_label"]) == 0.0
    assert all(e == EVENT_IDX["none"] for e in w["events"].numpy().tolist())
    assert w["action_weight"] == 1.0


def test_unmasked_episode_keeps_tactile_path(tmp_path, sampler):
    hw, ws = sampler
    ep = _write_episode(tmp_path, hw, masked=False, contact_from=0, contact_to=0)
    lo, hi = ws.valid_range(ep)
    w = ws.sample(ep, t0=(lo + hi) / 2)
    assert w["contact_weight"] == 1.0
    assert float(w["gate_label"]) == 0.0           # idle zero pads: no contact anywhere


def test_masked_episode_without_contact_gt_is_refused(tmp_path, sampler):
    hw, ws = sampler
    ep = _write_episode(tmp_path, hw, masked=True, contact_from=1, contact_to=2)
    (ep / f"{STREAM_CONTACT_GT}.zarr").rename(ep / "gone.zarr")
    with pytest.raises(KeyError):
        ws.sample(ep, t0=5.0)


def test_weighted_mean_drops_zero_weight_samples_without_nan():
    per_sample = torch.tensor([[1.0, 3.0], [100.0, 100.0]])
    assert float(L._weighted_mean(per_sample, torch.tensor([1.0, 0.0]))) == pytest.approx(2.0)
    assert float(L._weighted_mean(per_sample, None)) == pytest.approx(51.0)
    assert torch.isfinite(L._weighted_mean(per_sample, torch.tensor([0.0, 0.0])))
    assert float(L._weighted_mean(per_sample, torch.tensor([0.0, 0.0]))) == 0.0
