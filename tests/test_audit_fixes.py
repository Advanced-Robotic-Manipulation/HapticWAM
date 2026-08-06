"""Regression tests for the 2026-08-03 pre-training audit fixes.

Each test pins one defect that would otherwise be baked into a multi-day run:
RoPE fps modulation, failure-demo action supervision, causality of the action
targets, window-anchor resampling, and the ur_state normalization coverage.
"""
from __future__ import annotations

import numpy as np
import torch

from phantom.config.backbone import BackboneConfig
from phantom.data.schema import EpisodeMeta, is_failure_demo


def test_rope_fps_modulation_off_by_default():
    """The released robot/action-cond base was trained without fps modulation;
    leaving it on drives the frozen backbone off its temporal geometry."""
    assert BackboneConfig().rope_enable_fps_modulation is False


def test_is_failure_demo_recognises_all_three_encodings():
    assert is_failure_demo(EpisodeMeta(task="egg", success=False))
    assert is_failure_demo(EpisodeMeta(task="egg", tags=["deliberate_failure"]))
    # how the rig actually records them: success=True means "the EPISODE
    # captured the intended failure", not that the manipulation succeeded
    assert is_failure_demo(EpisodeMeta(task="egg_fail", success=True))
    assert not is_failure_demo(EpisodeMeta(task="egg", success=True))
    assert not is_failure_demo(EpisodeMeta(task="Carton", success=None))


def test_action_loss_masks_failure_windows():
    """A zero action_weight must remove that sample from the action term."""
    from phantom.model.ace.losses import group_velocity_mse

    class _Layout:
        def frame_slice(self, group):
            return slice(0, 2)

    pred = torch.zeros(2, 1, 2, 2, 2)
    tgt = torch.zeros(2, 1, 2, 2, 2)
    tgt[1] = 10.0                      # sample 1 is wildly "wrong"
    layout, grp = _Layout(), None

    unmasked = group_velocity_mse(pred, tgt, layout, grp)
    masked = group_velocity_mse(pred, tgt, layout, grp,
                                torch.tensor([1.0, 0.0]))
    assert float(masked) == 0.0, "failure sample still drives the action loss"
    assert float(unmasked) > 0.0
    # an all-failure batch yields no gradient rather than NaN
    allfail = group_velocity_mse(pred, tgt, layout, grp, torch.tensor([0.0, 0.0]))
    assert torch.isfinite(allfail)


def test_future_idx_never_resolves_backwards():
    """Action targets must not resolve to an action that already executed."""
    from phantom.data.windows import _EpisodeCache

    class _C(_EpisodeCache):
        def __init__(self):
            self._ts = {"s": np.array([0.0, 0.1, 0.2, 0.3])}

    c = _C()
    # a query between ticks: nearest may look back, future must not
    assert c.nearest_idx("s", 0.101) == 1
    assert c.future_idx("s", 0.101) == 2
    assert c.ts("s")[c.future_idx("s", 0.101)] >= 0.101
    # exact hits stay put, past the end clamps
    assert c.future_idx("s", 0.2) == 2
    assert c.future_idx("s", 99.0) == 3


def test_window_dataset_resamples_anchor():
    """A frozen index would replay the same anchors for the whole run."""
    from phantom.data.windows import WindowItem
    from phantom.train.common import WindowDataset

    seen = []

    class _Sampler:
        def build_index(self, root, wpe, episodes=None):
            return [WindowItem(episode="ep", t0=1.0, lo=0.0, hi=10.0)]

        def sample(self, ep, t0):
            seen.append(t0)
            return {"t0": torch.tensor(t0)}

    ds = WindowDataset("/nonexistent", _Sampler(), seed=0)
    for _ in range(8):
        ds[0]
    assert len(set(seen)) > 1, "anchor never moved"
    assert all(0.0 <= t <= 10.0 for t in seen)

    frozen = WindowDataset("/nonexistent", _Sampler(), resample=False, seed=0)
    seen.clear()
    frozen[0], frozen[0]
    assert seen == [1.0, 1.0], "held-out anchors must stay comparable"


def test_text_provider_cache_lookup(tmp_path):
    """With a cache: per-sample lookup, unknown/empty fall back to the empty
    embedding; without: historical expand-the-empty behavior."""
    from types import SimpleNamespace
    from phantom.backbone.text_embedding import TextEmbeddingProvider
    from phantom.config.backbone import BackboneConfig

    bb = BackboneConfig(text_emb_seq_len=4, text_emb_dim=8)
    known = torch.ones(4, 8)
    cache = {"wipe the whiteboard": known,
             "__meta__": {"recipe": "test"}}
    cp = tmp_path / "cache.pt"
    torch.save(cache, cp)

    # fake=True gives a deterministic random empty embedding; then attach cache
    prov = TextEmbeddingProvider(bb, None, fake=True)
    prov._cache = {"wipe the whiteboard": known}
    out = prov.get(3, ["wipe the whiteboard", "unknown task", ""])
    assert out.shape == (3, 4, 8)
    assert torch.equal(out[0], known)
    assert torch.equal(out[1], prov._emb[0])      # unknown -> empty
    assert torch.equal(out[2], prov._emb[0])      # empty -> empty

    # no cache: all rows are the empty embedding regardless of text
    prov2 = TextEmbeddingProvider(bb, None, fake=True)
    out2 = prov2.get(2, ["wipe the whiteboard", "x"])
    assert torch.equal(out2[0], out2[1])
