"""terminal_eval --dump-video-error: the imagined-future columns.

Scores the denoised VIDEO_GEN x0 latents against the VAE encode of the real
future frames, plus the GT-free across-seed agreement a deploy selector would
rank on. The default path must stay byte-identical, so the summary is checked
both ways.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import terminal_eval as TE  # noqa: E402

from phantom.model.sequence import FrameGroup  # noqa: E402


class _FakeLayout:
    """Minimal stand-in: VIDEO_GEN at frames [1:4) of an 8-frame sequence."""

    def __init__(self, has_video=True):
        self._has = has_video

    def has(self, g):
        return self._has and g == FrameGroup.VIDEO_GEN

    def frame_slice(self, g):
        assert g == FrameGroup.VIDEO_GEN
        return slice(1, 4)


class _FakePrediction:
    def __init__(self, x, *, acc=None, sigma=None):
        self.x_final_B_C_T_H_W = x
        self.acc = acc
        self.governor_sigma_B_Tc = sigma


def _seq(C=2, T=8, H=2, W=2, fill=0.0):
    x = torch.zeros(1, C, T, H, W)
    x[:, :, 1:4] = fill
    return x


def test_video_gen_latents_reads_only_the_imagined_frames():
    x = _seq(fill=3.0)
    x[:, :, 0] = 99.0          # VIDEO_COND must not leak in
    x[:, :, 4:] = -99.0        # nor CONTACT/ACTION
    v = TE.video_gen_latents(x, _FakeLayout())
    assert v.shape == (2, 3, 2, 2)
    assert np.allclose(v, 3.0)


def test_video_gen_latents_refuses_a_layout_without_an_imagined_future():
    with pytest.raises(SystemExit):
        TE.video_gen_latents(_seq(), _FakeLayout(has_video=False))


def test_video_error_is_the_mse_against_the_gt_latents_with_a_per_frame_split():
    gt = np.zeros((2, 3, 2, 2))
    pred = np.zeros_like(gt)
    pred[:, 0] = 1.0           # frame 0 off by 1, frames 1-2 exact
    pred[:, 2] = 2.0
    e = TE.video_error(pred, gt)
    assert e["video_err_frames"] == [1.0, 0.0, 4.0]
    assert np.isclose(e["video_err"], 5.0 / 3)
    assert e["video_gt_energy"] == 0.0
    # a perfect imagination scores exactly zero
    assert TE.video_error(gt, gt)["video_err"] == 0.0


def test_video_error_refuses_a_shape_mismatch():
    with pytest.raises(SystemExit):
        TE.video_error(np.zeros((2, 3, 2, 2)), np.zeros((2, 2, 2, 2)))


def test_seed_agreement_ranks_the_outlier_seed_last_and_needs_no_gt():
    base = np.zeros((1, 3, 2, 2))
    vids = [base, base, base, base + 4.0]          # three agree, one does not
    a = TE.seed_agreement(vids)
    assert a[0] == a[1] == a[2] == 0.0             # they ARE the median
    assert a[3] == 16.0
    assert np.argmin(a) != 3
    # one seed: distance to its own median is 0 by construction
    assert TE.seed_agreement([base + 7.0]) == [0.0]


def test_acc_row_pulls_the_gate_and_governor_sigma_and_tolerates_absence():
    from phantom.model.acc import AccOutput
    acc = AccOutput(g=torch.tensor([0.7]), g_ant=torch.tensor([0.8]),
                    g_react=torch.tensor([0.1]), alpha=torch.tensor([0.75]),
                    p_evt=torch.tensor([[0.1, 0.9, 0.2]]))
    r = TE.acc_row(_FakePrediction(_seq(), acc=acc,
                                   sigma=torch.tensor([[1.0, 3.0]])))
    assert np.isclose(r["acc_g"], 0.7) and np.isclose(r["acc_alpha"], 0.75)
    assert np.isclose(r["acc_p_evt_max"], 0.9)
    assert np.isclose(r["governor_sigma"], 2.0)
    assert TE.acc_row(_FakePrediction(_seq())) == {}      # student w/o ACC: no keys


def test_video_rows_joins_error_agreement_and_acc_per_seed():
    gt = np.zeros((1, 3, 2, 2))
    vids = [gt + 1.0, gt + 1.0, gt + 1.0, gt + 5.0]
    rows = TE.video_rows(vids, gt, [{"acc_g": 0.5}] * 4)
    assert [r["video_err"] for r in rows] == [1.0, 1.0, 1.0, 25.0]
    assert [r["video_err_to_median"] for r in rows] == [0.0, 0.0, 0.0, 16.0]
    assert all(r["acc_g"] == 0.5 for r in rows)


def _action_rows(video=False):
    gt = np.zeros((16, 7)); gt[:, 2] = -0.010
    pr = np.zeros((16, 7)); pr[:, 2] = -0.005
    rows = []
    for seed in range(2):
        r = {"episode": "e1", "task": "waffles", "t0": 1.0, "seed": seed,
             **TE.score_window(gt, pr, 0.2)}
        if video:
            r.update({"video_err": 1.0 + seed, "video_err_to_median": 0.5 * seed,
                      "acc_g": 0.6, "governor_sigma": 2.0})
        rows.append(r)
    return rows


def test_the_video_columns_reach_the_summary_only_when_asked():
    plain = TE.summarize(_action_rows(), nfe=1)
    assert not any(k.endswith("video_err") or k == "video_err" for k in plain)
    assert "acc_g" not in plain and "governor_sigma" not in plain

    with_video = TE.summarize(_action_rows(video=True), nfe=1,
                              extra_metrics=TE.VIDEO_METRICS)
    assert np.isclose(with_video["video_err"], 1.5)
    assert np.isclose(with_video["median_video_err"], 1.5)
    assert np.isclose(with_video["seed_std_video_err"], np.std([1.0, 2.0], ddof=1))
    assert np.isclose(with_video["per_task"]["waffles"]["video_err"], 1.5)
    assert np.isclose(with_video["acc_g"], 0.6)
    # the action metrics are untouched by the extra columns
    for k in TE.METRICS:
        assert (np.isnan(plain[k]) and np.isnan(with_video[k])) or plain[k] == with_video[k]


def test_default_summary_is_unchanged_by_the_feature():
    """Same rows, no extra_metrics -> the identical JSON the tables were built
    from (the flag is only allowed to ADD keys)."""
    import json
    a = json.dumps(TE.summarize(_action_rows(), nfe=1, guidance=1.0), sort_keys=True)
    b = json.dumps(TE.summarize(_action_rows(video=True), nfe=1, guidance=1.0),
                   sort_keys=True)
    assert a == b
