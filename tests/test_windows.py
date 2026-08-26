"""WindowSampler over synthetic episodes: shapes derive from config; value
mutations (different resolutions/rates) load identically."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom.config.backbone import BackboneConfig
from phantom.data.schema import NormStats
from phantom.data.synthetic import SyntheticEpisodeGenerator
from phantom.data.windows import WindowSampler, bilinear_resize
from phantom_test_utils import make_hw


def test_bilinear_resize_shapes():
    a = np.random.default_rng(0).random((48, 64, 3)).astype(np.float32)
    assert bilinear_resize(a, (24, 32)).shape == (24, 32, 3)
    assert bilinear_resize(a, (50, 70)).shape == (50, 70, 3)
    b = np.random.default_rng(0).random((48, 64)).astype(np.float32)
    assert bilinear_resize(b, (12, 16)).shape == (12, 16)
    assert np.allclose(bilinear_resize(a, (48, 64)), a)


@pytest.fixture(scope="module")
def episode_setup(tmp_path_factory):
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
    root = tmp_path_factory.mktemp("eps")
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    return hw, root


def _check_window(hw, root, student: bool):
    bb = BackboneConfig.tiny()
    sampler = WindowSampler(hw, bb, NormStats.identity(), student=student, seed=0)
    eps = list(root.glob("ep_*"))
    w = sampler.sample(eps[0])
    Tc = bb.t_video - 1
    Fn = hw.n_fingers
    cph, cpw = hw.cpk_shape
    assert w["video"].shape == (bb.frames_pix, 3, bb.res_h, bb.res_w)
    assert w["wrist"].shape == (hw.wrist_ft.window_len, 6)
    assert w["ur_state"].shape == (hw.ur_state_dim,)
    assert w["action_chunk"].shape == (hw.control.chunk_horizon, hw.control.action_dim)
    assert w["prev_chunk"].shape == w["action_chunk"].shape
    assert w["cpk_d_disp"].shape == (Tc, Fn, 3, cph, cpw)
    assert w["cpk_d_fz"].shape == (Tc, Fn, cph, cpw)
    assert w["cpk_wrist"].shape == (Tc, 6)
    assert w["events"].shape == (Tc,)
    assert w["gate_label"].shape == ()
    if student:
        assert "fields" not in w and "gel" not in w
    else:
        assert w["fields"].shape == (Fn, hw.recording.keyframe_ds.h,
                                     hw.recording.keyframe_ds.w, 8)
        assert w["gel"].shape == (Fn, 3, bb.res_h, bb.res_w)
        assert w["contact_state"].shape == (Fn, hw.contact_state_dim)
        assert torch.isfinite(w["reactive"])
    for k, v in w.items():
        if k == "cpk_cop":
            # NaN = "no CoP" by contract (derived.py); ContactPacker keys the
            # bump amplitude on it. Everything else must be finite.
            assert torch.isfinite(v[~torch.isnan(v)]).all()
            continue
        if torch.is_tensor(v) and v.is_floating_point():
            assert torch.isfinite(v).all(), f"non-finite values in {k}"


def test_teacher_window(episode_setup):
    hw, root = episode_setup
    _check_window(hw, root, student=False)


def test_student_window(episode_setup):
    hw, root = episode_setup
    _check_window(hw, root, student=True)


def test_mutated_resolution_still_works(tmp_path):
    """The user's requirement in miniature: change resolutions/rates in the
    config, regenerate data, and the entire window path must work unchanged."""
    hw2 = make_hw(
        tactile={"field": {"h": 96, "w": 96},
                 "raw_img": {"h": 60, "w": 80, "c": 3},
                 "infer_img": {"h": 60, "w": 80, "c": 3}, "rate_hz": 20.0},
        cameras={"scene": {"color": {"h": 120, "w": 160, "c": 3}, "fps": 15.0}},
        recording={"field_ds": {"h": 48, "w": 48}, "field_ds_rate_hz": 20.0,
                   "keyframe_rate_hz": 4.0, "keyframe_ds": {"h": 48, "w": 48},
                   "infer_img_rate_hz": 10.0, "zarr_chunk_frames": 16},
        derived={"cpk_downsample": 8},
        wrist_ft={"window_s": 0.08},
        control={"chunk_horizon": 20},
    )
    SyntheticEpisodeGenerator(hw2, seed=3, rate_scale=1.0).generate(
        tmp_path, task="approach_contact", duration_s=8.0)
    _check_window(hw2, tmp_path, student=False)
