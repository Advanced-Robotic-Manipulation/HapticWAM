"""Review panels must preserve the tactile input actually supplied at runtime."""

import json

import numpy as np
import pytest

# Panel rendering is OpenCV-only; a plain CI runner installs no cv2 (see the
# `requires_cv2` marker in tests/conftest.py — a marker cannot rescue a
# module-level import, so this module skips at collection time instead).
cv2 = pytest.importorskip(
    "cv2", reason="needs OpenCV (pip install '.[sim]'); absent on a plain CI runner"
)

from tools.sim import make_policy_video as video


class Video:
    def isOpened(self):
        return True

    def get(self, _):
        return 2

    def release(self):
        pass


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    np.savez(
        tmp_path / "sim_trace.npz",
        t=[0.0, 0.2],
        frame_t=[0.0, 0.2],
        pad_packet_normal_force=np.zeros((2, 2)),
    )
    (tmp_path / "effective_config.json").write_text("{}")
    (tmp_path / "run.json").write_text(
        json.dumps({"tactile_model": "measured_baseline_proxy"})
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Video())
    monkeypatch.setattr(
        video,
        "evaluate_policy_trace",
        lambda *_args, **_kwargs: {"control": {"stop_reason": None}},
    )
    return tmp_path


def frames():
    gel = np.zeros((2, 2, 288, 384), np.uint8)
    gel[0, 0, :, ::2] = 192
    gel[0, 1, ::2, :] = 73
    gel[1, 0] = 220
    gel[1, 1] = 140
    return gel


def test_video_uses_frozen_case_thresholds_for_placement(run_dir, monkeypatch):
    thresholds = {
        "require_support_verified_release": True,
        "support_robot_force_max_n": 0.1,
    }
    (run_dir / "case.json").write_text(json.dumps({"scoring_thresholds": thresholds}))
    captured = []

    def score(_trace, _scene, criteria, **_kwargs):
        captured.append(criteria)
        return {"control": {"stop_reason": None}}

    monkeypatch.setattr(video, "evaluate_policy_trace", score)
    panel = video.PolicyPanel(run_dir)
    assert captured == [thresholds]
    panel.close()


def test_saved_pixels_and_gel_force_override_reconstruction_and_net_force(run_dir):
    gel = frames()
    np.savez_compressed(
        run_dir / "policy_tactile.npz",
        t=[0.0, 0.125],
        gel=gel,
        gel_normal_force=[[2.0, 3.0], [4.0, 5.0]],
        pad_force=np.full((2, 2, 3), 99.0),
    )
    panel = video.PolicyPanel(run_dir)
    for index in range(2):
        for side in range(2):
            np.testing.assert_array_equal(
                panel._runtime_gray(index, side), gel[index, side]
            )
    assert panel._runtime_force(0, 0) == 2.0
    assert panel.tactile_mapping["formula"] is None
    assert "Static measured" in panel.tactile_mapping["model_description"]
    # No future tactile frame before the first causal sample.
    assert not panel._runtime_gray(-1, 0).any()
    assert panel._runtime_force(-1, 0) == 0.0
    panel.close()


def test_saved_gel_sidecar_does_not_require_obsolete_net_force(run_dir):
    np.savez_compressed(
        run_dir / "policy_tactile.npz",
        t=[0.0, 0.125],
        gel=frames(),
        gel_normal_force=np.ones((2, 2)),
    )
    panel = video.PolicyPanel(run_dir)
    np.testing.assert_array_equal(panel._runtime_gray(1, 0), 220)
    assert panel._runtime_force(1, 0) == 1.0
    panel.close()


def test_older_force_sidecar_retains_original_runtime_formula(run_dir):
    (run_dir / "run.json").write_text(json.dumps({"tactile_model": "force_proxy"}))
    net = np.zeros((2, 2, 3))
    net[1, :, 0] = 15.0
    np.savez(run_dir / "policy_tactile.npz", t=[0.0, 0.125], pad_force=net)
    panel = video.PolicyPanel(run_dir)
    np.testing.assert_array_equal(panel._runtime_gray(0, 0), 70)
    h, w = panel.field_hw
    gh, gw = panel.infer_hw
    yy, xx = np.mgrid[-1 : 1 : complex(h), -1 : 1 : complex(w)]
    expected = np.clip(
        cv2.resize(np.exp(-(xx * xx + yy * yy) / 0.18) * 0.8, (gw, gh)) * 150 + 70,
        0,
        255,
    ).astype(np.uint8)
    np.testing.assert_array_equal(panel._runtime_gray(1, 0), expected)
    assert panel._runtime_force(1, 0) == 15.0
    assert panel.tactile_mapping["formula"] is not None
    panel.close()


@pytest.mark.parametrize("bad", ["dtype", "shape", "force"])
def test_corrupt_saved_inputs_are_rejected_instead_of_silently_regenerated(
    run_dir, bad
):
    gel = frames()
    force = np.ones((2, 2))
    if bad == "dtype":
        gel = gel.astype(float)
    elif bad == "shape":
        gel = gel[:, :, :-1]
    else:
        force[0, 0] = np.nan
    np.savez_compressed(
        run_dir / "policy_tactile.npz",
        t=[0.0, 0.125],
        gel=gel,
        gel_normal_force=force,
    )
    with pytest.raises(ValueError, match="gel"):
        video.PolicyPanel(run_dir)
