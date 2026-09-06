"""Independent-clock and native-feedback checks for replay validation."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.sim.compare_replay import PHYSICS_CLOCK_TOLERANCE_S, state_metrics


def trace(t):
    t = np.asarray(t, dtype=float)
    return {"t": t, "q": np.zeros((len(t), 6)), "tcp": np.zeros((len(t), 6))}


def test_native_irregular_joint_feedback_preserves_between_video_frame_motion():
    real = trace([0, 0.1])  # A 10 Hz video grid completely misses this motion.
    real["native_arm_q_t"] = np.array([0, 0.013, 0.043, 0.080, 0.1])
    real["native_arm_q"] = np.zeros((5, 6))
    real["native_arm_q"][:, 0] = [0, 1, -0.5, 0.4, 0]
    sim = trace([0.013, 0.028, 0.043, 0.0615, 0.080])
    sim["q"][:, 0] = [1, 0.25, -0.5, -0.05, 0.4]
    metrics = state_metrics(real, sim)
    assert metrics["joint_error_rad"]["max_abs_all"] < 1e-12
    assert metrics["reference_sources"]["q"] == "native_arm_q"
    assert metrics["reference_sources"]["tcp"] == "tcp"


def test_native_tcp_slerp_crosses_pi_on_its_own_irregular_timestamps():
    real = trace([0, 0.11])
    real["tcp"][:, 5] = np.deg2rad([170, -170])
    real["native_arm_tcp_pose_t"] = np.array([0, 0.04, 0.11])
    real["native_arm_tcp_pose"] = np.zeros((3, 6))
    real["native_arm_tcp_pose"][:, 0] = [0, 0.020, 0.027]
    real["native_arm_tcp_pose"][:, 5] = np.deg2rad([170, 179, -170])
    sim = trace([0.04, 0.075, 0.11])
    sim["tcp"][:, 0] = [0.020, 0.0235, 0.027]
    sim["tcp"][:, 3:] = Rotation.from_euler(
        "z", [[179], [184.5], [190]], degrees=True
    ).as_rotvec()
    metrics = state_metrics(real, sim)
    assert metrics["tcp_translation"]["max"] < 1e-10
    assert metrics["tcp_rotation_geodesic"]["max"] < 1e-10
    assert metrics["reference_sources"]["tcp"] == "native_arm_tcp_pose"


def test_common_interval_is_intersection_of_each_native_stream_without_extrapolation():
    real = trace([0, 0.1])
    real["native_arm_q_t"] = np.array([0.01, 0.09])
    real["native_arm_q"] = np.zeros((2, 6))
    real["native_arm_tcp_pose_t"] = np.array([0.02, 0.08])
    real["native_arm_tcp_pose"] = np.zeros((2, 6))
    sim = trace([0, 0.01, 0.02, 0.05, 0.08, 0.1])
    result = state_metrics(real, sim)
    assert result["state_samples"] == 3
    assert result["excluded_sim_state_samples"] == 3
    assert result["common_state_interval_s"] == [0.02, 0.08]


@pytest.mark.parametrize(
    "offset, passed", [(0, True), (0.0001, True), (0.00051, False), (0.004, False)]
)
def test_independent_physics_clock_detects_shift_even_when_state_is_identical(
    offset, passed
):
    real = trace([0, 0.004, 0.008])
    sim = trace([0, 0.004, 0.008])
    sim["physics_t"] = sim["t"] + offset
    result = state_metrics(real, sim)["physics_clock_consistency"]
    assert result["passed"] is passed
    assert result["max_abs_error_s"] == pytest.approx(offset, abs=1e-15)
    assert result["tolerance_s"] == PHYSICS_CLOCK_TOLERANCE_S


def test_physics_clock_checks_samples_outside_reference_support():
    real = trace([0.01, 0.02])
    sim = trace([0, 0.01, 0.02, 0.03])
    sim["physics_t"] = np.array([0, 0.01, 0.02, 0.034])
    result = state_metrics(real, sim)
    assert result["state_samples"] == 2
    assert result["physics_clock_consistency"]["passed"] is False


def test_missing_optional_independent_traces_are_explicitly_unevaluated():
    result = state_metrics(trace([0, 0.01]), trace([0, 0.01]))
    assert result["physics_clock_consistency"]["status"] == "not_evaluated"
    assert result["physics_clock_consistency"]["passed"] is None
    assert result["asset_vs_nominal_fk"]["status"] == "not_evaluated"


def test_asset_vs_nominal_fk_reports_translation_and_geodesic_error_independently():
    real, sim = trace([0, 0.01]), trace([0, 0.01])
    sim["tcp_nominal_fk"] = np.zeros((2, 6))
    sim["tcp"][:, 0] = 0.012
    sim["tcp"][:, 3:] = Rotation.from_euler("x", 10, degrees=True).as_rotvec()
    result = state_metrics(real, sim)["asset_vs_nominal_fk"]
    assert result["tcp_translation"]["rmse"] == pytest.approx(12)
    assert result["tcp_rotation_geodesic"]["max"] == pytest.approx(10)
    assert result["samples"] == 2


def test_joint_metric_retains_multiturn_branch_difference():
    real, sim = trace([0, 0.01]), trace([0, 0.01])
    sim["q"][:, -1] = 2 * np.pi
    result = state_metrics(real, sim)
    assert result["joint_error_rad"]["max_abs_all"] == pytest.approx(2 * np.pi)


@pytest.mark.parametrize(
    "native",
    [
        {"native_arm_q": np.zeros((2, 6))},
        {"native_arm_q": np.zeros((2, 6)), "native_arm_q_t": np.array([0.01, 0.01])},
        {"native_arm_q": np.zeros((3, 6)), "native_arm_q_t": np.array([0, 0.01])},
    ],
)
def test_malformed_native_feedback_is_not_silently_replaced_by_video_grid(native):
    real = trace([0, 0.01])
    real.update(native)
    with pytest.raises(ValueError, match="Reference"):
        state_metrics(real, trace([0, 0.01]))


def test_cli_preserves_failed_clock_metrics_and_returns_nonzero(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    real, sim = trace([0, 0.004, 0.008]), trace([0, 0.004, 0.008])
    sim["physics_t"] = sim["t"] + 0.004
    np.savez(reference / "replay.npz", **real)
    np.savez(tmp_path / "sim_trace.npz", **sim)
    (reference / "manifest.json").write_text(
        json.dumps(
            {
                "episode": "synthetic_clock_failure",
                "split": "heldout",
                "meta_sha256": "test",
            }
        )
    )
    script = Path(__file__).resolve().parents[1] / "tools/sim/compare_replay.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--reference",
            str(reference),
            "--sim-trace",
            str(tmp_path / "sim_trace.npz"),
            "--out",
            str(tmp_path / "comparison"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    report = json.loads((tmp_path / "comparison/metrics.json").read_text())
    assert report["clock_check_failed"] is True
    assert report["state"]["physics_clock_consistency"]["status"] == "failed"


def test_tactile_previous_sample_never_uses_a_future_image():
    from tools.sim.tactile_panels import previous_sample

    ts = np.array([0.10, 0.31, 0.44])
    assert previous_sample(ts, 0.09) == (-1, None)
    index, age = previous_sample(ts, 0.30)
    assert index == 0
    assert age == pytest.approx(0.20)
    assert previous_sample(ts, 0.31) == (1, 0.0)


def test_tactile_contact_uv_moves_the_force_proxy_to_the_observed_contact():
    from tools.sim.tactile_panels import force_proxy

    image = force_proxy([0, 0, 6], uv=[0.6, -0.4], size=(101, 101))
    peak = np.array(np.nonzero(image[:, :, 0] == image[:, :, 0].max())).mean(axis=1)
    np.testing.assert_allclose(peak, [30, 80], atol=1)
    assert np.all(force_proxy([0, 0, 0]) == 70)
    with pytest.raises(ValueError, match="UV"):
        force_proxy([0, 0, 1], uv=[1.1, 0])


def _tactile_fixture(path):
    path.mkdir()
    np.savez(
        path / "tactile.npz",
        t0_master=np.asarray(100.0),
        left_image_t=np.array([0.1, 0.3]),
        right_image_t=np.array([0.2, 0.3]),
        left_image=np.stack(
            [np.full((12, 16), 20, np.uint8), np.full((12, 16), 40, np.uint8)]
        ),
        right_image=np.stack(
            [np.full((12, 16), 60, np.uint8), np.full((12, 16), 80, np.uint8)]
        ),
    )
    (path / "manifest.json").write_text(json.dumps({"streams": {}}))


def test_tactile_panels_prefer_packet_normal_force_and_mark_missing_stale_images(
    tmp_path,
):
    from tools.sim.tactile_panels import TactilePanels

    folder = tmp_path / "tactile"
    _tactile_fixture(folder)
    sim = trace([0, 1])
    sim["pad_force"] = np.full((2, 2, 3), 100.0)  # Unrelated body/table contacts.
    sim["pad_packet_force"] = np.tile([3.0, 4.0, 0.0], (2, 2, 1))
    sim["pad_packet_normal_force"] = np.full((2, 2), 2.0)
    sim["pad_contact_uv"] = np.full(
        (2, 2, 2), np.nan
    )  # Unknown is not a measured center.
    panels = TactilePanels(folder, sim)
    image = panels.render(0.19)
    assert image.shape == (300, 1280, 3)
    assert panels.rows[-1]["left_image_index"] == 0
    assert panels.rows[-1]["right_image_index"] == -1
    assert panels.rows[-1]["left_sim_force_norm_N"] == 5
    assert panels.rows[-1]["left_pressure_force_N"] == 2
    assert panels.rows[-1]["left_contact_location"] == "centered_prior"
    panels.render(0.9)
    assert panels.rows[-1]["left_image_fresh"] is False
    result = panels.save(tmp_path)
    assert result["sim_force_source"] == "pad_packet_force"
    assert result["fresh_image_frames"] == {"left": 1, "right": 0}


def test_synchronized_tactile_video_keeps_video_grid_and_native_image_times(tmp_path):
    import cv2

    from tools.sim.compare_replay import compare_videos

    tactile = tmp_path / "tactile"
    _tactile_fixture(tactile)
    video = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480)
    )
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((480, 640, 3), np.uint8))
    writer.release()
    real, sim = trace(np.arange(5) / 10), trace(np.arange(5) / 10)
    real["t0_master"] = np.asarray(100.0)
    sim["frame_t"] = sim["t"].copy()
    sim["pad_force"] = np.zeros((5, 2, 3))
    result = compare_videos(real, sim, video, video, tmp_path, 10, tactile)
    assert result["frames_compared"] == 5
    alignment = json.loads((tmp_path / "tactile_alignment.json").read_text())[
        "alignment"
    ]
    assert alignment[0]["left_image_index"] == -1
    assert (
        alignment[2]["left_image_index"] == 0
    )  # t=.2 uses image at .1, not future .3.
    for name, height in (("side_by_side.mp4", 818), ("tactile_side_by_side.mp4", 300)):
        capture = cv2.VideoCapture(str(tmp_path / name))
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
        capture.release()
