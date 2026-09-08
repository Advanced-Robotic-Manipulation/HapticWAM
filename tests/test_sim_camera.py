"""Projection checks for the SDK's square-pixel and centered-K pitfalls."""

import copy

import numpy as np
import pytest

from phantom.sim.camera import (
    camera_projection,
    configure_camera_intrinsics,
    validate_camera_intrinsics,
)


LEGACY = {
    "resolution": [640, 480], "fx": 742.893223660066, "fy": 742.893223660066,
    "cx": 320.0, "cy": 240.0,
}
EXPLICIT = {
    "resolution": [640, 480], "fx": 617.2, "fy": 611.8, "cx": 318.6, "cy": 242.1,
    "camera_model": "opencv_pinhole", "distortion_model": "opencv_brown_conrady",
    "distortion_coefficients": [-0.01, 0.005, 0.0001, -0.0002, 0.002],
}


class SdkCameraDouble:
    """Model the real SDK's coupling and deliberately misleading K getter."""

    def __init__(self):
        self.resolution = (640, 480)
        self.calls = []
        self.properties = None

    def get_resolution(self):
        return self.resolution

    def set_focal_length(self, value):
        self.calls.append(("focal", value))
        self.focal = value

    def set_horizontal_aperture(self, value, maintain_square_pixels=True):
        self.calls.append(("horizontal", value, maintain_square_pixels))
        self.horizontal = value
        if maintain_square_pixels:
            self.vertical = value * self.resolution[1] / self.resolution[0]

    def set_vertical_aperture(self, value, maintain_square_pixels=True):
        self.calls.append(("vertical", value, maintain_square_pixels))
        self.vertical = value
        if maintain_square_pixels:
            self.horizontal = value * self.resolution[0] / self.resolution[1]

    def set_opencv_pinhole_properties(self, **properties):
        self.properties = properties

    def get_opencv_pinhole_properties(self):
        p = self.properties
        return None if p is None else (p["cx"], p["cy"], p["fx"], p["fy"], p["pinhole"])

    def get_intrinsics_matrix(self):
        w, h = self.resolution
        return np.array([
            [w * self.focal / self.horizontal, 0, w / 2],
            [0, h * self.focal / self.vertical, h / 2],
            [0, 0, 1],
        ])


def test_legacy_calls_and_fov_are_preserved():
    camera = SdkCameraDouble()
    configure_camera_intrinsics(camera, LEGACY)
    assert camera.calls == [
        ("focal", 24.0),
        ("horizontal", 640 * 24 / LEGACY["fx"], True),
        ("vertical", 480 * 24 / LEGACY["fy"], True),
    ]
    assert camera.properties is None
    report = validate_camera_intrinsics(camera, LEGACY)
    assert report["undistorted_pinhole_fov_degrees"] == pytest.approx({
        "horizontal": 46.6077588692, "vertical": 35.8072668703,
    })


def test_explicit_projection_keeps_both_focal_lengths_and_off_center_principal_point():
    camera = SdkCameraDouble()
    configure_camera_intrinsics(camera, EXPLICIT)
    report = validate_camera_intrinsics(camera, EXPLICIT)
    assert np.asarray(report["intrinsics_px"]) == pytest.approx(camera_projection(EXPLICIT).matrix)
    # The SDK generic getter looks valid but loses the off-center principal point.
    assert camera.get_intrinsics_matrix()[0, 2] == 320
    assert report["intrinsics_px"][0][2] == 318.6
    assert camera.get_intrinsics_matrix()[0, 0] == pytest.approx(617.2)
    assert camera.get_intrinsics_matrix()[1, 1] == pytest.approx(611.8)
    assert report["distortion_coefficients_opencv"] == EXPLICIT["distortion_coefficients"] + [0] * 7


@pytest.mark.parametrize("patch", [{"fx": 740}, {"cx": 319}, {"cy": 241}])
def test_legacy_rejects_intrinsics_it_cannot_represent(patch):
    with pytest.raises(ValueError, match="use camera_model='opencv_pinhole'"):
        configure_camera_intrinsics(SdkCameraDouble(), {**LEGACY, **patch})


@pytest.mark.parametrize("patch", [
    {"fx": 0}, {"fy": float("nan")}, {"cx": float("inf")},
    {"resolution": [640.5, 480]}, {"resolution": [0, 480]},
    {"distortion_coefficients": [0.1, 0.2]},
    {"distortion_model": "inverse_brown_conrady"},
    {"distortion_model": "modified_brown_conrady"},
    {"distortion_model": "none"},
])
def test_bad_calibration_is_rejected_before_authoring(patch):
    camera = SdkCameraDouble()
    with pytest.raises(ValueError):
        configure_camera_intrinsics(camera, {**EXPLICIT, **patch})
    assert camera.calls == []


def test_missing_distortion_is_explicitly_zeroed_for_new_lens():
    config = {**LEGACY, "camera_model": "opencv_pinhole"}
    camera = SdkCameraDouble()
    configure_camera_intrinsics(camera, config)
    assert camera.properties["pinhole"] == [0] * 12
    assert validate_camera_intrinsics(camera, config)["readback_matches_config"]


@pytest.mark.parametrize("mutation", ["fx", "cx", "distortion", "schema", "resolution"])
def test_readback_detects_runtime_overrides(mutation):
    camera = SdkCameraDouble()
    configure_camera_intrinsics(camera, copy.deepcopy(EXPLICIT))
    if mutation in {"fx", "cx"}:
        camera.properties[mutation] += 1
    elif mutation == "distortion":
        camera.properties["pinhole"][0] += 0.01
    elif mutation == "schema":
        camera.properties = None
    else:
        camera.resolution = (1280, 720)
    with pytest.raises(RuntimeError):
        validate_camera_intrinsics(camera, EXPLICIT)


def test_configuration_resolution_must_match_render_target():
    with pytest.raises(ValueError, match="render resolution"):
        configure_camera_intrinsics(SdkCameraDouble(), {**EXPLICIT, "resolution": [1280, 720]})
