"""Explicit Isaac camera projection configuration without importing Isaac.

Isaac Sim 6's aperture setters maintain square pixels by default, and its
``get_intrinsics_matrix`` always reports a centered principal point, including
for the OpenCV lens schema. Read that schema directly for calibrated cameras.
The legacy authoring calls remain unchanged for historical centered cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraProjection:
    model: str
    resolution: tuple[int, int]
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


def camera_projection(config: Mapping[str, Any]) -> CameraProjection:
    """Validate the supported projection and coefficient convention.

    Nonzero distortion requires explicit OpenCV Brown-Conrady semantics; a
    RealSense modified/inverse model must not be relabeled without conversion.
    """
    resolution = np.asarray(config["resolution"], dtype=float)
    if (
        resolution.shape != (2,)
        or not np.isfinite(resolution).all()
        or np.any(resolution <= 0)
        or np.any(resolution != np.floor(resolution))
    ):
        raise ValueError("Camera resolution must contain two positive integers")
    width, height = (int(value) for value in resolution)
    fx, fy = float(config["fx"]), float(config["fy"])
    cx, cy = float(config.get("cx", width / 2)), float(config.get("cy", height / 2))
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("Camera intrinsics must be finite with positive fx and fy")
    if not 0 <= cx <= width or not 0 <= cy <= height:
        raise ValueError("Camera principal point must lie within the configured image")
    model = config.get("camera_model", "legacy_centered_pinhole")
    if model not in {"legacy_centered_pinhole", "opencv_pinhole"}:
        raise ValueError(f"Unsupported camera_model: {model!r}")
    distortion = np.asarray(config.get("distortion_coefficients", []), dtype=float)
    if (
        distortion.ndim != 1
        or distortion.size not in {0, 4, 5, 8, 12}
        or not np.isfinite(distortion).all()
    ):
        raise ValueError("OpenCV distortion must have 0, 4, 5, 8 or 12 finite coefficients")
    distortion_model = config.get("distortion_model", "none")
    if distortion_model not in {"none", "opencv_brown_conrady"}:
        raise ValueError(f"Unsupported distortion_model: {distortion_model!r}")
    if distortion_model == "none" and np.any(distortion != 0):
        raise ValueError("Nonzero coefficients require distortion_model='opencv_brown_conrady'")
    if model == "legacy_centered_pinhole":
        if (
            not np.allclose([fx, cx, cy], [fy, width / 2, height / 2], rtol=0, atol=1e-6)
            or np.any(distortion != 0)
        ):
            raise ValueError(
                "Legacy camera requires fx=fy, centered principal point and zero distortion; "
                "use camera_model='opencv_pinhole' for explicit intrinsics"
            )
    return CameraProjection(
        model, (width, height), fx, fy, cx, cy,
        tuple(np.pad(distortion, (0, 12 - len(distortion))).tolist()),
    )


def configure_camera_intrinsics(camera: Any, config: Mapping[str, Any]) -> None:
    """Author the projection before initialization; preserve legacy lens calls."""
    projection = camera_projection(config)
    if tuple(camera.get_resolution()) != projection.resolution:
        raise ValueError("Camera render resolution differs from intrinsics calibration")
    camera.set_focal_length(24.0)
    if projection.model == "legacy_centered_pinhole":
        camera.set_horizontal_aperture(projection.resolution[0] * 24 / projection.fx)
        camera.set_vertical_aperture(projection.resolution[1] * 24 / projection.fy)
    else:
        camera.set_horizontal_aperture(
            projection.resolution[0] * 24 / projection.fx, maintain_square_pixels=False
        )
        camera.set_vertical_aperture(
            projection.resolution[1] * 24 / projection.fy, maintain_square_pixels=False
        )
        # Native Isaac 6 lens schema supports fx != fy and off-center cx/cy.
        camera.set_opencv_pinhole_properties(
            cx=projection.cx, cy=projection.cy, fx=projection.fx, fy=projection.fy,
            pinhole=list(projection.distortion),
        )


def validate_camera_intrinsics(camera: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Read back the actual lens after initialization and fail on mismatch.

    Schema readback verifies authoring, not calibration accuracy or pixel-level
    renderer fidelity. Those still require an independently projected fixture.
    """
    projection = camera_projection(config)
    if tuple(camera.get_resolution()) != projection.resolution:
        raise RuntimeError("Initialized camera resolution differs from configured intrinsics")
    if projection.model == "opencv_pinhole":
        properties = camera.get_opencv_pinhole_properties()
        if properties is None or len(properties) != 5:
            raise RuntimeError("Initialized camera has no OpenCV pinhole lens schema")
        cx, cy, fx, fy, distortion = properties
        actual = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        distortion = np.asarray(distortion, dtype=float)
        source = "Isaac OpenCV pinhole lens schema (generic K getter ignores principal point)"
    else:
        actual = np.asarray(camera.get_intrinsics_matrix(), dtype=float)
        distortion = np.zeros(12)
        source = "Isaac centered aperture intrinsics getter"
    if (
        actual.shape != (3, 3)
        or not np.isfinite(actual).all()
        or not np.allclose(actual, projection.matrix, rtol=0, atol=1e-3)
    ):
        raise RuntimeError(f"Initialized camera K mismatch: expected {projection.matrix}, got {actual}")
    if (
        distortion.shape != (12,)
        or not np.isfinite(distortion).all()
        or not np.allclose(distortion, projection.distortion, rtol=1e-6, atol=1e-7)
    ):
        raise RuntimeError("Initialized camera distortion differs from configured coefficients")
    width, height = projection.resolution
    fx, fy, cx, cy = actual[0, 0], actual[1, 1], actual[0, 2], actual[1, 2]
    return {
        "camera_model": projection.model,
        "resolution": list(projection.resolution),
        "intrinsics_px": actual.tolist(),
        "distortion_coefficients_opencv": distortion.tolist(),
        "intrinsics_source": source,
        "readback_matches_config": True,
        "undistorted_pinhole_fov_degrees": {
            "horizontal": float(np.degrees(np.arctan(cx / fx) + np.arctan((width - cx) / fx))),
            "vertical": float(np.degrees(np.arctan(cy / fy) + np.arctan((height - cy) / fy))),
        },
        "validation_scope": "Lens schema readback only; not device calibration or rendered landmark validation",
    }
