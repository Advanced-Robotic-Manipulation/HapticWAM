#!/usr/bin/env python3
"""Fit an approximate scene camera to manual landmarks on a fit episode.

Run from repository root with numpy/scipy/OpenCV. This is reproducible image
registration, not hardware calibration. The holdout set is not used here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phantom.sim.kinematics import link_transforms

# Each row contains wrist_1 cyan cap face center and center between pad tips.
# These hand annotations have uncertainty; neither is a fiducial measurement.
ANNOTATIONS = {
    0: ([534, 157], [308, 398]),
    55: ([468, 219], [308, 409]),
    83: ([440, 265], [307, 419]),
    111: ([434, 273], [307, 422]),
    139: ([432, 276], [306, 422]),
    167: ([438, 274], [306, 417]),
    195: ([491, 225], [299, 407]),
    223: ([559, 135], [271, 335]),
}
# Official DAE blue mesh face center transformed by the URDF visual origin.
WRIST1_CAP_LOCAL = np.array([0, 0.0464, -0.00177, 1])


def project(parameters, points):
    rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
    camera_points = points @ rotation.T + parameters[3:6]
    return camera_points[:, :2] / camera_points[:, 2, None] * parameters[6] + [320, 240]


def fit(frames):
    points, pixels, names = [], [], []
    for frame in frames:
        index = int(frame["frame"][6:10])
        if index not in ANNOTATIONS:
            continue
        wrist = link_transforms(frame["q"])["wrist_1_link"]
        points.extend([(wrist @ WRIST1_CAP_LOCAL)[:3], frame["tcp"][:3]])
        pixels.extend(ANNOTATIONS[index])
        names.extend([f"{index}:wrist_1_cap", f"{index}:tcp"])
    points, pixels = np.array(points), np.array(pixels, dtype=float)
    if len(points) < 8:
        raise ValueError("Need at least four annotated frames")
    intrinsic = np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1.0]])
    ok, rotation, translation = cv2.solvePnP(
        points, pixels, intrinsic, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP
    )
    if not ok:
        raise RuntimeError("PnP initialization failed")

    def residual(parameters):
        return np.r_[
            (project(parameters, points) - pixels).ravel() / 3,
            (parameters[6] - 615) / 120,
        ]

    initial = np.r_[rotation.ravel(), translation.ravel(), 615.0]
    result = least_squares(
        residual,
        initial,
        bounds=(np.r_[[-np.inf] * 6, 400], np.r_[[np.inf] * 6, 900]),
        loss="soft_l1",
        f_scale=2,
        max_nfev=1000,
    )
    parameters = result.x
    r = Rotation.from_rotvec(parameters[:3]).as_matrix()
    camera_points = points @ r.T + parameters[3:6]
    if not result.success or np.any(camera_points[:, 2] <= 0):
        raise RuntimeError("Camera fit failed or landmarks are behind the camera")
    camera_from_base = np.eye(4)
    camera_from_base[:3, :3] = r
    camera_from_base[:3, 3] = parameters[3:6]
    # USD camera local -Z forward / +Y up; OpenCV +Z forward / +Y down.
    base_from_usd = np.linalg.inv(camera_from_base) @ np.diag([1, -1, -1, 1])
    projected = project(parameters, points)
    return {
        "status": "estimated_from_manual_RGB_landmarks_not_hardware_calibration",
        "resolution": [640, 480],
        "K": [[parameters[6], 0, 320], [0, parameters[6], 240], [0, 0, 1]],
        "distortion_assumed": [0, 0, 0, 0, 0],
        "T_camera_cv_from_ur_base": camera_from_base.tolist(),
        "T_ur_base_from_usd_camera": base_from_usd.tolist(),
        "camera_position_ur_base_m": base_from_usd[:3, 3].tolist(),
        "usd_camera_quaternion_wxyz": np.roll(
            Rotation.from_matrix(base_from_usd[:3, :3]).as_quat(), 1
        ).tolist(),
        "fit_reprojection_rmse_px": float(
            np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1)))
        ),
        "landmarks": [
            {
                "name": name,
                "world_m": point.tolist(),
                "pixel": pixel.tolist(),
                "projected_pixel": pred.tolist(),
                "error_px": float(np.linalg.norm(pred - pixel)),
            }
            for name, point, pixel, pred in zip(names, points, pixels, projected)
        ],
        "assumptions": [
            "fx=fy fitted with weak 615+/-120 px prior; principal point fixed to image center.",
            "No distortion calibration, camera intrinsics or extrinsics available.",
            "Wrist1 cyan face center from nominal DAE and visual transform: [0,.0464,-.00177] in wrist_1_link.",
            "TCP pixel approximates center between pad tips; physical TCP may not coincide exactly.",
            "In-sample fit error includes uncertain hand labels and is not a held-out image metric.",
        ],
    }


def estimate_layout(camera):
    transform = np.array(camera["T_camera_cv_from_ur_base"])
    intrinsic_inv = np.linalg.inv(camera["K"])
    center = np.array(camera["camera_position_ur_base_m"])

    def on_plane(pixels, height):
        rays = (intrinsic_inv @ np.c_[pixels, np.ones(len(pixels))].T).T @ transform[
            :3, :3
        ]
        return center + rays * ((height - center[2]) / rays[:, 2])[:, None]

    rim_pixels = [[194, 92], [430, 92], [439, 249], [178, 249]]
    rim = on_plane(rim_pixels, 0.14 - 0.022)
    width, depth = rim[1] - rim[0], rim[3] - rim[0]
    mat_pixels = [[224, 287], [449, 287], [466, 479], [210, 479]]
    mat = on_plane(mat_pixels, -0.02)
    return {
        "status": "estimates_from_uncalibrated_camera_not_measured_dimensions",
        "table_surface_z_ur_base_m": -0.022,
        "bin": {
            "rim_pixels": rim_pixels,
            "rim_corners_world_m": rim.tolist(),
            "center_world_xy_m": rim.mean(axis=0)[:2].tolist(),
            "height_prior_m": 0.14,
            "width_estimate_m": float(np.linalg.norm(width)),
            "depth_estimate_m": float(np.linalg.norm(depth)),
            "yaw_estimate_rad": float(np.arctan2(width[1], width[0])),
        },
        "mat": {
            "visible_extent_pixels": mat_pixels,
            "visible_corners_world_m": mat.tolist(),
            "visible_width_m": float(np.linalg.norm(mat[1] - mat[0])),
            "visible_depth_lower_bound_m": float(np.linalg.norm(mat[3] - mat[0])),
            "notes": "Top is guessed under the bin. Bottom is cropped; depth is a lower bound.",
        },
        "assumptions": [
            "Table surface -22 mm below base and bin height 140 mm are priors.",
            "Uncalibrated camera uncertainty propagates to all sizes and positions.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("artifacts/isaac_waffles/evidence/landmark_frames.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/isaac_waffles/evidence")
    )
    args = parser.parse_args()
    camera = fit(json.loads(args.frames.read_text()))
    camera["source"] = (
        str(args.frames) + " and associated RGB frames; calibration episode only"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "camera_fit.json").write_text(json.dumps(camera, indent=2) + "\n")
    (args.out / "layout_estimates.json").write_text(
        json.dumps(estimate_layout(camera), indent=2) + "\n"
    )
    print(f"Manual-landmark fit RMSE: {camera['fit_reprojection_rmse_px']:.2f} pixels")
    print(f"Camera position: {camera['camera_position_ur_base_m']}")


if __name__ == "__main__":
    main()
