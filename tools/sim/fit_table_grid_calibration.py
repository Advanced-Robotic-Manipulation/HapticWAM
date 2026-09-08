"""Fit a metric table-hole lattice in the recorded September 4 scene frame.

This image-specific diagnostic uses explicit seed annotations, not robot poses.
Run with uv run --no-project --with numpy --with opencv-python-headless --with
scipy python tools/sim/fit_table_grid_calibration.py --image IMAGE --profile
PROFILE --output OUTPUT. No camera stream, simulator or robot is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_points(image: Path, threshold: int) -> np.ndarray:
    bgr = cv2.imread(str(image))
    if bgr is None or bgr.shape != (480, 640, 3):
        raise ValueError("The explicit seed annotations require the 640x480 reference image")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, _, stats, centers = cv2.connectedComponentsWithStats((gray < threshold).astype("uint8"))
    points = np.array([p for s, p in zip(stats, centers)
                       if 4 <= s[4] <= 140 and 2 <= s[2] <= 17 and 2 <= s[3] <= 17])
    # Connected unobstructed table region; boundary-clipped holes are excluded.
    keep = ((points[:, 0] < 170) | ((points[:, 1] < 80) & (points[:, 0] < 450)))
    keep &= (points[:, 0] > 4) & (points[:, 1] > 5) & (points[:, 1] < 475)
    return points[keep]


def assign_lattice(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    # Origin: hole near pixel44.6,35.4. i increases image-right; j follows
    # successive table rows toward the camera/image-bottom. Pitch is one unit.
    rows = [
        (18., [49.6, 72.6, 95.7, 118.9, 142.3]),
        (35.3, [44.6, 68.1, 91.5, 115.3, 138.8]),
        (53., [38.5, 63.4, 87.1, 111.4, 135.4, 159.2]),
        (71., [33.5, 58.5, 82.6, 107.1, 131.9, 155.8]),
    ]
    seed_px, seed_ij, annotations = [], [], []
    for j, (y, xs) in enumerate(rows):
        for i, x in enumerate(xs):
            p = points[np.linalg.norm(points - [x, y], axis=1).argmin()]
            if np.linalg.norm(p - [x, y]) > 3:
                raise ValueError(f"Missing seed hole near {x,y}")
            seed_px.append(p)
            seed_ij.append([i, j - 1])
            annotations.append({"pixel_hint": [x, y], "ij": [i, j - 1]})
    h, _ = cv2.findHomography(np.asarray(seed_ij, dtype=float), np.asarray(seed_px))
    # Expanding the image region prevents a noisy small initial homography
    # from jumping by a whole grid cell at distant rows.
    for ymax in [90, 120, 160, 200, 250, 300, 360, 420, 480]:
        for _ in range(4):
            selected = points[points[:, 1] < ymax]
            continuous = cv2.perspectiveTransform(selected[None], np.linalg.inv(h))[0]
            integer = np.round(continuous)
            keep = np.linalg.norm(continuous - integer, axis=1) < .23
            h, _ = cv2.findHomography(integer[keep], selected[keep], cv2.RANSAC, 1.7)
    continuous = cv2.perspectiveTransform(points[None], np.linalg.inv(h))[0]
    integer = np.round(continuous).astype(int)
    keep = np.linalg.norm(continuous - integer, axis=1) < .12
    selected, integer = points[keep], integer[keep]
    # Keep at most one observed dark component per lattice site.
    prediction = cv2.perspectiveTransform(integer.astype(float)[None], h)[0]
    residual = np.linalg.norm(prediction - selected, axis=1)
    unique = {}
    for n, ij in enumerate(integer):
        key = tuple(ij)
        if key not in unique or residual[n] < residual[unique[key]]:
            unique[key] = n
    pick = sorted(unique.values())
    selected, integer = selected[pick], integer[pick]
    h, _ = cv2.findHomography(integer.astype(float), selected)
    return integer, selected, h, annotations


def projected(object_xyz: np.ndarray, rotation: np.ndarray, center: np.ndarray,
              k: np.ndarray) -> np.ndarray:
    r = cv2.Rodrigues(rotation)[0]
    return cv2.projectPoints(object_xyz, rotation, -r @ center, k, np.zeros(5))[0][:, 0]


def metrics(errors: np.ndarray) -> dict:
    return {"count": len(errors), "rms_px": float(np.sqrt(np.mean(errors ** 2))),
            "median_px": float(np.median(errors)), "p95_px": float(np.quantile(errors, .95)),
            "max_px": float(np.max(errors))}


def fit_pose(xyz: np.ndarray, pixels: np.ndarray, k: np.ndarray,
             train: np.ndarray, fixed_height: float | None = None) -> dict:
    _, rotation, translation = cv2.solvePnP(xyz[train], pixels[train], k, np.zeros(5))
    r = cv2.Rodrigues(rotation)[0]
    center = (-r.T @ translation).ravel()
    initial = np.r_[rotation.ravel(), center[:2] if fixed_height else center]

    def unpack(parameters):
        c = np.r_[parameters[3:5], fixed_height] if fixed_height else parameters[3:]
        return parameters[:3], c

    def residual(parameters):
        rv, c = unpack(parameters)
        return (projected(xyz[train], rv, c, k) - pixels[train]).ravel()

    fit = least_squares(residual, initial, loss="soft_l1", f_scale=1.0,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    rv, c = unpack(fit.x)
    r = cv2.Rodrigues(rv)[0]
    t = -r @ c
    projection = projected(xyz, rv, c, k)
    errors = np.linalg.norm(projection - pixels, axis=1)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, t
    return {
        "status": "converged" if fit.success else "optimizer_not_converged",
        "fixed_height_above_table_m": fixed_height,
        "camera_from_grid": transform.tolist(),
        "grid_from_camera": np.linalg.inv(transform).tolist(),
        "camera_center_in_grid_m": c.tolist(),
        "grid_up_normal_in_camera": r[:, 2].tolist(),
        "height_above_table_m": float(c[2]),
        "inclination_from_table_normal_deg": float(np.degrees(np.arccos(abs(r[2, 2])))),
        "fit_error": metrics(errors[train]),
        "validation_error": metrics(errors[~train]) if np.any(~train) else None,
        "all_error": metrics(errors),
        "projection_pixels": projection.tolist(),
        "point_errors_px": errors.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pitch-m", type=float, default=.050)
    parser.add_argument("--camera-above-mat-m", type=float, default=.963)
    parser.add_argument("--mat-thickness-m", type=float, default=.003)
    parser.add_argument("--threshold", type=int, default=60)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    k = np.array(profile["intrinsics"]["K"], dtype=float)
    distortion = profile["intrinsics"]["distortion_coefficients"]
    if any(float(d) != 0. for d in distortion):
        raise ValueError("This diagnostic requires the queried all-zero distortion coefficients")
    points = table_points(args.image, args.threshold)
    ij, pixel, h, annotations = assign_lattice(points)
    # Right-handed physical frame with Z above the table, toward the camera.
    xyz = np.column_stack((ij[:, 0] * args.pitch_m, -ij[:, 1] * args.pitch_m,
                           np.zeros(len(ij))))
    train = (ij[:, 0] + 3 * ij[:, 1]) % 5 != 0
    fixed_height = args.camera_above_mat_m + args.mat_thickness_m
    all_points = np.ones(len(xyz), dtype=bool)
    fits = {
        "free_height_all_points": fit_pose(xyz, pixel, k, all_points),
        "measured_height_all_points": fit_pose(xyz, pixel, k, all_points, fixed_height),
        "free_height_withheld_grid_sites": fit_pose(xyz, pixel, k, train),
        "measured_height_withheld_grid_sites": fit_pose(xyz, pixel, k, train, fixed_height),
    }
    columns = np.linalg.inv(k) @ h
    orthogonal_angle = np.degrees(np.arccos(np.clip(
        np.dot(columns[:, 0], columns[:, 1]) /
        (np.linalg.norm(columns[:, 0]) * np.linalg.norm(columns[:, 1])), -1., 1.)))
    result = {
        "schema_version": 1, "status": "metric_table_plane_fit_with_unresolved_robot_registration",
        "inputs": {
            "image_path": str(args.image.resolve()), "image_sha256": digest(args.image),
            "profile_path": str(args.profile.resolve()), "profile_sha256": digest(args.profile),
            "script_sha256": digest(Path(__file__)), "table_hole_pitch_m": args.pitch_m,
            "camera_above_mat_reported_m": args.camera_above_mat_m,
            "mat_thickness_assumed_m": args.mat_thickness_m,
            "distortion_used": distortion, "K": k.tolist(), "dark_threshold": args.threshold,
        },
        "grid_frame": {
            "origin": "Visible hole near pixel (44.6,35.4), arbitrarily assigned lattice(0,0).",
            "x": "Toward the adjacent right-hand hole near pixel(68.1,35.3).",
            "y": "Opposite successive image-down table rows; XYZ=(pitch*i,-pitch*j,0).",
            "z": "Right-handed table normal pointing above the table toward the camera.",
            "robot_registration": "Unknown translation in table plane and yaw about table normal; cannot identify UR origin from a repeating grid alone.",
        },
        "assignment": {
            "method": "Explicit top-left seed lattice; progressively extend a projective homography through connected visible table regions; reject >0.12-cell assignment residual and duplicate components.",
            "seed_annotations": annotations,
            "homography_pixel_from_ij": h.tolist(),
            "projective_axis_angle_after_K_inverse_deg": float(orthogonal_angle),
            "projective_axis_norm_ratio_after_K_inverse": float(np.linalg.norm(columns[:, 0]) / np.linalg.norm(columns[:, 1])),
            "lattice_ij": ij.tolist(), "object_xyz_m": xyz.tolist(), "observed_pixels": pixel.tolist(),
            "within_image_validation_mask": (~train).tolist(),
        },
        "fits": fits,
        "limitations": [
            "The measured device intrinsics are current available-profile calibration; recording-era intrinsics were not saved.",
            "The table pitch is user-reported and supported by current ruler photos, not an independent recording-era measurement.",
            "Thresholded dark-hole centroids can shift with illumination/reflections and are not surveyed fiducials; residuals are diagnostic, not metrology uncertainty.",
            "The five-way lattice-site split holds out image points after lattice assignment, not independent frames or setup sessions.",
            "This fit fixes the table plane scale and normal in camera coordinates; robot-frame yaw/translation still require independent robot or mounting landmarks.",
            "Changing grid pitch scales fitted camera height and translation proportionally. This tool does not alter any scene, policy or recording.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in fit.items()
                            if key not in ("projection_pixels", "point_errors_px")}
                      for name, fit in fits.items()}, indent=2))


if __name__ == "__main__":
    main()
