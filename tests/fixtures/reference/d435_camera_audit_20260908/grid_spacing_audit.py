"""Measure image-space grid spacing without changing camera or geometry.

Run with: uv run --no-project --with opencv-python-headless --with numpy \
  python grid_spacing_audit.py --real /path/to/scene_0000.png \
  --sim-root /path/to/previews/sept04 --output grid_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(path: Path) -> dict:
    im = cv2.imread(str(path))
    if im is None or im.shape != (480, 640, 3):
        raise ValueError(f"Expected 640x480 image: {path}")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, _, stats, centers = cv2.connectedComponentsWithStats(
        (gray < 60).astype("uint8")
    )
    points = np.array([
        p for s, p in zip(stats, centers)
        if 4 <= s[4] <= 140 and 2 <= s[2] <= 17 and 2 <= s[3] <= 17
    ])
    rois = {
        "upper_left": [15, 10, 160, 85],
        "middle_left": [10, 185, 160, 285],
        "lower_left": [10, 340, 160, 470],
    }
    out = {"source_path": str(path.resolve()), "sha256": sha256(path), "table": {}}
    for name, (x0, y0, x1, y1) in rois.items():
        selected = points[(points[:, 0] > x0) & (points[:, 0] < x1)
                          & (points[:, 1] > y0) & (points[:, 1] < y1)]
        pairs = []
        for p in selected:
            delta = selected - p
            candidates = np.where((delta[:, 0] > 10) & (delta[:, 0] < 55)
                                  & (np.abs(delta[:, 1]) < 8))[0]
            if len(candidates):
                j = candidates[np.linalg.norm(delta[candidates], axis=1).argmin()]
                pairs.append({
                    "from_xy": p.tolist(), "to_xy": selected[j].tolist(),
                    "distance_px": float(np.linalg.norm(delta[j])),
                    "image_row_angle_deg": float(np.degrees(np.arctan2(
                        delta[j, 1], delta[j, 0]))),
                })
        distances = [p["distance_px"] for p in pairs]
        angles = [p["image_row_angle_deg"] for p in pairs]
        out["table"][name] = {
            "roi_xyxy": [x0, y0, x1, y1],
            "selected_hole_centers_xy": selected.tolist(), "pairs": pairs,
            "pair_count": len(pairs),
            "median_pitch_px": float(np.median(distances)),
            "median_row_angle_deg": float(np.median(angles)),
        }
    # Shared clear strip; x250..435 excludes every candidate's mat edge.
    profile = np.mean(gray[460:465], axis=0).astype(np.float32)
    residual = profile - cv2.GaussianBlur(profile.reshape(1, -1), (31, 1), 0)[0]
    peaks = [x for x in range(250, 436) if residual[x] > 6
             and residual[x] > residual[x - 1] and residual[x] >= residual[x + 1]]
    out["mat"] = {
        "roi_xyxy": [250, 460, 436, 465],
        "vertical_grid_line_peak_x_px": peaks,
        "adjacent_spacing_px": np.diff(peaks).tolist(),
        "median_pitch_px": float(np.median(np.diff(peaks))),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"real": args.real}
    paths.update({name: args.sim_root / name / "sim_first.png" for name in
                  ("r2_original", "rgb_fov_only", "rgb_pose_fit")})
    results = {name: measure(path) for name, path in paths.items()}
    for name, result in results.items():
        if name == "real":
            continue
        for roi, metrics in result["table"].items():
            ref = results["real"]["table"][roi]
            metrics["relative_pitch_error_percent"] = 100 * (
                metrics["median_pitch_px"] / ref["median_pitch_px"] - 1)
            metrics["row_angle_difference_deg"] = (
                metrics["median_row_angle_deg"] - ref["median_row_angle_deg"])
        result["mat"]["relative_pitch_error_percent"] = 100 * (
            result["mat"]["median_pitch_px"] / results["real"]["mat"]["median_pitch_px"] - 1)
    report = {
        "schema_version": 1,
        "status": "independent_image_spacing_diagnostic_not_camera_calibration",
        "method": {
            "table": "Gray<60 connected components, area4..140px and width/height2..17px; fixed unobstructed left-table ROIs; nearest right-hand neighbor with dx10..55px and abs(dy)<8px; median Euclidean adjacent-hole spacing and image row angle.",
            "mat": "Mean grayscale scanline y460..464; subtract Gaussian31x1 local background; local maxima above6gray levels in x250..435; this clear strip excludes mat edges, packet and gripper.",
            "library_versions": {"opencv": cv2.__version__, "numpy": np.__version__},
            "script_sha256": sha256(Path(__file__)),
        },
        "limits": [
            "Fixed image regions compare projected local scale; they are not registered physical landmarks or a plane homography.",
            "Threshold-derived hole centers can vary with lighting, reflection and irregular hole visibility. Allow approximately1-2px center uncertainty; the small residual percentages are diagnostic, not statistical confidence bounds.",
            "The same 85mm mat pitch and50mm table pitch do not by themselves fix mat yaw/position, table height, distortion, camera pose or current-to-September setup continuity.",
            "PhoneIMG2038 shows repeated hole centers close to ruler2.5/7.5/12.5/17.5/22.5cm, supporting50mmhole pitch on the photographed setup; exact centers are partly occluded. It does not establish recording-era pitch independently.",
            "No camera intrinsics, scene geometry or policy configuration is optimized by this diagnostic.",
        ],
        "interpretation": [
            "Changing RGB FOV alone removes most original table-scale excess in all three ROIs.",
            "Robot-landmark pose fit leaves opposite-sign upper/lower table-scale residuals and rotates table rows relative to real. A low robot landmark error does not establish globally correct camera calibration.",
            "The bottom mat spacing remains larger in both new candidates; this could involve mat geometry/pose, image projection, distortion or setup continuity. It does not identify a unique cause.",
        ],
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {
        "table": {roi: {k: v for k, v in m.items() if k.startswith("median_")}
                  for roi, m in data["table"].items()}, "mat": data["mat"]}
        for name, data in results.items()}, indent=2))


if __name__ == "__main__":
    main()
