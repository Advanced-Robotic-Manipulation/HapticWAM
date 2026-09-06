#!/usr/bin/env python3
"""Read-only visual/telemetry analysis of prepared real waffle episodes.

Writes separate evidence artifacts. HSV tracks are visible wrapper-pixel
centroids, not calibrated object centers or six-dimensional object poses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
FIT = "ep_teacher_waffles_1788535016_005"
COMPLETE = "ep_waffles_1787395928_000"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def color_component(frame, green, offset=0):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = [32, 70 + offset, 35] if green else [0, 45 + offset, 90]
    high = [90, 255, 255] if green else [38, 255, 255]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    mask[: 80 if green else 250] = 0
    mask[:, :130] = 0
    mask[:, 490:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    valid = [j for j in range(1, n) if stats[j, 4] >= 150]
    if not valid:
        return None
    j = max(valid, key=lambda j: stats[j, 4])
    y, x = np.where(labels == j)
    rect = cv2.minAreaRect(np.c_[x, y].astype(np.float32))
    return centers[j], stats[j], cv2.boxPoints(rect), (labels == j).astype(np.uint8)


def visible_white_caps(frame, patch_center):
    """Image upper/lower white-shell candidates; no physical handedness claim."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([179, 58, 255]))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for j in range(1, n):
        _x, _y, w, h, area = stats[j]
        center = centers[j]
        if not (100 <= area <= 2800 and 10 <= w <= 110 and 7 <= h <= 110):
            continue
        ys, xs = np.where(labels == j)
        _, dims, _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        if min(dims) < 3 or max(dims) / min(dims) < 1.35:
            continue
        if not patch_center[0] - 15 <= center[0] <= patch_center[0] + 140:
            continue
        distance = np.linalg.norm(center - (patch_center + [50, -10]))
        if distance < 115:
            candidates.append((distance, center))
    return sorted(
        [c for _, c in sorted(candidates, key=lambda x: x[0])[:2]], key=lambda x: x[1]
    )


def sample_native(data, stream, grid):
    values = data["native_" + stream]
    stamps = data["native_" + stream + "_t"]
    if values.ndim == 1:
        return np.interp(grid, stamps, values)
    return np.stack(
        [np.interp(grid, stamps, values[:, k]) for k in range(values.shape[1])], 1
    )


def first_sustained(t, condition, duration=0.20, after=0):
    start = None
    for stamp, yes in zip(t, condition):
        if stamp < after or not yes:
            start = None
        elif start is None:
            start = float(stamp)
        elif stamp - start >= duration:
            return start
    return None


def analyze_episode(directory, output):
    data = np.load(directory / "replay.npz", allow_pickle=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    green = directory.name == COMPLETE
    video = cv2.VideoCapture(str(directory / "reference.mp4"))
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    times = data["t"]
    rows, annotated = [], []
    shown = set(np.linspace(0, count - 1, 12).round().astype(int))
    for index in range(count):
        ok, frame = video.read()
        if not ok:
            break
        t = float(times[min(index, len(times) - 1)])
        detection = color_component(frame, green)
        row = {
            "frame": index,
            "grid_t_s": t,
            "camera_t_s": float(data["camera_t"][index]),
            "track": "visible_green_patch" if green else "visible_warm_wrapper_pixels",
            "x_px": None,
            "y_px": None,
            "area_px": None,
            "threshold_spread_px": None,
            "uncertainty_px": None,
            "visibility": "missing",
            "upper_cap_x_px": None,
            "upper_cap_y_px": None,
            "lower_cap_x_px": None,
            "lower_cap_y_px": None,
            "cap_measurement_status": "unreviewed_automatic_candidate",
        }
        if detection is not None:
            center, stats, box, _ = detection
            ensemble = [color_component(frame, green, k) for k in (-15, 15)]
            spread = max(
                [
                    float(np.linalg.norm(x[0] - center))
                    for x in ensemble
                    if x is not None
                ]
                + [0]
            )
            # Hand-inspected occlusion/blur ranges, kept distinct from threshold
            # sensitivity. This is a conservative annotation bound, not a CI.
            occluded = (green and 14.35 <= t <= 16.20) or (
                not green and 9.1 <= t <= 11.1
            )
            external_hand = not green and 4.4 <= t <= 5.6
            blur = green and 8.65 <= t <= 10.05
            uncertainty = max(
                3.0, spread + 2.0, 12.0 if occluded else 7.0 if blur else 0
            )
            row.update(
                x_px=float(center[0]),
                y_px=float(center[1]),
                area_px=int(stats[4]),
                threshold_spread_px=spread,
                uncertainty_px=uncertainty,
                visibility="external_hand"
                if external_hand
                else "partial_occlusion"
                if occluded
                else "motion_blur"
                if blur
                else "visible",
            )
            if green and index in (0, 27, 54, 81, 108, 135, 161):
                row["cap_measurement_status"] = (
                    "visually_reviewed_shell_centroid_approximation"
                )
            caps = visible_white_caps(frame, center)
            for prefix, cap in zip(("upper_cap", "lower_cap"), caps):
                row[prefix + "_x_px"], row[prefix + "_y_px"] = map(float, cap)
                cv2.circle(frame, tuple(np.round(cap).astype(int)), 5, (255, 0, 255), 2)
            cv2.polylines(frame, [box.round().astype(int)], True, (0, 255, 255), 1)
            cv2.circle(frame, tuple(center.round().astype(int)), 5, (0, 255, 0), 2)
        rows.append(row)
        if index in shown:
            cv2.putText(
                frame,
                f"{t:.2f}s {row['visibility']}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                frame,
                f"{t:.2f}s {row['visibility']}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            annotated.append(cv2.resize(frame, (400, 300)))
    video.release()
    destination = output / directory.name
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "image_tracks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reviewed = [
        {**row, "cap_uncertainty_px": 8 if row["grid_t_s"] >= 8.5 else 5}
        for row in rows
        if row["cap_measurement_status"].startswith("visually_reviewed")
    ]
    (destination / "reviewed_shell_landmarks.json").write_text(
        json.dumps(reviewed, indent=2) + "\n"
    )
    if len(annotated) == 12:
        cv2.imwrite(
            str(destination / "tracking_review.jpg"),
            np.concatenate(
                [np.concatenate(annotated[i : i + 3], 1) for i in range(0, 12, 3)], 0
            ),
        )

    grid = np.arange(0, float(times[-1]) + 0.0001, 0.02)
    tcp = sample_native(data, "arm_tcp_pose", grid)
    gripper = sample_native(data, "gripper", grid)[:, 0]
    loads = []
    for side in ("left", "right"):
        stream = "tactile_" + side + "_wrench"
        raw = data["native_" + stream]
        ts = data["native_" + stream + "_t"]
        baseline = float(np.median(raw[(ts >= 0) & (ts <= 1), 2]))
        loads.append(np.abs(sample_native(data, stream, grid)[:, 2] - baseline))
    close = first_sustained(
        grid,
        (gripper >= 0.45) & (gripper - np.minimum.accumulate(gripper) >= 0.15),
        0.1,
    )
    contact = first_sustained(grid, (loads[0] > 2) & (loads[1] > 2), 0.2)
    minimum_index = int(np.argmin(tcp[:, 2]))
    minimum_t = float(grid[minimum_index])
    lift = first_sustained(
        grid, tcp[:, 2] > tcp[minimum_index, 2] + 0.02, 0.15, minimum_t
    )
    release = None
    if contact is not None:
        after = grid > contact
        running = np.maximum.accumulate(np.where(after, gripper, 0))
        release = first_sustained(
            grid, after & (gripper < running - 0.15), 0.1, contact
        )
    unload = first_sustained(
        grid,
        (loads[0] < 2) & (loads[1] < 2),
        0.3,
        (release if release is not None else float(times[-1]) + 1),
    )
    telemetry = {
        "closure_training_threshold_s": close,
        "dual_pad_contact_above_2_sensor_units_s": contact,
        "minimum_tcp_z_s": minimum_t,
        "minimum_tcp_z_m": float(tcp[minimum_index, 2]),
        "lift_20mm_above_min_s": lift,
        "release_closure_drop_0p15_s": release,
        "both_pads_unloaded_below_2_sensor_units_s": unload,
        "tcp_start_xyz_m": tcp[0, :3].tolist(),
        "tcp_end_xyz_m": tcp[-1, :3].tolist(),
        "maximum_tcp_z_m": float(tcp[:, 2].max()),
    }
    visual = {
        "complete_pick_place": bool(green),
        "external_hand_interval_s": None if green else [4.4, 5.6],
        "conclusion": "packet visibly deposited in bin; empty gripper retreats"
        if green
        else "packet remains held; recording ends after lift",
        "visual_lift_interval_s": [7.5, 8.0] if green else [10.9, 11.6],
        "visual_release_interval_s": [13.5, 14.0] if green else None,
        "unoccluded_final_object_interval_s": [17.0, 19.7] if green else None,
    }
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for k, label in enumerate(("x", "y", "z")):
        axes[0].plot(grid, tcp[:, k] * 1000, label=label)
    axes[0].set_ylabel("Measured TCP / mm")
    axes[0].legend(ncol=3)
    axes[1].plot(grid, gripper)
    axes[1].set_ylabel("Measured closure")
    for side, force in zip(("left", "right"), loads):
        axes[2].plot(grid, force, label=side)
    axes[2].axhline(2, color="grey", linestyle=":")
    axes[2].legend()
    axes[2].set_ylabel("Native |fz − baseline| / sensor units")
    for coordinate in ("x_px", "y_px"):
        axes[3].plot(
            [r["grid_t_s"] for r in rows],
            [np.nan if r[coordinate] is None else r[coordinate] for r in rows],
            label=coordinate,
        )
    axes[3].set_ylabel("Visible wrapper centroid / px")
    axes[3].set_xlabel("Episode-relative seconds")
    axes[3].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
        for stamp in (contact, lift, release):
            if stamp is not None:
                ax.axvline(stamp, color="grey", linestyle="--", alpha=0.5)
    fig.suptitle(
        directory.name
        + "\nReal streams; 2D color-patch location is not calibrated object pose"
    )
    fig.tight_layout()
    fig.savefig(destination / "timeline.png", dpi=160)
    plt.close(fig)
    report = {
        "episode": str(directory),
        "source_meta_sha256": manifest["meta_sha256"],
        "prepared_video_sha256": sha(directory / "reference.mp4"),
        "duration_s": float(times[-1]),
        "telemetry_events": telemetry,
        "visual_adjudication": visual,
        "time_uncertainty_s": {
            "video_half_frame": 1 / 30,
            "telemetry_contact": "approximately half native tactile frame spacing; plus chosen 0.2s persistence criterion",
        },
        "image_measurement_limitations": [
            "Native tactile magnitudes are sensor-output units here; no Newton calibration is asserted.",
            "Centroid describes visible colored wrapper pixels, not full-object center; viewpoint/occlusion shifts it.",
            "Uncertainty_px is a conservative heuristic annotation bound, not statistical confidence.",
            "White cap coordinates are image-space candidate shell centroids, not physical gel contact centers; reject misdetections on tracking_review.jpg.",
            "No calibrated 3D object position, friction, compliance, or object mass is inferred from this tracker.",
        ],
    }
    (destination / "timeline.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def green_pose_and_texture(cfg, episode, output):
    camera = cfg["camera"]
    world_from_cv = np.asarray(camera["world_from_cv"])
    cv_from_world = np.linalg.inv(world_from_cv)
    initial_quad = np.array([[254, 408], [367, 363], [377, 383], [266, 430]], float)
    size = np.asarray(cfg["waffle"]["size"])
    center_z = 0.050
    corners = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * size[:2] / 2

    def project(parameters):
        c, s = np.cos(parameters[2]), np.sin(parameters[2])
        rot = np.array([[c, -s], [s, c]])
        xyz = np.c_[
            corners @ rot.T + parameters[:2], np.full(4, center_z + size[2] / 2)
        ]
        xyz = xyz @ cv_from_world[:3, :3].T + cv_from_world[:3, 3]
        return np.c_[
            camera["fx"] * xyz[:, 0] / xyz[:, 2] + camera["cx"],
            camera["fy"] * xyz[:, 1] / xyz[:, 2] + camera["cy"],
        ]

    fit = least_squares(
        lambda x: (project(x) - initial_quad).ravel(), [-0.368, -0.256, 0.44]
    )
    rng = np.random.default_rng(4242)
    samples = []
    for _ in range(200):
        perturbed = initial_quad + rng.normal(0, 3, initial_quad.shape)
        samples.append(
            least_squares(
                lambda x, perturbed=perturbed: (project(x) - perturbed).ravel(), fit.x
            ).x
        )
    cap = cv2.VideoCapture(str(episode / "reference.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 285)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read unoccluded green wrapper frame")
    texture_quad = np.array(
        [[266, 147], [353, 191], [345, 203], [263, 159]], np.float32
    )
    width, height = 1024, 180
    transform = cv2.getPerspectiveTransform(
        texture_quad,
        np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            np.float32,
        ),
    )
    texture = cv2.warpPerspective(
        frame, transform, (width, height), flags=cv2.INTER_LINEAR
    )
    target = ROOT / "assets/sim/waffles/packet_green_top.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), texture)
    provenance = {
        "asset": str(target.relative_to(ROOT)),
        "source_episode": str(episode),
        "source_video_sha256": sha(episode / "reference.mp4"),
        "source_frame": 285,
        "grid_t_s": 19.0,
        "source_top_face_quad_xy_px": texture_quad.tolist(),
        "homography": transform.tolist(),
        "method": "perspective rectification of visible recorded pixels; linear resampling; no inpainting or generative synthesis",
        "limitations": "Upsampled approximately 95x17-pixel source face; blur and baked lighting remain. Metric aspect derives from estimated packet geometry.",
    }
    target.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    result = {
        "episode": str(episode),
        "source_frame": 0,
        "manual_top_face_quad_xy_px": initial_quad.tolist(),
        "camera": camera,
        "assumed_packet_size_m": size.tolist(),
        "assumed_packet_center_z_m": center_z,
        "conditional_initial_packet_center_m": [
            float(fit.x[0]),
            float(fit.x[1]),
            center_z,
        ],
        "conditional_initial_packet_yaw_rad": float(fit.x[2]),
        "projected_top_corners_xy_px": project(fit.x).tolist(),
        "pixel_coordinate_rmse": float(
            np.sqrt(np.mean((project(fit.x) - initial_quad) ** 2))
        ),
        "annotation_only_95pct_interval_xy_yaw": np.percentile(
            samples, [2.5, 97.5], axis=0
        ).tolist(),
        "uncertainty": "Conditional fit only. Bootstrap perturbs corners by Gaussian3px; excludes camera/plane/size systematic errors and is not calibrated metric accuracy.",
        "green_texture": provenance,
    }
    (output / "teleop_initial_packet_pose.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/isaac_waffles/evidence/pick_place_analysis",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    directories = [
        ROOT / "artifacts/isaac_waffles/evidence/fit" / name for name in (FIT, COMPLETE)
    ]
    reports = [analyze_episode(path, args.out) for path in directories]
    pose = green_pose_and_texture(
        json.loads((ROOT / "configs/sim/waffles.json").read_text()),
        directories[1],
        args.out,
    )
    print(
        json.dumps(
            {
                "reports": reports,
                "teleop_initial_pose": pose["conditional_initial_packet_center_m"],
                "teleop_yaw": pose["conditional_initial_packet_yaw_rad"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
