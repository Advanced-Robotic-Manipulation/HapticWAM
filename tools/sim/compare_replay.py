#!/usr/bin/env python3
"""Compare an Isaac trace with timestamped measured PHANTOM replay.

Requires numpy, scipy and OpenCV, but no Isaac/robot drivers. State metrics
use measured feedback, interpolated only within common time support. Videos
are nearest-frame resampled on a common grid; every signed snap error is
saved. Full-image RGB error is an appearance metric, never geometric error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PHYSICS_CLOCK_TOLERANCE_S = 0.0005
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def nearest_indices(ts, target):
    right = np.searchsorted(ts, target).clip(0, len(ts) - 1)
    left = (right - 1).clip(0, len(ts) - 1)
    return np.where(abs(target - ts[left]) <= abs(ts[right] - target), left, right)


def interpolate(ts, values, times):
    return np.stack(
        [np.interp(times, ts, values[:, j]) for j in range(values.shape[1])], axis=1
    )


def validate_trace(data, name):
    for key in ("t", "q", "tcp"):
        if key not in data:
            raise ValueError(f"{name} has no {key}")
    t = data["t"]
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(
            f"{name}: t must contain at least two finite increasing timestamps"
        )
    for key in ("q", "tcp"):
        if data[key].shape != (len(t), 6) or not np.isfinite(data[key]).all():
            raise ValueError(f"{name}: {key} must be a finite (N,6) array")


def magnitude_stats(values, units):
    values = np.asarray(values)
    return {
        "units": units,
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def reference_stream(real, key, native_name):
    """Prefer original feedback; never interpolate an already resampled grid twice."""
    native_key = f"native_{native_name}"
    native_t = f"{native_key}_t"
    if native_key in real or native_t in real:
        if native_key not in real or native_t not in real:
            raise ValueError(
                f"Reference native stream requires both {native_key} and {native_t}"
            )
        ts, values, source = (
            np.asarray(real[native_t]),
            np.asarray(real[native_key]),
            native_key,
        )
    else:
        ts, values, source = np.asarray(real["t"]), np.asarray(real[key]), key
    if (
        ts.ndim != 1
        or len(ts) < 2
        or not np.isfinite(ts).all()
        or np.any(np.diff(ts) <= 0)
    ):
        raise ValueError(
            f"Reference {source}: timestamps must be finite and strictly increasing"
        )
    if values.shape != (len(ts), 6) or not np.isfinite(values).all():
        raise ValueError(f"Reference {source}: samples must be finite (N,6)")
    return ts, values, source


def physics_clock_metrics(sim):
    if "physics_t" not in sim:
        return {
            "status": "not_evaluated",
            "passed": None,
            "reason": "Trace has no independent physics_t clock",
        }
    physics_t, reported_t = np.asarray(sim["physics_t"]), np.asarray(sim["t"])
    if (
        physics_t.shape != reported_t.shape
        or not np.isfinite(physics_t).all()
        or np.any(np.diff(physics_t) <= 0)
    ):
        raise ValueError(
            "physics_t must be finite, strictly increasing and aligned with t"
        )
    delta = physics_t - reported_t
    maximum = float(np.max(abs(delta)))
    passed = maximum <= PHYSICS_CLOCK_TOLERANCE_S
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "max_abs_error_s": maximum,
        "rmse_s": float(np.sqrt(np.mean(delta**2))),
        "tolerance_s": PHYSICS_CLOCK_TOLERANCE_S,
        "comparison": "Independent elapsed PhysX clock minus reported elapsed episode t",
    }


def state_metrics(real, sim):
    from scipy.spatial.transform import Rotation, Slerp

    validate_trace(real, "reference")
    validate_trace(sim, "simulation")
    qt, qvalues, qsource = reference_stream(real, "q", "arm_q")
    pt, pvalues, psource = reference_stream(real, "tcp", "arm_tcp_pose")
    st = sim["t"]
    mask = (st >= max(qt[0], pt[0])) & (st <= min(qt[-1], pt[-1]))
    times = st[mask]
    if not len(times):
        raise ValueError("Reference/simulation state timestamps do not overlap")
    rq = interpolate(qt, qvalues, times)
    rp = interpolate(pt, pvalues[:, :3], times)
    rr = Slerp(pt, Rotation.from_rotvec(pvalues[:, 3:]))(times)
    sr = Rotation.from_rotvec(sim["tcp"][mask, 3:])
    qerr = sim["q"][mask] - rq  # Preserve real measured IK branch, no modulo wrapping.
    xyzerr = sim["tcp"][mask, :3] - rp
    angle = (rr.inv() * sr).magnitude()
    result = {
        "state_samples": len(times),
        "common_state_interval_s": [float(times[0]), float(times[-1])],
        "excluded_sim_state_samples": int((~mask).sum()),
        "reference_sources": {
            "q": qsource,
            "tcp": psource,
            "time_domain": "seconds relative to exported t0_master; no clock offset reapplied",
        },
        "physics_clock_consistency": physics_clock_metrics(sim),
        "joint_error_rad": {
            "rmse_all": float(np.sqrt(np.mean(qerr**2))),
            "rmse_per_joint": np.sqrt(np.mean(qerr**2, axis=0)).tolist(),
            "max_abs_all": float(np.max(abs(qerr))),
            "max_abs_per_joint": np.max(abs(qerr), axis=0).tolist(),
            "branch_convention": "raw measured angles; no 2pi wrapping",
        },
        "tcp_translation": magnitude_stats(np.linalg.norm(xyzerr, axis=1) * 1000, "mm"),
        "tcp_rotation_geodesic": magnitude_stats(angle * 180 / np.pi, "degrees"),
    }
    if "tcp_nominal_fk" in sim:
        nominal = np.asarray(sim["tcp_nominal_fk"])
        if nominal.shape != sim["tcp"].shape or not np.isfinite(nominal).all():
            raise ValueError(
                "tcp_nominal_fk must be finite (N,6) aligned with simulation t/tcp"
            )
        # Check ALL simulated states, including those beyond reference support:
        # this checks the imported asset against the nominal chain, not footage.
        position = np.linalg.norm(sim["tcp"][:, :3] - nominal[:, :3], axis=1)
        rotation = (
            Rotation.from_rotvec(nominal[:, 3:]).inv()
            * Rotation.from_rotvec(sim["tcp"][:, 3:])
        ).magnitude()
        result["asset_vs_nominal_fk"] = {
            "status": "evaluated",
            "samples": len(st),
            "tcp_translation": magnitude_stats(position * 1000, "mm"),
            "tcp_rotation_geodesic": magnitude_stats(rotation * 180 / np.pi, "degrees"),
            "interpretation": "Imported asset TCP versus independent nominal FK; not a real-camera or factory-calibration test",
        }
    else:
        result["asset_vs_nominal_fk"] = {
            "status": "not_evaluated",
            "reason": "Trace has no tcp_nominal_fk",
        }
    return result


class VideoReader:
    def __init__(self, path):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot decode {path}")
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.last_index = -1
        self.last = None

    def get(self, index):
        index = int(index)
        if index == self.last_index:
            return self.last.copy()
        if index != self.last_index + 1:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {index}")
        self.last_index, self.last = index, frame
        return frame.copy()

    def close(self):
        self.cap.release()


def blue_bin_mask(bgr):
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([90, 100, 35], dtype=np.uint8),
        np.array([135, 255, 255], dtype=np.uint8),
    )
    # Choose largest saturated blue component to remove small indicators/cables.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    chosen = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == chosen


def compare_videos(real, sim, real_video, sim_video, out, fps, tactile_reference=None, *, measure_bin=True):
    import cv2

    rr, sr = VideoReader(real_video), VideoReader(sim_video)
    writer = None
    tactile_writer = None
    tactile = None
    try:
        rt = real["t"]
        if "frame_t" not in sim:
            raise ValueError(
                "Simulation trace needs explicit frame_t for timestamp-correct video comparison"
            )
        st = np.asarray(sim["frame_t"])
        if len(rt) != rr.count or len(st) != sr.count:
            raise ValueError(
                f"Video/trace count mismatch: real {rr.count}/{len(rt)}, sim {sr.count}/{len(st)}"
            )
        if len(st) < 2 or not np.isfinite(st).all() or np.any(np.diff(st) <= 0):
            raise ValueError(
                "Simulation frame_t must be finite and strictly increasing"
            )
        t0, t1 = max(rt[0], st[0]), min(rt[-1], st[-1])
        if t1 <= t0:
            raise ValueError("Videos have no common timestamp interval")
        grid = t0 + np.arange(int(np.floor((t1 - t0) * fps)) + 1) / fps
        ri, si = nearest_indices(rt, grid), nearest_indices(st, grid)
        first = rr.get(ri[0])
        height, width = first.shape[:2]
        tactile_height = 0
        if tactile_reference is not None:
            from tools.sim.tactile_panels import TactilePanels

            tactile = TactilePanels(tactile_reference, sim)
            if not np.isclose(
                float(tactile.data["t0_master"]),
                float(real["t0_master"]),
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError("Tactile/reference t0_master mismatch")
            tactile_height = 300
            tactile_writer = cv2.VideoWriter(
                str(out / "tactile_side_by_side.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width * 2, tactile_height),
            )
            if not tactile_writer.isOpened():
                raise RuntimeError("Cannot initialize tactile MP4 writer")
        writer = cv2.VideoWriter(
            str(out / "side_by_side.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * 2, height + 38 + tactile_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot initialize comparison MP4 writer")
        maes, ious, centroids = [], [], []
        for time, a, b in zip(grid, ri, si):
            reference, rendered = rr.get(a), sr.get(b)
            if reference.shape != rendered.shape:
                raise ValueError(
                    f"Image dimensions differ {reference.shape} versus {rendered.shape}; refusing implicit resize"
                )
            maes.append(
                float(
                    np.abs(
                        reference.astype(np.float32) - rendered.astype(np.float32)
                    ).mean()
                    / 255
                )
            )
            rm, sm = (blue_bin_mask(reference), blue_bin_mask(rendered)) if measure_bin else (
                np.zeros(reference.shape[:2], bool), np.zeros(rendered.shape[:2], bool))
            union = (rm | sm).sum()
            ious.append(float((rm & sm).sum() / union) if union else None)
            if rm.any() and sm.any():
                rc = np.array(np.nonzero(rm)).mean(axis=1)
                sc = np.array(np.nonzero(sm)).mean(axis=1)
                centroids.append(float(np.linalg.norm(rc - sc)))
            else:
                centroids.append(None)
            panel = np.zeros(
                (height + 38 + tactile_height, width * 2, 3), dtype=np.uint8
            )
            panel[38 : 38 + height, :width], panel[38 : 38 + height, width:] = (
                reference,
                rendered,
            )
            if tactile is not None:
                tactile_row = tactile.render(time, scene_width=width, sim_time=st[b])
                panel[38 + height :] = tactile_row
                tactile_writer.write(tactile_row)
            for x, label in (
                (10, "REAL / measured recording"),
                (width + 10, "ISAAC SIM / reconstruction"),
            ):
                cv2.putText(
                    panel,
                    label,
                    (x, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                panel,
                f"t={time:.3f}s",
                (width - 115, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            writer.write(panel)
            if time == grid[0]:
                cv2.imwrite(str(out / "comparison_first.png"), panel)
        np.savez_compressed(
            out / "video_alignment.npz",
            t=grid,
            real_index=ri,
            sim_index=si,
            real_frame_t=rt[ri],
            sim_frame_t=st[si],
            rgb_mae=np.asarray(maes),
            blue_bin_iou=np.array([np.nan if v is None else v for v in ious]),
            blue_bin_centroid_error_px=np.array(
                [np.nan if v is None else v for v in centroids]
            ),
        )
        valid_ious = [v for v in ious if v is not None]
        valid_centroids = [v for v in centroids if v is not None]
        source_snap = (
            real["camera_snap_error_s"][ri]
            if "camera_snap_error_s" in real
            else np.zeros(len(ri))
        )
        result = {
            "frames_compared": len(grid),
            "fps": fps,
            "common_video_interval_s": [float(grid[0]), float(grid[-1])],
            "real_decode_grid_snap_max_s": float(max(abs(rt[ri] - grid))),
            "real_source_camera_snap_max_s": float(max(abs(source_snap))),
            "sim_frame_snap_max_s": float(max(abs(st[si] - grid))),
            "rgb_appearance_mae_0_1": {
                "mean": float(np.mean(maes)),
                "max": float(np.max(maes)),
                "interpretation": "photometric difference; not geometric alignment",
            },
            "blue_bin_silhouette": {
                "mean_iou": float(np.mean(valid_ious)) if valid_ious else None,
                "mean_centroid_error_px": float(np.mean(valid_centroids))
                if valid_centroids
                else None,
                "valid_iou_frames": len(valid_ious),
                "valid_centroid_frames": len(valid_centroids),
                "method": "largest HSV-blue component; hue90..135 saturation>=100 value>=35",
                "interpretation": "uncalibrated static-bin silhouette proxy; does not validate robot, waffle, or depth alignment",
            },
        }
        if tactile is not None:
            result["tactile"] = tactile.save(out)
        if not measure_bin:
            result["blue_bin_silhouette"] = {"status": "not_applicable", "reason": "egg task has tray and holders, no blue bin"}
        return result
    finally:
        rr.close()
        sr.close()
        if writer is not None:
            writer.release()
        if tactile_writer is not None:
            tactile_writer.release()
        if tactile is not None:
            tactile.data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, required=True, help="Prepared episode directory"
    )
    parser.add_argument("--sim-trace", type=Path, required=True)
    parser.add_argument("--sim-video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--tactile-reference",
        type=Path,
        help="Separate native tactile sidecar directory; adds real/proxy panels",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("fps must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    real, sim = np.load(args.reference / "replay.npz"), np.load(args.sim_trace)
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if args.tactile_reference:
        tactile_manifest = json.loads(
            (args.tactile_reference / "manifest.json").read_text()
        )
        if (
            tactile_manifest["reference_replay_sha256"]
            != hashlib.sha256((args.reference / "replay.npz").read_bytes()).hexdigest()
        ):
            raise ValueError(
                "Tactile sidecar was prepared against a different replay file"
            )
    result = {
        "schema_version": 1,
        "episode": manifest["episode"],
        "split": manifest["split"],
        "reference_meta_sha256": manifest["meta_sha256"],
        "sim_trace": str(args.sim_trace.resolve()),
        "state": state_metrics(real, sim),
    }
    video = args.sim_video or args.sim_trace.with_name("sim.mp4")
    if video.exists():
        result["image"] = compare_videos(
            real,
            sim,
            args.reference / "reference.mp4",
            video,
            args.out,
            args.fps,
            args.tactile_reference,
            measure_bin=str(manifest.get("meta", {}).get("task", "")).lower().removesuffix("_fail") != "egg",
        )
    else:
        result["image"] = {
            "status": "not_evaluated",
            "reason": f"Simulation video absent: {video}",
        }
    result["limitations"] = [
        "State replay metrics do not establish closed-loop policy success or physically calibrated contact.",
        "No independent 3D object pose or calibrated per-frame robot landmark ground truth is available.",
        "An image silhouette score is not a substitute for metric camera calibration.",
        "Heldout means excluded from scene fitting; policy training overlap is not audited here.",
    ]
    clock_failed = result["state"]["physics_clock_consistency"]["passed"] is False
    result["clock_check_failed"] = clock_failed
    if clock_failed:
        result["limitations"].insert(
            0,
            "FAILED physics clock consistency: timestamp-based replay metrics are invalid until clock mismatch is fixed.",
        )
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 2 if clock_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
