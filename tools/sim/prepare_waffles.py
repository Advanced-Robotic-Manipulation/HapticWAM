#!/usr/bin/env python3
"""Read-only PHANTOM episode export for Isaac replay, with explicit provenance.

Run using PHANTOM's recording environment (numpy, scipy, zarr, OpenCV):
  .venv/bin/python tools/sim/prepare_waffles.py --data-root ../data --out ../sim_evidence

Each export contains replay.npz, reference.mp4, nine diagnostic PNGs and a
manifest. ``t`` is a uniform replay grid relative to ``t0_master``; q/qd/tcp
are measured feedback interpolated on that grid, never recorded actions.
TCP rotations are axis-angle radians and use SO(3) SLERP. Gripper discrete
object state is held from the previous feedback sample. Native sample data
and timestamps are retained under ``native_<stream>``/``native_<stream>_t``.
Camera nearest-neighbor indices and signed snap errors are recorded. Zarr
timestamps are ALREADY master-clock seconds: metadata offsets are preserved,
not applied again. Fit and held-out memberships are fixed before fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Allows invocation by path without installing the repository.
for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "phantom/data/episode_store.py").is_file():
        sys.path.insert(0, str(ancestor))
        break

DEFAULT_EPISODES = {
    "fit": [
        "episodes/deploy/20260904/ep_teacher_waffles_1788535016_005",
        "full/tasks/waffles/ep_waffles_1787395928_000",
    ],
    "heldout": [
        "episodes/deploy/20260904/ep_teacher_waffles_1788535066_006",
        "episodes/deploy/20260904/ep_teacher_waffles_1788538100_000",
        "episodes/deploy/20260901/ep_teacher_waffles_1788262056_000",
    ],
}
NUMERIC_STREAMS = (
    "arm_q",
    "arm_qd",
    "arm_tcp_pose",
    "arm_tcp_speed",
    "gripper",
    "arm_ft",
    "tactile_left_wrench",
    "tactile_right_wrench",
    "tactile_left_area",
    "tactile_right_area",
    "actions",
    "actions_plan",
    "actions_abs",
    "actions_qtarget",
)


def nearest_indices(ts: np.ndarray, times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(ts, times).clip(0, len(ts) - 1)
    left = (right - 1).clip(0, len(ts) - 1)
    return np.where(abs(times - ts[left]) <= abs(ts[right] - times), left, right)


def checked_timestamps(ts: np.ndarray, name: str) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all():
        raise ValueError(f"{name}: need at least two finite timestamps")
    if np.any(np.diff(ts) <= 0):
        raise ValueError(
            f"{name}: non-monotonic/duplicate timestamps; refusing silent repair"
        )
    return ts


def interpolate(ts: np.ndarray, values: np.ndarray, times: np.ndarray) -> np.ndarray:
    # np.interp deliberately retains the measured joint branch (no wrapping).
    values = np.asarray(values)
    flat = values.reshape(len(values), -1)
    result = np.stack(
        [np.interp(times, ts, flat[:, i]) for i in range(flat.shape[1])], axis=-1
    )
    return result.reshape((len(times),) + values.shape[1:])


def export_episode(
    episode: Path, out: Path, split: str, fps: float, max_seconds: float | None = None
) -> dict:
    import cv2
    from scipy.spatial.transform import Rotation, Slerp

    from phantom.data.episode_store import EpisodeReader

    episode = episode.resolve()
    out = out.resolve()
    if out == episode or episode in out.parents:
        raise ValueError("Output must be outside source episode")
    if (out / "manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing evidence export: {out}"
        )
    reader = EpisodeReader(episode)
    meta_bytes = (episode / "meta.json").read_bytes()
    meta = json.loads(meta_bytes)
    if str(meta.get("task", "")).lower() not in ("waffles", "waffles_fail"):
        raise ValueError(f"Not a waffles episode: {episode}")
    modes = meta.get("driver_modes", {})
    if modes.get("drivers") != "real" or any(
        v != "real" for v in modes.get("overrides", {}).values()
    ):
        raise ValueError(
            f"Replay truth requires explicit all-real driver provenance: {modes}"
        )
    required = ("arm_q", "arm_qd", "arm_tcp_pose", "gripper", "camera_scene_color")
    missing = [name for name in required if not reader.has(name)]
    if missing:
        raise ValueError(f"Missing required streams in {episode}: {missing}")
    stamps = {
        name: checked_timestamps(reader.ts(name), name)
        for name in reader.streams()
        if name in NUMERIC_STREAMS or name.startswith("camera_")
    }
    # Only render inside common observed support; never endpoint-extrapolate.
    t0 = max(stamps[s][0] for s in required)
    end = min(stamps[s][-1] for s in required)
    if max_seconds is not None:
        end = min(end, t0 + max_seconds)
    if end <= t0:
        raise ValueError("Required streams have no common time interval")
    grid = t0 + np.arange(int(np.floor((end - t0) * fps)) + 1, dtype=np.float64) / fps
    arrays = {"t": grid - t0, "t0_master": np.asarray(t0), "fps": np.asarray(fps)}
    stream_summary = {}
    for name, ts in stamps.items():
        delta = np.diff(ts)
        stream_summary[name] = {
            "samples": len(ts),
            "first_master_s": float(ts[0]),
            "last_master_s": float(ts[-1]),
            "median_hz": float(1 / np.median(delta)),
            "max_gap_s": float(delta.max()),
        }
        arrays[f"native_{name}_t"] = ts - t0
        if name in NUMERIC_STREAMS:
            values = np.asarray(reader.data(name)[:])
            if len(values) != len(ts) or not np.isfinite(values).all():
                raise ValueError(f"{name}: length mismatch or nonfinite samples")
            arrays[f"native_{name}"] = values
    for source, key in (("arm_q", "q"), ("arm_qd", "qd"), ("arm_tcp_pose", "tcp")):
        values = arrays[f"native_{source}"]
        if values.shape != (len(stamps[source]), 6):
            raise ValueError(f"{source}: expected (N,6), got {values.shape}")
        arrays[key] = interpolate(stamps[source], values, grid)
    arrays["tcp"][:, 3:] = Slerp(
        stamps["arm_tcp_pose"],
        Rotation.from_rotvec(arrays["native_arm_tcp_pose"][:, 3:]),
    )(grid).as_rotvec()
    grip_i = (np.searchsorted(stamps["gripper"], grid, side="right") - 1).clip(0)
    arrays["gripper"] = arrays["native_gripper"][grip_i]
    ct = stamps["camera_scene_color"]
    ci = nearest_indices(ct, grid)
    arrays["camera_index"] = ci
    arrays["camera_t"] = ct[ci] - t0
    arrays["camera_snap_error_s"] = ct[ci] - grid
    arrays["camera_q"] = interpolate(stamps["arm_q"], arrays["native_arm_q"], ct[ci])
    arrays["joint_names"] = np.asarray(
        [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "replay.npz", **arrays)
    frame_data = reader.data("camera_scene_color")
    first = frame_data[int(ci[0])]
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out / "reference.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open MP4 encoder")
    sample_rows = np.unique(np.linspace(0, len(grid) - 1, 9).astype(int))
    samples, tiles = [], []
    try:
        for j, index in enumerate(ci):
            rgb = frame_data[int(index)]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            if j in sample_rows:
                filename = f"frame_{j:05d}.png"
                cv2.imwrite(str(out / filename), bgr)
                samples.append(
                    {
                        "file": filename,
                        "grid_index": j,
                        "source_index": int(index),
                        "t": float(grid[j] - t0),
                        "camera_t": float(ct[index] - t0),
                        "q": arrays["camera_q"][j].tolist(),
                        "tcp": arrays["tcp"][j].tolist(),
                    }
                )
                tile = cv2.resize(bgr, (320, 240))
                cv2.putText(
                    tile,
                    f"{split} t={grid[j] - t0:.2f}s",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
    finally:
        writer.release()
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(
        str(out / "contact_sheet.jpg"),
        np.concatenate(
            [np.concatenate(tiles[i : i + 3], axis=1) for i in range(0, len(tiles), 3)],
            axis=0,
        ),
    )
    trace = episode / "planner_trace.json"
    manifest = {
        "schema_version": 1,
        "episode": str(episode),
        "split": split,
        "meta_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "meta": meta,
        "planner_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()
        if trace.exists()
        else None,
        "t0_master_s": float(t0),
        "fps": fps,
        "frames": len(grid),
        "duration_s": float(grid[-1] - grid[0]),
        "resolution_wh": [width, height],
        "streams": stream_summary,
        "samples": samples,
        "source_meta_unchanged": (episode / "meta.json").read_bytes() == meta_bytes,
        "source_zarr_open_mode": "r",
        "synchronization": {
            "timestamps": "Zarr ts already in MasterClock domain; no second clock-offset application",
            "grid": "intersection of measured q, qd, TCP, gripper and RGB timestamp support",
            "arm": "linear interpolation of measured q/qd/xyz; SO(3) SLERP of measured TCP orientation",
            "gripper": "previous recorded feedback sample; position closed fraction and categorical object state",
            "rgb": "nearest recorded frame with explicit signed timestamp residual",
            "camera_snap_abs_max_s": float(np.max(abs(arrays["camera_snap_error_s"]))),
        },
        "warnings": [
            "Recorded actions are not replay truth; proposals and rederived arrays are retained only for audit.",
            "Rule-derived success labels are not independent operator/lab outcome validation.",
            "Frames use recorded RGB; camera calibration and object dimensions need separate estimation.",
            "Held-out here means excluded from scene fitting, not excluded from policy training.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "episode": episode.name,
                "split": split,
                "frames": len(grid),
                "out": str(out),
            }
        ),
        flush=True,
    )
    return manifest


def inventory(data_root: Path) -> list[dict]:
    records = []
    for parent in (data_root / "episodes/deploy", data_root / "full/tasks/waffles"):
        if not parent.exists():
            continue
        for filename in sorted(parent.rglob("ep_*/meta.json")):
            meta = json.loads(filename.read_text())
            if "waffles" not in str(meta.get("task", "")).lower():
                continue
            streams = sorted(p.stem for p in filename.parent.glob("*.zarr"))
            records.append(
                {
                    "episode": str(filename.parent),
                    "status": meta.get("status"),
                    "success": meta.get("success"),
                    "tags": meta.get("tags", []),
                    "driver_modes": meta.get("driver_modes", {}),
                    "meta_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                    "streams": streams,
                    "complete_motion_rgb": all(
                        s in streams
                        for s in (
                            "arm_q",
                            "arm_qd",
                            "arm_tcp_pose",
                            "gripper",
                            "camera_scene_color",
                        )
                    ),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/physicalai/phantom-icra-2027/data"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--episode", type=Path, help="Export just this source episode")
    parser.add_argument("--split", choices=("fit", "heldout"), default="heldout")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--tactile-only",
        action="store_true",
        help="Write separate native tactile sidecars; leave existing evidence unchanged",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="Existing evidence root; with --episode, that episode's prepared directory",
    )
    args = parser.parse_args()
    if not 0 < args.fps <= 125:
        parser.error("--fps must be in (0, 125]")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.tactile_only:
        if args.prepared_root is None:
            parser.error(
                "--tactile-only requires --prepared-root for the original t0_master"
            )
        from tools.sim.tactile_panels import export_tactile_sidecar

        if args.episode:
            export_tactile_sidecar(args.episode, args.prepared_root, args.out)
        else:
            for split, paths in DEFAULT_EPISODES.items():
                for relative in paths:
                    episode = args.data_root / relative
                    export_tactile_sidecar(
                        episode,
                        args.prepared_root / split / episode.name,
                        args.out / split / episode.name,
                    )
        return
    if args.episode:
        export_episode(args.episode, args.out, args.split, args.fps, args.max_seconds)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    records = inventory(args.data_root)
    (args.out / "inventory.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "split.json").write_text(json.dumps(DEFAULT_EPISODES, indent=2) + "\n")
    if not args.inventory_only:
        for split, paths in DEFAULT_EPISODES.items():
            for relative in paths:
                path = args.data_root / relative
                export_episode(
                    path,
                    args.out / split / path.name,
                    split,
                    args.fps,
                    args.max_seconds,
                )


if __name__ == "__main__":
    main()
