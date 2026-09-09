#!/usr/bin/env python
"""Export PHANTOM episodes to a LeRobot v3.0 dataset for pi0.5 fine-tuning.

Target format: whatever the INSTALLED `lerobot` package writes (v3.0 codebase
version in lerobot 0.4.4). Nothing here guesses field names -- the frame dict
is validated by `lerobot.datasets.utils.validate_frame` against the `features`
spec we hand to `LeRobotDataset.create`.

Data contract (must match phantom/deploy so an adapter can serve a fine-tuned
pi0.5 through our own policy server):

  frame grid   one LeRobot frame per row of the episode's `actions` stream,
               minus the last (the last row has no successor to supervise).
               `actions` is ALREADY on the control.action_rate_hz grid and is
               the ONLY action stream WindowSampler reads.

  observation.state   (7,) float32 = [tcp x,y,z, rx,ry,rz (rotvec, base frame),
                      gripper aperture] sampled NEAREST to the frame time,
                      exactly like WindowSampler._EpisodeCache.at().

  action              (7,) float32 = the NEXT actions row =
                      [pose_delta(pose_k, pose_k+1) | gripper command at k+1].
                      That is byte-for-byte what WindowSampler puts in
                      `action_chunk` (it reads the same stream with a
                      strict-future index at t0 + (j+1)/rate) and what
                      deploy/executor.py `_pose_at` integrates as
                      t0_pose + cumsum(deltas).

  observation.images.scene   camera_scene_color nearest to the frame time,
                      resized to 224x224 RGB uint8 with the same
                      `phantom.data.windows.bilinear_resize` the training and
                      deploy paths use.

  task                meta.text or meta.task (the WindowSampler fallback).

Episode gating mirrors the training path: the manifest picks the episodes and
their train/val split; `is_trainable_episode` and `needs_rederive` are still
enforced; deliberate failure demos are DROPPED by default because LeRobot has
no per-sample action weight and our pipeline trains them at action_weight 0.

Usage
-----
    python tools/export_lerobot.py \
        --episodes-root data/phantom-episodes \
        --out-root ~/data2/lerobot/phantom_pi05 \
        --repo-id phantom/scene_pi05 \
        --splits train val

Add `--limit 10` for a smoke export.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phantom.data.episode_store import EpisodeReader  # noqa: E402
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ARM_TCP_POSE,  # noqa: E402
                                 STREAM_CAMERA_SCENE, STREAM_GRIPPER,
                                 EpisodeMeta, is_failure_demo,
                                 is_trainable_episode, needs_rederive)
from phantom.data.windows import bilinear_resize  # noqa: E402

log = logging.getLogger("export_lerobot")

STATE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz", "gripper"]
ACTION_NAMES = ["d_x", "d_y", "d_z", "d_rx", "d_ry", "d_rz", "gripper_cmd"]

OBS_IMAGE_KEY = "observation.images.scene"
OBS_STATE_KEY = "observation.state"
ACTION_KEY = "action"


# ---------------------------------------------------------------------------
# sampling helpers (same semantics as WindowSampler._EpisodeCache)
# ---------------------------------------------------------------------------

def nearest_idx(ts: np.ndarray, t: float) -> int:
    """Index of the sample nearest `t` -- WindowSampler.nearest_idx."""
    i = int(np.searchsorted(ts, t))
    if i <= 0:
        return 0
    if i >= len(ts):
        return len(ts) - 1
    return i if (ts[i] - t) < (t - ts[i - 1]) else i - 1


def nearest_idxs(ts: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Vectorised `nearest_idx` over a whole grid."""
    i = np.searchsorted(ts, times)
    i = np.clip(i, 1, len(ts) - 1)
    take_left = (times - ts[i - 1]) <= (ts[i] - times)
    out = np.where(take_left, i - 1, i)
    out[times <= ts[0]] = 0
    out[times >= ts[-1]] = len(ts) - 1
    return out.astype(np.int64)


def resize_scene(img: np.ndarray, size: int, mode: str) -> np.ndarray:
    """(H, W, 3) uint8 -> (size, size, 3) uint8.

    `squash` is the training/deploy convention (WindowSampler._rgb_at resizes
    straight to the backbone resolution and does not pad). `pad` keeps the
    aspect ratio on a black canvas, matching openpi's resize_with_pad, and is
    offered for an ablation -- it is NOT what our own encoder sees.
    """
    img = np.asarray(img)
    if mode == "squash":
        out = bilinear_resize(img.astype(np.float32), (size, size))
    elif mode == "pad":
        h, w = img.shape[:2]
        scale = min(size / h, size / w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        small = bilinear_resize(img.astype(np.float32), (nh, nw))
        out = np.zeros((size, size, 3), dtype=np.float32)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        out[y0:y0 + nh, x0:x0 + nw] = small
    else:
        raise ValueError(f"unknown resize mode {mode!r}")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# per-episode extraction
# ---------------------------------------------------------------------------

class SkipEpisode(Exception):
    """Raised with a human-readable reason; counted, never fatal."""


def episode_frames(ep_dir: Path, *, image_size: int, resize_mode: str) -> dict:
    """Read one episode into aligned arrays. Returns a dict with

        times   (N,)            frame times (t_master seconds)
        state   (N, 7) float32
        action  (N, 7) float32
        images  (N, size, size, 3) uint8   -- lazily produced by `images_iter`
        task    str
        stats   dict of per-episode diagnostics

    N = len(actions) - 1: frame k observes the rig at the k-th action-grid
    tick and is supervised with the delta that moves it to tick k+1.
    """
    reader = EpisodeReader(ep_dir)
    meta = reader.meta

    for stream in (STREAM_ACTIONS, STREAM_ARM_TCP_POSE, STREAM_GRIPPER,
                   STREAM_CAMERA_SCENE):
        if not reader.has(stream):
            raise SkipEpisode(f"missing stream {stream}")

    act_ts = reader.ts(STREAM_ACTIONS)
    actions = np.asarray(reader.data(STREAM_ACTIONS)[:], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise SkipEpisode(f"actions shape {actions.shape} is not (T, 7)")
    if len(act_ts) < 3:
        raise SkipEpisode(f"only {len(act_ts)} action rows")

    tcp_ts = reader.ts(STREAM_ARM_TCP_POSE)
    tcp = np.asarray(reader.data(STREAM_ARM_TCP_POSE)[:], dtype=np.float64)
    grip_ts = reader.ts(STREAM_GRIPPER)
    grip = np.asarray(reader.data(STREAM_GRIPPER)[:], dtype=np.float32)
    cam_ts = reader.ts(STREAM_CAMERA_SCENE)

    # frame k <- action-grid tick k; supervised by the action AT tick k+1
    times = act_ts[:-1]
    action = actions[1:]
    n = len(times)

    # the camera must actually cover the grid, or half the episode would be
    # served the same clamped frame
    lo, hi = float(cam_ts[0]), float(cam_ts[-1])
    inside = (times >= lo) & (times <= hi)
    if inside.sum() < 2:
        raise SkipEpisode("scene camera does not overlap the action grid")
    # `times` is sorted and the camera span is an interval, so the kept rows
    # are contiguous: record the slice so a downstream check can map frame k
    # back to action-stream row act_lo + k.
    act_lo = int(np.argmax(inside))
    act_hi = act_lo + int(inside.sum())
    if inside.sum() < n:
        times = times[inside]
        action = action[inside]
        n = len(times)

    tcp_i = nearest_idxs(tcp_ts, times)
    grip_i = nearest_idxs(grip_ts, times)
    cam_i = nearest_idxs(cam_ts, times)

    state = np.concatenate([tcp[tcp_i], grip[grip_i, :1]], axis=1).astype(np.float32)

    dt = np.diff(act_ts)
    stats = {
        "n_frames": int(n),
        "n_action_rows": int(len(actions)),
        # frame k came from actions.zarr row act_lo + k (observation) and is
        # supervised by row act_lo + k + 1 (action)
        "act_index_lo": act_lo,
        "act_index_hi": act_hi,
        "first_frame_t": round(float(times[0]), 6),
        "duration_s": round(float(act_ts[-1] - act_ts[0]), 3),
        "action_dt_mean_s": round(float(dt.mean()), 5),
        "action_dt_max_s": round(float(dt.max()), 5),
        "action_rate_hz": round(float(1.0 / dt.mean()), 3),
        "camera_rate_hz": round(float((len(cam_ts) - 1) / (cam_ts[-1] - cam_ts[0])), 3),
        "max_obs_camera_lag_s": round(float(np.abs(cam_ts[cam_i] - times).max()), 4),
        "max_obs_tcp_lag_s": round(float(np.abs(tcp_ts[tcp_i] - times).max()), 4),
        "clipped_to_camera_span": bool(inside.sum() < len(act_ts) - 1),
    }

    return {
        "times": times,
        "state": state,
        "action": action.astype(np.float32),
        "cam_idx": cam_i,
        "reader": reader,
        "task": (meta.text or meta.task),
        "meta": meta,
        "stats": stats,
        "image_size": image_size,
        "resize_mode": resize_mode,
    }


def images_iter(ep: dict):
    """Decode + resize scene frames one at a time (a 480x640x3 episode is
    ~130 MB decoded; the whole export must not hold one in RAM per worker)."""
    data = ep["reader"].data(STREAM_CAMERA_SCENE)
    size, mode = ep["image_size"], ep["resize_mode"]
    last_i, last_img = -1, None
    for i in ep["cam_idx"]:
        i = int(i)
        if i != last_i:
            last_img = resize_scene(np.asarray(data[i]), size, mode)
            last_i = i
        yield last_img


# ---------------------------------------------------------------------------
# reconstruction check
# ---------------------------------------------------------------------------

def reconstruction_error(ep: dict) -> dict:
    """Integrate the exported actions the way deploy/executor.py `_pose_at`
    does (t0_pose + cumsum of deltas) and compare with the recorded TCP path.

    Returns max/mean translation error in mm and rotation error in degrees.
    """
    reader = ep["reader"]
    tcp_ts = reader.ts(STREAM_ARM_TCP_POSE)
    tcp = np.asarray(reader.data(STREAM_ARM_TCP_POSE)[:], dtype=np.float64)

    t0_pose = ep["state"][0, :6].astype(np.float64)
    pred = t0_pose + np.cumsum(ep["action"][:, :6].astype(np.float64), axis=0)

    # the delta at frame k lands the arm on tick k+1, i.e. at times[k+1]
    target_t = np.concatenate([ep["times"][1:], ep["times"][-1:] + ep["stats"]["action_dt_mean_s"]])
    ref = tcp[nearest_idxs(tcp_ts, target_t)]

    trans_mm = np.linalg.norm(pred[:, :3] - ref[:, :3], axis=1) * 1000.0
    rot_deg = np.degrees(np.linalg.norm(pred[:, 3:] - ref[:, 3:], axis=1))
    return {
        "n": int(len(pred)),
        "trans_max_mm": round(float(trans_mm.max()), 4),
        "trans_mean_mm": round(float(trans_mm.mean()), 4),
        "rot_max_deg": round(float(rot_deg.max()), 4),
        "rot_mean_deg": round(float(rot_deg.mean()), 4),
    }


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# main export
# ---------------------------------------------------------------------------

def build_features(image_size: int, use_videos: bool) -> dict:
    return {
        OBS_IMAGE_KEY: {
            "dtype": "video" if use_videos else "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        OBS_STATE_KEY: {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        ACTION_KEY: {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }


def export_split(rows: list[dict], *, episodes_root: Path, out_root: Path,
                 repo_id: str, args) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_root} already exists (pass --overwrite)")
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        features=build_features(args.image_size, not args.no_videos),
        root=out_root,
        robot_type=args.robot_type,
        use_videos=not args.no_videos,
        vcodec=args.vcodec,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
    )

    exported: list[dict] = []
    skipped: list[dict] = []
    recon_checks: list[dict] = []
    t_start = time.time()

    for row in rows:
        ep_dir = episodes_root / row["path"]
        name = row["episode"]
        try:
            meta = EpisodeMeta.load(ep_dir / "meta.json")
        except Exception as e:
            skipped.append({"episode": name, "reason": f"unreadable meta.json: {e}"})
            continue
        if not is_trainable_episode(meta):
            skipped.append({"episode": name, "reason": "not is_trainable_episode"})
            continue
        if needs_rederive(ep_dir, meta):
            skipped.append({"episode": name, "reason": "policy rollout without re-derived actions"})
            continue
        if is_failure_demo(meta) and not args.include_failure_demos:
            skipped.append({"episode": name, "reason": "deliberate failure demo (action_weight 0)"})
            continue

        try:
            ep = episode_frames(ep_dir, image_size=args.image_size,
                                resize_mode=args.resize_mode)
        except SkipEpisode as e:
            skipped.append({"episode": name, "reason": str(e)})
            continue
        except Exception as e:                       # corrupt zarr, short stream
            skipped.append({"episode": name, "reason": f"{type(e).__name__}: {e}"})
            continue

        ep_index = ds.meta.total_episodes
        task = ep["task"]
        for img, st, ac in zip(images_iter(ep), ep["state"], ep["action"], strict=True):
            ds.add_frame({
                OBS_IMAGE_KEY: img,
                OBS_STATE_KEY: np.asarray(st, dtype=np.float32),
                ACTION_KEY: np.asarray(ac, dtype=np.float32),
                "task": task,
            })
        ds.save_episode()

        if len(recon_checks) < args.recon_checks:
            recon_checks.append({"episode": name, **reconstruction_error(ep)})

        exported.append({
            "episode_index": ep_index,
            "episode": name,
            "path": row["path"],
            "task_name": row.get("task"),
            "task_text": task,
            "split": row.get("split"),
            "success": row.get("success"),
            "failure_demo": row.get("failure_demo"),
            "session": row.get("session"),
            "operator": row.get("operator"),
            **ep["stats"],
        })
        if len(exported) % 25 == 0:
            el = time.time() - t_start
            log.info("%s: %d/%d episodes, %d frames, %.1f s elapsed (%.2f s/ep)",
                     repo_id, len(exported), len(rows),
                     sum(e["n_frames"] for e in exported), el, el / len(exported))

    ds.finalize()

    summary = {
        "repo_id": repo_id,
        "root": str(out_root),
        "n_manifest_rows": len(rows),
        "n_episodes": len(exported),
        "n_skipped": len(skipped),
        "n_frames": int(sum(e["n_frames"] for e in exported)),
        "total_duration_s": round(float(sum(e["duration_s"] for e in exported)), 1),
        "elapsed_s": round(time.time() - t_start, 1),
        "reconstruction_checks": recon_checks,
    }
    (out_root / "phantom_episodes.json").write_text(json.dumps(exported, indent=1))
    (out_root / "phantom_skipped.json").write_text(json.dumps(skipped, indent=1))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes-root", type=Path, required=True,
                   help="data/phantom-episodes (the dir holding tasks/ and manifests/)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="default <episodes-root>/manifests/all.jsonl")
    p.add_argument("--out-root", type=Path, required=True,
                   help="parent dir; one subdir per split is created under it")
    p.add_argument("--repo-id", default="phantom/scene_pi05")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--limit", type=int, default=0,
                   help="cap episodes PER SPLIT (0 = all) -- the smoke export")
    p.add_argument("--fps", type=int, default=10,
                   help="NOMINAL fps written into the dataset; must equal "
                        "hardware control.action_rate_hz")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--resize-mode", choices=["squash", "pad"], default="squash")
    p.add_argument("--robot-type", default="ur5e_phantom")
    p.add_argument("--include-failure-demos", action="store_true",
                   help="keep deliberate failure demos (our own trainer zeroes "
                        "their action loss; LeRobot cannot, so they would be "
                        "imitated)")
    p.add_argument("--no-videos", action="store_true",
                   help="store PNG frames instead of encoded video (much bigger)")
    p.add_argument("--vcodec", default="libsvtav1",
                   help="h264 / hevc / libsvtav1 / auto")
    p.add_argument("--batch-encoding-size", type=int, default=1)
    p.add_argument("--image-writer-processes", type=int, default=0)
    p.add_argument("--image-writer-threads", type=int, default=8)
    p.add_argument("--recon-checks", type=int, default=3,
                   help="episodes per split to run the cumsum-vs-TCP check on")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    manifest = args.manifest or (args.episodes_root / "manifests" / "all.jsonl")
    rows = load_manifest(manifest)
    log.info("manifest %s: %d rows", manifest, len(rows))

    import lerobot
    from importlib.metadata import version

    out_root = args.out_root.expanduser()
    summaries = {}
    for split in args.splits:
        srows = [r for r in rows if r.get("split") == split]
        if args.limit:
            srows = srows[:args.limit]
        if not srows:
            log.warning("split %r: no manifest rows", split)
            continue
        log.info("split %r: %d manifest rows", split, len(srows))
        summaries[split] = export_split(
            srows,
            episodes_root=args.episodes_root.expanduser(),
            out_root=out_root / split,
            repo_id=f"{args.repo_id}_{split}",
            args=args,
        )
        log.info("split %r done: %s", split, json.dumps(summaries[split]["reconstruction_checks"]))

    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lerobot_version": version("lerobot"),
        "lerobot_path": lerobot.__file__,
        "phantom_commit": git_commit(REPO_ROOT),
        "argv": sys.argv,
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "action_convention": (
            "frame k: observation at action-grid tick k; action = actions.zarr "
            "row k+1 = [pose_delta(pose_k, pose_k+1) | gripper command at k+1]. "
            "Integrate as t0_pose + cumsum(action[:, :6]) (deploy/executor.py "
            "_pose_at)."),
        "splits": summaries,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "phantom_export_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
