#!/usr/bin/env python
"""Validate a LeRobot export produced by tools/export_lerobot.py.

Loads the dataset back through `lerobot.datasets.LeRobotDataset` (so the
parquet + encoded video path is exercised, not just the writer's buffers) and
checks, per episode:

  * feature shapes / dtypes / fps / episode + frame counts;
  * pi0.5 readiness -- an `observation.images.*` key, `observation.state`,
    `action`, and a non-empty `task` string on every sample;
  * that integrating the loaded actions the way deploy/executor.py `_pose_at`
    does (t0_pose + cumsum of the 6-D deltas) reproduces the recorded TCP path
    of the source episode, reported as max error in mm and degrees.

Usage:
    python tools/validate_lerobot_export.py \
        --root ~/data2/lerobot/phantom_pi05_smoke/train \
        --episodes-root data/phantom-episodes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phantom.data.episode_store import EpisodeReader  # noqa: E402
from phantom.data.schema import STREAM_ACTIONS, STREAM_ARM_TCP_POSE  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="the split dir")
    p.add_argument("--episodes-root", type=Path, default=None,
                   help="source episodes root; enables the TCP reconstruction check")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--n-recon", type=int, default=5)
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.root.expanduser()
    info = json.loads((root / "meta" / "info.json").read_text())
    repo_id = args.repo_id or info.get("repo_id") or "local/dataset"
    ds = LeRobotDataset(repo_id=repo_id, root=root)

    print("=" * 72)
    print(f"root              {root}")
    print(f"codebase_version  {info.get('codebase_version')}")
    print(f"robot_type        {ds.meta.robot_type}")
    print(f"fps               {ds.fps}")
    print(f"episodes          {ds.meta.total_episodes}")
    print(f"frames            {ds.meta.total_frames}")
    print(f"tasks             {ds.meta.total_tasks}")
    print(f"video keys        {list(ds.meta.video_keys)}")
    print(f"camera keys       {list(ds.meta.camera_keys)}")
    print("features:")
    for k, ft in ds.meta.features.items():
        print(f"  {k:32s} dtype={ft['dtype']:8s} shape={tuple(ft['shape'])} names={ft.get('names')}")

    sample = ds[0]
    print("sample[0]:")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k:32s} {tuple(v.shape)} {v.dtype}"
                  f"{'' if v.numel() > 8 else '  ' + str(v.tolist())}")
        else:
            print(f"  {k:32s} {v!r}")

    problems: list[str] = []
    img_keys = [k for k in ds.meta.features if k.startswith("observation.images.")]
    if not img_keys:
        problems.append("no observation.images.* key -- pi0.5 needs at least one camera")
    for key, want in (("observation.state", 7), ("action", 7)):
        if key not in ds.meta.features:
            problems.append(f"missing feature {key}")
        elif tuple(ds.meta.features[key]["shape"]) != (want,):
            problems.append(f"{key} shape {ds.meta.features[key]['shape']} != ({want},)")
    if not isinstance(sample.get("task"), str) or not sample.get("task"):
        problems.append("sample['task'] is not a non-empty string")
    for key in img_keys:
        v = sample[key]
        if v.ndim != 3 or v.shape[0] != 3:
            problems.append(f"{key} sample is {tuple(v.shape)}, expected (3, H, W)")
        elif float(v.min()) < 0.0 or float(v.max()) > 1.0:
            problems.append(f"{key} values outside [0, 1]: [{float(v.min())}, {float(v.max())}]")

    # a few random samples must all carry a task string
    rng = np.random.default_rng(0)
    for i in rng.integers(0, len(ds), size=min(32, len(ds))):
        if not ds[int(i)].get("task"):
            problems.append(f"sample {int(i)} has an empty task")
            break

    # ---- reconstruction from the LOADED dataset --------------------------
    recon = []
    if args.episodes_root is not None:
        eps = json.loads((root / "phantom_episodes.json").read_text())
        froms = ds.meta.episodes["dataset_from_index"]
        tos = ds.meta.episodes["dataset_to_index"]
        for rec in eps[:args.n_recon]:
            ei = rec["episode_index"]
            lo, hi = int(froms[ei]), int(tos[ei])
            acts = np.stack([ds[i]["action"].numpy() for i in range(lo, hi)]).astype(np.float64)
            st0 = ds[lo]["observation.state"].numpy().astype(np.float64)

            src = EpisodeReader(args.episodes_root.expanduser() / rec["path"])
            act_ts = src.ts(STREAM_ACTIONS)
            tcp_ts = src.ts(STREAM_ARM_TCP_POSE)
            tcp = np.asarray(src.data(STREAM_ARM_TCP_POSE)[:], dtype=np.float64)

            n = hi - lo
            # frame k came from actions row act_lo + k; the delta it carries
            # lands the arm on row act_lo + k + 1
            act_lo = int(rec.get("act_index_lo", 0))
            target_t = act_ts[act_lo + 1:act_lo + n + 1]
            if len(target_t) != n:
                problems.append(f"{rec['episode']}: act_index_lo bookkeeping mismatch")
                continue
            idx = np.clip(np.searchsorted(tcp_ts, target_t), 1, len(tcp_ts) - 1)
            take_left = (target_t - tcp_ts[idx - 1]) <= (tcp_ts[idx] - target_t)
            idx = np.where(take_left, idx - 1, idx)
            ref = tcp[idx]

            pred = st0[:6] + np.cumsum(acts[:, :6], axis=0)
            trans_mm = np.linalg.norm(pred[:, :3] - ref[:, :3], axis=1) * 1000.0
            rot_deg = np.degrees(np.linalg.norm(pred[:, 3:] - ref[:, 3:], axis=1))
            recon.append({
                "episode": rec["episode"], "n": int(n),
                "trans_max_mm": round(float(trans_mm.max()), 4),
                "trans_mean_mm": round(float(trans_mm.mean()), 4),
                "rot_max_deg": round(float(rot_deg.max()), 4),
            })
        print("reconstruction (t0_pose + cumsum(action[:, :6]) vs recorded TCP):")
        for r in recon:
            print(f"  {r['episode']:34s} n={r['n']:4d} "
                  f"trans_max={r['trans_max_mm']:.3f} mm  "
                  f"mean={r['trans_mean_mm']:.3f} mm  "
                  f"rot_max={r['rot_max_deg']:.4f} deg")
        # The residual is NOT an export artefact: `actions` rows are deltas
        # between TCP poses the recorder read at its own tick, while the
        # reference here is the 125 Hz TCP stream sampled nearest to the
        # nominal grid time. Sub-tick lag at ~100 mm/s is a few mm at the
        # peak of a fast reach. Gate on the mean, warn on the peak.
        worst = max((r["trans_max_mm"] for r in recon), default=0.0)
        worst_mean = max((r["trans_mean_mm"] for r in recon), default=0.0)
        if worst_mean > 1.0:
            problems.append(f"cumsum reconstruction mean error {worst_mean:.3f} mm (> 1 mm)")
        if worst > 10.0:
            problems.append(f"cumsum reconstruction peak error {worst:.3f} mm (> 10 mm)")

    print("=" * 72)
    if problems:
        for pr in problems:
            print(f"FAIL: {pr}")
        raise SystemExit(1)
    print("OK: export is loadable and pi0.5-shaped")


if __name__ == "__main__":
    main()
