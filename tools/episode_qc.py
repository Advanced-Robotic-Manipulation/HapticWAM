"""Per-episode QC for a collection session (recovery-demo sessions included).

    python tools/episode_qc.py <dir-with-ep_*> [--task waffles]

Reports per episode: labels (success/failure_demo/tags), duration, start TCP
and its sigma vs the task's demo start distribution (configs/start_poses.yaml
— for RECOVERY demos the start SHOULD be 1.5-4 sigma off), gripper start
opening + first close + OBJ==2 fraction (did it actually hold something),
camera mean brightness (lighting drift vs ~90-135 in the v3 demos), tactile
stream health. Exit 1 if any episode has malformed streams.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import zarr


def qc_episode(ep: Path, stats) -> dict:
    meta = json.loads((ep / "meta.json").read_text())
    out = {"episode": ep.name, "task": meta.get("task"),
           "success": meta.get("success"), "failure_demo": meta.get("failure_demo"),
           "tags": meta.get("tags"), "operator": meta.get("operator")}
    try:
        p = np.asarray(zarr.open(str(ep / "arm_tcp_pose.zarr"))["data"])
        ts = np.asarray(zarr.open(str(ep / "arm_tcp_pose.zarr"))["ts"])
        g = np.asarray(zarr.open(str(ep / "gripper.zarr"))["data"])
        out["duration_s"] = round(float(ts[-1] - ts[0]), 1)
        out["start_xyz_mm"] = [round(float(x) * 1000) for x in p[0, :3]]
        if stats is not None:
            std = np.where(stats.tcp_std > 1e-9, stats.tcp_std, np.inf)
            sig = np.abs((p[0] - stats.tcp_mean) / std)
            out["start_sigma_xyz"] = [round(float(x), 1) for x in sig[:3]]
        out["grip_start"] = round(float(g[0, 0]), 2)
        out["grip_max"] = round(float(g[:, 0].max()), 2)
        out["obj2_frac"] = round(float((g[:, 1] == 2).mean()), 2)
        hit = np.nonzero((g[:, 0] > 0.45)
                         & (g[:, 0] - np.minimum.accumulate(g[:, 0]) > 0.15))[0]
        out["closes"] = bool(len(hit))
        try:
            import io
            import imageio.v2 as iio
            z = zarr.open(str(ep / "camera_scene_color.zarr"))["data"]
            f = iio.imread(io.BytesIO(bytes(z[min(3, len(z) - 1)])))
            out["cam_brightness"] = round(float(np.asarray(f).mean()), 1)
        except Exception as e:                            # noqa: BLE001
            out["cam_brightness"] = f"ERR {e}"
        for side in ("left", "right"):
            try:
                d = zarr.open(str(ep / f"tactile_{side}_fields_ds.zarr"))["data"]
                out[f"tactile_{side}_n"] = len(d)
            except Exception:                             # noqa: BLE001
                out[f"tactile_{side}_n"] = 0
    except Exception as e:                                # noqa: BLE001
        out["MALFORMED"] = str(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--task", default=None,
                    help="task whose start distribution to compare against "
                         "(default: each episode's own meta task)")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from phantom.deploy.start_pose import load_start_stats
    all_stats = load_start_stats()
    eps = sorted(Path(p) for p in glob.glob(str(Path(args.root) / "ep_*")))
    if not eps:
        eps = sorted(Path(p) for p in glob.glob(str(Path(args.root) / "*" / "ep_*")))
    rows, bad = [], 0
    for ep in eps:
        task = args.task or json.loads((ep / "meta.json").read_text()).get("task", "")
        st = all_stats.get(task.removesuffix("_fail"))
        r = qc_episode(ep, st)
        rows.append(r)
        bad += "MALFORMED" in r
        print(json.dumps(r))
    n = len(rows)
    print(f"\n== {n} episodes, {bad} malformed", file=sys.stderr)
    if n:
        lab = sum(1 for r in rows if r.get("success") is not None or r.get("failure_demo"))
        held = sum(1 for r in rows if isinstance(r.get("obj2_frac"), float) and r["obj2_frac"] > 0)
        print(f"   labeled: {lab}/{n} | held object (OBJ2>0): {held}/{n}", file=sys.stderr)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
