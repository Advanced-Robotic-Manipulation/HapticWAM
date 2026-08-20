"""Regenerate configs/start_poses.yaml from a dataset root (first frame of
arm_tcp_pose.zarr + gripper.zarr per success episode).

    python tools/gen_start_poses.py /path/to/phantom-episodes/tasks

Run on the box that holds the FULL dataset (compute). Success tasks only —
*_fail episodes are never deployed.
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import zarr

TASKS = ("Carton", "egg", "waffles", "whiteboard")


def main(root: str) -> int:
    lines = [
        "# Per-task demo START-pose distributions - generated from the FULL dataset",
        "# (first frame of arm_tcp_pose.zarr/gripper.zarr, success episodes only).",
        "# Regenerate with tools/gen_start_poses.py after any dataset change.",
        "# tcp: [x, y, z, rx, ry, rz]  (m / axis-angle rad, UR base frame)",
        f"generated: {time.strftime('%Y-%m-%d')}",
        f"source: {root}",
        "tasks:",
    ]
    for task in TASKS:
        eps = sorted(glob.glob(f"{root}/{task}/ep_*"))
        poses, grips = [], []
        for e in eps:
            try:
                poses.append(np.asarray(zarr.open(e + "/arm_tcp_pose.zarr")["data"][0]))
                grips.append(float(np.ravel(zarr.open(e + "/gripper.zarr")["data"][0])[0]))
            except Exception as ex:            # noqa: BLE001 — skip broken episode, keep stats honest
                print(f"skip {e}: {ex}", file=sys.stderr)
        P, G = np.array(poses), np.array(grips)
        assert len(P) >= 20, f"{task}: only {len(P)} episodes — refusing thin stats"
        fmt = lambda a: "[" + ", ".join(f"{x:.4f}" for x in a) + "]"  # noqa: E731
        lines += [
            f"  {task}:",
            f"    n: {len(P)}",
            f"    tcp_mean: {fmt(P.mean(0))}",
            f"    tcp_std:  {fmt(P.std(0))}",
            f"    gripper_mean: {G.mean():.3f}",
            f"    gripper_std: {G.std():.3f}",
        ]
    out = Path(__file__).resolve().parents[1] / "configs" / "start_poses.yaml"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
