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
        "# q_mean/q_std: start joint vector (rad); tcp_z_min: lowest demo TCP z (m)",
        f"generated: {time.strftime('%Y-%m-%d')}",
        f"source: {root}",
        "tasks:",
    ]
    for task in TASKS:
        eps = sorted(glob.glob(f"{root}/{task}/ep_*"))
        poses, grips, qs, zmins, lo, hi = [], [], [], [], [], []
        for e in eps:
            try:
                tcp = np.asarray(zarr.open(e + "/arm_tcp_pose.zarr")["data"])
                poses.append(tcp[0])
                zmins.append(float(tcp[:, 2].min()))
                lo.append(tcp[:, :3].min(0)); hi.append(tcp[:, :3].max(0))
                qs.append(np.asarray(zarr.open(e + "/arm_q.zarr")["data"][0]))
                grips.append(float(np.ravel(zarr.open(e + "/gripper.zarr")["data"][0])[0]))
            except Exception as ex:            # noqa: BLE001 — skip broken episode, keep stats honest
                print(f"skip {e}: {ex}", file=sys.stderr)
        P, G, Q, Z = np.array(poses), np.array(grips), np.array(qs), np.array(zmins)
        assert len(P) >= 20, f"{task}: only {len(P)} episodes — refusing thin stats"
        fmt = lambda a: "[" + ", ".join(f"{x:.4f}" for x in a) + "]"  # noqa: E731
        lines += [
            f"  {task}:",
            f"    n: {len(P)}",
            f"    tcp_mean: {fmt(P.mean(0))}",
            f"    tcp_std:  {fmt(P.std(0))}",
            f"    gripper_mean: {G.mean():.3f}",
            f"    gripper_std: {G.std():.3f}",
            # demo START joint configuration (the policy reads raw joints: a
            # wrapped wrist / flipped IK branch must gate OUT) + the lowest TCP
            # height any demo reached (run_deploy's z no-go floor = this - margin)
            f"    q_n: {len(Q)}",
            f"    q_mean: {fmt(Q.mean(0))}",
            f"    q_std:  {fmt(Q.std(0))}",
            f"    tcp_z_min: {Z.min():.4f}",
            # demo TCP envelope over ALL frames -> run_deploy's STOP hitbox (+ margin)
            f"    tcp_min: {fmt(np.array(lo).min(0))}",
            f"    tcp_max: {fmt(np.array(hi).max(0))}",
        ]
    out = Path(__file__).resolve().parents[1] / "configs" / "start_poses.yaml"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
