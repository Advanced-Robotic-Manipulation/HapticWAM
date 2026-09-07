#!/usr/bin/env python3
"""Supplementary CPU audit aligned at first post-acquisition 460 mm radius."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def radius(q):
    return np.sqrt(
        0.24365**2 + 0.21325**2 + 0.11235**2 + 2 * 0.24365 * 0.21325 * np.cos(q[2])
    )


def angle(a, b):
    return float(
        np.rad2deg(
            (Rotation.from_rotvec(a) * Rotation.from_rotvec(b).inv()).magnitude()
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.nice(19)
    if args.out.exists():
        raise FileExistsError(args.out)
    audit = json.loads(args.audit.read_text())
    result = []
    for case in audit["cases"]:
        p = Path(case["directory"])
        t0 = case["stages"]["first_radius_460mm"]["t_s"]
        for f in ["execution_trace.jsonl", "planner_trace.json"]:
            assert (
                hashlib.sha256((p / f).read_bytes()).hexdigest()
                == case["input_sha256"][f]
            )
        e = [
            json.loads(x)
            for x in (p / "execution_trace.jsonl").read_text().splitlines()
            if x
        ]
        plans = json.loads((p / "planner_trace.json").read_text())
        end = case["stop_t_s"] if case["stop_t_s"] is not None else case["finish_t_s"]
        snaps = []
        for dt in [0, 0.5, 1, 2]:
            if t0 + dt > end + 1e-9:
                continue
            r = min(e, key=lambda r: abs(r["t"] - (t0 + dt)))
            snaps.append(
                {
                    "relative_t_s": dt,
                    "actual_t_s": r["t"],
                    "tcp": r["measured_tcp"],
                    "q": r["measured_q"],
                    "radius_m": float(radius(r["measured_q"])),
                }
            )
        ids = {r["active_replan_id"] for r in e if t0 <= r["t"] <= min(t0 + 2, end)}
        raw = []
        for plan in plans:
            if plan["replan_id"] not in ids:
                continue
            a = np.asarray(plan["actions"])[:10].sum(axis=0)
            pose = np.asarray(plan["t0_pose"])
            raw.append(
                {
                    "id": plan["replan_id"],
                    "capture_t_s": plan["t"],
                    "raw_head10_xyz_m": a[:3].tolist(),
                    "raw_head10_SO3_change_deg": angle(pose[3:] + a[3:6], pose[3:]),
                }
            )
        result.append(
            {
                "group": case["group"],
                "seed": case["seed"],
                "full_task": case["outcomes"]["full_task"],
                "critical_t_s": t0,
                "snapshots": snaps,
                "raw_plans_actually_active_during_critical2s": raw,
            }
        )
    args.out.write_text(
        json.dumps(
            {
                "scope": "Stage-aligned descriptive samples; raw first10 proposals can include unplayed future rows. No intervention or score change.",
                "main_audit_sha256": hashlib.sha256(
                    args.audit.read_bytes()
                ).hexdigest(),
                "cases": result,
            },
            indent=2,
        )
        + "\n"
    )
    for r in result:
        if r["full_task"] or r["seed"] in [904501, 904505, 904511, 904302]:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
