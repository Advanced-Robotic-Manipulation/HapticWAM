#!/usr/bin/env python3
"""Compare the FIRST N vs LAST M whiteboard episodes across every modality.

The operator drove the first batch at max throttle and the last few carefully;
this quantifies what actually differs — not just force, but speed, smoothness,
contact behaviour, gripper load and how often the arm faulted.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(os.environ.get("PHANTOM_NUC_DRIVE", "/media/nuc/phantom_drive")) / "phantom_episodes"
FIRST_N, LAST_M = int(sys.argv[1]) if len(sys.argv) > 1 else 40, \
                  int(sys.argv[2]) if len(sys.argv) > 2 else 5


def ep_metrics(ep: Path) -> dict:
    def g(stream):
        p = ep / f"{stream}.zarr"
        if not p.is_dir():
            return None, None
        z = zarr.open(str(p), mode="r")
        return np.asarray(z["data"]), np.asarray(z["ts"])

    m = {}
    tcp, t_tcp = g("arm_tcp_pose")
    spd, _ = g("arm_tcp_speed")
    ft, _ = g("arm_ft")
    grip, _ = g("gripper")
    act, t_act = g("actions")

    if tcp is not None and len(tcp) > 2:
        pos = tcp[:, :3]
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        dur = float(t_tcp[-1] - t_tcp[0])
        m["path_len_m"] = float(step.sum())
        m["dur_s"] = dur
        m["speed_mean_mm_s"] = float(step.sum() / max(dur, 1e-6)) * 1000
    if spd is not None and len(spd) > 2:
        lin = np.linalg.norm(spd[:, :3], axis=1)
        m["tcp_speed_mean_mm_s"] = float(lin.mean()) * 1000
        m["tcp_speed_p95_mm_s"] = float(np.percentile(lin, 95)) * 1000
        # jerk-ish: how abruptly the commanded motion changes
        m["accel_rms_mm_s2"] = float(np.sqrt((np.diff(lin) ** 2).mean())) * 1000
    if ft is not None and len(ft) > 2:
        fmag = np.linalg.norm(ft[:, :3], axis=1)
        m["wrist_F_mean_N"] = float(fmag.mean())
        m["wrist_F_p95_N"] = float(np.percentile(fmag, 95))
    if grip is not None and len(grip) > 2:
        m["grip_pos_mean"] = float(grip[:, 0].mean())
    if act is not None and len(act) > 2:
        d = np.linalg.norm(act[:, :3], axis=1)
        m["action_dstep_mean_mm"] = float(d.mean()) * 1000
        m["action_dstep_p95_mm"] = float(np.percentile(d, 95)) * 1000
        m["n_actions"] = int(len(act))

    for side in ("left", "right"):
        w, _ = g(f"tactile_{side}_wrench")
        a, _ = g(f"tactile_{side}_area")
        if w is not None and len(w) > 1:
            f = np.linalg.norm(w[:, :3], axis=1)
            m[f"tac_{side}_F_mean_N"] = float(f.mean())
            m[f"tac_{side}_F_peak_N"] = float(f.max())
        if a is not None and len(a) > 1:
            m[f"tac_{side}_area_mean"] = float(np.asarray(a).mean())
    return m


def summarize(eps, label):
    rows = []
    for ep in eps:
        try:
            rows.append(ep_metrics(ep))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {ep.name}: {e}")
    keys = sorted({k for r in rows for k in r})
    out = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    print(f"\n=== {label}: {len(rows)} episodes")
    return out


def main():
    eps = []
    for sess in sorted(p for p in ROOT.iterdir()
                       if p.is_dir() and p.name.endswith("whiteboard")):
        for ep in sorted(sess.glob("ep_*")):
            if (ep / "meta.json").exists():
                eps.append(ep)
    print(f"whiteboard episodes found: {len(eps)} (chronological)")
    a = summarize(eps[:FIRST_N], f"FIRST {FIRST_N} (max throttle)")
    b = summarize(eps[-LAST_M:], f"LAST {LAST_M} (careful)")

    print(f"\n{'metric':<26} {'first':>12} {'last':>12} {'change':>10}")
    print("-" * 64)
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            continue
        va, vb = a[k], b[k]
        pct = (vb - va) / abs(va) * 100 if abs(va) > 1e-9 else float("nan")
        print(f"{k:<26} {va:>12.2f} {vb:>12.2f} {pct:>9.0f}%")


if __name__ == "__main__":
    main()
