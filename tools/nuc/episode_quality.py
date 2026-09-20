#!/usr/bin/env python3
"""Per-episode quality audit over the PHANTOM dataset on the external drive.

Reads ONLY cheap things per episode: every stream's ts array (small), the
small numeric streams (wrench/area/gripper/actions/arm_ft), shapes of the
heavy ones, and decodes exactly one camera frame. Emits one CSV row per
episode + a cohort-relative outlier report (median/MAD per task) so nothing
depends on hard-coded rate assumptions.

Output: ~/phantom-icra-2027/episode_quality.csv + summary on stdout.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(os.environ.get("PHANTOM_NUC_DRIVE", "/media/nuc/phantom_drive")) / "phantom_episodes"
OUT = Path.home() / "phantom-icra-2027" / "episode_quality.csv"
TASKS = ("Carton", "waffles", "egg", "whiteboard", "Carton_fail", "waffles_fail", "egg_fail", "whiteboard_fail")

EXPECTED = ["actions", "actions_abs", "arm_ft", "arm_q", "arm_qd",
            "arm_tcp_pose", "arm_tcp_speed", "camera_scene_color", "gripper",
            "tactile_left_area", "tactile_left_fields_ds",
            "tactile_left_infer_img", "tactile_left_keyframes",
            "tactile_left_wrench", "tactile_right_area",
            "tactile_right_fields_ds", "tactile_right_infer_img",
            "tactile_right_keyframes", "tactile_right_wrench"]
SMALL = {"actions", "actions_abs", "arm_ft", "gripper",
         "tactile_left_wrench", "tactile_right_wrench",
         "tactile_left_area", "tactile_right_area"}
PAD_CEILING_N = 30.0


def audit_episode(ep: Path) -> dict:
    r = {"session": ep.parent.name, "ep": ep.name, "problems": []}
    try:
        meta = json.loads((ep / "meta.json").read_text())
        r["task"] = meta.get("task", "?")
        r["success"] = meta.get("success")
        if meta.get("status") != "finalized":
            r["problems"].append(f"status={meta.get('status')}")
    except Exception as e:  # noqa: BLE001
        r["problems"].append(f"meta_unreadable:{e}")
        r["task"] = "?"
        return r

    missing = [s for s in EXPECTED if not (ep / f"{s}.zarr").is_dir()]
    if missing:
        r["problems"].append("missing_streams:" + ",".join(missing))
    durs, peak_force = [], 0.0
    for s in EXPECTED:
        p = ep / f"{s}.zarr"
        if not p.is_dir():
            continue
        try:
            g = zarr.open(str(p), mode="r")
            ts = np.asarray(g["ts"])
            n = len(ts)
            r[f"n_{s}"] = n
            if n == 0:
                r["problems"].append(f"empty:{s}")
                continue
            if n > 1:
                d = np.diff(ts)
                if (d < 0).any():
                    r["problems"].append(f"nonmonotonic_ts:{s}")
                dur = float(ts[-1] - ts[0])
                durs.append(dur)
                r[f"rate_{s}"] = round(n / max(dur, 1e-9), 2)
                r[f"gap_{s}"] = round(float(d.max()), 3)
            if s in SMALL:
                a = np.asarray(g["data"])
                if not np.isfinite(a).all():
                    r["problems"].append(f"nonfinite:{s}")
                if s.endswith("wrench") and a.ndim == 2 and a.shape[1] >= 3:
                    f = np.linalg.norm(a[:, :3], axis=1)
                    peak_force = max(peak_force, float(f.max()))
                if s.endswith("area"):
                    r[f"max_{s}"] = round(float(a.max()), 1)
            elif s == "camera_scene_color":
                try:
                    import cv2
                    frame = g["data"][n // 2]
                    buf = (np.frombuffer(frame, np.uint8)
                           if isinstance(frame, (bytes, bytearray))
                           else np.asarray(frame, dtype=np.uint8).reshape(-1))
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is None:
                        r["problems"].append("camera_frame_undecodable")
                    elif float(img.std()) < 3.0:
                        r["problems"].append("camera_frame_flat")
                except Exception as e:  # noqa: BLE001
                    r["problems"].append(f"camera_check_failed:{e}")
        except Exception as e:  # noqa: BLE001
            r["problems"].append(f"unreadable:{s}:{e}")
    r["dur_s"] = round(float(np.median(durs)), 2) if durs else 0.0
    r["peak_force_N"] = round(peak_force, 2)
    if peak_force > PAD_CEILING_N:
        r["problems"].append(f"force_over_pad_ceiling:{peak_force:.1f}N")
    contact = max(r.get("max_tactile_left_area", 0.0),
                  r.get("max_tactile_right_area", 0.0))
    r["max_contact_mm2"] = contact
    if contact < 1.0:
        r["problems"].append("no_tactile_contact")
    return r


def main() -> int:
    rows = []
    eps = [ep for sess in sorted(ROOT.iterdir()) if sess.is_dir()
           and sess.name.split("_", 2)[-1] in TASKS
           for ep in sorted(sess.glob("ep_*")) if ep.is_dir()]
    for i, ep in enumerate(eps):
        rows.append(audit_episode(ep))
        if (i + 1) % 25 == 0:
            print(f"...{i + 1}/{len(eps)}", flush=True)

    keys = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in ("session", "ep", "task", "success",
                                           "dur_s", "peak_force_N",
                                           "max_contact_mm2", "problems"), k))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "problems": ";".join(r["problems"])})

    # cohort-relative outliers per task: robust z on duration, contact, rates
    def mad_z(vals, v):
        med = np.median(vals)
        mad = np.median(np.abs(np.asarray(vals) - med)) or 1e-9
        return abs(v - med) / (1.4826 * mad)

    print(f"\n=== {len(rows)} episodes audited ===")
    for task in TASKS:
        cohort = [r for r in rows if r.get("task") == task]
        if not cohort:
            continue
        hard = [r for r in cohort if r["problems"]]
        scored = []
        durs = [r["dur_s"] for r in cohort]
        contacts = [r["max_contact_mm2"] for r in cohort]
        cam = [r.get("rate_camera_scene_color", 0) for r in cohort]
        for r in cohort:
            z = (mad_z(durs, r["dur_s"]) + mad_z(contacts, r["max_contact_mm2"])
                 + mad_z(cam, r.get("rate_camera_scene_color", 0))
                 + 10.0 * len(r["problems"]))
            scored.append((z, r))
        scored.sort(key=lambda t: -t[0])
        print(f"\n--- {task}: {len(cohort)} eps, {len(hard)} with hard problems")
        for z, r in scored[:6]:
            print(f"  score={z:5.1f}  {r['session']}/{r['ep']}  dur={r['dur_s']}s "
                  f"contact={r['max_contact_mm2']}mm2 "
                  f"cam={r.get('rate_camera_scene_color', '?')}Hz "
                  f"peakF={r['peak_force_N']}N  {';'.join(r['problems']) or 'ok'}")
    print(f"\nfull table: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
