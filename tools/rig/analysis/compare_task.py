"""Compare one arm's takes with the teacher's on the same task and cells: start, early motion, lowest point, end.
usage: compare_task.py <label> <task> [teacher label]"""
import glob
import json
import os
import sys

import numpy as np
import zarr

from phantom.eval import stats as S

D = os.path.expanduser("~/phantom-icra-2027/data/episodes/deploy/20260915_experiment")
LABEL, TASK = sys.argv[1], sys.argv[2]
TEACH = sys.argv[3] if len(sys.argv) > 3 else "v6_simft_mt1500"


def A(ep, name):
    g = zarr.open(os.path.join(ep, name), mode="r")
    return np.asarray(g["data"]), np.asarray(g["ts"])


def summ(ep):
    m = json.load(open(ep + "/meta.json"))
    tcp, tt = A(ep, "arm_tcp_pose.zarr")
    act, _ = A(ep, "actions.zarr")
    lw, _ = A(ep, "tactile_left_wrench.zarr")
    rw, _ = A(ep, "tactile_right_wrench.zarr")
    st = json.load(open(ep + "/stop.json")) if os.path.exists(ep + "/stop.json") else {}
    xyz = tcp[:, :3] * 1000
    grip = act[:, 6] if act.ndim == 2 and act.shape[1] > 6 else np.zeros(len(act))
    closes = int(((grip[1:] >= 0.45) & (grip[:-1] < 0.45)).sum()) if len(grip) > 1 else 0
    pk = max(np.linalg.norm(lw[:, :3], axis=1).max() if len(lw) else 0, np.linalg.norm(rw[:, :3], axis=1).max() if len(rw) else 0)
    i2 = int(np.argmin(np.abs(tt - (tt[0] + 2.0))))
    i4 = int(np.argmin(np.abs(tt - (tt[0] + 4.0))))
    imz = int(np.argmin(xyz[:, 2]))
    ev = [e.get("kind") for e in (st.get("safety_events") or [])]
    return dict(cell=S.episode_seed(m) - 100, verdict=str(m.get("notes"))[:11], start=xyz[0].round(0), d2=(xyz[i2] - xyz[0]).round(0),
                d4=(xyz[i4] - xyz[0]).round(0), minz=xyz[imz].round(0), end=xyz[-1].round(0), closes=closes, pad=round(float(pk), 1),
                stop=st.get("stopped_reason"), ev=ev[:3], replans=st.get("n_replans"), dur=round(float(tt[-1] - tt[0]), 1))


by = {}
for e in sorted(glob.glob(D + "/ep_*")):
    m = json.load(open(e + "/meta.json"))
    if m.get("task") != TASK:
        continue
    lab = [t[6:] for t in m["tags"] if t.startswith("label:")]
    lab = lab[0] if lab else "?"
    if lab in (LABEL, TEACH):
        by.setdefault(lab, {})[S.episode_seed(m) - 100] = summ(e)
stu, tea = by.get(LABEL, {}), by.get(TEACH, {})
print(f"{LABEL} on {TASK}: {len(stu)} takes")
for c in sorted(stu):
    s = stu[c]
    print(f"cell {c:2d} {LABEL[:7]:7s} {s['verdict']:11s} start {s['start']} d2s {s['d2']} d4s {s['d4']} minz {s['minz']} end {s['end']} closes {s['closes']} pad {s['pad']}N {s['stop']} {s['ev']} {s['replans']}r {s['dur']}s")
    if c in tea:
        t = tea[c]
        print(f"        teacher {t['verdict']:11s} start {t['start']} d2s {t['d2']} d4s {t['d4']} minz {t['minz']} end {t['end']} closes {t['closes']} pad {t['pad']}N {t['stop']} {t['replans']}r")
