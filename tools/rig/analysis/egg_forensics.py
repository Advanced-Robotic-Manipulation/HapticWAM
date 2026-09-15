"""Per-episode sensor forensics for today's egg takes: start pose, gripper closing, pad forces, wrist F/T, rule label."""
import glob
import json
import os
import sys

import numpy as np
import zarr

D = sys.argv[1] if len(sys.argv) > 1 else "../data/episodes/deploy/20260915"
PAT = sys.argv[2] if len(sys.argv) > 2 else "ep_teacher_egg_*"
eps = sorted(glob.glob(os.path.join(D, PAT)))


def A(ep, name):
    g = zarr.open(os.path.join(ep, name), mode="r")
    return np.asarray(g["data"]), np.asarray(g["ts"])


hw = None
try:
    from phantom.eval.grasp_label import label_episode
    from phantom.config.hardware import load_hardware
    hw = load_hardware("configs/hardware.nuc.yaml")
except Exception as e:  # noqa: BLE001
    print("no rule:", type(e).__name__, e)

print("ep | verdict | dur | start xyz mm | minz | grip(min,t) | Lpad peakN@t area | Rpad peakN@t area | wristdF | tcp@padpeak | end xyz | stop | rule")
for ep in eps:
    m = json.load(open(ep + "/meta.json"))
    stop = json.load(open(ep + "/stop.json")) if os.path.exists(ep + "/stop.json") else {}
    tcp, tt = A(ep, "arm_tcp_pose.zarr")
    g, gt = A(ep, "gripper.zarr")
    lw, lt = A(ep, "tactile_left_wrench.zarr")
    rw, rt = A(ep, "tactile_right_wrench.zarr")
    la, _ = A(ep, "tactile_left_area.zarr")
    ra, _ = A(ep, "tactile_right_area.zarr")
    ft, ftt = A(ep, "arm_ft.zarr")
    t0 = tt[0]
    dur = tt[-1] - t0
    st = tcp[0, :3] * 1000
    en = tcp[-1, :3] * 1000
    minz = tcp[:, 2].min() * 1000
    gp = g[:, 0]
    gi = int(np.argmin(gp))

    def peak(w, ts):
        f = np.linalg.norm(w[:, :3], axis=1)
        i = int(np.argmax(f))
        return f[i], ts[i] - t0, i

    lf, lti, li = peak(lw, lt)
    rf, rti, ri = peak(rw, rt)
    fb = np.linalg.norm(ft[:, :3] - ft[:20, :3].mean(0), axis=1)
    tpk = lt[li] if lf >= rf else rt[ri]
    k = int(np.argmin(np.abs(tt - tpk)))
    tcpk = tcp[k, :3] * 1000
    rule = ""
    if hw is not None:
        try:
            lab = label_episode(ep, hw, task="egg")
            rule = (f"ok={lab.grasp_ok} zc={lab.z_close_mm} hold={lab.hold_s} ch={lab.c_hold} "
                    f"lift={lab.lift_mm} n={lab.n_close_attempts} {list(lab.reasons)[:2]}")
        except Exception as e:  # noqa: BLE001
            rule = f"rule err {type(e).__name__}: {e}"[:80]
    status = str(m.get("status"))[:5]
    succ = str(m.get("success"))[:5]
    notes = str(m.get("notes"))[:14]
    print(f"{os.path.basename(ep)[15:]} | {status}/{succ} {notes} | {dur:5.1f}s | {st[0]:.0f},{st[1]:.0f},{st[2]:.0f} | {minz:.0f} | "
          f"{gp.min():.2f}@{gt[gi] - t0:.1f}s | L {lf:.2f}N@{lti:.1f}s a{la.max():.2f} | R {rf:.2f}N@{rti:.1f}s a{ra.max():.2f} | "
          f"{fb.max():.1f}N | {tcpk[0]:.0f},{tcpk[1]:.0f},{tcpk[2]:.0f} | {en[0]:.0f},{en[1]:.0f},{en[2]:.0f} | {stop.get('stopped_reason')} | {rule}")
