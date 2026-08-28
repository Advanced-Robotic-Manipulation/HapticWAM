"""Rig episode decomposition: commanded vs executed vs ACTUAL, per replan.

    python tools/rig_trace_decompose.py deploy <deploy_day_dir> [...]   # e.g. data/episodes/deploy/20260828
    python tools/rig_trace_decompose.py demos  <task_dir> [...]         # demo reference: where/when/how demos close

`deploy` prints one summary row per episode (label, z at start / min, gripper
close time + TCP at close, plateau aperture, Robotiq OBJ states seen,
commanded vs actual dz/dy over the episode) followed by a per-replan table
that aligns planner_trace.json (perf_counter clock) to the zarr streams
(master clock) through meta.clock_calibration.offset and compares the chunk
prefix the executor actually played (until the next replan) with the TCP
motion measured over that window. `demos` prints per-task close statistics
(t_close, xyz at close, aperture plateau, z_min) for the same rule as
training's close_index (pos > 0.45 with a > 0.15 rise).

Written for the 2026-08-28 session: it showed the executor tracks the
commanded prefix within a few mm during descent (executor exonerated), the
policy decelerating and closing 65-120 mm above the demo grasp height, and
16/26 episodes started from a wrapped-wrist / flipped-branch joint
configuration (see docs/rig_session_v5.md, safety batch).
"""
import json, sys, glob, os
import numpy as np, zarr
def stream(ep, name):
    p = os.path.join(ep, name + ".zarr")
    if not os.path.isdir(p): return None, None
    g = zarr.open(p, mode="r"); return np.asarray(g["data"]), np.asarray(g["ts"])
def at(ts, d, t):
    i = int(np.clip(np.searchsorted(ts, t), 0, len(ts)-1)); return d[i]
def close_idx(gp):
    for i in range(1, len(gp)):
        if gp[i] > 0.45 and gp[i] - gp[:i].min() > 0.15: return i
    return None
mode = sys.argv[1]; roots = sys.argv[2:]
if mode == "deploy":
    print(f"{'episode':42s} {'ckpt':10s} {'lab':5s} {'dur':>5s} {'z0':>4s} {'zmin':>4s} {'t_close':>7s} {'xyz@close':>18s} {'grip_pl':>7s} {'objs':>6s} {'cmd_dz_sum':>10s} {'act_dz':>7s} {'cmd_dy':>7s} {'act_dy':>7s}")
    rows_detail = []
    for root in roots:
        for ep in sorted(glob.glob(os.path.join(root, "ep_*")), key=os.path.getmtime):
            meta = json.load(open(os.path.join(ep, "meta.json")))
            off = meta["clock_calibration"]["offset"]
            tcp, tts = stream(ep, "arm_tcp_pose"); grip, gts = stream(ep, "gripper"); act, ats = stream(ep, "actions")
            trp = os.path.join(ep, "planner_trace.json"); tr = json.load(open(trp)) if os.path.exists(trp) else []
            ck = [t[5:] for t in meta.get("tags", []) if t.startswith("ckpt")]; ck = ck[0] if ck else "?"
            lab = {True: "S", False: "F", None: "-"}[meta.get("success")] + ("c" if "contaminated" in meta.get("tags", []) else "")
            if tcp is None or len(tr) < 3: 
                print(f"{os.path.basename(ep):42s} {ck:10s} {lab:5s} (short/aborted, {len(tr)} replans)"); continue
            mm = tcp[:, :3]*1000; t = tts - tts[0]
            gp = grip[:, 0]; ci = close_idx(gp)
            if ci is not None:
                tc = gts[ci]; p = at(tts, mm, tc); pl = gp[(gts >= tc) & (gts <= tc + 2.0)].max(); tcs = f"{tc - tts[0]:5.1f}s"; xyz = f"({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})"
            else:
                tcs, xyz, pl = "never", "-", gp.max()
            a = np.asarray(act)
            print(f"{os.path.basename(ep):42s} {ck:10s} {lab:5s} {t[-1]:5.1f} {mm[0,2]:4.0f} {mm[:,2].min():4.0f} {tcs:>7s} {xyz:>18s} {pl:7.2f} {str(sorted(set(grip[:,1].astype(int).tolist()))):>6s} {a[:,2].sum()*1000:10.0f} {mm[-1,2]-mm[0,2]:7.0f} {a[:,1].sum()*1000:7.0f} {mm[-1,1]-mm[0,1]:7.0f}")
            # per-replan detail
            det = [f"--- {os.path.basename(ep)} {ck} {lab}: replan | t | gate | none | cmd chunk dz | cmd window dz | ACTUAL window dz | actual z | cmd window dy | ACTUAL dy | grip_cmd_max | grip_actual"]
            for i, r in enumerate(tr):
                a_ = np.array(r["actions"]); c = np.cumsum(a_[:, :3], 0)*1000
                t0 = r["t"] + off; t1 = (tr[i+1]["t"] + off) if i+1 < len(tr) else t0 + 0.9
                # executor plays from action_times[0] = t0 + latency; window executed = [t0+lat, t1+lat_next]
                lat = r["latency_s"]; latn = tr[i+1]["latency_s"] if i+1 < len(tr) else lat
                w0, w1 = t0 + lat, t1 + latn
                ksteps = int(np.clip(round((w1 - w0) * 10), 1, 16))
                p0 = at(tts, mm, w0); p1 = at(tts, mm, w1); g1 = at(gts, gp, w1)
                det.append(f"  {i:2d} {t0-tr[0]['t']-off:5.1f} {r['gate']:.2f} {r['p_evt'][0]:.2f} {c[-1,2]:7.1f} {c[ksteps-1,2]:7.1f} {p1[2]-p0[2]:7.1f} {p0[2]:6.0f} {c[ksteps-1,1]:7.1f} {p1[1]-p0[1]:7.1f} {a_[:,6].max():.2f} {g1:.2f}")
            rows_detail.append("\n".join(det))
    print("\n\n".join(rows_detail))
else:  # demos
    for root in roots:
        eps = sorted(glob.glob(os.path.join(root, "ep_*")))
        rows = []
        for ep in eps:
            try:
                meta = json.load(open(os.path.join(ep, "meta.json")))
                if meta.get("status") not in (None, "finalized"): continue
                tcp, tts = stream(ep, "arm_tcp_pose"); grip, gts = stream(ep, "gripper")
                if tcp is None or grip is None: continue
                mm = tcp[:, :3]*1000; gp = grip[:, 0]; ci = close_idx(gp)
                if ci is None: continue
                tc = gts[ci]; p = at(tts, mm, tc); pl = gp[(gts >= tc) & (gts <= tc + 2.0)].max()
                objs = sorted(set(grip[:, 1].astype(int).tolist()))
                rows.append((meta.get("task"), meta.get("success"), tts[-1]-tts[0], tc - tts[0], mm[0,2], p[0], p[1], p[2], gp[0], pl, 2 in objs, mm[:,2].min()))
            except Exception as e:
                print("skip", ep, e)
        if not rows: print(root, "no rows"); continue
        R = np.array([r[2:] for r in rows], dtype=float)
        print(f"{root}: n={len(rows)} task={rows[0][0]} success={sum(1 for r in rows if r[1])}/{len(rows)}")
        names = ["dur_s","t_close_s","z_start","x_close","y_close","z_close","grip_start","grip_plateau","obj2_seen","z_min"]
        for j, n in enumerate(names):
            print(f"   {n:13s} mean {R[:,j].mean():8.1f}  sd {R[:,j].std():6.1f}  min {R[:,j].min():8.1f}  max {R[:,j].max():8.1f}")
