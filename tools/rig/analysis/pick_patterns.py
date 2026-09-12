import json, os, glob, re, numpy as np, zarr, datetime
BOX_Y = -0.055   # release-volume near edge (y); box interior beyond
rows = []
for E in sorted(glob.glob("data/episodes/deploy/20260912/ep_*"), key=os.path.getmtime):
    m = json.load(open(E + "/meta.json")); tags = m["tags"]
    label = next((t[6:] for t in tags if t.startswith("label:")), None)
    if label not in ("v6", "stu_ftA_r2", "stu_v6_r2", "stu_v6", "ftA"): continue
    preset7 = "boundary_projection" in m.get("deploy_overrides", {})
    seed = next((int(t[5:]) for t in tags if t.startswith("seed:")), None)
    sr = next((t for t in tags if t.startswith("start_req:")), "")
    mm = re.match(r"start_req:(-?\d+),(-?\d+),(-?\d+)mm/g([\d.]+)", sr)
    start = tuple(int(v) / 1000 for v in mm.groups()[:3]) if mm else None
    g0 = float(mm.group(4)) if mm else None
    stop = next((t[5:] for t in tags if t.startswith("stop:")), None)
    try:
        g = zarr.open(E + "/arm_tcp_pose.zarr", mode="r"); ts = np.asarray(g["ts"][:]); d = np.asarray(g["data"][:]); t0 = ts[0]
        dur = float(ts[-1] - t0)
    except Exception:
        continue
    grasp_t = None; lifted = False; reach_box = False; approach_y = float(d[:, 1].max())
    try:
        ga = zarr.open(E + "/actions.zarr", mode="r"); ad = np.asarray(ga["data"][:]); ats = np.asarray(ga["ts"][:]) - t0
        gg = zarr.open(E + "/gripper.zarr", mode="r"); gd = np.asarray(gg["data"][:]); gts = np.asarray(gg["ts"][:]) - t0
        closed_cmd = np.where(ad[:, 6] > 0.5)[0]
        if len(closed_cmd):
            tc = float(ats[closed_cmd[0]])
            # measured closure after the command and pads loaded?  use tactile wrench magnitude as load proxy
            loads = []
            for side in ("left", "right"):
                p = E + f"/tactile_{side}_wrench.zarr"
                if os.path.exists(p):
                    gt = zarr.open(p, mode="r"); tt = np.asarray(gt["ts"][:]) - t0; td = np.asarray(gt["data"][:])
                    sel = (tt > tc) & (tt < tc + 4.0)
                    loads.append(float(np.linalg.norm(td[sel, :3], axis=1).max()) if sel.any() else 0.0)
            if len(loads) == 2 and min(loads) > 1.0:
                grasp_t = tc
                zc = float(d[np.searchsorted(ts - t0, tc) - 1, 2]) if tc > 0 else float(d[0, 2])
                after = d[(ts - t0) > tc]
                if len(after) and after[:, 2].max() - zc > 0.08:
                    lifted = True
                    reach_box = bool(after[:, 1].max() > BOX_Y)
    except Exception as e:
        pass
    success = m.get("success")
    if success is True: stage, name = 4, "placed"
    elif reach_box: stage, name = 3, "reached_box"
    elif lifted: stage, name = (2.5, "lift_toward_box") if approach_y > -0.15 else (2, "lift")
    elif grasp_t is not None: stage, name = 1, "grasp"
    else: stage, name = 0, "no_grasp"
    if success is False and stage == 4: stage, name = 3, "reached_box"
    rows.append(dict(ep=os.path.basename(E)[-8:], label=label, preset7=preset7, seed=seed, start=start, g0=g0, dur=round(dur, 1),
                     success=success, stop=stop, stage=stage, name=name, grasp_t=None if grasp_t is None else round(grasp_t, 1),
                     max_y=round(approach_y, 3), max_z=round(float(d[:, 2].max()), 3),
                     reach=None if start is None else round(float(np.linalg.norm(start)), 3)))
json.dump(rows, open("/tmp/pick_patterns.json", "w"), indent=1)
print(f"{'ep':>8} {'label':11s} p7 {'seed':>4} {'start x,y,z (m)':22s} {'reach':>5} {'g0':>4} {'dur':>5} {'succ':>5} {'stage':>5} {'outcome':16s} {'grasp_t':>7} {'max_y':>6} {'max_z':>5} stop")
for r in rows:
    st = "" if r["start"] is None else "%.3f,%.3f,%.3f" % r["start"]
    print(f"{r['ep']:>8} {r['label']:11s} {'7' if r['preset7'] else '1':>2} {str(r['seed']):>4} {st:22s} {str(r['reach']):>5} {str(r['g0']):>4} {r['dur']:5.1f} {str(r['success']):>5} {r['stage']:5} {r['name']:16s} {str(r['grasp_t']):>7} {r['max_y']:+6.3f} {r['max_z']:5.3f} {r['stop']}")
