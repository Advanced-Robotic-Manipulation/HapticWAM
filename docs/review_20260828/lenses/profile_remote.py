import json, glob, os, sys, collections
import numpy as np, zarr

W = os.path.expanduser("~/phantom-icra-2027/data")
ns = json.load(open(f"{W}/val_eval/tasks/norm_stats.json"))
names = [f"q{i}" for i in range(6)] + [f"qd{i}" for i in range(6)] + ["x","y","z","rx","ry","rz"] + \
        ["vx","vy","vz","wx","wy","wz"] + ["grip_pos","grip_obj"]
print("== ur_state norm stats")
for n, m, s in zip(names, ns["mean"]["ur_state"], ns["std"]["ur_state"]):
    print(f"  {n:9s} mean {m:+.4f} std {s:.4f}")
print("== action norm stats (dx dy dz drx dry drz grip)")
print("  mean", [round(x, 5) for x in ns["mean"]["action"]])
print("  std ", [round(x, 5) for x in ns["std"]["action"]])
print("== wrist_ft std", [round(x, 3) for x in ns["std"]["wrist_ft"]], "mean", [round(x, 3) for x in ns["mean"]["wrist_ft"]])

rows = [json.loads(l) for l in open(f"{W}/val_eval/manifests/all.jsonl") if l.strip()]
c = collections.Counter((r["task"], r["split"]) for r in rows)
print("== manifest rows by (task, split):", dict(c))
print("   on disk under val_eval/tasks:", {t: len(glob.glob(f"{W}/val_eval/tasks/{t}/ep_*")) for t in sorted(os.listdir(f"{W}/val_eval/tasks")) if os.path.isdir(f"{W}/val_eval/tasks/{t}")})

def close_index(pos):
    pos = np.asarray(pos, dtype=np.float64)
    run_min = np.minimum.accumulate(pos)
    hit = np.nonzero((pos > 0.45) & (pos - run_min > 0.15))[0]
    if len(hit): return int(hit[0])
    hit = np.nonzero((pos >= pos.max() - 0.05) & (pos - run_min > 0.10))[0]
    return int(hit[0]) if len(hit) else None

def z(p, s):
    g = zarr.open(f"{p}/{s}.zarr", mode="r"); return np.asarray(g["data"][:]), np.asarray(g["ts"][:])

eps = []
for t in ("Carton", "egg", "waffles", "whiteboard"):
    eps += sorted(glob.glob(f"{W}/val_eval/tasks/{t}/ep_*"))[:30]
rec = sorted(glob.glob(f"{W}/incoming_recovery/**/ep_*", recursive=True))
eps += rec[:60]
print("== profiling", len(eps), "episodes (val_eval<=30/task + incoming_recovery<=60)")
out = collections.defaultdict(list)
for ep in eps:
    try:
        meta = json.load(open(f"{ep}/meta.json")); task = meta["task"]
        gp, gts = z(ep, "gripper"); tp, tts = z(ep, "arm_tcp_pose"); sp, sts = z(ep, "arm_tcp_speed")
        ac, ats = z(ep, "actions")
    except Exception as e:
        print("skip", ep, e); continue
    ci = close_index(gp[:, 0])
    dta = np.diff(ats)
    out["act_dt_med"].append(float(np.median(dta))); out["act_dt_p90"].append(float(np.percentile(dta, 90)))
    out["act_dt_max"].append(float(dta.max()))
    # delta semantics: cumsum(actions over window) vs measured displacement over same ts span
    i0, i1 = len(ats)//3, len(ats)//3 + 16
    if i1 < len(ats):
        cs = ac[i0+1:i1+1, :3].sum(0)
        p0 = tp[np.searchsorted(tts, ats[i0])]; p1 = tp[np.searchsorted(tts, ats[i1])]
        out["cumsum_vs_meas_mm"].append(float(np.linalg.norm(cs - (p1[:3]-p0[:3]))*1000))
    if ci is None:
        out["no_close"].append(task); continue
    tc = gts[ci]
    band = (sts >= tc-1.5) & (sts <= tc-0.2)
    if band.sum() < 5: continue
    v = sp[band, :3]; vn = np.linalg.norm(v, axis=1)
    out[f"{task}::speed_band_mm_s"].append(float(vn.mean()*1000))
    out[f"{task}::descent_mm_s"].append(float(-v[:, 2].mean()*1000))
    out[f"{task}::hover_frac"].append(float((vn < 0.01).mean()))
    # whole-episode dwell fraction (time with |v|<10mm/s) -> uniform-anchor weighting
    vall = np.linalg.norm(sp[:, :3], axis=1)
    out[f"{task}::dwell_frac_all"].append(float((vall < 0.01).mean()))
    pc = tp[np.searchsorted(tts, tc).clip(0, len(tp)-1)]
    out[f"{task}::close_xyz_mm"].append((pc[:3]*1000).tolist())
    # gripper obj codes around/after close
    j = np.searchsorted(gts, tc + 1.0).clip(0, len(gp)-1)
    out[f"{task}::obj_after_close"].append(int(gp[j, 1]))
    out[f"{task}::pos_at_close"].append(float(gp[ci, 0]))
    out[f"{task}::grip_cmd_max"].append(float(ac[:, 6].max()))
    # commanded (action) close vs measured close lag
    cj = close_index(ac[:, 6])
    if cj is not None: out[f"{task}::cmd_close_lead_s"].append(float(tc - ats[cj]))
    out[f"{task}::dur_s"].append(float(tts[-1]-tts[0])); out[f"{task}::tc_s"].append(float(tc - tts[0]))
    # vel channels for a 30% slower descent in normalized units
    m = np.array(ns["mean"]["ur_state"]); s = np.array(ns["std"]["ur_state"])
    vmean = v.mean(0)
    dz = ((0.7*vmean - m[20:23])/s[20:23]) - ((vmean - m[20:23])/s[20:23])
    out[f"{task}::dsigma_v_30pct_slower"].append(np.abs(dz).max())

print("== actions stream dt: median", np.median(out["act_dt_med"]), "p90", np.median(out["act_dt_p90"]), "max(med over eps)", np.median(out["act_dt_max"]))
print("== |cumsum(actions,16) - measured disp| mm: median", np.median(out["cumsum_vs_meas_mm"]), "p90", np.percentile(out["cumsum_vs_meas_mm"], 90))
print("== no_close:", collections.Counter(out["no_close"]))
for k in sorted(out):
    if "::" not in k: continue
    v = out[k]
    if k.endswith("close_xyz_mm"):
        a = np.array(v); print(f"{k:38s} n={len(v)} std_xy={a[:, :2].std(0).round(1)} z_mean={a[:, 2].mean():.1f} z_std={a[:, 2].std():.1f}")
    elif k.endswith("obj_after_close"):
        print(f"{k:38s} {dict(collections.Counter(v))}")
    else:
        a = np.array(v, dtype=float); print(f"{k:38s} n={len(a)} mean={a.mean():.3f} p10={np.percentile(a,10):.3f} p50={np.median(a):.3f} p90={np.percentile(a,90):.3f}")
