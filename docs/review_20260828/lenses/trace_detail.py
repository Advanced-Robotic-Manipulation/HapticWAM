import json, glob, os, numpy as np, zarr, torch
H = os.path.expanduser("~/phantom-icra-2027")
def stream(ep, name):
    p = os.path.join(ep, name + ".zarr")
    if not os.path.isdir(p): return None, None
    g = zarr.open(p, mode="r"); return np.asarray(g["data"]), np.asarray(g["ts"])
def at(ts, d, t):
    i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)); return d[i]
# (a) per-replan detail for a few 08-28 episodes: actual z, head/tail dz, grip cmd
picks = ["ep_teacher_waffles_1787923675_000", "ep_teacher_waffles_1787922904_000", "ep_teacher_whiteboard_1787941222_002", "ep_teacher_Carton_1787937561_000"]
for name in picks:
    ep = os.path.join(H, "data/episodes/deploy/20260828", name)
    meta = json.load(open(os.path.join(ep, "meta.json"))); off = meta["clock_calibration"]["offset"]
    tr = json.load(open(os.path.join(ep, "planner_trace.json")))
    tcp, tts = stream(ep, "arm_tcp_pose"); grip, gts = stream(ep, "gripper"); spd, sts = stream(ep, "arm_tcp_speed")
    print("---", name, "success", meta.get("success"), "notes", (meta.get("notes") or "")[:60])
    print("  i   t  z_act(mm) vz_act(mm/s) | head dz  tail dz (mm) | grip cmd head/tail max | grip act | p_none")
    for i, r in enumerate(tr):
        t0 = r["t"] + off + r["latency_s"]
        A = np.array(r["actions"])
        z = at(tts, tcp, t0)[2] * 1000; vz = at(sts, spd, t0)[2] * 1000; g = at(gts, grip, t0)[0]
        print("  %2d %4.1f %7.1f %8.1f | %7.1f %7.1f | %.2f %.2f | %.2f | %.2f" % (i, t0 - tts[0], z, vz, A[:9, 2].sum() * 1000, A[9:, 2].sum() * 1000, A[:9, 6].max(), A[9:, 6].max(), g, r["p_evt"][0]))
# (b) demo terminal windows: head (first 0.9 s) vs tail (last 0.7 s) descent in the 1.6 s before close, plus descent speed 1-3 s before close
def close_idx(gp):
    run_min = np.minimum.accumulate(gp)
    hit = np.nonzero((gp > 0.45) & (gp - run_min > 0.15))[0]
    return int(hit[0]) if len(hit) else None
for task in ("waffles", "whiteboard", "Carton", "egg"):
    roots = glob.glob(os.path.join(H, "data/phantom-episodes/tasks", task, "ep_*"))
    heads, tails, vs = [], [], []
    for ep in sorted(roots)[:120]:
        try:
            tcp, tts = stream(ep, "arm_tcp_pose"); grip, gts = stream(ep, "gripper")
            if tcp is None or grip is None: continue
            ci = close_idx(grip[:, 0])
            if ci is None: continue
            tc = gts[ci]
            z = lambda t: at(tts, tcp, t)[2] * 1000
            heads.append(z(tc - 0.7) - z(tc - 1.6)); tails.append(z(tc) - z(tc - 0.7)); vs.append((z(tc - 1.0) - z(tc - 3.0)) / 2.0)
        except Exception as e:
            pass
    if heads:
        print("DEMO %-10s n=%3d  dz head(1.6..0.7s before close) mean %6.1f  tail(0.7..0s) mean %6.1f  | descent speed 3..1 s before close %6.1f mm/s" % (task, len(heads), np.mean(heads), np.mean(tails), np.mean(vs)))
    else:
        print("DEMO", task, "no episodes at", os.path.join(H, "data/phantom-episodes/tasks", task))
# (c) text cache keys + checkpoint text provenance
for c in glob.glob(os.path.join(H, "data/phantom-episodes/tasks/text_embeddings.pt")) + glob.glob(os.path.join(H, "phantom/runs/*/text_embeddings.pt")):
    raw = torch.load(c, map_location="cpu", weights_only=True); raw.pop("__meta__", None)
    print("TEXT CACHE", c, sorted(raw.keys()))
print(open(os.path.join(H, "phantom/configs/paths.local.yaml")).read())
for ck in ("phantom/runs/teacher_v5_batch0822/v5_6.pt", "phantom/runs/teacher_v4_790eps/teacher_020000.pt"):
    p = os.path.join(H, ck)
    if os.path.exists(p):
        pl = torch.load(p, map_location="cpu", weights_only=False)
        cf = pl["configs"]
        print("CKPT", ck, "text_conditioning", cf.get("text_conditioning"), "acc", cf["model"].get("acc"), "cond_dropout", cf["model"].get("cond_dropout_p"), "rope", cf["model"].get("rope_time_mode"), "nfe", cf["model"].get("nfe"))
        print("   action norm mean", np.round(np.asarray(pl["norm_stats"]["mean"]["action"]), 4).tolist(), "std", np.round(np.asarray(pl["norm_stats"]["std"]["action"]), 4).tolist())
