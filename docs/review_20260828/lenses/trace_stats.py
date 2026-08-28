import json, glob, os, numpy as np
def scale(s):
    u = np.clip((s - 0.1) / 0.9, 0, 1); return 1 - u * 0.85
for day in ("20260820", "20260828"):
    print("=== day", day)
    allsig, alllat, alldt = [], [], []
    for ep in sorted(glob.glob(os.path.expanduser(f"~/phantom-icra-2027/data/episodes/deploy/{day}/ep_*"))):
        p = os.path.join(ep, "planner_trace.json")
        if not os.path.exists(p):
            continue
        tr = json.load(open(p))
        if len(tr) < 3:
            continue
        meta = json.load(open(os.path.join(ep, "meta.json")))
        tags = [t for t in meta.get("tags", []) if t.startswith(("ckpt", "nfe", "g1", "pnoise", "fresh"))]
        lat = np.array([r["latency_s"] for r in tr]); dt = np.diff([r["t"] for r in tr])
        sig = np.array([max(r["sigma"]) for r in tr]); gate = np.array([r["gate"] for r in tr])
        A = np.array([r["actions"] for r in tr])
        dz_head = A[:, :9, 2].sum(1) * 1000; dz_tail = A[:, 9:, 2].sum(1) * 1000
        desc = A[:, :, 2].sum(1) < -0.005
        allsig += sig.tolist(); alllat += lat.tolist(); alldt += dt.tolist()
        hd = dz_head[desc].mean() if desc.any() else 0.0
        tl = dz_tail[desc].mean() if desc.any() else 0.0
        print("%-38s succ=%-5s n=%2d lat %.2fs dt %.2fs sigma_max mean %.2f max %.2f scale mean %.2f min %.2f | dz head/tail %6.1f/%6.1f mm | gate %.2f | %s"
              % (os.path.basename(ep)[:38], meta.get("success"), len(tr), lat.mean(), dt.mean(), sig.mean(), sig.max(),
                 scale(sig).mean(), scale(sig).min(), hd, tl, gate.mean(), tags))
    if allsig:
        s = np.array(allsig)
        print("day summary: sigma_max pct 10/50/90 = %.2f/%.2f/%.2f  scale<0.95 in %.0f%% of replans; latency mean %.2f; replan interval mean %.2f"
              % (np.percentile(s, 10), np.percentile(s, 50), np.percentile(s, 90), 100 * (scale(s) < 0.95).mean(), np.mean(alllat), np.mean(alldt)))
