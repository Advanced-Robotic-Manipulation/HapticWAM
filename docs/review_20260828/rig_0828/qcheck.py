import glob, json, os, numpy as np, zarr
def q0(ep):
    return np.asarray(zarr.open(os.path.join(ep, "arm_q.zarr"), mode="r")["data"])[0]
demo = {}
for ep in glob.glob("/home/physicalai/phantom-icra-2027/data/val_eval/tasks/*/ep_*") + glob.glob("/home/physicalai/phantom-icra-2027/data/incoming_recovery/*/ep_*"):
    try:
        m = json.load(open(os.path.join(ep, "meta.json")))
        if m.get("success") is False: continue
        demo.setdefault(m["task"], []).append(q0(ep))
    except Exception: pass
stats = {t: (np.array(v).mean(0), np.array(v).std(0), len(v)) for t, v in demo.items()}
for t, (mu, sd, n) in sorted(stats.items()):
    print(f"DEMO {t:11s} n={n:3d} q_mean(deg)={np.round(np.degrees(mu),0)} q_std(deg)={np.round(np.degrees(sd),1)}")
print()
for ep in sorted(glob.glob("/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260828/ep_*"), key=os.path.getmtime):
    m = json.load(open(os.path.join(ep, "meta.json"))); t = m["task"]
    try: q = q0(ep)
    except Exception as e: print(os.path.basename(ep), "no q"); continue
    mu, sd, n = stats[t]; sig = np.abs(q - mu) / sd
    ck = [x[5:] for x in m.get("tags", []) if x.startswith("ckpt")][0]
    print(f"{os.path.basename(ep):40s} {ck:10s} q0(deg)={np.round(np.degrees(q),0)}  sigma={np.round(sig,1)}  worst={sig.max():.1f}")
