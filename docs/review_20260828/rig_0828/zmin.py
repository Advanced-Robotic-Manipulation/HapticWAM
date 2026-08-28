import glob, json, os, numpy as np, zarr
eps = glob.glob("/home/physicalai/phantom-icra-2027/data/val_eval/tasks/*/ep_*") + glob.glob("/home/physicalai/phantom-icra-2027/data/incoming_recovery/*/ep_*")
per = {}
for ep in eps:
    try:
        m = json.load(open(os.path.join(ep, "meta.json")))
        if m.get("status", "finalized") != "finalized" or m.get("success") is False: continue
        z = np.asarray(zarr.open(os.path.join(ep, "arm_tcp_pose.zarr"), mode="r")["data"])[:, 2] * 1000
        per.setdefault(m["task"], []).append((z.min(), z.max()))
    except Exception as e: pass
for t, v in sorted(per.items()):
    a = np.array(v); print(f"{t:12s} n={len(a):3d}  z_min: min {a[:,0].min():6.1f} p5 {np.percentile(a[:,0],5):6.1f} mean {a[:,0].mean():6.1f}   z_max: max {a[:,1].max():6.1f} p95 {np.percentile(a[:,1],95):6.1f}")
