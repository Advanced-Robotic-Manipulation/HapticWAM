import glob, json, os, numpy as np, zarr
demo = {}
for ep in glob.glob("/home/physicalai/phantom-icra-2027/data/val_eval/tasks/*/ep_*") + glob.glob("/home/physicalai/phantom-icra-2027/data/incoming_recovery/*/ep_*"):
    try:
        m = json.load(open(os.path.join(ep, "meta.json")))
        if m.get("success") is False or m.get("status", "finalized") != "finalized": continue
        t = np.asarray(zarr.open(os.path.join(ep, "arm_tcp_pose.zarr"), mode="r")["data"])[:, :3]
        sp = np.asarray(zarr.open(os.path.join(ep, "arm_tcp_speed.zarr"), mode="r")["data"])[:, :3]
        demo.setdefault(m["task"], []).append((t.min(0), t.max(0), np.linalg.norm(sp, axis=1).max(), np.percentile(np.linalg.norm(sp, axis=1), 99)))
    except Exception as e: pass
fmt = lambda a: "[" + ", ".join(f"{x:.4f}" for x in a) + "]"
for t in ("Carton", "egg", "waffles", "whiteboard"):
    lo = np.array([a for a, _, _, _ in demo[t]]).min(0); hi = np.array([b for _, b, _, _ in demo[t]]).max(0)
    vmax = np.array([v for _, _, v, _ in demo[t]]); v99 = np.array([v for _, _, _, v in demo[t]])
    print(f"  {t}:\n    tcp_min: {fmt(lo)}\n    tcp_max: {fmt(hi)}\n    # demo |tcp speed| max {vmax.max():.3f} m/s (median-episode max {np.median(vmax):.3f}, p99 median {np.median(v99):.3f})")
