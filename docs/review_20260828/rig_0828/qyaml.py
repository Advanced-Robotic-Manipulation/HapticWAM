import glob, json, os, numpy as np, zarr
demo = {}
for ep in glob.glob("/home/physicalai/phantom-icra-2027/data/val_eval/tasks/*/ep_*") + glob.glob("/home/physicalai/phantom-icra-2027/data/incoming_recovery/*/ep_*"):
    try:
        m = json.load(open(os.path.join(ep, "meta.json")))
        if m.get("success") is False or m.get("status", "finalized") != "finalized": continue
        q = np.asarray(zarr.open(os.path.join(ep, "arm_q.zarr"), mode="r")["data"])[0]
        z = np.asarray(zarr.open(os.path.join(ep, "arm_tcp_pose.zarr"), mode="r")["data"])[:, 2]
        demo.setdefault(m["task"], []).append((q, z.min()))
    except Exception: pass
fmt = lambda a: "[" + ", ".join(f"{x:.4f}" for x in a) + "]"
for t in ("Carton", "egg", "waffles", "whiteboard"):
    Q = np.array([q for q, _ in demo[t]]); Z = np.array([z for _, z in demo[t]])
    print(f"  {t}:\n    q_n: {len(Q)}\n    q_mean: {fmt(Q.mean(0))}\n    q_std:  {fmt(Q.std(0))}\n    tcp_z_min: {Z.min():.4f}")
