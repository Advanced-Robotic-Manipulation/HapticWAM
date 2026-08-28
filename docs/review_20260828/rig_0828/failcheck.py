import glob, json, os, numpy as np, zarr
rows = []
for ep in [os.path.dirname(p) for p in sorted(glob.glob("/private/tmp/claude-501/-Users-sannikov/a5af36c2-8450-43d1-9a34-fa0cdb820706/scratchpad/rig/faildemos/archive/*fail*/ep_*/meta.json"))]:
    if not (os.path.isdir(os.path.join(ep, "gripper.zarr")) and os.path.isdir(os.path.join(ep, "arm_tcp_pose.zarr"))): continue
    m = json.load(open(os.path.join(ep, "meta.json")))
    g = zarr.open(os.path.join(ep, "gripper.zarr"), mode="r"); gp = np.asarray(g["data"])[:, 0]; obj = np.asarray(g["data"])[:, 1]; gts = np.asarray(g["ts"])
    t = zarr.open(os.path.join(ep, "arm_tcp_pose.zarr"), mode="r"); tcp = np.asarray(t["data"])[:, :3] * 1000; tts = np.asarray(t["ts"])
    dur = tts[-1] - tts[0]
    # close/open events on the gripper trace
    closed = gp > 0.45
    edges = np.diff(closed.astype(int)); closes = np.nonzero(edges == 1)[0]; opens = np.nonzero(edges == -1)[0]
    tc = [gts[i] - gts[0] for i in closes]
    z_at = [tcp[min(np.searchsorted(tts, gts[i]), len(tcp) - 1), 2] for i in closes]
    zmin_after = None
    if len(closes):
        i0 = np.searchsorted(tts, gts[closes[0]]); zmin_after = tcp[i0:, 2].min()
    rows.append((os.path.basename(os.path.dirname(ep))[-20:], os.path.basename(ep)[-8:], m.get("task"), m.get("success"), dur, len(closes), len(opens), [round(x, 1) for x in tc], [round(float(z)) for z in z_at], None if zmin_after is None else round(float(zmin_after)), round(float(tcp[-1, 2])), int((obj == 2).any())))
print("session  ep  task  success  dur  n_close n_open  t_closes  z_at_closes  zmin_after_first_close  z_end  obj2")
for r in rows: print(r)
print("n =", len(rows), " with >=2 closes:", sum(1 for r in rows if r[5] >= 2), " re-descend >30mm after first close:", sum(1 for r in rows if r[9] is not None and r[8] and r[8][0] - r[9] > 30))
