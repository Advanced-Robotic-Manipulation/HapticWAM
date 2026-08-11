"""Analyze one or more PHANTOM episodes and print the stream table.
Run on the NUC: .venv/bin/python analyze_episodes_obj.py <ep_dir1> <ep_dir2> ...
"""
import sys
import glob
import os
import numpy as np
import zarr

CFG_HZ = {
    "actions": 10.0,
    "actions_abs": 10.0,
    "arm_q": 125.0,
    "arm_qd": 125.0,
    "arm_ft": 125.0,
    "arm_tcp_pose": 125.0,
    "arm_tcp_speed": 125.0,
    "gripper": 100.0,
    "camera_scene_color": 15.0,
    "tactile_left_fields_ds": 8.0,
    "tactile_right_fields_ds": 8.0,
    "tactile_left_wrench": 8.0,
    "tactile_right_wrench": 8.0,
    "tactile_left_area": 8.0,
    "tactile_right_area": 8.0,
    "tactile_left_keyframes": 3.0,
    "tactile_right_keyframes": 3.0,
    "tactile_left_infer_img": 8.0,
    "tactile_right_infer_img": 8.0,
    "tactile_left_cop": 8.0,
    "tactile_right_cop": 8.0,
    "tactile_left_slip": 8.0,
    "tactile_right_slip": 8.0,
    "tactile_left_mask_frac": 8.0,
    "tactile_right_mask_frac": 8.0,
    "tactile_left_events": 8.0,
    "tactile_right_events": 8.0,
}

GAP_FACTOR = 2.5


def find_ep_roots(episode_dir):
    """A given path may itself be an ep_* dir, may directly contain one, or
    (a multi-episode collection session) may contain SEVERAL ep_* dirs --
    return all of them, sorted."""
    if os.path.basename(episode_dir.rstrip("/")).startswith("ep_"):
        return [episode_dir]
    matches = glob.glob(os.path.join(episode_dir, "**", "ep_*"), recursive=True)
    matches = sorted(m for m in matches if os.path.isdir(m))
    if matches:
        return matches
    return [episode_dir]


def human_mb(n_bytes):
    return n_bytes / (1024 * 1024)


def analyze(ep_root):
    zarr_dirs = sorted(glob.glob(os.path.join(ep_root, "*.zarr")))
    rows = []
    duration = None
    # first pass: figure out episode duration from arm_q (or any high-rate stream) timestamps
    for zd in zarr_dirs:
        name = os.path.basename(zd)[:-5]
        if name == "arm_q":
            g = zarr.open_group(zd, mode="r")
            if "ts" in g:
                ts = np.asarray(g["ts"][:])
                duration = float(ts[-1] - ts[0])
    for zd in zarr_dirs:
        name = os.path.basename(zd)[:-5]
        g = zarr.open_group(zd, mode="r")
        keys = list(g.array_keys())
        data_key = "data" if "data" in keys else keys[0]
        arr = g[data_key]
        n = arr.shape[0]
        shape_tail = arr.shape[1:]
        dtype = arr.dtype
        nbytes_stored = 0
        for k in keys:
            try:
                nbytes_stored += g[k].nbytes_stored
            except Exception:
                pass
        mb = human_mb(nbytes_stored)
        per_sample = nbytes_stored / n if n else 0
        chunk = arr.chunks[0] if arr.chunks else n

        ts = None
        for tkey in ("ts", "t_host", "timestamp", "t"):
            if tkey in keys:
                ts = np.asarray(g[tkey][:])
                break
        if ts is not None and len(ts) > 1:
            span = float(ts[-1] - ts[0])
            hz = (n - 1) / span if span > 0 else 0.0
            dt = np.diff(ts)
            med = np.median(dt) if len(dt) else 0.0
            gaps = int(np.sum(dt > GAP_FACTOR * med)) if med > 0 else 0
        else:
            span = duration if duration else 0.0
            hz = (n - 1) / span if span and span > 0 else 0.0
            gaps = None

        cfg = CFG_HZ.get(name)
        hit = f"{min(100, round(100 * hz / cfg))}%" if cfg else "--"

        def fmt_bytes(b):
            if b >= 1024:
                return f"{b/1024:.1f} KB"
            return f"{b:.0f} B"

        shape_str = "x".join(str(d) for d in shape_tail) if shape_tail else "1"
        rows.append({
            "stream": name, "MB": round(mb, 1), "n": n,
            "Hz": round(hz, 1), "cfg": cfg if cfg else "--",
            "hit": hit, "chunk": chunk, "per_sample": fmt_bytes(per_sample),
            "shape_dtype": f"{shape_str} {dtype}", "gaps": gaps if gaps is not None else "--",
        })
    return rows, duration


def main():
    for episode_dir in sys.argv[1:]:
        for ep_root in find_ep_roots(episode_dir):
            print(f"\n=== {ep_root} ===")
            rows, duration = analyze(ep_root)
            print(f"duration: {duration:.2f}s" if duration else "duration: unknown")
            header = f"{'stream':<28}{'MB':>7}{'n':>6}{'Hz':>7}{'cfg':>6}{'hit':>6}{'chunk':>7}{'per-sample':>12}{'shape/dtype':>16}{'gaps':>6}"
            print(header)
            for r in sorted(rows, key=lambda r: r["stream"]):
                print(f"{r['stream']:<28}{r['MB']:>7}{r['n']:>6}{r['Hz']:>7}{str(r['cfg']):>6}"
                      f"{r['hit']:>6}{r['chunk']:>7}{r['per_sample']:>12}{r['shape_dtype']:>16}{str(r['gaps']):>6}")


if __name__ == "__main__":
    main()
