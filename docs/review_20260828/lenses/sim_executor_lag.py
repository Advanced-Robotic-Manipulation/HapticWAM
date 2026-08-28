"""Executor timing simulation: a PERFECT state-feedback policy (returns the demo
deltas for the phase matching the observed z) driven through the deploy loop
(obs -> latency L -> submit -> play from index 0 anchored at last_cmd)."""
import numpy as np

RATE = 10.0; H = 16
def demo_profile(z0=0.20, zstop=0.072, v_max=0.047, decel_s=1.0, dt=1/RATE, T=8.0):
    # descend at v_max, then decelerate linearly to 0 over `decel_s` ending at zstop
    z = [z0]; t = 0.0
    d_decel = 0.5 * v_max * decel_s
    while t < T:
        rem = z[-1] - zstop
        if rem <= 0: v = 0.0
        elif rem < d_decel: v = v_max * np.sqrt(rem / d_decel)   # linear decel in time ~ sqrt in distance
        else: v = v_max
        z.append(z[-1] - v * dt); t += dt
    return np.array(z)

def policy(z_obs, z_demo):
    """deltas for the 16 steps following the demo phase whose z matches z_obs"""
    k = int(np.argmin(np.abs(z_demo - z_obs)))
    fut = z_demo[k + 1:k + 1 + H] - z_demo[k:k + H]
    if len(fut) < H: fut = np.concatenate([fut, np.zeros(H - len(fut))])
    return fut

def run(L=0.9, mode="index0", T=12.0, tick=1/125):
    z_demo = demo_profile()
    z_cmd = z_demo[0]; t = 0.0
    plan = None; play = 0.0; pending = None
    zs, ts = [], []
    next_obs = 0.0
    while t < T:
        if pending is not None and t >= pending[0]:
            _, deltas, t_obs = pending
            if mode == "index0":
                play = 0.0                 # current executor: play from index 0 at last_cmd
            else:
                play = (t - t_obs)         # skip the elapsed head, anchored at last_cmd
            cum = np.cumsum(deltas); u = min(play * RATE, H - 1e-6); k = int(u)
            c0 = (cum[k-1] if k > 0 else 0.0) + (u - k) * (cum[k] - (cum[k-1] if k > 0 else 0.0))
            plan = (z_cmd - c0, cum); pending = None
        if t >= next_obs and pending is None:
            deltas = policy(z_cmd, z_demo)         # perfect policy, observes the current pose
            pending = (t + L, deltas, t); next_obs = t + L
        if plan is not None:
            play += tick
            u = min(play * RATE, H - 1e-6); k = int(u)
            base, cum = plan
            prev = cum[k-1] if k > 0 else 0.0
            z_cmd = base + prev + (u - k) * (cum[k] - prev)
        zs.append(z_cmd); ts.append(t); t += tick
    return np.array(ts), np.array(zs), z_demo

for L in (0.0, 0.9, 1.3):
    for mode in ("index0", "skip"):
        ts, zs, zd = run(L=L, mode=mode)
        t_demo = np.arange(len(zd)) / RATE
        # descent speed over the constant-velocity phase, final z, time to reach within 5mm of stop
        i_fast = (zs > 0.12) & (zs < 0.19)
        v = -np.gradient(zs, ts)[i_fast].mean() * 1000 if i_fast.any() else float('nan')
        z_min = zs.min() * 1000
        print(f"L={L:.1f} mode={mode:6s} descent {v:5.1f} mm/s  z_min {z_min:6.1f} mm  (demo stop 72.0, demo v {47:.0f})")
