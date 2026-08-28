"""Closed-loop simulation: a velocity-continuous 1-NN imitation policy (state = z, vz, gripper)
driven through the deploy loop (obs -> latency L -> submit -> executor plays from index 0 anchored at
last_cmd, back-to-back replans). Question: does loop timing alone make a demo-faithful policy descend
slowly and/or stop high?  Units: mm, mm/s, gripper 0..1.  RATE 10 Hz, H=16."""
import numpy as np
RATE, H, DT = 10.0, 16, 0.1
SZ, SV, SG = 89.0, 66.0, 0.19        # norm-stat stds: tcp z, tcp_speed z, gripper pos

def demo(z0=300., zstop=65., vmax=47., acc_s=1.0, dec_s=1.0, g_open=0.23, g_close=0.50, seed=0):
    """one demo: rest -> accelerate -> cruise -> decelerate to zstop -> close (0.5 s) -> hold (0.5 s) -> lift."""
    z, v, g = [z0], [0.0], [g_open]
    d_dec = 0.5 * vmax * dec_s
    while True:
        rem = z[-1] - zstop
        if rem <= 0.5: break
        if v[-1] < vmax and rem > d_dec + 0.5 * vmax * acc_s:
            vn = min(vmax, v[-1] + vmax / acc_s * DT)
        elif rem < d_dec:
            vn = max(3.0, vmax * np.sqrt(max(rem, 0) / d_dec))
        else:
            vn = vmax
        z.append(z[-1] - vn * DT); v.append(vn); g.append(g_open)
    for k in range(5):                      # close over 0.5 s at rest
        z.append(z[-1]); v.append(0.0); g.append(g_open + (g_close - g_open) * (k + 1) / 5)
    for k in range(5):                      # hold
        z.append(z[-1]); v.append(0.0); g.append(g_close)
    for k in range(20):                     # lift 100 mm/s
        z.append(z[-1] + 10.0); v.append(-100.0); g.append(g_close)
    return np.array(z), np.array(v), np.array(g)

class NNPolicy:
    """1-NN over a demo library in normalised (z, vz, g); returns the next H (dz, g) rows."""
    def __init__(self, demos, use_v=True, use_g=True):
        self.lib = []
        for z, v, g in demos:
            dz = np.diff(z); 
            for k in range(len(dz) - H):
                self.lib.append((z[k], -v[k], g[k], dz[k:k + H], g[k + 1:k + 1 + H]))
        self.Z = np.array([r[0] for r in self.lib]); self.V = np.array([r[1] for r in self.lib]); self.G = np.array([r[2] for r in self.lib])
        self.use_v, self.use_g = use_v, use_g
    def __call__(self, z, vz, g):
        d = ((self.Z - z) / SZ) ** 2
        if self.use_v: d += ((self.V - vz) / SV) ** 2
        if self.use_g: d += ((self.G - g) / SG) ** 2
        k = int(np.argmin(d)); return self.lib[k][3].copy(), self.lib[k][4].copy()

def run(policy, L=0.95, mode="index0", T=14.0, tick=1 / 125., z0=300., g0=0.23):
    z_cmd, v_cmd, g_cmd = z0, 0.0, g0
    t = 0.0; plan = None; play = 0.0; pending = None; next_obs = 0.0
    log = []
    while t < T:
        if pending is not None and t >= pending[0]:
            _, dz, gg, t_obs, z_at_obs = pending
            play = 0.0 if mode == "index0" else (t - t_obs)   # skip-head = phase-correct playback
            cum = np.cumsum(dz); u = min(play * RATE, H - 1e-6); k = int(u)
            c0 = (cum[k - 1] if k > 0 else 0.0) + (u - k) * (cum[k] - (cum[k - 1] if k > 0 else 0.0))
            plan = (z_cmd - c0, cum, gg); pending = None
        if t >= next_obs and pending is None:
            dz, gg = policy(z_cmd, v_cmd, g_cmd)             # observes the CURRENT commanded state
            pending = (t + L, dz, gg, t, z_cmd); next_obs = t + L
        if plan is not None:
            play += tick; u = min(play * RATE, H - 1e-6); k = int(u)
            base, cum, gg = plan; prev = cum[k - 1] if k > 0 else 0.0
            z_new = base + prev + (u - k) * (cum[k] - prev)
            v_cmd = (z_new - z_cmd) / tick; z_cmd = z_new; g_cmd = gg[min(k, H - 1)]
        log.append((t, z_cmd, v_cmd, g_cmd)); t += tick
    return np.array(log)

def report(tag, log):
    t, z, v, g = log.T
    close = np.nonzero(g > 0.45)[0]
    zc = z[close[0]] if len(close) else float('nan'); tc = t[close[0]] if len(close) else float('nan')
    cruise = (z < 250) & (z > 120) & (t < (tc if len(close) else 99))
    vmean = -v[cruise].mean() if cruise.any() else float("nan")
    print(f"{tag:46s} descent(250..120mm) {vmean:5.1f} mm/s   z_min {z.min():6.1f}   close at z={zc:6.1f} t={tc:4.1f}s")

if __name__ == "__main__":
    lib = [demo(z0=z0, vmax=vm, zstop=zs) for z0 in (280., 300., 320.) for vm in (42., 47., 52.) for zs in (55., 65., 72.)]
    print("demo: cruise 47 mm/s, stop 65 mm, close 0.5 s later\n")
    for use_v in (False, True):
        pol = NNPolicy(lib, use_v=use_v, use_g=True)
        for L in (0.0, 0.5, 0.95, 1.3):
            for mode in ("index0", "skip"):
                if L == 0.0 and mode == "skip": continue
                report(f"NN(z{',v' if use_v else ''},g)  L={L:.2f} {mode}", run(pol, L=L, mode=mode))
        print()
