"""Replay the descend-then-release supervisor over the 31 latched carries of 2026-09-12.

The real `DescendThenReleaseSupervisor` is driven tick by tick (125 Hz) over a trajectory
reconstructed from the forensics table, so what this prints is the shipped state machine's
own decision, not a paper model of it.

Reconstruction, per carry, over the 3 s window that ends at the halt (the `record_tail_s`
tail excluded), from `phantom/logs/run_deploy_*.log` + `stop.json` + the arm zarr as
tabulated in the 09-12 forensics follow-up:

  measured z  linear from `z 3 s before halt` to `z at halt` (the table's mean vz agrees)
  measured y  held at `y at halt`
  commanded z equal to the measured z (NO command lead), which UNDERSTATES how often the
              crate floor binds: while the arm is blocked the executor's commanded pose
              keeps descending away from the measured one
  latch       held for the whole window, except that a carry which released naturally
              unlatches when its measured z crosses the height the log reports
  carry apex  seeded at 0.35 m before the window, satisfying the lift gate (all 31 rows
              are lift >= 50 mm carries whose apex predates the window)

Known limits, stated rather than hidden: the two-sample z model cannot show a descent that
flattens inside the last few hundred ms (a jam), and a carry whose logged release height
lies outside the window's z range is marked `not evaluable` instead of guessed.

Run: .venv/bin/python tools/rig/analysis/descend_release_replay.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phantom.deploy.descend_then_release import (
    DescendThenReleaseConfig,
    DescendThenReleaseSupervisor,
)

CONFIG = Path("configs/placement_descent_rig_0913.json")
DT = 0.008
WINDOW_S = 3.0

# episode, arm, task, latch, released_z (None = never released), z 3 s before halt,
# z at halt, y at halt, vz mm/s, min z after apex, dF at halt (N)
CARRIES = [
    ("student_Carton_1789234301_002", "stu_ftA_r2", "Carton", 0.52, None, 0.3016, 0.2212, -0.0303, -27, 0.2211, 13.6),
    ("student_waffles_1789231488_001", "stu_ftA_r2", "waffles", 0.62, None, 0.2332, 0.1165, 0.0402, -39, 0.1131, 123.7),
    ("student_waffles_1789231581_003", "stu_ftA_r2", "waffles", 0.64, 0.123, 0.2100, 0.1084, 0.0914, -34, 0.1065, 68.1),
    ("student_waffles_1789231704_004", "stu_ftA_r2", "waffles", 0.64, None, 0.2440, 0.1238, 0.0829, -40, 0.1172, 171.3),
    ("student_waffles_1789231763_005", "stu_ftA_r2", "waffles", 0.63, 0.117, 0.1287, 0.0979, 0.0578, -10, 0.0966, 14.2),
    ("student_waffles_1789231831_006", "stu_ftA_r2", "waffles", 0.65, 0.105, 0.3302, 0.3184, -0.1470, -4, 0.3178, 4.9),
    ("student_waffles_1789231924_008", "stu_ftA_r2", "waffles", 0.68, 0.159, 0.2022, 0.1077, 0.0743, -32, 0.1033, 129.6),
    ("student_waffles_1789231983_009", "stu_ftA_r2", "waffles", 0.67, None, 0.2225, 0.1005, 0.0477, -41, 0.0978, 69.4),
    ("student_Carton_1789234063_007", "stu_v6_r2", "Carton", 0.56, None, 0.2512, 0.1332, 0.0489, -39, 0.1294, 83.9),
    ("student_waffles_1789232168_000", "stu_v6_r2", "waffles", 0.71, 0.160, 0.3618, 0.3203, -0.3266, -14, 0.3201, 37.8),
    ("student_waffles_1789232255_001", "stu_v6_r2", "waffles", 0.69, None, 0.3587, 0.2395, 0.0671, -40, 0.2387, 44.0),
    ("student_waffles_1789232513_006", "stu_v6_r2", "waffles", 0.71, 0.159, 0.3348, 0.3379, -0.2707, 1, 0.0928, 61.6),
    ("student_waffles_1789232608_008", "stu_v6_r2", "waffles", 0.71, None, 0.3448, 0.3282, 0.0297, -6, 0.3280, 42.9),
    ("student_waffles_1789232732_000", "stu_v6_r2", "waffles", 0.68, None, 0.3991, 0.2898, 0.0466, -36, 0.2889, 37.3),
    ("student_waffles_1789232787_001", "stu_v6_r2", "waffles", 0.76, 0.159, 0.3423, 0.3718, -0.2446, 10, 0.3715, 57.3),
    ("student_waffles_1789232963_000", "stu_v6_r2", "waffles", 0.74, None, 0.2977, 0.3766, 0.0070, 26, 0.3764, 66.0),
    ("student_waffles_1789233128_004", "stu_v6_r2", "waffles", 0.68, 0.123, 0.1956, 0.0954, -0.3043, -34, 0.0952, 13.9),
    ("teacher_Carton_1789236912_000", "v6", "Carton", 0.49, None, 0.3479, 0.3055, 0.0333, -14, 0.3050, 17.7),
    ("teacher_Carton_1789236959_001", "v6", "Carton", 0.54, None, 0.3013, 0.1329, 0.0654, -56, 0.1284, 57.7),
    ("teacher_Carton_1789237173_008", "v6", "Carton", 0.51, None, 0.3933, 0.3016, 0.0205, -31, 0.3015, 28.2),
    ("teacher_Carton_1789237857_004", "v6", "Carton", 0.52, None, 0.3564, 0.3883, 0.0119, 11, 0.3881, 100.1),
    ("teacher_waffles_1789225475_000", "v6", "waffles", 0.63, 0.112, 0.3930, 0.3792, -0.2018, -5, 0.3785, 40.8),
    ("teacher_waffles_1789227318_000", "v6", "waffles", 0.63, 0.079, 0.3608, 0.1580, -0.2332, -68, 0.1516, 183.3),
    ("teacher_waffles_1789227625_003", "v6", "waffles", 0.64, None, 0.3681, 0.3608, -0.0174, -2, 0.3607, 89.3),
    ("teacher_waffles_1789227696_004", "v6", "waffles", 0.63, None, 0.3330, 0.3273, 0.0311, -2, 0.3244, 36.4),
    ("teacher_waffles_1789228092_003", "v6", "waffles", 0.64, 0.078, 0.3465, 0.3914, -0.1552, 15, 0.3914, 57.8),
    ("teacher_waffles_1789228202_005", "v6", "waffles", 0.68, None, 0.3330, 0.2249, 0.0567, -36, 0.2245, 43.6),
    ("teacher_waffles_1789228261_006", "v6", "waffles", 0.65, None, 0.3319, 0.3147, 0.0399, -6, 0.3145, 36.2),
    ("teacher_waffles_1789228368_008", "v6", "waffles", 0.66, None, 0.1845, 0.0826, -0.0895, -34, 0.0790, 79.7),
    ("teacher_waffles_1789228444_009", "v6", "waffles", 0.65, 0.077, 0.3492, 0.1735, -0.2356, -59, 0.1734, 23.4),
    ("teacher_waffles_1789228724_000", "v6", "waffles", 0.63, None, 0.3066, 0.3714, -0.0152, 22, 0.3713, 42.1),
]
OPEN_APERTURE = {"waffles": 0.25, "Carton": 0.25}   # task demo start aperture (stats.gripper_mean)


def replay(row, config, *, y_gate=None):
    """Drive the real supervisor over one reconstructed carry."""
    ep, arm, task, latch, released_z, z0, z1, y, vz, z_min, df = row
    if y_gate is not None:
        config = DescendThenReleaseConfig(**{**config.to_dict(),
                                            "release_y_min_m": y_gate})
    sup = DescendThenReleaseSupervisor(config)
    # carry apex before the window (the lift gate); latched, high, off the band
    t = 0.0
    for _ in range(10):
        sup.step(t, measured_tcp=np.array([-0.35, y, 0.35, 0.0, 0.0, 0.0]),
                 latched=True, commanded_target=np.array([-0.35, y, 0.35, 0.0, 0.0, 0.0]),
                 policy_grip=latch, open_aperture=OPEN_APERTURE[task])
        t += DT
    t_nat = None
    if released_z is not None and min(z0, z1) - 1e-9 <= released_z <= max(z0, z1) + 1e-9:
        t_nat = WINDOW_S * (z0 - released_z) / (z0 - z1) if z0 != z1 else 0.0
    fired_at = fired_z = None
    n = int(WINDOW_S / DT)
    for k in range(n):
        s = k * DT
        z = z0 + (z1 - z0) * (s / WINDOW_S)
        latched = not (t_nat is not None and s >= t_nat)
        pose = np.array([-0.35, y, z, 0.0, 0.0, 0.0])
        d = sup.step(t + s, measured_tcp=pose, latched=latched,
                     commanded_target=pose.copy(), policy_grip=latch,
                     open_aperture=OPEN_APERTURE[task])
        if d.clear_latch and fired_at is None:
            fired_at, fired_z = s, z
    return {
        "ep": ep, "arm": arm, "task": task, "released_z": released_z,
        "z0": z0, "z1": z1, "y": y, "df": df,
        "t_nat": t_nat, "fired_at": fired_at, "fired_z": fired_z,
        "before_halt": None if fired_at is None else WINDOW_S - fired_at,
        "evaluable": released_z is None or t_nat is not None,
    }


def main():
    spec = json.loads(CONFIG.read_text())
    configs = {t: DescendThenReleaseConfig.from_spec(spec, t) for t in ("waffles", "Carton")}
    results = {}
    for label, y_gate in (("shipped (release y gate = 0.0, inside the crate)", None),
                          ("spec variant (release y gate = crate edge -0.10)", -0.10),
                          ("strict variant (release y gate = demo p10 release y 0.048)", 0.048)):
        rows = [replay(row, configs[row[2]], y_gate=y_gate) for row in CARRIES]
        results[label] = rows
        non = [r for r in rows if r["released_z"] is None]
        rel = [r for r in rows if r["released_z"] is not None]
        print(f"\n=== {label}")
        print(f"{'episode':38s} {'arm':11s} {'task':8s} {'nat':>6s} {'fires':>6s} "
              f"{'at z':>7s} {'s before halt':>14s}")
        for r in rows:
            nat = "-" if r["released_z"] is None else f"{r['released_z']:.3f}"
            if r["released_z"] is not None and not r["evaluable"]:
                nat += "*"
            print(f"{r['ep']:38s} {r['arm']:11s} {r['task']:8s} {nat:>6s} "
                  f"{'YES' if r['fired_at'] is not None else 'no':>6s} "
                  f"{'-' if r['fired_z'] is None else format(r['fired_z'], '.3f'):>7s} "
                  f"{'-' if r['before_halt'] is None else format(r['before_halt'], '.2f'):>14s}")
        fires_non = [r for r in non if r["fired_at"] is not None]
        print(f"-- non-releases converted: {len(fires_non)}/{len(non)}: "
              f"{', '.join(r['ep'] for r in fires_non)}")
        print(f"-- release height of those conversions: "
              f"{[round(r['fired_z'], 3) for r in fires_non]}")
        pre = [r for r in rel if r["evaluable"] and r["fired_at"] is not None
               and r["fired_at"] < r["t_nat"]]
        late = [r for r in rel if r["evaluable"] and (r["fired_at"] is None
                                                     or r["fired_at"] >= r["t_nat"])]
        unk = [r for r in rel if not r["evaluable"]]
        print(f"-- of the 12 natural releases: {len(pre)} preempted, {len(late)} not, "
              f"{len(unk)} not evaluable (logged release height outside the window)")
        for r in pre:
            print(f"   preempted {r['ep']}: natural {r['released_z']:.3f} m, "
                  f"supervisor {r['fired_z']:.3f} m, {r['t_nat'] - r['fired_at']:.2f} s earlier")
    # lever sweep: the forced release fires dwell_s after the tool enters the
    # release zone, so the gate height and the dwell together set the release
    # height. Lower gate = lower release, fewer conversions.
    print("\n=== lever sweep (shipped y gate), 19 non-releases")
    print(f"{'gate z':>7s} {'dwell':>6s} {'conversions':>12s} {'release z (m)':>28s}")
    for gate in (0.16, 0.15, 0.14, 0.13, 0.12):
        for dwell in (0.35, 0.5):
            fired = []
            for row in CARRIES:
                if row[4] is not None:
                    continue
                base = configs[row[2]].to_dict()
                base.update(release_gate_z_m=gate, dwell_s=dwell)
                if base["release_gate_z_m"] <= base["release_z_m"]:
                    fired = None
                    break
                r = replay(row, DescendThenReleaseConfig(**base))
                if r["fired_at"] is not None:
                    fired.append(round(r["fired_z"], 3))
            if fired is None:
                print(f"{gate:7.3f} {dwell:6.2f} {'n/a (gate below the Carton floor)':>41s}")
                continue
            print(f"{gate:7.3f} {dwell:6.2f} {len(fired):>9d}/19 {str(sorted(fired)):>28s}")

    # phases that must never fire
    cfgw = configs["waffles"]
    from phantom.deploy.descend_then_release import DescendThenReleaseSupervisor
    for phase, tcp, latched in (("reach (unlatched, over the object)", (-0.35, -0.27, 0.09), False),
                                ("carry apex (latched, over the crate)", (-0.35, 0.06, 0.35), True),
                                ("high stall (latched, over the crate, z 0.24-0.38)", (-0.35, 0.06, 0.28), True)):
        s = DescendThenReleaseSupervisor(cfgw)
        pose = np.array([*tcp, 0.0, 0.0, 0.0])
        fired = False
        for k in range(1000):
            d = s.step(k * DT, measured_tcp=pose, latched=latched,
                       commanded_target=pose.copy(), policy_grip=0.64, open_aperture=0.25)
            fired = fired or d.clear_latch
        print(f"\n{phase}: fires = {fired} (8 s of ticks)")


if __name__ == "__main__":
    main()
