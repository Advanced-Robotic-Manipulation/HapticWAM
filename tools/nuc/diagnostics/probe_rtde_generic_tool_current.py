#!/usr/bin/env python3
"""Diagnostic probe (Route 2, generic fallback): read tool_output_current
via the OFFICIAL low-level RTDE client (package `rtde`, a different library
from the `ur_rtde`/`rtde_receive` PHANTOM already depends on).

Only useful if you've already run probe_rtde_tool_current.py and it printed:
  "subscription ACCEPTED" (firmware publishes the field) AND
  "no new getter method appeared" (ur_rtde has no Python accessor for it)
-- which is exactly what this rig showed on 2026-07-31: the E-series
controller DOES publish tool_output_current over RTDE, but the ur_rtde
Python bindings PHANTOM uses have no getter for it. This script reaches the
field directly, bypassing ur_rtde's fixed getter set entirely.

NOT part of the PHANTOM pipeline. Read-only: this uses RTDE's OUTPUT
recipe only (robot -> client), never an input/control recipe, so it cannot
command anything -- same safety property as RTDEReceiveInterface.

SETUP (only if you want to pursue this route)
----------------------------------------------
This needs a NEW, separate pip package -- not a PHANTOM dependency, purely
for this diagnostic:
    .venv/bin/pip install git+https://github.com/UniversalRobots/RTDE_Python_Client_Library.git@main

Then run:
    .venv/bin/python diagnostics/probe_rtde_generic_tool_current.py --seconds 25

TIMING (same protocol as probe_gripper_current.py -- this matters, an idle
-only run just re-confirms the baseline is 0 and proves nothing about the
route):
  0-5s:   leave the gripper alone (baseline, should read ~0)
  5-20s:  actually close the gripper on something solid enough to STALL it
          (via teleop leader or the pendant's Robotiq panel) -- it must be
          a real motor stall, not just a normal open/close motion
  20-25s: release / reopen (clean trailing baseline)

Writes every sample to a CSV (default ./rtde_current_probe_<timestamp>.csv,
override with --out) so the squeeze window can be checked after the fact,
exactly like probe_gripper_current.py's output.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phantom.config.hardware import load_hardware  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", default="configs/hardware.nuc.yaml")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else Path(
        f"rtde_current_probe_{int(time.time())}.csv")

    try:
        import rtde.rtde as rtde_ll
    except ImportError:
        print("the generic 'rtde' package is not installed. Install it with:")
        print("  pip install git+https://github.com/UniversalRobots/"
             "RTDE_Python_Client_Library.git@main")
        print("then re-run this script.")
        return 1

    hw = load_hardware(args.hardware)
    ip = hw.arm.ip

    names = ["timestamp", "tool_output_current"]
    types = ["DOUBLE", "DOUBLE"]

    print(f"connecting rtde.RTDE({ip!r}, 30004) (read-only OUTPUT recipe)...")
    con = rtde_ll.RTDE(ip, 30004)
    con.connect()
    con.get_controller_version()
    if not con.send_output_setup(names, types, frequency=args.hz):
        print("controller REJECTED the output recipe -- tool_output_current "
             "is not available on this firmware after all (unexpected, "
             "given the earlier ur_rtde subscription test accepted it).")
        con.disconnect()
        return 1
    if not con.send_start():
        print("controller refused to start data synchronization")
        con.disconnect()
        return 1

    print(f"writing every sample to {out_path}")
    print(f"streaming for {args.seconds:.0f}s -- squeeze something into the "
         "gripper jaws now.")
    t0 = time.perf_counter()
    n = 0
    lo = hi = None
    rows: list[dict] = []
    try:
        while time.perf_counter() - t0 < args.seconds:
            state = con.receive()
            if state is None:
                continue
            val = state.tool_output_current
            t = time.perf_counter() - t0
            n += 1
            lo = val if lo is None else min(lo, val)
            hi = val if hi is None else max(hi, val)
            rows.append({"t": t, "tool_output_current_mA": val})
            if n % 5 == 0:
                print(f"  t={t:6.2f}s  tool_output_current = {val:8.4f} mA")
    except KeyboardInterrupt:
        pass
    finally:
        con.send_pause()
        con.disconnect()

    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t", "tool_output_current_mA"])
            w.writeheader()
            w.writerows(rows)

    print()
    if n == 0:
        print("no samples received -- something is wrong with the recipe/connection.")
        return 1
    print(f"=== {n} samples, tool_output_current range: "
         f"[{lo:.4f}, {hi:.4f}] mA (delta {hi - lo:.4f} mA) ===")
    if hi - lo > 0:
        print("VARIED -- check the CSV/printed timeline against when you "
             "actually squeezed; if it tracks the squeeze, Route 2 works.")
    else:
        print("CONSTANT throughout -- no signal seen (but note this is the "
             "TOTAL tool-port current, which also powers any always-on "
             "tool electronics, not just the gripper motor -- it may simply "
             "be too coarse/noisy to show a 2F-85 stall).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
