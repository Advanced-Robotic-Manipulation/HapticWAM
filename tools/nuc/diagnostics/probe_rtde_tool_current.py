#!/usr/bin/env python3
"""Diagnostic probe (Route 2): can we read tool_output_current -- the
UR controller's own current draw on the tool connector, which powers the
gripper -- over RTDE on THIS rig's ur_rtde install and firmware?

NOT part of the PHANTOM pipeline -- does not modify or import any
phantom.* driver module. Opens its OWN, separate RTDEReceiveInterface
(read-only transport; UR's RTDE protocol supports multiple simultaneous
receive clients -- see phantom/recording/workers.py's ArmStateWorker for the
same reasoning already relied on in production). This script never sends a
single control command -- RTDEReceiveInterface has no way to move anything.

This has TWO independent unknowns, both checked here:
  (a) Is "tool_output_current" a field this robot's firmware actually
      publishes over RTDE at all? (checked by asking ur_rtde to subscribe to
      it explicitly and seeing whether the connection is accepted)
  (b) Does the ur_rtde Python package expose a getter for it? (checked by
      introspecting the connected object -- ur_rtde's public API reference
      has no documented getToolOutputCurrent()/getActualToolCurrent(), so
      this is genuinely unknown until checked against your installed
      version)

If ur_rtde does NOT expose a getter for a field it CAN subscribe to, this
script also checks whether the separate, official "RTDE_Python_Client
Library" (UniversalRobots/RTDE_Python_Client_Library on GitHub -- a
different, lower-level package, import name `rtde`, NOT the `rtde_receive`
ur_rtde already uses) is installed, since that one supports ANY named RTDE
field generically. It is NOT installed by this script -- if useful, it
prints the exact install command for you to run yourself.

HOW TO USE
----------
Run, from the phantom-icra-2027 venv:
    .venv/bin/python diagnostics/probe_rtde_tool_current.py

This first part needs NO physical action -- it's pure introspection + a
connection test. If it finds a working getter, it will then poll it for
--seconds (default 15s) and print live values; squeeze something into the
gripper jaws during that window the same way as the other probe script.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phantom.config.hardware import load_hardware  # noqa: E402

CANDIDATE_NAMES = [
    "getToolOutputCurrent", "getActualToolCurrent", "getToolCurrent",
    "getActualToolOutputCurrent",
]


def introspect(recv) -> list[str]:
    return sorted(m for m in dir(recv)
                 if re.search(r"current|tool", m, re.IGNORECASE) and not m.startswith("_"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", default="configs/hardware.nuc.yaml")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    hw = load_hardware(args.hardware)
    ip = hw.arm.ip

    try:
        import rtde_receive
    except ImportError:
        print("ur_rtde (rtde_receive) is not importable in this venv -- "
              "unexpected, since the production driver depends on it.")
        return 1

    print(f"connecting default RTDEReceiveInterface({ip!r}) (read-only)...")
    recv = rtde_receive.RTDEReceiveInterface(ip)
    print("connected.\n")

    print("=== step (b): does the DEFAULT connection expose a tool-current getter? ===")
    found_default = introspect(recv)
    print("methods with 'current' or 'tool' in the name:")
    for m in found_default:
        print(f"  {m}")
    hit = next((m for m in CANDIDATE_NAMES if hasattr(recv, m)), None)
    if hit is None:
        print("-> none of the expected names exist on this ur_rtde build "
              "(matches the public API reference, which documents no such "
              "getter as of 1.6.3).")
    recv.disconnect()

    print()
    print("=== step (a): will the firmware even ACCEPT a subscription to "
         "'tool_output_current'? ===")
    try:
        recv2 = rtde_receive.RTDEReceiveInterface(
            ip, variables=["timestamp", "tool_output_current"])
        print("-> subscription ACCEPTED: this robot's firmware does publish "
             "tool_output_current over RTDE.")
        found_custom = introspect(recv2)
        new_methods = [m for m in found_custom if m not in found_default]
        if new_methods:
            print("   new getter(s) appeared after requesting it explicitly:")
            for m in new_methods:
                print(f"     {m}")
        else:
            print("   ...but no new getter method appeared -- ur_rtde accepted "
                 "the field into the wire recipe but doesn't expose a Python "
                 "accessor for it. This is the case the original chat "
                 "flagged as unverified, and it looks like the answer is "
                 "'accepted by firmware, not exposed by this ur_rtde build'.")
        recv2.disconnect()
    except Exception as e:
        print(f"-> subscription REJECTED ({type(e).__name__}): {e}")
        print("   this firmware/robot does not publish tool_output_current "
             "over RTDE at all -- Route 2 is a dead end on this rig via "
             "ur_rtde, regardless of Python bindings.")
        found_custom = []
        new_methods = []

    hit_name = hit or (new_methods[0] if new_methods else None)
    if hit_name:
        print(f"\nfound a usable getter: {hit_name}() -- polling it for "
             f"{args.seconds:.0f}s. Squeeze something into the gripper jaws now.")
        recv3 = rtde_receive.RTDEReceiveInterface(
            ip, variables=["timestamp", "tool_output_current"])
        getter = getattr(recv3, hit_name)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            t = time.perf_counter() - t0
            print(f"  t={t:6.2f}s  {hit_name}() = {getter()}")
            time.sleep(0.1)
        recv3.disconnect()
        return 0

    print()
    print("=== fallback: the official generic RTDE client (different package) ===")
    try:
        import rtde.rtde as rtde_ll  # noqa: F401
        print("the generic 'rtde' package IS installed -- it can subscribe to "
             "ANY named field regardless of ur_rtde's fixed getter set. Ask me "
             "to write the generic-field probe against it if step (a) above "
             "said the firmware accepts tool_output_current.")
    except ImportError:
        print("the generic 'rtde' package is NOT installed. If step (a) above "
             "said the firmware ACCEPTS tool_output_current but ur_rtde has no "
             "getter for it, that package is the way to reach it generically:")
        print("  pip install git+https://github.com/UniversalRobots/"
             "RTDE_Python_Client_Library.git@main")
        print("(a NEW, separate diagnostic dependency -- not required by "
             "PHANTOM itself -- only install it if you want to pursue this.)")

    print()
    print("VERDICT: Route 2 (RTDE tool_output_current) is "
         f"{'REACHABLE via a Python getter' if hit_name else 'NOT reachable via ur_rtde on this build'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
