#!/usr/bin/env python3
"""Diagnostic probe (Route 1 + Route 3): is there ANY current/force-proxy
signal hiding in the Robotiq 2F-85's URCap socket variable set on THIS rig?

NOT part of the PHANTOM pipeline -- does not import or modify any
phantom.* module (except read-only use of the existing config loader to
avoid hardcoding this rig's IP/port a second time). Opens its OWN, separate
TCP connection to the URCap socket; never touches phantom's live connection.

READ-ONLY / SAFE: sends ONLY "GET <VAR>" -- this script NEVER sends a SET or
move command, so it cannot actuate the gripper and is safe to run alongside
(or instead of) a live collect.sh session. It will not interfere with the
production GripperPilot; the URCap socket bridge answers each client's GET
independently.

Routes covered in one pass (so one squeeze test gives you both answers):
  Route 1 (preferred): COU, MSC, PCO, DST -- undocumented-but-answering
      registers that MIGHT carry a real current/force signal. Unknown until
      tested under load (the original probe found them at 0 while idle,
      which proves nothing).
  Route 3 (fallback, "guaranteed to work"): PRE - POS -- position-tracking
      error while the gripper is commanded to close further than it can
      physically reach (an object in the jaws). POS/PRE/OBJ are confirmed
      answering already, so this is a proxy, not a new unknown.
  (Route 2 -- RTDE tool_output_current -- is a different transport and is
  covered by probe_rtde_tool_current.py instead.)

HOW TO USE
----------
1. (Optional but recommended for a clean first read) stop collect.sh so
   nothing else is mid-move on the gripper while you read baseline values.
   Not required -- this script only ever GETs, never SETs -- but a quiet
   baseline is easier to read.
2. Run, from the phantom-icra-2027 venv:
     .venv/bin/python diagnostics/probe_gripper_current.py --seconds 25
3. For the first ~5s just let it sit (gripper open, unloaded) -- this is
   your baseline. Then, using the teleop leader (if a session is up) or by
   jogging the gripper from the UR pendant's Robotiq panel, CLOSE the
   gripper on something solid enough that it stalls (can't reach the
   commanded position) for the remaining ~15-20s. Then let it open again
   for the last couple seconds if you can.
4. It prints one line per sample AND writes a CSV
   (default ./gripper_probe_<timestamp>.csv) with every column, plus a
   min/max/range summary per column at the end -- any column whose range
   jumps meaningfully during the "loaded" window (vs. the open baseline)
   is a viable channel. A column that never moves at all (or a column
   whose GET always failed, marked UNSUPP) is not.

Registers polled: POS, OBJ, STA, PRE, CUR, COU, MSC, PCO, DST.
"""

from __future__ import annotations

import argparse
import csv
import socket
import sys
import time
from pathlib import Path

# repo root must be on sys.path for this read-only config import to work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phantom.config.hardware import load_hardware  # noqa: E402

REGISTERS = ["POS", "OBJ", "STA", "PRE", "CUR", "COU", "MSC", "PCO", "DST"]


def _cmd(sock: socket.socket, text: str) -> str:
    sock.sendall((text + "\n").encode("ascii"))
    return sock.recv(1024).decode("ascii").strip()


def _get(sock: socket.socket, var: str) -> int | None:
    """Returns the int value, or None if this URCap build doesn't expose it
    (answers "VAR ?") or the response was malformed."""
    try:
        resp = _cmd(sock, f"GET {var}")
    except (OSError, socket.timeout):
        return None
    parts = resp.split()
    if len(parts) != 2 or parts[0] != var or parts[1] == "?":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", default="configs/hardware.nuc.yaml")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--hz", type=float, default=20.0,
                    help="poll rate (kept modest -- each tick is up to 9 "
                         "round-trips over the same socket)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hw = load_hardware(args.hardware)
    ip, port = hw.arm.ip, hw.gripper.port
    out_path = Path(args.out) if args.out else Path(
        f"gripper_probe_{int(time.time())}.csv")

    print(f"connecting to gripper URCap socket at {ip}:{port} (read-only GETs only)...")
    sock = socket.create_connection((ip, port), timeout=2.0)
    print("connected. Polling", ", ".join(REGISTERS))
    print(f"writing {out_path}")
    print()
    print(f"{'t(s)':>7}  " + "  ".join(f"{r:>5}" for r in REGISTERS) + "   PRE-POS")

    period = 1.0 / args.hz
    t0 = time.perf_counter()
    rows: list[dict] = []
    next_t = t0
    try:
        while time.perf_counter() - t0 < args.seconds:
            t = time.perf_counter() - t0
            vals = {r: _get(sock, r) for r in REGISTERS}
            pre_pos = (None if vals["PRE"] is None or vals["POS"] is None
                       else vals["PRE"] - vals["POS"])
            rows.append({"t": t, **vals, "PRE_MINUS_POS": pre_pos})
            fmt = lambda v: "  ?  " if v is None else f"{v:5d}"
            print(f"{t:7.2f}  " + "  ".join(fmt(vals[r]) for r in REGISTERS)
                  + f"   {fmt(pre_pos)}")
            next_t += period
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("\ninterrupted by user")
    finally:
        sock.close()

    if not rows:
        print("no samples collected")
        return 1

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"=== summary over {len(rows)} samples, {rows[-1]['t']:.1f}s ===")
    print(f"{'column':>14}  {'min':>6}  {'max':>6}  {'range':>6}  verdict")
    cols = REGISTERS + ["PRE_MINUS_POS"]
    for col in cols:
        vals = [r[col] for r in rows if r[col] is not None]
        n_unsupp = sum(1 for r in rows if r[col] is None)
        if not vals:
            print(f"{col:>14}  {'--':>6}  {'--':>6}  {'--':>6}  "
                  f"UNSUPPORTED by this URCap build ({n_unsupp}/{len(rows)} GETs failed)")
            continue
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        verdict = "VARIED -- worth a closer look" if rng > 0 else "constant (no signal seen)"
        print(f"{col:>14}  {lo:6d}  {hi:6d}  {rng:6d}  {verdict}")
    print()
    print("Look for a column whose range grew specifically during the window you")
    print("were squeezing an object -- open the CSV and check timestamps against")
    print("when you actually closed the gripper on something.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
