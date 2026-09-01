#!/bin/bash
# live wrist-3 readout while jogging: run me and watch; Ctrl-C when done
~/phantom-icra-2027/phantom/.venv/bin/python - <<'PY'
import math, time
import rtde_receive
r = rtde_receive.RTDEReceiveInterface("192.168.88.56")
print("jog wrist 3 (bottom joint) with the MINUS arrow; target: -178\n")
while True:
    w = math.degrees(r.getActualQ()[5])
    bar = "#" * max(0, int((w + 200) / 8))
    mark = "  <-- GOOD, STOP HERE" if -185 < w < -170 else ""
    print(f"\rwrist3: {w:7.1f} deg {mark}   ", end="", flush=True)
    time.sleep(0.25)
PY
