#!/bin/bash
# READ_TCP.sh — print the arm's live TCP (base frame, mm + rotvec) and joints, read-only over a
# second RTDE receive socket (safe while a policy server or run_deploy is idle; do not run during
# an episode). Use it to RECORD a reference pose, e.g. jog the gripper to the correct egg-holder
# place pose and run this once — the number goes into the run sheet / verdict notes.
#   ./READ_TCP.sh            prints once
#   ./READ_TCP.sh 5          prints 5 samples one second apart
BASE=${PHANTOM_RIG_BASE:-$HOME/phantom-icra-2027}
N=${1:-1}
cd "$BASE/phantom" && .venv/bin/python - "$N" <<'PY'
import sys, time, numpy as np
from rtde_receive import RTDEReceiveInterface
r = RTDEReceiveInterface("192.168.88.56")
for i in range(int(sys.argv[1])):
    p = np.array(r.getActualTCPPose()); q = np.degrees(r.getActualQ())
    print(f"TCP xyz mm: {p[0]*1000:8.1f} {p[1]*1000:8.1f} {p[2]*1000:8.1f}   rotvec: {p[3]:.3f} {p[4]:.3f} {p[5]:.3f}   "
          f"q deg: {' '.join(f'{v:.1f}' for v in q)}", flush=True)
    if i + 1 < int(sys.argv[1]): time.sleep(1.0)
PY
