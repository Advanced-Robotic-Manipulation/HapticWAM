#!/bin/bash
# PHANTOM v4 one-shot launch (2026-08-20). Just run: ~/phantom-icra-2027/GO.sh
set -e
cd ~/phantom-icra-2027/phantom
echo "== preflight =="
ping -c 1 -W 2 192.168.88.56 >/dev/null && echo "arm: OK" || { echo "arm NOT reachable — power the robot first"; exit 1; }
.venv/bin/python - <<'PY'
import pyrealsense2 as rs
d = list(rs.context().query_devices())
assert d, "NO CAMERA on USB"
usb = d[0].get_info(rs.camera_info.usb_type_descriptor)
print("camera:", usb)
assert usb.startswith("3"), "camera on USB2 — replug into the USB3 hub port"
PY
echo "== launching (waffles, 3 episodes, model=20k) =="
exec .venv/bin/python -m phantom.scripts.run_deploy --system teacher \
    --ckpt runs/teacher_v4_790eps/DEMO.pt --ema \
    --task waffles --hardware configs/hardware.nuc.yaml \
    --episodes 3 --device cuda --persistent-noise
