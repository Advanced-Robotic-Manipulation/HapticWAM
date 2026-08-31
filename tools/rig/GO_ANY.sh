#!/bin/bash
# usage: [CKPT=...] [EXTRA="--home-joints --max-tcp-speed 0.15"] GO_ANY.sh <task> [episodes] [nfe] [guidance]
set -e
TASK=${1:?task}; EPS=${2:-3}; NFE=${3:-5}; G=${4:-1.0}
CKPT=${CKPT:-runs/teacher_v4_790eps/DEMO.pt}
cd ~/phantom-icra-2027/phantom
ping -c 1 -W 2 192.168.88.56 >/dev/null && echo "arm: OK" || { echo "arm NOT reachable"; exit 1; }
.venv/bin/python -c "
import pyrealsense2 as rs
d = list(rs.context().query_devices()); assert d, \"NO CAMERA\"
u = d[0].get_info(rs.camera_info.usb_type_descriptor); print(\"camera:\", u); assert u.startswith(\"3\")"
echo "== launching $TASK x$EPS nfe=$NFE guidance=$G (model=$CKPT) extra=[$EXTRA] =="
exec .venv/bin/python -m phantom.scripts.run_deploy --system teacher \
    --ckpt "$CKPT" --ema \
    --task "$TASK" --hardware configs/hardware.nuc.yaml \
    --episodes "$EPS" --device cuda --persistent-noise --nfe "$NFE" --guidance "$G" $EXTRA
