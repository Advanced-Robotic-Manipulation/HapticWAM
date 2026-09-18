#!/bin/bash
# after esuite3: E2 at NFE 50 on ONE slow episode (8 seeds) — bimodality check without blocking the queue
[ -e ~/phantom-icra-2027/esuite4.lock ] && exit 0; touch ~/phantom-icra-2027/esuite4.lock
until grep -q "ALL DONE3" ~/phantom-icra-2027/esuite3.log; do sleep 60; done
cd ~/phantom-icra-2027/phantom
D=$HOME/phantom-icra-2027/data/episodes/deploy/20260828
.venv/bin/python tools/replay_rig.py --hardware configs/hardware.nuc.yaml --ckpt runs/teacher_v5_batch0822/v5_6.pt --episodes $D/ep_teacher_waffles_1787923675_000 --nfe 50 --seeds 8 --out runs/replay/e2_nfe50_1ep.json > runs/replay/e2_nfe50_1ep.log 2>&1; echo "e2_nfe50_1ep rc=$? $(date -u +%H:%M:%S)"
echo "ALL DONE4 $(date -u +%H:%M:%S)"
