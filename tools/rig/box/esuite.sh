#!/bin/bash
cd ~/phantom-icra-2027/phantom
D=$HOME/phantom-icra-2027/data/episodes/deploy/20260828
V5="$D/ep_teacher_waffles_1787923675_000 $D/ep_teacher_waffles_1787929940_000 $D/ep_teacher_waffles_1787930224_002"
V4="$D/ep_teacher_waffles_1787922904_000 $D/ep_teacher_waffles_1787923361_000"
C5=runs/teacher_v5_batch0822/v5_6.pt; C4=runs/teacher_v4_790eps/teacher_020000.pt; HW=configs/hardware.nuc.yaml
mkdir -p runs/replay
run(){ tag=$1; shift; echo "== $tag $(date -u +%H:%M:%S)"; .venv/bin/python tools/replay_rig.py --hardware $HW --persistent-noise "$@" --out runs/replay/$tag.json > runs/replay/$tag.log 2>&1; echo "$tag rc=$? $(date -u +%H:%M:%S)"; }
run e0_v5 --ckpt $C5 --episodes $V5 --nfe 5 --seeds 8 --prev-chunk proposal --prev-cpk chained
run e0_v4 --ckpt $C4 --episodes $V4 --nfe 5 --seeds 8 --prev-chunk proposal --prev-cpk chained
run e3_measured_v5 --ckpt $C5 --episodes $V5 --nfe 5 --seeds 8 --prev-chunk measured
run e3_zeros_v5 --ckpt $C5 --episodes $V5 --nfe 5 --seeds 8 --prev-chunk zeros
run e3_nocpk_v5 --ckpt $C5 --episodes $V5 --nfe 5 --seeds 8 --prev-cpk none
run e2_nfe1_v5 --ckpt $C5 --episodes $V5 --nfe 1 --seeds 8
run e2_nfe10_v5 --ckpt $C5 --episodes $V5 --nfe 10 --seeds 16
run e2_nfe50_v5 --ckpt $C5 --episodes $V5 --nfe 50 --seeds 16
echo "ALL DONE $(date -u +%H:%M:%S)"
