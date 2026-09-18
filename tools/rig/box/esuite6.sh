#!/bin/bash
[ -e ~/phantom-icra-2027/esuite6.lock ] && exit 0; touch ~/phantom-icra-2027/esuite6.lock
until grep -q "ALL DONE5" ~/phantom-icra-2027/esuite5.log; do sleep 60; done
cd ~/phantom-icra-2027/phantom
D=$HOME/phantom-icra-2027/data/episodes/deploy/20260828
EPS="$D/ep_teacher_waffles_1787923675_000 $D/ep_teacher_waffles_1787929940_000 $D/ep_teacher_waffles_1787930224_002"
C5=runs/teacher_v5_batch0822/v5_6.pt; HW=configs/hardware.nuc.yaml
run(){ tag=$1; shift; echo "== $tag $(date -u +%H:%M:%S)"; .venv/bin/python tools/replay_deploy_path.py --hardware $HW "$@" --out runs/replay/$tag.json > runs/replay/$tag.log 2>&1; echo "$tag rc=$? $(date -u +%H:%M:%S)"; }
run e0d_deploypath_nfe5 --ckpt $C5 --episodes $EPS --nfe 5 --seeds 4 --persistent-noise
run e0d_deploypath_nfe5_nomerge --ckpt $C5 --episodes $EPS --nfe 5 --seeds 4 --persistent-noise --no-merge
echo "ALL DONE6 $(date -u +%H:%M:%S)"
