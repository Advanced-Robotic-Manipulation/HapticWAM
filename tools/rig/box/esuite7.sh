#!/bin/bash
[ -e ~/phantom-icra-2027/esuite7.lock ] && exit 0; touch ~/phantom-icra-2027/esuite7.lock

cd ~/phantom-icra-2027/phantom
D=$HOME/phantom-icra-2027/data/episodes/deploy/20260828
HW=configs/hardware.nuc.yaml
run(){ tag=$1; shift; echo "== $tag $(date -u +%H:%M:%S)"; .venv/bin/python tools/replay_deploy_path.py --hardware $HW "$@" --out runs/replay/$tag.json > runs/replay/$tag.log 2>&1; echo "$tag rc=$? $(date -u +%H:%M:%S)"; }
run e0r_deployrng_v5 --ckpt runs/teacher_v5_batch0822/v5_6.pt --episodes $D/ep_teacher_waffles_1787923675_000 $D/ep_teacher_waffles_1787929940_000 --nfe 5 --persistent-noise --deploy-rng
run e0r_deployrng_v4 --ckpt runs/teacher_v4_790eps/teacher_020000.pt --episodes $D/ep_teacher_waffles_1787922904_000 $D/ep_teacher_waffles_1787923361_000 --nfe 5 --persistent-noise --deploy-rng
echo "ALL DONE7 $(date -u +%H:%M:%S)"
