#!/bin/bash
# after esuite2: replay WITH deploy's LoRA fold (bf16) — does the merged model reproduce the rig?
[ -e ~/phantom-icra-2027/esuite5.lock ] && exit 0; touch ~/phantom-icra-2027/esuite5.lock
until grep -q "ALL DONE2" ~/phantom-icra-2027/esuite2.log; do sleep 60; done
cd ~/phantom-icra-2027/phantom
D=$HOME/phantom-icra-2027/data/episodes/deploy/20260828
EPS="$D/ep_teacher_waffles_1787923675_000 $D/ep_teacher_waffles_1787929940_000 $D/ep_teacher_waffles_1787930224_002"
C5=runs/teacher_v5_batch0822/v5_6.pt; HW=configs/hardware.nuc.yaml
run(){ tag=$1; shift; echo "== $tag $(date -u +%H:%M:%S)"; .venv/bin/python tools/replay_rig.py --hardware $HW "$@" --out runs/replay/$tag.json > runs/replay/$tag.log 2>&1; echo "$tag rc=$? $(date -u +%H:%M:%S)"; }
run e0m_merged_nfe5 --ckpt $C5 --episodes $EPS --nfe 5 --seeds 8 --persistent-noise --merge-lora
run e0m_merged_nfe1 --ckpt $C5 --episodes $EPS --nfe 1 --seeds 8 --persistent-noise --merge-lora
echo "ALL DONE5 $(date -u +%H:%M:%S)"
