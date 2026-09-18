#!/bin/bash
# waits for esuite.sh, then re-runs E0 on the two slow v5 episodes with 16 seeds, storing per-seed values
until grep -q "ALL DONE" ~/phantom-icra-2027/esuite.log; do sleep 60; done
cd ~/phantom-icra-2027/phantom
D=$HOME/phantom-icra-2027/data/episodes/deploy/20260828
SLOW="$D/ep_teacher_waffles_1787923675_000 $D/ep_teacher_waffles_1787929940_000"
C5=runs/teacher_v5_batch0822/v5_6.pt; HW=configs/hardware.nuc.yaml
run(){ tag=$1; shift; echo "== $tag $(date -u +%H:%M:%S)"; .venv/bin/python tools/replay_rig.py --hardware $HW "$@" --out runs/replay/$tag.json > runs/replay/$tag.log 2>&1; echo "$tag rc=$? $(date -u +%H:%M:%S)"; }
run e0b_seeds16_pn --ckpt $C5 --episodes $SLOW --nfe 5 --seeds 16 --persistent-noise
run e0b_seeds16_fresh --ckpt $C5 --episodes $SLOW --nfe 5 --seeds 16
echo "ALL DONE2 $(date -u +%H:%M:%S)"
