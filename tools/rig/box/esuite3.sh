#!/bin/bash
# after esuite2: replan-latency profile on the DEPLOY GPU (the rig box, RTX 5090) for the K-seed / NFE levers (P4/P6)
until grep -q "ALL DONE6" ~/phantom-icra-2027/esuite6.log; do sleep 60; done
cd ~/phantom-icra-2027/phantom
C5=runs/teacher_v5_batch0822/v5_6.pt; HW=configs/hardware.nuc.yaml
mkdir -p runs/bench
for nfe in 5 3 1; do for k in 1 2 4; do
  tag=lat_nfe${nfe}_k${k}; echo "== $tag $(date -u +%H:%M:%S)"
  .venv/bin/python -m phantom.scripts.bench_inference --ckpt $C5 --hardware $HW --ema --nfe $nfe --k-seeds $k --iters 10 --warmup 3 > runs/bench/$tag.log 2>&1; echo "$tag rc=$?"; grep -iE "mean|median|p95|latency" runs/bench/$tag.log | tail -2
done; done
tag=lat_nfe5_k1_compile; echo "== $tag"; .venv/bin/python -m phantom.scripts.bench_inference --ckpt $C5 --hardware $HW --ema --nfe 5 --k-seeds 1 --compile --iters 10 --warmup 3 > runs/bench/$tag.log 2>&1; echo "$tag rc=$?"; grep -iE "mean|median|p95|latency" runs/bench/$tag.log | tail -2
echo "ALL DONE3 $(date -u +%H:%M:%S)"
