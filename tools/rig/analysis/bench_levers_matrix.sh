#!/bin/bash
# GPU-gated inference-lever benchmark on compute3 (v6 checkpoint, K=4, nfe=1). Never runs while a
# run_deploy process exists or with < 9 GB free; each config loads the model in its own process.
cd /home/physicalai/phantom-icra-2027/phantom
PY=.venv/bin/python; OUT=/home/physicalai/phantom-icra-2027/logs/bench_levers_20260912; mkdir -p $OUT
CK=runs/teacher_v6/teacher_020000.pt
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $2-$1}'; }
while true; do
  busy=$(pgrep -f "phantom.scripts.run_deploy" | wc -l)
  if [ "$busy" -eq 0 ] && [ "$(free_mib)" -ge 9000 ]; then
    sleep 300   # a quiet rig for 5 minutes before touching the GPU
    busy=$(pgrep -f "phantom.scripts.run_deploy" | wc -l)
    [ "$busy" -eq 0 ] && [ "$(free_mib)" -ge 9000 ] && break
  fi
  sleep 120
done
echo "bench start $(date -u +%FT%TZ) free=$(free_mib) MiB" > $OUT/progress.txt
run() { name=$1; shift; echo "--- $name $(date -u +%T)" >> $OUT/progress.txt
  if [ "$busy" ]; then :; fi
  if pgrep -f "phantom.scripts.run_deploy" >/dev/null; then echo "rig became busy; aborting before $name" >> $OUT/progress.txt; exit 0; fi
  env PYTHONDONTWRITEBYTECODE=1 timeout 1500 $PY -m phantom.scripts.bench_inference --ckpt $CK --hardware configs/hardware.nuc.yaml --system teacher --nfe 1 --ema --iters 10 --warmup 3 --seed 7 --out $OUT/$name.json --dump $OUT/$name.npz --k-seeds 4 "$@" > $OUT/$name.log 2>&1
  echo "rc=$? $(grep -E "replan wall|PARITY" $OUT/$name.log | tail -2 | tr "\n" " ")" >> $OUT/progress.txt; }
run baseline
run compile_default   --compile --compile-mode default --parity-against $OUT/baseline.npz
run compile_cudagraph --compile --compile-mode reduce-overhead --parity-against $OUT/baseline.npz
run flex              --flex --parity-against $OUT/baseline.npz
run flex_compile      --flex --compile --compile-mode default --parity-against $OUT/baseline.npz
run fp8               --fp8 --parity-against $OUT/baseline.npz
run fp8_flex_compile  --fp8 --flex --compile --compile-mode default --parity-against $OUT/baseline.npz
run k2_baseline       --k-seeds 2
run k1_baseline       --k-seeds 1
echo "BENCH DONE $(date -u +%FT%TZ)" >> $OUT/progress.txt
