#!/bin/bash
# SESSION_0912.sh — warm every model today's run sheet needs, pair rows first, baselines while
# GPU memory allows; then PICK.sh <row> attaches to the warm server by checkpoint sha.
#   ./SESSION_0912.sh            warm 6 (v6) + 2 (stu_ftA_r2), then 5 (pi0.5) 9 (dp) 10 (xvla) if they fit
#   ./SESSION_0912.sh status     what is warm + GPU memory
#   ./SESSION_0912.sh stop       stop OUR servers (by pid of the menu ports; never anything else)
#   ./SESSION_0912.sh baselines  warm only 5 9 10 (after the pair is done, if they did not fit before)
# Never kills another user's process: if the GPU is held by someone else it says so and exits.
cd "$(dirname "$0")"
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
declare -A NEED=( [6]=8500 [2]=8500 [5]=8500 [9]=3000 [10]=5000 )
warm() {
  for r in "$@"; do
    F=$(free_mib)
    if [ "$F" -lt "${NEED[$r]}" ]; then echo "!! row $r needs ~${NEED[$r]} MiB, only $F free — skipped (stop a finished row or wait)"; continue; fi
    ./serve_bg.sh "$r"
  done
}
case "${1:-warm}" in
  warm)
    F=$(free_mib); echo "GPU free: $F MiB"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
    if [ "$F" -lt 17000 ]; then
      echo "!! less than 17 GB free: another job holds the 5090 (Ilya's sim lanes?). Do NOT kill it — wait for it or ask."; exit 3
    fi
    warm 6 2          # the pre-registered pair: teacher v6 vs pad-free student
    warm 5 9 10       # baselines, each only if it fits
    ./serve_bg.sh status; echo "GPU free now: $(free_mib) MiB";;
  baselines) warm 5 9 10; ./serve_bg.sh status;;
  status) ./serve_bg.sh status; echo "GPU free: $(free_mib) MiB"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader;;
  stop) shift; [ $# -eq 0 ] && set -- 6 2 5 9 10; for r in "$@"; do ./serve_bg.sh stop "$r"; done;;
  *) sed -n 2,9p "$0";;
esac
