#!/bin/bash
# SESSION_0912.sh — warm every model today's run sheet needs, pair rows first, baselines while
# GPU memory allows; then PICK.sh <row> attaches to the warm server by checkpoint sha.
#   ./SESSION_0912.sh            warm 1 (v6) + 2 (stu_ftA_r2), then 4 (pi0.5) 5 (dp) 6 (xvla) if they fit
#   ./SESSION_0912.sh status     what is warm + GPU memory
#   ./SESSION_0912.sh stop       stop OUR servers (by pid of the menu ports; never anything else)
#   ./SESSION_0912.sh baselines  warm only 4 5 6 (after the pair is done, if they did not fit before)
#   ./SESSION_0912.sh blockc     Block C: stop 2 4 5 6 (keeps 1 v6), warm 3 stu_v6_r2
# Never kills another user's process: if the GPU is held by someone else it says so and exits.
cd "$(dirname "$0")"
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
declare -A NEED=( [1]=7000 [2]=7000 [3]=7000 [4]=8500 [5]=2500 [6]=3000 [7]=7000 [8]=7000 )   # measured 09-12: v6 6.6 · stu 6.4 · pi05 7.9 · dp 1.8 · xvla 2.4 GB
listening() { ss -ltn 2>/dev/null | grep -q ":$((7776 + $1)) "; }
ours_mib() {  # GPU memory held by OUR menu servers (so the gate only counts other people's jobs)
  local t=0 pid m
  for r in 1 2 3 4 5 6 7 8 9; do
    pid=$(ss -ltnp 2>/dev/null | grep ":$((7776 + r)) " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
    [ -n "$pid" ] && { m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$pid" '$1==p{print $2}'); t=$((t + ${m:-0})); }
  done; echo $t
}
warm() {
  for r in "$@"; do
    if listening "$r"; then echo "row $r already warm on :$((7776 + r))"; continue; fi
    F=$(free_mib)
    if [ "$F" -lt "${NEED[$r]:-7000}" ]; then echo "!! row $r needs ~${NEED[$r]:-7000} MiB, only $F free — skipped (./SESSION_0912.sh stop <finished row> first)"; continue; fi
    ./serve_bg.sh "$r"
  done
}
case "${1:-warm}" in
  warm)
    F=$(free_mib); O=$(ours_mib); echo "GPU free: $F MiB (our warm servers hold $O MiB)"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
    if [ $((F + O)) -lt 17000 ]; then
      echo "!! less than 17 GB available even counting our own servers: another job holds the 5090. Do NOT kill it — wait for it or ask."; exit 3
    fi
    warm 1 2          # the pre-registered pair: teacher v6 vs pad-free student
    warm 4 5 6        # baselines, each only if it fits
    ./serve_bg.sh status; echo "GPU free now: $(free_mib) MiB";;
  baselines) warm 4 5 6; ./serve_bg.sh status;;
  blockc) ./serve_bg.sh stop 2; ./serve_bg.sh stop 4; ./serve_bg.sh stop 5; ./serve_bg.sh stop 6; warm 3; ./serve_bg.sh status;;
  status) ./serve_bg.sh status; echo "GPU free: $(free_mib) MiB (our servers hold $(ours_mib) MiB)"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader;;
  stop) shift; [ $# -eq 0 ] && set -- 1 2 3 4 5 6 7 8; for r in "$@"; do ./serve_bg.sh stop "$r"; done;;
  *) sed -n 2,9p "$0";;
esac
