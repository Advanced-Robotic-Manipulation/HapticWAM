#!/bin/bash
# SESSION_0912.sh — warm every model today's run sheet needs, pair rows first, baselines while
# GPU memory allows; then PICK.sh <row> attaches to the warm server by checkpoint sha.
#   ./SESSION_0912.sh            warm 1 (v6) + 2 (stu_ftA_r2), then 4 (pi0.5) 5 (dp) 6 (xvla) if they fit
#   ./SESSION_0912.sh status     what is warm + GPU memory
#   ./SESSION_0912.sh stop       stop OUR servers (by pid of the menu ports; never anything else)
#   ./SESSION_0912.sh baselines  warm only 4 5 6 (after the pair is done, if they did not fit before)
#   ./SESSION_0912.sh blockc     Block C: stop 2 4 5 6 (keeps 1 v6), warm 3 stu_v6_r2
#   ./SESSION_0912.sh monday     09-14: warm 9 (v6_simft2k, --flex --compile) + 2 (stu_ftA_r2), then 1 (v6) if it fits
#   ./SESSION_0912.sh rows 9 17 14   09-15: warm exactly these menu rows, in this order, each only if it fits
#   (09-15: EXPERIMENT.sh drives this per block)
# Never kills another user's process: if the GPU is held by someone else it says so and exits.
cd "$(dirname "$(readlink -f "$0")")"
TSV=${PHANTOM_RIG_BASE:-$HOME/phantom-icra-2027}/MODELS.tsv
NROWS=$(awk -F'\t' '!/^#/ && $1!=""' "$TSV" 2>/dev/null | wc -l | tr -d ' '); NROWS=${NROWS:-13}
ALL_ROWS=$(seq 1 "$NROWS" | tr '\n' ' ')
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
# GPU need per row, derived from the live menu (measured: v6 6.6 GB, student 6.4, --flex --compile teacher 7.6, pi05 7.9, dp 1.8, xvla 2.5)
declare -A NEED=(); _r=0
while IFS=$'\t' read -r _label _ckpt _note _system _extra; do
  [ -z "$_label" ] && continue; case "$_label" in \#*) continue;; esac
  _r=$((_r + 1)); _n=7000; [ -n "$_extra" ] && _n=8000
  case "$_label" in pi05) _n=8500;; dp) _n=2500;; xvla) _n=3000;; esac
  NEED[$_r]=$_n
done < "$TSV"
listening() { ss -ltn 2>/dev/null | grep -q ":$((7776 + $1)) "; }
ours_mib() {  # GPU memory held by OUR menu servers (so the gate only counts other people's jobs)
  local t=0 pid m
  for r in $ALL_ROWS; do
    pid=$(ss -ltnp 2>/dev/null | grep ":$((7776 + r)) " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
    [ -n "$pid" ] && { m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$pid" '$1==p{print $2}'); t=$((t + ${m:-0})); }
  done; echo $t
}
warm() {
  for r in "$@"; do
    if listening "$r"; then echo "row $r already warm on :$((7776 + r))"; continue; fi
    F=$(free_mib)
    if [ "$F" -lt "${NEED[$r]:-7500}" ]; then echo "!! row $r needs ~${NEED[$r]:-7500} MiB, only $F free — skipped (./SESSION_0912.sh stop <finished row> first)"; continue; fi
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
    warm 4 5          # baselines (pi0.5, dp), each only if it fits; 6 xvla is off the rig (needs 3 camera views)
    ./serve_bg.sh status; echo "GPU free now: $(free_mib) MiB";;
  baselines) warm 4 5; ./serve_bg.sh status;;
  monday)
    F=$(free_mib); O=$(ours_mib); echo "GPU free: $F MiB (our warm servers hold $O MiB)"
    if [ $((F + O)) -lt 15000 ]; then
      echo "!! less than 15 GB available even counting our own servers: another job holds the 5090. Do NOT kill it — wait for it or ask."; exit 3
    fi
    warm 9 2          # Block P: the sim-expert teacher vs the pad-free student
    warm 1            # base v6, only if it fits (three Cosmos servers is the ceiling)
    ./serve_bg.sh status; echo "GPU free now: $(free_mib) MiB";;
  blockc) ./serve_bg.sh stop 2; ./serve_bg.sh stop 4; ./serve_bg.sh stop 5; ./serve_bg.sh stop 6; warm 3; ./serve_bg.sh status;;
  rows)
    shift; [ $# -gt 0 ] || { echo "usage: ./SESSION_0912.sh rows <menu row> [row ...]"; exit 2; }
    F=$(free_mib); O=$(ours_mib); echo "GPU free: $F MiB (our warm servers hold $O MiB); menu has $NROWS rows"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
    T=0; for r in "$@"; do listening "$r" || T=$((T + ${NEED[$r]:-7500})); done
    if [ "$T" -gt 0 ] && [ "$F" -lt "$T" ]; then
      echo "!! rows $* need ~$T MiB more, only $F MiB free: another job holds the 5090 (or our other servers do)."
      echo "!! Do NOT kill anyone else's job — wait for it, ask, or stop a finished row of ours. Nothing warmed (no half-warm blocks)."; exit 3
    fi
    warm "$@"
    ./serve_bg.sh status; echo "GPU free now: $(free_mib) MiB";;
  status) ./serve_bg.sh status; echo "GPU free: $(free_mib) MiB (our servers hold $(ours_mib) MiB)"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader;;
  stop) shift; [ $# -eq 0 ] && set -- $ALL_ROWS; for r in "$@"; do ./serve_bg.sh stop "$r"; done;;
  *) sed -n 2,11p "$0";;
esac
