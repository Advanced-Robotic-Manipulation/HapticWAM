#!/bin/bash
# serve_bg.sh — hold menu models warm in the background, one policy server per menu slot.
#   ./serve_bg.sh 1 2        warm menu rows 1 and 2, wait until each prints READY
#   ./serve_bg.sh all        warm every row in MODELS.tsv
#   ./serve_bg.sh status     what is listening on the menu ports (ckpt + busy/idle)
#   ./serve_bg.sh stop 2     stop the server of menu row 2 (ours only; kills by pid of the listener)
# Ports: 7776 + menu row (same as SERVE.sh and PICK.sh, which attaches by checkpoint sha).
BASE=${PHANTOM_RIG_BASE:-$HOME/phantom-icra-2027}; TSV=$BASE/MODELS.tsv
labels=(); ckpts=(); systems=(); n=0
while IFS=$'\t' read -r label ckpt note system; do
  [ -z "$label" ] && continue; case "$label" in \#*) continue;; esac
  n=$((n+1)); labels[$n]=$label; ckpts[$n]=$ckpt; systems[$n]=${system:-teacher}
done < "$TSV"
[ "$n" -gt 0 ] || { echo "no rows in $TSV"; exit 1; }
cd "$BASE/phantom" || exit 1

listener_pid() { ss -ltnp "sport = :$1" 2>/dev/null | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2; }

case "${1:-}" in
  ""|-h|--help) sed -n 2,7p "$0"; exit 0;;
  status)
    for m in $(seq 1 $n); do
      PORT=$((7776 + m)); PID=$(listener_pid $PORT)
      if [ -n "$PID" ]; then
        INFO=$(.venv/bin/python -m phantom.scripts.policy_server --probe --port $PORT 2>/dev/null | head -1)
        echo "  $m) ${labels[$m]}  :$PORT  pid $PID  ${INFO:-(no status)}"
      else
        echo "  $m) ${labels[$m]}  :$PORT  -"
      fi
    done; exit 0;;
  stop)
    shift; for m in "$@"; do
      PORT=$((7776 + m)); PID=$(listener_pid $PORT)
      [ -n "$PID" ] && { kill "$PID" && echo "stopped ${labels[$m]} (pid $PID, :$PORT)"; } || echo "nothing on :$PORT"
    done; exit 0;;
  all) set -- $(seq 1 $n);;
esac

mkdir -p "$BASE/logs"
started=()
for m in "$@"; do
  L=${labels[$m]}; C=${ckpts[$m]}; S=${systems[$m]}
  [ -n "$C" ] || { echo "bad menu number $m (1..$n)"; continue; }
  [ -f "$C" ] || { echo "checkpoint missing on disk: $BASE/phantom/$C"; continue; }
  PORT=$((7776 + m)); LOG=$BASE/logs/serve_${L}.log
  if [ -n "$(listener_pid $PORT)" ]; then echo "$L already listening on :$PORT"; continue; fi
  nohup .venv/bin/python -m phantom.scripts.policy_server --ckpt "$C" --system "$S" \
      --hardware configs/hardware.nuc.yaml --port $PORT > "$LOG" 2>&1 &
  echo "$L ($S) starting on :$PORT, pid $!, log $LOG"
  started+=("$m")
done

for m in "${started[@]}"; do
  L=${labels[$m]}; LOG=$BASE/logs/serve_${L}.log; PID=""
  for i in $(seq 1 180); do
    grep -q "^READY " "$LOG" 2>/dev/null && { echo "READY  $m) $L on :$((7776 + m))"; break; }
    grep -qiE "Traceback|Error|out of memory" "$LOG" 2>/dev/null && { echo "FAILED $m) $L — see $LOG"; tail -n 3 "$LOG"; break; }
    sleep 2
  done
done
