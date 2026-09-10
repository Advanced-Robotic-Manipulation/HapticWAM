#!/bin/bash
# serve_bg.sh <menu#> : hold a menu model warm in the background (same ports as SERVE.sh)
m=${1:?menu number}
BASE=${PHANTOM_RIG_BASE:-$HOME/phantom-icra-2027}; TSV=$BASE/MODELS.tsv
i=0; while IFS=$'\t' read -r label ckpt note system; do
  [ -z "$label" ] && continue; case "$label" in \#*) continue;; esac
  i=$((i+1)); [ "$i" = "$m" ] && { L=$label; C=$ckpt; S=${system:-teacher}; }
done < "$TSV"
[ -n "$C" ] || { echo "bad menu number $m"; exit 1; }
PORT=$((7776 + m)); LOG=$BASE/logs/serve_${L}.log; mkdir -p $BASE/logs
if ss -ltn "sport = :$PORT" | grep -q ":$PORT"; then echo "port $PORT already listening ($L)"; exit 0; fi
cd $BASE/phantom
nohup .venv/bin/python -m phantom.scripts.policy_server --ckpt "$C" --system "$S" \
    --hardware configs/hardware.nuc.yaml --port $PORT > "$LOG" 2>&1 &
echo "$L ($S) starting on $PORT, pid $!, log $LOG"
