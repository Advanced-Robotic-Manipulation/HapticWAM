#!/bin/bash
# Terminal A: hold a model warm on the GPU. Terminal B's PICK.sh then attaches
# in seconds (no per-launch model load). Ctrl-C stops the server; restart with
# a different model to switch.
set -e
BASE=~/phantom-icra-2027
TSV=$BASE/MODELS.tsv
[ -f "$TSV" ] || { echo "no $TSV"; exit 1; }
echo "== PHANTOM policy server =="
i=0; labels=(); ckpts=(); systems=()
while IFS=$'	' read -r label ckpt note system; do
  [ -z "$label" ] && continue
  case "$label" in \#*) continue;; esac
  i=$((i+1)); labels[$i]=$label; ckpts[$i]=$ckpt; systems[$i]=${system:-teacher}
  printf "  %d) %-6s %s%s\n" "$i" "$label" "$note" "$([ "${system:-teacher}" = student ] && echo "  [STUDENT: sensor-free]")"
done < "$TSV"
read -p "model to hold [1]: " m; m=${m:-1}
CKPT=${ckpts[$m]}; SYSTEM=${systems[$m]:-teacher}
[ -n "$CKPT" ] || { echo "bad choice"; exit 1; }
# --system must match the checkpoint's input set: a student (camera + proprio,
# no pads) built as a teacher fails at load (rig 09-08: the 4th MODELS.tsv
# column was read by PICK.sh but not here)
# one port per menu slot (7777, 7778, ...) so several models can stay warm at
# once — PICK.sh probes all of them and attaches to the matching one (A/B
# alternation with no reload)
PORT=$((7776 + m))
cd $BASE/phantom
echo ">> serving ${labels[$m]} (system=$SYSTEM) on 127.0.0.1:$PORT"
exec .venv/bin/python -m phantom.scripts.policy_server \
    --ckpt "$CKPT" --system "$SYSTEM" --hardware configs/hardware.nuc.yaml --port $PORT
