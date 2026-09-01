#!/bin/bash
# Terminal A: hold a model warm on the GPU. Terminal B's PICK.sh then attaches
# in seconds (no per-launch model load). Ctrl-C stops the server; restart with
# a different model to switch.
set -e
BASE=~/phantom-icra-2027
TSV=$BASE/MODELS.tsv
[ -f "$TSV" ] || { echo "no $TSV"; exit 1; }
echo "== PHANTOM policy server =="
i=0; labels=(); ckpts=()
while IFS=$'	' read -r label ckpt note; do
  [ -z "$label" ] && continue
  case "$label" in \#*) continue;; esac
  i=$((i+1)); labels[$i]=$label; ckpts[$i]=$ckpt
  printf "  %d) %-6s %s\n" "$i" "$label" "$note"
done < "$TSV"
read -p "model to hold [1]: " m; m=${m:-1}
CKPT=${ckpts[$m]}
[ -n "$CKPT" ] || { echo "bad choice"; exit 1; }
cd $BASE/phantom
exec .venv/bin/python -m phantom.scripts.policy_server \
    --ckpt "$CKPT" --hardware configs/hardware.nuc.yaml
