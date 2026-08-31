#!/bin/bash
# Interactive rig launcher: pick a model, pick an inference preset, confirm, run.
# The model menu is the curated list in ~/phantom-icra-2027/MODELS.tsv
# (label<TAB>ckpt<TAB>note) — keep ONLY builds worth running on the rig in there.
set -e
BASE=~/phantom-icra-2027
TSV=$BASE/MODELS.tsv
[ -f "$TSV" ] || { echo "no $TSV — create it (label<TAB>ckpt<TAB>note per line)"; exit 1; }

echo "== PHANTOM rig launcher =="
echo "-- models --"
i=0
labels=(); ckpts=()
while IFS=$'	' read -r label ckpt note; do
  [ -z "$label" ] && continue
  case "$label" in \#*) continue;; esac
  i=$((i+1)); labels[$i]=$label; ckpts[$i]=$ckpt
  printf "  %d) %-6s %s\n" "$i" "$label" "$note"
done < "$TSV"
read -p "model [1]: " m; m=${m:-1}
CKPT=${ckpts[$m]}; MODEL=${labels[$m]}
[ -n "$CKPT" ] || { echo "bad choice"; exit 1; }
[ -f "$BASE/phantom/$CKPT" ] || { echo "checkpoint missing on disk: $BASE/phantom/$CKPT"; exit 1; }

echo "-- inference presets --"
echo "  1) LEVERS  (recommended): nfe=1 + terminal-veto + parity-fixes + k-seeds 4 + max-episode-s 35 + max-replans 200"
echo "  2) PLAIN   nfe=5, no extras (pre-fix inference style — attribution control only)"
echo "  3) VETO    nfe=5 + terminal-veto + parity-fixes (quality sampling, safety gate on)"
echo "  4) CUSTOM  type your own nfe + flags"
read -p "preset [1]: " p; p=${p:-1}
case $p in
  1) NFE=1; EXTRA="--terminal-veto --parity-fixes --k-seeds 4 --max-episode-s 35 --max-replans 200"; PRESET=LEVERS;;
  2) NFE=5; EXTRA=""; PRESET=PLAIN;;
  3) NFE=5; EXTRA="--terminal-veto --parity-fixes"; PRESET=VETO;;
  4) read -p "nfe [5]: " NFE; NFE=${NFE:-5}; read -p "flags: " EXTRA; PRESET=CUSTOM;;
  *) echo "bad choice"; exit 1;;
esac
read -p "task [waffles]: " TASK; TASK=${TASK:-waffles}
read -p "episodes [3]: " EPS; EPS=${EPS:-3}
read -p "append flags (Enter for none): " MORE
[ -n "$MORE" ] && EXTRA="$EXTRA $MORE"

echo
echo ">> $MODEL ($CKPT) | $PRESET | $TASK x$EPS | nfe=$NFE | extra: [$EXTRA]"
echo ">> reminders: no --seed on 1-episode processes; interleave arms within each grid cell;"
echo ">>            joint gate must be green; stay attended until a gripper release is seen working."
read -p "Enter to launch (Ctrl-C to abort) "
CKPT="$CKPT" EXTRA="$EXTRA" exec "$BASE/GO_ANY.sh" "$TASK" "$EPS" "$NFE" 1.0
