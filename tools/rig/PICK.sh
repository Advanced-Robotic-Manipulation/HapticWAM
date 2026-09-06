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
labels=(); ckpts=(); systems=()
while IFS=$'	' read -r label ckpt note system; do
  [ -z "$label" ] && continue
  case "$label" in \#*) continue;; esac
  i=$((i+1)); labels[$i]=$label; ckpts[$i]=$ckpt; systems[$i]=${system:-teacher}
  printf "  %d) %-6s %s%s\n" "$i" "$label" "$note" "$([ "${system:-teacher}" = student ] && echo '  [STUDENT: sensor-free]')"
done < "$TSV"
read -p "model [1]: " m; m=${m:-1}
CKPT=${ckpts[$m]}; MODEL=${labels[$m]}; SYSTEM=${systems[$m]:-teacher}
[ -n "$CKPT" ] || { echo "bad choice"; exit 1; }
[ -f "$BASE/phantom/$CKPT" ] || { echo "checkpoint missing on disk: $BASE/phantom/$CKPT"; exit 1; }

echo "-- inference presets --"
echo "  1) LEVERS  (recommended): nfe=1 + terminal-veto + parity-fixes + k-seeds 4 + max-episode-s 150 + max-replans 200"
echo "  2) PLAIN   nfe=5, no extras (pre-fix inference style — attribution control only)"
echo "  3) VETO    nfe=5 + terminal-veto + parity-fixes (quality sampling, safety gate on)"
echo "  4) CUSTOM  type your own nfe + flags"
read -p "preset [1]: " p; p=${p:-1}
case $p in
  1) NFE=1; EXTRA="--terminal-veto --parity-fixes --k-seeds 4 --max-episode-s 150 --max-replans 200"; PRESET=LEVERS;;
  2) NFE=5; EXTRA=""; PRESET=PLAIN;;
  3) NFE=5; EXTRA="--terminal-veto --parity-fixes"; PRESET=VETO;;
  4) read -p "nfe [5]: " NFE; NFE=${NFE:-5}; read -p "flags: " EXTRA; PRESET=CUSTOM;;
  *) echo "bad choice"; exit 1;;
esac
read -p "task [waffles]: " TASK; TASK=${TASK:-waffles}
read -p "episodes [3]: " EPS; EPS=${EPS:-3}
read -p "append flags (Enter for none): " MORE
[ -n "$MORE" ] && EXTRA="$EXTRA $MORE"

# attach to a warm policy server when one holds the SAME model (start one in
# another terminal with ./SERVE.sh — launches then take seconds, not minutes)
if [[ "$EXTRA" != *"--policy-server"* ]]; then
  ATTACHED=""; FOUND=""
  WANT=$(basename "$(readlink -f "$BASE/phantom/$CKPT" 2>/dev/null || echo "$CKPT")")
  for PORT in 7777 7778 7779; do
    # probe prints "<ckpt> <idle|busy> <sha>"; exit 1 = nothing listens,
    # exit 2 = something listens but does not answer (AMBIGUOUS). A busy or
    # ambiguous server must never turn into a competing local load that
    # fights the running process for the arm (issue #8, rig 09-04).
    RC=0; LINE=$(cd $BASE/phantom && timeout 15 .venv/bin/python -m phantom.scripts.policy_server --probe --port $PORT 2>/dev/null) || RC=$?
    if [ "$RC" = "2" ] || [ "$RC" = "124" ]; then
      echo "!! a process listens on :$PORT but did not answer the probe — hung server or foreign process."
      echo "!! Not launching: check SERVE.sh terminals / 'ss -ltnp | grep $PORT' first."; exit 3
    fi
    [ "$RC" = "0" ] && [ -n "$LINE" ] || continue
    read -r SRV_CKPT SRV_STATE SRV_SHA <<< "$LINE"
    FOUND="$FOUND $PORT:$(basename "$SRV_CKPT")[$SRV_STATE]"
    HAVE=$(basename "$(readlink -f "$BASE/phantom/$SRV_CKPT" 2>/dev/null || echo "$SRV_CKPT")")
    if [ "$HAVE" = "$WANT" ] || [ "$(basename "$SRV_CKPT")" = "$(basename "$CKPT")" ]; then
      if [ "$SRV_STATE" = "busy" ]; then
        echo "!! the policy server on :$PORT holds $SRV_CKPT but is BUSY: another run_deploy is attached"
        echo "!! (possibly Ctrl-Z'd: 'jobs' / 'ps -ef | grep run_deploy'). Bring it to the foreground and end it."
        echo "!! Not launching a competing process."; exit 3
      fi
      echo ">> warm policy server on :$PORT holds $SRV_CKPT (sha $SRV_SHA) — attaching (fast start)"
      EXTRA="$EXTRA --policy-server 127.0.0.1:$PORT"; ATTACHED=1; break
    fi
  done
  if [ -z "$ATTACHED" ]; then
    if [ -n "$FOUND" ]; then
      echo ">> NOTE: running servers hold [$FOUND ] but you picked $CKPT."
      echo ">>       loading locally (slow). ./SERVE.sh with this model makes it fast next time."
    else
      echo ">> no policy server running — loading the model in-process (~3 min)."
      echo ">>    tip: run ./SERVE.sh in another terminal once per model; later launches attach in seconds."
    fi
  fi
fi

echo
echo ">> $MODEL ($CKPT, system=$SYSTEM) | $PRESET | $TASK x$EPS | nfe=$NFE | extra: [$EXTRA]"
echo ">> reminders: PAIR the arms: same --seed <cell#> on BOTH arms of a cell (the append-flags prompt), new number per cell;"
echo ">>            never reuse one seed across cells; interleave arms within each grid cell;"
echo ">>            joint gate must be green; stay attended until a gripper release is seen working."
echo ">>            during an episode: type  x  + Enter to end it cleanly (motion stops, gripper stays);"
echo ">>            a bare Enter is IGNORED (stray newlines ended 5 episodes on 09-04)."
read -p "Enter to launch (Ctrl-C to abort) "
CKPT="$CKPT" SYSTEM="$SYSTEM" EXTRA="$EXTRA" exec "$BASE/GO_ANY.sh" "$TASK" "$EPS" "$NFE" 1.0
