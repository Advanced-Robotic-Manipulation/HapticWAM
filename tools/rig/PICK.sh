#!/bin/bash
# Interactive rig launcher: pick a model, pick an inference preset, confirm, run.
# The model menu is the curated list in ~/phantom-icra-2027/MODELS.tsv
# (label<TAB>ckpt<TAB>note) — keep ONLY builds worth running on the rig in there.
set -e
BASE=${PHANTOM_RIG_BASE:-$HOME/phantom-icra-2027}
TSV=$BASE/MODELS.tsv
[ -f "$TSV" ] || { echo "no $TSV — create it (label<TAB>ckpt<TAB>note per line)"; exit 1; }

echo "== PHANTOM rig launcher =="
echo "-- models --"
i=0
labels=(); ckpts=(); systems=(); extras=()
while IFS=$'	' read -r label ckpt note system extra; do
  [ -z "$label" ] && continue
  case "$label" in \#*) continue;; esac
  i=$((i+1)); labels[$i]=$label; ckpts[$i]=$ckpt; systems[$i]=${system:-teacher}; extras[$i]=$extra
  printf "  %d) %-6s %s%s\n" "$i" "$label" "$note" "$([ "${system:-teacher}" = student ] && echo '  [STUDENT: sensor-free]')"
done < "$TSV"
# 09-15: the row may be given on the command line (./PICK.sh 9, ./PICK.sh stu_simft_000500); a label is accepted too
m=${1:-}
if [ -z "$m" ]; then read -p "model [1]: " m; m=${m:-1}; fi
if ! [[ "$m" =~ ^[0-9]+$ ]]; then for j in $(seq 1 $i); do [ "${labels[$j]}" = "$m" ] && { m=$j; break; }; done; fi
[[ "$m" =~ ^[0-9]+$ ]] || { echo "bad choice: $m (a menu number or an exact label)"; exit 1; }
CKPT=${ckpts[$m]}; MODEL=${labels[$m]}; SYSTEM=${systems[$m]:-teacher}
[ -n "$CKPT" ] || { echo "bad choice"; exit 1; }
[ -e "$BASE/phantom/$CKPT" ] || { echo "checkpoint missing on disk: $BASE/phantom/$CKPT"; exit 1; }

echo "-- inference presets --"
echo "  7) PLACEMENT_FIX (waffles, sim-validated 09-12, rig-unverified): LEVERS + boundary projection v3 (3 cm raw band, 2 cm apex cap) + descend-then-release supervisor (release only below TCP z 0.21 m) + minimal_v5; NOT the default until a rig session confirms the release volume"
echo "  1) LEVERS  (paired sessions, both arms): nfe=1 + terminal-veto + parity-fixes + k-seeds 4 + pose plays 16 / gripper 10 + max-episode-s 150 + max-replans 200 (+ bounded reach, ON by default since 09-11)"
echo "  2) PLAIN   nfe=5, no extras (pre-fix inference style — attribution control only)"
echo "  3) VETO    nfe=5 + terminal-veto + parity-fixes (quality sampling, safety gate on)"
echo "  4) CUSTOM  type your own nfe + flags"
echo "  5) BOUNDED REACH: LEVERS + explicit --servo-reach-profile bounded_v1 (same controller as 1 now that it is the default)"
echo "  6) NO REACH LIMITER: LEVERS + --servo-reach-profile none (attribution control only: elbow may straighten at the box)"
# The default preset must NOT depend on the model: a paired cell runs both
# arms under the SAME controller or the pair measures the controller, not the
# model. BOUNDED REACH (5) is an explicit choice for teacher-only trials.
DEFAULT_PRESET=1
LEROBOT_TYPE=""
case "$SYSTEM" in lerobot) LEROBOT_TYPE=pi05;; lerobot:*) LEROBOT_TYPE=${SYSTEM#lerobot:};; esac
if [ -n "$LEROBOT_TYPE" ]; then
  # LeRobot arms (pi0.5 / diffusion / xvla): no terminal veto / parity fixes / k-seeds (the adapter
  # refuses them); nfe = the checkpoint's own inference steps (pi0.5 flow 10; diffusion / xvla use
  # their config default, 0 = leave it); play caps per docs/pi05_baseline.md, shared by all three
  # so the baseline column runs one controller.
  p=lerobot; NFE=10; [ "$LEROBOT_TYPE" != pi05 ] && NFE=0
  EXTRA="--max-play-steps 16 --grip-play-steps 16 --min-replan-s 0.5 --max-episode-s 150 --max-replans 400"; PRESET=LEROBOT_$LEROBOT_TYPE
  echo "  ($LEROBOT_TYPE row: preset fixed to LEROBOT = nfe $NFE, plays 16/16, replan >= 0.5 s, no veto)"
else
if [ -n "${PICK_PRESET:-}" ]; then p=$PICK_PRESET; echo "preset: $p (PICK_PRESET)"; else read -p "preset [$DEFAULT_PRESET]: " p; p=${p:-$DEFAULT_PRESET}; fi
fi
case $p in
  lerobot) ;;
  1) NFE=1; EXTRA="--terminal-veto --parity-fixes --k-seeds 4 --max-play-steps 16 --grip-play-steps 10 --max-episode-s 150 --max-replans 200"; PRESET=LEVERS;;   # pose plays the whole chunk (09-08: the turn to the box lives in the tail); gripper capped at the validated head (tail openings flapped the fingers)
  2) NFE=5; EXTRA=""; PRESET=PLAIN;;
  3) NFE=5; EXTRA="--terminal-veto --parity-fixes"; PRESET=VETO;;
  4) read -p "nfe [5]: " NFE; NFE=${NFE:-5}; read -p "flags: " EXTRA; PRESET=CUSTOM;;
  5) NFE=1; EXTRA="--terminal-veto --parity-fixes --k-seeds 4 --max-play-steps 16 --grip-play-steps 10 --max-episode-s 150 --max-replans 200 --servo-reach-profile bounded_v1"; PRESET=BOUNDED_REACH;;
  6) NFE=1; EXTRA="--terminal-veto --parity-fixes --k-seeds 4 --max-play-steps 16 --grip-play-steps 10 --max-episode-s 150 --max-replans 200 --servo-reach-profile none"; PRESET=NO_REACH_LIMITER;;
  7) NFE=1; EXTRA="--terminal-veto --parity-fixes --k-seeds 4 --max-play-steps 16 --grip-play-steps 10 --max-episode-s 150 --max-replans 200 --boundary-projection-config configs/boundary_projection_d3.json --placement-release-config configs/placement_release_descent_sim_waffles.json --placement-controller-profile minimal_v5"; PRESET=PLACEMENT_FIX;;   # sim zoo 09-12: boundary v3 + 2 cm apex cap + descend-then-release; release volume = sim waffle scene, UNVERIFIED on the rig
  *) echo "bad choice"; exit 1;;
esac
if [ -n "${PICK_TASK:-}" ]; then TASK=$PICK_TASK; echo "task: $TASK (PICK_TASK)"; else read -p "task [waffles]: " TASK; TASK=${TASK:-waffles}; fi
# the grasp rule and the pairing key are case-sensitive (Z_MAX_MM: waffles Carton egg whiteboard)
case "$(echo "$TASK" | tr "A-Z" "a-z")" in waffles) TASK=waffles;; carton) TASK=Carton;; egg) TASK=egg;; whiteboard|sponge|marker) TASK=whiteboard;;
  *) echo "!! unknown task '$TASK' — use one of: waffles Carton egg whiteboard"; exit 1;; esac
if [ -n "$LEROBOT_TYPE" ]; then
  case "$TASK" in Carton) PHRASE="pick up the carton";; whiteboard) PHRASE="pick up the whiteboard marker";; *) PHRASE="pick up the $TASK";; esac
  EXTRA="$EXTRA --text '$PHRASE'"
  # 09-11 forensics: on whiteboard pi0.5 bottoms out 21-26 mm below the sponge (sd 5 mm) and rams
  # the brick; tell the policy it is 23 mm lower than it is (policy-side only, recorded in overrides)
  [ "$TASK" = whiteboard ] && [ "$LEROBOT_TYPE" = pi05 ] && EXTRA="$EXTRA --policy-z-offset-m -0.023"
fi
# 09-12 forensics: a multi-episode launch increments the seed per episode, so the second arm of the
# cell never shares a seed and 142 episodes produced ZERO teacher-vs-student pairs. 09-13 evening: the
# hard stop is gone (operator request); multi-episode launches are allowed. 09-15: the episodes prompt
# moved BELOW the cell proposal so that the other arm of a batch defaults to the same count (same seeds).
# Paired cells: the seed is 100 + cell, the SAME for both arms of a cell
# (sampler noise + start jitter both come from it). The last cell used is
# remembered per day, so the second arm of a cell just presses Enter; type
# the next number when the placement changes. (09-07: 43 episodes, 43 seeds,
# 0 pairs — the operator was left to type --seed by hand.)
CELLF=$BASE/.pick_cell_$(date +%Y%m%d); LAST=$(cat "$CELLF" 2>/dev/null || echo 0)
# 09-11: automatic cell numbering. Every launch is logged as "cell task label episodes";
# the proposal is the SAME cell while this task's last cell has run on fewer than two
# different models (the other arm is still due), else the next free number: last cell +
# the batch size that ran there (a 5-episode batch uses seeds cell..cell+4, so the next
# placement starts 5 higher and never reuses a seed). Enter accepts; a typed number wins.
RUNS=$BASE/.pick_runs_$(date +%Y%m%d); touch "$RUNS"   # set -e: every awk below must see a file
LASTT=$(awk -F"\t" -v t="$TASK" '$2==t{c=$1} END{print c+0}' "$RUNS" || echo 0); LASTT=${LASTT:-0}
if [ "$LASTT" -gt 0 ]; then
  ARMS=$(awk -F"\t" -v t="$TASK" -v c="$LASTT" '$2==t && $1==c{print $3}' "$RUNS" | sort -u | tr "\n" " " || true)
  NARMS=$(echo $ARMS | wc -w | tr -d " "); EPSL=$(awk -F"\t" -v t="$TASK" -v c="$LASTT" '$2==t && $1==c{e=$4} END{print e+0}' "$RUNS" || echo 1)
  case " $ARMS " in *" $MODEL "*) ALREADY=1;; *) ALREADY=0;; esac
  if [ "$NARMS" -lt 2 ] && [ "$ALREADY" = 0 ]; then
    PROPOSE=$LASTT; WHY="cell $LASTT on $TASK has run on [$ARMS] only — this is the OTHER arm"
  else
    PROPOSE=$((LASTT + (EPSL > 0 ? EPSL : 1))); WHY="cell $LASTT on $TASK is done on [$ARMS] — new placement, next free number"
  fi
else
  PROPOSE=$(( LAST > 0 ? LAST + 1 : 1 )); WHY="first $TASK cell today"
fi
echo ">> $WHY"
if [ -n "${PICK_CELL:-}" ]; then CELL=$PICK_CELL; echo "cell: $CELL (PICK_CELL)"; else read -p "cell number [Enter = $PROPOSE]: " CELL; fi
CELL=${CELL:-$PROPOSE}
[[ "$CELL" =~ ^[0-9]+$ ]] && [ "$CELL" -ge 1 ] || { echo "cell must be a positive integer (start at 1)"; exit 1; }
echo "$CELL" > "$CELLF"
# episodes per launch: defaults to the batch size the OTHER arm ran on this cell (same seeds -> pairs), else 1
DEF_EPS=1
if [ "$LASTT" -gt 0 ] && [ "$CELL" = "$LASTT" ] && [ "${NARMS:-0}" -lt 2 ] && [ "${ALREADY:-0}" = 0 ] && [ "${EPSL:-1}" -gt 1 ]; then
  DEF_EPS=$EPSL; echo ">> the other arm ran $EPSL episodes on cell $CELL — the same count pairs them (seeds $((100 + CELL))..$((100 + CELL + EPSL - 1)))"
fi
if [ -n "${PICK_EPS:-}" ]; then EPS=$PICK_EPS; echo "episodes: $EPS (PICK_EPS)"; else read -p "episodes [$DEF_EPS]: " EPS; EPS=${EPS:-$DEF_EPS}; fi
[[ "$EPS" =~ ^[0-9]+$ ]] && [ "$EPS" -ge 1 ] || { echo "episodes must be a positive integer"; exit 1; }
if [ "$EPS" -gt 1 ]; then
  echo "!! note: $EPS episodes in one launch use seeds $((100 + CELL)) .. $((100 + CELL + EPS - 1)) (one per episode). The OTHER arm pairs only if it is"
  echo "!!       launched on the SAME cell with the SAME count — press Enter at both prompts, PICK proposes exactly that."
  echo "!!       Verdict r (redo) DELETES the take and re-runs the same seed inside the batch."
fi
printf "%s\t%s\t%s\t%s\n" "$CELL" "$TASK" "$MODEL" "$EPS" >> "$RUNS"
SEED=$((100 + CELL))
EXTRA="$EXTRA --seed $SEED"
if [ -n "${PICK_MORE+x}" ]; then MORE=$PICK_MORE; [ -n "$MORE" ] && echo "append flags: $MORE (PICK_MORE)"; else read -p "append flags (Enter for none): " MORE; fi
[ -n "$MORE" ] && EXTRA="$EXTRA $MORE"

# attach to a warm policy server when one holds the SAME model (start one in
# another terminal with ./SERVE.sh — launches then take seconds, not minutes)
if [[ "$EXTRA" != *"--policy-server"* ]]; then
  ATTACHED=""; FOUND=""
  WANT=$(basename "$(readlink -f "$BASE/phantom/$CKPT" 2>/dev/null || echo "$CKPT")")
  # one port per menu slot (SERVE.sh: 7776 + slot); probe every slot (09-09
  # audit: only 7777-7779 were probed, so students on slots 4+ never attached
  # and every launch reloaded the model in-process)
  for PORT in $(seq 7777 $((7776 + i))); do
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
    # Match by CONTENT (sha256[:12] of the file the server loaded), never by
    # basename: eight menu rows share three basenames (student_001200.pt x3,
    # teacher_001200.pt x3, teacher_006000.pt x2), so a basename match would
    # attach stu_v5_6's launch to a warm stu_ftA_r1 server and record the
    # wrong model under the right label (audit 09-10).
    if [ -d "$BASE/phantom/$CKPT" ]; then
      WANT_SHA=$(cd $BASE/phantom && .venv/bin/python -c "from phantom.scripts.lerobot_server import dir_digest; print(dir_digest('$CKPT') or '')" 2>/dev/null)
    else
      WANT_SHA=$(sha256sum "$BASE/phantom/$CKPT" 2>/dev/null | cut -c1-12)
    fi
    if [ -n "$WANT_SHA" ] && [ "$SRV_SHA" = "$WANT_SHA" ]; then
      if [ "$SRV_STATE" = "busy" ]; then
        echo "!! the policy server on :$PORT holds $SRV_CKPT but is BUSY: another run_deploy is attached"
        echo "!! (possibly Ctrl-Z'd: 'jobs' / 'ps -ef | grep run_deploy'). Bring it to the foreground and end it."
        echo "!! Not launching a competing process."; exit 3
      fi
      echo ">> warm policy server on :$PORT holds $SRV_CKPT (sha $SRV_SHA = menu $MODEL) — attaching (fast start)"
      EXTRA="$EXTRA --policy-server 127.0.0.1:$PORT"; ATTACHED=1; break
    fi
  done
  if [ -z "$ATTACHED" ] && [ -n "$LEROBOT_TYPE" ]; then
    echo "!! $LEROBOT_TYPE runs only through its warm server: ./serve_bg.sh $m  (lerobot_server), then re-run PICK."; exit 3
  fi
  if [ -z "$ATTACHED" ] && [ -n "${extras[$m]}" ]; then
    # 09-15: rows served with extra flags (--flex --compile) must never run in-process: that would be a
    # different policy from the paired takes that went through the warm server.
    echo "!! row $m ($MODEL) is served with [${extras[$m]}] — an in-process launch would run a different policy."
    echo "!! Warm it first: ./serve_bg.sh $m   (or ./EXPERIMENT.sh <block>), then re-run PICK."; exit 3
  fi
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

EXTRA="$EXTRA --tag label:$MODEL"     # menu row label on every episode (stats.py pairs by it)
# 09-13 session levers, applied to EVERY row (both arms of a cell, phantom and LeRobot alike):
#  - descend-then-release placement supervisor (rig 09-12: policies plunged through the demo release band
#    into the 45 N wrench guard; replay converts 5 of 19 non-placements) + 0.35 s latch release dwell
#  - homing start bound y <= -0.28 m on waffles/Carton only (4-day analysis, 382 episodes: the start-y effect
#    holds on every day/task/model — box-side starts score 0.35 vs 1.66; the bad starts are INSIDE the demo
#    ±1σ draw, so this is a disclosed workaround). Egg: its own box since 09-15 (below). Whiteboard: none.
# PICK_NO_SESSION_EXTRA=1 disables them (attribution runs only).
# 09-15: folded into EXTRA BEFORE the confirmation so the operator sees the levers before launching.
SESSION_EXTRA="--placement-descent configs/placement_descent_rig_0913.json --grip-latch-release-s 0.35"
# 09-14 success envelope (188 episodes, 09-12+09-13, rule ordinal): placed starts sit higher — waffles z in
# [0.325, 0.362] places 0.24 vs 0.13 outside; Carton z >= 0.285 (weaker, n=8 placed); x carries nothing;
# y <= -0.28 confirmed on the unbounded 09-12 day (0.18 vs 0.08). Deterministic clamp: a paired cell keeps
# the same start on both arms.
# 09-15 egg (70 deploy episodes, 0 placements ever): the only signal is the LIFT proxy — all 5 held lifts
# (>50 mm, carried over the box) started at y <= -225 mm and z <= 280 mm (4/11 inside vs 1/52 outside,
# Fisher p = 0.0025; within one day+checkpoint 4/7 vs 0/10, p = 0.015). Egg's demo start is 46-70 mm lower
# than waffles/Carton, so the bound is a z CEILING, not a floor; y <= -0.28 is unreachable for egg draws.
case "$TASK" in
  waffles) SESSION_EXTRA="$SESSION_EXTRA --home-bounds y_max=-0.28,z_min=0.325";;
  Carton)  SESSION_EXTRA="$SESSION_EXTRA --home-bounds y_max=-0.28,z_min=0.285";;
  # 09-15 15:15: the one egg GRAB today (ep_teacher_egg_1789471598_004, row 14, marked 'c' after a safety stop):
  # both pads loaded (L 11 N / R 18 N), gripper closed at 2.1 s, held 15 s, lifted 135 mm, carried over the box.
  # It started at x -348, y -255, z 263 mm — inside the lift envelope above. Box of +-15 mm around that start
  # (y draws never go below -255, so y effectively sits in [-255, -240]).
  egg)     SESSION_EXTRA="$SESSION_EXTRA --home-bounds x_min=-0.363,x_max=-0.333,y_min=-0.270,y_max=-0.240,z_min=0.248,z_max=0.278";;
esac
# 09-14: PHANTOM rows also record the imagined-future agreement of the K seeds per replan (diag only,
# the default rule still picks the chunk) — the rig-side record for the world-model reliability claim.
[ -z "$LEROBOT_TYPE" ] && SESSION_EXTRA="$SESSION_EXTRA --agreement-shadow"
if [ -z "$PICK_NO_SESSION_EXTRA" ]; then EXTRA="$EXTRA $SESSION_EXTRA"; echo ">> session levers: $SESSION_EXTRA"; else echo "!! session levers DISABLED (PICK_NO_SESSION_EXTRA)"; fi
echo
echo ">> $MODEL ($CKPT, system=$SYSTEM) | $PRESET | $TASK x$EPS | CELL $CELL (seed $SEED) | nfe=$NFE | extra: [$EXTRA]"
if [ "$PRESET" = BOUNDED_REACH ]; then
  echo ">> reach fix: require the 'servo reach profile bounded_v1' effective-settings line at startup."
  echo ">> This enables the apex limiter. It does not configure a box release volume or qualify full placement."
fi
echo ">> reminders: both arms of cell $CELL share seed $SEED (same placement!); type the next cell number when the placement changes;"
echo ">>            a censored end (control_lost / servo_hold_timeout / servo_branch_fault) = re-run this cell, same number;"
echo ">>            verdicts: s placed | f failed | c CRUSHED (placed, but too hard on the object; counts as placed,"
echo ">>            tagged for the haptic benchmark) | r REDO: the take is DELETED as if it never ran and the same"
echo ">>            seed runs again (bad placement, false start, hand in frame). Append d for damage (fd, cd)."
echo ">>            joint gate must be green; stay attended until a gripper release is seen working."
echo ">>            during an episode: press Enter TWICE (within 1.5 s) or type  x  + Enter to end it cleanly"
echo ">>            (motion stops, gripper stays). A single Enter is ignored (stray newlines, 09-04)."
echo ">>            DO NOT use the robot E-stop to end an episode: it kills the control script (pendant reset)."
if [ -n "${PICK_GO:-}" ]; then echo ">> launching (PICK_GO)"; else read -p "Enter to launch (Ctrl-C to abort) "; fi
# Use the launcher from this same checkout; parent-directory copies can lag
# behind a git pull and previously left the fixed controller disabled.
LAUNCHER="$BASE/phantom/tools/rig/GO_ANY.sh"
[ -f "$LAUNCHER" ] || { echo "missing tracked launcher: $LAUNCHER"; exit 1; }
[ -n "$LEROBOT_TYPE" ] && SYSTEM=student
CKPT="$CKPT" SYSTEM="$SYSTEM" EXTRA="$EXTRA" exec bash "$LAUNCHER" "$TASK" "$EPS" "$NFE" 1.0
