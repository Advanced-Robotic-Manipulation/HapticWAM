#!/bin/bash
# EXPERIMENT.sh — 2026-09-15 session driver: exactly the models the paper needs, warmed per block, in order.
#   ./EXPERIMENT.sh P       Block P   : 9 v6_simft2k (teacher) + S = its pad-free student (newest stu_simft_* row; STUDENT=<label> overrides)
#   ./EXPERIMENT.sh M       Block M   : + 14 v6_simft_mt1500 (multitask teacher), keeps 9 + S
#   ./EXPERIMENT.sh SV      Blocks S/V and N : row 9 only (nothing new warmed)
#   ./EXPERIMENT.sh PB      Block P-B : 14 + newest stu_mt_* row (MT_STUDENT=<label> overrides); stops the rest of ours
#   ./EXPERIMENT.sh E       Block E   : 12 stu_nowrist_4k + 2 stu_ftA_r2 next to 9; stops 14 and the mt student
#   ./EXPERIMENT.sh B       Block C   : baselines 4 pi05 + 5 dp + 6 xvla next to 9 (X-VLA sees one of its three views; reported as is)
#   ./EXPERIMENT.sh T1      Block C   : base teacher 1 v6 next to 9 (never with row 10)
#   ./EXPERIMENT.sh FINAL   the evening students: 9 + stu_simft_001000 (+ stu_mt_001000 with FINAL_MT=1); stops everything else of ours
#   ./EXPERIMENT.sh run <stage>   ONE SCRIPT: warm the stage AND launch PICK.sh for each of its models on the waffles cell plan
#                           (20 episodes per core arm, 10 per comparison arm, batches of 10, all takes into
#                           data/episodes/deploy/<day>_experiment/); only homing Enter + verdict prompts remain.
#   ./EXPERIMENT.sh status | stop | list | next
# Rows are resolved by LABEL from MODELS.tsv, so the row numbers fetch_student.sh appends do not matter; each block prints
# the PICK.sh row numbers to use. Never kills another user's process (SESSION_0912.sh / serve_bg.sh stop only our menu ports).
set -u
cd "$(dirname "$(readlink -f "$0")")"   # works through the ~/phantom-icra-2027/EXPERIMENT.sh symlink too
BASE=${PHANTOM_RIG_BASE:-$HOME/phantom-icra-2027}; TSV=$BASE/MODELS.tsv
ANCHOR=${EXP_ANCHOR:-v6_simft_mt1500}   # the teacher every arm pairs against (16:40: the multitask teacher; EXP_ANCHOR=v6_simft2k for the 09-13 one)
listening() { ss -ltn 2>/dev/null | grep -q ":$((7776 + $1)) "; }
warm_student() {  # the Block P student: a stu_simft_* row that is already warm wins (never switch arms mid-block), else the newest
  local r=0 label rest
  while IFS=$'\t' read -r label rest; do
    [ -z "$label" ] && continue; case "$label" in \#*) continue;; esac
    r=$((r + 1)); case "$label" in stu_simft_*) listening "$r" && { echo "$label"; return; };; esac
  done < "$TSV"
  newest stu_simft_; }
nrows() { awk -F'\t' '!/^#/ && $1!=""' "$TSV" | wc -l | tr -d ' '; }
row_of() { awk -F'\t' -v l="$1" '!/^#/ && $1!="" {n++; if ($1==l) {print n; exit}}' "$TSV"; }
newest() {  # newest row label with this prefix, by the zero-padded step in the label (stu_simft_001000 > stu_simft_000500)
  awk -F'\t' -v p="$1" '!/^#/ && $1!="" && index($1,p)==1 {print $1}' "$TSV" | sort | tail -1; }
need() { local r; r=$(row_of "$1"); [ -n "$r" ] || { echo "!! no menu row labelled '$1' (fetch_student.sh first? ./serve_bg.sh status shows the menu)" >&2; return 1; }; echo "$r"; }
warm_rows() { ./SESSION_0912.sh rows "$@"; }
stop_ours_except() {  # stop OUR servers (menu ports only) whose row is not in the keep list
  local keep=" $* " r
  local stopped=0
  for r in $(seq 1 "$(nrows)"); do
    case "$keep" in *" $r "*) continue;; esac
    ss -ltn 2>/dev/null | grep -q ":$((7776 + r)) " && { ./serve_bg.sh stop "$r"; stopped=1; }
  done
  # 17:05 lesson: a stopped server releases its GPU memory a few seconds after the kill; warming the next
  # model immediately OOM'd pi0.5. Wait until nvidia-smi shows the memory back (up to 30 s).
  if [ "$stopped" = 1 ]; then
    local i=0 prev=-1 used
    while [ $i -lt 15 ]; do
      sleep 2; i=$((i + 1))
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ $i -ge 3 ] && [ "$used" = "$prev" ] && break     # two equal readings after 6 s = released
      prev=$used
    done
    echo ">> GPU memory settled after ${i}x2 s: ${used} MiB used"
  fi; }
disk_check() {
  local free; free=$(df -BG --output=avail "$BASE" 2>/dev/null | tail -1 | tr -d ' G')
  echo ">> disk free on the box: ${free:-?} GB (a rig episode is ~0.25 GB; 120 launches ~ 30 GB)"
  [ -n "$free" ] && [ "$free" -lt 30 ] && echo "!! under 30 GB free — the sim campaign is filling the disk; check before launching"; return 0; }
show() { echo; ./serve_bg.sh status; echo ">> PICK.sh rows for this block: $*"; }
snap() {  # scene screenshot (training reference vs live camera) saved next to the takes and shown on the box screen
  local label=$1 c0=$2 c1=$3 dir="$OUT/screenshots" f
  mkdir -p "$dir"; f="$dir/$(date +%H%M%S)_${label}_${TASK}_cells${c0}-${c1}.png"
  if "$BASE/phantom/.venv/bin/python" "$BASE/snap.py" "$TASK" >/dev/null 2>&1 && [ -s /tmp/snap.png ]; then
    cp /tmp/snap.png "$f"; echo ">> screenshot saved: $f"
    DISPLAY=${DISPLAY:-:1} bash -c 'pkill -f "eog" 2>/dev/null; setsid nohup eog -f "$0" >/dev/null 2>&1 < /dev/null &' "$f"
  else echo "!! screenshot skipped (camera busy or snap.py failed)"; fi; }
case "${1:-}" in
  P)     disk_check; T=$(need "$ANCHOR") || exit 1; S=${STUDENT:-$(warm_student)}; [ -n "$S" ] || { echo "!! no stu_simft_* row yet — fetch_student.sh hid_simft 000500 first"; exit 1; }
         echo ">> Block P student: $S (a warm stu_simft_* row is kept; STUDENT=<label> overrides)"
         SR=$(need "$S") || exit 1; MT=$(row_of v6_simft_mt1500)
         # keep the multitask teacher warm if it already is (Block M follows P on the same cells); three servers is the ceiling
         stop_ours_except "$T" "$SR" "$MT"; warm_rows "$T" "$SR"; [ -n "$MT" ] && warm_rows "$MT"
         show "teacher v6_simft2k = $T   student $S = $SR${MT:+   (multitask teacher $MT stays warm for Block M if it fit)}";;
  M)     T=$(need "$ANCHOR") || exit 1; warm_rows "$T"; show "anchor teacher $ANCHOR = $T (the arm every other arm pairs against)";;
  S9)    T=$(need "$ANCHOR") || exit 1; S9=$(need v6_simft2k) || exit 1
         stop_ours_except "$T" "$S9"; warm_rows "$T" "$S9"; show "09-13 sim-expert teacher v6_simft2k = $S9 (optional, one launch per cell, pairs against $ANCHOR)";;
  SV)    T=$(need "$ANCHOR") || exit 1; warm_rows "$T"; show "row $T with preset 4 + the recorded 09-15 flags (S: --select-by video_agreement; V: + --agreement-veto THR; N: no --terminal-veto)";;
  PB)    MT=$(need v6_simft_mt1500) || exit 1; MS=${MT_STUDENT:-$(newest stu_mt_)}; [ -n "$MS" ] || { echo "!! no stu_mt_* row yet — fetch_student.sh hid_mt 000500 first"; exit 1; }
         MSR=$(need "$MS") || exit 1; T=$(row_of "$ANCHOR")
         stop_ours_except "$T" "$MT" "$MSR"; warm_rows "$MT" "$MSR"; show "multitask teacher = $MT   its student $MS = $MSR";;
  E)     T=$(need "$ANCHOR") || exit 1; W=$(need stu_nowrist_4k) || exit 1; F=$(need stu_ftA_r2) || exit 1
         stop_ours_except "$T" "$W" "$F"; warm_rows "$T" "$W" "$F"; show "wrist-masked student = $W   ftA student = $F (one launch per cell, vs row $T's Block P episodes)";;
  B)     T=$(need "$ANCHOR") || exit 1; P=$(need pi05) || exit 1; DPR=$(need dp) || exit 1   # baselines next to the anchor (pi0.5 7.9 GB own venv, DP 1.8 GB); X-VLA dropped 19:15 (not task-directed)
         stop_ours_except "$T" "$P" "$DPR"; warm_rows "$T" "$P" "$DPR"
         show "pi0.5 = $P   diffusion policy = $DPR (one launch per cell, vs the anchor's episodes)";;
  T1)    T=$(need "$ANCHOR") || exit 1; V=$(need v6) || exit 1; F10=$(row_of v6_fast)                  # base teacher v6 (never together with row 10, same checkpoint)
         [ -n "$F10" ] && listening "$F10" && ./serve_bg.sh stop "$F10"
         stop_ours_except "$T" "$V"; warm_rows "$T" "$V"; show "base teacher v6 = $V (one launch per cell, vs row $T's Block P episodes)";;
  NV)    T=$(need "$ANCHOR") || exit 1; S=$(need "${STUDENT:?STUDENT=<label> required}") || exit 1
         stop_ours_except "$T" "$S"; warm_rows "$T" "$S"; show "student $STUDENT = $S with the terminal veto OFF (preset 4, nfe 1, same levers)";;
  FINAL) disk_check; T=$(need "$ANCHOR") || exit 1; S=$(need "${STUDENT:-stu_simft_001000}") || exit 1; keep="$T $S"; MS=""
         if [ "${FINAL_MT:-0}" = 1 ]; then MS=$(need stu_mt_001000) || exit 1; keep="$keep $MS"; fi
         stop_ours_except $keep; warm_rows $keep; show "teacher v6_simft2k = $T   1000-step student = $S${MS:+   mt 1000-step student = $MS}";;
  run)    # ONE SCRIPT: warm the stage, then launch PICK.sh for every model of the stage on the waffles cell plan,
          # every take into its own experiment folder ($EXP_OUT). Only the deploy prompts remain (homing Enter,
          # verdicts). Resume after an interruption with EXP_FROM=<cell> (and EXP_ONLY=<label> for one model).
          #   EXP_TASK=waffles EXP_N_CORE=20 EXP_N_CMP=10 EXP_BATCH=10 EXP_OUT=<dir> ./EXPERIMENT.sh run <stage>
          # EXP_MORE="<run_deploy flags>" is appended to every launch of the run, e.g. EXP_MORE=--pad-free for a
          # student on a gripper WITHOUT tactile pads (pads never opened/read/recorded; episodes tagged padfree:on).
         STAGE=${2:?usage: ./EXPERIMENT.sh run <M|B|E|T1|P|FINAL|S9|NV>}; TASK=${EXP_TASK:-waffles}; NCORE=${EXP_N_CORE:-20}; NCMP=${EXP_N_CMP:-10}; BATCH=${EXP_BATCH:-10}
         # The experiment folder is sticky across midnight: the first run of a session writes it to
         # $BASE/data/episodes/deploy/.experiment_out and later runs reuse it (EXP_OUT overrides;
         # delete that file to start a new folder on a new day).
         STICKY=$BASE/data/episodes/deploy/.experiment_out
         OUT=${EXP_OUT:-$(cat "$STICKY" 2>/dev/null || true)}
         [ -n "$OUT" ] || OUT=$BASE/data/episodes/deploy/$(date +%Y%m%d)_experiment
         mkdir -p "$OUT"; [ -n "${EXP_OUT:-}" ] || echo "$OUT" > "$STICKY"   # an explicit EXP_OUT is a one-off (e.g. video takes) and does not move the sticky folder
         case $STAGE in
           M) MODELS="$ANCHOR"; N=$NCORE;;  S9) MODELS="v6_simft2k"; N=$NCORE;;  B) MODELS="pi05 dp"; N=$NCMP;;  E) MODELS="stu_ftA_r2 stu_nowrist_4k"; N=$NCMP;;
           T1) MODELS="v6"; N=$NCORE;;  P) MODELS="${STUDENT:-$(warm_student)}"; N=$NCORE;;  FINAL) MODELS="${STUDENT:-stu_simft_001000}"; N=$NCORE;;
           NV) MODELS="${STUDENT:?STUDENT=<label> required}"; N=$NCORE;;
           *) echo "!! unknown stage $STAGE"; exit 2;;
         esac
         [ -n "${EXP_ONLY:-}" ] && MODELS=$EXP_ONLY
         PRESET=1; NFE=""; FLAGS=""
         if [ "$STAGE" = NV ]; then PRESET=4; NFE=1; FLAGS="--parity-fixes --k-seeds 4 --max-play-steps 16 --grip-play-steps 10 --max-episode-s 150 --max-replans 200"; fi
         "$0" "$STAGE" || { echo "!! warm-up failed — nothing launched"; exit 1; }
         echo "$STAGE" > "$BASE/.experiment_stage"
         echo; echo ">> RUN $STAGE: task $TASK, $N episodes per model in batches of $BATCH, recordings -> $OUT"; echo ">> models: $MODELS"
         [ -n "${EXP_MORE:-}" ] && echo ">> extra run_deploy flags on every launch: $EXP_MORE"
         FROM=${EXP_FROM:-1}
         for L in $MODELS; do
           r=$(row_of "$L"); [ -n "$r" ] || { echo "!! no row for $L"; exit 1; }
           c=$FROM; FROM=1      # EXP_FROM applies to the first model only; the following models start at cell 1
           while [ "$c" -le "$N" ]; do
             n=$BATCH; [ $((c + n - 1)) -gt "$N" ] && n=$((N - c + 1))
             echo; echo "=============== $L (row $r): $TASK cells $c..$((c + n - 1)) ($n episodes) ==============="
             snap "$L" "$c" "$((c + n - 1))"
             PICK_TASK=$TASK PICK_PRESET=$PRESET PICK_NFE=$NFE PICK_FLAGS="$FLAGS" PICK_CELL=$c PICK_EPS=$n PICK_MORE="--out $OUT${EXP_MORE:+ $EXP_MORE}" PICK_GO=1 ./PICK.sh "$r" \
               || { echo "!! launch of $L at cell $c ended with an error. Fix, then resume: EXP_ONLY=$L EXP_FROM=$c ./EXPERIMENT.sh run $STAGE"; exit 1; }
             c=$((c + n))
           done
           echo ">> $L done ($N episodes). Placed so far in $OUT: $(grep -l '"success": true' "$OUT"/ep_*/meta.json 2>/dev/null | wc -l | tr -d ' ') of $(ls -d "$OUT"/ep_* 2>/dev/null | wc -l | tr -d ' ') takes"
         done
         echo; echo ">> STAGE $STAGE COMPLETE. Next: ./EXPERIMENT.sh run <next stage>   (order: M B E T1 P FINAL, S9 optional)";;
  next)   # sequential mode: advance through the evening's order, one stage per call (state in $BASE/.experiment_stage)
         ORDER="M B E T1 P FINAL S9"; SF=$BASE/.experiment_stage; cur=$(cat "$SF" 2>/dev/null || echo "")
         nxt=""; if [ -z "$cur" ]; then nxt=${ORDER%% *}; else found=0; for s in $ORDER; do [ "$found" = 1 ] && { nxt=$s; break; }; [ "$s" = "$cur" ] && found=1; done; fi
         [ -n "$nxt" ] || { echo ">> all stages done ($ORDER). Use an explicit stage to repeat one."; exit 0; }
         echo ">> stage $nxt (after: ${cur:-start}); order = $ORDER"; echo "$nxt" > "$SF"; exec "$0" "$nxt";;
  list)   echo "order: M  B  E  T1  P  FINAL  S9   (current: $(cat "$BASE/.experiment_stage" 2>/dev/null || echo none))"
         echo "  M     anchor teacher: $ANCHOR (EXP_ANCHOR overrides)"
         echo "  S9    optional: row 9, the 09-13 sim-expert teacher, 20 episodes"
         echo "  P     + newest stu_simft_* student (arm B of the paired table)"
         echo "  B     anchor + pi0.5 + diffusion policy"
         echo "  E     row 9 + stu_nowrist_4k + stu_ftA_r2"
         echo "  T1    row 9 + base teacher v6"
         echo "  FINAL row 9 + stu_simft_001000 (+ stu_mt_001000 with FINAL_MT=1)"
         echo "  (PB = multitask student, only on a GO; SV = pilots on row 9)";;
  status) ./SESSION_0912.sh status; disk_check;;
  stop)   ./SESSION_0912.sh stop;;
  *) sed -n 2,13p "$0"; echo "  ./EXPERIMENT.sh next | list   sequential mode"; exit 2;;
esac
