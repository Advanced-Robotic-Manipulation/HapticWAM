#!/bin/bash
# EXPERIMENT.sh — 2026-09-15 session driver: exactly the models the paper needs, warmed per block, in order.
#   ./EXPERIMENT.sh P       Block P   : 9 v6_simft2k (teacher) + S = its pad-free student (newest stu_simft_* row; STUDENT=<label> overrides)
#   ./EXPERIMENT.sh M       Block M   : + 14 v6_simft_mt1500 (multitask teacher), keeps 9 + S
#   ./EXPERIMENT.sh SV      Blocks S/V and N : row 9 only (nothing new warmed)
#   ./EXPERIMENT.sh PB      Block P-B : 14 + newest stu_mt_* row (MT_STUDENT=<label> overrides); stops the rest of ours
#   ./EXPERIMENT.sh E       Block E   : 12 stu_nowrist_4k + 2 stu_ftA_r2 next to 9; stops 14 and the mt student
#   ./EXPERIMENT.sh B       Block C   : baselines 4 pi05 + 5 dp next to 9 (one launch per cell); X-VLA cannot run on the rig (3 views)
#   ./EXPERIMENT.sh T1      Block C   : base teacher 1 v6 next to 9 (never with row 10)
#   ./EXPERIMENT.sh FINAL   the evening students: 9 + stu_simft_001000 (+ stu_mt_001000 with FINAL_MT=1); stops everything else of ours
#   ./EXPERIMENT.sh status | stop
# Rows are resolved by LABEL from MODELS.tsv, so the row numbers fetch_student.sh appends do not matter; each block prints
# the PICK.sh row numbers to use. Never kills another user's process (SESSION_0912.sh / serve_bg.sh stop only our menu ports).
set -u
cd "$(dirname "$(readlink -f "$0")")"   # works through the ~/phantom-icra-2027/EXPERIMENT.sh symlink too
BASE=${PHANTOM_RIG_BASE:-$HOME/phantom-icra-2027}; TSV=$BASE/MODELS.tsv
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
  for r in $(seq 1 "$(nrows)"); do
    case "$keep" in *" $r "*) continue;; esac
    ss -ltn 2>/dev/null | grep -q ":$((7776 + r)) " && ./serve_bg.sh stop "$r"
  done; }
disk_check() {
  local free; free=$(df -BG --output=avail "$BASE" 2>/dev/null | tail -1 | tr -d ' G')
  echo ">> disk free on the box: ${free:-?} GB (a rig episode is ~0.25 GB; 120 launches ~ 30 GB)"
  [ -n "$free" ] && [ "$free" -lt 30 ] && echo "!! under 30 GB free — the sim campaign is filling the disk; ask Ilya before launching"; return 0; }
show() { echo; ./serve_bg.sh status; echo ">> PICK.sh rows for this block: $*"; }
case "${1:-}" in
  P)     disk_check; T=$(need v6_simft2k) || exit 1; S=${STUDENT:-$(warm_student)}; [ -n "$S" ] || { echo "!! no stu_simft_* row yet — fetch_student.sh hid_simft 000500 first"; exit 1; }
         echo ">> Block P student: $S (a warm stu_simft_* row is kept; STUDENT=<label> overrides)"
         SR=$(need "$S") || exit 1; MT=$(row_of v6_simft_mt1500)
         # keep the multitask teacher warm if it already is (Block M follows P on the same cells); three servers is the ceiling
         stop_ours_except "$T" "$SR" "$MT"; warm_rows "$T" "$SR"; [ -n "$MT" ] && warm_rows "$MT"
         show "teacher v6_simft2k = $T   student $S = $SR${MT:+   (multitask teacher $MT stays warm for Block M if it fit)}";;
  M)     T=$(need v6_simft2k) || exit 1; MT=$(need v6_simft_mt1500) || exit 1
         warm_rows "$T" "$MT"; show "multitask teacher v6_simft_mt1500 = $MT (one launch per Block P cell, pairs against row $T's Block P episodes)";;
  SV)    T=$(need v6_simft2k) || exit 1; warm_rows "$T"; show "row $T with preset 4 + the flags in RUN_SHEET_0915.md (S: --select-by video_agreement; V: + --agreement-veto THR; N: no --terminal-veto)";;
  PB)    MT=$(need v6_simft_mt1500) || exit 1; MS=${MT_STUDENT:-$(newest stu_mt_)}; [ -n "$MS" ] || { echo "!! no stu_mt_* row yet — fetch_student.sh hid_mt 000500 first"; exit 1; }
         MSR=$(need "$MS") || exit 1; T=$(row_of v6_simft2k)
         stop_ours_except "$T" "$MT" "$MSR"; warm_rows "$MT" "$MSR"; show "multitask teacher = $MT   its student $MS = $MSR";;
  E)     T=$(need v6_simft2k) || exit 1; W=$(need stu_nowrist_4k) || exit 1; F=$(need stu_ftA_r2) || exit 1
         stop_ours_except "$T" "$W" "$F"; warm_rows "$T" "$W" "$F"; show "wrist-masked student = $W   ftA student = $F (one launch per cell, vs row $T's Block P episodes)";;
  B)     T=$(need v6_simft2k) || exit 1; P=$(need pi05) || exit 1; DPR=$(need dp) || exit 1; XV=$(need xvla) || exit 1   # baselines next to row 9 (pi0.5 7.9 GB own venv, DP 1.8 GB, X-VLA 3 GB; ~20 GB total)
         stop_ours_except "$T" "$P" "$DPR" "$XV"; warm_rows "$T" "$P" "$DPR" "$XV"
         show "pi0.5 = $P   diffusion policy = $DPR   X-VLA = $XV (one launch per cell, vs row $T's Block P episodes; X-VLA sees only the scene camera of its 3 declared views — reported as is)";;
  T1)    T=$(need v6_simft2k) || exit 1; V=$(need v6) || exit 1; F10=$(row_of v6_fast)                  # base teacher v6 (never together with row 10, same checkpoint)
         [ -n "$F10" ] && listening "$F10" && ./serve_bg.sh stop "$F10"
         stop_ours_except "$T" "$V"; warm_rows "$T" "$V"; show "base teacher v6 = $V (one launch per cell, vs row $T's Block P episodes)";;
  FINAL) disk_check; T=$(need v6_simft2k) || exit 1; S=$(need stu_simft_001000) || exit 1; keep="$T $S"; MS=""
         if [ "${FINAL_MT:-0}" = 1 ]; then MS=$(need stu_mt_001000) || exit 1; keep="$keep $MS"; fi
         stop_ours_except $keep; warm_rows $keep; show "teacher v6_simft2k = $T   1000-step student = $S${MS:+   mt 1000-step student = $MS}";;
  next)   # sequential mode: advance through the evening's order, one stage per call (state in $BASE/.experiment_stage)
         ORDER="M B E T1 P FINAL"; SF=$BASE/.experiment_stage; cur=$(cat "$SF" 2>/dev/null || echo "")
         nxt=""; if [ -z "$cur" ]; then nxt=${ORDER%% *}; else found=0; for s in $ORDER; do [ "$found" = 1 ] && { nxt=$s; break; }; [ "$s" = "$cur" ] && found=1; done; fi
         [ -n "$nxt" ] || { echo ">> all stages done ($ORDER). Use an explicit stage to repeat one."; exit 0; }
         echo ">> stage $nxt (after: ${cur:-start}); order = $ORDER"; echo "$nxt" > "$SF"; exec "$0" "$nxt";;
  list)   echo "order: M  B  E  T1  P  FINAL   (current: $(cat "$BASE/.experiment_stage" 2>/dev/null || echo none))"
         echo "  M     row 9 sim-expert teacher + row 14 multitask teacher"
         echo "  P     + newest stu_simft_* student (arm B of the paired table)"
         echo "  B     row 9 + pi0.5 + diffusion policy + X-VLA"
         echo "  E     row 9 + stu_nowrist_4k + stu_ftA_r2"
         echo "  T1    row 9 + base teacher v6"
         echo "  FINAL row 9 + stu_simft_001000 (+ stu_mt_001000 with FINAL_MT=1)"
         echo "  (PB = multitask student, only on a GO; SV = pilots on row 9)";;
  status) ./SESSION_0912.sh status; disk_check;;
  stop)   ./SESSION_0912.sh stop;;
  *) sed -n 2,13p "$0"; echo "  ./EXPERIMENT.sh next | list   sequential mode"; exit 2;;
esac
