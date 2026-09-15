#!/bin/bash
# EXPERIMENT.sh — 2026-09-15 session driver: exactly the models the paper needs, warmed per block, in order.
#   ./EXPERIMENT.sh P       Block P   : 9 v6_simft2k (teacher) + S = its pad-free student (newest stu_simft_* row; STUDENT=<label> overrides)
#   ./EXPERIMENT.sh M       Block M   : + 14 v6_simft_mt1500 (multitask teacher), keeps 9 + S
#   ./EXPERIMENT.sh SV      Blocks S/V and N : row 9 only (nothing new warmed)
#   ./EXPERIMENT.sh PB      Block P-B : 14 + newest stu_mt_* row (MT_STUDENT=<label> overrides); stops the rest of ours
#   ./EXPERIMENT.sh E       Block E   : 12 stu_nowrist_4k + 2 stu_ftA_r2 next to 9; stops 14 and the mt student
#   ./EXPERIMENT.sh FINAL   the evening students: 9 + stu_simft_001000 (+ stu_mt_001000 with FINAL_MT=1); stops everything else of ours
#   ./EXPERIMENT.sh status | stop
# Rows are resolved by LABEL from MODELS.tsv, so the row numbers fetch_student.sh appends do not matter; each block prints
# the PICK.sh row numbers to use. Never kills another user's process (SESSION_0912.sh / serve_bg.sh stop only our menu ports).
set -u
cd "$(dirname "$0")"
BASE=${PHANTOM_RIG_BASE:-$HOME/phantom-icra-2027}; TSV=$BASE/MODELS.tsv
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
  P)     disk_check; T=$(need v6_simft2k) || exit 1; S=${STUDENT:-$(newest stu_simft_)}; [ -n "$S" ] || { echo "!! no stu_simft_* row yet — fetch_student.sh hid_simft 000500 first"; exit 1; }
         SR=$(need "$S") || exit 1
         stop_ours_except "$T" "$SR"; warm_rows "$T" "$SR"; show "teacher v6_simft2k = $T   student $S = $SR";;
  M)     T=$(need v6_simft2k) || exit 1; MT=$(need v6_simft_mt1500) || exit 1
         warm_rows "$T" "$MT"; show "multitask teacher v6_simft_mt1500 = $MT (one launch per cell, pairs against row $T's Block P episodes)";;
  SV)    T=$(need v6_simft2k) || exit 1; warm_rows "$T"; show "row $T with preset 4 + the flags in RUN_SHEET_0915.md (S: --select-by video_agreement; V: + --agreement-veto THR; N: no --terminal-veto)";;
  PB)    MT=$(need v6_simft_mt1500) || exit 1; MS=${MT_STUDENT:-$(newest stu_mt_)}; [ -n "$MS" ] || { echo "!! no stu_mt_* row yet — fetch_student.sh hid_mt 000500 first"; exit 1; }
         MSR=$(need "$MS") || exit 1; T=$(row_of v6_simft2k)
         stop_ours_except "$T" "$MT" "$MSR"; warm_rows "$MT" "$MSR"; show "multitask teacher = $MT   its student $MS = $MSR";;
  E)     T=$(need v6_simft2k) || exit 1; W=$(need stu_nowrist_4k) || exit 1; F=$(need stu_ftA_r2) || exit 1
         stop_ours_except "$T" "$W" "$F"; warm_rows "$T" "$W" "$F"; show "wrist-masked student = $W   ftA student = $F (one launch per cell, vs row $T's Block P episodes)";;
  FINAL) disk_check; T=$(need v6_simft2k) || exit 1; S=$(need stu_simft_001000) || exit 1; keep="$T $S"; MS=""
         if [ "${FINAL_MT:-0}" = 1 ]; then MS=$(need stu_mt_001000) || exit 1; keep="$keep $MS"; fi
         stop_ours_except $keep; warm_rows $keep; show "teacher v6_simft2k = $T   1000-step student = $S${MS:+   mt 1000-step student = $MS}";;
  status) ./SESSION_0912.sh status; disk_check;;
  stop)   ./SESSION_0912.sh stop;;
  *) sed -n 2,11p "$0"; exit 2;;
esac
