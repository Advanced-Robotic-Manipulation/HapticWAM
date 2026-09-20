#!/bin/bash
# Pull a rental student checkpoint from the hub into the menu (09-15). Run with the token inline:
#   HF_TOKEN=... ./fetch_student.sh <hid_simft|hid_mt> <000500|000750|001000> [row label]
# Downloads straight into runs/<run>/student_<step>.pt (no duplicate copy), appends a MODELS.tsv row
# (label stu_simft_000500 / stu_mt_000500, system=student, repo-relative path as PICK.sh expects). Does NOT touch
# the repo mirror tools/rig/MODELS.tsv (a dirty tracked file would block git pull).
set -euo pipefail; : "${HF_TOKEN:?}"; RUN=$1; ST=$2; LABEL=${3:-stu_${RUN#hid_}_$ST}
# hub repos after the 2026-09 armteam restructure — the run decides the repo
case "$RUN" in
  hid_mt)      HUB=armteam/hapticwam-ablations ;;   # multitask ablation student
  hid_simft|*) HUB=armteam/hapticwam-student ;;     # default: pad-free student runs
esac
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python - "$RUN" "$ST" "$HUB" <<PY
import os, sys
from huggingface_hub import hf_hub_download
run, st, hub = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(hub, f"{run}/student_{st}.pt", repo_type="model", local_dir="runs", token=os.environ["HF_TOKEN"])
print("fetched", p, os.path.getsize(p))
PY
[ -s "runs/$RUN/student_$ST.pt" ] || { echo "!! runs/$RUN/student_$ST.pt missing after download"; exit 1; }
cd /home/physicalai/phantom-icra-2027
if [ "$RUN" = hid_simft ]; then T="the 09-13 sim-expert teacher, row 9"; else T="the multitask teacher, row 14"; fi
grep -q "^$LABEL	" MODELS.tsv || printf "%s\t%s\t%s\t%s\n" "$LABEL" "runs/$RUN/student_$ST.pt" "PAD-FREE STUDENT of $T (rental 09-15, $ST steps, r2 recipe; val124 number in hub eval_mt/${RUN}_student_$ST.json) - Block P arm B candidate" "student" >> MODELS.tsv
echo "row added: $LABEL = row $(awk -F"\t" "!/^#/ && \$1!=\"\" {n++; if (\$1==\"$LABEL\") print n}" MODELS.tsv) (rows now $(awk -F"\t" "!/^#/ && \$1!=\"\"" MODELS.tsv | wc -l))"
