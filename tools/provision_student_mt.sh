#!/bin/bash
# Pad-free student of a SIM-EXPERT teacher on a rented box. Default teacher = the 09-13
# teacher_v6_simft/teacher_002000.pt (already on the hub in armteam/hapticwam-teacher; placed waffles 7/12 + Carton 6/10 on the
# rig); pass the multitask teacher_v6_simft_multitask/teacher_001500.pt instead once uploaded. Run AFTER tools/provision_distill.sh AND tools/provision_dagger_r2.sh in the
# same workspace (they leave: repo, venv, cosmos, dataset_v3 demos, the 09-01..09-09 rollouts
# and the r3 manifest installed as manifests/all.jsonl). Pulls the teacher, writes the runner +
# hub watcher. Launches NOTHING.
#   HF_TOKEN=hf_xxx bash tools/provision_student_mt.sh [/workspace/phantom-hid] [STEPS=1000] [EVERY=250] \
#        [TEACHER=teacher_v6_simft/teacher_002000.pt] [RUN=hid_simft]
set -euo pipefail
W=${1:-/workspace/phantom-hid}; STEPS=${2:-1000}; EVERY=${3:-250}
TEACHER=${4:-teacher_v6_simft/teacher_002000.pt}; RUN=${5:-hid_simft}
# hub repos after the 2026-09 armteam restructure:
HUB_TEACHER=armteam/hapticwam-teacher     # teacher_v6_simft/, teacher_v6/
HUB_STUDENT=armteam/hapticwam-student     # hid_simft/
HUB_ABL=armteam/hapticwam-ablations       # teacher_v6_simft_multitask/, hid_mt/, eval_mt/
# which repo holds the requested teacher, and which one this run's checkpoints go to
case "$TEACHER" in
  teacher_v6_simft/*|teacher_v6/*) TEACHER_HUB=$HUB_TEACHER ;;
  *)                               TEACHER_HUB=$HUB_ABL ;;   # multitask + every other v6 variant
esac
case "$RUN" in
  hid_mt)      RUN_HUB=$HUB_ABL ;;
  hid_simft|*) RUN_HUB=$HUB_STUDENT ;;   # default: a pad-free student run belongs to the student repo
esac
: "${HF_TOKEN:?set HF_TOKEN inline}"
D=$W/data/phantom-episodes; PY=$W/.venv/bin/python
cd "$W/phantom"
NW=$(nproc); [ "$NW" -gt 16 ] && NW=16

hfget() { $PY - "$1" "$2" "${3:-$W/dl}" <<'EOF2'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], repo_type="model", local_dir=sys.argv[3]))
EOF2
}

echo "== teacher $TEACHER -> run $RUN"
mkdir -p "$W/runs/teacher/$(dirname $TEACHER)"
T=$W/runs/teacher/$TEACHER
[ -f "$T" ] || { hfget "$TEACHER_HUB" "$TEACHER" >/dev/null; cp "$W/dl/$TEACHER" "$T"; }
ls -la "$T"
[ -f "$D/manifests/all.jsonl" ] || { echo "FATAL: r3 manifest missing — run tools/provision_dagger_r2.sh first"; exit 1; }
grep -q "rollout" "$D/manifests/all.jsonl" || echo "WARNING: manifests/all.jsonl has no rollout rows — is this the r3 manifest?"

echo "== 1-step gate (window count + the teacher loads into the student layout)"
$PY -m phantom.train.distill_hid --teacher-ckpt "$T" --data "$D/tasks" --hardware configs/hardware.nuc.yaml \
    --run-name gate_$RUN --dagger-round 2 --max-steps 1 --grasp-frac 0.3 --photo-aug 1.0 \
    --teacher-nfe -1 --w-sigma 1.0 --batch-size 1 --grad-accum 1 --num-workers 4 --device cuda 2>&1 | tee /tmp/gate_mt.log | grep -E "HID dataset|val:|Error|error" | head -5
grep -qiE "error|Traceback" /tmp/gate_mt.log && { echo "FATAL: gate failed, see /tmp/gate_mt.log"; exit 1; }

mkdir -p "$W/eval"
cat > "$W/distill_mt.sh" <<EOF3
#!/bin/bash
# Pad-free student of $TEACHER: the exact stu_ftA_r2 recipe (DAgger round-2 manifest,
# batch 2 x accum 4, teacher-nfe -1, w-sigma 1.0), $STEPS steps, checkpoint + val124 eval every $EVERY.
set -uo pipefail
cd $W/phantom
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$T; D=$D/tasks; HW=configs/hardware.nuc.yaml
ev() { echo "=== terminal_eval \$2 \$(date)"; $PY tools/terminal_eval.py --ckpt "\$1" --data \$D --hardware \$HW --nfe 1 --seeds 4 --split val --out $W/eval/\$2.json > $W/eval/\$2.log 2>&1 || echo "!!! eval \$2 FAILED"; grep -oE "\"endpoint_err_mm\": [0-9.]+" $W/eval/\$2.json | head -1; }
echo "=== $RUN (pad-free student of $TEACHER) \$(date)"
( $PY -m phantom.train.distill_hid --teacher-ckpt "\$T" --data \$D --hardware \$HW --run-name $RUN --dagger-round 2 \\
    --max-steps $STEPS --ckpt-every $EVERY --eval-every $EVERY --grasp-frac 0.3 --photo-aug 1.0 --teacher-nfe -1 --w-sigma 1.0 \\
    --batch-size 2 --grad-accum 4 --num-workers $NW --device cuda || echo "!!! $RUN FAILED" ) &
TP=\$!
done_ck=""
while true; do
  for ck in \$(ls $W/runs/hid/${RUN}_r2/student_*.pt 2>/dev/null | sort); do
    st=\$(basename \$ck .pt); case " \$done_ck " in *" \$st "*) continue;; esac
    sleep 60; ev "\$ck" ${RUN}_\$st; done_ck="\$done_ck \$st"
  done
  kill -0 \$TP 2>/dev/null || { sleep 90; for ck in \$(ls $W/runs/hid/${RUN}_r2/student_*.pt 2>/dev/null | sort); do st=\$(basename \$ck .pt); case " \$done_ck " in *" \$st "*) continue;; esac; ev "\$ck" ${RUN}_\$st; done_ck="\$done_ck \$st"; done; break; }
  sleep 120
done
echo "ALL MT DONE \$(date) evaluated: \$done_ck"
EOF3
chmod +x "$W/distill_mt.sh"
cat > "$W/ckpt_watch_mt.sh" <<EOF3
#!/bin/bash
: "\${HF_TOKEN:?}"; export HF_HUB_DISABLE_XET=1
while true; do
  DIR=$W/runs/hid/${RUN}_r2; [ -d "\$DIR" ] && $PY $W/phantom/tools/upload_run_ckpts.py "\$DIR" --repo $RUN_HUB --run-name $RUN --log $W/distill_mt.log 2>&1 | tail -1
  ls $W/eval/${RUN}_*.json >/dev/null 2>&1 && $PY - <<'PY'
import glob, os
from huggingface_hub import HfApi
api = HfApi()
for f in glob.glob("$W/eval/${RUN}_*.json") + glob.glob("$W/eval/${RUN}_*.log"):
    api.upload_file(path_or_fileobj=f, path_in_repo="eval_mt/" + os.path.basename(f), repo_id="$HUB_ABL", repo_type="model")
PY
  grep -q "ALL MT DONE" $W/distill_mt.log 2>/dev/null && { sleep 120; echo "ALL UPLOADS DONE"; exit 0; }
  sleep 300
done
EOF3
chmod +x "$W/ckpt_watch_mt.sh"
echo "READY (mt). Launch on GO:  nohup bash $W/distill_mt.sh > $W/distill_mt.log 2>&1 &   and   HF_TOKEN=hf_xxx nohup bash $W/ckpt_watch_mt.sh > $W/ckpt_watch_mt.log 2>&1 &"
