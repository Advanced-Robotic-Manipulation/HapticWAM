#!/bin/bash
# Student of the MULTITASK sim-expert teacher (teacher_v6_simft_multitask/teacher_001500.pt)
# on a rented box. Run AFTER tools/provision_distill.sh AND tools/provision_dagger_r2.sh in the
# same workspace (they leave: repo, venv, cosmos, dataset_v3 demos, the 09-01..09-09 rollouts
# and the r3 manifest installed as manifests/all.jsonl). Pulls the teacher, writes the runner +
# hub watcher. Launches NOTHING.
#   HF_TOKEN=hf_xxx bash tools/provision_student_mt.sh [/workspace/phantom-hid] [STEPS=1000] [EVERY=250]
set -euo pipefail
W=${1:-/workspace/phantom-hid}; STEPS=${2:-1000}; EVERY=${3:-250}
HUB=armteam/phantom-checkpoints
: "${HF_TOKEN:?set HF_TOKEN inline}"
D=$W/data/phantom-episodes; PY=$W/.venv/bin/python
cd "$W/phantom"
NW=$(nproc); [ "$NW" -gt 16 ] && NW=16

hfget() { $PY - "$1" "${2:-$W/dl}" <<'EOF2'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("armteam/phantom-checkpoints", sys.argv[1], repo_type="model", local_dir=sys.argv[2]))
EOF2
}

echo "== multitask teacher (step 1500 = Ilya's pick: lowest sampled action MSE on his 78-episode val)"
mkdir -p "$W/runs/teacher/teacher_v6_simft_multitask"
T=$W/runs/teacher/teacher_v6_simft_multitask/teacher_001500.pt
[ -f "$T" ] || { hfget teacher_v6_simft_multitask/teacher_001500.pt >/dev/null; cp "$W/dl/teacher_v6_simft_multitask/teacher_001500.pt" "$T"; }
ls -la "$T"
[ -f "$D/manifests/all.jsonl" ] || { echo "FATAL: r3 manifest missing — run tools/provision_dagger_r2.sh first"; exit 1; }
grep -q "rollout" "$D/manifests/all.jsonl" || echo "WARNING: manifests/all.jsonl has no rollout rows — is this the r3 manifest?"

echo "== 1-step gate (window count + the teacher loads into the student layout)"
$PY -m phantom.train.distill_hid --teacher-ckpt "$T" --data "$D/tasks" --hardware configs/hardware.nuc.yaml \
    --run-name gate_mt --dagger-round 2 --max-steps 1 --grasp-frac 0.3 --photo-aug 1.0 \
    --teacher-nfe -1 --w-sigma 1.0 --batch-size 1 --grad-accum 1 --num-workers 4 --device cuda 2>&1 | tee /tmp/gate_mt.log | grep -E "HID dataset|val:|Error|error" | head -5
grep -qiE "error|Traceback" /tmp/gate_mt.log && { echo "FATAL: gate failed, see /tmp/gate_mt.log"; exit 1; }

mkdir -p "$W/eval"
cat > "$W/distill_mt.sh" <<EOF3
#!/bin/bash
# Pad-free student of the multitask teacher: the exact stu_ftA_r2 recipe (DAgger round-2 manifest,
# batch 2 x accum 4, teacher-nfe -1, w-sigma 1.0), $STEPS steps, checkpoint + val124 eval every $EVERY.
set -uo pipefail
cd $W/phantom
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$T; D=$D/tasks; HW=configs/hardware.nuc.yaml
ev() { echo "=== terminal_eval \$2 \$(date)"; $PY tools/terminal_eval.py --ckpt "\$1" --data \$D --hardware \$HW --nfe 1 --seeds 4 --split val --out $W/eval/\$2.json > $W/eval/\$2.log 2>&1 || echo "!!! eval \$2 FAILED"; grep -oE "\"endpoint_err_mm\": [0-9.]+" $W/eval/\$2.json | head -1; }
echo "=== hid_mt_r2 (pad-free student of teacher_v6_simft_multitask/1500) \$(date)"
( $PY -m phantom.train.distill_hid --teacher-ckpt "\$T" --data \$D --hardware \$HW --run-name hid_mt_r2 --dagger-round 2 \\
    --max-steps $STEPS --ckpt-every $EVERY --eval-every $EVERY --grasp-frac 0.3 --photo-aug 1.0 --teacher-nfe -1 --w-sigma 1.0 \\
    --batch-size 2 --grad-accum 4 --num-workers $NW --device cuda || echo "!!! hid_mt_r2 FAILED" ) &
TP=\$!
done_ck=""
while true; do
  for ck in \$(ls $W/runs/hid/hid_mt_r2_r2/student_*.pt 2>/dev/null | sort); do
    st=\$(basename \$ck .pt); case " \$done_ck " in *" \$st "*) continue;; esac
    sleep 60; ev "\$ck" mt_\$st; done_ck="\$done_ck \$st"
  done
  kill -0 \$TP 2>/dev/null || { sleep 90; for ck in \$(ls $W/runs/hid/hid_mt_r2_r2/student_*.pt 2>/dev/null | sort); do st=\$(basename \$ck .pt); case " \$done_ck " in *" \$st "*) continue;; esac; ev "\$ck" mt_\$st; done_ck="\$done_ck \$st"; done; break; }
  sleep 120
done
echo "ALL MT DONE \$(date) evaluated: \$done_ck"
EOF3
chmod +x "$W/distill_mt.sh"
cat > "$W/ckpt_watch_mt.sh" <<EOF3
#!/bin/bash
: "\${HF_TOKEN:?}"; export HF_HUB_DISABLE_XET=1
while true; do
  DIR=$W/runs/hid/hid_mt_r2_r2; [ -d "\$DIR" ] && $PY $W/phantom/tools/upload_run_ckpts.py "\$DIR" --repo $HUB --run-name hid_mt_r2 --log $W/distill_mt.log 2>&1 | tail -1
  ls $W/eval/mt_*.json >/dev/null 2>&1 && $PY - <<'PY'
import glob, os
from huggingface_hub import HfApi
api = HfApi()
for f in glob.glob("$W/eval/mt_*.json") + glob.glob("$W/eval/mt_*.log"):
    api.upload_file(path_or_fileobj=f, path_in_repo="eval_mt/" + os.path.basename(f), repo_id="$HUB", repo_type="model")
PY
  grep -q "ALL MT DONE" $W/distill_mt.log 2>/dev/null && { sleep 120; echo "ALL UPLOADS DONE"; exit 0; }
  sleep 300
done
EOF3
chmod +x "$W/ckpt_watch_mt.sh"
echo "READY (mt). Launch on GO:  nohup bash $W/distill_mt.sh > $W/distill_mt.log 2>&1 &   and   HF_TOKEN=hf_xxx nohup bash $W/ckpt_watch_mt.sh > $W/ckpt_watch_mt.log 2>&1 &"
