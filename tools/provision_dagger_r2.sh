#!/bin/bash
# DAgger round-2 overlay for a rented box. Run AFTER tools/provision_distill.sh in
# the same workspace (base dataset, repo, venv, cosmos, the ftA teacher). Places the
# 09-01/09-04 + 09-08/09-09 rig rollouts, installs the r3 manifest AS all.jsonl
# (distill_hid reads manifests/all.jsonl by name), runs a 1-step window-count gate,
# and writes the two-student runner + watcher. Launches NOTHING.
#   HF_TOKEN=hf_xxx bash tools/provision_dagger_r2.sh [/workspace/phantom-hid]
set -euo pipefail
W=${1:-/workspace/phantom-hid}
HUB=armteam/phantom-checkpoints
: "${HF_TOKEN:?set HF_TOKEN inline}"
D=$W/data/phantom-episodes; export D
PY=$W/.venv/bin/python
cd "$W/phantom"

hfget() { $PY - "$1" "${2:-$W/dl}" <<'EOF2'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("armteam/phantom-checkpoints", sys.argv[1], repo_type="model", local_dir=sys.argv[2]))
EOF2
}

echo "== teacher ftA"
mkdir -p "$W/runs/teacher/teacher_v5_ftA"
[ -f "$W/runs/teacher/teacher_v5_ftA/teacher_001500.pt" ] || { hfget teacher_v5_ftA/teacher_001500.pt >/dev/null; cp "$W/dl/teacher_v5_ftA/teacher_001500.pt" "$W/runs/teacher/teacher_v5_ftA/"; }
T=$W/runs/teacher/teacher_v5_ftA/teacher_001500.pt

echo "== rollouts 09-08/09-09 (re-derived + labelled, tasks/<task>_rollout layout)"
hfget dataset_v3_packed/rollouts_0908_0909.tar.zst >/dev/null
tar --zstd -xf "$W/dl/dataset_v3_packed/rollouts_0908_0909.tar.zst" -C "$D"
echo "== rollouts 09-01/09-04 (session layout -> symlinks, rederive, label)"
hfget rollouts/rollouts_0901_0904.tar.zst >/dev/null
mkdir -p "$D/rollouts_0901_0904" && tar --zstd -xf "$W/dl/rollouts/rollouts_0901_0904.tar.zst" -C "$D/rollouts_0901_0904"
$PY - <<'EOF2'
import json, os
from pathlib import Path
D = Path(os.environ.get("D", "/workspace/phantom-hid/data/phantom-episodes"))
n = 0
for mp in sorted((D/"rollouts_0901_0904").rglob("ep_*/meta.json")):
    dst = D/"tasks"/f"{json.loads(mp.read_text())['task']}_rollout"/mp.parent.name
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists(): os.symlink(mp.parent, dst); n += 1
print("linked", n, "old rollouts")
EOF2
$PY tools/rederive_rollout_actions.py "$D/rollouts_0901_0904"
$PY tools/label_grasps.py "$D/rollouts_0901_0904" --include-unfinalized --json-out /tmp/labels_0901_0904.json
$PY tools/apply_grasp_labels.py /tmp/labels_0901_0904.json "$D/rollouts_0901_0904"

echo "== manifest r3 AS all.jsonl"
hfget dataset_v3_packed/manifests_r3.tar >/dev/null
mkdir -p /tmp/r3 && tar -xf "$W/dl/dataset_v3_packed/manifests_r3.tar" -C /tmp/r3
[ -f "$D/manifests/all_base.jsonl" ] || cp "$D/manifests/all.jsonl" "$D/manifests/all_base.jsonl"
cp /tmp/r3/all_r3_windows.jsonl "$D/manifests/all.jsonl"
$PY - "$D/manifests/all.jsonl" <<'EOF2'
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
c = collections.Counter(r.get("split") for r in rows); print("manifest:", dict(c))
assert c["train"] == 1090 and c["val"] == 124, "expected 1090/124 (all_r3_windows)"
EOF2

NW=$(( $(nproc) / 2 )); [ "$NW" -gt 16 ] && NW=16; [ "$NW" -lt 2 ] && NW=2
echo "== 1-step window-count gate"
$PY -m phantom.train.distill_hid --teacher-ckpt "$T" --data "$D/tasks" --hardware configs/hardware.nuc.yaml \
    --run-name gate_r2 --dagger-round 2 --max-steps 1 --grasp-frac 0.3 --photo-aug 1.0 \
    --teacher-nfe -1 --w-sigma 1.0 --batch-size 1 --grad-accum 1 --num-workers $NW --device cuda 2>&1 | tee /tmp/gate_r2.log | grep -E "HID dataset|val:|Error" | head -3
grep -q "8720 windows" /tmp/gate_r2.log || { echo "GATE FAILED: expected 'HID dataset: 8720 windows' — manifest/rollout placement wrong"; exit 1; }
rm -rf "$W/runs/hid/gate_r2_r2"

STEPS=${STEPS:-2000}
mkdir -p "$W/eval"
cat > "$W/distill_r2.sh" <<EOF3
#!/bin/bash
# Two students from ftA: hid_r2_ftA (demos + 123 rig rollouts, DAgger round 2) and
# hid_2k_ftA (demos only, same recipe/steps = the controlled comparison). Then evals.
set -uo pipefail
cd $W/phantom
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$T; D=$D/tasks; HW=configs/hardware.nuc.yaml
ev() { echo "=== terminal_eval \$2 \$(date)"; $PY tools/terminal_eval.py --ckpt "\$1" --data \$D --hardware \$HW --nfe 1 --seeds 4 --split val --out $W/eval/\$2.json > $W/eval/\$2.log 2>&1 || echo "!!! eval \$2 FAILED"; grep -oE "endpoint_err_mm[^,}]*" $W/eval/\$2.json | head -1; }
echo "=== hid_r2_ftA (rollouts) \$(date)"
$PY -m phantom.train.distill_hid --teacher-ckpt "\$T" --data \$D --hardware \$HW --run-name hid_r2_ftA --dagger-round 2 \\
    --max-steps $STEPS --ckpt-every 500 --eval-every 500 --grasp-frac 0.3 --photo-aug 1.0 --teacher-nfe -1 --w-sigma 1.0 \\
    --batch-size 2 --grad-accum 4 --num-workers $NW --device cuda || echo "!!! hid_r2_ftA FAILED"
for ck in \$(ls $W/runs/hid/hid_r2_ftA_r2/student_*.pt 2>/dev/null | sort); do ev "\$ck" r2_\$(basename \$ck .pt); done
echo "=== hid_2k_ftA (demos only) \$(date)"
cp $D/manifests/all_base.jsonl $D/manifests/all.jsonl
$PY -m phantom.train.distill_hid --teacher-ckpt "\$T" --data \$D --hardware \$HW --run-name hid_2k_ftA \\
    --max-steps $STEPS --ckpt-every 500 --eval-every 500 --grasp-frac 0.3 --photo-aug 1.0 --teacher-nfe -1 --w-sigma 1.0 \\
    --batch-size 2 --grad-accum 4 --num-workers $NW --device cuda || echo "!!! hid_2k_ftA FAILED"
for ck in \$(ls $W/runs/hid/hid_2k_ftA_r0/student_*.pt 2>/dev/null | sort); do ev "\$ck" 2k_\$(basename \$ck .pt); done
echo "ALL R2 DONE \$(date)"
EOF3
chmod +x "$W/distill_r2.sh"
cat > "$W/ckpt_watch_r2.sh" <<EOF3
#!/bin/bash
: "\${HF_TOKEN:?}"; export HF_HUB_DISABLE_XET=1
while true; do
  for pair in "$W/runs/hid/hid_r2_ftA_r2:hid_r2_ftA" "$W/runs/hid/hid_2k_ftA_r0:hid_2k_ftA"; do
    DIR=\${pair%%:*}; N=\${pair##*:}; [ -d "\$DIR" ] || continue
    $PY $W/phantom/tools/upload_run_ckpts.py "\$DIR" --repo $HUB --run-name \$N --log $W/distill_r2.log 2>&1 | tail -1
  done
  ls $W/eval/*.json >/dev/null 2>&1 && $PY - <<'PY'
import glob, os
from huggingface_hub import HfApi
api = HfApi()
for f in glob.glob("$W/eval/*.json") + glob.glob("$W/eval/*.log"):
    api.upload_file(path_or_fileobj=f, path_in_repo="eval_r2/" + os.path.basename(f), repo_id="$HUB", repo_type="model")
PY
  grep -q "ALL R2 DONE" $W/distill_r2.log 2>/dev/null && { sleep 120; echo "ALL UPLOADS DONE"; exit 0; }
  sleep 600
done
EOF3
chmod +x "$W/ckpt_watch_r2.sh"
echo "READY (r2). Launch on GO:  nohup bash $W/distill_r2.sh > $W/distill_r2.log 2>&1 &   and   HF_TOKEN=hf_xxx nohup bash $W/ckpt_watch_r2.sh > $W/ckpt_watch_r2.log 2>&1 &"
