#!/bin/bash
# v6 overlay for a rented CUDA box. Run AFTER tools/provision_distill.sh in the
# same workspace (it does the repo/venv/cosmos/dataset/pytest work and pulls the
# v5 teachers; those are harmless here). This script:
#   1. pulls the v6 teacher checkpoint from the hub,
#   2. pins the manifests to compute's exact split (all.jsonl 991/124 — the
#      split v6 was trained and will be scored on; all_r2.jsonl = +46 rollouts),
#   3. verifies the v6 payload carries wrench_baseline_rows=8 (the v6 data fix
#      that distill/control/eval inherit from the payload, never from a flag),
#   4. writes distill_v6.sh: student (round-2 recipe: teacher at full NFE +
#      trained sigma readout) -> terminal_eval; no-distill control (same inputs,
#      init from v6) -> terminal_eval; teacher v6 terminal_eval for the same
#      split/seeds; and ckpt_watch_v6.sh for token-inline egress.
# Launches NOTHING.
#
#   HF_TOKEN=hf_xxx V6_CKPT=teacher_v6/teacher_020000.pt bash tools/provision_distill_v6.sh [/workspace/phantom-hid]
#
# Budget (H100 SXM $2.40/h): HID round-2 step at batch 2 x GA 4 ~ 12 s -> 2000
# steps ~ 6.7 h; control teacher step ~ 5 s -> 1200 steps ~ 1.7 h; three
# terminal_evals ~ 0.5 h. ~9 h ~ $22 all in. STEPS/CTRL_STEPS override.
set -euo pipefail
W=${1:-/workspace/phantom-hid}
# hub repos after the 2026-09 armteam restructure:
HUB_TEACHER=armteam/hapticwam-teacher         # teacher_v6/, teacher_v6_simft/
HUB_ABL=armteam/hapticwam-ablations           # teacher_v6_ftA/, ctrl_v6/, eval_v6/, ...
HUB_TELEOP=armteam/hapticwam-teleop-dataset   # DATASET repo, FLAT: manifests_v6.tar
: "${HF_TOKEN:?set HF_TOKEN inline}"
: "${V6_CKPT:?set V6_CKPT=teacher_v6/<file>.pt (hub path of the chosen v6 checkpoint)}"
cd "$W/phantom"
PY=$W/.venv/bin/python

hfget() {  # hfget <repo> <path-in-repo> [dest-dir] [repo-type: model|dataset]
  $PY - "$1" "$2" "${3:-$W/dl}" "${4:-model}" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], repo_type=sys.argv[4], local_dir=sys.argv[3]))
EOF
}

# which repo owns the requested v6 teacher: the shipped v6 runs live in the
# teacher repo, every v6 variant (ftA / video1p0 / simft_multitask) in ablations
case "$V6_CKPT" in
  teacher_v6/*|teacher_v6_simft/*) V6_HUB=$HUB_TEACHER ;;
  *)                               V6_HUB=$HUB_ABL ;;
esac

echo "== v6 teacher: $V6_CKPT"
mkdir -p "$W/runs/teacher/teacher_v6"
hfget "$V6_HUB" "$V6_CKPT" "$W/dl" >/dev/null
cp "$W/dl/$V6_CKPT" "$W/runs/teacher/teacher_v6/"
TEACHER_V6="$W/runs/teacher/teacher_v6/$(basename "$V6_CKPT")"
ls -la "$TEACHER_V6"

echo "== manifests: pin to compute's split"
MAN=$W/data/phantom-episodes/manifests
mkdir -p "$MAN"
if hfget "$HUB_TELEOP" manifests_v6.tar "$W/dl" dataset >/dev/null 2>&1; then
  tar -xf "$W/dl/manifests_v6.tar" -C "$MAN"
  echo "placed manifests_v6.tar (all.jsonl + all_r2.jsonl)"
else
  echo "WARNING: $HUB_TELEOP/manifests_v6.tar not on hub — using the manifest intake_recovery generated; verifying its counts instead"
fi
$PY - "$MAN/all.jsonl" <<'EOF'
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
c = collections.Counter(r.get("split") for r in rows)
print("manifest:", dict(c))
assert c["train"] == 991 and c["val"] == 124, f"split mismatch vs compute (991/124): {dict(c)}"
EOF

echo "== v6 payload: wrench_baseline_rows"
$PY - "$TEACHER_V6" <<'EOF'
import sys, torch
from phantom.train.common import wrench_baseline_rows_of
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
n = wrench_baseline_rows_of(p)
print("wrench_baseline_rows =", n, "| step", p.get("step"))
assert n == 8, "the v6 teacher must carry wrench_baseline_rows=8 — wrong checkpoint?"
EOF

EPS=$(find -L "$W/data/phantom-episodes/tasks" -maxdepth 2 -mindepth 2 -type d -name "ep_*" | wc -l)
echo "episodes on disk: $EPS (expect 1115)"
[ "$EPS" -eq 1115 ] || { echo "FATAL: episode count $EPS != 1115"; exit 1; }

NW=$(( $(nproc) / 2 )); [ "$NW" -gt 16 ] && NW=16; [ "$NW" -lt 2 ] && NW=2
STEPS=${STEPS:-2000}          # student; ckpt+eval every 500 so the best is pickable
CTRL_STEPS=${CTRL_STEPS:-1200}
SEEDS=${SEEDS:-4}
mkdir -p "$W/eval"
cat > "$W/distill_v6.sh" <<EOF
#!/bin/bash
# v6 chain: teacher eval -> student (round-2 recipe) -> control (no distill) -> evals.
# Token-FREE. Launch ONLY on explicit GO:
#   nohup bash $W/distill_v6.sh > $W/distill_v6.log 2>&1 &
set -uo pipefail
cd $W/phantom
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$TEACHER_V6
D=$W/data/phantom-episodes/tasks
HW=configs/hardware.nuc.yaml
ev() {  # ev <ckpt> <tag> [extra terminal_eval flags]
  echo "=== terminal_eval \$2 \$(date)"
  $PY tools/terminal_eval.py --ckpt "\$1" --data \$D --hardware \$HW --nfe 1 --seeds $SEEDS --split val \\
      --out $W/eval/\$2.json \${3:-} > $W/eval/\$2.log 2>&1 || echo "!!! eval \$2 FAILED rc=\$?"
  grep -oE "endpoint_err_mm[^,}]*" $W/eval/\$2.json | head -2
}
ev "\$T" v6_teacher_nfe1
ev "\$T" v6_teacher_nfe5 "--nfe 5"

echo "=== student hid_r1_v6 from \$T  \$(date)"
$PY -m phantom.train.distill_hid \\
    --teacher-ckpt "\$T" --data \$D --hardware \$HW --run-name hid_r1_v6 \\
    --max-steps $STEPS --ckpt-every 500 --eval-every 500 \\
    --grasp-frac 0.3 --photo-aug 1.0 --teacher-nfe -1 --w-sigma 1.0 \\
    --batch-size 2 --grad-accum 4 --num-workers $NW --device cuda \\
    \${RESUME_STUDENT:+--resume \$RESUME_STUDENT} || echo "!!! hid_r1_v6 FAILED rc=\$?"
echo "=== hid_r1_v6 finished \$(date)"
for ck in \$(ls $W/runs/hid/hid_r1_v6_r0/student_*.pt 2>/dev/null | sort); do
  ev "\$ck" v6_student_\$(basename \$ck .pt)
done

echo "=== control ctrl_v6 (student inputs, init from v6, NO distillation) \$(date)"
$PY -m phantom.train.train_teacher \\
    --data \$D --hardware \$HW --allow-config-drift --student --init-weights "\$T" \\
    --run-name ctrl_v6 --max-steps $CTRL_STEPS --split train \\
    --grasp-frac 0.3 --photo-aug 1.0 --acc-two-pass \\
    --contact-nll-beta 0.5 --action-noise-per-strip --no-action-t-max-of-two \\
    --event-band-weight 0 --cond-dropout 0 \\
    --lr 1e-4 --lr-new-modules 3e-4 --warmup-steps 500 --ema-decay 0.999 \\
    --ckpt-every $CTRL_STEPS --eval-every $CTRL_STEPS \\
    --batch-size 2 --grad-accum 4 --num-workers $NW --device cuda || echo "!!! ctrl_v6 FAILED rc=\$?"
ck=\$(ls $W/runs/teacher/ctrl_v6/teacher_*.pt 2>/dev/null | sort | tail -1)
[ -n "\$ck" ] && ev "\$ck" v6_control
echo "ALL V6 DONE \$(date)"
EOF
chmod +x "$W/distill_v6.sh"

cat > "$W/ckpt_watch_v6.sh" <<EOF
#!/bin/bash
# Every 10 min push student/control checkpoints + logs + evals to the hub.
#   HF_TOKEN=hf_xxx nohup bash $W/ckpt_watch_v6.sh > $W/ckpt_watch_v6.log 2>&1 &
: "\${HF_TOKEN:?set HF_TOKEN inline for the watcher}"
export HF_HUB_DISABLE_XET=1
while true; do
  # dir:run-name:repo — hid_r1_v6 is a superseded round that was not copied forward; a NEW one
  # egresses to the ablations repo, never to the retiring one. ctrl_v6 is an ablation.
  for triple in "$W/runs/hid/hid_r1_v6_r0:hid_r1_v6:$HUB_ABL" "$W/runs/teacher/ctrl_v6:ctrl_v6:$HUB_ABL"; do
    D=\${triple%%:*}; R=\${triple##*:}; N=\${triple%:*}; N=\${N##*:}
    [ -d "\$D" ] || continue
    $PY $W/phantom/tools/upload_run_ckpts.py "\$D" --repo \$R --run-name \$N --log $W/distill_v6.log 2>&1 | tail -1
  done
  ls $W/eval/*.json >/dev/null 2>&1 && $PY - <<'PY'
import glob, os
from huggingface_hub import HfApi
api = HfApi()
for f in glob.glob("$W/eval/*.json") + glob.glob("$W/eval/*.log"):
    api.upload_file(path_or_fileobj=f, path_in_repo="eval_v6/" + os.path.basename(f), repo_id="$HUB_ABL", repo_type="model")
PY
  grep -q "ALL V6 DONE" $W/distill_v6.log 2>/dev/null && { sleep 120; echo "ALL UPLOADS DONE"; exit 0; }
  sleep 600
done
EOF
chmod +x "$W/ckpt_watch_v6.sh"

echo
echo "READY (v6). Runner: $W/distill_v6.sh — student $STEPS steps (ckpt/eval every 500), control $CTRL_STEPS, evals x$SEEDS seeds."
echo "Launch (only on explicit GO):"
echo "  nohup bash $W/distill_v6.sh > $W/distill_v6.log 2>&1 &"
echo "  HF_TOKEN=hf_xxx nohup bash $W/ckpt_watch_v6.sh > $W/ckpt_watch_v6.log 2>&1 &"
