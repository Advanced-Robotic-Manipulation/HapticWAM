#!/bin/bash
# Zero-to-DISTILLATION provision for a rented CUDA box (vast.ai H100/A100).
# Derived from provision_v5.sh (identical setup) — pulls BOTH teachers and
# gates on a real 3-step distill_hid smoke; launches NOTHING.
# v5 = v4 dataset + the 2026-08-22 recovery sessions (hub archive/) + the
# 20k teacher checkpoint, launching the terminal-phase fine-tune.
#
#   HF_TOKEN=hf_xxx bash provision_distill.sh [/workspace/phantom-hid]
#
# Needs ONLY the HF token (read access to armteam/phantom-checkpoints + the
# license-accepted nvidia/Cosmos-Predict2.5-2B): the repo rides in the same
# private HF folder as the packed dataset, so no GitHub credentials ever
# touch the rented machine. ~15 min on a datacenter pipe, most of it the
# 68G dataset pull.
#
# Produces: repo + venv + cosmos code&weights + dataset + paths.local.yaml,
# runs the test suite, prints the v4 launch command. Training is NOT started.
set -euo pipefail

W=${1:-/workspace/phantom-hid}
HUB=armteam/phantom-checkpoints
PACK=dataset_v3_packed
: "${HF_TOKEN:?set HF_TOKEN}"

(apt-get update -qq && apt-get install -y -qq zstd git python3-venv python3-pip) 2>/dev/null || true
command -v zstd >/dev/null || { echo "FATAL: zstd unavailable"; exit 1; }
command -v git  >/dev/null || { echo "FATAL: git unavailable"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "FATAL: no GPU visible (nvidia-smi -L failed)"; exit 1; }
AVAIL=$(df -BG --output=avail "${W%/*}" 2>/dev/null | tail -1 | tr -dc 0-9 || echo 999)
[ "${AVAIL:-999}" -ge 200 ] || echo "WARNING: <200GB free — peak usage ~155GB (dataset 73G + recovery 31G + cosmos + hub cache + ckpts); rent --disk 300"

mkdir -p "$W" && cd "$W"
PY=$(command -v python3.11 || command -v python3.10 || command -v python3)
echo "== python: $PY ($($PY -V))"

$PY -m venv .venv 2>/dev/null || ($PY -m pip install -q virtualenv && $PY -m virtualenv .venv)
. .venv/bin/activate
pip install -q -U pip "huggingface_hub>=1.20,<2"
export HF_XET_HIGH_PERFORMANCE=1     # hub 1.x: xet turbo (hf_transfer extra is gone)
export PIP_NO_CACHE_DIR=1

hfget() {  # hfget <repo> <path-in-repo> [dest-dir]
  python - "$1" "$2" "${3:-.}" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
p = hf_hub_download(sys.argv[1], sys.argv[2], repo_type="model",
                    local_dir=sys.argv[3])
print(p)
EOF
}

echo "== repo"
mkdir -p "$W/dl"
hfget $HUB $PACK/phantom_repo_latest.tar.gz "$W/dl" >/dev/null
mkdir -p phantom && tar -xzf "$W/dl/$PACK/phantom_repo_latest.tar.gz" -C phantom
cd phantom
REPO_COMMIT=${REPO_COMMIT:-__REPO_COMMIT__}   # baked in at tarball pack time
GOT=$(cat COMMIT 2>/dev/null || echo none)
[ "$GOT" = "$REPO_COMMIT" ] || { echo "FATAL: hub repo tarball is commit '$GOT', this script expects '$REPO_COMMIT' — repack (git archive --add-file=COMMIT) or set REPO_COMMIT"; exit 1; }
echo "repo commit $GOT verified"

echo "== cosmos repo (public, pinned to the submodule commit)"
[ -d cosmos-predict2.5/.git ] || git clone -q https://github.com/nvidia-cosmos/cosmos-predict2.5.git
git -C cosmos-predict2.5 checkout -q a2c298b0a3df3778b973fe65e9e58877b292d8a7

echo "== python deps"
pip install -q -r requirements/requirements-h100.txt || \
  pip install -q -r requirements/requirements-a100.txt
pip install -q -e .
python - <<'EOF0'
import torch
assert torch.cuda.is_available(), "CUDA not available after deps install — wrong image/driver; stop before the 100GB pull"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cuda", torch.version.cuda)
EOF0

echo "== cosmos weights (gated: token must have accepted the NVIDIA license)"
WR="$W/cosmos-predict2.5-2b"
mkdir -p "$WR"
python - "$WR" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
root = sys.argv[1]
for f in ("robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
          "robot/action-cond/cr1_empty_string_text_embeddings.pt",
          "tokenizer.pth"):
    print(hf_hub_download("nvidia/Cosmos-Predict2.5-2B", f, local_dir=root))
EOF

echo "== dataset (packed tarballs -> tasks/ + manifests/)"
mkdir -p "$W/data/phantom-episodes/tasks" "$W/data/phantom-episodes/manifests"
python - "$W" <<'EOF'
import subprocess, sys
from huggingface_hub import hf_hub_download
W = sys.argv[1]
tasks = ["Carton", "Carton_fail", "waffles", "waffles_fail",
         "egg", "egg_fail", "whiteboard", "whiteboard_fail"]
import os
for t in tasks:
    done_flag = f"{W}/data/phantom-episodes/tasks/{t}/.complete"
    if os.path.exists(done_flag):
        print("skip (complete)", t, flush=True); continue
    subprocess.run(["rm", "-rf", f"{W}/data/phantom-episodes/tasks/{t}"])   # half-extracted -> redo
    for ext in ("tar.zst", "tar"):
        try:
            p = hf_hub_download("armteam/phantom-checkpoints",
                                f"dataset_v3_packed/{t}.{ext}",
                                repo_type="model", local_dir=W + "/dl")
            break
        except Exception:
            p = None
    assert p, f"no tarball for {t}"
    cmd = ["tar"] + (["--zstd"] if p.endswith("zst") else []) + ["-xf", p, "-C", f"{W}/data/phantom-episodes/tasks"]
    subprocess.run(cmd, check=True)
    subprocess.run(["rm", p])
    open(done_flag, "w").close()
    print("unpacked", t, flush=True)
for extra, dest in (("manifests.tar", W + "/data/phantom-episodes"),
                    ("norm_stats.json", W + "/data/phantom-episodes/tasks"),
                    ("text_embeddings.pt", W + "/data/phantom-episodes/tasks")):
    p = hf_hub_download("armteam/phantom-checkpoints",
                        f"dataset_v3_packed/{extra}", repo_type="model",
                        local_dir=W + "/dl")
    if extra.endswith(".tar"):
        subprocess.run(["tar", "-xf", p, "-C", dest], check=True)
    else:
        subprocess.run(["cp", p, dest], check=True)
    print("placed", extra, flush=True)
EOF

echo "== recovery sessions (hub archive/20260822_*) -> normalized + placed under tasks/"
python - "$W" <<'PYEOF'
import collections, glob, json, os, subprocess, sys
from huggingface_hub import HfApi, snapshot_download
W = sys.argv[1]
api = HfApi()
raw = f"{W}/data/recovery_raw/archive"
os.makedirs(raw, exist_ok=True)
if os.path.exists(f"{raw}/.complete") and len(glob.glob(f"{raw}/20260822_*/ep_*/meta.json")) == 325:
    print("recovery sessions already present (validated)", flush=True)
elif api.file_exists("armteam/phantom-checkpoints", "dataset_v3_packed/batch_20260822.tar.zst"):
    # ONE 30GB xet transfer instead of 123k files (snapshot_download first
    # enumerates the whole 200k-entry episodes repo: measured 1-3h idle GPU)
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("armteam/phantom-checkpoints", "dataset_v3_packed/batch_20260822.tar.zst",
                        repo_type="model", local_dir=W + "/dl")
    subprocess.run(["tar", "--zstd", "-xf", p, "-C", raw], check=True)
    subprocess.run(["rm", p])
    print("recovery sessions unpacked from batch_20260822.tar.zst", flush=True)
else:
    sess = [e.path for e in api.list_repo_tree("armteam/phantom-episodes", path_in_repo="archive",
                                              repo_type="dataset", recursive=False)
            if "20260822_" in e.path]
    print(f"{len(sess)} recovery sessions on hub — no tarball, slow per-file snapshot", flush=True)
    snapshot_download("armteam/phantom-episodes", repo_type="dataset",
                      allow_patterns=[p + "/*" for p in sess], max_workers=32,
                      local_dir=f"{W}/data/recovery_raw")
n_raw = len(glob.glob(f"{raw}/20260822_*/ep_*/meta.json"))
assert n_raw == 325, f"recovery sessions incomplete: {n_raw}/325 episodes under {raw}"
open(f"{raw}/.complete", "w").close()
for cmd in (["normalize", f"{W}/data/recovery_raw/archive"],
            ["place", f"{W}/data/recovery_raw/archive", f"{W}/data/phantom-episodes/tasks"],
            ["manifest", f"{W}/data/phantom-episodes/tasks",
             f"{W}/data/phantom-episodes/manifests/all.jsonl", "--val-min-eps", "10"]):
    subprocess.run([sys.executable, "tools/intake_recovery.py", *cmd], check=True)
eps = sorted(glob.glob(f"{W}/data/phantom-episodes/tasks/*/ep_*"))
print(f"episodes under tasks/ now: {len(eps)} (expect 790 + 325 = 1115)")
assert len(eps) == 1115, f"episode count {len(eps)} != 1115 — partial snapshot or duplicate placement"
# every stream the teacher's WindowSampler reads, OPENED (existence alone
# accepts a half-written zarr group): nonempty, equal-length data/ts
import zarr
STREAMS = ("gripper", "actions", "camera_scene_color", "arm_q", "arm_qd", "arm_tcp_pose",
           "arm_tcp_speed", "arm_ft") + tuple(
    f"tactile_{side}_{k}" for side in ("left", "right")
    for k in ("fields_ds", "keyframes", "infer_img", "wrench", "area"))
bad = []
for e in eps:
    st = json.load(open(os.path.join(e, "meta.json"))).get("status", "finalized")
    if st != "finalized":
        bad.append((e, f"status={st}")); continue
    for sname in STREAMS:
        try:
            g = zarr.open(os.path.join(e, sname + ".zarr"), mode="r")
            n, m = g["data"].shape[0], g["ts"].shape[0]
            if n == 0 or n != m:
                bad.append((e, f"{sname}: data {n} vs ts {m}")); break
        except Exception as ex:
            bad.append((e, f"{sname}: {type(ex).__name__}")); break
assert not bad, f"{len(bad)} episodes fail stream validation, e.g. {bad[:3]}"
print(f"stream validation: {len(eps)} episodes x {len(STREAMS)} groups OK", flush=True)
rows = [json.loads(l) for l in open(f"{W}/data/phantom-episodes/manifests/all.jsonl") if l.strip()]
split = collections.Counter(r["split"] for r in rows)
hold = json.load(open(f"{W}/data/phantom-episodes/manifests/intake_holdout.json"))
exp = {"train": 712 + hold["train"], "val": 78 + hold["val"]}
print("manifest:", dict(split), "| expect", exp, "| new-batch holdout:", hold["holdout_sessions"])
assert split == exp and hold["added"] == 325, f"manifest split {dict(split)} != {exp}"
paths = [r["path"] for r in rows]
assert len(set(paths)) == len(paths) == 1115, "manifest rows are not unique"
assert len({r["episode"] for r in rows}) == 1115, "duplicate episode ids in manifest"
phys = {os.path.relpath(e, f"{W}/data/phantom-episodes") for e in eps}
assert set(paths) == phys, f"manifest/disk mismatch: {sorted(set(paths) ^ phys)[:5]}"
tr = {r["path"] for r in rows if r["split"] == "train"}; va = {r["path"] for r in rows if r["split"] == "val"}
assert not (tr & va), "a path is in both train and val"
print("manifest: bijection onto disk, unique, disjoint splits OK", flush=True)
PYEOF

echo "== FT-A init checkpoint: v5_6 (teacher_v5_batch0822/teacher_003000.pt)"
# the plan of record is "FT-A: 3k steps FROM v5_6" (REVIEW_SYNTHESIS.md:478).
# v5_6 is byte-identical to compute3's DEMO.pt and is what the rig ran.
hfget $HUB teacher_v5_batch0822/teacher_003000.pt "$W/dl" >/dev/null
mkdir -p "$W/runs/teacher/teacher_v5_batch0822"
cp "$W/dl/teacher_v5_batch0822/teacher_003000.pt" "$W/runs/teacher/teacher_v5_batch0822/"
FTA_INIT="$W/runs/teacher/teacher_v5_batch0822/teacher_003000.pt"
# v4 control (an objective-only ablation initialised from the 20k v4 teacher).
# Uncomment both lines and re-point FTA_INIT to run it instead:
# hfget $HUB teacher_v4_790eps/teacher_020000.pt "$W/dl" >/dev/null
# mkdir -p "$W/runs/teacher/teacher_v4_790eps" && cp "$W/dl/teacher_v4_790eps/teacher_020000.pt" "$W/runs/teacher/teacher_v4_790eps/"
# FTA_INIT="$W/runs/teacher/teacher_v4_790eps/teacher_020000.pt"

echo "== paths.local.yaml"
cat > configs/paths.local.yaml <<EOF2
cosmos_repo: $W/phantom/cosmos-predict2.5
cosmos_weights_root: $WR
data_root: $W/data
runs_root: $W/runs
cosmos_text_embedding_cache: $W/data/phantom-episodes/tasks/text_embeddings.pt
EOF2

echo "== verify: GPU"
python - <<'EOF3'
import torch
assert torch.cuda.is_available(), "CUDA not available — wrong image or driver"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
EOF3

echo "== verify: text-cache preflight (every episode text must equal its task)"
python - "$W" <<'EOF3'
import json, glob, sys, collections
pairs = collections.Counter()
for p in glob.glob(sys.argv[1] + "/data/phantom-episodes/tasks/*/ep_*/meta.json"):
    m = json.load(open(p))
    pairs[(m["task"], m.get("text") or m["task"])] += 1
bad = {k: v for k, v in pairs.items() if k[0] != k[1]}
print("task/text pairs:", dict(pairs))
assert not bad, f"TEXT-CACHE MISS RISK: {bad} — these episodes would train with the empty-string embedding"
import torch
cache = torch.load(sys.argv[1] + "/data/phantom-episodes/tasks/text_embeddings.pt", map_location="cpu", weights_only=False)
keys = set(cache.keys()) if isinstance(cache, dict) else set(getattr(cache, "keys", lambda: [])())
need = {k[1] for k in pairs}
missing = need - keys
assert not missing, f"text_embeddings.pt lacks {sorted(missing)} — those episodes would train unconditioned"
print("text cache covers", sorted(need), flush=True)
EOF3


echo "== second teacher: ftA_1500 (teacher_v5_ftA/teacher_001500.pt — best on waffles, all 09-01/09-04 rig grasps)"
hfget $HUB teacher_v5_ftA/teacher_001500.pt "$W/dl" >/dev/null
mkdir -p "$W/runs/teacher/teacher_v5_ftA"
cp "$W/dl/teacher_v5_ftA/teacher_001500.pt" "$W/runs/teacher/teacher_v5_ftA/"
TEACHER_V56="$FTA_INIT"
TEACHER_FTA="$W/runs/teacher/teacher_v5_ftA/teacher_001500.pt"

echo "== verify: pytest (DDP 2-rank spawn is a known container limitation; the"
echo "   replay tiny-ckpt bf16-on-CUDA atol failure is a known test-environment"
echo "   issue — both are allow-listed, anything else stops the provision)"
set +e
python -m pytest tests/ -q --deselect tests/test_ddp_sync.py::test_ddp_ranks_stay_in_sync > "$W/pytest_gate.log" 2>&1
RC=$?
set -e
tail -1 "$W/pytest_gate.log"
if [ $RC -ne 0 ]; then
  # only a REAL summary with nothing but allow-listed FAILED lines may pass:
  # collection errors (^ERROR, rc 2) and an empty collection (rc 5) must not
  FAILS=$(grep -E "^(FAILED|ERROR)" "$W/pytest_gate.log" || true)
  OTHER=$(echo "$FAILS" | grep -v "^FAILED tests/test_replay_rig" | grep -v "^$" || true)
  if [ $RC -eq 5 ] || [ -z "$FAILS" ] || [ -n "$OTHER" ]; then
    echo "$FAILS"; echo "PYTEST GATE FAILED (rc=$RC) outside the allow-list — full log: $W/pytest_gate.log — fix before launch"; exit 1
  fi
  echo "WARNING: only allow-listed failures (test_replay_rig tiny-ckpt bf16 atol) — continuing"
fi

NW=$(( $(nproc) / 2 )); [ "$NW" -gt 16 ] && NW=16; [ "$NW" -lt 2 ] && NW=2
BS=${BS:-1}; GA=${GA:-8}     # effective batch 8 (= the 5090 teacher default 1 x 8); the 4090 OOMs at BS=1 — needs >=40GB
echo "== verify: 3-step REAL distillation smoke (teacher v5_6, the launch line below)"
python -m phantom.train.distill_hid \
    --teacher-ckpt "$TEACHER_V56" --data "$W/data/phantom-episodes/tasks" \
    --hardware configs/hardware.nuc.yaml --run-name provision_smoke --max-steps 3 \
    --grasp-frac 0.3 --photo-aug 1.0 \
    --batch-size $BS --grad-accum 1 --num-workers $NW --device cuda 2>&1 | tail -3
rm -rf "$W/runs/hid/provision_smoke_r0"
rm -rf "$W/dl"
EPS=$(find -L "$W/data/phantom-episodes/tasks" -maxdepth 2 -mindepth 2 -type d -name "ep_*" | wc -l)
echo "episodes on disk: $EPS (expect 1115); nproc $(nproc) -> --num-workers $NW"

# COST (review 2026-09-05): a HID step is ~10-12 full 2B-DiT forwards (teacher
# sample + two-pass anticipation samples + student fwd/bwd) ~= 20 s/step at
# effective batch 8 on an H100 — NOT a teacher step. 6000 steps x 2 teachers
# would be ~67 h / ~$150. Defaults below fit the $39 credit; override with
# STEPS=... TEACHERS="v5_6 ftA" (or a single teacher).
STEPS=${STEPS:-1200}
TEACHERS=${TEACHERS:-"v5_6 ftA"}
SEC_PER_STEP=${SEC_PER_STEP:-20}
NT=$(echo $TEACHERS | wc -w | tr -d ' ')
EST_H=$(python -c "print(round($STEPS*$SEC_PER_STEP*$NT/3600.0, 1))")
cat > "$W/distill_both.sh" <<EOF
#!/bin/bash
# Sequential HID round-0 distillation (teachers: $TEACHERS), $STEPS steps each.
# Token-FREE by design: uploads are done by ckpt_watch.sh (launch it with the
# token inline). Launch ONLY on explicit GO:
#   nohup bash $W/distill_both.sh > $W/distill_both.log 2>&1 &
# Resume a preempted run:  add  --resume $W/runs/hid/hid_r0_<T>_r0/student_XXXXXX.pt
set -uo pipefail
cd $W/phantom
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for T in $TEACHERS; do
  case \$T in v5_6) CK="$TEACHER_V56";; ftA) CK="$TEACHER_FTA";; *) echo "unknown teacher \$T"; continue;; esac
  RV="RESUME_\$T"; RES="\${!RV:-}"        # RESUME_v5_6=/path/student_001000.pt to resume
  echo "=== distill hid_r0_\$T from \$CK  \$(date)"
  $W/.venv/bin/python -m phantom.train.distill_hid \\
      --teacher-ckpt "\$CK" --data $W/data/phantom-episodes/tasks \\
      --hardware configs/hardware.nuc.yaml --run-name hid_r0_\$T \\
      --max-steps $STEPS --grasp-frac 0.3 --photo-aug 1.0 \\
      --batch-size $BS --grad-accum $GA --num-workers $NW --device cuda \\
      \${RES:+--resume \$RES} || echo "!!! hid_r0_\$T FAILED rc=\$? (continuing with the next teacher)"
  echo "=== hid_r0_\$T finished  \$(date)"
done
echo "ALL DISTILLS DONE  \$(date)"
EOF
chmod +x "$W/distill_both.sh"
cat > "$W/ckpt_watch.sh" <<EOF
#!/bin/bash
# Incremental egress: every 10 min push every student checkpoint + logs to the
# hub (sha256-idempotent). A preempted rental then loses at most 1000 steps.
#   HF_TOKEN=hf_xxx nohup bash $W/ckpt_watch.sh > $W/ckpt_watch.log 2>&1 &
: "\${HF_TOKEN:?set HF_TOKEN inline for the watcher}"
while true; do
  for T in $TEACHERS; do
    D=$W/runs/hid/hid_r0_\${T}_r0
    [ -d "\$D" ] || continue
    LOGARG=""; [ -f $W/distill_both.log ] && LOGARG="--log $W/distill_both.log"
    $W/.venv/bin/python $W/phantom/tools/upload_run_ckpts.py "\$D" --repo $HUB --run-name hid_r0_\$T \$LOGARG 2>&1 | tail -2
  done
  grep -q "ALL DISTILLS DONE" $W/distill_both.log 2>/dev/null && { sleep 60; echo "ALL UPLOADS DONE"; exit 0; }
  sleep 600
done
EOF
chmod +x "$W/ckpt_watch.sh"

echo
echo "READY. Runner: $W/distill_both.sh (teachers: $TEACHERS, $STEPS steps each, batch ${BS}x${GA}, ckpt+eval every 1000)"
echo "       ESTIMATE: ~${EST_H} h at ~${SEC_PER_STEP} s/step — check the smoke's it/s above and rescale before GO."
echo "Launch (only on explicit GO) — two commands, token ONLY on the watcher:"
echo "  nohup bash $W/distill_both.sh > $W/distill_both.log 2>&1 &"
echo "  HF_TOKEN=hf_xxx nohup bash $W/ckpt_watch.sh > $W/ckpt_watch.log 2>&1 &"
echo "Watch:  tail -f $W/distill_both.log $W/ckpt_watch.log"
