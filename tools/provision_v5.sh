#!/bin/bash
# Zero-to-FINE-TUNE provision for a rented CUDA box (vast.ai H100/A100/5090).
# v5 = v4 dataset + the 2026-08-22 recovery sessions (hub archive/) + the
# 20k teacher checkpoint, launching the terminal-phase fine-tune.
#
#   HF_TOKEN=hf_xxx bash provision_v5.sh [/workspace/phantom-v5]
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

W=${1:-/workspace/phantom-v5}
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

echo "== 20k teacher checkpoint (fine-tune init)"
hfget $HUB teacher_v4_790eps/teacher_020000.pt "$W/dl" >/dev/null
mkdir -p "$W/runs/teacher/teacher_v4_790eps"
cp "$W/dl/teacher_v4_790eps/teacher_020000.pt" "$W/runs/teacher/teacher_v4_790eps/"

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

echo "== verify: pytest"
python -m pytest tests/ -q 2>&1 | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "PYTEST FAILED — fix before launch"; exit 1; }

NW=$(( $(nproc) / 2 )); [ "$NW" -gt 16 ] && NW=16; [ "$NW" -lt 2 ] && NW=2
BS=${BS:-4}; GA=${GA:-2}     # effective batch 8; export BS=1 GA=8 on a 24-32GB card, BS=2 GA=4 on 40GB
echo "== verify: 2-step REAL training smoke (same flags as the launch line, incl. the"
echo "   --init-weights / --grasp-frac guards that would otherwise first run at paid launch)"
python -m phantom.train.train_teacher \
    --data "$W/data/phantom-episodes/tasks" --hardware configs/hardware.nuc.yaml \
    --allow-config-drift --run-name provision_smoke --max-steps 2 \
    --init-weights "$W/runs/teacher/teacher_v4_790eps/teacher_020000.pt" \
    --grasp-frac 0.3 --photo-aug 1.0 --acc-two-pass \
    --lr 2e-5 --lr-new-modules 6e-5 --warmup-steps 150 \
    --batch-size $BS --grad-accum $GA --num-workers $NW \
    --device cuda 2>&1 | tail -3
rm -rf "$W/runs/teacher/provision_smoke"
rm -rf "$W/dl"
EPS=$(find -L "$W/data/phantom-episodes/tasks" -maxdepth 2 -mindepth 2 -type d -name "ep_*" | wc -l)
echo "episodes on disk: $EPS (expect 1115 = 790 v4 + 325 batch_20260822; symlinks counted)"
echo "nproc: $(nproc) -> --num-workers $NW; smoke ran batch ${BS}x${GA}"

echo
echo "READY. Launch (only on explicit GO):"
echo "  unset HF_TOKEN   # training does not need it; keep it out of the process environment"
echo "  cd $W/phantom && nohup $W/.venv/bin/python -m phantom.train.train_teacher \\"
echo "    --data $W/data/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml \\"
echo "    --allow-config-drift --run-name teacher_v5_batch0822 --max-steps 3000 \\"
echo "    --init-weights $W/runs/teacher/teacher_v4_790eps/teacher_020000.pt \\"
echo "    --grasp-frac 0.3 --photo-aug 1.0 --acc-two-pass \\"
echo "    --lr 2e-5 --lr-new-modules 6e-5 --warmup-steps 150 --ckpt-every 500 --eval-every 500 \\"
echo "    --batch-size $BS --grad-accum $GA --num-workers $NW \\"
echo "    --device cuda > train_v5.log 2>&1 &"
echo "  # fine-tune LR = 1/5 of the from-scratch peak (audit 2026-08-26: full peak = 2.7x LoRA-B"
echo "  #   weight-scale displacement budget); --init-ema (default) starts from the deployed EMA weights;"
echo "  #   ckpt/eval every 500 -> SELECT the best checkpoint with tools/terminal_eval.py, do not ship step 3000"
echo "  #   val = frozen v4 78 eps + the last whole session(s) per task of batch_20260822 (>=10 eps/task),"
echo "  #   see manifests/intake_holdout.json — in-run val_* mixes both; terminal_eval per task for the split"
echo "  # effective batch 8 everywhere: 4x2 needs ~62GB (H100 NVL/80GB, measured 61.5GB);"
echo "  # 40GB -> --batch-size 2 --grad-accum 4; 5090 32GB / 4090 24GB -> --batch-size 1 --grad-accum 8"
echo "  # (4090 measured 19.9GiB at 1x8 with --acc-two-pass; batch 2 on a 5090 is unmeasured)"
echo "  # 3000 steps ~= 5h on H100 NVL. Offline check: tools/terminal_eval.py + tools/episode_qc.py"
