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
AVAIL=$(df -BG --output=avail "${W%/*}" 2>/dev/null | tail -1 | tr -dc 0-9 || echo 999)
[ "${AVAIL:-999}" -ge 150 ] || echo "WARNING: <150GB free — dataset+staging+ckpts need ~150GB"

mkdir -p "$W" && cd "$W"
PY=$(command -v python3.11 || command -v python3.10 || command -v python3)
echo "== python: $PY ($($PY -V))"

$PY -m venv .venv 2>/dev/null || ($PY -m pip install -q virtualenv && $PY -m virtualenv .venv)
. .venv/bin/activate
pip install -q -U pip huggingface_hub
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

echo "== cosmos repo (public, pinned to the submodule commit)"
[ -d cosmos-predict2.5/.git ] || git clone -q https://github.com/nvidia-cosmos/cosmos-predict2.5.git
git -C cosmos-predict2.5 checkout -q a2c298b0a3df3778b973fe65e9e58877b292d8a7

echo "== python deps"
pip install -q -r requirements/requirements-h100.txt || \
  pip install -q -r requirements/requirements-a100.txt
pip install -q -e .

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
    if os.path.isdir(f"{W}/data/phantom-episodes/tasks/{t}"):
        print("skip (present)", t, flush=True); continue
    for ext in ("tar.zst", "tar"):
        try:
            p = hf_hub_download("armteam/phantom-checkpoints",
                                f"dataset_v3_packed/{t}.{ext}",
                                repo_type="model", local_dir=W + "/dl")
            break
        except Exception:
            p = None
    assert p, f"no tarball for {t}"
    flags = "--zstd" if p.endswith("zst") else ""
    subprocess.run(f"tar {flags} -xf {p} -C {W}/data/phantom-episodes/tasks",
                   shell=True, check=True)
    subprocess.run(["rm", p])
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
import glob, subprocess, sys
from huggingface_hub import HfApi, snapshot_download
W = sys.argv[1]
api = HfApi()
sess = [e.path for e in api.list_repo_tree("armteam/phantom-episodes", path_in_repo="archive",
                                          repo_type="dataset", recursive=False)
        if "20260822_" in e.path]
print(f"{len(sess)} recovery sessions on hub")
snapshot_download("armteam/phantom-episodes", repo_type="dataset",
                  allow_patterns=[p + "/*" for p in sess],
                  local_dir=f"{W}/data/recovery_raw")
for cmd in (["normalize", f"{W}/data/recovery_raw/archive"],
            ["place", f"{W}/data/recovery_raw/archive", f"{W}/data/phantom-episodes/tasks"],
            ["manifest", f"{W}/data/phantom-episodes/tasks",
             f"{W}/data/phantom-episodes/manifests/all.jsonl"]):
    subprocess.run([sys.executable, "tools/intake_recovery.py", *cmd], check=True)
n = len(glob.glob(f"{W}/data/phantom-episodes/tasks/*/ep_*"))
print(f"episodes under tasks/ now: {n} (expect 790 + ~315)")
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
EOF3

echo "== verify: pytest"
python -m pytest tests/ -q 2>&1 | tail -1 || echo "PYTEST FAILED — investigate before launch"

echo "== verify: 2-step REAL training smoke (same flags as the launch line)"
python -m phantom.train.train_teacher \
    --data "$W/data/phantom-episodes/tasks" --hardware configs/hardware.nuc.yaml \
    --allow-config-drift --run-name provision_smoke --max-steps 2 \
    --device cuda --acc-two-pass 2>&1 | tail -3
rm -rf "$W/runs/teacher/provision_smoke"
rm -rf "$W/dl"
EPS=$(find "$W/data/phantom-episodes/tasks" -maxdepth 2 -mindepth 2 -type d -name "ep_*" | wc -l)
echo "episodes on disk: $EPS (expect 790)"

echo
echo "READY. Launch (only on explicit GO):"
echo "  cd $W/phantom && nohup $W/.venv/bin/python -m phantom.train.train_teacher \\"
echo "    --data $W/data/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml \\"
echo "    --allow-config-drift --run-name teacher_v5_recovery --max-steps 3000 \\"
echo "    --init-weights $W/runs/teacher/teacher_v4_790eps/teacher_020000.pt \\"
echo "    --grasp-frac 0.3 --photo-aug 1.0 --acc-two-pass \\"
echo "    --batch-size 4 --grad-accum 2 --num-workers 16 \\"
echo "    --device cuda > train_v5.log 2>&1 &"
echo "  # batch 4x2 needs ~62GB (H100 NVL/80GB); on 40GB use --batch-size 2 --grad-accum 4"
echo "  # 3000 steps ~= 5h on H100 NVL. Offline check: tools/terminal_eval.py + tools/episode_qc.py"
