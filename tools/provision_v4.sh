#!/bin/bash
# Zero-to-training provision for a rented CUDA box (vast.ai H100/A100/5090).
#
#   HF_TOKEN=hf_xxx bash provision_v4.sh [/workspace/phantom-v4]
#
# Needs ONLY the HF token (read access to armteam/hapticwam-teleop-dataset,
# armteam/hapticwam-teacher + the
# license-accepted nvidia/Cosmos-Predict2.5-2B). The CODE is cloned from the
# PUBLIC GitHub repo Advanced-Robotic-Manipulation/HapticWAM, so no GitHub
# credentials touch the rented machine either; the python package inside that
# repo is still named `phantom`. ~15 min on a datacenter pipe, most of it the
# 68G dataset pull.
#
# Produces: repo + venv + cosmos code&weights + dataset + paths.local.yaml,
# runs the test suite, prints the v4 launch command. Training is NOT started.
set -euo pipefail

W=${1:-/workspace/phantom-v4}
# hub repos after the 2026-09 armteam restructure (one repo per artifact kind):
HUB_TELEOP=armteam/hapticwam-teleop-dataset   # DATASET repo: packed task tarballs,
                                              # manifests*.tar, norm_stats.json, batch_*.tar.zst — FLAT (no dataset_v3_packed/ prefix)
HUB_TEACHER=armteam/hapticwam-teacher         # text_embeddings.pt: the Cosmos text-embedding
                                              # cache for the task prompts, a training INPUT rather
                                              # than a checkpoint, kept beside the teacher it feeds
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

hfget() {  # hfget <repo> <path-in-repo> [dest-dir] [repo-type: model|dataset]
  python - "$1" "$2" "${3:-.}" "${4:-model}" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
p = hf_hub_download(sys.argv[1], sys.argv[2], repo_type=sys.argv[4],
                    local_dir=sys.argv[3])
print(p)
EOF
}

echo "== repo (public GitHub clone)"
mkdir -p "$W/dl"
REPO_URL=${REPO_URL:-https://github.com/Advanced-Robotic-Manipulation/HapticWAM.git}
REPO_REF=${REPO_REF:-}                        # optional branch, tag or FULL commit sha
clone_repo() {   # clone_repo <url> <ref-or-empty> <dir>
  if [ -d "$3/.git" ]; then return 0; fi
  if [ -z "$2" ]; then
    git clone -q --depth 1 "$1" "$3"
    return 0
  fi
  # a branch or tag clones directly; a raw commit sha needs a shallow fetch
  if git clone -q --depth 1 --branch "$2" "$1" "$3" 2>/dev/null; then
    return 0
  fi
  git init -q "$3"
  git -C "$3" remote add origin "$1" 2>/dev/null || git -C "$3" remote set-url origin "$1"
  git -C "$3" fetch -q --depth 1 origin "$2"
  git -C "$3" checkout -q FETCH_HEAD
}
clone_repo "$REPO_URL" "$REPO_REF" phantom
cd phantom
echo "repo commit $(git rev-parse HEAD) from $REPO_URL"

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
python - "$W" "$HUB_TELEOP" "$HUB_TEACHER" <<'EOF'
import subprocess, sys
from huggingface_hub import hf_hub_download
W, HUB_TELEOP, HUB_TEACHER = sys.argv[1], sys.argv[2], sys.argv[3]
tasks = ["Carton", "Carton_fail", "waffles", "waffles_fail",
         "egg", "egg_fail", "whiteboard", "whiteboard_fail"]
import os
for t in tasks:
    if os.path.isdir(f"{W}/data/phantom-episodes/tasks/{t}"):
        print("skip (present)", t, flush=True); continue
    for ext in ("tar.zst", "tar"):
        try:
            p = hf_hub_download(HUB_TELEOP, f"{t}.{ext}",
                                repo_type="dataset", local_dir=W + "/dl")
            break
        except Exception:
            p = None
    assert p, f"no tarball for {t}"
    flags = "--zstd" if p.endswith("zst") else ""
    subprocess.run(f"tar {flags} -xf {p} -C {W}/data/phantom-episodes/tasks",
                   shell=True, check=True)
    subprocess.run(["rm", p])
    print("unpacked", t, flush=True)
# (file, dest, repo, repo_type, path-in-repo). text_embeddings.pt is the Cosmos
# text-embedding cache and lives at the ROOT of the teacher repo.
for extra, dest, repo, rtype, inrepo in (
        ("manifests.tar", W + "/data/phantom-episodes", HUB_TELEOP, "dataset", "manifests.tar"),
        ("norm_stats.json", W + "/data/phantom-episodes/tasks", HUB_TELEOP, "dataset", "norm_stats.json"),
        ("text_embeddings.pt", W + "/data/phantom-episodes/tasks", HUB_TEACHER, "model", "text_embeddings.pt")):
    p = hf_hub_download(repo, inrepo, repo_type=rtype, local_dir=W + "/dl")
    if extra.endswith(".tar"):
        subprocess.run(["tar", "-xf", p, "-C", dest], check=True)
    else:
        subprocess.run(["cp", p, dest], check=True)
    print("placed", extra, flush=True)
EOF

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
echo "    --allow-config-drift --run-name teacher_v4_790eps --max-steps 12000 \\"
echo "    --device cuda --acc-two-pass > train_v4.log 2>&1 &"
echo "  # v4 defaults active: time_true rope, cond-dropout 0.1, action-t-max2, fp32-master"
echo "  # 12000 steps ~= 35h/~\$30 on A100 (v3 val was flat past 9k); 20000 ~= 58h/~\$55"
echo "  # watch: grep EVAL-SAMPLED train_v4.log   (mag_ratio -> 1.0, dir_cosine up)"
