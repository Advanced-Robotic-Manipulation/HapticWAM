#!/bin/bash
# continuation of pi05_setup2.sh after the phantom-adapter import failed on missing deps (zarr, ...):
# expose the phantom venv's site-packages at LOWER priority than pi05venv's own (same python 3.11,
# same torch 2.13.0+cu130), then the checkpoint download + processor patch + digest.
#   HF_TOKEN=hf_xxx bash pi05_setup3.sh
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN inline}"
B=$HOME/phantom-icra-2027; P=$B/phantom; V=$B/pi05venv
SP=$("$V/bin/python" -c "import site; print(site.getsitepackages()[0])")
PSP=$("$P/.venv/bin/python" -c "import site; print(site.getsitepackages()[0])")
echo "$P" > "$SP/zz_phantom_repo.pth"
echo "$PSP" > "$SP/zzz_phantom_venv_fallback.pth"
"$V/bin/python" - <<'PY'
import importlib, torch, transformers, lerobot, numpy
print("torch", torch.__version__, torch.__file__.split("/site-packages/")[0].split("/")[-3])
print("transformers", transformers.__version__, "lerobot", lerobot.__version__, "numpy", numpy.__version__)
import phantom.inference.lerobot_policy, phantom.scripts.lerobot_server
print("phantom adapter importable")
from transformers.models.siglip import check; print("replace ok:", check.check_whether_transformers_replace_is_installed_correctly())
PY
mkdir -p "$P/runs/pi05_20k"
cd "$P"
HF_HUB_DISABLE_XET=1 "$V/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
# pi05_phantom_expert_v1/ moved to the baselines repo (2026-09 restructure); paths unchanged
p = snapshot_download("armteam/hapticwam-baselines", repo_type="model",
                      allow_patterns=["pi05_phantom_expert_v1/020000/pretrained_model/*",
                                      "pi05_phantom_expert_v1/paligemma_tokenizer/*"],
                      local_dir=os.path.expanduser("~/phantom-icra-2027/phantom/runs/pi05_20k/hub"))
print("downloaded", p)
PY
ln -sfn hub/pi05_phantom_expert_v1/020000/pretrained_model runs/pi05_20k/pretrained_model
ln -sfn hub/pi05_phantom_expert_v1/paligemma_tokenizer runs/pi05_20k/paligemma_tokenizer
"$V/bin/python" tools/patch_pi05_processors.py --policy-dir runs/pi05_20k/pretrained_model --tokenizer "$P/runs/pi05_20k/paligemma_tokenizer"
"$V/bin/python" -c "from phantom.scripts.lerobot_server import dir_digest; print('ckpt sha', dir_digest('runs/pi05_20k/pretrained_model'))"
df -h / | awk 'NR>1{print "free", $4}'
echo "PI05 BOX SETUP DONE — warm with: ./serve_bg.sh 5 (needs ~8 GB GPU free)"
