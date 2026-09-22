#!/bin/bash
# Rig box: clean pi0.5 venv — python 3.11 (same ABI as the phantom venv), torch cu130 for the 5090,
# lerobot 0.4.4 + transformers 4.53.2 + lerobot's patched transformers model files (tf_replace.tgz,
# taken from the training box's working venv). Then the 20k checkpoint + tokenizer from the hub and the
# processor patch. CPU/disk only — no GPU use.
#   HF_TOKEN=hf_xxx bash pi05_setup2.sh
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN inline}"
B=$HOME/phantom-icra-2027; P=$B/phantom; V=$B/pi05venv
rm -rf "$V"
~/.local/bin/uv venv --python 3.11 "$V"
~/.local/bin/uv pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0" "torchvision==0.28.0" \
  || ~/.local/bin/uv pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cu130 torch torchvision
"$V/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
~/.local/bin/uv pip install --python "$V/bin/python" "lerobot==0.4.4" "transformers==4.53.2"
"$V/bin/python" -c "import lerobot, transformers; print('lerobot', lerobot.__version__, 'transformers', transformers.__version__)"
SP=$("$V/bin/python" -c "import site; print(site.getsitepackages()[0])")
[ -f "$B/tf_replace.tgz" ] && (cd "$SP/transformers/models" && tar xzf "$B/tf_replace.tgz" && echo "transformers_replace applied")
"$V/bin/python" -c "from transformers.models.siglip import check; print('replace ok:', check.check_whether_transformers_replace_is_installed_correctly())"
# the phantom package itself (lerobot_server, the adapter) — as a path, no install
echo "$P" > "$SP/zz_phantom_repo.pth"
"$V/bin/python" -c "import phantom.inference.lerobot_policy, phantom.scripts.lerobot_server; print('phantom adapter importable')"
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
