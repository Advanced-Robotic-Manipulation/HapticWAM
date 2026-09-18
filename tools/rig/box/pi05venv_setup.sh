#!/bin/bash
# venv only (CPU/disk, ~1 GB): python3.12 + .pth onto the phantom venv (torch cu130) + lerobot/transformers pins
set -euo pipefail
B=$HOME/phantom-icra-2027; P=$B/phantom; V=$B/pi05venv
if [ ! -x "$V/bin/python" ]; then
  ~/.local/bin/uv venv --python 3.12 "$V"
  SP=$("$P/.venv/bin/python" -c "import site; print(site.getsitepackages()[0])")
  echo "$SP" > "$V/lib/python3.12/site-packages/zz_phantom_venv.pth"
fi
"$V/bin/python" -c "import torch; print(\"torch via .pth\", torch.__version__, torch.cuda.is_available())"
~/.local/bin/uv pip install --python "$V/bin/python" "lerobot==0.4.4" "transformers==4.53.2"
"$V/bin/python" -c "import lerobot, transformers, torch; print(\"lerobot\", lerobot.__version__, \"transformers\", transformers.__version__, \"torch\", torch.__version__)"
echo "PI05 VENV DONE $(date)"
