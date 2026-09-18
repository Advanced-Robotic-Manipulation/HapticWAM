#!/bin/bash
# non-destructive: mirror the un-mirrored rig deploy days to the hub (rollouts/deploy_<day>/), like deploy_20260908
: "${HF_TOKEN:?}"; export HF_HUB_DISABLE_XET=1
cd ~/phantom-icra-2027/data/episodes/deploy
for d in 20260811 20260813 20260814 20260818 20260820 20260828 20260901 20260904 20260907; do
  [ -d $d ] || continue
  echo "=== $d $(date +%H:%M)"
  ~/phantom-icra-2027/phantom/.venv/bin/python - $d <<'PY'
import sys
from huggingface_hub import HfApi
d = sys.argv[1]
HfApi().upload_folder(folder_path=d, path_in_repo=f"rollouts/deploy_{d}", repo_id="armteam/phantom-checkpoints", repo_type="model", ignore_patterns=["*.tmp"])
print("uploaded", d)
PY
done
echo "OLD DEPLOY UPLOAD DONE $(date +%H:%M)"
