#!/bin/bash
# non-destructive: mirror the un-mirrored rig deploy days to the hub, like deploy_20260908.
# 2026-09 restructure: the loose zarr tree is no longer uploaded folder-by-folder — each
# session is ONE uncompressed tar at armteam/hapticwam-rollouts:zarr_rollouts/deploy_<day>.tar
# whose members start with deploy_<day>/ (the layout the new repo publishes).
: "${HF_TOKEN:?}"; export HF_HUB_DISABLE_XET=1
HUB_ROLLOUTS=armteam/hapticwam-rollouts
cd ~/phantom-icra-2027/data/episodes/deploy
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
for d in 20260811 20260813 20260814 20260818 20260820 20260828 20260901 20260904 20260907; do
  [ -d $d ] || continue
  echo "=== $d $(date +%H:%M)"
  # --transform renames the members 20260811/... -> deploy_20260811/... (GNU tar)
  tar -cf "$T/deploy_$d.tar" --exclude='*.tmp' --transform "s|^$d|deploy_$d|" "$d" \
    || { echo "!!! tar $d failed"; rm -f "$T/deploy_$d.tar"; continue; }
  ~/phantom-icra-2027/phantom/.venv/bin/python - "$T/deploy_$d.tar" $d "$HUB_ROLLOUTS" <<'PY'
import sys
from huggingface_hub import HfApi
f, d, repo = sys.argv[1:4]
HfApi().upload_file(path_or_fileobj=f, path_in_repo=f"zarr_rollouts/deploy_{d}.tar",
                    repo_id=repo, repo_type="dataset")
print("uploaded", d)
PY
  rm -f "$T/deploy_$d.tar"
done
echo "OLD DEPLOY UPLOAD DONE $(date +%H:%M)"
