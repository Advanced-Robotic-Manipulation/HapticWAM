#!/bin/bash
# mirror un-mirrored rig deploy days to the hub as ONE tar.zst each (the hub repo is at the 20k-file
# limit, raw folders are rejected). One day at a time; the local tar is removed after the hub confirms.
# 2026-09 restructure: deploy-day tars now live FLAT in the rollouts DATASET repo.
#   HF_TOKEN=hf_xxx nohup bash pack_upload_deploy.sh > logs/pack_upload_deploy.log 2>&1 &
: "${HF_TOKEN:?}"; export HF_HUB_DISABLE_XET=1
B=$HOME/phantom-icra-2027; D=$B/data/episodes/deploy; P=$D/_pack; mkdir -p "$P"
PY=$B/phantom/.venv/bin/python
HUB_ROLLOUTS=armteam/hapticwam-rollouts
for d in 20260811 20260813 20260814 20260818 20260820 20260828 20260901 20260904 20260907; do
  [ -d "$D/$d" ] || continue
  echo "=== $d $(date +%H:%M) free $(df --output=avail -BG / | tail -n 1 | tr -dc 0-9)G"
  ( cd "$D" && tar -cf - "$d" | zstd -T4 -q -o "$P/deploy_$d.tar.zst" ) || { echo "!!! tar $d failed"; rm -f "$P/deploy_$d.tar.zst"; continue; }
  ls -la "$P/deploy_$d.tar.zst" | awk '{print $5}'
  "$PY" - "$P/deploy_$d.tar.zst" "$d" "$HUB_ROLLOUTS" <<'PY' || { echo "!!! upload $d failed"; continue; }
import sys, os
from huggingface_hub import HfApi
f, d, repo = sys.argv[1:4]
api = HfApi()
api.upload_file(path_or_fileobj=f, path_in_repo=f"deploy_{d}.tar.zst",
                repo_id=repo, repo_type="dataset")
info = api.get_paths_info(repo, [f"deploy_{d}.tar.zst"], repo_type="dataset")
size = getattr(info[0], "size", None) or getattr(getattr(info[0], "lfs", None), "size", None)
local = os.path.getsize(f)
print("hub size", size, "local", local)
assert size == local, "size mismatch"
print("uploaded+verified", d)
PY
  rm -f "$P/deploy_$d.tar.zst"
done
echo "PACK UPLOAD DONE $(date +%H:%M)"
