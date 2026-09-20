#!/bin/bash
# Pin a provisioning script to the exact commit a rental is allowed to run.
#
#   bash tools/pack_repo.sh [out-dir]   -> <out>/provision_v5.<short>.sh      (scp THIS to the box)
#                                          <out>/provision_distill.<short>.sh
#
# The provisioning scripts clone the code from the PUBLIC GitHub repo
# Advanced-Robotic-Manipulation/HapticWAM. A pinned script refuses to run if the
# clone lands on any commit other than the one baked in here, so a rental can
# never silently run code other than the audited commit (2026-08-26).
#
# This no longer builds phantom_repo_latest.tar.gz and nothing uploads a source
# tarball to the hub any more: the clone replaces it.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "FATAL: tracked changes not committed"; exit 1; }
OUT=${1:-.}
SHA=$(git rev-parse HEAD)              # FULL sha: `git fetch --depth 1 origin <sha>` requires it
SHORT=$(git rev-parse --short HEAD)
git fetch -q origin 2>/dev/null || true
git merge-base --is-ancestor HEAD origin/main 2>/dev/null || \
  echo "WARNING: $SHORT is not on origin/main — a rental cannot clone a commit the remote does not have"
mkdir -p "$OUT"
for s in provision_v5 provision_distill; do
  sed "s/__REPO_COMMIT__/$SHA/" "tools/$s.sh" > "$OUT/$s.$SHORT.sh"
  chmod +x "$OUT/$s.$SHORT.sh"
done
echo "pinned $SHA -> $OUT/provision_v5.$SHORT.sh, $OUT/provision_distill.$SHORT.sh"
