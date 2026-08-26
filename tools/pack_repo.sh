#!/bin/bash
# Pack the repo at HEAD for a rental (provision_v5.sh unpacks it from the hub)
# and pin the provision script to that exact commit.
#
#   bash tools/pack_repo.sh [out-dir]      -> <out>/phantom_repo_latest.tar.gz
#                                             <out>/provision_v5.<sha>.sh  (scp THIS to the box)
#                                             then upload the tarball to
#                                             armteam/phantom-checkpoints/dataset_v3_packed/
#
# The tarball carries a COMMIT file; the pinned script refuses any tarball
# whose COMMIT differs, so a rental can never silently run code other than
# the audited commit (Codex review 2026-08-26).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "FATAL: tracked changes not committed"; exit 1; }
OUT=${1:-.}
SHA=$(git rev-parse --short HEAD)
mkdir -p "$OUT"
TMP=$(mktemp -d)
echo "$SHA" > "$TMP/COMMIT"
git archive --format=tar.gz --add-file="$TMP/COMMIT" -o "$OUT/phantom_repo_latest.tar.gz" HEAD
sed "s/__REPO_COMMIT__/$SHA/" tools/provision_v5.sh > "$OUT/provision_v5.$SHA.sh"
chmod +x "$OUT/provision_v5.$SHA.sh"
rm -rf "$TMP"
echo "packed $SHA -> $OUT/phantom_repo_latest.tar.gz ($(du -h "$OUT/phantom_repo_latest.tar.gz" | cut -f1)), pinned script $OUT/provision_v5.$SHA.sh"
