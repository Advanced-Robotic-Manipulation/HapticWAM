#!/bin/bash
# Package a fine-tune data root for transfer (symlinks dereferenced): <root>.tar (no compression,
# the zarr chunks are already compressed). Usage: package_simft_root.sh <root> <out.tar>
set -euo pipefail
ROOT=$1; OUT=$2
cd "$(dirname "$ROOT")"
tar --dereference -cf "$OUT" "$(basename "$ROOT")"/manifests "$(basename "$ROOT")"/norm_stats.json "$(basename "$ROOT")"/tasks
ls -la "$OUT"; sha256sum "$OUT"
