#!/usr/bin/env bash
set -euo pipefail
PHANTOM_SIM_RUN_ROOT="${PHANTOM_SIM_RUN_ROOT:-/home/physicalai/phantom-icra-2027/sim/waffles}"
PHANTOM_SIM_TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PHANTOM_SIM_TOOLS/launch_waffles.sh" \
  --episode "$PHANTOM_SIM_RUN_ROOT/evidence/fit/ep_waffles_1787395928_000" \
  --config "$PHANTOM_SIM_TOOLS/../../configs/sim/waffles_pick_place.json" \
  --output "$PHANTOM_SIM_RUN_ROOT/runs/view_$(date -u +%Y%m%d_%H%M%S)_$$" \
  --mode dynamics --gui "$@"
