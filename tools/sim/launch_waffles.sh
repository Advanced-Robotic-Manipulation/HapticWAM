#!/usr/bin/env bash
set -euo pipefail
SIM_ROOT="${ISAAC_SIM_ROOT:-/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CARB_APP_PATH="$SIM_ROOT/kit" ISAAC_PATH="$SIM_ROOT" EXP_PATH="$SIM_ROOT/apps"
# The supplied 6.0 python.sh overwrites LD_PRELOAD and loads an incompatible
# system NCCL before its bundled torch. Source the same environment, then use
# the matching bundled NCCL without changing the shared Isaac installation.
set +u
source "$SIM_ROOT/setup_python_env.sh"
set -u
export LD_PRELOAD="$SIM_ROOT/kit/libcarb.so:$SIM_ROOT/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib/libnccl.so.2"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$SIM_ROOT/kit/python/bin/python3" "$REPO_ROOT/tools/sim/run_waffles.py" "$@"
