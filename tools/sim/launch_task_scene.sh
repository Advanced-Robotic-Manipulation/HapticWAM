#!/usr/bin/env bash
# Shared simulator runtime; this launcher never constructs a real-device driver.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK=""
RUN_ARGS=()
while (($#)); do
  case "$1" in
    --task)
      if (($# < 2)); then
        echo "--task requires carton or egg" >&2
        exit 2
      fi
      TASK="${2,,}"
      shift 2
      ;;
    --task=*) TASK="${1#*=}"; TASK="${TASK,,}"; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: launch_task_scene.sh --task carton|egg --episode PREPARED --output OUTPUT [runner options]

Selects configs/sim/<task>_teleop_reconstruction_fit.json and runs Isaac headless.
These defaults correspond to the reviewed clean teleoperation fit episodes.
For a session-specific reconstruction, pass --config PATH to override that default.
All remaining options are forwarded to run_waffles.py; see launch_waffles.sh --help.
Set ISAAC_SIM_ROOT to use a different Isaac installation. No hardware is launched.
EOF
      exit 0
      ;;
    *) RUN_ARGS+=("$1"); shift ;;
  esac
done
case "$TASK" in
  carton|egg) ;;
  *) echo "A supported --task is required: carton or egg" >&2; exit 2 ;;
esac
exec "$REPO_ROOT/tools/sim/launch_waffles.sh" \
  --config "$REPO_ROOT/configs/sim/${TASK}_teleop_reconstruction_fit.json" "${RUN_ARGS[@]}"
