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

Selects the reviewed clean teleoperation fit scene and runs Isaac headless.
Carton uses carton_teleop_capacity270_fit.json (full 250 ml, estimated dimensions).
Egg uses egg_teleop_reconstruction_fit.json.
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
  carton) SCENE_CONFIG="carton_teleop_capacity270_fit.json" ;;
  egg) SCENE_CONFIG="egg_teleop_reconstruction_fit.json" ;;
  *) echo "A supported --task is required: carton or egg" >&2; exit 2 ;;
esac
exec "$REPO_ROOT/tools/sim/launch_waffles.sh" \
  --config "$REPO_ROOT/configs/sim/$SCENE_CONFIG" "${RUN_ARGS[@]}"
