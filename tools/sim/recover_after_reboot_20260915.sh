#!/bin/bash
# Site-specific: host aliases and absolute paths below are our lab's -- adapt to your own setup.
# Rig box: restore the sim-generation working set after a reboot (/dev/shm is wiped) and restart the night supervisor.
# Backups live on the data disk: data/shm_backup_20260915/{phantom_sim_zoo_driver_20260912,phantom_sim_zoo_inputs_20260912}.
set -euo pipefail
B=/home/physicalai/phantom-icra-2027/data/shm_backup_20260915
REPO=/home/physicalai/phantom-icra-2027/phantom
RT=/dev/shm/phantom_sim_zoo_runtime_v18_20260914
[ -d /dev/shm/phantom_sim_zoo_driver_20260912 ] || cp -r $B/phantom_sim_zoo_driver_20260912 /dev/shm/
[ -d /dev/shm/phantom_sim_zoo_inputs_20260912 ] || cp -r $B/phantom_sim_zoo_inputs_20260912 /dev/shm/
mkdir -p /dev/shm/phantom_sim_zoo_raw_20260912
if [ ! -d $RT ]; then
  mkdir -p $RT && (cd $REPO && git archive HEAD | tar -x -C $RT) && cp $REPO/configs/paths.local.yaml $RT/configs/
fi
mkdir -p $RT/configs/sim/expert && cp $REPO/configs/sim/expert/*.json $RT/configs/sim/expert/
pgrep -f "night_supervisor.s[h]" >/dev/null || (cd /tmp && nohup bash /dev/shm/phantom_sim_zoo_driver_20260912/night_supervisor.sh > /dev/null 2>&1 &)
echo "recovered $(date -u +%FT%TZ): runtime $(ls $RT | wc -l) entries, supervisor $(pgrep -f 'night_supervisor.s[h]' | wc -l)"
