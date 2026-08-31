#!/bin/bash
# v5 fine-tune arm: same as GO_ANY.sh but with the v5 checkpoint staged by stage_v5.sh.
#   GO_v5_ANY.sh <task> [episodes=3] [nfe=5] [guidance=1.0]
# Control arm = GO_<task>.sh (v4 DEMO.pt). Interleave v4/v5 episodes within each grid cell.
CKPT=runs/teacher_v5_batch0822/DEMO.pt exec ~/phantom-icra-2027/GO_ANY.sh "$@"
