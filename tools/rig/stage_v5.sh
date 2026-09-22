#!/bin/bash
# Site-specific: host aliases and absolute paths below are our lab's -- adapt to your own setup.
# Pull one v5 checkpoint from the rental and stage it under a TEAM LABEL (no step numbers).
#   ssh -A compute3 ~/phantom-icra-2027/stage_v5.sh <step> <label>     e.g. 500 v5_1
set -euo pipefail
STEP=$(printf "%06d" "${1:?step}"); LABEL=${2:?label e.g. v5_1}
RUN=${RUN:-teacher_v5_batch0822}; PORT=${PORT:-34688}; HOST=${HOST:-ssh3.vast.ai}
D=~/phantom-icra-2027/phantom/runs/$RUN; mkdir -p "$D"
scp -o StrictHostKeyChecking=no -P "$PORT" "root@$HOST:/workspace/phantom-v5/runs/teacher/$RUN/teacher_$STEP.pt" "$D/$LABEL.pt.part"
mv "$D/$LABEL.pt.part" "$D/$LABEL.pt"
~/phantom-icra-2027/phantom/.venv/bin/python -c "import torch,sys; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); assert p.get('ema'), 'no EMA'; print('ok', sys.argv[2])" "$D/$LABEL.pt" "$LABEL"
ln -sfn "$LABEL.pt" "$D/DEMO.pt"
echo "$(date -u +%FT%TZ) $LABEL <- teacher_$STEP.pt" >> "$D/.stage_map"
echo "staged: $D/DEMO.pt -> $LABEL.pt   (GO_v5_<task>.sh uses it)"
