#!/bin/bash
# Rebuild the /dev/shm working set of the scripted-expert campaign on compute3 after a reboot
# (everything under /dev/shm is lost; exported episodes on the data disk survive).
#
#   bash tools/sim/rebuild_expert_campaign_env.sh [<commit-or-branch>=main]
#
# Produces: runtime tree (git archive + paths.local.yaml), zoo inputs (from
# docs/results/sim_zoo_20260912/inputs + the js4 hardware variant), driver dir (score_trial.py
# + campaign launchers), start_pool.json (real start poses from deploy rollouts + val demos).
set -euo pipefail
REV=${1:-main}
REPO=/home/physicalai/phantom-icra-2027/phantom
PY=$REPO/.venv/bin/python
RT=/dev/shm/phantom_sim_zoo_runtime_v17_20260912
IN=/dev/shm/phantom_sim_zoo_inputs_20260912
DRV=/dev/shm/phantom_sim_zoo_driver_20260912
RAW=/dev/shm/phantom_sim_zoo_raw_20260912
mkdir -p "$RT" "$IN" "$DRV" "$RAW"
cd "$REPO"
git archive "$REV" | tar -x -C "$RT"
cp configs/paths.local.yaml "$RT/configs/"
cp -r docs/results/sim_zoo_20260912/inputs/. "$IN/"
sed "s/joint_speed_stop_rad_s: .*/joint_speed_stop_rad_s: 4.0/" "$IN/hardware_input.yaml" > "$IN/hardware_input_expert_js4.yaml"
cp tools/sim/zoo/score_trial.py "$DRV/"
# start pool: first rows of every waffle deploy rollout + the val demos
CUDA_VISIBLE_DEVICES="" $PY - <<'EOF'
import zarr, numpy as np, glob, json, os
pool=[]
def first_rows(E):
    q=zarr.open(E+"/arm_q.zarr","r"); p=zarr.open(E+"/arm_tcp_pose.zarr","r"); g=zarr.open(E+"/gripper.zarr","r"); f=zarr.open(E+"/arm_ft.zarr","r")
    return dict(q=[float(x) for x in q["data"][0]], measured_tcp_pose=[float(x) for x in p["data"][0]], gripper=float(g["data"][0][0]), wrist_ft=[float(x) for x in f["data"][0]], t_master=float(q["ts"][0]))
for E in sorted(glob.glob("/home/physicalai/phantom-icra-2027/data/episodes/deploy/*/ep_*waffles_*")):
    try: m=json.load(open(E+"/meta.json")); r=first_rows(E)
    except Exception: continue
    r.update(source="deploy", episode=os.path.basename(E), policy=m.get("policy"), success=m.get("success")); pool.append(r)
for E in sorted(glob.glob("/home/physicalai/phantom-icra-2027/data/val_eval/tasks/waffles/ep_*")):
    try: r=first_rows(E)
    except Exception: continue
    r.update(source="val_demo", episode=os.path.basename(E)); pool.append(r)
json.dump({"states": pool, "band_rule": "y<=-0.28 and z>=0.31"}, open("/dev/shm/phantom_sim_zoo_inputs_20260912/start_pool.json","w"))
print("start pool", len(pool))
EOF
# launchers
cat > "$DRV/launch_campaign.sh" <<'EOS'
#!/bin/bash
# usage: launch_campaign.sh <scene.json> <n> <lanes>   (keep lanes <= 3: six Isaac lanes exhausted RAM on 2026-09-12)
SCENE=$1; N=${2:-300}; LANES=${3:-3}
RT=/dev/shm/phantom_sim_zoo_runtime_v17_20260912; IN=/dev/shm/phantom_sim_zoo_inputs_20260912
RAW=/dev/shm/phantom_sim_zoo_raw_20260912/expert_campaign; OUT=/home/physicalai/phantom-icra-2027/data/sim_expert_20260912
IDLE=/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260911/ep_teacher_waffles_1789123632_000
mkdir -p $RAW; cd $RT
for L in $(seq 0 $((LANES-1))); do
  nohup env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python tools/sim/expert_campaign.py \
    --n $N --lanes $LANES --lane $L --runtime $RT --inputs $IN --driver /dev/shm/phantom_sim_zoo_driver_20260912 \
    --raw-root $RAW --out-root $OUT --start-pool $IN/start_pool.json --scene $SCENE \
    --hardware-input $IN/hardware_input_expert_js4.yaml --hardware configs/hardware.nuc.yaml --idle-tactile-from $IDLE \
    --duration 20 --holdout ep_waffles_1785593867_004 ep_waffles_1785594044_014 ep_waffles_1785594169_018 ep_waffles_1785594205_020 ep_waffles_1785594503_010 \
    > $RAW/lane$L.log 2>&1 &
  echo "lane $L pid $!"
done
EOS
cat > "$DRV/launch_finetune.sh" <<'EOS'
#!/bin/bash
# usage: launch_finetune.sh <data_root> <max_steps> [batch] [accum]
ROOT=$1; STEPS=${2:-2500}; BS=${3:-2}; ACC=${4:-4}
cd /home/physicalai/phantom-icra-2027/phantom
V6=runs/teacher_v6/teacher_020000.pt
mkdir -p logs
nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m phantom.train.train_teacher \
  --data $ROOT/tasks --hardware configs/hardware.nuc.yaml --allow-config-drift \
  --init-weights $V6 --max-steps $STEPS --split train \
  --wrench-baseline-rows 8 --grasp-frac 0.3 --photo-aug 1.0 --acc-two-pass \
  --lr 2e-5 --lr-new-modules 6e-5 --warmup-steps 150 --ema-decay 0.995 \
  --ckpt-every 500 --eval-every 500 --event-band-weight 0 \
  --batch-size $BS --grad-accum $ACC --num-workers 6 --device cuda \
  --run-name teacher_v6_simft --allow-skipped-episodes > logs/train_teacher_v6_simft_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
echo "train pid $!"
EOS
chmod +x "$DRV"/*.sh
# v6 norm stats for the fine-tune root
mkdir -p /home/physicalai/phantom-icra-2027/data/sim_expert_20260912
CUDA_VISIBLE_DEVICES="" $PY - <<'EOF'
import json, torch, numpy as np
ck=torch.load("/home/physicalai/phantom-icra-2027/phantom/runs/teacher_v6/teacher_020000.pt", map_location="cpu", weights_only=False)
ns=ck["norm_stats"]
json.dump({"mean":{k:np.asarray(v).tolist() for k,v in ns["mean"].items()}, "std":{k:np.asarray(v).tolist() for k,v in ns["std"].items()}},
          open("/home/physicalai/phantom-icra-2027/data/sim_expert_20260912/norm_stats_v6.json","w"), indent=1)
print("norm stats written")
EOF
echo "rebuilt: $RT $IN $DRV"
