# Rig session — v5 fine-tune vs v4 (how to run inference on compute3)

State as of 2026-08-28. Everything below was verified live on compute3 (repo `b0e87b5`, torch 2.13 + CUDA OK).

## What is on the box
```
~/phantom-icra-2027/phantom/runs/teacher_v5_batch0822/DEMO.pt -> v5_6.pt   # v5 fine-tune, final (best) build
~/phantom-icra-2027/phantom/runs/teacher_v5_batch0822/v5_1.pt              # first v5 build (for ablation only)
~/phantom-icra-2027/phantom/runs/teacher_v4_790eps/DEMO.pt   -> teacher_020000.pt   # v4 CONTROL (unchanged)
```
`v5_6` is byte-identical (sha256) to hub `armteam/phantom-checkpoints/teacher_v5_batch0822/teacher_003000.pt`.
All six v5 builds, the training log and every offline eval JSON are on the hub in that folder.
Build labels on the box hide step numbers on purpose; the mapping is `runs/teacher_v5_batch0822/.stage_map`.

## Preflight (once per session)
1. Robot ON (pendant: Enable robot, Start), gripper powered, RealSense on the **USB3** hub port, gel sensors connected.
2. `ping 192.168.88.56` must answer (UR). `GO_ANY.sh` checks the arm and the camera USB3 mode and refuses otherwise.
3. Nobody else on the GPU: `nvidia-smi` (a few GB from other users is fine; a training job is not).
4. Place the taped 3x3 grid (40 mm spacing) centred on the demo object position.

## Launch — two arms, same flow, same flags
```
cd ~/phantom-icra-2027
./GO_waffles.sh     [episodes=3]   # v4 control  (runs/teacher_v4_790eps/DEMO.pt)
./GO_v5_waffles.sh  [episodes=3]   # v5 build    (runs/teacher_v5_batch0822/DEMO.pt)
```
(same for `Carton`, `egg`, `whiteboard`). Both go through `GO_ANY.sh` -> `run_deploy.py` with
`--ema --persistent-noise --nfe 5 --guidance 1.0`, `configs/hardware.nuc.yaml`, EMA weights (deploy default).
Per-episode flow (unchanged from LAUNCH.txt): wait ~3 min for model load + warmup; **Enter #1** homes the arm
slowly to the task's demo start pose (clear its path, e-stop in hand); the sigma gate prints live-vs-demo per axis
and refuses > 2.5 sigma (jog and Enter to re-check); **Enter #2** starts the episode; at the end answer the
outcome prompt: `s` success / `f` failure / `c` contaminated (a hand in frame etc.).

## Protocol for a comparable A/B
- One task first (waffles). Per grid cell: one v4 episode, then one v5 episode (interleaved), same placement.
- >= 10 episodes per arm per task (20 to see anything smaller than a night-and-day effect).
- Primary number = miss distance / z at close from `planner_trace.json` (not just grasp count). Note the cell id in
  the episode notes prompt. Film 2 successes + 2 misses per arm.
- Episodes land under `runs/deploy_*/ep_teacher_<task>_<epoch>_<seq>/` with `planner_trace.json`, condition tags
  in `meta.json` (`nfe5`, `g1.0`, `pnoise`, `ckpt:<file>`, ...).

## Staging a different build (needs the rental to exist; otherwise pull from the hub)
```
# from the hub (any time):
cd ~/phantom-icra-2027/phantom && .venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download; import shutil
p = hf_hub_download("armteam/phantom-checkpoints", "teacher_v5_batch0822/teacher_002500.pt", repo_type="model")
shutil.copy(p, "runs/teacher_v5_batch0822/v5_5.pt")
PY
ln -sfn v5_5.pt runs/teacher_v5_batch0822/DEMO.pt
```

## If something refuses to start
- `arm NOT reachable` -> robot off / pendant not started / wrong subnet.
- `NO CAMERA` or USB type not 3.x -> replug the RealSense into the back USB3 hub port.
- Gate refuses with a sigma table -> jog the arm toward the demo start (or place the object mid-mat), Enter to re-check.
- After a protective stop: clear it on the pendant when prompted; `run_deploy` rebuilds RTDE control itself
  between episodes and refuses the next episode if the script did not come back (exit code 4/5 = restart the process).
- Stale camera / stale arm stream now end the episode cleanly (stop reason in the log) instead of crashing.

## Offline numbers behind v5_6 (terminal_eval, 124 val episodes x 2 seeds, EMA)
endpoint error v4 20.7 mm -> v5_6 17.4 mm; z-at-end -4.7 -> -2.4 mm; commit ratio 1.72 -> 1.43; close timing +0.7 -> -0.1 steps;
new-batch holdout 20.7 -> 14.9 mm; every task improved. Offline != rig: the rig decides.
