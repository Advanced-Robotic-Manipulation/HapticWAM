# Inference: running a trained teacher

How to load a trained teacher checkpoint and run it — mocked dry run or real
rig. Complements [deployment_runtime.md](deployment_runtime.md) (loop
architecture) and [launch_guide.md](launch_guide.md) (all commands by stage).
Verified end-to-end with `teacher_v3_790eps` on the 5090 box, 2026-08-11.

## What a checkpoint needs at inference

| Piece | Where | Notes |
|---|---|---|
| teacher ckpt (`teacher_020000.pt`) | hub `armteam/phantom-checkpoints` (private) → `runs/<run_name>/` | payload keys: `lora`, `phantom_modules`, `ema`, `norm_stats`, `configs` (incl. `text_conditioning` provenance), `optimizer`/`scheduler` (training-only) |
| Cosmos base weights + tokenizer | `paths.local.yaml::cosmos_weights_root` | same files as training |
| text-embedding cache | `paths.local.yaml::cosmos_text_embedding_cache` | **required for v3+** (trained text-conditioned); ships next to the ckpt on the hub. **Do NOT set it for v2** — v2 trained on the empty-string embedding and must run that way |
| hardware yaml | `--hardware configs/hardware.nuc.yaml` | must be the config family the data was recorded with: shapes are hard-asserted at ckpt load; a VALUES-only drift logs a warning and proceeds |

Check a checkpoint's own record before running it:
`torch.load(ckpt, map_location="cpu")["configs"]["text_conditioning"]` —
`active: True` + the task list means the cache is mandatory.

## Task prompt

The prompt IS the task name — raw dataset tokens, not natural language:
`Carton` | `waffles` | `egg` | `whiteboard`. `--task` sets both the episode
metadata and the conditioning text (`--text` overrides the latter only).
The string must be a key of the cache; unknown text falls back to the
empty-string embedding with a warning (that's v2 behavior, not v3).
`*_fail` names exist in the cache for completeness but are never prompted at
deploy. Always pass `--ema` (EMA weights are the deploy weights).

## Mocked dry run with the real model (no hardware)

`configs/hardware.nuc.mock.yaml` = the rig config with `mode.drivers: mock` —
identical shapes (so the ckpt loads) but synthetic drivers, so it runs on any
CUDA box:

```bash
python -m phantom.scripts.run_deploy --system teacher \
    --ckpt runs/teacher_v3_790eps/teacher_020000.pt --ema \
    --task whiteboard --hardware configs/hardware.nuc.mock.yaml \
    --episodes 1 --max-replans 5 --device cuda \
    --out /path/to/scratch/deploy_smoke
```

Healthy output: `replan N: latency=~1.0s gate=... sigma_max=... accepted=True`
per replan, then `episode 0: <path> (replans=N stop=None safety_events=0)`,
and the episode directory contains the full zarr stream set +
`planner_trace.json` (per-replan σ/gate/p_evt/actions — the thing to eyeball).

## Real rig

Same command with `--hardware configs/hardware.nuc.yaml`, no `--out` needed
(defaults under `episodes_root/deploy/<date>`). Operator flow, safety matrix
and protective-stop recovery: [deployment_runtime.md](deployment_runtime.md).
Modes other than `teacher` pair with their own checkpoints
(`--system student|vision_only|...`).

## Measured (5090, teacher_v3_790eps, mock drivers, default nfe)

- replan latency: **1.03–1.41 s** (first replan warms up) — inside the 1.6 s
  chunk duration @ H=16, so the receding-horizon loop is real-time viable
- model + VAE load to first replan: ~3 min
- GPU footprint: single-digit GiB (ran alongside ~8 GiB of other jobs on a
  32 GiB card with ample headroom; exact peak not yet profiled)

## Box setup used (compute3 reference)

```
~/phantom-icra-2027/
  cosmos-predict2.5-2b/          # base weights + tokenizer.pth
  phantom/                       # repo, .venv inside
    runs/teacher_v3_790eps/      # teacher_020000.pt + text_embeddings.pt
    configs/paths.local.yaml     # cosmos_* roots + cosmos_text_embedding_cache
```

Known first-run gotcha (fixed): the first snapshot used to race the sensor
workers' first frames ("no camera frames yet") — `run_episode` now blocks up
to 10 s for ring warm-up, which also covers the real rig's first episode
right after connect.
