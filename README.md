# PHANTOM

**P**redictive **H**aptic **ANT**icipation with **O**cclusion-robust **M**anipulation — the
full research codebase for the ICRA 2027 pipeline specified in [pipeline.md](pipeline.md).

PHANTOM fine-tunes **Cosmos-Predict2.5-2B `robot/action-cond`** into a tactile
world-action model (teacher), then **distills the fingertip sensors away**
(student) so a robot can anticipate and pre-shape for contact it can no longer
feel. Modules: **HHT** (heterogeneous haptic tokenizer), **ACC** (anticipatory
contact gate), **ACE head** (asymmetric generation + contact-event
reparametrization), **LFA** (latent-frame action injection), **HID / HID-S**
(haptic-imagination distillation + force-safety fine-tune).

Hardware: UR arm (RTDE) + Robotiq 2F gripper + 2× DM-Tac W2L optical-tactile
sensors + RealSense. Training: 8×H100 (Linux). Inference: RTX 5090.

## Two files you configure, nothing else

| File | What lives there |
|---|---|
| [configs/hardware.yaml](configs/hardware.yaml) | **Every hardware hyperparameter** — sensor resolutions, rates, force-unit scales, thresholds, arm generation, safety limits. Fields marked `# BENCH:` are unverified defaults that the [day-1 bench](docs/hardware_bench_day1.md) fills in. No shape/rate/threshold is hardcoded anywhere in `phantom/` — change a value here and the whole stack follows. |
| [configs/paths.yaml](configs/paths.yaml) | Where the cloned `cosmos-predict2.5` repo and the `cosmos-predict2.5-2b` weights live, plus data/run roots. Per-machine overrides go in `configs/paths.local.yaml` (gitignored). |

## Quickstart (no hardware, no GPU)

For the two Linux target machines use the per-machine manifests in
[requirements/](requirements/README.md) (`requirements-5090.txt` for the
deployment/recording box, `requirements-h100.txt` for the training cluster).
On a dev box:

```bash
pip install -e .[dev,train]

# end-to-end recording pipeline on mock drivers:
#   mock rig -> 2 tactile worker processes -> ring buffers -> zarr episode
#   -> offline derived pass -> event timeline recovered
python -m phantom.scripts.mock_smoke

# model side on a tiny CPU backbone + synthetic episodes:
#   build PhantomDiT -> teacher training step -> 5-NFE sample -> drop-video
#   -> HID distill step -> HID-S step -> checkpoint round trip
python -m phantom.scripts.smoke_test --tiny --synthetic

# unit tests (config validators, derived-channel closure, packing round trips,
# sequence-layout invariants, ring buffers, tiny-model end-to-end)
pytest tests/
```

Everything above runs identically after you change values in
`configs/hardware.yaml` — that robustness is itself under test
(`tests/test_windows.py::test_mutated_resolution_still_works`).

## Repository map

```
configs/            hardware.yaml + paths.yaml (+ eval campaign example)
phantom/
  config/           pydantic schemas + validators for the two config files
  data/             derived contact channels, zarr episode store, training windows,
                    synthetic episode generator
  drivers/          hardware ABCs; real drivers (dmrobotics/ur_rtde/Robotiq/RealSense)
                    and mocks driven by one shared contact scenario
  recording/        shared-memory rings, per-sensor worker processes, episode recorder
  timesync/         RTDE master clock
  teleop/           keyboard / SpaceMouse teleoperation
  backbone/         cosmos-predict2.5 integration: import shims, checkpoint loader,
                    LoRA injection, VAE + text-embedding wrappers
  model/            SequenceLayout, PhantomDiT, HHT, ACC, ACE (packing/heads/losses),
                    PhantomRectifiedFlow
  train/            4 training programs + DAgger driver (plain PyTorch, torchrun DDP)
  inference/        PhantomPolicy.replan()
  deploy/           receding-horizon runtime: planner, executor, governor, safety
  dagger/           student rollouts, teacher relabeling, manifests
  eval/             trial runner (ledger), metrics, recovery/retention aggregation
  scripts/          all entry points (see the launch guide)
tests/              cosmos-free unit tests + tiny-model integration tests
docs/               the documentation set below
```

## Documentation

| Doc | Read it when |
|---|---|
| [docs/code_structure.md](docs/code_structure.md) | you want the module-by-module map and the design rules |
| [docs/launch_guide.md](docs/launch_guide.md) | you need the exact command for any stage, on any machine |
| [docs/hardware_bench_day1.md](docs/hardware_bench_day1.md) | the sensors/arm arrive — fills the `BENCH:` fields |
| [docs/data_collection_sop.md](docs/data_collection_sop.md) | collecting the 750-episode teleop dataset |
| [docs/training_playbook.md](docs/training_playbook.md) | running pretrain → teacher → HID → DAgger → HID-S |
| [docs/deployment_runtime.md](docs/deployment_runtime.md) | deploying on the robot; safety layer; latency budget |

## Project workflow (bird's eye)

1. **Day-1 bench** → fill `configs/hardware.yaml`, set `bench_verified: true`.
2. **Contact play** (~2–4 h unscripted poking) → `pretrain_tactile`.
3. **Teleop collection** (750 episodes / 5 tasks + failure episodes) → `dump_norm_stats` → `postprocess_episodes`.
4. **Teacher training** (8×H100, LoRA + new modules, ~60–120 M trainable).
5. **HID distillation** → student; **2 DAgger rounds** (rollouts on the rig, teacher relabels).
6. Optional **HID-S** safety fine-tune.
7. **Evaluation campaign** (5 systems × 5 tasks × occlusion variants) → recovery-ratio / retention tables + the ACC lead-time plot.
