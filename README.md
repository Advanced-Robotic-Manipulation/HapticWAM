# HapticWAM

**Haptic World-Action Model** — a tactile world-action model that learns to *imagine* contact,
then gives that imagination to a robot with no tactile sensors at all.

HapticWAM fine-tunes the frozen **Cosmos-Predict2.5-2B `robot/action-cond`** video-diffusion
transformer into a **teacher** that takes optical-tactile input and, at every replan, generates a
structured *contact package* — per-finger displacement and normal-force deltas, contact mask,
centre of pressure, slip, wrist wrench and a typed contact event — on the same latent grid as the
action chunk. The fingertip sensors are then **distilled away**: a **pad-free student** with an
identical architecture but no tactile input path keeps generating that contact future from vision,
proprioception and wrist F/T alone, so it can anticipate and pre-shape for contact it can no
longer feel.

Modules keep their names: **HHT** (heterogeneous haptic tokenizer), **ACC** (anticipatory contact
coupling gate), **ACE head** (asymmetric generation + contact-event reparametrization), **LFA**
(latent-frame action injection, Cosmos-Policy style), **HID / HID-S** (haptic-imagination
distillation + force-safety fine-tune).

The full specification is [pipeline.md](pipeline.md); start reading the operational state at
[docs/STATUS.md](docs/STATUS.md).

> HapticWAM was developed under the working name PHANTOM until 2026-09-19; the Python package and
> CLI keep the name `phantom`.

## Install

```bash
git clone --recurse-submodules git@github.com:Advanced-Robotic-Manipulation/HapticWAM.git
git submodule update --init cosmos-predict2.5   # the DiT code, imported as a library
pip install -e .[dev,train,cosmos]
```

Nothing from `cosmos-predict2.5` is pip-installed — it is imported from the path in
`configs/paths.yaml::cosmos_repo`. On machines without Transformer Engine / megatron builds the
import shims in `phantom/backbone/compat.py` are installed automatically.

Per-machine pinned manifests are in [requirements/](requirements/README.md):
`requirements-5090.txt` (the deployment/recording box), `requirements-h100.txt` /
`requirements-a100.txt` (training nodes). Non-pip steps — the Daimon `dmrobotics` SDK, RealSense
udev rules, SpaceMouse hidapi — are listed there too.

### Two files you configure, nothing else

| File | What lives there |
|---|---|
| [configs/hardware.yaml](configs/hardware.yaml) | **Every hardware hyperparameter** — sensor resolutions, rates, force-unit scales, thresholds, arm generation, safety limits. No shape/rate/threshold is hardcoded anywhere in `phantom/`; change a value here and the whole stack follows (`tests/test_windows.py::test_mutated_resolution_still_works` enforces it). |
| [configs/paths.yaml](configs/paths.yaml) | Where the cloned `cosmos-predict2.5` repo and the `cosmos-predict2.5-2b` weights live, plus data/run roots. Per-machine overrides go in `configs/paths.local.yaml` (gitignored). |

Training targets are picked in `configs/compute.yaml` (`target: rtx5090` | `h100x8` | `a100x8`).

## Hardware

UR arm over RTDE + Robotiq 2F gripper + 2× **DM-Tac W2L** optical-tactile fingertip sensors +
RealSense D435. The lab rig is a UR3 CB3 with a wrist F/T sensor. Compute: a **single RTX 5090**
for training (LoRA, 256–480p) and for deployment; training only can retarget to 8×H100 / 8×A100.
Verified sensor-SDK facts: [docs/sensor_sdk.md](docs/sensor_sdk.md). Bench procedure for filling
the `# BENCH:` fields: [docs/hardware_bench_day1.md](docs/hardware_bench_day1.md).

## Data and checkpoints

Everything heavy lives on the Hugging Face hub under `armteam`:

| Repo | What |
|---|---|
| [`armteam/hapticwam-teleop-raw`](https://huggingface.co/datasets/armteam/hapticwam-teleop-raw) | raw teleop episode sessions + the packed `dataset_v3_packed` training tarballs |
| [`armteam/hapticwam-sim-episodes`](https://huggingface.co/datasets/armteam/hapticwam-sim-episodes) | Isaac Sim episodes |
| [`armteam/hapticwam-rig-episodes`](https://huggingface.co/datasets/armteam/hapticwam-rig-episodes) | closed-loop rig deployment episodes |
| [`armteam/hapticwam-results`](https://huggingface.co/datasets/armteam/hapticwam-results) | the result media (videos, plots, frame dumps) archived out of `docs/results/` |
| [`armteam/phantom-checkpoints`](https://huggingface.co/armteam/phantom-checkpoints) | teacher / student / baseline weights — **id not renamed** |

`tools/provision_v5.sh` and `tools/provision_distill.sh` pull a training box's dataset and
checkpoints from the hub with nothing but an `HF_TOKEN`. The Markdown reports stay in the repo
under [docs/results/](docs/results/README.md); what moved to the hub and why is recorded in
[docs/history/RESULTS_ARCHIVE_20260918.md](docs/history/RESULTS_ARCHIVE_20260918.md).

## Run it

### No hardware, no GPU

```bash
# end-to-end recording pipeline on mock drivers:
#   mock rig -> 2 tactile worker processes -> ring buffers -> zarr episode
#   -> offline derived pass -> event timeline recovered
python -m phantom.scripts.mock_smoke

# model side on a tiny CPU backbone + synthetic episodes:
#   build the DiT -> teacher training step -> 5-NFE sample -> drop-video
#   -> HID distill step -> HID-S step -> checkpoint round trip
python -m phantom.scripts.smoke_test --tiny --synthetic
```

### Collect → train → distill → deploy

```bash
# 1. record, then close the derived channels and the norm stats
python -m phantom.scripts.record_episodes --task fragile_grasp --teleop spacemouse
python -m phantom.scripts.postprocess_episodes --data data/episodes/<date>
python -m phantom.scripts.dump_norm_stats     --data data/episodes/<date>

# 2. tactile-encoder SSL pretrain (single GPU)
python -m phantom.train.pretrain_tactile --data <contact_play_root>

# 3. teacher — tactile in, contact package + action chunk out
torchrun --nproc_per_node 8 -m phantom.train.train_teacher \
    --data <episodes_root> \
    --tactile-pretrain runs/tactile_pretrain/tactile_encoder_pretrain.pt \
    --run-name teacher_v1

# 4. HID distillation -> the pad-free student, then 2 DAgger rounds
torchrun --nproc_per_node 8 -m phantom.train.distill_hid \
    --teacher-ckpt runs/teacher/teacher_v1/teacher_050000.pt --data <episodes_root>
python -m phantom.train.dagger_driver --round 1 \
    --teacher-ckpt <teacher.pt> --demos <episodes_root> --rollouts <rollout_root>

# 5. optional force-safety fine-tune
python -m phantom.train.finetune_hids --student-ckpt runs/hid/hid_r2/student_030000.pt \
    --data <episodes_root>

# 6. deploy: teacher with pads, student without (tactile is still recorded in both)
python -m phantom.scripts.run_deploy --system teacher --ckpt <teacher.pt> --task fragile_grasp
python -m phantom.scripts.run_deploy --system student --ckpt <student.pt> --task fragile_grasp

# 7. evaluation campaign -> recovery-ratio / retention tables + the ACC lead-time plot
python -m phantom.scripts.run_eval --campaign configs/eval_campaign.yaml
```

Every training program smoke-runs anywhere with `--tiny --synthetic --max-steps 2`; every deploy
path dry-runs with `--tiny` and `mode.drivers: mock`. Exact commands per machine and per stage:
[docs/launch_guide.md](docs/launch_guide.md).

## Repository map

```
configs/            hardware.yaml + paths.yaml + compute.yaml (+ eval campaign example)
phantom/
  config/           pydantic schemas + validators for the config files
  data/             derived contact channels, zarr episode store, training windows,
                    synthetic episode generator
  drivers/          hardware ABCs; real drivers (dmrobotics/ur_rtde/Robotiq/RealSense)
                    and mocks driven by one shared contact scenario
  recording/        shared-memory rings, per-sensor worker processes, episode recorder
  timesync/         RTDE master clock
  teleop/           keyboard / SpaceMouse teleoperation
  backbone/         cosmos-predict2.5 integration: import shims, checkpoint loader,
                    LoRA injection, VAE + text-embedding wrappers
  model/            SequenceLayout, the DiT subclass, HHT, ACC, ACE (packing/heads/losses),
                    the rectified-flow wrapper
  train/            4 training programs + DAgger driver (plain PyTorch, torchrun DDP)
  inference/        policy.replan()
  deploy/           receding-horizon runtime: planner, executor, governor, safety
  dagger/           student rollouts, teacher relabeling, manifests
  eval/             trial runner (ledger), metrics, recovery/retention aggregation
  scripts/          all entry points (see the launch guide)
tests/              cosmos-free unit tests + tiny-model integration tests
tools/              rig launchers, provisioning, hub upload, sim and analysis tooling
papers/             the manuscript source (edited on Overleaf — do not rewrite in-repo)
docs/               the documentation set below
```

## Documentation

| Doc | Read it when |
|---|---|
| [docs/STATUS.md](docs/STATUS.md) | **first** — what is done, what remains, exact next actions |
| [docs/ARCH_EXPLAINER_0912.md](docs/ARCH_EXPLAINER_0912.md) | you want the "what do we actually train, and why should it work" argument |
| [docs/code_structure.md](docs/code_structure.md) | you want the module-by-module map and the design rules |
| [docs/launch_guide.md](docs/launch_guide.md) | you need the exact command for any stage, on any machine |
| [docs/training_playbook.md](docs/training_playbook.md) | running pretrain → teacher → HID → DAgger → HID-S |
| [docs/deployment_runtime.md](docs/deployment_runtime.md) | deploying on the robot; safety layer; latency budget |
| [docs/sensor_sdk.md](docs/sensor_sdk.md) | anything touches the DM-Tac sensors or the `dmrobotics` SDK |
| [docs/hardware_bench_day1.md](docs/hardware_bench_day1.md) | the sensors/arm arrive — fills the `BENCH:` fields |
| [docs/data_collection_sop.md](docs/data_collection_sop.md) | collecting the teleop dataset |
| [docs/isaac_sim.md](docs/isaac_sim.md) | the Isaac Sim reconstruction: measured-motion replay and policy rollouts |
| [docs/rig_session_v5.md](docs/rig_session_v5.md) | you are about to run the physical rig |
| [docs/results/](docs/results/README.md) | the archived experiment reports (historical text, PHANTOM naming) |
| [docs/history/README.md](docs/history/README.md) | resolving pre-2026-09-18 commit hashes |

## Tests and CI

```bash
pytest tests/                       # config validators, derived-channel closure, packing
                                    # round trips, sequence-layout invariants, ring buffers,
                                    # tiny-model end-to-end
```

The tests run without the Cosmos submodule and without the ~4.8 GB Cosmos-Predict2.5-2B weights:
the cases that need them carry the `requires_cosmos_repo` / `requires_cosmos_weights` markers and
skip. CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the same suite on Python
3.11 — the rig venv's interpreter, since `dmrobotics` needs 3.8–3.11 — with the submodule
deliberately left uninitialised.

## Licence

Apache-2.0 — see [LICENSE](LICENSE).
