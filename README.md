# HapticWAM

**Haptic World-Action Model** — a tactile world-action model that learns to *imagine* contact,
then gives that imagination to a robot with no tactile sensors at all.

> ### HapticWAM: Distilling Imagined Touch into a World-Action Model without Inference-Time Tactile Sensing
>
> Paper: **[arXiv:2609.23888](https://arxiv.org/abs/2609.23888)** ·
> [PDF](https://arxiv.org/pdf/2609.23888) ·
> Models and data: **[HapticWAM — ICRA 2027](https://huggingface.co/collections/armteam/hapticwam-icra-2027-6ab234bdfe24c383b5f9fb57)** on the Hugging Face hub
>
> Submitted to the **IEEE International Conference on Robotics and Automation (ICRA) 2027**.
> If you use this code, the checkpoints or the datasets, please [cite the paper](#citation).

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

The full specification is [pipeline.md](pipeline.md); the module-by-module map is
[docs/code_structure.md](docs/code_structure.md).

> HapticWAM was developed under the working name PHANTOM until 2026-09-14; the Python package and
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

Every weight and every episode is public on the Hugging Face hub under
[`armteam`](https://huggingface.co/armteam), collected in
**[HapticWAM — ICRA 2027](https://huggingface.co/collections/armteam/hapticwam-icra-2027-6ab234bdfe24c383b5f9fb57)**.

### Models

Four model repos. Each carries an `index.jsonl` listing every file with its size and LFS
sha256, and each is **Apache-2.0**.

| Repo | Size | What it is |
|---|---|---|
| [`armteam/hapticwam-teacher`](https://huggingface.co/armteam/hapticwam-teacher) | 2.0 GB | the tactile-input **teacher** — the model that sees the fingertip pads and is distilled away. Also carries `text_embeddings.pt`, the Cosmos prompt-embedding cache the training and deploy scripts need. |
| [`armteam/hapticwam-student`](https://huggingface.co/armteam/hapticwam-student) | 4.5 GB | the distilled **pad-free student** — the model that actually runs on the rig with no tactile sensor. Plus the `nowrist` sensor-free ablation arms. |
| [`armteam/hapticwam-baselines`](https://huggingface.co/armteam/hapticwam-baselines) | 12.5 GB | the three comparison policies — pi0.5 expert fine-tune, Diffusion Policy, X-VLA — with the step-selection sweeps that chose their steps. |
| [`armteam/hapticwam-ablations`](https://huggingface.co/armteam/hapticwam-ablations) | 26 GB | every training arm that is **not** deployed (world-model-loss study, multitask, no-distillation controls, the v5 lineage, the pi0.5 60k resume) and the complete, canonical evaluation sweeps. |

#### Which checkpoint the paper deploys

Each repo holds several rungs of each run. These are the exact files behind the reported
numbers — pick anything else and you are not reproducing the paper:

| Role | Repo | Path |
|---|---|---|
| **teacher** (the `teacher` arm in the rig tables) | `hapticwam-teacher` | `teacher_v6_simft/teacher_002000.pt` |
| base teacher before the sim fine-tune (rig arm **A**) | `hapticwam-teacher` | `teacher_v6/teacher_020000.pt` |
| **student** (pad-free, 1 000 distillation steps) | `hapticwam-student` | `hid_simft/student_001000.pt` |
| **pi0.5** baseline | `hapticwam-baselines` | `pi05_phantom_expert_v1/020000/pretrained_model/` — step 20 000, **not** the 60k resume (that one is in `hapticwam-ablations/pi05_phantom_expert_v1_resume60k/`) |
| **Diffusion Policy** baseline | `hapticwam-baselines` | `diffusion_100k/pretrained_model/` |
| X-VLA baseline | `hapticwam-baselines` | `xvla_20k/pretrained_model/` |

#### Get the deployed student in three lines

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("armteam/hapticwam-student", "hid_simft/student_001000.pt")
print(ckpt)   # -> pass to: python -m phantom.scripts.run_deploy --system student --ckpt <ckpt>
```

```bash
hf download armteam/hapticwam-student hid_simft/student_001000.pt --local-dir runs/hid_simft
python -m phantom.scripts.run_deploy --system student \
    --ckpt runs/hid_simft/hid_simft/student_001000.pt --task whiteboard \
    --hardware configs/hardware.nuc.mock.yaml --episodes 1     # mocked, no hardware
```

#### …and the teacher

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("armteam/hapticwam-teacher", "teacher_v6_simft/teacher_002000.pt")
text = hf_hub_download("armteam/hapticwam-teacher", "text_embeddings.pt")  # prompt cache, required
```

```bash
hf download armteam/hapticwam-teacher teacher_v6_simft/teacher_002000.pt --local-dir runs/teacher_v6_simft
hf download armteam/hapticwam-teacher text_embeddings.pt --local-dir data/phantom-episodes/tasks
```

(`hf` is the current Hugging Face CLI; on older installs the same commands are
`huggingface-cli download …`.)

### Datasets

Six dataset repos, all **CC-BY-4.0**.

| Repo | Size | What it is |
|---|---|---|
| [`armteam/hapticwam-teleop-dataset`](https://huggingface.co/datasets/armteam/hapticwam-teleop-dataset) | 102.7 GB | **the training corpus** — 1,115 teleoperated episodes with fingertip tactile, packed as one `.tar.zst` per task, plus the manifests and `norm_stats.json`. Start here. |
| [`armteam/hapticwam-teleop-raw`](https://huggingface.co/datasets/armteam/hapticwam-teleop-raw) | ~112 GB | the same teleoperation as loose, as-recorded sessions (`tasks/`, `archive/`, `collect/`, `manifests/`) — provenance for the packed corpus above, and a superset that also holds pre-cleanup takes. |
| [`armteam/hapticwam-sim-episodes`](https://huggingface.co/datasets/armteam/hapticwam-sim-episodes) | 82.7 GB | Isaac Sim expert episodes (`sim_expert_20260912/`, `sim_expert_20260914/`) as per-trial tars with an `index.jsonl`. These drive the sim fine-tune that produces the deployed teacher. |
| [`armteam/hapticwam-rig-episodes`](https://huggingface.co/datasets/armteam/hapticwam-rig-episodes) | 24.3 GB | the **closed-loop rig takes the paper's numbers are computed from** — `20260915_experiment/` is the analysis set, with `_extra` and `_superseded` kept for provenance. |
| [`armteam/hapticwam-rollouts`](https://huggingface.co/datasets/armteam/hapticwam-rollouts) | 50 GB | policy-driven (not teleoperated) rollouts: the DAgger rounds and the earlier deploy days, as per-day archives plus the round-3 manifest. |
| [`armteam/hapticwam-results`](https://huggingface.co/datasets/armteam/hapticwam-results) | 1.05 GB | result media archived out of this repository — videos, plots, frame dumps, per-trial evidence — with a `MANIFEST.tsv` mapping each file back to its old `docs/results/` path. |

How they fit together: the teacher trains on **teleop-dataset**, is sim-fine-tuned with
**sim-episodes** and **rollouts**, is distilled into the student, and both are then evaluated
on the rig into **rig-episodes**; **teleop-raw** is the provenance of the packed corpus and
**results** holds the media.

`tools/provision_v5.sh` and `tools/provision_distill.sh` pull a training box's dataset and
checkpoints from the hub with nothing but an `HF_TOKEN`.

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
  fixtures/         test fixtures and the frozen study inputs the sim tooling reads
tools/              rig launchers, provisioning, hub upload, sim and analysis tooling
docs/               the documentation set below
```

## Documentation

| Doc | Read it when |
|---|---|
| [docs/code_structure.md](docs/code_structure.md) | you want the module-by-module map and the design rules |
| [docs/launch_guide.md](docs/launch_guide.md) | you need the exact command for any stage, on any machine |
| [docs/training_playbook.md](docs/training_playbook.md) | running pretrain → teacher → HID → DAgger → HID-S |
| [docs/inference.md](docs/inference.md) | loading a trained teacher and running it, mocked or on the rig |
| [docs/deployment_runtime.md](docs/deployment_runtime.md) | deploying on the robot; safety layer; latency budget |
| [docs/tactile_prediction_error.md](docs/tactile_prediction_error.md) | you need the TPE metric definition and how it is computed |
| [docs/sensor_sdk.md](docs/sensor_sdk.md) | anything touches the DM-Tac sensors or the `dmrobotics` SDK |
| [docs/hardware_bench_day1.md](docs/hardware_bench_day1.md) | the sensors/arm arrive — fills the `BENCH:` fields |
| [docs/data_collection_sop.md](docs/data_collection_sop.md) | collecting the teleop dataset |
| [docs/data_collect_app.md](docs/data_collect_app.md) | running the `phantom.data_collect` operator app |
| [docs/pi05_baseline.md](docs/pi05_baseline.md) | exporting, fine-tuning and deploying the pi0.5 baseline |
| [docs/rig_session_v5.md](docs/rig_session_v5.md) | you are about to run the physical rig |

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

Code and model weights: **Apache-2.0** — see [LICENSE](LICENSE).
Datasets on the hub (`hapticwam-teleop-dataset`, `-teleop-raw`, `-sim-episodes`,
`-rig-episodes`, `-rollouts`, `-results`): **CC-BY-4.0**.

## Citation

```bibtex
@article{sannikov2026hapticwam,
  title   = {{HapticWAM}: Distilling Imagined Touch into a World-Action Model
             without Inference-Time Tactile Sensing},
  author  = {Sannikov, Mikhail and Mikhalchuk, Ilya and Gubernatorov, Konstantin
             and Kovalev, Petr and Oluwatobi, Ogunwoye Faith and Tsetserukou, Dzmitry},
  journal = {arXiv preprint arXiv:2609.23888},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.23888}
}
```

