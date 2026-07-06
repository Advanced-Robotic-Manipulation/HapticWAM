# Launch guide

Exact commands for every module, grouped by stage. Machines:

- **[ANY]** — any machine with `pip install -e .[dev,train]` (Windows dev box OK)
- **[RIG]** — the recording machine wired to the robot (`pip install -e .[hw]` extra;
  `dmrobotics` SDK installed per Daimon's instructions)
- **[5090]** — the Linux inference box (CUDA torch)
- **[H100]** — the 8×H100 training node(s) (Linux, CUDA torch, NCCL)

Global flags on most entry points: `--hardware <yaml>` (defaults to
`configs/hardware.yaml`), `--synthetic` (generated data), `--tiny` (small CPU
backbone), `--device cuda|cpu`.

---

## 0. Environment

Per-machine manifests live in [../requirements/](../requirements/README.md):

```bash
# 5090 box:   pip install -r requirements/requirements-5090.txt && pip install -e .
# H100 node:  pip install -r requirements/requirements-h100.txt && pip install -e .
# dev box:    pip install -e .[dev,train]
```

The cosmos repo is imported as a library from `configs/paths.yaml::cosmos_repo`
— nothing from it needs to be pip-installed; its pip-installable import-chain
deps are already in the requirements files. On machines without Transformer
Engine / megatron builds, import shims are installed automatically
(`phantom/backbone/compat.py`). Non-pip steps (dmrobotics SDK, realsense udev
rules, SpaceMouse hidapi) are listed in the requirements README.

**Trust boundary:** until `verify_backbone.py` has passed a shim-parity check
against real TE on Linux (below), treat full-checkpoint outputs computed
through the shims as smoke-level only.

## 1. Smoke tests [ANY]

```bash
pytest tests/                                        # unit + tiny integration
python -m phantom.scripts.mock_smoke                 # recording pipeline e2e
python -m phantom.scripts.smoke_test --tiny --synthetic   # model e2e on CPU
```

## 2. Backbone verification [5090 first, then H100]

```bash
# loads the REAL 2B checkpoint; subclass parity must be exact
python -m phantom.scripts.verify_backbone --save-ref ref_shim.pt
# on a machine with real transformer_engine installed:
python -m phantom.scripts.verify_backbone --check-ref ref_shim.pt
# full-checkpoint smoke + VRAM measurement of the EXTENDED layout (week-1 item):
python -m phantom.scripts.smoke_test --synthetic --device cuda
```

## 3. Day-1 hardware bench [RIG]

Follow [hardware_bench_day1.md](hardware_bench_day1.md); the semi-automated
runner:

```bash
python -m phantom.scripts.bench_day1 --items b c a d arm
```

Update `configs/hardware.yaml` with the measured values, set
`meta.bench_verified: true`, then rerun `pytest` + `mock_smoke` (with
`mode.drivers: real` overrides as appropriate).

## 4. Data collection [RIG]

```bash
# contact play (no task, ~2-4 h total) — record with the normal recorder:
python -m phantom.scripts.record_episodes --task contact_play --teleop keyboard

# teleop demos per task (see data_collection_sop.md for quotas/hotkeys):
python -m phantom.scripts.record_episodes --task fragile_grasp --teleop spacemouse

# after each session: derived channels + normalization stats
python -m phantom.scripts.postprocess_episodes --data data/episodes/<date>
python -m phantom.scripts.dump_norm_stats --data data/episodes/<date>
```

Rerun `dump_norm_stats` whenever `tactile.force_unit_to_N` (or any unit scale)
changes — the norm stats are what absorb unit changes for training.

## 5. Training [H100]

```bash
# (1) tactile encoder SSL pretrain (single GPU is fine)
python -m phantom.train.pretrain_tactile --data <contact_play_root>

# (2) teacher PHANTOM
torchrun --nproc_per_node 8 -m phantom.train.train_teacher \
    --data <episodes_root> --tactile-pretrain runs/tactile_pretrain/tactile_encoder_pretrain.pt \
    --run-name teacher_v1

# (3) HID distillation (round 0, offline)
torchrun --nproc_per_node 8 -m phantom.train.distill_hid \
    --teacher-ckpt runs/teacher/teacher_v1/teacher_050000.pt --data <episodes_root>

# (3b) DAgger rounds 1..2 — rollouts happen on the rig first (step 6), then:
python -m phantom.train.dagger_driver --round 1 \
    --teacher-ckpt <teacher.pt> --demos <episodes_root> --rollouts <rollout_root>

# (4) optional HID-S safety fine-tune
python -m phantom.train.finetune_hids --student-ckpt runs/hid/hid_r2/student_030000.pt \
    --data <episodes_root>
```

Every program smoke-runs anywhere with `--tiny --synthetic --max-steps 2`.

## 6. Deployment + DAgger rollouts [5090]

```bash
# teacher with sensors / student without (tactile still records in ALL modes):
python -m phantom.scripts.run_deploy --system teacher --ckpt <teacher.pt> --task fragile_grasp
python -m phantom.scripts.run_deploy --system student --ckpt <student.pt> --task fragile_grasp

# DAgger rollout collection (~50/task/round), then hand off to dagger_driver:
python -m phantom.scripts.run_dagger_round --round 1 --ckpt <student_r0.pt> \
    --tasks fragile_grasp slippery_place --rollouts-per-task 50
```

Dry run without hardware or checkpoints: add `--tiny` and keep
`mode.drivers: mock`.

## 7. Evaluation campaign [5090 + RIG]

```bash
cp configs/eval_campaign.example.yaml configs/eval_campaign.yaml   # fill checkpoints
python -m phantom.scripts.run_eval --campaign configs/eval_campaign.yaml
# aggregation only (tables + ACC lead-time data):
python -m phantom.scripts.run_eval --campaign configs/eval_campaign.yaml \
    --aggregate-only --episode-metrics
```

The ledger CSV is append-only and the runner resumes from it — safe across a
multi-week campaign.

## Ablation flags → pipeline.md §8 mapping

| Ablation | How |
|---|---|
| tactile-as-image (VAE only) vs full HHT | train teacher with `PhantomModelConfig(contact_obs_frames=0)` variant / drop OBS_MECH in a custom layout (RQ1 build-up: gel → +fields → +force channels via `tactile.field_channels` zeroing in the dataset) |
| ACC α=0 (CASA) / α=1 / learned | `AccConfig` + freeze `W_alpha` at ∓∞ logits (small code toggle) |
| rope aligned vs append | `PhantomModelConfig(rope_time_mode=...)` |
| video frames on/off at inference | `--drop-video` on run_deploy / `drop_video=True` in sample() |
| HID weighting uniform/saliency/+confidence | `HIDConfig(saliency_kappa=0, confidence_temp=1e9)` etc. |
| feature-alignment term | `HIDConfig(feature_align=True)` |
| DAgger on/off | manifest with/without rollouts |
| speed governor on/off | `safety.governor.min_scale: 1.0` |
