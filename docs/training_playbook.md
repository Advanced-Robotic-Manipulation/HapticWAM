# Training playbook

The four training programs (pipeline.md §4–§7), what each trains, what it
consumes and produces. All are plain-PyTorch with torchrun DDP; every one
smoke-runs anywhere with `--tiny --synthetic --max-steps 2`.

Checkpoint format (all programs): never re-saves the frozen 2B base —
`{lora, phantom_modules, ema, config snapshots (hardware shapes asserted on
load), norm_stats, optimizer, step}`.

## NEW — choosing the compute target (`configs/compute.yaml`)

Training (and only training) is retargeted between machines by ONE file: set
`target:` in `configs/compute.yaml`, or per-machine in the gitignored
`configs/compute.local.yaml` (e.g. a single line `target: h100x8`), or one-off
with `--compute <profile|yaml>` on any train program. Precedence per value:
dataclass defaults < profile < explicit CLI flags.

| target | you run | dtype | teacher/HID per-GPU batch×accum | effective batch |
|---|---|---|---|---|
| `rtx5090` (default) | `python -m phantom.train.<prog> …` | legacy rule | 1×8 | 8 — **bit-identical to before** |
| `h100x8` | `torchrun --nproc_per_node 8 -m phantom.train.<prog> …` | bf16 | 1×1 (×8 ranks) | 8 |
| `a100x8` | `torchrun --nproc_per_node 8 -m phantom.train.<prog> …` | bf16 | 1×1 (×8 ranks) | 8 |

Effective batch is preserved on purpose — no automatic lr scaling; override
lr/warmup in the profile explicitly if you scale batch. Under torchrun the
programs shard data with a `DistributedSampler` (per-epoch reshuffle), build
each rank's model on its own GPU, and checkpoint from rank 0 only.
`pretrain_tactile` gets batch 4/GPU via the profile's `programs:` block
(4×8 = the same effective 32) and stays fp32. Note: resuming a 5090 checkpoint
on 8 GPUs works (weights + rank-0 optimizer state) but shifts the effective
schedule semantics — prefer finishing a run on the target it started on.

---

## (1) `phantom.train.pretrain_tactile` — contact-play SSL

| | |
|---|---|
| Trains | `TactileFieldEncoder` (~5 M conv over the 8-ch field stack) |
| Objectives | masked-patch reconstruction (all 8 ch) + cross-channel: predict the 5 force channels from displacement-only input (free native supervision) |
| Data | tactile keyframes from contact-play episodes (`ContactPlayDataset`) |
| Output | `runs/tactile_pretrain/tactile_encoder_pretrain.pt` |
| Scale | single GPU, ~30 k steps, batch 32 |

## (2) `phantom.train.train_teacher` — teacher PHANTOM

| | |
|---|---|
| Frozen | 2B DiT base, Wan VAE, text embedding |
| Trains | LoRA r=16 on the DiT (~30–60 M) + HHT (~10 M) + ACC (~2 M) + ACE heads + frame-type embeddings + per-block bias λ — **~60–120 M total** |
| Objective | grouped RF: λ_a·MSE(v)[ACTION] + λ_v·MSE(v)[VIDEO] + λ_c·heteroNLL(x0)[CONTACT] + λ_w wrist + λ_evt event CE + ACC gate BCE (weights in `PhantomModelConfig.loss`) |
| Data | all 5 tasks co-trained; `WindowSampler` windows; `norm_stats.json` beside the data (rerun `dump_norm_stats` after unit changes — raw force channels can be O(10³) in SDK units and the stats absorb that) |
| Init | `--tactile-pretrain <program-1 ckpt>` |
| Scale | single 5090 (default `rtx5090` target) or 8×H100 / 8×A100 (`configs/compute.yaml` target `h100x8`/`a100x8` + torchrun), batch 1–2/GPU + grad accum, bf16, activation checkpointing (cosmos SAC is on by default). Teacher sequence is 4160 tokens at full res — measure VRAM with `smoke_test --synthetic --device cuda` first |
| Output | `runs/teacher/<run>/teacher_XXXXXX.pt` |

ACC self-anticipation during training defaults to `"gt_noised"` (GT package +
noise stands in for the previous replan). For the final headline runs and the
lead-time evaluation switch `AccConfig.self_anticipation="two_pass"`.

## (3) `phantom.train.distill_hid` — HID distillation

| | |
|---|---|
| Teacher | program-2 checkpoint, frozen, eval mode |
| Student | same classes with `student=True` layout (no OBS_GEL/OBS_MECH frames), initialized from the teacher's shared weights (teacher-only keys dropped automatically) |
| Losses | (i) trajectory distillation: student CONTACT x0 → teacher's imagined package, weighted w_τ = s_τ·c_τ (saliency from teacher event probs, confidence from teacher σ); (ii) soft event KL vs teacher; (iii) behavior: ACTION-frame velocity matching at shared (x_t, t) with identical noise on shared groups; (iv) GT grounding on real actions/video/wrist + ACC auxiliaries |
| Data | the same demo windows (windows still carry tactile TARGETS — the rig's sensors label; the student model just never receives tactile INPUTS) |
| Flags | `--dagger-round k --extra-data <rollout roots>`; `HIDConfig(feature_align=True)` for the ablation row |
| Output | `runs/hid/<run>_rK/student_XXXXXX.pt` |

## (3b) `phantom.train.dagger_driver` — DAgger rounds (mandatory, ×2)

1. Collect ~50 student rollouts/task on the rig
   (`scripts/run_dagger_round`) — full sensor streams are recorded because the
   rig still wears the sensors.
2. `dagger_driver --round k` → offline teacher relabeling
   (`dagger/relabel.py`, writes `<episode>/relabels/teacher.pt`) → manifest →
   re-invokes `distill_hid` with rollouts merged.

## (4) `phantom.train.finetune_hids` — HID-S safety fine-tune (optional row)

| | |
|---|---|
| Start | post-HID student; a frozen copy is the KL-leash reference |
| Reward | r = −λ_F·max(0, peak_normal_force − τ_obj); recorded force when calibrated, cumulative-depth fallback otherwise (automatic via the config sentinel) |
| τ_obj | auto-calibrated per task: `tau_quantile` (default 0.93) of per-episode peak normal force over SUCCESSFUL episodes — the dataset defines "gentle enough" |
| Update | advantage-weighted regression exp(A/β) on the ACTION-frame RF loss + velocity-MSE leash to the frozen reference at shared (x_t, t) |
| Scale | small LR (1e-5), ~5 k steps |

## Order of operations

```
bench → contact play → (1) pretrain_tactile
      → teleop demos → dump_norm_stats → postprocess
      → (2) train_teacher
      → (3) distill_hid            (round 0, offline)
      → rollouts → (3b) dagger_driver --round 1
      → rollouts → (3b) dagger_driver --round 2
      → (4) finetune_hids          (optional)
      → eval campaign
```

## Comparative systems (pipeline.md §8) — how each is produced

| System | Recipe |
|---|---|
| Teacher PHANTOM | program (2) |
| HID student (ours) | programs (3)+(3b) |
| Vision-only WAM | program (2) with a layout that drops all tactile+F/T inputs (custom `PhantomModelConfig` variant) |
| No-distillation control | student-layout model trained with program (2)'s objective from scratch (run `train_teacher` with `student=True` build — critical control) |
| Drop-tactile-add-nothing | program (3) with the wrist-F/T path also removed |

---

## Launch checklist (2026-08-03 audit — do not skip)

A pre-flight audit before the first real run found defects that would each
have wasted the full multi-day run. They are fixed in code, but two of them
depend on HOW the run is launched, so any machine must follow this:

```bash
python -m phantom.train.train_teacher \
    --data <root>/tasks \
    --hardware configs/hardware.nuc.yaml \   # the config the DATA was recorded under
    --run-name <name> --max-steps 20000 \
    --device cuda --acc-two-pass             # required for the RQ2 gate lead-time
```

* `--hardware` must name the config the episodes were **recorded** under, not
  the repo default. Training under the wrong one changes window semantics and
  produces a checkpoint the rig rejects on load (wrist window 125 vs 31). The
  run now HARD-FAILS on config drift; `--allow-config-drift` overrides.
* `--split` defaults to `train`, which honours `manifests/all.jsonl`. The
  held-out `val` episodes are evaluated every `eval_every` steps — watch the
  `EVAL step N val_*` lines, they are the only signal that the run is healthy.
* Deliberate-failure demos (`<task>_fail`, or tagged `deliberate_failure`, or
  `success=False`) are excluded from ACTION supervision automatically and keep
  supervising contact/event/gate. Do not "clean" them out of the data root.
* `rope_enable_fps_modulation` stays **False**: the released base was trained
  without it, and enabling it drives the frozen backbone off its temporal
  geometry.
