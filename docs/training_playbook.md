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

Selecting a multi-GPU profile **without** `torchrun` is now fatal
(`ComputeProfile.check_world`): `h100x8` on one process is per-GPU batch 1 ×
accum 1 = effective batch 1, i.e. a rented 8-GPU node silently training the
wrong recipe. `PHANTOM_ALLOW_WORLD_MISMATCH=1` overrides it for a deliberate
single-GPU debug run. Validation loaders are built unsharded
(`make_loader(..., shard=False)`): only rank 0 evaluates, so a
`DistributedSampler` made `val_*` a shuffled, `drop_last` 1/world sample.

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

## (2) `phantom.train.train_teacher` — teacher HapticWAM

| | |
|---|---|
| Frozen | 2B DiT base, Wan VAE, text embedding |
| Trains | LoRA r=16 on the DiT (~30–60 M) + HHT (~10 M) + ACC (~2 M) + ACE heads + frame-type embeddings + per-block bias λ — **~60–120 M total** |
| Objective | grouped RF: λ_a·MSE(v)[ACTION] + λ_v·MSE(v)[VIDEO] + λ_c·heteroNLL(x0)[CONTACT] + λ_w wrist + λ_evt event CE + ACC gate BCE (weights in `PhantomModelConfig.loss`) |
| Data | all 5 tasks co-trained; `WindowSampler` windows; `norm_stats.json` beside the data (rerun `dump_norm_stats` after unit changes — raw force channels can be O(10³) in SDK units and the stats absorb that) |
| Init | `--tactile-pretrain <program-1 ckpt>` |
| Scale | single 5090 (default `rtx5090` target) or 8×H100 / 8×A100 (`configs/compute.yaml` target `h100x8`/`a100x8` + torchrun), batch 1–2/GPU + grad accum, bf16, activation checkpointing (cosmos SAC is on by default). Teacher sequence is 4160 tokens at full res — measure VRAM with `smoke_test --synthetic --device cuda` first |
| Output | `runs/teacher/<run>/teacher_XXXXXX.pt` — on a rental, egress them as they land: `HF_TOKEN=… python tools/upload_run_ckpts.py runs/teacher/<run> --log train_<run>.log` (idempotent, skips what is already on the hub; the destination folder is the run name, so never reuse `teacher_v5_batch0822`) |

ACC self-anticipation during training defaults to `"gt_noised"` (GT package +
noise stands in for the previous replan). For the final headline runs and the
lead-time evaluation switch `AccConfig.self_anticipation="two_pass"`.

### FT-A — the recommended objective bundle for the next teacher fine-tune

The P5, P6 and P7 findings of 2026-08-28 are all
*objective* defects: the model is fine, what it is asked to minimise is not.
Each is a flag, each defaults OFF, and with all of them off the run is
bit-identical to v4/v5 — so the shipped checkpoints stay reproducible.

```bash
    --contact-nll-beta 0.5 \
    --action-noise-per-strip --no-action-t-max-of-two \
    --ema-decay 0.995 --cond-dropout 0 \
    --acc-two-pass
```

| flag | finding | what it changes |
|---|---|---|
| `--contact-nll-beta B` | P5 | β-NLL (Seitzer 2022): the contact term is multiplied by `sigma^(2B)` **detached**. The trunk's contact gradient is `2r/sigma^2`, and the v4/v5 logs pin `log sigma ≈ −2.35` (`1/sigma^2 ≈ 110`), so the LoRA — the only capacity that can route observations into ACTION tokens — is optimised as a contact-field forecaster and the action objective trains at ~1/10–1/50 rate. β=1 hands the trunk exactly the plain-MSE gradient; β=0.5 halves the exponent and keeps some heteroscedastic weighting. σ stays calibrated for the speed governor either way. |
| `--contact-nll-detach-weight` | P5 | The *alternative* to β-NLL, not a companion: σ head on the detached residual, trunk on plain MSE. Pick one. |
| `--no-wrist-region-mse` | P5 | Drops the `λ_w` term. Packed channel 15 is also the NLL's `wrist` sigma group, i.e. supervised twice. Not in the bundle — it changes the loss *scale* as well as the balance; enable it only in an explicit ablation. |
| `--contact-self-forcing` *(NOT in the recommended bundle — see below)* | P7 | The ACTION frames are denoised alongside the model's **own** predicted contact package (the `--acc-two-pass` inner sample, already computed — zero extra forward passes) instead of the co-noised GT future, which is 98% decodable at the ACTION head's median training `t=0.90`. GT remains the CONTACT loss target, so only the network *input* changes. Needs `--acc-two-pass`; the run hard-fails without it. **Removed from the recommended FT-A bundle 2026-08-30**: the corrected E9 premise test shows the "98% decodable GT" measurement conflated pinning with content — pinning the CONTACT frames to *zeros* hurts commit as much as pinning to GT (0.58 vs 0.65, real 0.94), so E9 provides no exposure-bias gap and no offline evidence that self-forcing helps. The flag remains available for a controlled ablation; do not spend the primary FT-A run on it. |
| `--action-noise-per-strip` | P6 | `eps_action = ActionPacker.pack(randn(B,H,A))`: one noise draw per action *value* instead of 160–192 i.i.d. latent cells whose strip mean is a structured 0.072–0.079σ offset. Honoured by `training_step` **and** `sample()` — deploy must draw the noise the model was trained to denoise. The checkpoint records the flag and `run_deploy`/`replay` rebuild from it. |
| `--no-action-t-max-of-two` | P6 | Drops the max-of-two ACTION timestep. Under max-of-two the ACTION frames see median `t=0.896` and `P(t<0.556)=0.7%`, while the two Euler steps that actually resolve the chunk run at `t≤0.556` — the head is barely trained where it is read. |
| `--ema-decay 0.995` | — | At 0.999 the EMA averages over ~1000 steps, i.e. a third of a 3000-step fine-tune, so the deployed artifact lags the fine-tune it paid for. |
| `--cond-dropout 0` | — | v5 trained at 0.1; on a short fine-tune 10% of the gradient goes to the unconditional branch, which nothing downstream uses unless guidance is on. Set it back to 0.1 for a from-scratch run. |

`--init-weights` warns (rather than fails) when these flags differ from the
checkpoint's — changing them *is* the fine-tune — while `--resume`, which
continues one run, still refuses any difference. Everything else
(`rope_time_mode`, `acc.self_anticipation`, …) still hard-fails on both.
(Until 2026-08-30 that was only half true: the P10B assert inside
`load_phantom_checkpoint` ran *before* the tolerant check and killed the whole
bundle at load — `--cond-dropout 0` alone was enough. `--init-weights` now
passes `FINETUNE_MUTABLE_MODEL_FIELDS` down as `tolerate_model_fields`;
`tests/test_fixnow_train_0830.py` runs the documented bundle through
`train_teacher.main` end to end so the CLI itself is covered.)

`--event-band-weight` is persisted in the checkpoint's `configs.train` and
re-applied on `--resume`: a spot-instance kill used to bring the term back at
`mc.loss.event` (0.5) for the rest of the run, unrecorded. Passing a
*different* value on a resume is refused.

Two D8 weighting knobs, both default-off: `EpisodeMeta.weight` (meta.json,
default 1.0) multiplies a window's `action_weight` per episode — failure demos
stay at 0 — and `--commit-band-weight` multiplies it again for windows
anchored inside the pre-close commit band. `group_velocity_mse` normalises by
the batch's weight sum, so these are *relative* weights within a batch.

Verify before spending the run: the per-term LoRA grad-norm probe
(`scratchpad/research/gradprobe.py`, ~10 GPU-min on the real v5 checkpoint) —
per-term norms should land within ~10× of each other instead of the current
>99% squared-norm share on `contact_nll`.

### The world-model ablation — `--loss-video` and `--video-attend`

Architecture note (09-12) §6 item 2: today the 3 imagined VIDEO_GEN frames
buy shared LoRA weights and nothing at inference — `0.1·video_v_mse` shapes the
adapter, the structural mask forbids ACTION/CONTACT from attending them, and
`drop_video` is off at deploy. Two flags make that measurable, both default to
the shipped behaviour:

| flag | asks | mechanism |
|---|---|---|
| `--loss-video 0` | does the video objective help the actions **at all**? | overrides `mc.loss.video`. At 0 the term is *skipped* (not multiplied by zero, so no gradient reaches the video branch) while the VIDEO_GEN frames stay in the layout — identical token count, attention mask, RoPE and cond mask, so the arm differs from the control in the objective alone. `video_v_mse` is still logged, so the arm stays readable. |
| `--video-attend` | does letting the actions **read** the imagined future help? | `SequenceLayout.structural_attn_bias` stops masking CONTACT/ACTION queries from VIDEO_GEN keys, at training and at inference identically (one code path: `attention_bias.structural_bias_tokens`). The frames stop being droppable, so `rf.sample`, `tools/terminal_eval.py` and `run_deploy` all **refuse** `--drop-video` for such a checkpoint, and `PhantomModelConfig` refuses `video_attend` + `drop_video_at_inference` outright. |

Both are recorded in the checkpoint's `configs.model` (`video_attend`,
`loss.video`) and restored by `PhantomModelConfig.from_dict`, so eval, replay
and deploy rebuild the arm they are scoring. `--init-weights` treats them as
fine-tune-mutable (warn); `--resume` still refuses any difference, so resuming
run A must re-pass `--loss-video 0`.

**Budget-matched trio, 5,000 steps each from the v6 teacher.** Use
`--init-weights`, never `--resume`: a resume would restore v6's optimizer
moments, its cosine schedule and step 20,000, which is a continuation of v6
rather than three comparable fine-tunes. `--init-ema` (default) starts all
three from the EMA weights the rig and `terminal_eval` actually score. Every
flag outside the one under test is identical across A, B and C — same steps,
data, split, seed, lr, warmup, EMA decay and batch×accum.

```bash
# 0) read v6's own recipe off the checkpoint — the --init-weights drift guard
#    hard-fails on any behavioural mismatch and names the flags to re-pass
python - <<'PY'
import json, torch
c = torch.load("runs/teacher_v6/teacher_020000.pt", map_location="cpu",
               weights_only=False)["configs"]
print(json.dumps({"model": c["model"], "train": c.get("train")},
                 indent=1, default=str))
PY

# 1) the shared block: v6's recipe (fill the objective knobs from the dump
#    above; wrench_baseline_rows=8 is v6's)
V6=runs/teacher_v6/teacher_020000.pt
COMMON="--data $DATA --hardware configs/hardware.nuc.yaml --allow-config-drift \
  --init-weights $V6 --max-steps 5000 --split train \
  --wrench-baseline-rows 8 --grasp-frac 0.3 --photo-aug 1.0 --acc-two-pass \
  --lr 2e-5 --lr-new-modules 6e-5 --warmup-steps 150 --ema-decay 0.995 \
  --ckpt-every 1000 --eval-every 1000 \
  --batch-size 4 --grad-accum 2 --num-workers $NW --device cuda"

# A — no video objective (frames kept, layout unchanged)
python -m phantom.train.train_teacher $COMMON \
    --run-name teacher_v6_ft_noVideoLoss --loss-video 0

# B — world-action coupling (actions attend the imagined future)
python -m phantom.train.train_teacher $COMMON \
    --run-name teacher_v6_ft_videoAttend --video-attend

# C — matched control: the same fine-tune with the defaults
python -m phantom.train.train_teacher $COMMON \
    --run-name teacher_v6_ft_control
```

Read the three with `tools/terminal_eval.py` (C is the reference, not v6
itself — a 5k fine-tune moves the model on its own). B must be evaluated
**without** `--drop-video`; A and C may be run both ways, and A's
`--drop-video` delta is the direct measurement of what the video loss was
buying. On one H100 at effective batch 8 a teacher step is ~7 s
(measured), so each arm is ≈ 9.7 GPU-h and the trio ≈ 29 GPU-h; the same
ablation from scratch prices at ≈ 35 h per arm.

## (3) `phantom.train.distill_hid` — HID distillation

| | |
|---|---|
| Teacher | program-2 checkpoint, frozen, eval mode |
| Student | same classes with `student=True` layout (no OBS_GEL/OBS_MECH frames), initialized from the teacher's shared weights (teacher-only keys dropped automatically) |
| Losses | (i) trajectory distillation: student CONTACT x0 → teacher's imagined package, weighted w_τ = s_τ·c_τ (saliency from teacher event probs, confidence from teacher σ); (ii) soft event KL vs teacher; (iii) behavior: ACTION-frame velocity matching at shared (x_t, t) with identical noise on shared groups; (iv) GT grounding on real actions/video/wrist + ACC auxiliaries |
| Data | the same demo windows (windows still carry tactile TARGETS — the rig's sensors label; the student model just never receives tactile INPUTS) |
| Flags | `--split train` (**default**, since 2026-08-30 — both HID programs used to index every episode, val included), `--dagger-round k --extra-data <rollout roots>`; `HIDConfig(feature_align=True)` for the ablation row |
| Output | `runs/hid/<run>_rK/student_XXXXXX.pt` |

## (3b) `phantom.train.dagger_driver` — DAgger rounds (mandatory, ×2)

1. Collect ~50 student rollouts/task on the rig
   (`scripts/run_dagger_round`) — full sensor streams are recorded because the
   rig still wears the sensors.
2. `dagger_driver --round k` → offline teacher relabeling
   (`dagger/relabel.py`, writes `<episode>/relabels/teacher.pt`) → manifest →
   re-invokes `distill_hid` with rollouts merged.

**Run `tools/rederive_rollout_actions.py` on every rollout root first.** A
deploy rollout's `actions` stream is the executor's raw *proposal* on the
governor-warped clock (pre-clamp, pre-rate-limit); training on it imitates
commands the safety layer refused. `intake_recovery.py manifest` now REFUSES a
policy rollout with no `actions_plan.zarr`, and `WindowSampler.build_index`
(the `--extra-data` path, which sees no manifest) warns loudly. The re-derived
gripper channel is the **commanded** aperture, not the measured position — the
measured one is the grasp outcome and would leak it into the action target.

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
| Teacher HapticWAM | program (2) |
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
