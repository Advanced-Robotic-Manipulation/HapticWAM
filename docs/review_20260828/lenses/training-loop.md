# PHANTOM review — lens: TRAINING LOOP + EVALUATION (2026-08-28, HEAD 3539988)

Scope: `phantom/train/{train_teacher,common,builder}.py`, `phantom/config/training.py`
(the prompt's `configs/training.py` does not exist; the dataclasses live in
`phantom/config/training.py`), `tools/terminal_eval.py`, `tools/provision_v5.sh`, plus the
code they depend on for the questions asked (`phantom/data/windows.py`, `phantom/model/rf.py`,
`phantom/model/phantom_dit.py`, `phantom/model/acc.py`, `phantom/inference/policy.py`,
`phantom/deploy/{planner,executor,runtime}.py`, `tools/intake_recovery.py`). Everything cited
was read; quick numeric checks were run with the repo venv. Nothing under the repo was modified.

Checks run:
- `pytest tests/test_terminal_eval_close.py tests/test_v5_train_fixes.py tests/test_windows.py` → 15 passed.
- RF timestep mass vs the NFE=5 Euler grid; EMA lag at decay 0.999; cosine-LR values and the
  per-element displacement budget (sum of lr) for the 3k fine-tune vs a 20k from-scratch run.
- `WindowSampler.sample()` on a recorded deploy episode dir (`data/episodes/deploy/20260827/...`):
  the schema is identical to demos (19 zarr streams + `planner_trace.json` + `meta.json`); the
  only blocker on the local mock episode is the 4.7 s minimum stream overlap (the mock has 0 s;
  real rig episodes are 16–31 s).
- The v5 training log / per-checkpoint eval JSONs are on the private hub only (no `HF_TOKEN` in
  this shell, nothing cached locally except `dataset_v3_packed/{norm_stats.json,manifests.tar}`),
  so raw-vs-EMA per-checkpoint numbers could NOT be checked here — see finding 4 for what to look at.

---------------------------------------------------------------------------------------------------

## 0. Bottom line

The training loop is mechanically sound for the run that was done (single GPU, 4×2, fine-tune):
grad accumulation, clipping, fp32-master AdamW, EMA keyed by state_dict names, worker RNG
reseeding, manifest split, label-sanity gate, config-drift refusals — all verified in code and
by the shipped tests. The things that are wrong are *what* the loop is asked to learn and *how*
its output is judged:

1. The policy is conditioned on its OWN previous proposal at deploy but on MEASURED motion in
   training, through two paths (ACC intent MLP and the pretrained action-AdaLN of the DiT).
   Together with persistent noise this is a textbook self-reinforcing "slow mode" — a mechanism
   that matches the rig observation (decelerates, stops high, one seed descends at full speed)
   and that no offline number so far can see.
2. `terminal_eval` samples states from DEMOS. It measures one-step imitation from on-distribution
   states, so by construction it cannot see covariate shift. A rollout-state replay metric can be
   built with ~150 lines and zero new sampler code (design in §3).
3. The 45 under-grasp failure demos are action-weight 0 for the WHOLE episode. If those episodes
   contain the re-open/retry after the failed close, the loop discards precisely the corrective
   supervision the paper says the action head never sees. Verify on the data before the next run.
4. The 3k fine-tune at 1/5 LR moved the LoRA by ≈3 % of the from-scratch displacement budget
   (sum lr = 0.030 vs 1.000), and every evaluated v5 checkpoint is a v4→v5 EMA interpolation
   (61 % v4 at step 500, 5 % at step 3000). "Monotone over 6 checkpoints" is what EMA lag alone
   produces; the reported 17.4 mm was selected on the same 124 episodes it is reported on.
   v5 ≈ v4 is the expected outcome of this recipe, and the rig agrees (0/26 both).

---------------------------------------------------------------------------------------------------

## 1. Findings in detail

### F1 (HIGH) — prev_chunk: training conditions on measured motion, deploy on the previous proposal; the intent path is a velocity-copy shortcut

Evidence:
- Training: `phantom/data/windows.py:277` `prev = self._action_grid(c, t0 - (H - arange(H))/rate)`
  from `STREAM_ACTIONS`, which for demos is Δ-EE derived from MEASURED TCP (docs/STATUS.md:38
  "dataset actions stay Δ-EE derived from measured TCP").
- Deploy: `phantom/inference/policy.py:101-102` `prev = normalize("action", prev_plan.actions)` —
  the previous plan's raw 16-step proposal, before governor time-scaling, chunk blending, the
  kinematic rate limit and the z-floor clamp (`phantom/deploy/executor.py:182-200, 222-230`).
  `phantom/deploy/planner.py:284-290` documents this as "DEFERRED (deliberately, not an
  oversight)".
- Where it enters the network: (a) `phantom/model/acc.py:37,62,84` `intent_B_H_A` → `intent_mlp`
  → the ACC fused embedding (→ gate, event probs, per-block bias λ); (b)
  `phantom/model/phantom_dit.py:145-156, 227-230` the first 12 of the 16 intent actions go through
  the PRETRAINED Cosmos `action_embedder_B_D / _B_3D` and are ADDED to the timestep embedding /
  AdaLN-LoRA of the VIDEO_GEN frames (`use_action_adaln_intent=True`, `phantom/config/model.py:68`).
  The action frames attend to those tokens in every block.
- In smooth teleop demos the next 1.6 s chunk is highly predictable from the previous 1.6 s of
  measured motion, so the cheapest way to lower `action_v_mse` is "continue the previous chunk".
  Nothing in training breaks that correlation (no intent dropout/perturbation; `_null_obs_batch`
  at `rf.py:214-234` explicitly leaves `prev_chunk` untouched during cond-dropout — "it is intent,
  not observation").

Failure scenario (matches the rig): first replan → `prev` = normalized ZERO physical action
(`policy.py:104-110`) = "I was stationary"; the model proposes a gentle start; the executor plays
it 0.9 s late and time-scaled by the governor; the next replan is conditioned on that gentle
proposal (not on what the arm did) AND on proprio that says "slow"; with `reuse_noise` the
chunk is a near-deterministic function of (obs, prev) → the slow mode locks in, descent runs at
33 vs 45–50 mm/s, the demo-like deceleration+close pattern fires at whatever height the arm has
reached. The "one seed descended at full speed" observation is exactly what a shortcut with a
noise-dependent basin looks like.

Fix (cheapest first):
1. Diagnostic, no training (15 min on the 5090, part of the replay tool in §3): at each recorded
   rig replan state, sample with prev_chunk = (i) as deployed, (ii) measured Δ-EE from
   `arm_tcp_pose` over the previous 1.6 s (training semantics), (iii) normalized zeros, (iv) a
   demo-like descending chunk. If (iv)/(ii) recover the descent and (iii) kills it, F1 is confirmed.
   Also on val demo windows: endpoint error with prev_chunk zeroed vs true — a large jump = strong
   reliance.
2. Training (one 3k fine-tune, ~5 h H100): intent augmentation in `WindowDataset.__getitem__`
   (`common.py:444-468`): with p≈0.3 replace `prev_chunk` by normalized zeros (the first-replan
   input) and with p≈0.3 time-scale its Δ-pose channels by U(0.4, 1.6) (keep gripper). This
   decorrelates "next velocity" from "previous velocity" while keeping the intent useful for
   phase. No model-config change, so `--init-weights` drift checks still pass.
3. Deploy: feed the MEASURED previous motion (Δ-EE from the arm ring over the last 1.6 s), not
   `prev_plan.actions` — that is what training saw. (`executor.record_action` at
   `executor.py:207-212` records PLAN steps, so DAgger rollouts inherit the same mismatch; see §3.)

### F2 (HIGH) — `terminal_eval` samples DEMO states only; build a rollout-state replay (design in §3)

Evidence: `tools/terminal_eval.py:118-127` iterates `ds.index` built from the val manifest and
anchors `t0 = t_close - lead`; `make_batch` (96-115) is `WindowSampler.sample(ep, t0)` on the demo
episode. The proprio (`windows.py:265-271`) therefore carries demo `qd`/`tcp_speed`, the camera
frame shows the demo hover, and prev_chunk is the demo's own measured motion. Every quantity that
differs on the rig is held at its demo value. The docs already concede this ("Offline != rig: the
rig decides", rig_session_v5.md:103), yet checkpoint selection (`provision_v5.sh:281`) and the
paper's offline claim rest on it.

Fix: the replay metric of §3. It uses only existing code paths (`WindowSampler.sample` on a deploy
episode dir, `rig_trace_decompose.py:60`'s clock alignment, `terminal_eval.make_batch`).

### F3 (HIGH, verify on data) — failure demos are action-weight 0 for the WHOLE episode

Evidence: `phantom/data/windows.py:395-403` `w["action_weight"] = 0.0 if is_failure_demo(meta)`;
`phantom/data/schema.py:167-171` flags the episode by `success is False`, the
`deliberate_failure` tag, or a `<task>_fail` task name; `phantom/model/ace/losses.py:49-51`
applies it per sample. There is no time-resolved weighting. `WindowDataset._close_time`
(`common.py:409-424`) still finds the close-on-air in those episodes, so `--grasp-frac 0.3`
anchors 30 % of their windows around a failed close that contributes nothing to the action term
(slots wasted, not harmful).

Why it matters: `ur_state` ALREADY contains "did the close succeed" — the Robotiq OBJ channel is
the last element (norm_stats: mean 2.40, std 1.03; 2 = stopped on an object, 3 = closed on
nothing). The CONTEXT's statement that "the action head never conditions on 'did the close
succeed'" is true only because no training window with action weight 1 ever has OBJ=3 after a
close. If the `_fail` episodes contain the operator re-opening and re-grasping after the failed
close, that segment is the ONLY action supervision in the dataset for the rig's actual end-state,
and the current rule throws it away.

Fix: (1) verify with `python tools/rig_trace_decompose.py demos tasks/waffles_fail` (count
gripper re-opens after the first close; ≥2 closes = a recovery segment exists). (2) If yes, make
the weight window-level in `windows.py:403`: weight 0 only when the chunk (t0, t0+1.6] overlaps
[t_failclose − 1.6 s, t_failclose + 0.5 s]; weight 1 elsewhere (and keep `success is False`
episodes at 0 if their failure is not localized). (3) If no recovery segment exists, the 45
episodes are still useful contact negatives but the next collection session should record
"close on air → re-open → descend → grasp" explicitly (10 per task is enough for a fine-tune).

### F4 (MEDIUM) — EMA lag + same-set selection confound the v5 checkpoint story; the fine-tune budget says v5 ≈ v4

Evidence:
- `common.py:162` shadow = current params at construction; `common.py:728` `EMA(base, 0.999)` is
  built AFTER `--init-weights` loaded the v4 EMA; no bias correction in `update` (170). Computed:
  weight still on the v4 init at ckpt 500/1000/1500/2000/2500/3000 = 0.606/0.368/0.223/0.135/
  0.082/0.050. Deploy and `terminal_eval` use EMA by default (`terminal_eval.py:66`,
  rig_session_v5.md:28).
- LR: `make_scheduler` (`common.py:131-137`) cosine → exactly 0 at `max_steps`; with peak 2e-5 and
  warmup 150 the LR at the six checkpoints is 1.93e-5 / 1.59e-5 / 1.08e-5 / 5.5e-6 / 1.5e-6 / 0.
  Σlr over the fine-tune = 0.030 vs 1.000 for the 20k run (per-element Adam displacement budget,
  the same argument `train_teacher.py:71-74` uses). The v5 LoRA is a ≤3 % perturbation of v4.
- Selection: `provision_v5.sh:281` "SELECT the best checkpoint with tools/terminal_eval.py" on the
  124-episode val set; rig_session_v5.md:101-103 reports the same set.

Failure scenario: the "monotone improvement over 6 checkpoints" is exactly what an un-corrected
EMA unrolling from v4 toward a raw model that stopped moving by ~step 700 looks like. The number
17.4 mm is then (a) partially an interpolation artefact and (b) a max over 6 draws on the
reporting set. Neither invalidates the fine-tune, but a reviewer would.

Fix: (1) `eval_ckpt.sh` already produces `te_<tag>_raw.json` — tabulate raw vs EMA per
checkpoint; if raw is flat while EMA improves, say so and report raw (or the final EMA only).
(2) For short fine-tunes use `ema_decay` 0.99–0.995 (horizon 100–200 steps) or bias-correct the
shadow (initialize it at the first `update`, or divide by 1 − d^k). (3) Select on the v4-78
half and report on the new-batch-46 half (or the reverse) — the split already exists in
`manifests/intake_holdout.json`. (4) State plainly that a 1/5-LR 3k fine-tune cannot be expected
to change closed-loop behaviour; the fix must come from conditioning/data (F1, F3, F5), not more
of the same.

### F5 (MEDIUM) — the grasp band strengthens the "slow ⇒ close now" association the rig is failing on

Evidence: `common.py:379,449-455` anchors windows in [t_close − 1.5 s, t_close − 0.2 s] — the
phase where demo TCP speed falls to zero and the gripper closes. The proprio input has demo
velocity (`windows.py:265-271`: q, qd, tcp_pose, tcp_speed, gripper). norm_stats: tcp_speed z std
= 0.066 m/s, so demo descent 45–50 mm/s ≈ 0.7σ and the rig's 33 mm/s ≈ 0.5σ — the model can tell
them apart and, with 30 % of windows in the deceleration band, is rewarded for mapping "slowing
down" → "close next". Both v4 and v5 fail identically on the rig, so this is not the cause of the
failure, but the recipe pushes in the wrong direction unless velocity is decorrelated.

Fix: (1) replay counterfactual (§3): scale/zero the qd and tcp_speed channels of `ur_state` at
rig states and watch the predicted descent. (2) If sensitive: proprio-velocity augmentation in
`__getitem__` (scale channels 6:12 and 18:24 by U(0.3, 1.5), or zero with p=0.3). (3) Anchor the
band by HEIGHT (z − z_close ∈ [40, 150] mm via `arm_tcp_pose`) rather than by time, so the band
also covers windows where the demo still moves at full speed.

### F6 (MEDIUM) — under DDP the val loader is silently sharded AND shuffled; `shuffle=False` is ignored

Evidence: `common.py:510-518`: when `world > 1` a `DistributedSampler(shuffle=True, seed)` is
always built; the `shuffle` argument only flips the DataLoader flag, which is ignored when a
sampler is given; `train_teacher.py:306` requests `shuffle=False` for val. `evaluate()`
(`common.py:555-561`) then strides over a shuffled 1/world shard on rank 0 only; `drop_last=True`
discards the remainder. Not exercised by v5 (single GPU) — the 20k from-scratch run on 8×H100
would be.

Fix: for `shuffle is False` build `DistributedSampler(shuffle=False)` or pass no sampler and let
rank 0 evaluate the whole val set. Add a `dist.barrier()` before `destroy_process_group()`
(`common.py:816-819`) so ranks 1..N do not exit while rank 0 writes the final checkpoint.

### F7 (LOW) — `terminal_eval --max-episodes` default 80 on a task-sorted index

`terminal_eval.py:65,118`: `ds.index[:80]` with `windows_per_episode=1` and a manifest-ordered
index evaluates only the first ~80 of 124 val episodes (the first task folders).
`scratchpad/probe_v5.sh` used the default; `eval_ckpt.sh` used 200 — two v4 baselines on
different subsets. Also `close_step_err` at `lead = chunk_s` has the GT close at step 15/16, so
its range is [−15, +1] and it mostly encodes "closed inside the chunk or not". Fix: default to all
episodes, report per-task medians, and add `pred_close_height_mm` (z at the first predicted close
minus demo z_close) which is the quantity the rig fails on.

### F8 (LOW) — `evaluate()` materialises every val batch; `sampled_eval` uses 8 windows

`common.py:557-559` iterates the whole loader and skips by `i % stride` — 248 window builds
(VAE-scale frames + tactile derive) to use 24. `train_teacher.py:337` `n_windows=8`: the probe
log shows `sampled_mag_ratio=1.24` at step 30 — 8 samples cannot steer anything. Fix: a
`SubsetRandomSampler`/`Subset` for the strided indices; 32–64 sampled windows.

### F9 (LOW) — cosine tied to `cfg.max_steps`; `--resume` semantics; `log_every=1`

`common.py:135` computes progress from `cfg.max_steps`, so resuming with a larger `--max-steps`
jumps the LR discontinuously, and resuming AT `max_steps` trains at LR 0. `train_teacher.py:64`
sets `log_every=1` whenever `--max-steps` is passed (20k lines for a from-scratch run).

### Notes that are NOT findings (verified fine)
- DataLoader determinism: `_seed_worker_rng` (`common.py:525-537`) reseeds per (seed, epoch,
  worker); workers are re-forked each epoch (no `persistent_workers`), so `ds.epoch` set at
  rollover (`common.py:767`) is honoured; `torch.manual_seed` before `iter(loader)`; val uses
  `resample=False` (photo-aug is gated on `resample`, `common.py:457`).
- DDP gradient path: `_StepModule` routes `training_step` through `DDP.forward`
  (`common.py:716-725`), `no_sync` on all but the last micro-batch, clip on the same tensors,
  `find_unused_parameters=True` (needed: the two-pass inner `sample()` is no-grad). Rank noise
  streams are separated (`common.py:745-751`). Test `test_ddp_sync.py` covers it on gloo.
- `Fp32MasterAdamW`: masters built after `--init-weights` load (train_loop line 726), grads
  clipped on live params then copied; weight decay decoupled — at these LRs negligible.
- RF timesteps: training t (logit-normal, shift 5) has P(t > 0.87) = 0.38, with
  `action_t_max_of_two` 0.62; the NFE=5 grid is [1, .952, .882, .769, .556, 0] — the action band
  is trained where sampling looks. Consistent between v4 and v5 (drift check would refuse otherwise).
- `--init-weights` refuses model-config drift and norm-stats drift (`train_teacher.py:206-214,
  237-253`); `--acc-two-pass` in the fine-tune implies v4 was two-pass too.
- Failure demos are excluded from the val hold-out (`intake_recovery.py:105-111`), correct.

---------------------------------------------------------------------------------------------------

## 2. Answers to the specific questions

**LR/EMA/cosine for the 3k fine-tune.** Warmup 150 → peak 2e-5 (LoRA) / 6e-5 (new modules) →
cosine to 0 at 3000. Σlr = 0.030; EMA(0.999) horizon ≈ 1000 steps, uncorrected, initialized at
v4-EMA. Consequences: F4. Recommended: decay 0.99–0.995 or bias correction; do not end the cosine
at exactly 0 if you intend to `--resume` (F9).

**For a 20k from-scratch run.** Peak 1e-4/3e-4, warmup 500, cosine to 0, EMA horizon 1k = 5 % of
the run, Σlr = 1.0. Fine. On 8×H100 (`h100x8`: 1×1×8 = effective 8) the LR is not rescaled on
purpose (compute.yaml:16-17) — correct since the effective batch is preserved. Mind F6 (val
shard) before launching, and `check_world` only WARNS on a profile/launch mismatch
(`compute.py:83-91`): launching `h100x8` without torchrun silently trains at effective batch 1.

**Grasp-frac vs failure demos.** Anchoring fires on the failed close (gripper-only rule), the
windows carry weight 0, so ≈4 % of the action-loss slots are empty; the normalization
`(w·l).sum()/w.sum()` keeps the loss unbiased. The real issue is F3 (whole-episode zeroing).

**DataLoader determinism / DDP.** Verified fine except F6 (val sharding/shuffle) and the missing
end-of-run barrier.

**Checkpoint selection EMA vs raw.** Everything downstream (deploy, terminal_eval, `--init-ema`)
uses EMA. Because the EMA is an uncorrected blend with v4 for the whole 3k run, per-checkpoint
EMA comparisons are not comparisons of training progress (F4). Compare raw vs EMA; if raw is
better at step 1000–1500 than EMA at 3000, that says the fine-tune plateaued early.

**Is terminal_eval a valid proxy?** As a regression guard on imitation quality: yes. As a proxy for
the rig failure: no (F2). Note also that its 20.7→14.9 mm "new-batch holdout" is the last whole
session(s) per task of a two-day, one-operator batch (`intake_recovery.py:101-123`) — same
lighting/placements as the training sessions, so it is optimistic.

**Anything in the v5 recipe likely to have REDUCED closed-loop commitment?** Nothing in the recipe
is a smoking gun, and the rig shows v4 and v5 failing the same way; but three ingredients push
toward less commitment rather than more: the time-anchored deceleration band (F5), the
unchanged intent conditioning (F1), and `--event-band-weight 0` (harmless: v4 never learned that
band either). `--photo-aug 1.0` only makes the camera less trustworthy, which shifts reliance
toward proprio/intent — i.e. toward the velocity shortcuts. Unverified but worth one replay run:
train with photo-aug OFF and see whether the replay descent changes.

---------------------------------------------------------------------------------------------------

## 3. The cheapest offline metric that CAN see covariate shift — rollout-state replay

Why it is cheap: a deploy episode directory has the SAME schema as a demo (verified on
`data/episodes/deploy/20260827/ep_teacher_smoke_1787827101_000`: `actions, arm_q, arm_qd,
arm_tcp_pose, arm_tcp_speed, arm_ft, camera_scene_color, gripper, tactile_{left,right}_{fields_ds,
keyframes, infer_img, wrench, area}.zarr`, `meta.json`, `planner_trace.json`). `WindowSampler.sample
(ep, t0)` runs on it unchanged (it only needs ≥4.7 s of stream overlap, `windows.py:170-172`, which
every real rig episode has). `planner_trace.json` is a list of replans with `t` (perf_counter),
`latency_s`, `gate`, `p_evt`, `sigma`, `accepted`, `actions` (16×7, denormalized), `diag{nfe,
guidance}` (`planner.py:262-271`); `meta.clock_calibration.offset` maps `t` to the zarr master
clock (`rig_trace_decompose.py:40,60`).

Build (`tools/rollout_replay.py`, ~150 lines, reuse `terminal_eval.make_batch`):
1. For each episode under `data/episodes/deploy/<day>/ep_*` with `status == finalized` and ≥3
   accepted replans: `off = meta.clock_calibration.offset`; anchors `t_i = trace[i].t + off` for
   accepted replans; keep those with `lo ≤ t_i ≤ hi` from `sampler.valid_range(ep)`.
2. Batch at `t_i`: `item = sampler.sample(ep, t_i)` → `make_batch` (zero `events`/`cpk_*` exactly
   as `terminal_eval.py:112-114`). This reproduces the deploy observation: the camera frame at
   t_i (deploy tiles the current frame, `policy.py:83-86`; the sampler reads frames at t_i +
   j/fps — for a replay set `item["video"][1:] = item["video"][0]` to match deploy exactly), the
   wrist window resampled on the same grid (`planner.py:129-138` mirrors `windows.py:260`), the
   ur_state at t_i, tactile at t_i.
3. prev_chunk: three variants — (a) `trace[i-1].actions` normalized (what deploy actually used),
   (b) the sampler's default (= `actions.zarr` = plan steps as PLAYED, `executor.py:207-212`),
   (c) measured Δ-EE from `arm_tcp_pose` over the last 1.6 s (training semantics; same
   construction as `data/derived.py pose_delta`). Report (a) as the primary, (c) as the diagnostic.
4. prev_cpk: not in the trace (the package is a tensor) → the model's own sample at `t_i − 1.6 s`
   as in `terminal_eval.py:137-140`.
5. Sample with the trace's `nfe/guidance`, K = 4 seeds. (Persistent-noise seeds were not recorded —
   `meta.tags` has `seed:none` — so exact reproduction of the trace chunk is impossible; check that
   `trace[i].actions` falls inside the K-seed spread as the replay's validity test.)
6. Targets — there is no GT, so score against geometry and against demo states:
   - `z_now = arm_tcp_pose[t_i].z`; `z_close_task` from `configs/start_poses.yaml tcp_z_min` (or
     `rig_trace_decompose demos` z_close mean). Metrics per replan with `z_now − z_close > 15 mm`:
     `closure_frac = −Σdz_pred / (z_now − z_close)` (demo ≈ 1 within one chunk when < 80 mm
     high), `pred_descent_mm_s`, `pred_close_height_mm = z_now + Σdz_pred[:k_close] − z_close`
     (k_close from `close_steps` with the task's plateau aperture), `p_close_in_chunk`.
   - Nearest-demo-state target: for the same task, demo frames with |Δxyz| < 15 mm, gripper open,
     OBJ = 3-or-0 → their GT chunks; report endpoint error to the nearest and the spread.
   - Multi-modality: per-state std of Σdz across K seeds; fraction of seeds that commit.
7. Counterfactuals (one-line batch edits, run on the same states): zero/scale `ur_state[6:12]`
   and `[18:24]` (velocity); prev_chunk (a)/(b)/(c)/zeros/demo-like; guidance 1.0 vs 1.5/2.0
   (the model was trained with `cond_dropout_p = 0.1`, so obs-guidance is a zero-training knob
   that can be evaluated here before the rig). Each pair of numbers is a hypothesis test for the
   CONTEXT's three suspects.

Cost: 26 episodes × ~20 replans × (1 + 1 prev) × 4 seeds × 5 NFE ≈ 4k forward passes ≈ 20–30 min
on the 5090; minutes on an H100. Output: one JSONL row per (episode, replan, variant, seed) +
a per-task summary. Run it on the 08-20 and 08-28 sessions for v4 and v5 first — that alone will
say whether v5's 17.4 mm means anything at rig states.

---------------------------------------------------------------------------------------------------

## 4. What I would do in the next 72 h (training/eval lens only)

1. Build the replay tool (§3) and run F1/F5 counterfactuals on the existing 26 + 13 episodes.
2. `rig_trace_decompose demos tasks/*_fail` → decide F3; if a recovery segment exists, add
   window-level action weights.
3. One fine-tune from v4-EMA with: intent augmentation (F1), velocity augmentation (F5), height-
   anchored grasp band, window-level failure weights (F3), `ema_decay 0.995`, photo-aug 0.5 —
   3k steps, checkpoints every 500, raw AND EMA replay numbers per checkpoint, selection on the
   replay metric at rig states, not on `terminal_eval`.
4. Only then spend rig time, with the replay's predicted `pred_close_height_mm` as the
   pre-registered number to beat.
