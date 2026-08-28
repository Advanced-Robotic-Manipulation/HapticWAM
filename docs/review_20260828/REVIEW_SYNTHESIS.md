# PHANTOM — review synthesis and 3-week plan to ICRA 2027

Inputs: 7 Opus narrative lenses (`scratchpad/review/*.md`), 5 Codex lenses (`scratchpad/review/codex/*.md`),
6 literature reports (`scratchpad/research/0*.md`), and the adversarial confirm/refute round
(`scratchpad/review/round_1.json` + the verdict JSON: 15 confirmed, 14 refuted).
Repo `~/GitHub/phantom` @ `3539988`. Written 2026-08-28. Deadline ~Sep 15 2026.
Constraints: 1 UR3 rig, 2 operators, **15–20 rig-hours total**, rented H100s (~$2.5/h), compute3 (5090) free.

Certainty labels: **[certain]** read in code / measured by a reviewer; **[believe]** strong inference;
**[guess]** hypothesis with a named test. Every finding below survived a two-voter adversarial round;
numbers that a voter corrected are stated in their corrected form.

---

## 0. The one-paragraph verdict

There is no paper until the teacher grasps, and the single biggest reason the team cannot make the
teacher grasp is that **no number it currently computes can see the failure**. `terminal_eval` conditions
every input on a demo that is already descending correctly at a demo-placed object, so it measures the
terminal *prior*, not perception-driven commit; it moved 20.7→17.4 mm while the rig went 0/26, and its
`commit_ratio` moved 1.72→1.43 in the *wrong* direction (offline the model over-descends while the rig
under-descends). The first deliverable of the next three weeks is not a fix, it is a **replay evaluator that
runs the checkpoint on the rig's own recorded states** — 37 episodes with full zarr streams and
`planner_trace.json` already exist and have never been fed back through the policy. Everything else in this
document is ordered behind that, because it decides which of four live root-cause hypotheses to spend rig
hours on. Second: the "closes on air then lifts" half of the failure is **by construction** — nothing in the
loss or the architecture ties close-outcome to the lift, and the ACC gate that correctly says `p_none 0.99`
is provably decorative at deploy (it is logged, never consumed). A ~40-line deploy-time terminal veto fixes
that half without training and *is* the paper's tactile-grounded-terminal-control story. Third: the paper's
headline arm (RQ3, sensor-free student) currently **cannot be built** (no `--student` flag, wrist F/T welded
into every layout) and **would be silently mis-built if it were** (three training programs construct models
from default `PhantomModelConfig` instead of the checkpoint's). Those are ~1 dev-day and are on the critical
path for a 35-h control run.

---

## 1. The ten most important confirmed problems

Ordered by (expected effect on the rig outcome or on paper validity) ÷ (cost to fix). Two items merge
closely-related confirmed findings; the four remaining confirmed findings are in §1.11.

### P1 — The offline metric conditions on demo-consistent inputs and structurally cannot see the rig failure
**Where:** `tools/terminal_eval.py:118-143` (batch built from one `sampler.sample(ep, t0)`, `t0 = tc − 1.6`,
only `events`/`cpk_*` nulled) and `tools/terminal_eval.py:89-91` (episode set = `manifest_split(...,"val")`;
`data/episodes/deploy` is never read by any tool that loads a checkpoint — `tools/rig_trace_decompose.py:43`
is post-hoc only). Also `tools/terminal_eval.py:65-66` (`--max-episodes` default 80 slices a task-sorted
124-episode index) and `phantom/train/common.py:334` (missing manifest silently falls back to *all* episodes).

**Why it matters:** this is the number that blessed v5_6 and will decide how 15–20 rig-hours are spent. It
teacher-forces GT scene video, GT `ur_state` (incl. `qd`/`tcp_speed` at demo cruise), GT tactile/`contact_state`,
and — decisively — the GT demo `prev_chunk`, while deploy feeds the policy's *own* previous proposal
(`phantom/inference/policy.py:101-111`). The rig's state family (hover at 150–250 mm, |v| < 30 mm/s, aperture
drifting 0.31→0.52) is never evaluated. Corroborating measurement: the v4 conditioning ablation
(`EVAL_RESULTS_v4.txt:11-12`) shows knocking out video shifts the sampled chunk by 1.85 mm, `prev_chunk`
0.93 mm, wrist 0.25 mm — on these windows the chunk is dominated by prior + noise seed, so a fine-tune that
sharpens the terminal prior improves the metric and does nothing on the rig. That is exactly what v5 did.

**Exact fix (two parts, both cheap):**
1. `tools/replay_rig.py` (~250 lines, skeleton in `scratchpad/review/closed-loop-root-cause.md` §5; a Codex
   variant is `tools/deploy_replay_eval.py`). For each accepted replan of each recorded deploy episode:
   `t = trace[i]["t"] + meta.clock_calibration["offset"]` (`tools/rig_trace_decompose.py:41,60`), rebuild the
   `ObsSnapshot` exactly as `SnapshotBuilder.build` did (`phantom/deploy/planner.py:77-185`: latest-≤t rows,
   not nearest; wrist window on `linspace(t−0.25, t, 31)`; `video[1:] = video[0]` to match the tiling at
   `phantom/inference/policy.py:85-89`), sample with the trace's `nfe/guidance`, K=4–8 seeds, and report
   `head_dz` (cum z over steps 0–8), `tail_dz`, `close_step`, `grip_max`, `pred_close_height_mm`, plus the
   **across-seed std**. Validity test: the trace's own chunk must fall inside the K-seed spread (seeds were
   not recorded — `meta.tags` has `seed:none`).
2. Harden `terminal_eval` regardless: default `--max-episodes` to all, hard-fail on a missing manifest,
   report per-task medians, add `pred_close_height_mm`, and add a `--null {tactile,wrist,prev_cpk,obs,all}`
   flag driving the existing `_null_obs_batch` / `null_video_cond` paths (`phantom/model/rf.py:214-234,396-397`).
   Cost: ~1 dev-day for both; then every checkpoint decision is made on the rig's own states.

### P2 — `prev_chunk` at deploy is the policy's own future-shifted proposal, not the executed past
**Where:** `phantom/inference/policy.py:101-111` (`normalize("action", prev_plan.actions)`) vs
`phantom/data/windows.py:277` (16 **measured** `pose_delta` rows on the grid `t0 − (16−k)/10`, written by
`phantom/data_collect/session.py:733-750`). `phantom/deploy/planner.py:284-291` marks this DEFERRED.

**Why it matters:** it is worse than the DEFERRED comment says. `policy.py:166-172` anchors
`action_times[0] = obs.t + latency` and `planner.py:259-260` builds the next snapshot immediately after
`submit` with no pacing, so at the rig's 0.96 s cadence the "previous chunk" spans roughly **[t0, t0+1.6]** —
essentially *zero* overlap with the executed past, ~7 of 16 steps never executed, and everything reshaped by
governor/blend/rate-limit. This channel is never dropped under conditioning dropout
(`phantom/model/rf.py:283-284`: "prev_chunk stays (it is intent, not observation)"), it modulates every DiT
block through the pretrained action-AdaLN (`phantom/model/phantom_dit.py:142-156`) and drives ACC's
`intent_mlp` (`phantom/model/acc.py:62,84`). So every replan conditions on a fixed point of the policy's own
output over nearly the window it is being asked to predict — the textbook copycat/self-consistency loop
(1905.11979, 2010.14876) and the most plausible carrier of "slow mode locks in". Note: the parity lens's
claim that this path is *structurally inert for actions* is **false** — `structural_attn_bias`
(`phantom/model/sequence.py:190-202`) blocks only CONTACT/ACTION→VIDEO_GEN; OBS_*/VIDEO_COND queries read
VIDEO_GEN freely, and `scratchpad/review/probe_leak.py` measures a non-zero 2-hop path that goes to exactly
0.000000 once all non-video queries are blocked.

**Exact fix (~30 lines in `SnapshotBuilder.build`):** take `rings["arm"].latest(n ≥ 1.8 s × 125 Hz)`, pick the
measured `tcp_pose` nearest each of the 16 grid times `t_now − (16−k)/10`, chain `derived.pose_delta`, and
append the gripper command actually sent per step (the executor knows `plan.actions[k,6]` per entered step).
Keep the first-replan normalized-zero case. Ship it behind `--parity-fixes` so the rig A/B can attribute it.
Validate first offline with replay swap E3 (§2).

### P3 — Nothing ties "did the close succeed" to the lift, and the ACC gate is decorative at deploy
**Where:** `phantom/model/attention_bias.py:196-203` (`acc_key_bias = softplus(beta_raw)·p_evt·g` on haptic
keys, wired at `phantom/model/phantom_dit.py:249-251`), `phantom/model/acc.py:95-104` (`g ∈ [0,1]`),
`phantom/data/windows.py:403` (`action_weight = 0` for failure demos) + `phantom/model/rf.py:310-318`,
`phantom/deploy/planner.py:263,276` (`plan.gate` / `plan.p_evt` are **logged only** — grep confirms no code
path in `phantom/deploy/` lets the gate modify, clamp or veto a chunk).

**Why it matters:** this is the second half of the rig failure and it is *by construction*, not a bug to hunt.
The bias is non-negative and collapses to ~0 as g→0, so "no contact" removes the haptic up-weighting rather
than producing any suppressive signal; the 45 under-grasp failure demos teach zero corrective action; and the
only proprio success bit (`gripper.obj`) is unusable — only 285/788 successful demos ever show OBJ=2 during
the closed phase (see P8). Meanwhile the numerical size of the ACC injection is `|λ·β·g| ≤ 0.157` nats
(measured on the real v5 EMA checkpoint: `phantom_bias_lambdas` max 0.109, median 0.033, 7 of 28 negative,
`softplus(beta_raw)` 1.18–1.44) — a ≤17% key re-weight in the strongest block, ~4% typically. For the paper
this means the RQ2 "anticipated contact is coupled into the policy through ACC attention" claim is
**unsupported as written** and must either be re-parameterised or downgraded to "auxiliary anticipation head".

**Exact fix, in order:**
1. *Deploy-time terminal veto, no training* (~40 lines in `PlannerLoop.run`): after a commanded close, if
   `plan.p_evt[none] > 0.9` for one replan → command the task open aperture, clamp the next chunk to `z ≥ z_now`
   (no lift), replan; log `terminal_veto`. Add a **close-mask**: veto any commanded gripper-close transition
   unless `p_contact > θ_close` (start 0.5, calibrate on the 08-20 gate trace) **or** `z ≤ tcp_z_min + 15 mm`.
   Literature precedent: 2503.23835 gets 100% disturbance resilience from exactly this scripted override vs
   20–70% for the learned alternative; 2410.13124 reports even a *tactile-equipped* diffusion policy still has
   11.5% phantom grasps, so a scripted veto is defensible, not embarrassing.
2. *λ:=0 inference ablation* (one flag, GPU-minutes) to settle whether the ACC bias does anything at all.
   If inert → either `lambda_bias_init` 0.5–1.0 in the final teacher, or downgrade the RQ2 claim.
3. *Medium term:* DAgger-lite relabels of rig failures (open + re-descend) with action loss **on** — see P9/§3.
   Do **not** simply un-zero the 45 `_fail` demos: `tools/intake_recovery.py:13-18` records that they continue
   with an empty gripper ("phantom carry"), i.e. imitating them teaches the rig failure. This is why the naive
   "restore action_weight on failure demos" finding was refuted (§5).

### P4 — The executed chunk head is one latency stale from a pose the model never observed; the tail never runs
**Where:** `phantom/inference/policy.py:165-172` (`action_times = obs.t + latency + k/rate`),
`phantom/deploy/executor.py:96-111` (`u0 = (now − action_times[0])·rate ≈ 0` → playback starts at index 0)
and `:104-106` (`t0_pose = _last_cmd − c0`, rebasing onto a pose the observation never contained), vs
`phantom/data/windows.py:276` (training chunk anchored at `t0` with zero latency, all 16 steps supervised).

**Why it matters:** measured on the 37 rig traces, latency == replan interval == 0.97 s (08-28) and 1.69 s
(08-20), so **only steps 0–9 of every chunk ever execute** and they execute over `[t_obs+L, t_obs+2L]`. In
`whiteboard_1787941222_002`, 17/17 replans put 2–4× more descent in the never-executed tail than in the head —
a "slow now, fast later" mode is indistinguishable from "stop" on this rig. Two consequences: (a) the team's
headline decomposition ("the executor tracks the commanded chunk within mm, therefore the *policy*
under-commits") is confounded — the executor faithfully tracks a *time-shifted head*; (b) a 1 s pure delay in
a position loop means a *perfect* policy cannot stop within ±40–60 mm of where it intends. Two independent
simulations agree the loop alone produces **overshoot**, never a stall (`sim_executor_lag.py`: z_min 34 mm at
L=0.9 s vs a 72 mm demo stop; `sim_velocity_loop.py`: 1.5 mm), which is why latency is a **fix constraint and
a confound, not the root cause** — but it means any repair of the policy's z-judgement without fixing the loop
trades under-commit for table strikes. (The reviewer's citation of "40 mm below any demo" at
`docs/rig_session_v5.md:66-68` does not check out; the sim result stands on its own.)

**Exact fix, in order:** (1) **cut latency first** — the loop is compute-bound; benchmark `--nfe 3` and
`--compile` (already wired at `phantom/scripts/run_deploy.py:297-315`, unused by `GO_ANY.sh`), target L ≤ 0.5 s.
(2) **Phase-correct playback + committed-prefix inpainting** (RTC, 2506.07339): set `t_exec0 = obs.t` so
`submit`'s existing skip-elapsed branch starts at `k0 = ceil(L·rate)`, and in `rf.sample` pin the first `k0`
action slots to the *noised committed actions* at every Euler step, exactly as cond frames are pinned
(`rf.py:412,457`); `ActionPacker` maps action `a` of frame `k` to channel `a` (`packing.py:284-287`), so the
mask is per (frame, channel). **Do not ship skip-head alone** — the team already tried and reverted it for
cause (`executor.py:88-94`), and the sim shows it halves descent speed and exhausts the chunk at L=1.3 s.
Keep the z-floor. Decide the rule offline with replay experiment E8 before spending rig time.

### P5 — The contact heteroscedastic NLL owns the shared trunk's gradient; the action objective trains at ~1/10–1/50 rate
**Where:** `phantom/model/ace/losses.py:77` — `d_B_Tc / log_var.exp() + log_var` with **no detach** on the
residual, which flows through `x0_pred` (`phantom/model/rf.py:308`) into the LoRA. `sigma_reg` is
`0.01·(log σ)²` (`rf.py:327`, `config/model.py:37`), which shifts the NLL stationary point by ~1% and cannot
hold σ up. The v4/v5 logs pin the regime: `sigma_reg = 4.4–6.5` with `contact_nll = −2.9…−3.7` ⇒ **log σ ≈
−2.05…−2.35, 1/σ² ≈ 60–160**.

**Why it matters:** the LoRA (22.9 M params) is the *only* capacity that can teach the frozen 2B trunk to
route camera/proprio/tactile into ACTION tokens, and it is being optimised as a contact-field forecaster.
Two independent probes: at init on the tiny teacher, contact_nll is 93–98% of the LoRA squared-gradient share
(|g| 4.31 vs action 0.27); in a realistic-regime output-space probe calibrated to the logged losses,
‖∂L/∂v_pred‖ contact/action = **14.8×** (vs 0.13× at σ=1). At true convergence the residual co-shrinks with σ,
so the honest number is a **10–50× norm ratio, >99% of the squared-norm share** — not the 114× first claimed.
Two sub-arguments in the original finding are **wrong** and must not be repeated: `grad_clip 1.0` is a uniform
rescale to which AdamW is invariant, and the "95–100%" figure was an at-init tiny-model number. Even corrected,
this is the most parsimonious explanation for the weak obs→action coupling seen everywhere (`sampled_dir_cosine`
0.62/0.82 on val, consecutive-replan cosine 0.16–0.35 on the rig, ≤1.85 mm knockout sensitivities). It also
**confounds the planned baselines**: `vision_only`/`no_distill` carry no contact NLL at all, so a teacher-vs-
baseline gap could be objective imbalance rather than tactile information.

**Exact fix (3 lines + one fine-tune):** β-NLL (Seitzer 2022) — multiply the NLL term by
`sigma.pow(2*beta).detach()`, β ∈ [0.5, 1]; or detach the weighting from the trunk path:
`d.detach()/log_var.exp() + log_var` for the σ head plus `w_c · d` (plain MSE) for the trunk. Drop the
duplicate `wrist_region_mse` (`losses.py:79-86` double-supervises channel 15, also the NLL's `wrist` group).
Confirm with the per-term LoRA grad probe on the **real v5 checkpoint** (10 GPU-min, script in
`scratchpad/review/model-losses.md` appendix and `scratchpad/research/gradprobe.py`) before spending the run.

### P6 — 5-step Euler with `rf_shift=5` emits x0 at an almost-untrained t; persistent noise becomes a fixed per-episode velocity offset
**Where:** `phantom/model/rf.py:430` (`ts = [1, 0.952, 0.882, 0.769, 0.556, 0]` under `_time_shift(rf_shift=5)`,
`config/backbone.py:70`), `rf.py:448-452` (the last update is algebraically `x0_pred(t=0.556)`),
`rf.py:270-276` + `train_teacher.py:161-166` (`action_t_max_of_two=True` in both shipped checkpoints, confirmed
from the saved `configs.model`), `phantom/model/ace/packing.py:274-288` (each action value tiled over 160–192
i.i.d. latent cells), `rf.py:405-411` + `run_deploy.py:448` (`--persistent-noise` reuses one eps tensor for the
whole episode).

**Why it matters (all measured):** under max-of-two the ACTION frames see median t = 0.896 and
**P(t < 0.556) = 0.7%, P(t < 0.7) = 5%** — the two Euler steps that actually resolve the mode run where the
action head has essentially never trained. Strip-mean noise std is 0.072–0.079; the Bayes weight the network
should place on the `x_t` reading is 0.06 / 0.32 / 0.71 / **0.95** across the four steps, so the final chunk is
dominated by a strip-mean offset that, under persistent noise, is **identical at every replan of an episode** —
a fixed 0.1–0.5σ per-step bias ≈ 0.7–3 mm/step ≈ 7–30 mm/s on z. That is a precise mechanism for "one seed
descended at full speed, the others ~30% slower", and it is invisible to `terminal_eval`, which reports
seed-averaged means only. With fresh noise the plan direction re-rolls every 0.9 s (rig consecutive-replan
cosine 0.16–0.35, recorded in `rf.py:367-371`), i.e. the noise term is *not* small relative to the
obs-conditioned term at rig states.

**Exact fix:** *today, no retraining* — `terminal_eval`/replay at `--nfe 1/2/5/10 × 8 seeds`, reporting mean
**and across-seed std**, with a `--persistent-noise` flag added there. `--nfe 1` gives
`x0 = eps − v(eps, t=1) = E[x0|obs]`: deterministic, no offset, ~5× lower replan latency (which also buys P4).
Then the top deploy-side lever: **K-seed sampling with contact-consistent selection** (BID 2408.17355 /
Q-Planning 2608.21204) — batch K=4–8 seeds in one denoise, reject chunks whose head descent is <50% of the
K-max while `p_contact` is low, pick the mode nearest the previous plan. *Retrain-level:* draw ACTION/CONTACT
noise **per strip** (`eps_action = ActionPacker.pack(randn(B,H,A))`) in both `training_step` and `sample`, then
drop `action_t_max_of_two` and use an unshifted per-frame schedule for ACTION frames.

### P7 — Co-denoised GT future contact/video is a training-time shortcut for the ACTION head; the "droppable video" mask leaks
**Where:** `phantom/model/sequence.py:190-202` (bias set only on [CONTACT/ACTION rows, VIDEO_GEN cols];
`tests/test_layout.py:58-68` asserts video-query freedom and *nothing* about OBS rows), `rf.py:270-276`
(ACTION at `max(t,t')`, CONTACT/VIDEO at `t`, so `t_contact ≤ t_action` always), `rf.py:130-134` (GT future cpk
packed into those CONTACT frames), vs `rf.py:433-434` (at deploy every non-cond frame shares one t).

**Why it matters (measured with the real packers, optimal linear read-out):** at the ACTION head's median
training noise t = 0.90 the co-noised GT contact package is **98% decodable** (event-class accuracy 0.980,
wrist r 0.85, wrench r 0.85, slip 0.94, own-action r 0.83); at t = 0.85 everything is ≥0.92. So the cheapest way
to lower `action_v_mse` is to read "onset at +1 s" off tokens that at deploy are the model's own mean-regressed
imagination. This is classic exposure bias and it is a *direct* mechanism for "closes and lifts on an imagined
onset schedule while the gate correctly reports p_none 0.99". Separately, OBS_*/VIDEO_COND queries are
unrestricted on VIDEO_GEN keys, so (a) generated video reaches actions in two hops and (b) **the "droppable
video without train/test mismatch" claim in the docstring and the paper is false** — and `drop_video=True` also
shifts every later frame's learnable absolute temporal pos-emb (`minimal_v4_dit.py:836`, `pos_emb_t[:T]`).

**Exact fix:** diagnose first (30 GPU-min): `terminal_eval`/replay with CONTACT frames cond-pinned to GT vs
zeros; a >5 mm shift means the head leans on tokens it never has at deploy. Then pick one, both need a
fine-tune: (1) **group-causal mask** — VIDEO_COND/OBS attend only conditioning tokens; ACTION attends
cond+ACTION (+CONTACT); CONTACT attends cond+CONTACT(+ACTION); VIDEO_GEN attends all. One function; the flex
BlockMask follows; bonus, cond-token K/V can then be cached across the 5 NFE steps (~2× faster replans, which
buys P4 again). (2) **Self-forcing** — pack the two-pass inner sample's own `pred.cpk` (already computed,
`rf.py:163-168`) into the CONTACT x0 the ACTION frames are denoised alongside, keeping GT as the *loss target*.
Zero extra forward passes. Literature: ViPRA (2511.07732) reports exactly this collapse (joint coupling helps
at pretrain, 53%→31% when kept at finetune) attributed to compounding drift on OOD states; HarmoWAM
(2605.10942) names the same tension. Cite both; do not let a reviewer predict our result from our architecture.

### P8 — Robotiq OBJ is not a grasp-success signal on this rig, and the literature note has the codes inverted
**Where:** `phantom/drivers/base.py:70-73` and `phantom/drivers/real/robotiq.py:90-94` (gOBJ: 0 moving,
1 contact-while-opening, 2 contact-while-closing, 3 at requested position) vs
`scratchpad/research/06_failure_demos_success_reward.md:34`, which states the reverse and recommends it as a
near-ground-truth success label. `tools/episode_qc.py:89` already prints `obj2_frac>0` as "held object".

**Why it matters (recomputed over all 1115 episodes):** among *successful* demos, `obj2_frac_closed > 0` in
5.2% of Carton, 1.6% of egg, 22.8% of waffles, 99.2% of whiteboard — while the over-squeeze `*_fail` demos show
it 85–100% and the 45 under-grasp fails show 0% *and still lift 79–277 mm with an empty gripper*. OBJ=2 flags
**over-squeeze/stall**, not grasp: with position-streamed teleop the fingers reach the commanded aperture on
soft objects and the Robotiq reports 3. An "OBJ ∈ {1,2} after close" auto-label would mark ~690/1070 genuine
successes as failures, collapsing the positive set to whiteboard and building the paper's self-improvement
curve on mislabeled data.

**Exact fix — label grasp success from tactile contact sustained through the lift** (this is also the paper's
point: mechanical object-detect fails on soft objects, the visuotactile gel does not):
```
t_close = ts_grip[close_index(pos)]                     # phantom/train/common.py:300-321; None -> label 0
contact(t) = max_f mask_frac_f(t) > tau_contact_area(0.025)   # the training gate definition, windows.py:331,353
hold    = [t_close+0.5 s, t_release);  c_hold = mean contact over hold
grasp_ok = z_close <= Z_MAX[task] (demo p95+15 mm: waffles 103, Carton 144, egg 101, whiteboard 181 mm)
        and len(hold) >= 2 s and c_hold >= 0.8 and lift >= 50 mm while in contact
stall    = mean(obj==2 over hold) > 0.5                  # over-squeeze FLAG only, never part of grasp_ok
task_ok  = operator verdict 's'                          # place/wipe completion stays human
```
Validate on compute3 before the rig: expect ≥95% positive on the 1037 successes, ~0% on the 45 under-grasp
fails and on the 39 rig rollouts, `grasp_ok=1 ∧ stall=1` on the 80 over-squeeze fails. Write it as a pytest
fixture so it is regression-tested.

### P9 — Unlabeled and `contaminated` rollouts would train at full action weight
**Where:** `phantom/data/schema.py:167-171` (`is_failure_demo` is True only for `success is False`, tag
`deliberate_failure`, or a `*_fail` task name — verified by execution to return **False** for
`success=None, tags=['contaminated']`), `phantom/data/windows.py:403`, `phantom/deploy/runtime.py:225`
(`recorder.stop(success=None)` leaves `status='finalized'`), `phantom/scripts/run_deploy.py:467-489`
(Enter = skip leaves `success=None`; `c` sets `success=None` + tag `contaminated`; `--no-label-prompt` skips
the prompt entirely), `tools/intake_recovery.py:141-152` (appends every finalized `ep_*` as `split='train'`
ignoring success/tags), `phantom/train/common.py:355-360` (`manifest_split` filters by status only).
The collect app fixed this exact hole (`phantom/data_collect/session.py:357-377`, `status='aborted'` +
tag `unlabeled`); deploy did not.

**Why it matters:** latent today (v5 trained only on teleop demos) but it detonates on the **first** DAgger-lite
round, which is on this plan's critical path: `train/dagger_driver.py:50-60` hands the rollout root to
`distill_hid --extra-data`, merged with no success filter, grounded through `act_w`. A session where the
operator skips six prompts (or the process dies before the prompt) trains six on-air closes at weight 1.0 on
exactly the on-policy states where the model is already wrong — reinforcing the bug and invalidating any
"self-improvement helped" claim.

**Exact fix (3 arguments + 2 lines):** in `run_deploy`/`run_episode`, no verdict ⇒
`recorder.relabel(status='aborted', tags=['unlabeled'])` (the API exists, `recorder.py:207-222`, and its
docstring warns about this exact case); make the rollout intake refuse `policy != ''` episodes with
`success is None` or tag `contaminated`; make `manifest_split` skip rows tagged `contaminated`/`unlabeled`.
Related, same intake pass: **re-derive `actions.zarr` from `arm_tcp_pose`+`gripper` on the 10 Hz grid**
(`executor.py:205-212` records the raw *proposal* at governor-warped times, before clamp and rate limit, while
demos record measured `pose_delta` — `session.py:733-750`), keeping the original as `actions_plan.zarr`; and
stamp deploy episodes with the **base** hardware hash (`run_deploy.py:262-289` builds an overridden `hw` whose
`config_hash` makes every rollout fail `train_teacher`'s CONFIG DRIFT check, `train_teacher.py:286-296`).

### P10 — The paper's headline arm cannot be built, and would be silently mis-built if it were
**Where (A, unbuildable):** `phantom/train/train_teacher.py:189,194,258` hardcode `student=False` and the
argparse block has no `--student`, so the "critical control" `no_distill` named at
`docs/training_playbook.md:112` is not runnable. `phantom/model/hht/hht.py:43-49,84-91` fuse
`[WristTCN(wrist) ‖ URStateMLP(ur_state)]` into OBS_PROPRIO for teacher **and** student;
`phantom/deploy/planner.py:27` sets `TACTILE_INPUT_MODES=("teacher",)` and `SnapshotBuilder.build` computes the
wrist window in every mode; `grep mask_wrist` returns nothing. So `vision_only`, `no_distill` and
`drop_tactile` are input-identical, and `phantom/eval/aggregate.py:72-79`'s headline
`recovery_ratio = (student − vision_only)/(teacher − vision_only)` has **no producible denominator**.
**(B, mis-built):** `phantom/train/distill_hid.py:190-193`, `phantom/dagger/relabel.py:61-62` and
`phantom/train/finetune_hids.py:143-146` call `build_model(...)` with **no `mc`**, so `builder.py:46-47` falls
back to `PhantomModelConfig()` defaults — `rope_time_mode='aligned'` (ACTION RoPE positions `[1,2,3,3]` instead
of `[0.4,0.8,1.2,1.6]`) and `acc.self_anticipation='gt_noised'` — while v4/v5 were trained `time_true` +
`two_pass`. `common.py:245-255` asserts hardware shapes only; the RoPE table is recomputed at runtime from
`mc` (`phantom_dit.py:96`), so **no saved key can rescue it** and no error is raised.
`run_deploy.py:38-50`, `terminal_eval.py:77-79` and `train_teacher.py:205-215` were all hardened in the v4
audit; these three were missed.

**Why it matters:** RQ3 is the paper's headline. (B) would silently invalidate the student number and every
DAgger relabel after ~$70–90 of H100; (A) means a reviewer attributes any student gain to the surviving wrist
F/T signal. Also note the related honesty item: on this rig wrist F/T is `source: ur_internal`
(`configs/hardware.nuc.yaml:37`) — the arm's own current-based estimate, so "sensor-free" is defensible as
**"fingertip-sensor-free"**, but write it that way (that refuted finding's one salvageable nugget).

**Exact fix (~1 dev-day):** (i) add `--student` to `train_teacher` (pass through to `PhantomModelConfig`,
`build_model`, `WindowSampler`); (ii) add `PhantomModelConfig.mask_wrist` honoured in `HHT.obs_frames` /
`wrist_feature` (zero the window) and in `SnapshotBuilder.build` for `vision_only`/`drop_tactile`;
(iii) in each of the three programs, `mc = PhantomModelConfig.from_dict(payload["configs"]["model"])`, build
teacher with `mc` and student with `dataclasses.replace(mc, student=True)`; (iv) assert in
`load_phantom_checkpoint` that `model.mc.to_dict()` matches the saved config modulo `student`, plus a test that
a `time_true` checkpoint refuses an `aligned` model. Then launch `no_distill` early — it is a ~35 h H100 job
(~$90) on the critical path and depends only on the final data/recipe.

### 1.11 Also confirmed (fix, but they do not compete for the top ten)
- **Evaluation protocol as coded cannot support the success-rate claims.** `phantom/eval/trial_runner.py:67-79`
  loops `system → task → occlusion → seed → trial` with the runtime opened per system (strictly system-major,
  interleaving structurally impossible); `:92-93` success/damage are operator y/n; `run_deploy.py:470-482` has
  no damage field; `phantom/eval/aggregate.py:43-53` clusters the bootstrap on `seed`, which is **never plumbed
  into the policy at all** (absent from `runtime.run_episode`'s signature — so a "≥3 seeds" claim would be for
  seeds that were never run); `configs/eval_campaign.example.yaml:4` names five tasks that do not exist
  (`fragile_grasp/slippery_place/insertion/wipe/regrasp` vs the real waffles/Carton/egg/whiteboard); the plan's
  ~2100+960 trials is ~10× the ~250 valid episodes obtainable in 15–20 rig-hours. Fix = the pre-registered
  protocol in §3 (interleaved per placement, objective labels, Wilson + Fisher, single checkpoint per arm
  stated plainly).
- **The ACC attention injection is numerically almost inert** (`phantom/model/phantom_dit.py:67`,
  `attention_bias.py:196-203`) — see P3; run the λ:=0 ablation and either re-parameterise or downgrade RQ2.
- **Trained with `cond_dropout 0.1`, deployed at `guidance 1.0`** (`train_teacher.py:156-160`,
  `provision_v5.sh:268-276`, `rf.py:394-455`, `run_deploy.py:170-173`): a correctly-implemented, fully-trained,
  **never-exercised** knob. `terminal_eval.py:59` already exposes `--guidance`. A/B 1.0/1.5/2.0 offline first —
  guidance >1 doubles NFE cost (~0.9 → ~1.8 s replan), which is itself a closed-loop confound, so it must beat
  the baseline offline before it costs a rig hour. Note the CFG null branch also leaves `acc_null` on its own
  predicted summary while the conditional branch is overwritten with the true `prev_cpk`
  (`rf.py:394-400,446`) — fix that before any guidance sweep is interpretable.
- **Deploy safety: the z no-go floor is an episode STOP, not a clamp** (`phantom/deploy/safety.py:198-210` +
  `run_deploy.py:264-278`: `apply_hitbox` intersects with the already-raised workspace, both tests run on the
  *unclamped* target, `max(CLAMP, STOP) = STOP`). Reproduced: `target z = floor − 1 mm → stop`. **A policy that
  finally descends far enough loses the episode outright and forces an `f` label — it biases the A/B against
  exactly the model we are trying to build.** One-line fix: evaluate the hitbox on `self.clamp_target(...)`.
  Also regenerate `configs/start_poses.yaml` envelopes over the full 250/task set (they came from n=17–37) and
  add `q_min/q_max` joint-envelope STOP. **Land this before any rig episode is counted.**

---

## 2. Ranked root-cause hypotheses and the decisive offline experiments

The phenomenon to explain (from `docs/rig_session_v5.md`, the traces, and CONTEXT): commanded chunk heads
−26, −23, −28, −22, −14, −3 mm per 0.96 s replan (≈25 mm/s where demos cruise at 45–50), reaching zero at
z ≈ 190 mm (demo close 46–72 mm); the gripper command rises 0.31 → 0.52 *while still descending*; the chunks
then turn positive (lift) while the gate says `p_none 0.99`; same shape for v4 and v5 in every valid episode;
one noise seed descended at full speed.

**H1 [top, believe] — Few-NFE mean collapse of a broadened conditional onto the marginal chunk, locked in by
the loop.** `norm_stats.action.mean = [0,0,0,0,0,0, 0.42]` — the marginal chunk of this dataset *is literally
the failure*: zero motion, gripper at the waffles close aperture. The NFE-5 sampler's first four steps run at
t ≥ 0.77 where `E[x0|x_t,obs] ≈ E[x0|obs]`, and the deployed chunk is `x0_pred(t=0.556)`, a t that received
0.7% of ACTION supervision (P6). Offline, at in-distribution demo states, the conditional is sharp and the
mean is right (17 mm, `commit_ratio` 1.43–1.72 — the model *over*-descends). At rig states the conditional
broadens ("continue" vs "stop/close") and the mean of two modes is "slow descent + partial close", which the
loop then feeds back. Explains every observation including the seed multi-modality and why v4 ≡ v5.
**Against:** nobody has sampled the rig snapshots at high NFE or across seeds. → **E2**

**H2 [high, believe] — The obs→action map was trained weakly, so at shifted states the chunk is prior + noise.**
Two independent, additive mechanisms: the contact NLL owns 10–50× the action gradient on the shared LoRA (P5),
and the action head can read the answer off co-noised GT contact/video tokens at 98% accuracy (P7). Supporting
measurements: knockout sensitivities 0.25–1.85 mm; `sampled_dir_cosine` 0.62 (v4) / 0.82 (v5); rig
consecutive-replan cosine 0.16–0.35. This is the *training-side* half of H1 — it explains why the conditional
broadens so easily. → **E1, E5, E9**

**H3 [high, believe] — Self-referential intent loop.** `prev_chunk` at deploy is the policy's own proposal
covering ~[t0, t0+1.6] (P2), never dropped under cond-dropout, feeding the AdaLN of every block and the ACC
intent MLP. Copycat/causal-confusion (1905.11979, 2010.14876): "slow arm ⇒ slow mode" is loss-free in training
and non-causal at deploy. Note the *proprio*-velocity version of this hypothesis is much weaker than it looks —
a 30% slower descent is only **0.05–0.07 σ** in normalised `tcp_speed` (measured; the stds are
transport-dominated), so if velocity feeds back it is through the intent channel (per-step dz std 6.7 mm), not
through `ur_state`. → **E3**

**H4 [certain in code, for the "lifts anyway" half] — The close→lift transition has no outcome signal.** P3.
The only open question is whether a deploy-time veto can flip the chunk, or whether the lift is
image/aperture-driven and needs data. → **E6**

**H5 [medium as cause, high as fix constraint] — Executor phase lag.** P4. Both sims say the loop produces
overshoot, not a stall, so it is exonerated as *the* cause but it (a) confounds the policy-vs-executor
decomposition and (b) will convert any successful z-fix into table strikes if left alone. → **E8**

**H6 [low-medium, guess] — Guidance / null-token blur.** `_null_obs_batch` zeroes in *normalised* space, so
the "null" observation is the **dataset mean state** — whose `tcp_pose.z` mean is 0.187 m, i.e. 187 mm, which
is where the waffles rollouts committed (182–194 mm). Probably a coincidence, but 10% of the (already starved)
action gradient trains "mean state + black image → marginal chunk", and guidance >1 extrapolates away from
"the policy at the mean state", not from an unconditional policy. → **E7**

**H7 [low, keep the probe] — Height read from the near-top-down image, proprio z under-used.** The two stated
mechanisms were refuted (normalisation is per-channel and affine-absorbable; a 65–120 mm error is 0.7–1.4σ in
`tcp_pose.z`, among the *largest* normalised deviations, not a diluted one). But nothing has measured
`∂head_dz/∂z`, and per-*group* conditioning dropout genuinely does not exist (`rf.py:257-263` nulls all
observations jointly), which is the fix if the probe comes back near zero. → **E5b**

### The experiment table (all offline; ≤2 GPU-hours total on compute3; run in this order)

Data paths (compute3): rig episodes `~/phantom-icra-2027/data/episodes/deploy/{20260820,20260828}/ep_*` —
use `waffles_1787923675_000` (v5_6), `waffles_1787922904_000` (v4), `whiteboard_1787941222_002` (v5_6),
`Carton_1787937561_000` (v5_6) plus the other valid 08-28 episodes, **skipping everything from 17:19 on**
(wrapped-wrist / flipped-IK, 16 of 26 invalid). Checkpoints `runs/teacher_v5_batch0822/v5_6.pt` and
`runs/teacher_v4_790eps/teacher_020000.pt`. Val demos `data/val_eval/tasks` + `manifests/all.jsonl` split
`val`; norm stats `dataset_v3_packed/norm_stats.json`.

| id | question | swap | expected signature | decision |
|---|---|---|---|---|
| **E0** | does the replay reproduce the rig at all? | none (as-deployed batch, chained `prev_cpk`, persistent noise, NFE 5), 8 seeds | per-replan `head_dz` within the 8-seed spread of the trace, same commit replan | **GATE G0.** If it fails, nothing below is interpretable; fall back to rig-only iteration |
| **E1** | is the trunk gradient really contact-owned? | per-term LoRA grad-norm probe on the **real v5 ckpt** with a real batch (10 min) | contact_nll/action norm ratio 10–50×, >99% squared-norm share | β-NLL goes into fine-tune FT-A |
| **E2** | H1 mean collapse? | NFE ∈ {1,5,10,50} × 16 seeds (batch the seeds), same conditioning | at NFE 50 `head_dz` is **bimodal** (cluster near −45 mm and near 0) with NFE-5 samples **between** the modes and `grip_max` between 0.25 and 0.5; the "continue" fraction falls replan by replan | if yes → K-seed selection + NFE from the winner go on the rig; `--no-action-t-max-of-two` goes into FT-A |
| **E3** | H3 intent loop? | `prev_chunk` ∈ {proposal (as deployed), measured Δ-TCP over the last 1.6 s (as trained), zeros, a demo-like descending chunk}; `prev_cpk` ∈ {chained, None} | \|Δ head_dz\| > 5 mm ⇒ the path is live; measured-history recovering the descent ⇒ ship the P2 fix to the rig | P2 fix ships / doesn't |
| **E4** | H7 image vs proprio for height? | hybrid batches: (rig image + demo proprio @ same z) and (demo image @ z + rig proprio) at the stall replans | whichever half decides `head_dz`/`close_step` | image-dominant ⇒ per-group image dropout in FT-B |
| **E5** | sensitivity table | finite differences ±1σ on `tcp_pose.z` (±89 mm), `tcp_speed_z`, `qd`, gripper pos, obj; fixed noise; 40 val terminal windows + rig snapshots | `∂head_dz/∂z ≈ 0` ⇒ vision-only height (H7); velocity gain ≈1 ⇒ H3's proprio variant | picks the FT-B augmentation |
| **E6** | what could veto the lift? | post-close rig snapshot: `obj` 3→2; `pos` → demo close; tactile ← a demo in-contact snapshot; ACC `g` forced to 1 | if only the **tactile** swap flips lift → re-open, the action head *can* read touch and a runtime veto suffices; if nothing flips it, the lift is image/aperture-driven → needs recovery data | scripted veto vs DAgger-lite priority |
| **E7** | H6 obs reliance | guidance ∈ {1, 1.5, 2, 3} (fix the `acc_null` mismatch first); report `‖v_cond − v_null‖/‖v_cond‖` at t = 0.556 per replan | commit height falls with guidance; reliance dips at the stalled replans | guidance goes on the rig only if it wins offline |
| **E8** | executor rule pre-validation | playback rule ∈ {index0, skip-head, skip + prefix-inpaint} × L ∈ {0.5, 0.95}, on a closed-loop *surrogate* (next snapshot's proprio rolled forward with the executed head, image frozen) | z_min and descent speed per rule | picks the executor change **before** rig time |
| **E9** | the paper's premise (run this whatever else happens) | 124 val windows + the 45 `_fail` windows: real / tactile-nulled / wrist-nulled / prev_cpk-nulled / CONTACT-pinned-to-GT-vs-zeros / λ:=0 | Δ endpoint, Δ close_step, Δ aperture per condition | if tactile-null ≈ real, the tactile-teacher story is dead and the paper pivots to the commitment analysis |

Runtime: one 5-NFE sample ≈ 1 s on the 5090; E0 ≈ 16 min; E2 ≈ 5 min if the 16 seeds are batched (B=16 fits
32 GB at inference); E1/E3–E9 are minutes each.

---

## 3. Day-by-day plan (D1 = Fri 2026-08-29 → D18 = Mon 2026-09-15)

Rig budget: **18.0 h planned of 15–20 available** (§4 says what to drop first).
GPU: H100 @ $2.5/h. Planned spend **≈ $330**, worst case $420. *GPU dollars are not the binding constraint —
rig-hours and calendar are. Do not economise on H100 time; do economise on rig time.*

### Week 1 — instrument, diagnose, one pilot rig session

**D1 Fri 08-29 — code day A (rig 0 h, GPU $0)**
- `tools/replay_rig.py` (P1). Deliverable: E0 runs end-to-end on one episode.
- Parity bundle behind `--parity-fixes`: `prev_chunk` from the arm ring (P2), `prev_cpk` step index
  `round(latency/latent_dt)`, `reactive` from the two latest `fields_ds` frames, contact-state `dt` from ring
  timestamps.
- **Safety one-liner**: hitbox on the clamped target + the missing floor+hitbox test (§1.11) — blocking for
  every rig session after this.
- Deploy no-verdict guard + `manifest_split` tag filter (P9).
- `mc`-from-checkpoint in `distill_hid`/`relabel`/`finetune_hids` + the `load_phantom_checkpoint` assertion (P10B).
- `--student` + `mask_wrist` (P10A).

**D2 Sat 08-30 — offline day 1 (rig 0 h, GPU ~$10)**
- Run **E0** → **GATE G0**. Then E1, E2, E9.
- Regenerate `configs/start_poses.yaml` envelopes over the full 250/task set; `gen_start_poses` must refuse
  when `q_n != n`.
- Validate the P8 success rule on all 1115 demos + the 39 rig rollouts (expect ≥95% / 0% / 0%).
- **Launch `no_distill` insurance run** (student layout, current v5 recipe, 20k from scratch, ~35 h, **$90**).
  It is an insurance policy: if the recipe changes materially at the D9 freeze it must be relaunched ($90 more).

**D3 Sun 08-31 — offline day 2 (rig 0 h, GPU ~$10)**
- E3, E4, E5, E6, E7, E8. → **GATE G1**.
- Implement the two no-training deploy levers: **K-seed sampling with contact-consistent selection** (P6) and
  the **terminal veto + close-mask** (P3). Both behind flags.
- Benchmark `--nfe 3` and `--compile`; target L ≤ 0.5 s (P4). Set `governor.min_scale: 1.0` for all A/Bs
  (it is inert anyway — `sigma_max` 0.07–0.15 gives scale ≥0.95 in 100% of replans — but it must not be a
  free variable in a paper number).

**D4 Mon 09-01 — fine-tune FT-A + rig prep (rig 0 h, GPU ~$20)**
- FT-A: 3k steps from v5_6 (~6.5 h, $16) with whatever G1 selected. Default bundle if G1 is ambiguous:
  β-NLL contact loss (P5) + self-forcing CONTACT (P7.2) + intent/velocity dropout p≈0.3 (P2/H3) +
  `ema_decay 0.995` (the 0.999 EMA over a 3k run is 5–61% v4 interpolation — "monotone over 6 checkpoints"
  is what uncorrected EMA lag alone produces) + `cond_dropout 0`.
- Score FT-A on **replay**, not `terminal_eval`. Report raw **and** EMA per checkpoint.
- Rig dry-run checklist; pre-register the A/B schedule (placement cells drawn before the session).

**D5 Tue 09-02 — RIG SESSION 1 (rig 3.0 h)**
- **Centre cell only** (the demo-mean placement), waffles + Carton. Arm A = v5_6 as shipped; Arm B = v5_6 +
  {parity fixes, K-seed selection, terminal veto, NFE from E2, governor off}. 8 episodes/arm/task, **interleaved
  per placement**, both policies resident. Objective labels via the P8 rule + operator verdict; log damage.
- 30 min reserved: after each Arm-B failure, open the gripper and record a teleop **recovery demo from the
  arm's exact failure state** (`docs/recovery_demos_protocol.md` steps 3–5, skipping step 2).
- → **GATE G2**.

**D6 Wed 09-03 — intake + FT-B (rig 0 h, GPU ~$20)**
- Rollout intake: label, re-derive `actions.zarr`, fix hash, weight, place, manifest (P9).
- FT-B: 3k steps on demos + recovery demos with the G2-selected changes.

### Week 2 — close the loop on the teacher, then freeze

**D7 Thu 09-04 — RIG SESSION 2 (rig 4.0 h)** — the data session. Best arm from D5, waffles + one
touch-relevant task (egg for damage rate, or whiteboard). ~40 rollouts + **~40 recovery demos from the policy's
own failure states**. This is the best-evidenced lever in the literature: CR-DAgger (2506.16685) and
TER-DAgger (2603.04038) both report **+37 to +64 points from 50–100 corrections/task** at exactly our base-demo
scale; FlowDAgger (2607.08877) gets +0.17–0.67 from **5–20** corrections on a flow policy. Auto-trigger the
takeover the TER-DAgger way: `p_none` high **and** z below the demo grasp band **and** descent decayed.

**D8 Fri 09-05 — v6 fine-tune (rig 0 h, GPU ~$20)** — 3k steps from the best of {v5_6, FT-A, FT-B} on
demos + recoveries + auto-labelled rollouts, with per-episode weights (`EpisodeMeta.weight`, honoured at
`windows.py:403`; `w=1` demos, `0` failures, `1` recoveries, `clip(1/(p_task+0.2),1,3)` successful rollouts,
×2 window multiplier in the commit band, ×3 oversampling of the small rollout pool). Score on replay + the
held-out recovery demos (the only true on-policy validation set that exists).

**D9 Sat 09-06 — RECIPE FREEZE → GATE G3 (rig 0 h, GPU ~$165)**
- Freeze the teacher recipe. Relaunch `no_distill` if the recipe moved (~$90).
- Launch the **HID student**: ~5k steps from the frozen teacher, ~28 h, **$70**. Primary distillation loss =
  **predict the teacher's `cpk` from vision+proprio** (HapticVLA 2603.15257 style, a compact structured target)
  over raw hidden-feature HID; keep `w_ground` on demos only. Do **not** run the 30k default.
- Fix `distill_hid`/`finetune_hids` to load `manifest_split(root,"train")` — both currently train on the
  held-out `val` split.

**D10 Sun 09-07 — buffer / video-head ablation (rig 0 h, GPU ~$20)** — the λ_v = 0 / drop-video run. A
reviewer from the WAM community asks this in the first paragraph. A 3k fine-tune proxy is acceptable if a
from-scratch run does not fit; report it honestly as a proxy. Fix or drop the "droppable video" claim (P7).

**D11 Mon 09-08 — RIG SESSION 3 (rig 3.0 h)** — interleaved v6 vs v5_6 vs v4, waffles + the second task,
10 episodes/arm, centre cell + ±1 cell. Primary numbers = continuous (miss distance, z-at-close), secondary =
tactile-confirmed success. Collect a second recovery batch on failures.

**D12 Tue 09-09 — v7 (final teacher) + student eval (rig 0 h, GPU ~$20)**

**D13 Wed 09-10 — GATE G4: paper framing decision (rig 0 h)**
Results paper vs analysis paper (§4). Whichever it is, the method section gets rewritten to the *actual*
pipeline today: 4 tasks not 5, no occlusion, no fragility instrumentation, task-token conditioning not
"language instruction", contact-play SSL **never run on real data**
(`runs/tactile_pretrain/` is the synthetic smoke output and no launch line passes `--tactile-pretrain`).

### Week 3 — the numbers that go in the paper

**D14 Thu 09-11 — dry-run + schedule (rig 0 h)** — freeze checkpoints, generate the randomised interleaved
placement schedule, verify both/three policies load simultaneously.

**D15 Fri 09-12 — RIG SESSION 4A (rig 3.0 h)** and **D16 Sat 09-13 — RIG SESSION 4B (rig 3.0 h)**
The main matrix: **2 tasks × 3 arms (teacher, HID student, no_distill) × 25 trials**, arms interleaved per
placement from a schedule drawn in advance, single checkpoint per arm (say so in the paper).
Power: 0.3 vs 0.7 needs ~23/arm; 0.5 vs 0.8 needs ~38/arm; Fisher at 25/arm gives 18/25 vs 10/25 → p=0.045 and
15/25 vs 10/25 → p=0.26. Wilson half-width at n=25, p=0.5 is ±0.18. Report Wilson CIs + Fisher exact + the
paired per-placement differences, plus miss-distance / z-at-close distributions from `planner_trace.json`,
damage % on egg, and peak tactile normal force. ~150 valid episodes ≈ 6 h at ~3.5 min/episode with resets.

**D17 Sun 09-14 — RIG SESSION 5, ablations (rig 2.0 h)** — terminal-veto on/off and video-on/off, 15 episodes
each, teacher only. → **GATE G5: all results in, freeze.** Everything after this is writing.

**D18 Mon 09-15 — submit.** Reserve 1.0 rig-hour unspent through D17 for a re-run of one confounded cell.

### Gates

| gate | when | condition | if it fails |
|---|---|---|---|
| **G0** | D2 | replay E0 reproduces the recorded trace inside the 8-seed spread on ≥4 of 6 episodes | replay tool is invalid → debug for ≤4 h, else iterate on the rig with the terminal veto + K-seed as the only interventions and plan for the analysis paper |
| **G1** | D3 | ≥1 offline intervention moves predicted commit height by **>30 mm** at rig states | no training-side lever is identified → go deploy-side only (veto + K-seed + NFE + latency), and the paper's contribution becomes the diagnosis |
| **G2** | D5 | Arm B ≥ **3/16** tactile-confirmed grasps on waffles centre cell (vs Arm A's expected 0) | if 0/16 for both: stop tuning the policy, spend D6–D8 entirely on recovery demos + the scripted terminal controller; re-gate D9 |
| **G3** | D9 | best teacher ≥ **40%** on waffles centre cell over ≥10 interleaved episodes | do not launch the student ($70 + calendar); cut RQ3 and write the teacher + analysis paper |
| **G4** | D13 | teacher ≥50% on 2 tasks **and** student trained | choose the narrowed paper (§4) |
| **G5** | D17 | all matrix + ablation cells collected | report what exists with explicit n per cell; never pad |

---

## 4. What to cut, in this order, if things slip

1. **HID-S / `finetune_hids`** — cut now, unconditionally. Student-only, force-reward, AWR weights that mix
   normalised `cpk_d_disp` cumsums against raw-depth `tau` (`finetune_hids.py:58-63` vs `:79-81`), and it
   ignores `action_weight` so the 115 `_fail` demos get AWR-weighted action imitation. It is an "optional row".
2. **RQ2 as a headline** — the gate lead-time "money plot". The contact package lives on a **1.0 s** latent grid
   and the gate lookahead is 0.5 s while the replan period is 0.9–1.6 s, so "transients lasting tens of ms" is
   two orders of magnitude below the model's temporal resolution; `metrics.py:100-106` computes an **uncapped**
   lead; `event_f1` is off by one step and uses only `sensors[0]` while lead-times use both. No CASA (α=0)
   variant exists. Keep the gate as a diagnostic figure; report AUROC/PR at replan times against a proprio-only
   logistic baseline if anything.
3. **The 3×3 placement grid** → centre cell only. Measured close-time TCP xy σ is 3.3/4.6 mm (egg),
   5.4/4.5 (whiteboard), 12.1/10.1 (Carton), 19–21/8–15 (waffles), so ±40 mm corner cells are 2σ for waffles
   and 6–10σ for egg/whiteboard. The A/B is paired so placement OOD largely cancels, but the corner cells buy
   generalisation evidence we cannot afford and cost episodes we need for power.
4. **`vision_only` as an arm** → rewrite RQ3 as "distilled vs non-distilled student, both vision+proprio+F/T"
   and report the three raw rates with paired differences instead of `recovery_ratio`. This saves a 35-h run
   and an unbuildable denominator.
5. **The second task in the main matrix** → waffles only at 25/arm × 3 arms (≈4 rig-h instead of 6).
6. **DAgger-lite round 2** (D11's rollout collection) → keep the recovery demos, drop the second autonomous
   rollout batch. Note Q-Planning (2608.21204) shows success-only filtered SFT **plateaus** (55%/30% vs 90%/80%),
   so the autonomous batch was never the load-bearing part; the human corrections are.
7. **The video-head ablation** → keep the claim out of the paper instead of buying the row. If cut, do **not**
   call the system a world-action model; call it "a video-DiT backbone fine-tuned as a joint contact+action flow
   policy with an auxiliary video objective", and drop "droppable at inference" entirely (it is false as
   implemented, P7).
8. **The student entirely** → teacher-only paper: tactile WAM teacher + the closed-loop commitment analysis +
   the gate-driven terminal controller + the DAgger-lite curve. This is still a paper.
9. **Last resort — the negative-result / analysis paper.** If D13 arrives without a teacher that grasps ≥50% on
   waffles: "commitment failure in flow-matching chunk policies with privileged contact heads" — the E1–E9
   experiment set, the trace decomposition, the offline-vs-closed-loop dissociation (17 mm vs 0/26), the
   marginal-chunk analysis, and the terminal-veto fix. ViPRA (2511.07732) and HarmoWAM (2605.10942) make this
   a *recognised* failure mode of the joint-modelling family, so the framing must be "we confirm it on a
   tactile WAM, localise it to the terminal commit, and fix it with X" — not "we found a failure mode".

---

## 5. Refuted findings — do not re-raise these

Each was proposed by a lens and killed in the adversarial round. One line each, with the reason.

1. **"Post-close proprio is degenerate: success ≡ empty close for 3/4 tasks"** — `ur_state` carries full
   `tcp_pose` (incl. z) and the video shows the un-grasped object; a close at 65–120 mm is nowhere near the
   success distribution. The lift is OOD extrapolation, not state confusion. (The OBJ statistics behind it are
   real and are used in P8.)
2. **"The 3×3 / 40 mm grid is far outside the demo placement distribution"** — true for egg/whiteboard corners,
   but the A/B is *paired* per cell and pooling `batch_20260822` gives waffles σ_x 19.3 mm over a 101.7 mm span,
   so ±40 mm is ~2σ for the first A/B task; and the failure reproduced 13/13 on 08-20 with no grid at all.
   Kept only as the one-line cut in §4.3.
3. **"Contact targets on a 1.0 s latent grid ⇒ no aligned near-term contact token"** — the ACC gate label *is*
   "contact within 0.5 s" with probes at 0.075–0.5 s (`windows.py:345-359`), BCE-supervised, injected as a
   per-block key bias the ACTION queries see. The near-term representation exists; the model just doesn't act
   on it (that is P3).
4. **"`prev_chunk` intent is structurally cut off from ACTION tokens"** — false: only ACTION/CONTACT→VIDEO_GEN
   is blocked; OBS_*/VIDEO_COND read VIDEO_GEN, so intent reaches ACTION in 2 hops across 28 blocks
   (measured non-zero, exactly 0.000000 with all non-video queries blocked). Also reaches ACC directly.
5. **"`prev_cpk` is a pure prediction fed back as fact — self-confirming loop"** — v4/v5 trained with
   `--acc-two-pass`, which feeds the model's own sampled package, so train and deploy match by design; and the
   gate correctly read `p_none 0.99` during the air-close, i.e. it demonstrably did not self-confirm.
6. **"Failure demos contribute zero action loss ⇒ un-zero them"** — mechanically true, deliberately correct:
   `tools/intake_recovery.py:13-18` records that those episodes continue with an **empty gripper**; imitating
   them teaches the exact rig failure. The real gap is the absence of *corrective* targets (P3, §3 D7).
7. **"grasp-frac anchor includes post-close lift while the evaluator ends at close"** — arithmetic right,
   inference wrong: `--grasp-frac` defaults to 0 and v4 (no grasp weighting) fails identically; ending the
   chunk at close would phase-lock the close to a fixed step index and train onto the eval anchor.
8. **"`--event-band-weight 0` leaves the deploy `prev_cpk` event channel unsupervised"** — mechanically true but
   inert: two-pass training feeds `flatten_summary` of the model's **own** sample, the same path deploy uses, so
   there is no mismatch; and the v4 init never learned that band (MSE ~0.9 = chance) so weight 0 preserves the
   input distribution ACC was trained against.
9. **"Training ignores inference latency ⇒ add latency augmentation"** — the *mechanism* survives as P4, but
   this framing was rejected: the deploy half of the proposed fix (skip-head anchoring) is exactly what the team
   already tried and reverted for cause (`executor.py:88-94`), and the sim shows index-0 replay biases the
   terminal **down** (overshoot), the opposite sign of the observed stall.
10. **"`prev_cpk` semantics differ between the training proxy and deploy"** — true and *documented*
    (`acc.py:13-19`), with `two_pass` as the configured alternative; and `terminal_eval` already reproduces the
    deploy prev_cpk path, so it cannot be the offline-vs-rig discriminator.
11. **"Height is read from the image; proprio z is under-used (1.4σ)"** — the "weak" claim is unproven:
    OBS_PROPRIO is a full pinned conditioning frame with VIDEO_COND's token budget, and 190 mm sits at the
    *densest* part of the training z distribution (mean 0.187 m), which explains the tight 182–194 mm commit
    cluster just as well. Kept only as the cheap E5b probe; the salvageable nugget is that per-group (image-only)
    conditioning dropout genuinely does not exist.
12. **"`ur_state` normalisation over whole episodes dilutes the grasp-height signal"** — normalisation is
    per-channel and affine, absorbable exactly by the first `nn.Linear` of `URStateMLP` (verified numerically:
    a 4× z rescale with the matching column rescale gives max abs diff 0.0), so the proposed 2×/4× ablation is a
    training-time no-op; and a 65–120 mm error is 0.7–1.4σ, among the largest deviations, not a modest one.
13. **"'Sensor-free student' is false because the student consumes wrist F/T"** — literally true but disclosed:
    `pipeline.md` defines sensor-free as no tactile stream, wrist F/T is `source: ur_internal` (the arm's own
    current estimate, no added hardware), and every baseline consumes it identically so it cancels in the
    contrast. Adopt the wording **"fingertip-sensor-free"** and move on.
14. **"The anticipatory-contact claim is one flag away from invalid"** — guarded four ways: both pinned launch
    scripts pass `--acc-two-pass`, `train_teacher.py:203-217` hard-fails on model-config drift for
    `--init-weights`/`--resume`, `run_deploy` rebuilds from the checkpoint's own saved config, and a loud startup
    warning fires otherwise. The **missing CASA (α=0) knob** is real and is handled as cut §4.2.

---

## Completeness critic

- **No perception lens — camera extrinsics/lighting drift between demo collection and the rig is never checked.** Both `inference-parity.md:215-226` and `closed-loop-root-cause.md:172` propose a 1-minute extrinsics check and the plan drops it; a re-aimed RealSense or the 08-18/20 lighting change is a sufficient standalone cause of a vision-read height failing at exactly the terminal phase, and it would make every training-side result at G1 uninterpretable. → Add to D1/D2: compare the first cond frame of each 08-28 deploy episode against a same-task demo at the gated start pose (gripper pixel bbox + background SSIM); if it moved, recalibrate or re-collect before spending a rig hour.
- **Tactile sensor baseline drift is nobody's lens, yet the gel is now the success label.** Gel zeroing, wear and re-mounting between demo collection (through 08-22) and the September sessions were reviewed by no one; P8 makes sustained `mask_frac` contact the auto-label that gates G2/G3 and produces every number in the paper. Also, `run_deploy.py:470-482` still has no damage field, so D5's "log damage" and D15's "damage % on egg" have nowhere to go. → Diff per-session `fields_ds` depth/`mask_frac` baselines (demos vs 08-20/08-28 rollouts) on D2, add a pre-session gel-zero + press-check to the rig checklist, and add the damage field on D1.
- **The tactile success rule has no validated positive class on a policy-driven grasp.** P8 is validated on demos and on 39 known-failure rollouts; there are zero successful rollouts, so the thresholds (`c_hold ≥ 0.8`, 2 s hold, per-task `Z_MAX`) have never seen a true positive produced by the policy — and G2, G3 and the headline success rates all read off it. → Dual-label every D5/D7 episode (rule + operator verdict), publish the confusion matrix, and freeze the rule only after ≥10 human-confirmed policy grasps.
- **G2's "≥3/16" is not a decision a 16-episode session can make.** 3/16 vs 0/16 is Fisher p ≈ 0.11, and Arm B bundles four interventions, so passing tells you neither "it works" nor "which part". → Make the D5 primary continuous (median z-at-close / miss distance, where n=16 has power), keep 3/16 only as a descriptive tripwire, and state the G2 rule that way in the pre-registration.
- **The rig arms are unattributable bundles and only the veto is ever ablated.** D5 Arm B = parity fixes + K-seed selection + terminal veto + new NFE + governor off; D17 ablates terminal-veto and video only. The paper can therefore never name what made the teacher grasp. → Split D5 into veto-only vs veto+parity+K-seed, add K-seed on/off and parity on/off to D17, and pre-register which single intervention the abstract claims.
- **K-seed sampling contradicts the L ≤ 0.5 s latency target and was never profiled on the deploy machine.** `research/01_closed_loop_chunk_execution.md:50` explicitly warns K=4–8 at NFE 5 through a 2B DiT "may not fit the 0.9 s replan budget … needs profiling before committing rig time"; the "1 s per sample" figure is compute3's 5090, not the rig NUC, and P4 simultaneously demands latency be cut. → Profile K × NFE on the actual deploy GPU on D3 as a gate on the lever; fall back to K=2–3 or score only every other replan.
- **E8 and E7 cannot change any decision as scheduled.** E8 chooses between index-0 / skip-head / RTC prefix-inpainting, but no day implements prefix inpainting (D1 = parity+safety, D3 = K-seed+veto+latency) and no arm carries an executor change; E7's guidance >1 doubles replan cost and is excluded from every arm by the latency target. → Either put "RTC prefix-inpaint behind `--parity-fixes`" into D3 with a rig slot, or cut E8/E7 outright and give the hours to E0/E2 seed coverage.
- **E9 — the test of the paper's own premise — is consumed by no gate.** Its stated consequence ("tactile-null ≈ real ⇒ the tactile-teacher story is dead, pivot") appears in no row of G0–G5 and is not referenced at the D13 framing gate, so the plan can spend ~$160 on the student and `no_distill` after the premise has already failed. → Add gate G1b at D3 with a numeric threshold (e.g. |Δ endpoint| < 3 mm and |Δ close_step| < 1 step under tactile-null ⇒ pivot to the commitment/analysis paper immediately).
- **The rig-hour budget does not add up.** D15+D16 book 6.0 h for 150 episodes at the plan's own 3.5 min/episode = 8.75 h; 16/26 of the last session's episodes were invalid and no attempt-to-valid margin exists anywhere; D7 packs ~40 rollouts *and* ~40 teleop recovery demos into 4.0 h (3 min each including reset); D14's dry-run is priced at 0 rig-h but needs the rig to verify three policies load. Planned 18 h + 1 h reserve already sits at the ceiling of a 15–20 h budget. → Re-price everything in *attempts* with a ≥1.4× invalid factor, adopt §4.5 (waffles only, 3 arms × 25) as the plan of record rather than a contingency, and book 0.5 h for D14.
- **The GPU ledger does not add up and never states how many H100s are rented.** Day lines sum to ≈$375–465 against a claimed "≈$330, worst case $420" (D2 is labelled ~$10 while launching a $90 run); D9 runs a 35 h `no_distill` relaunch concurrently with a 28 h student, plus D10/D12 fine-tunes — 2–3 simultaneous instances — and 34 h rented jobs have no preemption/resume story. → Restate the ledger per instance, book 2 concurrent H100s from D9, and require resume-safe checkpointing (and a resume smoke test) on every run over 12 h.
- **Arm parity — training data and deploy controller — is unstated, and it is the reviewer's kill-shot on RQ3.** The teacher is fine-tuned on recovery demos and auto-labelled rollouts harvested from its own failures, while `no_distill` was launched on the D2 recipe and the student is distilled at the D9 freeze; nothing in the plan asserts the three arms share one dataset, and nothing says the terminal veto / K-seed selection runs identically on all three. If either differs, the reported gap measures data or a scripted controller, not distillation. → Pre-register "identical training set and identical deploy-time controller across teacher, student and `no_distill`", and re-launch whichever arm violates it before D15.
- **There is no non-WAM baseline anywhere in the plan.** All three arms are the same Cosmos-2B flow policy, so the paper cannot answer "would a plain ACT / diffusion policy on these 1115 demos do better?", which is the first question an ICRA reviewer asks of a 2B video-DiT teacher; the related safety gap is the same shape — making the z-floor a clamp (§1.11) removes the last backstop exactly as every intervention pushes the policy lower, with no wrist-F/T or tactile force abort, protective-stop/E-stop still invisible to the safety layer (`deploy-safety.md:117`), no retry cap on the veto's open→re-descend loop, and no written RTDE handover procedure for D7's auto-takeover (whose transient will otherwise be recorded as supervised demo action). → Train one ACT/DP baseline on compute3 (free) and report it at least offline on replay; and land force-abort + protective-stop watch + veto retry cap + the handover procedure on D1, before any session is counted.
