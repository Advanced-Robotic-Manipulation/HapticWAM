# PHANTOM review — lens: MODEL + LOSSES (2026-08-28, HEAD 3539988)

Scope: `phantom/model/{rf,phantom_dit,sequence,acc,attention_bias}.py`, `phantom/model/ace/*`,
`phantom/model/hht/*`, `phantom/config/model.py`, the train programs' use of them, and the
deploy-side consumers (`inference/policy.py`, `deploy/planner.py`, `deploy/executor.py`)
where they determine what the model actually sees. Everything below was read in the code;
numbers marked **[measured]** come from small local runs (tiny cosmos model on the Mac,
pure-numpy simulations of the packers, the v4/v5 training logs and the two offline
terminal-eval JSONs in the scratchpad, and the real `teacher_v5_batch0822/teacher_003000.pt`
checkpoint pulled from the hub). Claims marked **[believe]** / **[educated guess]** are
inferences I could not run here (need the data + a GPU).

Run configuration actually used (from the v5 checkpoint's saved `configs.model`, and the v4 log):
`rope_time_mode=time_true`, `cond_dropout_p=0.1`, `action_t_max_of_two=True`,
`acc.self_anticipation=two_pass`, `nfe=5`, loss weights action 1.0 / video 0.1 / contact 1.0 /
wrist 0.5 / event 0.5 / gate_bce 0.2 / sigma_reg 0.01; v5: lr 2e-5, new modules 6e-5,
3000 steps, batch 4x2, grad_clip 1.0, EMA 0.999, `--event-band-weight 0`, `--grasp-frac 0.3`.

---

## 0. Answers to the lens questions, in one paragraph each

**Through which path do ACTION tokens see tactile/contact?** Directly. The only attention
restriction is `SequenceLayout.structural_attn_bias()` (sequence.py:190-202): CONTACT and
ACTION *queries* get -1e4 on VIDEO_GEN *keys*; nothing else is masked. So ACTION tokens attend
VIDEO_COND, OBS_GEL, OBS_MECH, OBS_PROPRIO, CONTACT (co-denoised) and ACTION keys in every one
of the 28 blocks. The ACC only adds a scalar key bias `lambda_block * beta(p_evt) * g` on the
haptic-group keys (attention_bias.py:196-203). The head is not structurally blind. It is,
however, *weakly trained* to use observations, for three reasons quantified below (§1, §3, §4),
and the ACC injection is numerically almost inert (§7): in the real v5 checkpoint
`|lambda| <= 0.109` over all 28 blocks, `softplus(beta_raw) ~ 1.3`, so the largest possible
attention-logit bias is `0.109 * 1.3 * 1.0 = 0.14` nats (a 15 % relative re-weighting of a
subset of keys in the strongest block) **[measured]**.

**Is cond-dropout / guidance applied to the right conditions?** Mostly, with one design flaw:
the "null" observation is the *dataset mean*, not a null token (§5). Guidance is 1.0 at deploy,
so the 10 % dropout currently only costs action-gradient budget and blurs the obs->action map
around the mean state; guidance > 1 would extrapolate away from "the policy at the mean state",
not from an unconditional policy.

**Do loss weights let the video term dominate?** No. **[measured]** on the tiny model the
weighted video term is < 0.1 % of the squared gradient norm on the shared LoRA. The term that
dominates is the *contact heteroscedastic NLL*: 94.6 % of the squared LoRA-gradient norm at
init, and ~100 % (`|g| = 82.2` vs action `0.72`, i.e. 114x) once `log sigma` sits at the value
the v4/v5 logs show it converged to (-2.35). See §1.

**Is the flow-matching target/timestep sampling sensible for a 16x7 chunk?** The v-target is
fine; the *noise geometry* is not: every action value is tiled over 160-192 latent cells with
i.i.d. noise, so the strip-average recovers x0 at r=0.84 already at t=0.90 **[measured]**; the
`action_t_max_of_two` draw (median t=0.896, P(t<0.7)=5 %, P(t<0.6)=1.3 % **[measured]**) then
concentrates supervision in the only band where obs matter, but the sampler's last two steps
(t=0.769, 0.556 under rf_shift=5) run at t-levels the action head has hardly seen, and the
final output *is* `x0_pred(t=0.556)`. See §4.

**Does prev_cpk feedback create a self-confirming loop?** Not through the ACC — its injection
is too small (§7) and the summary's event dims are unsupervised noise (§8). The loops that do
exist are (a) `prev_chunk` at deploy = the previous *plan*, not executed actions (§2), (b) the
arm's own velocity in `ur_state` (§2), and (c) *within* a replan, the ACTION head conditioning
on co-denoised CONTACT/VIDEO tokens that carried GT future at training time and carry the
model's own imagination at deploy (§3).

**ACC two-pass?** Works as designed but the inner pass's ACC input is still the GT-noised
summary (rf.py:163-175), so the "own prediction" the outer pass sees is GT-informed; the RQ2
lead-time claim is not GT-free (§6).

**What would produce systematic under-commit?** Offline, on demo states, there is no
under-commit (te_v4 z_end -4.7 mm, te_v5 -2.4 mm; commit_ratio median 1.18-1.30; the gripper
is the only shrunk channel: predicted max aperture 0.42 vs GT close 0.49). The rig under-commit
is closed-loop, and the model-side amplifiers are §2 (velocity/intent copycat with a
self-referential deploy input), §4 (noise-driven per-episode velocity offsets under persistent
noise, and few-NFE at under-trained t), and §1/§3 (obs->action coupling trained weakly, so the
policy falls back on shortcuts that are self-generated at deploy). EMA and normalisation are
clean (§9).

---

## 1. HIGH — the contact heteroscedastic NLL owns the shared trunk's gradient

`losses.contact_hetero_nll` (losses.py:54-78) is `d_g / sigma_g^2 + log sigma_g^2` per sigma
group, with `sigma` from `SigmaHead` clamped to `log sigma in [-5, 3]` (heads.py:34). The
gradient it sends into the trunk through `x0_pred` is `2 r / sigma^2` per element. Both the v4
and v5 logs show where sigma went: `sigma_reg = mean(log sigma^2) = 5.4-6.5` and
`contact_nll = -3.1 .. -3.7`, i.e. `log sigma ~ -2.35`, `1/sigma^2 ~ 110` (at the NLL optimum
`d = sigma^2`, so `1 + 2 log sigma = -3.7` -> `log sigma = -2.35`, consistent).

**[measured]** per-term gradient norm on the 40 LoRA tensors of the tiny teacher
(synthetic data, one batch of 4, each term back-propagated alone with the same noise seed):

| term (weight) | init `log sigma~0` | trained regime `log sigma=-2.5` |
|---|---|---|
| action_v_mse (1.0) | 0.718 (1.3 %) | 0.718 (0.0 %) |
| video_v_mse (0.1) | 0.046 (0.0 %) | 0.046 |
| contact_nll (1.0) | 6.01 (**94.6 %**) | 82.2 (**~100 %**) |
| contact_event_mse (0.5) | 1.03 (2.8 %) | 1.03 |
| wrist_mse (0.5) | 0.69 (1.3 %) | 0.69 |
| event_ce (0.5) | 0.10 | 0.10 |
| acc_* | 0 (no LoRA path) | 0 |

(share = squared-norm share). The probe script is in this file's appendix. It is a tiny random
model, so the *absolute* numbers are not the real run's, but the `1/sigma^2` factor is
structural and the log confirms sigma. With `grad_clip=1.0` on the total norm, the action
gradient is additionally scaled by ~1/80 before Adam sees it; Adam's per-parameter rescaling
does not restore direction. `sigma_reg = 0.01 * log sigma^2` cannot hold sigma up: the NLL
gains ~1 nat per unit of `log sigma`, the reg costs ~0.05.

Consequence **[believe]**: the LoRA (the only thing that can teach the frozen 2B trunk to route
observations into the ACTION tokens) is trained as a contact-field forecaster with the action
objective as a ~1-3 % perturbation. This is the most parsimonious explanation for the weak
obs->action coupling visible everywhere: `sampled_dir_cosine` 0.62 (v4) / 0.82 (v5) on val,
consecutive-replan direction cosine 0.16-0.35 on the rig, and a policy that ignores proprio
height while its tactile gate says p_none=0.99.

**Fix (3 lines, then a 2-3k-step fine-tune):** in `contact_hetero_nll` train the trunk on the
plain residual and the sigma head on the *detached* residual —
`terms.append(d_B_Tc.detach() / log_var.exp() + log_var)` for the sigma path plus
`w_c * d_B_Tc` (or beta-NLL: multiply the NLL term by `sigma^(2*beta).detach()`, beta=0.5-1,
Seitzer et al. 2022). Keep the governor's sigma calibrated (it still trains on the detached
residual). Verify with the same probe on the real checkpoint (10 min on the GPU box: per-term
LoRA grad norms should be within 10x of each other) and with `val_action_v_mse`,
`sampled_dir_cosine`, and `terminal_eval` before/after.

Related: `wrist_region_mse` (losses.py:81-86, weight 0.5) double-supervises channel 15, which
is also the NLL's `wrist` group — harmless, but drop one when rebalancing.

## 2. HIGH — deploy `prev_chunk` is the previous *plan*, training `prev_chunk` is the executed past; plus raw velocity in `ur_state`

Training (windows.py:277-279): `prev_chunk` = the 16 recorded, executed delta-actions on
`(t0-1.6 s, t0]`. Deploy (policy.py:101-111): `prev_chunk = normalize(prev_plan.actions)` — the
previous replan's 16 *proposed* actions, whose `action_times[0] = obs.t + latency`; at the next
snapshot (~0.9 s later, the loop replans back-to-back) essentially none of them has executed
yet, and what does execute is reshaped by rebasing, blending, the governor and the rate limit
(executor.py:69-112, 182-203). First replan: `prev_chunk = normalize(zeros)` = "I was
stationary for 1.6 s", a state the training windows almost never contain (valid_range starts
`max(0.25, 1.6, ...) = 1.6 s` into an episode, windows.py:170-172). `planner.py:284-291`
records this as DEFERRED.

Where the intent goes: `_action_adaln_embeddings` (phantom_dit.py:131-157) feeds it through the
pretrained action-AdaLN of the VIDEO_GEN frames, and `AccGate.intent_mlp` (acc.py:62, 84).
ACTION queries cannot attend VIDEO_GEN keys, but VIDEO_COND/OBS queries can, and ACTION attends
those — a 2-hop path that 28 blocks make fully usable. Independently, `ur_state` carries
`qd(6)` and `tcp_speed(6)` (12 of 26 dims, windows.py:266-271; planner.py:151-153) — the arm's
current velocity, which in smooth teleop demos predicts the next chunk almost perfectly.

Failure scenario **[believe]**: the policy learns "continue the recent motion" (causal
confusion / copycat). In closed loop the recent motion is its own previous output, filtered by
the executor: a slow first chunk (zero-motion prev_chunk, arm at rest) -> slow arm velocity and
a slow plan fed back as "what I did" -> slower plan. Rollouts descend at 33 vs 45-50 mm/s and
decelerate early; this loop is a direct model-side amplifier of that.

**Fix:** (a) deploy: keep a ring of the last 16 *executed* action-grid deltas (the executor
already emits exactly these through `record_action`, executor.py:205-212) and feed that as
`prev_chunk`; before the first plan, seed it from the measured TCP deltas of the last 1.6 s
(zeros are correct only if the arm was truly still, which after homing it is — but then
training must contain such windows, see (b)); (b) training: intent/velocity dropout — with
p~0.3 zero `prev_chunk` and `qd/tcp_speed` (normalized-zero = mean, so better: replace with a
learned null, see §5) so the policy cannot lean on them; also let `valid_range` start at 0.3 s
so at-rest starts are in-distribution. (c) Diagnostic, offline, 20 min: on val terminal
windows scale `qd`/`tcp_speed` by 0.5 and `prev_chunk` by 0.5 and measure the sampled chunk's
descent speed; if it follows the input, the loop is confirmed.

## 3. HIGH — co-denoised GT future is a training-time shortcut for the ACTION head (exposure bias), and the "droppable video" mask leaks

At training, CONTACT frames hold the packed GT future package noised at the *same* t as the
video, and ACTION frames at `max(t, t')` (rf.py:270-276), so `t_contact <= t_action` always.
The packed contact channels are heavily tiled (event bands: ±1 over 240-280 cells;
wrist/wrench: 6 values over 2x3 blocks of ~200 cells; slip: half-planes of 640 cells;
d_fz/mask: bilinear 36x48 -> 32x40), so their strip averages beat the noise early.
**[measured]** with the real packers (lat 32x40, optimal linear read-out `x_t/(1-t)`):

| t | event class acc | wrist r | wrench r | slip r | action r |
|---|---|---|---|---|---|
| 0.97 | 0.46 | 0.37 | 0.42 | 0.65 | 0.40 |
| 0.95 | 0.69 | 0.63 | 0.62 | 0.82 | 0.59 |
| 0.92 | 0.91 | 0.79 | 0.80 | 0.92 | 0.76 |
| 0.90 | **0.98** | **0.86** | 0.86 | 0.95 | 0.84 |
| 0.85 | 1.00 | 0.94 | 0.93 | 0.98 | 0.92 |

At the ACTION head's median training noise (t=0.896) the GT "onset at +1/+2/+3 s", the future
wrist F/T and slip are readable at ~98 % / r=0.86 from the CONTACT tokens it attends. The
VIDEO_GEN frames (true future frames, noised) are reachable by the same 2-hop OBS path as in
§2 because `structural_attn_bias` only restricts CONTACT/ACTION queries (sequence.py:195-202;
`test_structural_bias` asserts "video queries unrestricted" and nothing about OBS). So during
training the cheapest way to lower `action_v_mse` is: read the descent timing from the co-noised
GT future. At deploy those tokens are the model's own jointly-denoised imagination — the mean-
regressed E[contact|obs] injected at step 1 (§4) — so the action head follows its imagination
rather than the observations. This is the standard teacher-forcing / exposure-bias gap; the
paper's thesis ("anticipated contact closes the loop") needs the coupling, but it needs it to
be trained on the model's *own* anticipation.

**Fixes (pick one, both need a fine-tune):**
1. Group-causal mask: conditioning tokens (VIDEO_COND, OBS_*) attend only conditioning tokens;
   ACTION attends cond + ACTION (+ CONTACT if the coupling is wanted); CONTACT attends cond +
   CONTACT (+ ACTION); VIDEO_GEN attends everything. One function (`structural_attn_bias`);
   the flex BlockMask follows automatically. Bonus: cond-token hidden states no longer depend
   on x_t, so their K/V can be computed once per replan and cached across the 5 NFE steps
   (roughly halves replan latency).
2. Self-forcing: the two-pass already produces the model's own `pred.cpk` per training window
   (rf.py:163-168); pack *that* into the CONTACT x0 that the ACTION frames are denoised
   alongside (keep GT as the CONTACT loss target). Zero extra forward passes.
Diagnostic first (30 min offline): run `terminal_eval` twice with the CONTACT frames
(i) cond-pinned to GT and (ii) cond-pinned to zeros; if endpoint/z errors move by >5 mm the
action head is leaning on contact tokens it will never have at deploy.

## 4. HIGH — 5-step Euler with rf_shift=5 on strip-tiled actions: decisions at t>=0.95, output at an untrained t, persistent noise = per-episode velocity bias

Schedule (rf.py:430, `_time_shift` with s=5): `ts = [1, 0.952, 0.882, 0.769, 0.556, 0]`
**[measured]**. The last Euler step is `x_0 = x_t - 0.556 v_pred`, i.e. the sample is exactly
`x0_pred(t=0.556)`. ACTION-frame training t under `action_t_max_of_two` (config/model.py:67):
median 0.896, P(t>0.9)=48 %, P(t<0.7)=5.0 %, **P(t<0.6)=1.3 %** **[measured]** — the two
steps that produce the final chunk run where the action head almost never trained.

Noise geometry (packing.py:276-288, `ActionPacker`): 7 strips of width 5-6 over 32 rows ->
160-192 i.i.d. cells per action value; the strip-mean noise has std 0.072-0.079. The
likelihood std of x0 read from `x_t` is `t*0.072/(1-t)`: 1.44 (t=0.952), 0.54 (0.882),
0.24 (0.769), 0.09 (0.556) **[measured]**. With the obs-conditional residual std ~0.37
(`val_action_v_mse` 0.14), the Bayes weight the net should put on the `x_t` reading is
0.06 / 0.32 / 0.71 / 0.95 at the four steps. So: step 1 (t=1, pure noise) writes E[x0|obs];
steps 2-3 mix in the strip-mean noise at 6-32 % weight; steps 4-5 mostly copy what is there.
The strip-mean noise never averages out within an episode when `--persistent-noise` is on
(rf.py:405-411): every replan re-uses the same 16x7 strip-mean offsets, which act as a fixed
per-step velocity offset of order 0.1-0.5 sigma (0.7-3 mm/step, 7-30 mm/s on z). That is a
concrete mechanism for "one seed descended at full speed, the others did not" and for a
30 % descent deficit that depends on the seed, not the scene. With fresh noise it instead
re-rolls the plan direction every replan (the rig's 0.16-0.35 consecutive cosine), which says
the noise term is *not* small relative to the obs-conditioned term at rig states.

**Fix, cheapest first:**
1. Today, no retraining: `tools/terminal_eval.py --nfe 1 / 2 / 5 / 10 --seeds 4`, reporting
   mean *and across-seed std* of `z_end_err`/`endpoint_err`. `--nfe 1` gives
   `x_0 = eps - v_pred(eps, t=1) = E[x0|obs]` — a deterministic conditional-mean policy with no
   noise-driven offset and ~5x lower replan latency (tighter loop = less compounding). Put the
   winner (nfe 1 or 10) in the rig A/B; run persistent vs fresh noise as a condition, not a
   default.
2. Retrain-level: draw the ACTION (and CONTACT) noise *per strip* (`eps_action =
   ActionPacker.pack(randn(B,H,A))`) in both `training_step` and `sample`, so the per-value SNR
   follows the scalar schedule and the model is actually trained at the t the sampler uses;
   then drop `action_t_max_of_two` and use an unshifted (or shift 1-2) schedule for the ACTION
   frames (per-frame t is already supported by `t_B_T`).

## 5. MEDIUM — the conditioning-dropout "null" is the dataset mean, not a null token

`_null_obs_batch` (rf.py:213-234) zeroes `gel, fields, contact_state, wrist, ur_state,
reactive` in *normalized* space and blacks the video-cond latent. `NormStats.normalize` is
`(x-mean)/std` (schema.py:127-131), so zero = the per-dim dataset mean: `ur_state` null = the
mean joint configuration at the mean velocity, `wrist` null = the mean F/T window, `fields`
null = the mean tactile field, `gel` = mid-grey. These are plausible in-distribution
observations, not "absence of observation". Effect: 10 % of the action gradient (already
starved, §1) teaches the model that the *typical state* maps to the marginal action
distribution — a blur exactly around the states the policy visits most; and guidance > 1
(`sample`, rf.py:445-455) would extrapolate away from "the policy at the mean state", so the
guidance knob does not do what its docstring says. `text` is nulled correctly.

**Fix:** learned null embeddings per observation group (one extra `nn.Parameter` per HHT
encoder output, swapped in under dropout; for `ur_state` also usable for the velocity dropout
in §2), or set `cond_dropout_p=0` for the next fine-tune since deploy runs guidance 1.0.

## 6. MEDIUM — two-pass ACC still sees GT (one level removed); RQ2 lead-time is not GT-free

`_acc_inputs_train` (rf.py:150-183): outer pass -> `sample(nfe=2)` -> inner `build_x0` ->
`_in_anticipation_pass=True` -> the *inner* ACC input is `flatten_summary(gt_cpk) + 0.1 noise`.
The inner sample's gate/p_evt (and, through the key bias, its CONTACT prediction) are thus
GT-informed, and the outer pass's "own previous prediction" inherits that. The docstring in
acc.py:13-19 and the `--acc-two-pass` help text claim the gate "never sees leaked GT". Deploy's
first replan uses the same inner pass but with all-zero placeholders (policy.py:130-145) — a
different distribution again.

**Fix:** terminate the recursion with a *no-contact* summary (zeros package, event=none) instead
of the noised GT — that is exactly what deploy's first replan effectively feeds — and report
lead-time from a model trained that way. Also note the inner pass costs 2 NFE of the full net
per training sample (~30 % of step time) for a summary vector the ACC then weights at
`|lambda*beta*g| <= 0.14` nats (§7).

## 7. MEDIUM (paper validity) — the ACC attention injection is numerically almost inert

Real v5 EMA weights **[measured]**: `phantom_bias_lambdas` over 28 blocks = max 0.109, median
~0.03, six blocks negative; `softplus(beta_raw)` = 1.31-1.44. The key bias added to attention
logits on the haptic group (attention_bias.py:65-76, 196-203) is therefore at most
`0.109 * 1.44 * g <= 0.16` nats, i.e. a <=17 % relative up-weighting of OBS_MECH/OBS_GEL/
CONTACT keys in the strongest block and ~3 % in a typical one. `lambda_bias_init=0` with the
LoRA LR (2e-5 in v5, 1e-4 in v4, `lr_new_modules` for phantom_* so 6e-5 / 3e-4) never moved
them far. The ACC is in effect an auxiliary classifier whose output is logged and used by the
governor/UI; it does not materially route attention, and the `prev_cpk` feedback loop through
it cannot be strong (the "self-confirming imagination" worry through prev_cpk is answered: not
via ACC). The paper's ACC mechanism claim needs either an ablation showing the bias matters
(lambda := 0 at inference, compare) or a re-parameterisation that lets it matter
(`lambda_bias_init` ~0.5-1.0, or a multiplicative gate on the haptic tokens' values instead of
a logit bias).

## 8. LOW — the packed EVENT band is unsupervised in both shipped teachers; its unpack feeds ACC

v4 predates `event_band_mse`; v5 ran `--event-band-weight 0`. Logs: `contact_event_mse ~ 0.99`
at the end of v5 = the variance of ±1 targets = chance. `ContactPacker.unpack` (packing.py:222-
226) still turns that channel into `cpk.event` probabilities, `flatten_summary` puts them in
the first 5 of 33 summary dims (packing.py:252), and both the two-pass and deploy `prev_cpk`
paths feed them to `AccGate.cpk_mlp`. Consistent between train and deploy, but 15 % of the
ACC's prev-package input is noise. Fix: either supervise the band at 0.1-0.2 in the next
fine-tune or drop the event dims from the summary (the EventReadout logits are the trained
event signal; use those).

## 9. Things that are fine (checked, no finding)

- EMA (`common.EMA`, decay 0.999 over trainable params, initialised from the loaded weights
  for a fine-tune; deploy loads EMA by default, run_deploy.py:56-67). v5 EMA vs raw lambdas
  agree to 1e-3 — nothing stale.
- Normalisation: action mean ~0, per-step stds x 2.9 / y 7.2 / z 6.7 mm, gripper abs 0.42±0.28
  (scratchpad norm_stats.json); no shrinkage from normalisation; `--init-weights` refuses
  norm-stats drift (train_teacher.py:237-253).
- The action loss uses only the 4 live channels (rf.py:316-318); the failure demos get
  `action_weight=0` (windows.py:403, group_velocity_mse weighted mean).
- RoPE `time_true` positions [0.4,0.8,1.2,1.6] for ACTION, CONTACT 1..3, OBS 0; the model is
  rebuilt from the saved config at deploy (`PhantomModelConfig.from_dict`).
- DDP generator seeding per rank (common.py:748-751) — fixed 08-26.
- The `_null_cond_latent` is encoded standalone, consistent with `encode_gen=False` at deploy
  (causal Wan VAE) — parity verified by the team's deploy-parity batch, not re-verified here.
- Video weight 0.1: its LoRA gradient is ~6 % of the action term's — irrelevant either way.
- SigmaHead/EventReadout read the *last* NFE step's hidden state (t=0.556, rf.py:458-462),
  which is in the 11 % tail of the CONTACT training-t distribution (single draw, median 0.83);
  minor calibration drift of the governor sigma / diag p_evt at deploy. Low.

## 10. Missing experiments (all cheap; none needs the rig except E5)

- **E1** per-term LoRA grad-norm probe on the real v5 checkpoint with a real batch (appendix
  script; 10 min). Confirms §1 before spending a fine-tune on it.
- **E2** `terminal_eval --nfe {1,2,5,10} --seeds 4`, report mean and across-seed std of
  z_end / endpoint; add `--persistent-noise` as a flag there so the per-episode offset can be
  measured offline. Decides the rig's nfe/noise condition (§4).
- **E3** input-sensitivity on val terminal windows (one script, ~30 min GPU): (a) `ur_state`
  tcp z +50 mm -> does cum z at chunk end descend ~50 mm more? (proprio usage / z-blindness);
  (b) `qd`,`tcp_speed` x0.5 and `prev_chunk` x0.5 -> does predicted speed follow? (§2);
  (c) CONTACT frames pinned to GT vs zeros (§3); (d) OBS_GEL/OBS_MECH zeroed (tactile usage);
  (e) `lambda := 0` (§7). Each answers a paper question too.
- **E4** 2-3k-step fine-tune from v5_6 with: detached-sigma contact loss (§1), intent/velocity
  dropout (§2), event band 0.1 (§8), cond_dropout 0 (§5); re-run E2/E3 and the terminal eval.
- **E5** rig: executed-actions `prev_chunk` (§2) + nfe from E2, A/B against v5_6 as-is.

---

## Appendix — probe scripts used (run from the repo root with `.venv/bin/python`)

Per-term LoRA gradient norms (tiny model; swap in `build_model(... tiny=False)` +
`load_phantom_checkpoint` + a real `WindowDataset` batch on the GPU box):

```python
PYTHONPATH=tests python - <<'EOF'
import torch, tempfile, pathlib
from phantom_test_utils import make_hw
from phantom.config.paths import load_paths
from phantom.train.builder import build_model
from phantom.data.synthetic import SyntheticEpisodeGenerator
from phantom.data.windows import WindowSampler
from phantom.data.schema import NormStats
from phantom.train import common as C
hw = make_hw(tactile={"field": {"h": 48, "w": 64}, "raw_img": {"h": 60, "w": 80, "c": 1},
                      "infer_img": {"h": 60, "w": 80, "c": 1}, "rate_hz": 30.0},
             cameras={"scene": {"color": {"h": 60, "w": 80, "c": 3}, "fps": 10.0}},
             recording={"field_ds": {"h": 24, "w": 32}, "field_ds_rate_hz": 30.0, "keyframe_rate_hz": 5.0,
                        "keyframe_ds": {"h": 24, "w": 32}, "infer_img_rate_hz": 10.0, "zarr_chunk_frames": 16},
             derived={"cpk_downsample": 4}, wrist_ft={"window_s": 0.1})
pm = build_model(hw, load_paths(), student=False, tiny=True, load_base=False)
root = pathlib.Path(tempfile.mkdtemp())
SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(root, task="grasp_slip", duration_s=8.0)
ds = C.WindowDataset(root, WindowSampler(hw, pm.bb, NormStats.identity()), windows_per_episode=4)
batch = C.collate_windows([ds[i] for i in range(4)]); rf = pm.rf; rf.train()
lora = [p for n, p in rf.named_parameters() if "lora_" in n and p.requires_grad]
w = pm.mc.loss
weights = {"action_v_mse": w.action, "video_v_mse": w.video, "contact_nll": w.contact,
           "contact_event_mse": w.event, "wrist_mse": w.wrist, "event_ce": w.event,
           "acc_gate_bce": w.gate_bce, "acc_event_ce": w.event, "sigma_reg": w.sigma_reg}
def term_grad(term):
    for p in lora: p.grad = None
    rf._gen.manual_seed(0); parts = rf.training_step(batch)
    (weights[term] * parts[term]).backward()
    gs = [p.grad.float() for p in lora if p.grad is not None]
    return float(torch.sqrt(sum((x ** 2).sum() for x in gs))) if gs else 0.0
with torch.no_grad():   # emulate the trained regime seen in the logs (log sigma ~ -2.35)
    rf.phantom_sigma_head.net[3].weight.zero_(); rf.phantom_sigma_head.net[3].bias.fill_(-2.5)
for k in weights: print(k, term_grad(k))
EOF
```

Readability of co-noised CONTACT/ACTION frames vs t (pure packers, real geometry):
see the table in §3; script = pack a random package/chunk with `ContactPacker`/`ActionPacker`
built from `configs/hardware.nuc.yaml` + `BackboneConfig()`, form `x_t=(1-t)x0+t*eps`,
`unpack(x_t/(1-t))`, correlate with the truth.
