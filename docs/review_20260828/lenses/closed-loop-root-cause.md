# PHANTOM review — lens: CLOSED-LOOP ROOT CAUSE

Repo `~/GitHub/phantom` @ `3539988`, nothing modified. Read: `phantom/data/windows.py`, `phantom/model/{rf,phantom_dit,sequence,acc,attention_bias}.py`,
`phantom/model/ace/packing.py`, `phantom/model/hht/{hht,encoders}.py`, `phantom/config/{model,backbone,hardware}.py`,
`phantom/inference/policy.py`, `phantom/deploy/{planner,executor,runtime,governor}.py`, `phantom/scripts/run_deploy.py`,
`phantom/train/{common,train_teacher}.py`, `phantom/data_collect/session.py` (action recording), `tools/{terminal_eval,rig_trace_decompose}.py`,
`configs/{hardware.nuc.yaml,start_poses.yaml}`, the checkpoint norm stats (`scratchpad/norm_stats.json` = hub `dataset_v3_packed/norm_stats.json`,
identical to the v4/v5 checkpoint blobs per the parity lens), `docs/rig_session_v5.md`, `docs/training_playbook.md`, `docs/recovery_demos_protocol.md`,
and the earlier lens `scratchpad/review/inference-parity.md` (which read the 37 rig traces on compute3 — I reuse its measured numbers and
mark them "[parity-lens]"). Rig episodes are NOT on this machine (the `scratchpad/deploy20260820/*` zarr dirs are empty skeletons), so
everything below is code + demo statistics + two small simulations/probes I ran here:

* `scratchpad/review/sim_velocity_loop.py` — a demo-faithful 1-NN imitation policy over (z, v_z, gripper) driven through the exact deploy
  loop (latency L, back-to-back replans, index-0 playback rebased on `_last_cmd`).
* `scratchpad/review/probe_leak.py` — tiny random-init teacher: does information the structural mask is supposed to hide from ACTION
  queries reach the ACTION-frame output anyway.

Certainty labels: **[certain]** read in code / computed here; **[believe]** strong inference; **[guess]** hypothesis with a proposed test.

---

## 0. Executive summary

The phenomenon to explain (`docs/rig_session_v5.md`, CONTEXT.md, [parity-lens] §2.3): during the descent the commanded chunk *heads* are
−26, −23, −28, −22, −14, −3 mm per 0.96 s replan (≈ 25 mm/s where demos cruise at 45–50 mm/s), the heads reach zero at z ≈ 190 mm
(demo close z ≈ 46–72 mm, `scratchpad/v4z.json`), the gripper command rises 0.31 → 0.52 (demo close aperture 0.45–0.5), then the chunks
turn positive (lift) while the tactile gate says `p_none 0.99`. Same shape for v4 and v5, for every valid episode, and one noise seed
descended at full speed to the right height.

1. **[certain] Loop timing alone does NOT produce this.** A demo-faithful, velocity-continuous imitation policy driven through the actual
   loop (L = 0.95 s, index-0 playback) descends at the demo speed and *overshoots* the stop by 55–65 mm; it never stalls high
   (`sim_velocity_loop.py`, §3.5). So the slow heads and the stop at 190 mm are the **content of the model's chunks at the states the
   loop produces**, i.e. a covariate-shift failure of the policy — the executor and the latency are exonerated as *causes* (they
   matter for the fix, §3.5).
2. **[certain] The marginal-mean chunk of this dataset is literally the failure.** `norm_stats.action.mean = [0, 0, 0, 0, 0, 0, 0.42]`:
   zero motion, gripper 0.42 (the waffles close aperture). A sampler that regresses toward the conditional mean under state uncertainty
   emits exactly "decelerate, close to a demo-like aperture". The 5-NFE shifted schedule evaluates the network at
   t = [1, .952, .882, .769, .556] and the **last Euler step (56 % of the path) is a single x0-prediction at SNR 0.64**; with
   `action_t_max_of_two` (v4/v5 default) only **0.7 %** of ACTION supervision lands at t < 0.556 (computed §3.1). Offline, at
   in-distribution demo states, the conditional is unimodal and the mean is right (17 mm). At the shifted states the rig produces
   (slow velocity, drifting aperture, slightly different image) the conditional broadens ("continue" vs "stop/close") and the mean of
   the two modes is "slow descent + partial close" — which the loop then feeds back until it is "stop + close". This is my top-ranked
   root cause (H1) and it unifies every observation including the seed multi-modality. Decisive test: NFE 5 vs 50 × 16 seeds on the
   recorded rig snapshots (§4, exp E2).
3. **[believe] The feedback that locks the collapse is proprio velocity + gripper aperture (H2, H4)**: 12 of the 26 `ur_state` dims are
   velocities (`windows.py:266-271`), demo action targets are Δ of the *measured* TCP so velocity continuity is baked into the
   targets (`session.py:739-750`), and the rig heads sustain the observed 25 mm/s instead of accelerating to cruise. Decisive test:
   swap `qd/tcp_speed` for demo-cruise values at the same z on the rig snapshots (E3).
4. **[believe] Height is mostly read from the image, and the image says "aligned" at 190 mm (H3)**: proprio z is only 1.4 σ
   (σ_z = 89 mm) from the demo close height at 190 mm inside a 26-dim MLP; the camera is near top-down. Decisive test: proprio ↔ image
   swap between a rig snapshot and a demo window at the same z (E4), plus a finite-difference sensitivity of `head_dz` to z (E5).
5. **[certain] Two "structural" beliefs in the codebase/parity lens are false and matter for the experiments and the paper**:
   (a) the structural mask only blocks CONTACT/ACTION → VIDEO_GEN (`sequence.py:190-202`); OBS_*/VIDEO_COND queries still read VIDEO_GEN
   keys, so VIDEO_GEN latents *and the intent chunk* (which only enters the VIDEO_GEN AdaLN, `phantom_dit.py:131-157`) reach the ACTION
   output in two hops — measured non-zero on the tiny model and exactly zero once all non-video queries are blocked (`probe_leak.py`,
   §3.6). "Droppable video" and "prev_chunk cannot reach the action head" are both untrue; `drop_video=True` also shifts every
   non-video frame's learnable absolute temporal pos-emb (`minimal_v4_dit.py:836`, `pos_emb_t[:T]`).
   (b) the ACC gate bias is *key-only on the haptic tokens and proportional to g* (`attention_bias.py:196-203`): when the model has
   closed on air and the gate correctly says "no contact" (g → 0), the haptic tokens are **de-emphasised** exactly when they carry the
   "nothing there" evidence. Nothing in the action objective ties "did the close succeed" to the lift — failure demos carry
   `action_weight = 0` (`windows.py:403`) and `gripper.obj` (the only proprio success bit) is ≈ absent from many demos
   (`lift_v4.jsonl obj2_frac_closed` = 0.0 in 2 of the first 3 waffles episodes). The "lifts anyway" part is therefore by construction.
6. **[certain] No offline number currently measures any of this.** `tools/terminal_eval.py` anchors only at `t_close − 1.6 s` of demos
   (`:123`), with demo velocity/history/aperture — the rig's states (z ≈ 190, v ≈ −25 mm/s, aperture drifting) are never evaluated.
   All eight experiments in §4 run on the recorded rig episodes + val windows on compute3 in ≤ 1 GPU-hour total and decide between
   H1–H4 before the next rig session; the plan is a concrete script outline (§5).

---

## 1. What the loop actually feeds the model (code facts)

| Quantity | Training (`WindowSampler.sample`) | Deploy (`SnapshotBuilder.build` → `PhantomPolicy._batch_from_obs`) |
|---|---|---|
| `ur_state` (26) | `[q(6), qd(6), tcp_pose(6), tcp_speed(6), gripper(pos, obj)]` nearest `t0` (`windows.py:266-271`), normalised with `norm_stats.ur_state` | latest arm ring row + gripper ring row (`planner.py:151-153`), same normalisation (`policy.py:93-95`) |
| action targets | Δ of consecutive **measured** TCP poses on the 10 Hz record tick + last **sent** gripper command (`session.py:739-750`, `derived.pose_delta`) | denormalised chunk, cumsum from `_last_cmd` (`executor.py:96-104, 143-154`) |
| chunk anchor | `(t0, t0+1.6]` from the state at `t0` (`windows.py:276`) | played from index 0 starting at `obs.t + latency` (`policy.py:167-172`, `executor.py:96-98`) |
| `prev_chunk` | executed Δ rows `(t0−1.6, t0]` (`windows.py:277`) | previous **proposal** (`policy.py:101-102`; DEFERRED comment `planner.py:284-291`) |
| `prev_cpk` | own package for the current window (`two_pass`, `rf.py:163-168`) | previous plan's package, step 0 (`rf.py:398-400`) |
| noise | fresh per sample | fixed per episode (`rf.py:405-411`, `--persistent-noise`) |
| window anchors | uniform in `[start+1.6, end−3.1]` (`windows.py:159-177, 194`); v5 adds 30 % in `[t_close−1.5, t_close−0.2]` (`common.py:444-456`, `launch_v5.sh`) | — |

Normalisation numbers that matter (`norm_stats.json`, identical in the v4/v5 checkpoints [parity-lens] §2.4):

```
action   mean [0.0000 0.0001 -0.0001 -0.0006 0.0001 -0.0002 | 0.4219]   std [2.9 7.2 6.7 mm | 17.5 16.3 15.9 mrad | 0.28]
ur_state tcp_pose  mean z = 0.187 m   std z = 0.089 m
         tcp_speed std  (x,y,z) = (0.029, 0.070, 0.066) m/s        qd std 0.07-0.22 rad/s
         gripper   pos mean 0.354 std 0.191;  obj mean 2.40 std 1.03
start_poses waffles: tcp z 0.335 ± 0.027, gripper 0.232 ± 0.152, tcp_z_min 0.052
```

Three consequences: (i) the **normalised zero chunk is "stop + gripper 0.42"**; (ii) a commit at z = 190 mm vs 65 mm is a 1.4 σ change of
one of 26 inputs; the demo cruise 47 mm/s vs the rig 25 mm/s is a 0.33 σ change of `tcp_speed_z`; (iii) `_null_obs_batch` nulls `ur_state`
to normalised zeros = the *mean state* (`rf.py:230-233`), whose z is 187 mm — the height at which the waffles rollouts committed
(182–194 mm [parity-lens]). Probably coincidence, but it is one of the cheapest things to test (E7).

## 2. Sampler facts (computed)

```
rf_shift = 5 (backbone.py:70); NFE-5 schedule t = [1.000 0.952 0.882 0.769 0.556 0.000], step sizes [.048 .070 .113 .214 .556]
training t ~ shift(sigmoid(N(0,1))):   median 0.833,  P(t<0.556) = 8.3 %,  P(t<0.769) = 34 %
ACTION frames with action_t_max_of_two (v4/v5 default, train_teacher.py:161, config/model.py:67):
                                        median 0.896,  P(t<0.556) = 0.7 %,  P(t<0.769) = 11.7 %
last Euler step: x_0.556 = 0.444 x0 + 0.556 eps -> SNR 0.64; the step to t=0 IS the x0-prediction at that point (rf.py:456)
```

So the deployed action chunk is `x0_hat(x_{0.556})` where `x_{0.556}` came from four Euler steps at t ≥ 0.77; the network is asked for
that x0-prediction at a noise level that received 0.7 % of the ACTION supervision. Velocity targets are t-independent, so this is not a
bug, but it means the *mode-resolving* part of the sampler runs on the least-trained part of the field, and the first four steps —
evaluated at t ≥ 0.77 where E[x0 | x_t, obs] ≈ E[x0 | obs] — drag the sample toward the conditional mean. In-distribution this is
invisible (the conditional is sharp); it is exactly where a broadened conditional becomes a mean. `evaluate_sampled`'s `mag_ratio`
guard (`common.py:614-677`) is computed on val windows, so it cannot see a state-dependent collapse.

## 3. Ranked hypotheses

### H1 [top, believe] Covariate shift → broadened conditional → few-NFE mean collapse to the marginal chunk ("stop + 0.42"), locked in by the loop
**Mechanism.** At a rig state the model has not seen (slow descent, drifting aperture, slightly different lighting/placement), the
conditional over chunks is bimodal — "continue at cruise" vs "decelerate/close" — and the 5-NFE sample lands near the mean:
slower descent + partial close. The executor executes it, so the next observation is even slower and more closed; the mean moves
toward the "stop/close" mode; after 4–6 replans it *is* the stop mode. A different noise draw (persistent per episode) can start the
episode inside the "continue" mode — the "one seed descended at full speed" observation.
**For.** Marginal-mean chunk = the failure (§1); sampler schedule/training-t mismatch (§2); the head sequence −26 → −3 with the gripper
rising *while still descending* (a mixture, not a decision); offline `commit_ratio` 1.43–1.72 > 1 at demo states
(`docs/rig_session_v5.md:88`) — the model over-commits when the state is in-distribution, so the under-commit is state-specific;
seed multi-modality; identical behaviour for v4 and v5 (the fine-tune re-weighted terminal *demo* windows, which does not touch
off-distribution states).
**Against.** None in hand — nobody has sampled the rig snapshots at high NFE or across seeds.
**Decisive experiment (E2, §4).** Replay each rig replan snapshot offline; sample NFE ∈ {5, 10, 50} × 16 seeds. Signature of H1: at
NFE 50 the per-seed `head_dz` is **bimodal** (a cluster near −45 mm and one near 0) and the NFE-5 samples sit **between** the clusters
with the gripper channel between 0.25 and 0.5; the fraction in the "continue" mode falls replan by replan. If instead all NFE/seeds
agree on the stop chunk, H1 is out and the model's conditional itself is wrong (→ H3/H2).
**Fix (if confirmed).** Cheapest: mode-seeking at deploy — sample K seeds (batch K, one denoise, no extra latency beyond K× FLOPs on a
5090 that already fits batch 4), reject samples whose head descent is < 50 % of the K-max while `p_contact` is low, and pick the
mode nearest the previous plan (keeps persistent-noise coherence). Training-side: `--no-action-t-max-of-two` and/or a lower
`rf_shift` for the ACTION group in the ~3k-step fine-tune so the t < 0.77 field is supervised; NFE 8–10 with `--compile`.

### H2 [high, believe] Velocity self-consistency: proprio `qd`/`tcp_speed` anchor the chunk to the observed (slow) velocity; the index-0 loop seeds the slow state
**Mechanism.** Twelve of 26 proprio dims are velocities; the targets are velocity-continuous (Δ of measured TCP), so the model learns
`a_0 ≈ v_obs·0.1`. At the rig the observed velocity at replan n is chunk n−2's step-9 velocity (back-to-back replans, index-0 playback;
[parity-lens] §2.1), and the first two replans observe v = 0 (the arm has not moved yet), so the executed velocity restarts from a stale
value every 0.96 s. A demo-faithful prior would still re-accelerate to cruise (my simulation does); the rig heads instead **sustain**
25 mm/s at z = 280–205 mm, where every demo cruises at 45–50 — i.e. the model is anchored to v_obs but has lost the "accelerate to
cruise" prior at those states. This is H1's feedback channel, but it can also stand alone if the model simply extrapolates v_obs.
**For.** `windows.py:266-271` (velocities in the state); `session.py:739-750` (targets = measured Δ); the rig descends ≈ 30 % slower from the
first replans (CONTEXT); the stall watchdog and governor are inert ([parity-lens] §2.2). The sim (§3.5) shows the loop alone gives
45 mm/s, so sustained 25 mm/s needs the model.
**Against.** Velocity is a 0.33 σ signal; a 26→256 MLP may barely use it. Unknown until E3/E5.
**Decisive experiment (E3).** On the rig snapshots replace `qd` and `tcp_speed` with the demo-typical cruise values at the same z (take
them from the nearest val demo state by z, or scale the observed velocity ×1.8) and re-sample with the same noise. Signature: `head_dz`
goes from ≈ −25 mm to ≈ −45 mm (gain ≈ 1 → H2 confirmed); unchanged → the model ignores velocity and H2 is out. Complement with the
free trace statistic **anchoring gain** = slope of `actions[0,2]/0.1` on the measured `tcp_speed_z` at snapshot time across all rig
replans (demos give 1 by construction).
**Fix (if confirmed).** Deploy: phase-correct playback + committed-prefix inpainting ([parity-lens] §3.1) so the observed velocity is
the *intended* one; do not let the first two replans observe a resting arm twice (start the second replan only after chunk 0 has
played ≥ 0.3 s, or feed `prev_chunk` = chunk 0's head). Training: velocity-dropout on `qd/tcp_speed` (zero them in 30 % of windows) so
the prior, not the observed velocity, sets the cruise speed.

### H3 [high, believe] Height is read from the near-top-down image; proprio z is under-used; "x-y aligned" looks like "at the object"
**Mechanism.** The scene camera is near top-down: lateral alignment is a strong pixel cue, height is a weak one (apparent gripper size,
perspective offset). If the action head weights the image over the 1.4 σ proprio-z signal, a rig episode whose gripper sits over the
object at 190 mm can look like the demo pre-close state. The commit heights are consistent across v4/v5 and across episodes
(182–194 mm; retries at 170 mm [parity-lens]), which smells like a *perceptual* threshold rather than noise.
**For.** Camera dominance in the conditioning ablation (`common.py:398-404` comment); consistent commit height; the executor tracks within
mm so the height is the model's decision.
**Against.** Offline the same model commits at demo states with 17 mm error — but those windows have the demo image *and* demo proprio,
so they cannot separate the two. The A/B grid tape (`docs/rig_session_v5.md:14`) is a new visual element under the object in every
08-28 episode; 08-20 (no tape?) failed the same way, so the tape is not the cause, but it is an uncontrolled visual difference.
**Decisive experiment (E4 + E5).** E4: for a rig replan at z ≈ 190, build two hybrid batches: (rig image + demo proprio at z = 190 from a
val demo that is cruising there) and (demo image at z = 190 + rig proprio). Signature: if the chunk follows the image
(stop with the rig image regardless of proprio, cruise with the demo image), H3 is confirmed. E5: finite-difference sensitivity of
`head_dz` to `ur_state` tcp-z (±50/±100 mm, q left as is and also with q adjusted by IK if available) on 40 val terminal windows and
on the rig snapshots; a near-zero ∂head_dz/∂z means the policy is vision-only for height. Extrinsics check on the rig (1 min): gripper
bbox in the first deploy frame vs demos at the gated start pose.
**Fix (if confirmed).** Training-side: image dropout (already exists as joint cond-dropout; make it *per-group*: drop the image alone in
20 % of windows so proprio must carry height), or a proprio-z auxiliary target on the OBS_PROPRIO tokens; deploy-side: a z-conditioned
commit guard (do not accept a close below `tcp_z_min + 30 mm`... i.e. refuse a gripper close while z > task `tcp_z_min` + 40 mm and
`p_contact` is low — re-plan instead).

### H4 [high for the "lifts anyway" half, certain in code] Gripper-state coupling with no success signal: close → lift is unconditional
**Mechanism.** After a close, `ur_state.gripper.pos` ≈ 0.5 and the image shows a closed gripper; in every training window that state is
followed by a lift. The only proprio "did I grasp" bit is `gripper.obj` (Robotiq gOBJ), but demos closed with the 30 N force clamp on soft
pads and often report no object (`obj2_frac_closed` 0.0 / 0.869 / 0.0 in the first three `lift_v4.jsonl` rows), so obj is not a learnable
success signal. Tactile evidence exists (p_none 0.99) but reaches the action head only through attention to haptic tokens, whose key bias
is `λ_block · β(p_evt) · g` (`attention_bias.py:196-203`, `phantom_dit.py:249-252`) — **∝ g, so it vanishes exactly when the gate says
"nothing there"**. Failure demos (45 `<task>_fail`) supervise contact heads only (`windows.py:403`, `rf.py:310-318`), so no gradient ever
taught "closed on air ⇒ re-open and descend".
**For.** All of the above is read in code. The rig sequence (close → lift → transport → retry) is exactly the demo continuation.
**Decisive experiment (E6).** On the post-close rig snapshots swap `gripper.obj` (3 → 2) and `pos` (0.5 → demo close value) and
re-sample; then swap the tactile `fields/gel/contact_state` for a demo in-contact snapshot. Signature: if only the tactile swap flips the
chunk from lift to re-open/descend, the action head *can* read tactile — then a deploy-time gate that raises the ACC bias (or simply
vetoes lift while p_none > 0.9) is enough; if nothing flips it, the lift is image/aperture-driven and needs training data (DAgger-lite
relabels of the 26 rig failures) or a hand-coded terminal guard.
**Fix.** Immediate (no training): terminal guard in `PlannerLoop` — after a commanded close, if `plan.p_evt[none] > 0.9` for one replan,
open the gripper to the task aperture and clamp the next chunk's z ≥ current z (no lift) → the policy sees "open, at hover" and
re-descends; log it as `terminal_veto`. Paper-wise this *is* the tactile-grounded terminal control the next-phase plan wants.

### H5 [medium as a cause; high as a fix constraint, certain] Observation→execution latency (index-0 playback of a chunk anchored 1–2 L in the past)
**Facts.** L ≈ 0.96 s = replan interval; `action_times[0] = obs.t + latency` (`policy.py:167-172`); `submit` rebases so `pose_at(0) ==
_last_cmd` (`executor.py:96-104`); steps 0–9 execute, 10–15 never do ([parity-lens] §2.1).
**Simulation (`sim_velocity_loop.py`)**, demo-faithful 1-NN policy over (z, v, g) with a 47 mm/s cruise and a 65 mm stop:

```
                              descent(250..120)   z_min   close z
NN(z,v,g) L=0.00              52.0 mm/s           70.6    70.6      (reference)
NN(z,v,g) L=0.95 index0       45.0 mm/s            1.5     1.5      <- current executor: table strike, NO stall
NN(z,v,g) L=0.95 skip-head    28.2 mm/s           82.4    82.4      <- phase-correct playback alone: right height, 40 % slow
NN(z,v,g) L=1.30 index0       41.8 mm/s           -3.4    -0.7
NN(z,v,g) L=1.30 skip-head    10.0 mm/s          175.7    never     <- chunk exhausted
```
**Verdict.** The loop cannot make a demo-faithful policy stop high; it makes it overshoot by ~60 mm (the 08-28 whiteboard "40 mm below
any demo" slam is this). But it constrains the fix: (a) any repair of H1–H3 without phase correction trades under-commit for table
strikes — keep the z-floor; (b) phase-correct playback *alone* halves the descent speed (only the chunk tail is played) and at L = 1.3 s
exhausts the chunk — it needs prefix inpainting and L ≤ 0.6 s (NFE 3–4 with `--compile`) as the parity lens says. Both are testable
offline with the same harness (E8).

### H6 [medium, certain-in-code] `prev_chunk` is the previous *proposal* and it does reach the action head (2-hop), contrary to the parity lens
`policy.py:101-102` feeds the last accepted plan's 16 proposals (covering `(now−0.96, now+0.64]`) where training used the executed past
(`windows.py:277`). The parity lens called this "structurally inert for actions". It is not: `structural_attn_bias` blocks only
CONTACT/ACTION queries from VIDEO_GEN keys (`sequence.py:195-202`); VIDEO_COND and OBS_* queries read VIDEO_GEN, and ACTION reads them.
`probe_leak.py` on the tiny random-init teacher (ACC λ = 0, so the ACC path is off): randomising the intent chunk changes the ACTION
velocity by 1e-3 relative, randomising the VIDEO_GEN latents likewise; with all non-video queries blocked both become exactly 0.000000.
Two blocks at init give a small number; 28 trained blocks can give anything. It also enters ACC's `intent_mlp` (`acc.py:62, 84`).
**Effect on the loop.** The "previous proposal" contains the un-executed tail — in whiteboard 002 the tail carried 2–4× the head's descent
at every replan [parity-lens] — so the model is told "you already planned to descend a lot" while the arm did not; and after a stop, the
previous proposal is the stop, reinforcing it. Sign and size unknown → E1 swap.
**Experiment (E1).** Same snapshot, `prev_chunk` ∈ {previous proposal (as deployed), executed Δ-TCP over the last 1.6 s from the arm
stream (as trained), zeros}. Signature: any |Δhead_dz| > 5 mm means the intent path is live and must be made train-faithful
([parity-lens] §3.3 fix) before any other conclusion is drawn from replays.

### H7 [low] `prev_cpk` feedback
Step-0 stale ([parity-lens] §3.4) and multiplied by g ≈ 0 during the descent (gate low far from contact) → cannot drive the descent.
Post-close it says "no contact" and de-emphasises haptic keys (H4). E1 includes a `prev_cpk=None` swap; expect < 2 mm change during descent.

### H8 [low-medium, guess] Joint cond-dropout makes the "mean state" a half-unconditional token
`cond_dropout_p = 0.1` nulls image (black latent), wrist, tactile and `ur_state` **jointly** to normalised zeros (`rf.py:213-234, 257-261`).
Zeros in normalised space are the *mean state* (z = 187 mm, gripper 0.35), not an out-of-manifold null. 10 % of windows therefore train
"mean state + black image → marginal chunk (stop + 0.42)". With the image present the model can tell the difference; if H3 holds (image
weakly used for height), states near the proprio mean inherit some of the marginal. Cheap test E7: `guidance ∈ {1, 1.5, 2, 3}` on the
rig snapshots (the null branch exists, `rf.py:394-397`), and the scalar `‖v_cond − v_null‖/‖v_cond‖` at the last Euler step per replan
("obs reliance"). Signature: commit height drops with guidance / obs-reliance is low at the stalled replans → the model is partly
unconditional there; fix = per-group dropout with a learned null token (or −4 σ constants) instead of zeros.

### H9 [low] Terminal windows under-represented
v4: uniform anchors (~8 % of windows in the last 1.5 s before close); v5: `--grasp-frac 0.3` (`launch_v5.sh`) — and the rig outcome did not
change. What *is* absent from every demo is the rig's actual state family (hover at 150–250 mm with |v| < 30 mm/s and aperture 0.3–0.5);
the recovery demos start 3–8 cm off (`docs/recovery_demos_protocol.md`), not 12 cm, and start from rest with a settled aperture. So the
data gap is "slow/hesitant approach", not "terminal". Fix = DAgger-lite on the 26 rig failures + synthetic slow-approach windows
(time-stretch demo descents ×1.5 in the sampler — actions are Δ so stretching is a resample).

### H10 [low] Video-generation prior dominating
Only via the 2-hop leak (H6). Test E1 variant: `drop_video=True` vs False on the same snapshot (mind the pos-emb shift, §3.6). Note for the
paper: "video is droppable at inference" is not exact for this layout.

### H11 Time-anchored behaviour via visual cues
The model has no clock input; any time anchoring is an image cue (gripper apparent size/position, shadow) → folded into H3/E4.

### 3.5 Note on the simulation
`sim_velocity_loop.py` builds 27 synthetic demos (start 280–320 mm, cruise 42–52 mm/s, stop 55–72 mm, 1 s accel/decel, 0.5 s close,
lift), a 1-NN policy in normalised (z, v_z, g) using the real σ's, and the executor's index-0 rebase exactly as `executor.py:96-111`.
It is deliberately the *most favourable* policy (perfect imitation with velocity continuity); its failure mode under the loop is
overshoot, never a stall, which is what rules the loop out as the cause of the stall.

### 3.6 Structural facts established by `probe_leak.py`
```
intent chunk randomised          -> ACTION velocity rel change 0.001   (mask on, tiny 2-block net, ACC off)
VIDEO_GEN latents randomised     -> ACTION velocity rel change 0.001
OBS_PROPRIO frame randomised     -> ACTION velocity rel change 0.023
drop_video=True at inference     -> ACTION velocity rel change 0.015
CONTROL (all non-video queries blocked): intent -> 0.000000 ; VIDEO_GEN -> 0.000000 ; drop_video -> 0.015 (learnable abs. pos-emb shift)
```
`LearnablePosEmbAxis.generate_embeddings` uses `pos_emb_t[:T]` (`minimal_v4_dit.py:836`): the 14-frame extended sequence occupies
absolute temporal slots 0–13 of a table the base model trained for video frames 0–3; slots 4–13 are frozen pretrained "future video"
embeddings re-used for OBS/CONTACT/ACTION frames, and dropping VIDEO_GEN shifts every later frame by 3 slots. Hygiene for the rig
(deploy uses `drop_video=False`), but it kills the "droppable" claim.

---

## 4. The experiments (all offline, compute3, recorded rig episodes + val windows; ≤ 1 GPU-hour)

Data: `~/phantom-icra-2027/data/episodes/deploy/20260828/ep_teacher_{waffles_1787923675_000, waffles_1787922904_000,
whiteboard_1787941222_002, Carton_1787937561_000}` (+ the other valid 08-28 episodes; skip those from 17:19 on = wrapped wrist), the
08-20 waffles episodes, `runs/teacher_v5_batch0822/v5_6.pt` and `runs/teacher_v4_790eps/teacher_020000.pt`, val demos from
`data/phantom-episodes/manifests/all.jsonl` split `val`.

| id | question | inputs swapped | data | metric / expected signature |
|---|---|---|---|---|
| **E0** | does the offline replay reproduce the rig? | none (as-deployed batch, chained `prev_cpk`, persistent noise, NFE 5) | every replan of 6 rig episodes | per-replan `head_dz` vs trace within ±8 mm and the same commit replan (seed differs → compare across 8 seeds; the trace must be inside the seed spread) |
| **E1** | is the intent/cpk path live? | `prev_chunk` ∈ {proposal, executed Δ-TCP, zeros}; `prev_cpk` ∈ {chained, None}; `drop_video` ∈ {F, T} | same | |Δhead_dz| per swap; > 5 mm = live path, fix parity first |
| **E2** | H1 mean collapse? | NFE ∈ {5, 10, 50}, 16 seeds, same everything | the 6 replans around the stall per episode | histogram of `head_dz` per NFE: bimodal at NFE 50 with NFE 5 between the modes = H1; `grip_max` bimodal likewise |
| **E3** | H2 velocity anchoring? | `qd, tcp_speed` ← demo cruise at the same z (nearest val state by z) / ×1.8 / zeros | descent replans (z 300→190) | `head_dz` → ≈ −45 mm = H2; free check: anchoring gain from the traces (`actions[0,2]/0.1` vs measured `tcp_speed_z`) |
| **E4** | H3 image vs proprio for height? | (rig image + demo proprio@z) and (demo image@z + rig proprio) | stall replans at z≈190 + val demo windows at z≈190 | which half decides `head_dz`/`close_step` |
| **E5** | ∂head_dz/∂z, ∂/∂v, ∂/∂aperture | finite differences ±1σ on tcp-z (±89 mm), tcp_speed_z, gripper pos, obj; fixed noise | 40 val terminal windows + rig snapshots | sensitivity table; ≈0 for z = vision-only height |
| **E6** | H4 what could veto the lift? | post-close snapshot: obj 3→2; pos→demo close; tactile ← demo in-contact snapshot; ACC g forced to 1 | post-close replans | does the chunk flip from lift to open/descend? |
| **E7** | H8 obs reliance | guidance ∈ {1, 1.5, 2, 3}; `‖v_cond − v_null‖/‖v_cond‖` at t = 0.556 | all replans | commit height vs guidance; reliance dips at stalled replans |
| **E8** | fix pre-validation | executor rule ∈ {index0, skip-head, skip+prefix-inpaint}, L ∈ {0.5, 0.95} | closed-loop *surrogate*: the same replay but the next snapshot's proprio is rolled forward with the executed head (image frozen) | z_min and descent speed per rule; choose before rig time |

Reading the table: E0 must pass first (else nothing else is interpretable); E1 decides whether the replays need the parity fixes before
H1–H4 are judged; E2–E5 are the discriminating set; E6/E7 are cheap add-ons; E8 is the go/no-go for the executor change.

## 5. Script outline — `tools/replay_rig.py` (new tool; ~250 lines; reuses existing classes)

```python
"""Offline replay of recorded rig episodes through the deployed policy, with input swaps.
python tools/replay_rig.py --ckpt runs/teacher_v5_batch0822/v5_6.pt --hardware configs/hardware.nuc.yaml \
    --episodes data/episodes/deploy/20260828/ep_teacher_waffles_1787923675_000 ... \
    --val-root data/phantom-episodes/tasks --exp E0 E1 E2 E3 E4 E5 E6 E7 --seeds 8 --out replay.json"""

# ---- loading (copy of terminal_eval.py:71-85) ----------------------------------------------------
def load_policy(ckpt, hw_path, device="cuda"):            # -> pm, norm, mc (EMA weights, mc from checkpoint)
    ...  # build_model(..., mc=PhantomModelConfig.from_dict(payload["configs"]["model"]), inference=True); load_phantom_checkpoint(load_ema=True)

# ---- rig episode access (EpisodeReader; clock offset from meta.clock_calibration.offset, rig_trace_decompose.py:42,60) -------------
class RigEpisode:
    def __init__(self, path): reader, meta, trace = EpisodeReader(path), EpisodeMeta.load(path/"meta.json"), json.load(planner_trace.json)
    def t_master(self, i): return trace[i]["t"] + meta.clock_calibration["offset"]              # snapshot time of replan i
    def last_leq(self, stream, t): ...                                                          # index of the last sample with ts <= t (deploy took the LATEST frame, not nearest)
    def snapshot(self, i) -> ObsSnapshot:                                                       # rebuild exactly what SnapshotBuilder.build() saw (planner.py:77-185)
        t = self.t_master(i)
        rgb = camera_scene_color at last_leq(t)                                                 # JPEG-decoded uint8 HxWx3
        arm rows in (t - 2*window_s, t] -> wrist_window = np.interp on linspace(t_ft - 0.25, t_ft, 31)  (planner.py:129-138)
        ur_state = [q, qd, tcp_pose, tcp_speed] at last_leq(t) + gripper [pos, obj] at last_leq(t)
        fields = tactile_*_keyframes at last_leq(t); gel = tactile_*_infer_img; contact_state from fields_ds (last two) via dv.derive_timestep
        reactive = dv.reactive_score(fields_ds now, fields_ds at previous replan)                # deploy semantics (planner.py:179-184)
        return ObsSnapshot(t=t, rgb=rgb, wrist_window=..., ur_state=..., gel=..., fields=..., contact_state=..., reactive=...)
    def executed_prev_chunk(self, i):                                                           # training semantics (windows.py:277): Δ of measured TCP on the 10 Hz grid ending at t
        poses = tcp_pose resampled at t - (16-k)/10 ... ; return np.stack([pose_delta(p[k-1], p[k]) + [gripper pos at that time]])
    def trace_actions(self, i): return np.array(trace[i]["actions"])                            # (16, 7) denormalised, what the executor played (head = first 9-10 rows)

# ---- batch construction: PhantomPolicy._batch_from_obs is the deploy path; swaps are applied to the ObsSnapshot / batch ------------
def as_deployed_batch(policy, snap, prev_plan): return policy._batch_from_obs(snap, prev_plan)  # policy.py:80-146 (prev_chunk = previous proposal)
def apply_swaps(batch, snap, swaps: dict, norm, val_bank):
    # swaps: {"prev_chunk": "executed"|"zeros"|"proposal", "prev_cpk": "chained"|"none", "vel": "demo"|"x1.8"|"zero",
    #         "z": +/-m, "grip_pos": float, "obj": int, "image": "demo@z", "proprio": "demo@z", "tactile": "demo_contact"}
    # velocity dims of ur_state: qd = [6:12], tcp_speed = [18:24]; tcp z = [14]; gripper pos/obj = [24], [25]  (windows.py:266-271)
    # denormalise -> edit -> renormalise with norm.ur_state; image swap replaces batch["video"] (tiled, policy.py:85-89)

# ---- the val bank for "demo@z": val windows whose tcp z is within 15 mm of the target and |tcp_speed_z| > 30 mm/s (cruising) ------
class ValBank:  # WindowSampler(hw, bb, norm).sample(ep, t0) over manifest_split(val), indexed by task and z bin; also "in-contact" windows (events != none)

# ---- sampling (rf.sample, rf.py:355-469) ------------------------------------------------------------------------------------------
def sample_chunk(rf, batch, *, nfe, seed, prev_cpk, guidance=1.0, drop_video=False, noise=None):
    rf._gen = torch.Generator().manual_seed(seed); rf._episode_noise = noise                     # persistent-noise emulation across replans of one chain
    pred = rf.sample(batch, nfe=nfe, prev_cpk=prev_cpk, guidance_scale=guidance, drop_video=drop_video, reuse_noise=noise is not None)
    return denorm(pred.actions_B_H_A[0]), pred.cpk.detach(), pred.acc, pred.x_final_B_C_T_H_W
def obs_reliance(rf, batch, x_at_0556, ...):  # one extra net call with the null batch (rf.py:393-397, 445-455) at t=0.556 -> ||v_cond - v_null|| / ||v_cond||

# ---- metrics per sampled chunk -------------------------------------------------------------------------------------------------------
def metrics(a):  # a (16,7) denormalised
    cum = np.cumsum(a[:, :3], 0) * 1000
    return dict(head_dz=cum[8, 2], tail_dz=cum[15, 2] - cum[8, 2], head_dxy=np.linalg.norm(cum[8, :2]),
                close_step=first k with a[k, 6] > 0.45 else 16, grip_max=a[:, 6].max(), v0=a[0, 2] / 0.1 * 1000)

# ---- experiments --------------------------------------------------------------------------------------------------------------------
def run_chain(ep, swaps, nfe, seed):        # E0/E1/E3/E4/E6/E7: replay all replans sequentially, prev_cpk chained, persistent noise per chain
def run_modes(ep, replans, nfes, seeds):    # E2: per replan, K seeds x NFE -> head_dz / grip_max arrays; report bimodality (dip test or 2-means gap > 20 mm)
def run_sensitivity(bank, rig_snaps):       # E5: finite differences (+/-1 sigma) on z, tcp_speed_z, grip pos, obj, fixed noise
def run_executor_surrogate(ep, rule, L):    # E8: roll proprio forward with the executed head under rule in {index0, skip, skip+inpaint}; image frozen
# report: json {episode, replan, swaps, nfe, seed, metrics, obs_reliance, trace_head_dz}; a 10-line summary table per experiment.
```

Runtime: one 5-NFE sample ≈ 1 s on the 5090 → E0 (6 eps × 20 replans × 8 seeds) ≈ 16 min; E2 (6 eps × 6 replans × 16 seeds × (5+10+50 NFE)/5)
≈ 1.1 h if run naively — batch the 16 seeds (B = 16 fits on 32 GB at inference) → ≈ 5 min; E3–E7 minutes each.

## 6. What to change first (ordered by information per rig-hour)

1. E0–E2 (half a day). If H1 confirmed: deploy-time mode-seeking (K-seed batch + descent-consistent selection) and NFE 8 with `--compile`;
   fine-tune 1–3k steps with `--no-action-t-max-of-two`. If not: E3/E4 decide between velocity-dropout and image-dropout fine-tunes.
2. Regardless: the terminal guard (H4) — veto lift while `p_none > 0.9` after a close; re-open and re-descend. It converts the failure into a
   retry loop that the tactile teacher can *use*, and it is the paper's story.
3. Executor: phase-correct playback + prefix inpainting + L ≤ 0.6 s (H5) — validated on E8 before rig time; keep the z-floor.
4. Parity one-liners (prev_chunk from the arm ring, prev_cpk step index, reactive dt) so replays and the rig agree (H6/H7).
5. `terminal_eval --rig-states`: add the replay metrics (E0's `head_dz` at the rig's own states, seed spread, obs reliance) as the standing
   offline number; the current one cannot see the failure.
6. Paper hygiene: drop "droppable video"/"intent conditions the chunk" claims or fix the mask (block all non-VIDEO_GEN queries from
   VIDEO_GEN keys — a one-line change in `structural_attn_bias`, but it changes what OBS tokens see and needs a fine-tune).
