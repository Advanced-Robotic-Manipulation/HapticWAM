# PHANTOM review — lens: TRAIN/DEPLOY PARITY

Repo `~/GitHub/phantom` @ `3539988`. Read: `phantom/inference/policy.py`, `phantom/deploy/{planner,executor,governor,safety,runtime}.py`,
`phantom/scripts/run_deploy.py`, `phantom/data/windows.py`, `phantom/train/common.py`, plus the pieces they depend on
(`model/rf.py`, `model/acc.py`, `model/sequence.py`, `model/phantom_dit.py`, `model/ace/{packing,heads}.py`,
`data_collect/session.py`, `recording/{recorder,workers}.py`, `backbone/text_embedding.py`, `tools/terminal_eval.py`,
`tools/rig_trace_decompose.py`, `configs/hardware.nuc.yaml`, tests). Nothing under the repo was modified.

Data actually looked at (read-only, over ssh to compute3): all 37 rig `planner_trace.json` from
`data/episodes/deploy/{20260820,20260828}` (+ zarr arm/gripper streams for four 08-28 episodes), the v4/v5
checkpoint config/provenance/norm-stats blobs, the text-embedding cache, `paths.local.yaml`, `GO_ANY.sh`.
Scripts used: `scratchpad/review/{sim_executor_lag,trace_stats,trace_detail}.py`.

Certainty labels: **[certain]** = read in code / measured; **[believe]** = strong inference; **[guess]** = hypothesis with a proposed test.

---

## 0. Executive summary

1. **[certain] The closed loop is outside the regime training models.** Rig replan latency L ≈ 0.95–1.0 s (08-28; 1.33 s and 2.6 s
   on 08-20) and the planner replans back-to-back, so each 16-step chunk is played from index 0 for ~0.96 s (steps 0–9, 60 %)
   and then replaced. Those steps describe the motion for `(t_obs, t_obs+0.96]` but are executed over `[t_obs+0.96, t_obs+1.92]`
   from the pose the arm reached *after* the observation (`executor.submit` rebases every plan onto `_last_cmd`). Training
   windows (`windows.py:276`) anchor the chunk at the observation time with zero latency and supervise all 16 steps uniformly.
   A simulation with a *perfect* state-feedback policy driven through exactly this loop overshoots the demo stop height by
   38 mm at L = 0.9 s and 59 mm at L = 1.3 s — the same magnitude as the 08-28 "40 mm below any demo" slam that motivated the
   z-floor. The tails that are never executed are not innocuous: in whiteboard 002 the tail (steps 9–15) commands 2–4× the descent
   of the head at every one of 17 replans. Fix = phase-correct playback + RTC-style prefix inpainting + lower latency (details §3.1).
2. **[certain] The offline terminal metric (`tools/terminal_eval.py`) is not deploy-faithful** — GT executed history as
   `prev_chunk`, chunk anchored at the observation (no latency), the full 16 steps scored although only ~10 ever execute,
   `prev_cpk` from 1.6 s earlier instead of one replan period, fresh noise instead of persistent. It cannot see the failure the
   rig sees, which is why 17.4 mm offline coexists with 65–120 mm on the rig. A faithful variant is a ~40-line change and would
   let checkpoint/executor choices be made offline instead of spending rig hours.
3. **[certain] Speed governor is exonerated**: over all 37 rig episodes `sigma_max` is 0.07–0.15, giving playback scale ≥ 0.95
   in 100 % of replans. The 30 % slower descent is in the model's chunk heads (20–36 mm per 0.96 s), not in the executor.
4. Several smaller, exact parity defects that each cost 1–15 lines: `prev_chunk` is the previous *proposal* (covering
   −0.96…+0.64 s) rather than the executed past; `prev_cpk` is fed one latent step stale; the CASA reactive score compares
   tactile frames 0.96 s apart instead of 0.125 s; DAgger rollouts record proposals as `STREAM_ACTIONS`; the deploy start state
   (at rest, `prev_chunk = 0`) is excluded from training by construction.
5. Verified OK (no action): image path (RGB, resize, normalisation, cond-latent-only VAE encode), gel/fields/contact_state,
   `ur_state` composition + checkpoint norm stats, wrist F/T grid (tested), text conditioning (cache keys match the 8 texts in
   both checkpoints' provenance), EMA-by-default + LoRA merge, NFE/guidance/rope from the checkpoint, bf16 both sides.

---

## 1. Feature-by-feature parity table

Legend: T = training (`WindowSampler.sample`, `windows.py`), D = deploy (`SnapshotBuilder.build` → `PhantomPolicy._batch_from_obs`).

| Input | Training | Deploy | Verdict |
|---|---|---|---|
| **Video cond frame** | camera frame *nearest* `t0` (±33 ms @ 15 fps; can be after `t0`), JPEG round-trip RGB (`jpeg_codec.py:45-83`, RealSense `rgb8`), `bilinear_resize` → 256×320, `/127.5−1` (`windows.py:221-226, 253-255`); 13 frames encoded by the (causal) VAE, only latent 0 is conditioning | latest ring frame (age ≤ 67 ms + driver), raw RGB, same resize fn, same normalisation; tiled ×13, `encode_gen=False` encodes frame 0 only (`policy.py:85-89`, `rf.py:99-109`) | ✓ (photo-aug is train-only; frame ~50–100 ms older than proprio at deploy vs aligned within ±33 ms in T — minor) |
| **obs_gel** | `infer_img` nearest `t0`, gray→3ch, resize (`windows.py:228-236`) | latest `tactile_*_img` ring (`planner.py:164-167`), same conversion (`policy.py:116-121`) | ✓ |
| **obs_mech fields** | `keyframes` stream at `t0` (144×192×8), `norm("fields")` (`windows.py:363-365`) | `tactile_*_kf` ring latest (`planner.py:162-163`) | ✓ |
| **contact_state** | `[wrench(6), area, cop(2), slip, mask_frac]` from `fields_ds` at `t0` and its predecessor with the *measured* dt (`windows.py:368-378`) | same from `latest(2)` with nominal dt = 1/8 s (`planner.py:168-175`) | ✓ (dt nominal vs measured; ~8 Hz stream, fine) |
| **reactive (CASA)** | `reactive_score(cur, prev)` on two *consecutive* `fields_ds` frames, ~125 ms apart (`windows.py:379-384`) | `reactive_score(ds_now, self._prev_fields)` where `_prev_fields` is from the **previous snapshot ≈ 0.96 s earlier** (`planner.py:179-184`) | ✗ §3.5 |
| **ur_state** | `[q, qd, tcp_pose, tcp_speed, gripper(pos,obj)]`, each stream nearest `t0` (`windows.py:266-271`), `norm("ur_state")` | latest arm sample + latest gripper ring pushed by the executor's gripper thread (`planner.py:140-153`, `executor.py:316-319`) | ✓ composition, units, normalisation (ckpt `norm_stats`, `run_deploy.py:68-72`) |
| **wrist F/T window** | `np.interp` on `linspace(t0−0.25, t0, 31)` (`windows.py:258-263`); `zero_ft` at episode start (`session.py:684-691`) | same grid anchored at newest arm sample (`planner.py:129-138`); `zero_ft` after warm-up (`runtime.py:201-205`) | ✓ (tests `test_deploy_parity_fixes.py:456-510`) |
| **text** | `meta.text or meta.task` (`windows.py:393`); teleop sets `text = s.text or s.task` (`session.py:680`) | `args.text or args.task` (`run_deploy.py:89`); cache `runs/teacher_v3_790eps/text_embeddings.pt` keys = the 8 texts listed in both v4 and v5 `configs.text_conditioning.texts` (verified on compute3) | ✓ |
| **prev_chunk** | the **executed** Δ-EE rows nearest `t0 − (16−k)/10` s, i.e. `(t0−1.6, t0]` (`windows.py:277`) | the previous **accepted plan's 16 proposals**, which cover `(now−0.96, now+0.64]` (`planner.py:278-291`, `policy.py:101-111`); first replan = normalised zero action | ✗ §3.3 (but see the structural-mask note: it cannot reach the action head) |
| **prev_cpk summary** | `two_pass` (both v4/v5): the model's *own* package for the *current* window, `step=0` = `(t0, t0+1 s]` (`rf.py:150-183`) | `flatten_summary(prev_plan.cpk, step=0)` = the **previous** replan's step 0 = `(now−0.96, now+0.04]` (`rf.py:398-400`, `packing.py:247-261`) | ✗ §3.4 |
| **contact/cpk placeholders** | GT targets (masked out of conditioning by `cond_mask`) | zeros (`policy.py:130-145`) | ✓ (targets only) |
| **action chunk** | Δ-EE of *measured* TCP between consecutive 10 Hz record ticks for `(t0, t0+1.6]`, `future_idx` (`windows.py:276`, `session.py:735-752`, period = `1/action_rate_hz` `session.py:396`) | denormalised, cumsum'd from `_last_cmd`, played from index 0 starting at `t_obs + latency` (`policy.py:163-172`, `executor.py:96-111, 143-154`) | ✗ §3.1 (timing/phase, not units) |
| **gripper channel** | teleop's last *sent* command at the end of the tick | plan value applied at the *start* of step k (`executor.py:153`) | ✓ (0.1 s early; negligible) |
| **Noise** | fresh `eps` per sample | persistent per episode (`rf.py:405-412`), `--persistent-noise` in `GO_ANY.sh` | design choice; offline evals use fresh (§3.2, §3.10) |
| **Guidance** | `cond_dropout_p = 0.1` trained (ckpt) | `1.0` (no CFG) | ✓ (untested lever; 2× latency) |
| **NFE** | n/a (ckpt `nfe=5`) | 5 (`GO_ANY.sh`) | ✓ |
| **EMA** | eval uses EMA | EMA default + `merge_lora` (`run_deploy.py:54-77`) | ✓ |
| **RoPE / layout** | `time_true`, action frames at `[0.4,0.8,1.2,1.6]` latent units *from the cond frame* (`sequence.py:96-100, 178-180`) | built from ckpt `mc` (`run_deploy.py:39-49`) | ✓ config; ✗ physics (the executed steps actually run at +0.96…+1.92 s) |
| **Backbone config** | `BackboneConfig()` | `BackboneConfig()` default (`builder.py:49`); `payload["configs"]["backbone"]` is saved but never compared (`common.py:246-255` asserts hardware shapes only) | hygiene |
| **Episode start** | `t0 ≥ start + 1.6 s` (`windows.py:167-172`: `past = max(window_s, chunk_s, 2·field_dt)`) | first replan at rest, `qd = tcp_speed = 0`, `prev_chunk = 0` | ✗ §3.8 |
| **Rollout recording (DAgger)** | teleop `STREAM_ACTIONS` = measured Δ-EE | executor records the **plan proposal** at grid-step entry (`executor.py:207-212`) | ✗ §3.6 |
| **Governor** | — | `sigma_lo=0.1, hi=1.0, min_scale=0.15` (`hardware.nuc.yaml:264-267`) on `mean σ(d_fz, slip, mask)` (`heads.py:46-51`) | inert today (§2.2) |

---

## 2. Rig-trace facts used below (compute3, read-only)

### 2.1 Latency == replan interval == executed fraction of the chunk
```
day 20260820: latency mean 1.69 s (1.33 s in 6 eps, 2.62 s in 5 eps), replan interval mean 1.68 s
day 20260828: latency mean 0.97 s, replan interval mean 0.97 s   (all 26 eps: nfe5 g1.0 pnoise)
```
`PlannerLoop.run` builds the next snapshot immediately after `submit` (`planner.py:228-292`), so the interval is the latency and
`ChunkExecutor` plays `play_time = 0 … 0.96 s` of each plan, i.e. steps 0–9 (`executor.py:184-185`, `_pose_at`). On 08-20 the
2.62 s episodes exhausted the 1.6 s chunk and held for ~1 s per cycle (`_pose_at` clamps `u ≤ H−1e-6`).

### 2.2 Governor scale
`sigma_max` percentiles 10/50/90 = 0.07/0.09/0.10 (08-28), 0.07/0.09/0.11 (08-20); scale < 0.95 in **0 %** of replans.
So playback ran at 1.0× and `rig_trace_decompose`'s assumption `ksteps = round(Δt·10)` at 1.0× (`tools/rig_trace_decompose.py:64`)
was valid. Executor exonerated for speed; the slowness is in the commanded heads.

### 2.3 Chunk head vs tail (per-replan detail, `trace_detail.py`)
* `waffles_1787923675_000` (v5_6): descending heads −26, −23, −28, −22, −14, −3 mm per 0.96 s (z 307→194 mm), then heads flip
  positive (+13, +54, +115, +153 mm) while the gripper command rises 0.31→0.52 — the model commits (closes + lifts) at
  z ≈ 190 mm, ~120 mm above the demo close height; retries the same at z ≈ 170 mm. Tails shrink with the heads here
  (deceleration is inside the chunk).
* `waffles_1787922904_000` (v4): identical shape, commit at z ≈ 182 mm.
* `whiteboard_1787941222_002` (v5_6): **17 consecutive replans with head −6…−22 mm and tail −38…−47 mm** — the model puts
  2–4× more descent in the never-executed tail at every replan; measured descent ≈ 20 mm per replan.
* `Carton_1787937561_000` (v5_6): the first 6 replans dither (+0.1, +7.9, +2.0, +26.6, −10.3, +29.9 mm heads; z 322→377 mm)
  before descent starts at replan 7.

### 2.4 Checkpoints
Both v4 and v5: `acc.self_anticipation = two_pass`, `cond_dropout_p = 0.1`, `rope_time_mode = time_true`, `nfe = 5`;
identical action norm stats (mean ≈ 0, std = [2.9, 7.2, 6.7] mm/step xyz, gripper 0.42 ± 0.28) — v5 did not recompute stats.

---

## 3. Findings

### 3.1 [HIGH] Observation→execution phase mismatch: the executed chunk head is one latency stale, the tail is never executed, and training has no model of either
**Code.** `policy.replan` stamps `action_times = t_obs + latency + k/rate` (`policy.py:165-172`). `executor.submit` computes
`u0 = (now − action_times[0])·rate ≈ 0` and rebases the plan so `pose_at(u0) == _last_cmd` (`executor.py:96-104, 109`). Hence
step k of the chunk — which in training is the displacement over `(t0 + (k)/10, t0 + (k+1)/10]` from the state observed at `t0`
(`windows.py:276`) — is applied over `[t_obs + L + k/10, …]` from the pose the *previous* plan drove the arm to during `[t_obs, t_obs+L]`.
With L ≈ 0.96 s only steps 0–9 ever run (§2.1). Training samples `t0` uniformly (or grasp-weighted) and supervises all 16 steps
equally (`common.py:444-456`, `rf.py:316-318`); nothing in the data or loss encodes a 1 s observation lag or a 60 % execution prefix.

**What it does (simulation, `sim_executor_lag.py`, perfect state-feedback policy = demo deltas for the phase matching the
observed z, demo descends at 47 mm/s and decelerates over 1 s to stop at 72 mm):**
```
L=0.0            z_min 72.0 mm   (reference)
L=0.9 index0     descent 47 mm/s   z_min 34.1 mm   (current executor: 38 mm OVERSHOOT)
L=0.9 skip-head  descent 37 mm/s   z_min 81.9 mm   (phase-correct playback, chunk exhausts → 10 mm short, 22 % slow)
L=1.3 index0     z_min 13.0 mm   (59 mm overshoot)
L=1.3 skip-head  descent  9 mm/s   z_min 117.6 mm  (chunk mostly exhausted)
```
Two consequences. (i) Even a perfect policy cannot stop where it intends within ±40–60 mm: the current loop is a 1 s pure delay
in a position-feedback loop with a ~1 s deceleration — this is the mechanism behind the 08-28 "whiteboard rollout 40 mm below any
demo" slam (`docs/rig_session_v5.md:66-68`) and it means the observed 65–120 mm under-commit is *net of* ~40 mm of lag-induced
overshoot, i.e. the policy's intended stop is even higher than measured. (ii) The chunk tail is a dead output: whiteboard 002
put 70 % of its descent in the tail at every replan (§2.3) — a "slow now, fast later" mode is indistinguishable from "stop" on
this rig. Fixing the policy's z-judgement without fixing the loop trades under-commit for table strikes.

**Fix (deploy-side, no retraining, in this order):**
1. *Cut latency first.* The loop is compute-bound (interval == latency). Measure with `bench_inference` at `--nfe 3`
   (Euler on a shifted schedule; the terminal metric at NFE 3 vs 5 is one `terminal_eval --nfe 3` run) and with `--compile`
   (already wired, warm-up runs outside the episode `run_deploy.py:297-315`, not used by `GO_ANY.sh`). Target L ≤ 0.5 s.
2. *Phase-correct playback + committed-prefix inpainting (RTC-style, Black et al. 2025, arXiv:2506.07339).* Keep the
   observation-anchored chunk semantics the model was trained with: set `t_exec0 = obs.t` in `policy.replan` so `submit`'s
   existing skip-elapsed-head branch (`executor.py:96-98`, tested by `test_submit_stale_plan_skips_elapsed_head`) starts at
   `k0 = ceil(L·rate)`, and in `rf.sample` pin the first `k0` action slots to the *noised committed actions* (the prefix of the
   executing plan that will run during the latency window, which the executor knows) at every Euler step — the action packer
   maps action `a` of latent frame `k` to channel `a` (`packing.py:284-287`), so the mask is per (frame, channel):
   frames `0..k0//4−1` fully, frame `k0//4` channels `0..k0%4−1`. This is the same FRAME_REPLACE trick already used for
   conditioning frames (`rf.py:412, 457`), applied at sub-frame granularity. With L ≈ 0.5 s that leaves 11 fresh steps
   (1.1 s) per 0.5 s replan — the loop is then inside its design regime.
3. *Longer term (fine-tune, ~3k steps):* sample the chunk at `t0 + L_aug` with `L_aug ~ U(0, 1.0)` s and feed the executed
   actions over `(t0, t0+L_aug]` as the intent input, so the model learns latency compensation explicitly. Only worth it if
   (1)+(2) leave L > 0.6 s.

**Verification before rig time:** the deploy-faithful offline metric of §3.2 with the executor's playback rule applied to the
predicted chunk, run for `index0` vs `skip+inpaint` — pick the rule offline.

### 3.2 [HIGH] `tools/terminal_eval.py` does not measure what the rig executes
`terminal_eval.py:96-115` builds the window with `sampler.sample` → `prev_chunk` = GT executed history (`windows.py:277`),
`t0 = t_close − 1.6 s` with the chunk anchored at the observation (`:123`), `prev_cpk` from a window **1.6 s** earlier
(`:137-140`) although the rig period is ~1 s, fresh noise per seed (`:132`) although the rig uses persistent noise, and the
metrics integrate all 16 steps (`:145-149`) although only ~10 execute. Every rig-specific degradation of §3.1/§3.3/§3.4 is
absent, so v4→v5 improvements measured here (20.7→17.4 mm) say nothing about the rig, and checkpoint selection for a 15–20
rig-hour budget is being made on it. It also cannot detect the whiteboard "descent in the tail" mode because the tail is scored.

**Fix (~40 lines):** add `--deploy-faithful`: (a) anchor `t0 = t_close − 1.6 − L` and score only the *head* (`ceil(L·rate)`
steps) applied from the GT pose at `t0 + L` against the GT trajectory (endpoint of the head, z-at-head-end); (b) `prev_chunk`
= the previous *prediction* (window at `t0 − L`), `prev_cpk` = that prediction's package at `step = round(L/latent_dt)`;
(c) reuse the initial noise across the two windows; (d) also report `head_dz / tail_dz` and its spread over 8 seeds
(→ §3.10). Report the v4/v5 numbers under this mode before the A/B.

### 3.3 [MEDIUM] `prev_chunk` at deploy is the previous *proposal*, not the executed past — and it cannot reach the action head anyway
`planner.py:278-291` feeds `prev_plan` (the last accepted plan) and `policy.py:101-102` normalises its 16 proposals as
`prev_chunk`. Those cover `(now−0.96, now+0.64]`; training's `prev_chunk` covers `(t0−1.6, t0]` of *measured* motion
(`windows.py:277`, recorded by `session.py:735-752`). The DEFERRED comment acknowledges it. Two additional facts:
* **[certain] The intent path is structurally inert for actions.** `prev_chunk` enters (a) the pretrained action-AdaLN, which
  modulates **only VIDEO_GEN frames** (`phantom_dit.py:131-157`: `emb[:, vid] = a_emb`, all other frames zero) and
  (b) ACC's `intent_mlp` (`acc.py:62, 84`). The structural mask forbids CONTACT/ACTION queries from attending VIDEO_GEN keys
  (`sequence.py:188-197`). So the only route from `prev_chunk` to the sampled actions is the ACC key bias on haptic tokens
  (`phantom_dit.py:241-252`, λ init 0). The mismatch therefore mainly corrupts the video rollout and the ACC gate, not the
  chunk — and, for the paper, "the previously committed chunk conditions the next chunk" is not true of the action head.
* At the first replan the zero action is normalised correctly (`policy.py:103-111`, tested).

**Fix (exact parity, ~15 lines in `SnapshotBuilder.build`):** resample the arm ring's `tcp_pose` at `t_ft − (16−k)/10` s
(k = 0..15) and take `pose_delta` between consecutive samples (`derived.py:62-71`, the teleop convention) with the gripper
column from the executor's last sent targets (or the gripper ring); pass it in `ObsSnapshot` and use it instead of
`prev_plan.actions`. Keeps the first-replan zero case.

### 3.4 [MEDIUM] `prev_cpk` is fed one latent step stale
`rf.sample` overwrites the ACC summary with `flatten_summary(prev_cpk)` at the default `step=0` (`rf.py:398-400`,
`packing.py:247-261`). The latent grid step is `temporal_comp/fps = 1.0 s` (`windows.py:135-137, 282`), so step 0 of the
previous replan's package describes `(t_prev, t_prev+1] ≈ (now−0.96, now+0.04]` — the interval that has just elapsed — while
training in `two_pass` mode feeds step 0 of the *current* window's own prediction (`rf.py:163-168`), i.e. `(t0, t0+1]`
(the docstring's `c_hat_{t+1|t-1}` would be step 1 of the previous package).
**Fix:** `step = int(round(prev_plan.latency_s / latent_dt))` (= 1 at the rig's period) when building the summary in
`rf.sample`; or pass `flatten_summary(..., step=1)` explicitly from the policy. Affects only the ACC gate path.

### 3.5 [MEDIUM] CASA reactive score compares tactile frames 0.96 s apart instead of 0.125 s
`planner.py:179-184` keeps `self._prev_fields` from the *previous snapshot* (≈ latency ago; and during warm-up, from
`_wait_rings_warm`'s repeated builds) and computes `reactive_score(ds_now, _prev_fields)`. Training uses the fields_ds frame
and its immediate predecessor (`windows.py:379-384`, `_field_frame` `:202-213`), ~125 ms apart. The score is a mean-abs
difference (`derived.py:226-230`), so the deploy value is inflated by whatever the gel did over ~1 s. It drives `g_react`
via `psi_react` (`acc.py:100-102`) — teacher only, gate path only.
**Fix:** the two latest `fields_ds` frames are already fetched at `planner.py:168-170`; use `dv.reactive_score(cur, prev)`
from them (per finger, then the stack) and drop `_prev_fields`.

### 3.6 [MEDIUM] DAgger rollouts record plan proposals as `STREAM_ACTIONS`
`executor.py:207-212` calls `record_action(t0, plan.actions[k])` when a grid step is *entered* — the un-executed proposal
(tail included, rebased away at the next swap) — under the canonical `STREAM_ACTIONS` name (`recorder.py:103-110`).
Teleop records the measured Δ-EE of consecutive TCP poses at the *end* of each tick (`session.py:735-752`). When rollouts are
merged via `distill_hid --extra-data` / `dagger_driver`, their `action_chunk` and `prev_chunk` (`windows.py:276-279`) will be the
teacher's proposals, not the motion, and mis-timed by 0.1 s + the swap discontinuities. The relabel targets themselves are fine
(`relabel.py:38-49` re-samples the teacher), but every "GT grounding" term on rollout windows is wrong.
**Fix:** record measured Δ-EE at 10 Hz in the deploy path (a 10 Hz tick in `PlannerLoop`/recorder using the same `pose_delta`
on the arm ring, exactly `session.py:735-752`), and write proposals to a separate stream (e.g. `actions_proposed`).

### 3.7 [MEDIUM] Missing experiment / unverified assumption: does the action head read z from proprio at all, and are camera extrinsics unchanged?
The waffles traces show a clean commit at z ≈ 180–190 mm (§2.3), with `tcp_pose.z` in `ur_state` sitting ~110 mm (≈ several
σ of the demo z distribution) above any demo close. Either the model ignores proprio z (near-top-down camera is then the only
z cue, and gripper apparent size is a weak one), or the camera's view differs from collection (the 08-18/20 lighting change
already forced photo-aug). Both are cheap to test and neither has been:
1. **Proprio sensitivity (offline, minutes on compute3):** on terminal windows perturb `ur_state` tcp-z by ±50/±100 mm (and
   qd/tcp_speed ×0.7 for the velocity-self-consistency suspect) and measure `∂(head dz)/∂z`. If ≈ 0 the model is vision-only
   for height — the fix is training-side (proprio dropout of the *image*, or a z-conditioned auxiliary), not a deploy tweak.
2. **Extrinsics check (rig, 1 minute):** at the gated start pose (≤ 2.5 σ) compare the gripper's pixel bbox in the first deploy
   frame with demos at the same pose; > a few px means the model's z cue is shifted and every episode is OOD in the input that
   dominates the conditioning ablation.

### 3.8 [MEDIUM/LOW] The deploy start state is excluded from training by construction
`valid_range` requires `t0 ≥ start + max(window_s, chunk_s, 2·field_dt) = start + 1.6 s` (`windows.py:167-172`) so that a full
`prev_chunk` exists. Deploy's first replan is at rest (`qd = tcp_speed = 0`, `prev_chunk` = normalised zero, gripper settled) —
a state the sampler never produced. Carton 000 dithered for 6 replans / 55 mm upward before descending (§2.3); waffles started
descending at once, so this is task-dependent. **Fix:** allow `t0` down to `start + window_s` with the pre-episode part of
`prev_chunk` zero-padded (physically correct for a resting arm: teleop starts each episode's Δ chain fresh, `session.py:698-702`)
and let the v5-style fine-tune see those windows; verify the first-replan head on the faithful eval.

### 3.9 [LOW] Governor thresholds are uncalibrated to the sigma head (currently harmless)
Inert today (§2.2), but `sigma_lo = 0.1` sits at the *90th percentile* of observed σ and `sigma_reg` (`rf.py:327`) pushes
`log σ → 0`, i.e. σ → 1 → scale 0.15. Any re-calibration of the contact heads (e.g. the fine-tune on failure demos raising σ in
contact-rich phases) silently slows playback and re-creates the "slow arm ⇒ slow mode" loop. For the A/B set
`safety.governor.min_scale: 1.0` (documented switch, `docs/launch_guide.md:159`) or set `sigma_lo/hi` from offline σ percentiles.

### 3.10 [LOW, guess] Persistent noise locks a per-episode chunk-shape mode that the executor never gets past
With `--persistent-noise` the same initial noise is reused every replan (`rf.py:405-412`), so a sample-level mode such as
"slow head, fast tail" (whiteboard 002, 17/17 replans) or "commit early" becomes an episode-level constant, and the executor only
ever plays the head. The rig note "one noise seed descended at full speed" is consistent. Test = the seed spread of `head_dz`
on terminal windows in the faithful eval (§3.2 d). If the spread is large, either re-draw noise every N replans, or run paired
`--seed` trials and report the distribution rather than one seed.

### 3.11 [LOW] Hygiene
* Deploy builds `BackboneConfig()` defaults (`builder.py:49`) and never compares them to `payload["configs"]["backbone"]`
  (`common.py:246-255` asserts hardware shapes only) — add the assert next to the hardware one.
* Contact-state dt at deploy is nominal (`1/field_ds_rate_hz`, `planner.py:160`) while training uses the measured gap; use the
  ring timestamps from `latest(2)`.
* 08-20 had five episodes at 2.62 s latency (chunk exhausted every cycle) with no condition tags; the cause (guidance 2? GPU
  contention?) is not recorded anywhere — provenance tags now exist, keep them.

---

## 4. Executor rebase/blend math — checked for systematic shortening
`submit`: `u0 ≈ 0`, `c0 = u0·cum[0]`, `t0_pose = _last_cmd − c0` → `pose_at(u0) == _last_cmd` exactly (`executor.py:96-104`);
`_pose_at` interpolates cumulative deltas linearly (`:143-154`). Blend: prev plan continues on its own clock for 0.1 s and is
cross-faded (`:190-203`); both trajectories start at `_last_cmd`, so the blend error is ≤ one step's delta difference.
Rate limiter caps 0.25 m/s / 1 rad/s per measured tick (`:231-253`); demo descent (≤ 50 mm/s) never trips it; demo transport
peaks 0.24–0.34 m/s do (`docs/rig_session_v5.md:71-72`) — post-grasp only. Stale hold at 2.6 s (`:173-179`). **No shortening
in the executed prefix** — consistent with `rig_trace_decompose` (within mm). The mismatch is purely *which* steps run *when*
(§3.1), not their magnitude.

## 5. Verified-OK list (for the record)
RGB order through JPEG (`jpeg_codec.py:55, 79`, RealSense `rgb8` `realsense.py:73`); resize/normalisation identical function;
cond-latent-only encode is equivalent for a causal VAE (`rf.py:99-109`) — **[believe]**, not unit-tested against the real Wan VAE;
`ur_state` 26-dim order matches `dump_norm_stats.py:70-91`; wrist grid tested; gel 1-ch→3-ch both sides; text cache keys;
EMA + LoRA merge; `mc` from checkpoint (`run_deploy.py:39-49`); norm stats from checkpoint; bf16; NFE 5; guidance 1.0;
persistent noise reset per episode (`runtime.py:178-179`); warm-up replan outside the episode; `--seed` reseeds `_gen`.

## 6. Suggested order of work (3 weeks, rig-hours scarce)
1. Faithful offline eval (§3.2) — half a day; re-score v4/v5.
2. Latency: NFE 3 + `--compile` benchmark (§3.1 step 1) — hours.
3. Phase-correct playback + prefix inpainting (§3.1 step 2) — 1 day; validate on (1).
4. The three exact-parity one-liners (§3.3 prev_chunk from the ring, §3.4 step index, §3.5 reactive) — hours; governor off for the A/B.
5. Proprio-z sensitivity + extrinsics check (§3.7) before the next rig session — decides whether the remaining error is
   training-side.
6. Rollout action recording (§3.6) before the first DAgger/self-improvement round.
