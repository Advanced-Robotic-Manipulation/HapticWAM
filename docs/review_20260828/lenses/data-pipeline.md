# PHANTOM review — lens: DATA PIPELINE (2026-08-28, repo HEAD 3539988)

Scope: `phantom/data/{episode_store,windows,schema,derived}.py`, `phantom/train/common.py`
(WindowDataset, manifest_split, close_index, grasp-frac anchoring, photo-aug),
`tools/{intake_recovery,episode_qc,terminal_eval}.py`, `phantom/scripts/dump_norm_stats.py`,
`configs/start_poses.yaml`, and — because the question "is every input stream built identically
for training and the rig" cannot be answered from the data modules alone — the deploy-side
observation builders `phantom/deploy/planner.py` (SnapshotBuilder), `phantom/inference/policy.py`
(`_batch_from_obs`), `phantom/deploy/executor.py` (`record_action`), and the collection-side
action writer `phantom/data_collect/session.py`.

Method: read every file above in full; ran `pytest tests/test_windows.py tests/test_v5_intake.py
tests/test_terminal_eval_close.py tests/test_v5_holdout.py` (16 passed); ran a read-only
profiling script on compute3 over the real data that is there (`data/val_eval/tasks`, 78 v4 val
episodes + `norm_stats.json` + `manifests/all.jsonl`, and the first 60 episodes of
`data/incoming_recovery` = batch_20260822 waffles). Script:
`scratchpad/review/profile_remote.py`. No files under the repo were modified.

Certainty labels: **[certain]** = read in code / measured; **[believe]** = strong inference from
code + data; **[guess]** = hypothesis worth one cheap experiment.

---

## 0. TL;DR

1. **The proprio state after a *successful* close is indistinguishable from an empty close for 3 of 4
   tasks** [certain, measured]. In success demos the Robotiq OBJ code 1 s after the close is 3
   ("reached requested position, no object") in Carton 17/20, egg 11/17, waffles 24/36 (whiteboard:
   2 = 21/21). Operators command ~0.55–0.61 aperture on soft objects; the fingers reach it; OBJ=3.
   So (gripper closed to ~0.47, OBJ=3) is *the success signature* in training and is exactly what
   the rig produces when it closes on air. The only stream that can tell the two apart is tactile
   contact — and the mapping "closed + no contact → do not lift / reopen" has **zero action
   supervision** because failure demos carry `action_weight=0` (`windows.py:403`). The model does
   what the data says: closed ⇒ lift.
2. **`prev_chunk` is a different object at deploy than in training** [certain]. Training: the 16
   *measured* Δ-EE steps over `(t0-1.6 s, t0]` (`windows.py:277`). Deploy: the previous *accepted
   plan* — 16 *proposed* steps spanning roughly `[t0-0.9 s, t0+0.7 s]`, ~7 of which were never
   executed, all reshaped by governor/blend/rate-limit before the arm saw them
   (`policy.py:101-111`, `planner.py:284-291` marks this DEFERRED). This is a closed loop from the
   policy's own output to its "intent" input. It is the plausible carrier of "slow mode locks in";
   the proprio-velocity channels are *not* (a 30 % slower descent is 0.05–0.07 σ in `tcp_speed`,
   measured from the real `norm_stats.json`).
3. **The A/B placement protocol is outside the demo distribution** [certain from val data; the
   full-train spread may be somewhat wider]. TCP position at close (a proxy for object placement)
   has σ_xy ≈ 4/6 mm (egg), 6/7 mm (whiteboard), 17/11 mm (Carton), 21/15 mm (waffles). A 3×3 grid
   at 40 mm spacing (`docs/rig_session_v5.md:19,35`) puts off-centre cells at ~2 σ (waffles/Carton)
   to 6–10 σ (egg/whiteboard). Any grid-cell episode conflates "policy under-commits" with
   "object never seen there".
4. **The offline terminal metric cannot see the rig failure** [believe, with measured support].
   It conditions on *demo-consistent* `prev_chunk`, proprio, contact state and a demo scene frame;
   offline `commit_ratio` is 1.43–1.72 (model descends *more* than GT) while the rig under-commits.
   The v4 conditioning ablation on compute3 (`EVAL_RESULTS_v4.txt:11-12`) shows knocking out video
   shifts the sampled chunk by only 1.85 mm, prev_chunk 0.93 mm, wrist 0.25 mm — the action head is
   barely observation-driven on uniformly-anchored windows. Four cheap GPU-only variants of
   `terminal_eval` would expose the failure mode before spending a rig-hour (section 4).
5. Uniform-in-time anchoring is dwell-time weighting [certain]: 13–27 % of every demo is
   near-stationary, and in the pre-close band the hover fraction is egg 72 %, waffles 24 % (p90
   65 %), Carton 12 %, whiteboard 10 %. The v5 grasp band `[tc-1.5, tc-0.2]` concentrates on
   exactly the decelerate-and-hover segment.

What checks out (no finding): the Δ-EE action stream is built consistently and honestly
(collect-loop period median 0.1003 s, p90 0.1034 s; |cumsum of 16 recorded deltas − measured
displacement| median 1.1 mm, p90 2.4 mm); wrist F/T is zeroed at episode start on both paths and
resampled onto the same time grid; the video conditioning latent is a function of the current frame
only (causal Wan VAE, `rf.py:113-118`); `ur_state` is the same 26-vector in the same order on both
paths; failure demos never enter `val`; `manifest_split` enforces a bijection; NaN CoP targets are
intentional and observation features are `nan_to_num`'d; the close rule fires on every profiled
episode (0 `no_close` in 98).

---

## 1. Stream-by-stream parity: training window vs rig snapshot

| stream | training (`windows.py`) | rig (`planner.py` / `policy.py`) | verdict |
|---|---|---|---|
| `video` | 13 frames from t0 on the 4 Hz grid, nearest camera frame (±33 ms at 15 fps) `:253-255`; only latent 0 conditions (`rf.py:117-118`), causal VAE | current frame tiled ×13, `encode_gen=False` `policy.py:85-89`, `rf.py:390` | **parity** for the cond latent. Frame age ≤ ~70 ms + 0.9 s inference (see §3.6) |
| `wrist` | `arm_ft` resampled with `np.interp` onto `linspace(t0-0.25, t0, 31)` `:258-263`, after `zero_ft` at episode start (`session.py:685-690`) | same resampling anchored at the newest arm sample `planner.py:129-138`; `zero_ft` at `runtime.py:201-205` | **parity** (fixed 08-27) |
| `ur_state` | `[q, qd, tcp_pose, tcp_speed, gripper(pos,obj)]` nearest to t0 `:266-271` | `[q, qd, tcp_pose, tcp_speed, gripper(pos,obj)]` latest `planner.py:151-153`; gripper ring pushes `[position, obj]` `executor.py:318-319`, collection pushes the same `gripper.py:250-251` | **parity** in construction; **semantic problem** in what the gripper channels can tell (§3.1) |
| `prev_chunk` | 16 *measured* Δ-EE + commanded gripper over `(t0-1.6, t0]` `:277` | previous *accepted plan* (proposal, future-shifted) `policy.py:101-111` | **MISMATCH** (§3.2) |
| `action_chunk` target | first recorded action at time ≥ t0+(k+1)/10 `:276`, i.e. the motion in `(t0+k/10, t0+(k+1)/10]` | executed from `obs.t + latency` (0.9 s later), rebased to the current commanded pose `executor.py:96-111` | **latency mismatch**: training has 0 s obs→action lag, deploy 0.9 s (§3.6) |
| `reactive` | `reactive_score(cur, prev)` between *consecutive fields_ds frames* (~0.125 s) `:379-384` | `reactive_score(ds_now, _prev_fields)` where `_prev_fields` is from the *previous replan* (~0.9 s) `planner.py:182-184` | **MISMATCH** ×5–7 during motion (§3.8) |
| `fields`, `gel`, `contact_state` | keyframe / infer_img / derived at nearest t0 | latest ring rows; `derive_timestep` with fixed `dt_field=1/8` instead of the real Δts | parity (slip's `/dt` differs by ≤30 %; minor) |
| `text` | `meta.text or meta.task` `:393` | `args.text or args.task` `run_deploy.py:89` | parity; unknown keys fall back silently to the empty embedding with one warning (`text_embedding.py:69-83`) — grep the train log for "not in the embedding cache" once |
| recorded `actions` on rig episodes | n/a (demos: measured `pose_delta`, `session.py:733-750`) | `plan.actions[k]` = *planned* delta at the tick playback enters step k `executor.py:207-212` | **semantic MISMATCH** for any future use of rollouts as data (§3.7) |

---

## 2. Normalisation sanity (`dump_norm_stats.py`, real `norm_stats.json` on compute3)

`ur_state` stds (from `data/val_eval/tasks/norm_stats.json`, the file v4/v5 trained with —
`train_teacher.py:237-252` hard-fails if the fine-tune root's stats differ from the init
checkpoint's, so the fine-tune did keep them):

```
q0..q5  std  0.351 0.096 0.305 0.344 0.190 0.129   (q5 mean -3.056 rad = -175°, 5° from the ±π wrap)
qd0..5  std  0.176 0.074 0.219 0.195 0.097 0.078   rad/s
xyz     std  0.053 0.129 0.089 m ; rotvec std 0.33 0.32 0.29 rad
vxyz    std  0.029 0.070 0.066 m/s ; wxyz 0.090 0.127 0.173
grip    pos mean 0.354 std 0.191 ; obj mean 2.40 std 1.03
action  std  dx 2.9 mm  dy 7.2 mm  dz 6.7 mm per 0.1 s ; grip cmd mean 0.42 std 0.28
wrist   std  11.0 12.0 13.3 N / 2.3 3.8 2.4 Nm   (CB3 current estimate; noise-dominated)
```

* **A 30 % slower descent is invisible in normalised proprio** [certain]: measured per episode,
  `max_c |Δσ_c|` over `(vx,vy,vz)` for `v → 0.7 v` in the pre-close band = 0.05–0.07 σ (Carton
  0.067, waffles 0.051, whiteboard 0.059, egg 0.015). The velocity stds are set by transport
  (peaks 0.24–0.34 m/s), so the whole approach phase lives within ±0.5 σ. Conclusion: the
  "velocity self-consistency via proprio" suspect in CONTEXT.md is not a proprio phenomenon; if
  velocity feeds back, it is via `prev_chunk` (per-step dz std 6.7 mm ≈ 67 mm/s — same scale
  problem, but 16 correlated steps and an explicit "intent" pathway into ACC).
* `obj` codes: 3 is the modal value (mean 2.4); the codes are ordinal-encoded and z-scored, so
  0 (moving) = −2.3 σ, 2 (object stopped fingers) = −0.4 σ, 3 = +0.6 σ. Not a bug, but the
  encoding gives the model no reason to treat 2 vs 3 as categorical — and in 3/4 tasks it does not
  matter because success demos mostly end in 3 anyway (§3.1).
* `dump_norm_stats.py:49` uses `list_episodes(root)` → stats include val and `_fail` episodes.
  Harmless (statistics only), but note the file actually used is `dataset_v3_packed/norm_stats.json`
  pulled from the hub (`provision_v5.sh:116`), i.e. computed on an older dataset version. Consistent
  across v4→v5 (checked at `train_teacher.py:237-252`), so no action.
* Variance floor `1e-12` (`dump_norm_stats.py:79`) → std floor 1e-6. No channel is near it in the
  real stats; only a hazard for synthetic/mock data.
* `wrist_ft` std of ~12 N on a stream whose real signal is a few N (CB3 current-based estimate with
  ±18 N pose bias, re-zeroed at start) means the WristTCN sees mostly noise at ±0.3 σ. Consistent
  with the ablation's 0.25 mm sensitivity. Not a data bug; a paper-claim caveat (the ACC "wrist
  window" input is inert on this arm).

---

## 3. Findings (ranked)

### 3.1 HIGH — Post-close proprio is degenerate: success and empty close look the same; the recovery mapping has no action supervision
* **Where**: `phantom/data/windows.py:403` (`action_weight = 0.0 if is_failure_demo`),
  `phantom/data/schema.py:155-171`, `tools/intake_recovery.py:12-17` (under-grasp demos "continue the
  task with an EMPTY gripper — if they were ever imitated they would teach exactly the rig failure
  mode").
* **Evidence** [certain, measured on compute3]: OBJ code sampled 1 s after the measured close in
  *success* demos: Carton {3:17, 2:2, 0:1}, egg {3:11, 0:4, 2:2}, waffles {3:24, 2:7, 0:5},
  whiteboard {2:21}. Commanded close aperture (`actions[:,6].max`) 0.55–0.61 for the three soft
  objects vs 0.86 for the eraser; measured `pos_at_close` 0.46–0.49 everywhere. The fingers reach
  the commanded position on soft objects, so the Robotiq reports "no object". The rig failure
  ("closes on air to a demo-like aperture, then LIFTS") therefore produces a `ur_state` that is
  *inside* the success distribution. Only tactile contact differs — and every training window with
  a closed gripper *and* action supervision has contact, so the action head never had to consult
  the contact gate. The 45 under-grasp failure demos contain the exact rig state but contribute
  nothing to the action loss.
* **Failure scenario**: rig closes 65–120 mm high → gripper reaches ~0.5, OBJ=3, `p_none`=0.99 →
  the action head, conditioned on (closed, OBJ=3, scene shows arm at object) has only ever seen
  "lift and transport" for that proprio → lifts. Exactly the 08-28 trace.
* **Fix (data, minimal)**: give the closed-on-air state action supervision.
  (a) Cheapest: relabel the post-close segment of the 45 under-grasp demos (from the empty close to
  the end) with a fixed recovery primitive — `[0,0,0,0,0,0, open]` for 0.5 s then hold — with
  `action_weight=1` (a new `recovery_relabel` flag in `WindowSampler.sample`, keyed on
  `is_failure_demo and t0 > t_close`). ~45 × 8 windows, one fine-tune. (b) Better: record 20–30
  true recovery demos per task starting *from* the failure state (closed on air, 50–100 mm above the
  object): open → descend → grasp → lift. Tag `recovery`, `success=True`, weight 1. This is 2–3
  rig-hours and is the single data change most likely to move the rig outcome. (c) Independent of
  training, a deploy reflex "closed for > 0.5 s and `mask_frac` < τ → open and re-descend to the
  pre-close height" turns the tactile gate the model already trusts into the loop closure the paper
  claims; the trace already contains `p_evt`.

### 3.2 HIGH — `prev_chunk` at deploy is the policy's own future-shifted proposal, not the executed past
* **Where**: `phantom/inference/policy.py:101-111` (prev = `prev_plan.actions`);
  `phantom/deploy/planner.py:284-291` (explicitly DEFERRED); training construction
  `phantom/data/windows.py:277` (`t0 - (H - arange(H))/rate`, nearest recorded measured delta).
* **Evidence** [certain]: in training `prev_chunk[k]` is the *measured* `pose_delta` over
  `(t0-1.6+0.1k, t0-1.5+0.1k]` — it encodes the arm's realised velocity for the last 1.6 s. At
  deploy, with replans every ~0.9 s and latency 0.9 s, the previous accepted plan's `action_times`
  span `[t0-0.9, t0+0.7]`: ~45 % of the "past" is in the future, all of it is a proposal, and the
  executor plays back a governor-scaled, blended, rate-limited version of it (`executor.py:182-185,
  194-200, 240-253`). The intent that the ACC and DiT condition on is thus the policy's own previous
  output. `rf.py:176-182` feeds it to ACC as `intent_B_H_A`; `rf.py:441` feeds it to the DiT as
  `action=`. The 16k conditioning ablation measured 0.93 mm sensitivity to zeroing it on uniform
  windows — small on average, but that ablation used raw zeros (= the per-dim demo *mean* action,
  `conditioning_ablation.py:86`), and the terminal band was not isolated.
* **Failure scenario**: one seed samples a slow chunk → the next replan's intent says "we have been
  moving slowly" → the model stays in the slow mode (consistent with "one seed descended at full
  speed": the seed picks the mode, the intent loop keeps it).
* **Fix**: build `prev_chunk` on the rig exactly as in collection: take the arm ring
  (`rings["arm"].latest(n)` with n ≥ 1.8 s × 125 Hz — the ring holds 20 s), pick the measured
  `tcp_pose` nearest to each of the 16 grid times `t_now - (16-k)/10`, chain `pose_delta`, and
  append the gripper command actually sent at those times (the executor already knows
  `plan.actions[k,6]` per entered step). 30 lines in `SnapshotBuilder.build`. Validate offline with
  the terminal-eval variant in §4(b) before the rig.

### 3.3 HIGH — The 3×3 / 40 mm placement grid is outside the demo placement distribution
* **Where**: `docs/rig_session_v5.md:19` ("taped 3x3 grid (40 mm spacing)"), `:35` ("Per grid
  cell: one v4 episode, then one v5 episode"). Data: `configs/start_poses.yaml` encodes the *arm*
  start spread (σ 27–30 mm) but nothing about *object* placement; `tools/gen_start_poses.py` never
  computes it.
* **Evidence** [certain on the val subset, n=17–36 per task]: TCP xy at the measured close, σ:
  egg 3.6/6.0 mm, whiteboard 5.9/6.8 mm, Carton 16.7/10.9 mm, waffles 21.4/14.6 mm. The 40 mm
  neighbours of the centre are 2–3 σ out for waffles/Carton and 6–10 σ out for egg/whiteboard;
  corners are √2 further. Close height σ is 4–13 mm (egg 72 mm, waffles 79, Carton 104, whiteboard
  156).
* **Failure scenario**: an off-centre placement changes the scene frame in a way no demo covered;
  the near-top-down camera's z-ambiguity plus a translated object make "descend to the object" an
  extrapolation. The A/B then measures generalisation, not commit — and 0/26 could be partly
  protocol.
* **Fix**: run the A/B at the demo-mean cell first (10+10 episodes), then ±1 cell; log the cell id
  in `meta.json` tags (the notes prompt is free text today), and report per-cell miss distance. Add
  `close_xy_mean/std` per task to `start_poses.yaml` via `gen_start_poses.py` (it already reads
  `arm_tcp_pose` and `gripper`; `close_index` is importable) so the operator can see how far a cell
  is. If placement generalisation is a paper claim, the demos must be augmented with translated
  placements — that is a data collection item, not a training knob.

### 3.4 HIGH — The offline terminal metric conditions on demo-consistent inputs and cannot expose the failure
* **Where**: `tools/terminal_eval.py:118-143` (anchor `tc - 1.6`, `sampler.sample(ep, t0)` → GT
  `prev_chunk`, GT `ur_state`, GT scene, GT `contact_state`; `prev_cpk` from a prior GT window).
* **Evidence** [certain]: all conditioning comes from a demo that is already descending at demo
  speed with a demo-placed object; `commit_ratio` = 1.72 (v4) → 1.43 (v5) means the sampled
  chunk descends *more* than the demo over the last 1.6 s, while the rig under-commits. The metric
  moved in the fine-tune and the rig did not. The v4 knockout ablation (`EVAL_RESULTS_v4.txt`:
  video 1.85 mm, tactile 1.37, text 1.25, prev_chunk 0.93, wrist 0.25) says the sampled chunk is
  dominated by the prior + noise seed rather than by the observation — so a low endpoint error can
  largely be "the average terminal chunk".
* **Failure scenario**: a fine-tune that improves the prior for the terminal band improves
  `endpoint_err_mm` and does nothing on the rig (v5).
* **Fix**: four GPU-only variants of `terminal_eval` (each ~20 min on compute3), in this order:
  (a) *obs-null baseline* — same windows with `_null_obs_batch` + `null_video_cond` (the code path
  exists, `rf.py:396-397`): if endpoint error stays ~17 mm, the metric measures the prior;
  (b) *intent loop* — replace `prev_chunk` by the model's own sample from `t0-0.9 s` (time-shifted
  as at deploy); (c) *slow arm* — scale `prev_chunk[:, :6]` and `ur_state[qd, tcp_speed]` by 0.7 in
  physical units; (d) *latency* — anchor observations at `t0` but score against the GT chunk on
  `(t0+0.9, t0+2.5]`. Report commit_ratio and z_end for each. Whichever collapses is the rig
  mechanism, found without a rig-hour.

### 3.5 MEDIUM — Time-uniform anchoring is dwell-time weighting; the v5 grasp band sits on the hover
* **Where**: `phantom/data/windows.py:194` (`rng.uniform(lo, hi)`), `phantom/train/common.py:447-455`
  (`grasp_frac` band `[tc-1.5, tc-0.2]`, default `grasp_window_s=(1.5, 0.2)` at `:379`),
  `tools/provision_v5.sh:272` (`--grasp-frac 0.3`).
* **Evidence** [certain, measured]: whole-episode fraction with |v| < 10 mm/s: Carton 17 %, egg
  27 %, waffles 16 %, whiteboard 13 %. Inside the pre-close band `[tc-1.5, tc-0.2]`: hover fraction
  egg 72 %, waffles 24 % (p90 65 %), Carton 12.5 %, whiteboard 10 %; band-mean descent speed egg
  2.9 mm/s, waffles 20, Carton 25, whiteboard 22 mm/s. The demos are bimodal in the terminal phase
  (decisive vs decelerate-hover-close). Uniform anchoring supervises "stationary chunk" targets in
  proportion to time spent stationary; the grasp band supervises the hover in proportion to its
  length. The offline improvement in `commit_ratio` (1.72 → 1.43, i.e. *less* descent) is what
  fitting the hover better looks like.
* **Failure scenario**: with z-ambiguous vision, the "decelerate and settle" mode is entered at the
  wrong height and is self-consistent (it is a legitimate demo mode), then the close fires at the
  demo-like timing.
* **Fix**: (i) motion-weighted anchoring: draw `t0` with density ∝ `|v_tcp(t0)| + ε` (one extra
  read of `arm_tcp_speed` in `build_index`/`__getitem__`, cached per episode); (ii) move the grasp
  band earlier, e.g. `(2.5, 0.8)`, so the chunk covers the decisive descent rather than the settle;
  (iii) log per-batch GT displacement magnitude so a fine-tune's target distribution is visible in
  the training log (today `grasp_coverage()` only counts episodes).

### 3.6 MEDIUM — Zero observation→action latency in training vs 0.9 s at deploy
* **Where**: `phantom/data/windows.py:276` (targets start at `t0 + 0.1`), vs
  `phantom/inference/policy.py:167,172` (`t_exec0 = obs.t + latency`) and the rebase at
  `phantom/deploy/executor.py:96-111`.
* **Evidence** [certain]: the model is trained to emit the motion that starts *at* the observation;
  on the rig the chunk starts executing 0.9 s later from wherever the arm then is. At 30–45 mm/s
  that is 27–40 mm of unmodelled displacement per replan, ~56 % of the chunk duration. The executor
  rebases (continuity), so the arm executes "the first 0.9 s of a plan made for a state 0.9 s ago"
  every cycle; the model never trained on that pairing.
* **Failure scenario**: near the object, a chunk that (correctly) says "descend 30 mm then settle"
  is executed from 30 mm lower → settle at the wrong height; the next observation then sees a
  settled arm and continues the settle mode.
* **Fix**: latency augmentation in `WindowSampler.sample`: draw `lag ∈ U[0, 1.0]` s (or the
  measured 0.9 ± 0.2), keep observations at `t0`, take `action_chunk` on `(t0+lag, t0+lag+1.6]`
  and `prev_chunk` on `(t0+lag-1.6, t0+lag]`, and pass `lag` as an extra scalar input (or fix it at
  the deployed value). A `--obs-lag-s` flag on `train_teacher`; one fine-tune. Offline check = §4(d).

### 3.7 MEDIUM — Rig episodes record *planned* deltas as `actions`; demos record *measured* deltas
* **Where**: `phantom/deploy/executor.py:207-212` (`record_action(t0, plan.actions[k])` on entering
  step k) vs `phantom/data_collect/session.py:733-750` (`pose_delta(prev_tcp, tcp)` per 10 Hz tick).
* **Evidence** [certain]: between the plan and the arm sit the governor (`executor.py:182-185`,
  playback scaled by `max(sigma)` down to `min_scale=0.15`), the blend (`:194-200`) and the
  per-tick rate limit (`:240-253`); the recorded action is the pre-governor proposal. Timestamps
  are the tick at which playback *entered* the step, so a governed-slow chunk records 16 full-size
  deltas spread over > 1.6 s.
* **Failure scenario**: the planned self-improvement / DAgger-lite phase ingests successful rollouts
  (the standard trick) — `WindowSampler` would train on deltas the arm did not perform, with a
  time base that does not match `arm_tcp_pose`. `tools/rig_trace_decompose.py` separately compares
  planned vs actual and is unaffected.
* **Fix**: in the deploy recorder, write the measured `pose_delta` chain at 10 Hz (same code as
  collection; `record_action` already accepts a `stream=` argument) into `actions`, and the
  proposals into `actions_plan`. Then `is_failure_demo` + `success` labelling of rollouts works
  unchanged.

### 3.8 MEDIUM — `reactive` is computed over ~0.125 s in training and ~0.9 s at deploy
* **Where**: `phantom/data/windows.py:379-384` (consecutive `fields_ds` frames at t0) vs
  `phantom/deploy/planner.py:182-184` (`_prev_fields` is overwritten once per `build()`, i.e. per
  replan).
* **Evidence** [certain]: `reactive_score` is a mean absolute frame difference; during any tactile
  change it scales with elapsed time, so the deployed value is 5–7× the trained one at contact
  transitions and slightly *lower* during slow drift (fields_ds Δt jitter is 0.11–0.17 s in
  training vs the fixed 0.9 s). It feeds `AccInputs.react_score_B` (`rf.py:176-182`), i.e. the
  gate/event heads whose `p_none 0.99` the team is reading on the rig.
* **Failure scenario**: ACC gate behaves differently at contact onset/release on the rig than it
  was validated offline; no effect on the action head directly.
* **Fix**: in `SnapshotBuilder.build`, compute `reactive` from `tac["fields_ds"][-1]` vs
  `tac["fields_ds"][0]` — both are already fetched (`planner.py:168-170`) — and drop `_prev_fields`.

### 3.9 LOW — Velocity channels: nothing to fix, but stop chasing them
* `dump_norm_stats.py:70-72` accumulates each stream over its full length; the resulting
  `tcp_speed` stds (29/70/66 mm/s) are transport-dominated. A 30 % slower descent is 0.05–0.07 σ
  (measured, §2). Do not spend rig time on the proprio-velocity hypothesis; the intent channel
  (§3.2) is the testable one. If you want the model to *see* approach speed, per-channel
  normalisation on a log/asinh scale or a phase-conditional stat is the tool — not a priority.

### 3.10 LOW — Raw wrist-3 angle sits 5° from the ±π wrap; the gate rejects what a one-line unwrap would rescue
* **Where**: `configs/start_poses.yaml:50,62` (q5 ≈ −3.11…−3.15 rad), norm stats q5 mean −3.056 σ
  0.129; `phantom/deploy/planner.py:151-153` feeds raw `q`.
* **Evidence** [certain]: a +183° reading is the same TCP pose as −177° and is 60 σ away in the
  feature. The 08-28 gate now refuses such starts (16/26 episodes lost that day, rig doc §Safety).
* **Fix**: in `SnapshotBuilder.build`, `q[i] += 2π·round((q_mean[i]-q[i])/2π)` toward the task's
  `q_mean` from `start_poses.yaml` (identical physics for wrist 3; for other joints it changes
  nothing within the demo branch). Apply the same in `windows.py:266` for symmetry (no-op on the
  demos). Keep the gate for the flipped-elbow branch, which is a different configuration.

### 3.11 Hygiene (not ranked)
* `tools/terminal_eval.py:66,118`: `--max-episodes` defaults to 80 and slices a task-sorted,
  manifest-ordered index; with 124 val episodes the default silently drops the later tasks /
  the new-batch holdout. Default to `None` (all) and print per-task n.
* `phantom/data/windows.py:347-356`: gate-label probes use `nearest_idx`, so the first probe
  (`t0 + 0.075 s`) can resolve to a frame *before* t0; use `future_idx` for probes.
* `tools/intake_recovery.py:101-123`: failure demos from a held-out session stay in `train` — same
  day/lighting/placement as `val` for the contact heads. Minor leak; document it in
  `intake_holdout.json`.
* `phantom/dagger/manifest.py` writes `{"round","entries":[{"episode","weight","source"}]}` while
  training reads `manifests/all.jsonl` rows with `split`; `weight` is consumed nowhere. Unify before
  the DAgger phase or it will be a silent no-op.
* `phantom/scripts/dump_norm_stats.py:49`: stats over all episodes incl. val/fail (harmless).
* `episode_qc.py` prints `start_sigma_xyz` against `start_poses.yaml` but nothing about object
  placement; add close-xy sigma (see §3.3).
* `WindowSampler.valid_range` excludes the first 1.6 s and the last 3.1 s of every demo from
  anchoring — fine, but the deploy's first replan uses a normalised zero `prev_chunk`, i.e. "the arm
  was still for 1.6 s", which the training distribution contains only at `t0 = lo` on demos whose
  operator waited. Probably fine; worth one look at the first-replan traces.

---

## 4. Missing experiments, cheapest first (no rig time unless stated)

| # | experiment | cost | what it decides |
|---|---|---|---|
| a | `terminal_eval` with all observations nulled (`_null_obs_batch` + `null_video_cond`) | 20 min GPU | whether 17 mm endpoint error is perception or prior |
| b | `terminal_eval` with `prev_chunk` := model's own sample from `t0-0.9` (2-step autoregressive intent) | 40 min | whether the intent loop reproduces under-commit / slow mode offline |
| c | `terminal_eval` with `prev_chunk[:, :6]`, `qd`, `tcp_speed` scaled ×0.7 | 20 min | whether "slow arm" propagates (expected: no, per §2) |
| d | `terminal_eval` scoring the GT chunk on `(t0+0.9, t0+2.5]` while conditioning at `t0` | 20 min | the latency mismatch's size in mm |
| e | per-seed terminal eval on 8 seeds at the same window, report the bimodality of `commit_ratio` | 1 h | confirms the seed-selects-mode story; motivates ensembling/persistent-noise re-seeding on stall |
| f | centre-cell-only A/B, 10+10 episodes, waffles | 2 rig-hours | separates placement OOD from commit failure |
| g | 20–30 recovery demos per task from the closed-on-air state | 3 rig-hours | the data fix for §3.1; the paper's "tactile closes the loop" claim becomes trainable |
| h | fine-tune with §3.2 fix + §3.6 lag augmentation + §3.5 motion-weighted anchors | 1 H100-day | the offline variants (b, d) should move first; then rig |

---

## 5. Appendix — measurements (compute3, read-only)

Val-subset profile (`data/val_eval/tasks`, n = 20/17/20/21 + 16 waffles from `incoming_recovery`):

```
actions stream: dt median 0.1003 s, p90 0.1034 s, per-episode max (median over eps) 0.113 s
|cumsum(16 recorded deltas) - measured TCP displacement|: median 1.07 mm, p90 2.40 mm
commanded close leads measured close by 0.10-0.18 s (all tasks)
close_index found a close in 98/98 episodes

task        n  close xy σ (mm)  z̄ (mm)  band descent (mm/s) hover%  dwell% (all)  OBJ@+1s         cmd max  dur (s)  tc (s)
Carton     20     16.7 / 10.9    104     24.7 (p10 3.0)     12.5    17.2          3:17 2:2 0:1     0.61     18.5     5.4
egg        17      3.6 /  6.0     72      2.9 (p10 0.6)     72.1    26.7          3:11 0:4 2:2     0.55     24.8     8.0
waffles    36     21.4 / 14.6     79     20.4 (p10 9.9)     23.7    16.2          3:24 2:7 0:5     0.61     20.4     6.7
whiteboard 21      5.9 /  6.8    156     21.5 (p10 9.6)     10.2    13.3          2:21             0.86     22.3     5.5

Δσ in normalised tcp_speed for a 30 % slower band velocity: 0.015-0.067 (max over xyz, per-task mean)
```

Manifest on compute3 (`val_eval/manifests/all.jsonl`, v4-era): per task 160 train / 17–21 val /
20 `_fail` train (whiteboard 10) — failure demos never in val [certain].

Unit tests: `tests/test_windows.py tests/test_v5_intake.py tests/test_terminal_eval_close.py
tests/test_v5_holdout.py` → 16 passed.
