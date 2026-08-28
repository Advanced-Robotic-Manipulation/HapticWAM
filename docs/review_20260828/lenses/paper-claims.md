# PHANTOM review — lens: PAPER CLAIMS + EXPERIMENTAL DESIGN

Repo `/Users/sannikov/GitHub/phantom` @ `3539988`, reviewed 2026-08-28. Deadline ~Sep 15 (≈3 weeks), one UR3 rig,
~15–20 rig-hours, rented H100s. Everything below was read from code (file:line) or checked with
`.venv/bin/python`; nothing is inferred from names. 39/39 cosmos-free unit tests pass on the Mac
(`tests/test_layout.py tests/test_packing.py tests/test_windows.py`).

---

## 0. Executive verdict

1. **There is no paper until the teacher grasps.** 0/26 rig successes (08-28) and 13/13 misses (08-20). Every
   claim in `pipeline.md §8` is a ratio whose numerator or denominator is the teacher's success rate.
2. **The tactile premise itself is untested.** Nothing in the repo measures whether the action head *uses* touch.
   On the rig the ACC gate correctly said "nothing there" (p_none 0.99) while the policy closed on air and
   lifted — direct evidence that the contact heads do not steer the chunk. A seconds-per-window offline
   counterfactual (tactile real vs nulled) on `v5_6` would settle this today and must run before any rig hour is
   spent on baselines (§4.1).
3. **The student pipeline (RQ3 headline) is currently broken in a silent way**: `distill_hid`, `dagger/relabel`
   and `finetune_hids` build models with the default `PhantomModelConfig` (RoPE `aligned`, ACC `gt_noised`) while
   the checkpoints were trained `time_true` + `two_pass`; the loader only checks hardware shapes (§3.1).
4. **The critical control (`no_distill`) and the lower bound (`vision_only`) cannot be built** with the current
   CLI/architecture: `train_teacher` hardcodes `student=False`, and wrist F/T is welded into every layout and
   every deploy mode (§3.2).
5. **The "world-action model" label is not earned by the video head as built**: the action frames never attend
   generated video (structural mask), the video loss weight is 0.1, and — verified — the mask leaks video
   information into actions via OBS/VIDEO_COND queries, so "droppable without train/test mismatch" is false.
   Without a λ_v=0 ablation the video head is decoration (§2.2, §3.4).
6. **The evaluation protocol as coded cannot support the claims** (system-major trial order, operator y/n
   labels, "seeds" that are sampling seeds, a 3000-trial plan against ~250 available valid episodes) (§5).
7. A credible 3-week paper exists only as a **narrowed** one: 2 tasks, 3 arms (teacher / HID student /
   no-distill control), 25 interleaved trials per arm per task, objective success labels, plus the offline
   tactile-sensitivity study and one video-head ablation. Drop occlusion, the 5-task suite, RQ2 lead-time
   as a "money plot", 2×DAgger, HID-S, and contact-play pretraining from the claims (§6).

---

## 1. Claims inventory — what the paper intends to say vs what exists

Sources: `pipeline.md` (§3–§8), `README.md`, `docs/STATUS.md`, `docs/training_playbook.md`,
`docs/rig_session_v5.md`, `configs/eval_campaign.example.yaml`, `scratchpad/research/CONTEXT.md`.

| # | Claim (as written) | Evidence in repo today | What is missing | Hostile-reviewer attack |
|---|---|---|---|---|
| C1 | **Tactile WAM teacher**: Cosmos-Predict2.5-2B LoRA jointly denoises future video + contact package + action chunk; tactile via HHT; "full native SDK surface" consumed | Code complete (`model/rf.py`, `model/sequence.py`, `hht/`); v4 (20k) and v5 (3k FT) checkpoints; offline terminal metric 17.4 mm | **Any rig success.** Any test that the action head is sensitive to tactile input. λ_v ablation. | "Your teacher fails 100 % on pick-and-place of a waffle pack on a rigid table; what does touch add? Show a tactile-nulled counterfactual." |
| C2 | **"World-action model"** (video generation is guidance, droppable at inference, Fast-WAM-style mask) | `sequence.py:190-202`, `model.py:32` (λ_v=0.1), `rf.py:373-376` drop-video layout | Mask is one-hop only (verified §2.2) → video is NOT cleanly droppable; no ablation of video on/off in training; actions never read the imagined future | "This is a policy with an auxiliary video loss, not a world model — the policy cannot query its own imagination. Where is the λ_v=0 row?" |
| C3 | **Anticipatory contact (ACC > CASA), gate lead-time "money plot", ms transients** | `acc.py`, `metrics.py:75-106`, gate label `windows.py:349-360` | CASA (α=0) variant does not exist (`acc.py:95-102` learns α; no fixed-α knob); lead-time metric is uncapped and replan-quantised (§3.6); contact grid is **1.0 s** (`windows.py:136-137`), gate lookahead 0.5 s < replan period 0.9–1.6 s | "Your anticipation horizon is shorter than your replan period; your contact package lives on a 1 Hz grid; the 'lead time' is the label definition (contact-within-Δ) plus replan phase noise." |
| C4 | **HID: sensor-free student recovers most of the teacher–vision-only gap; no-distill control isolates distillation from wrist F/T** | `distill_hid.py` (losses implemented), `smoke_test` HID step | Zero student experiments; student built with wrong model config (§3.1); `no_distill` cannot be trained (§3.2); `vision_only` ≡ `no_distill` (wrist F/T in every layout, §3.2); DAgger rounds infeasible (400 rig eps) and relabels are dead code (§3.7) | "Without the control, everything you attribute to distillation is wrist F/T. Without a true vision-only arm the recovery ratio is undefined." |
| C5 | **HHT > tactile-as-image** (RQ1), contact-play SSL pretraining | `pretrain_tactile.py` exists; `runs/tactile_pretrain/` is **synthetic** only; v4/v5 launch lines (`tools/provision_v4.sh:157-160`, `provision_v5.sh:268-275`) have no `--tactile-pretrain` | No VAE-only-tactile variant (no config knob to drop OBS_MECH); no real contact-play data | "README says contact-play pretrained; your checkpoints were not." |
| C6 | **σ speed governor slows where hallucination is uncertain** | `governor.py`, `executor.py:182-183`, config `hardware.nuc.yaml:264-267` (min_scale 0.15) | No on/off ablation; no `--no-governor`; scale not in the trace; plausibly contributes to the 30 % slow descent | "The 'safety' governor is a confound in every rig number, including your executor-exoneration analysis." |
| C7 | **5-task suite, 2 under occlusion, fragility instrumented, ≥3 seeds × 20 trials/cell, 2 100 + 960 trials** | `eval_campaign.example.yaml` names tasks that do not exist (`fragile_grasp, slippery_place, insertion, wipe, regrasp`); data has 4 tasks (waffles, Carton, egg, whiteboard); no occlusion data; one checkpoint per system | Everything | "You promise 3 000 trials and deliver <200 from one checkpoint per arm, with hand-labelled success." |
| C8 | **HID-S force-safety fine-tune** | `finetune_hids.py` (never run on real data; same mc bug) | Cut. | — |
| C9 | **Language conditioning** | `inference.md`: "the prompt IS the task name" (task token, cached Reason1 embedding) | — | Do not call it language conditioning. |

---

## 2. Is this a WAM in the accepted sense? Is the video head defensible?

### 2.1 What the architecture actually does (from code)

* Sequence (teacher, real config, verified by `SequenceLayout.build`): T=14, 4480 tokens —
  `video_cond[0:1) | video_gen[1:4) | obs_gel[4:5) | obs_mech[5:6) | obs_proprio[6:7) | contact[7:10) | action[10:14)`;
  RoPE `time_true` positions `[0,1,2,3, 0,0,0, 1,2,3, 0.4,0.8,1.2,1.6]`.
* Conditioning is **one** current RGB frame tiled (`inference/policy.py:85-86`); no history except the 0.25 s wrist
  window and `prev_chunk`.
* **Actions never attend VIDEO_GEN keys** (`sequence.py:199-201`). The contact package (CONTACT) *is* visible to
  ACTION queries — so the *tactile* part of "world-action" (predict future contact, act jointly) is real in the
  joint-denoising sense. The *video* part is not: the policy cannot read its own imagined future.
* The intent chunk (`prev_chunk`) is injected only into the VIDEO_GEN frames' AdaLN (`phantom_dit.py:154-156`)
  and into the ACC gate (`acc.py:84`). It reaches ACTION only indirectly.
* Video gradients shape the shared LoRA (VIDEO_GEN queries attend ACTION/CONTACT keys), i.e. the video loss is an
  **auxiliary "actions must be predictive of the future frame" regulariser** at weight 0.1. That is a legitimate
  multi-task objective (UVA/Cosmos-Policy-style) but it is not a mechanism the policy uses at decision time.

### 2.2 The droppability claim is false as implemented (verified)

`python` check on the real config:

```
video_cond   queries -> VIDEO_GEN keys masked? False
obs_proprio  queries -> VIDEO_GEN keys masked? False
obs_gel      queries -> VIDEO_GEN keys masked? False
obs_mech     queries -> VIDEO_GEN keys masked? False
contact      queries -> VIDEO_GEN keys masked? True
action       queries -> VIDEO_GEN keys masked? True
```

`structural_attn_bias` (`sequence.py:195-202`) masks only CONTACT/ACTION rows. OBS_* and VIDEO_COND tokens read
VIDEO_GEN tokens in block *k*, and ACTION reads those OBS/VIDEO_COND tokens in block *k+1*. Hence (a) generated video
**does** influence actions through a two-hop path, and (b) dropping video at inference (`--drop-video`,
`rf.py:373-376`) changes what ACTION sees relative to training — exactly the "train/test attention mismatch" the
docstring says it avoids. The rig currently runs with video ON (run_deploy default), so no mismatch is paid, but
~21 % of the sequence (960/4480 tokens) is spent generating frames nobody consumes directly.

### 2.3 Verdict

* Call it a **contact-package world-action model** if — and only if — the tactile-counterfactual test (§4.1)
  shows the action chunk responds to touch/contact prediction. Otherwise it is a LoRA-finetuned video DiT used as a
  flow-matching chunk policy with auxiliary losses, and the honest title says so.
* The video head is defensible **only with an ablation**: same recipe with the VIDEO_GEN group removed from the
  layout during training (`drop_video=True` layout; `training_step` already tolerates a missing VIDEO_GEN,
  `rf.py:329-331`; `build_model` needs a `drop_video` passthrough — `builder.py:39-49` has none), compared on
  terminal_eval **and** on the rig. If λ_v=0 is no worse, the WAM framing goes; if it is better, the paper has a
  real (and cheap) finding. Either way, fix the mask (`VIDEO_GEN` keys visible only to `VIDEO_GEN` queries) before
  claiming droppability, or stop claiming it.
* Reviewers from the WAM community (UVA, Cosmos-Policy, WorldVLA, Dream-Tac, VT-WAM) will ask the λ_v=0 question in
  the first paragraph of the review.

---

## 3. Code-level findings that change paper outcomes (file:line)

### 3.1 HIGH — Student/DAgger/HID-S build models with the wrong model config (silent geometry mismatch)

* `phantom/train/distill_hid.py:190-193`: `teacher = build_model(hw, paths, student=False, ...)` and
  `student = build_model(hw, paths, student=True, ...)` — **no `mc`** → `builder.py:46-47` uses
  `PhantomModelConfig(student=...)` defaults: `rope_time_mode="aligned"` (`model.py:59`), `acc.self_anticipation="gt_noised"`
  (`model.py:24`), `cond_dropout_p=0`, `action_t_max_of_two=False`.
* v4/v5 were trained with `--acc-two-pass` and the v4 default `--rope-time-mode time_true`
  (`train_teacher.py:150-155`, `provision_v4.sh:157-161`, `provision_v5.sh:268-275`).
* `common.py:235-280 load_phantom_checkpoint` asserts hardware shapes only; the model-config snapshot is never
  compared. `run_deploy.build_policy` (`run_deploy.py:36-52`) and `tools/terminal_eval.py:87-90` were fixed in the v4
  audit to rebuild from `payload["configs"]["model"]`; the three training programs were not.
* Same defect: `phantom/dagger/relabel.py:59-61`, `phantom/train/finetune_hids.py:143-146`.
* Consequence: the HID teacher would produce its imagined packages and velocities under `aligned` RoPE
  (`[1,2,3,3]` action positions instead of `[0.4,0.8,1.2,1.6]`), the student is initialised from `time_true` weights
  but trained under `aligned`, and the ACC two-pass self-anticipation is replaced by leaked-GT `gt_noised`. The
  whole RQ3 experiment would be measuring a mis-built model.
* Fix (10 lines): in each program, `payload = torch.load(ckpt); mc_t = PhantomModelConfig.from_dict(payload["configs"]["model"])`;
  build teacher with `mc=mc_t`, student with `dataclasses.replace(mc_t, student=True)`; add to `load_phantom_checkpoint`
  an assertion that `model.mc.to_dict()` equals the saved config modulo `student`. Add a tiny test that a checkpoint
  saved with `time_true` refuses to load into an `aligned` model.

### 3.2 HIGH — The two baselines the headline ratio needs cannot be produced

* `train_teacher.py:189` `PhantomModelConfig(student=False, …)`, `:194` `build_model(..., student=False)`,
  `:258` `WindowSampler(..., student=False)` — **no `--student` flag**, so the "no-distillation control" (playbook
  §Comparative systems: "run `train_teacher` with `student=True` build — critical control") is not runnable.
* `hht.py:44-49, 84-91`: `OBS_PROPRIO = proj([WristTCN(wrist) ‖ URStateMLP(ur_state)])` in **both** teacher and
  student; there is no layout or config without wrist F/T. `planner.py:130-147` builds the wrist window in every
  mode; `TACTILE_INPUT_MODES = ("teacher",)` (`planner.py:28`) only gates tactile. Therefore `vision_only`,
  `no_distill` and `drop_tactile` are the **same model** under three names, and `aggregate.py:72-79`'s recovery
  ratio `(student − vision_only)/(teacher − vision_only)` has no valid denominator.
* Fix: (i) `--student` in `train_teacher` (build with `student=True`, sampler `student=True`); (ii) a
  `PhantomModelConfig.mask_wrist: bool` honoured in `HHT.obs_frames`/`wrist_feature` (zero the window) and in
  `SnapshotBuilder.build` for `vision_only`/`drop_tactile`; (iii) start the `no_distill` from-scratch run **now** —
  it does not depend on the teacher fix, only on the final data/recipe (≈35 h on one H100 ≈ $90). If compute is
  short, drop `vision_only` and rewrite RQ3 as "distilled vs non-distilled student, both vision+proprio+F/T".

### 3.3 HIGH — No evidence the teacher's actions depend on touch (the premise of C1/C4)

* ACC's only effect on the trunk is a key-side attention bias toward haptic tokens
  (`attention_bias.py:196-203`, `phantom_dit.py:249-252`, per-block λ init 0). Nothing conditions the gripper
  close/commit on the contact heads; `event_ce`, `contact_nll`, `gate_bce` are auxiliary readouts.
* Rig 08-28: gate/p_evt said `none` (0.99) while the chunk closed and lifted (CONTEXT, `rig_session_v5.md`).
* `evaluate_sampled` (`common.py:614-676`) and `tools/terminal_eval.py` never vary the tactile input.
* Fix — the cheapest, highest-value experiment in this review (§4.1): on the 124 terminal_eval windows, sample
  with (a) real inputs, (b) tactile nulled (gel/fields/contact_state zeroed — the model saw exactly this null under
  `cond_dropout_p=0.1`, `rf.py:229-233`), (c) wrist nulled, (d) prev_cpk nulled; report Δ endpoint, Δ close-step,
  Δ aperture. If (b)≈(a), the tactile-teacher story is dead and the paper must pivot to "what makes a chunk policy
  commit" (which is also what the rig needs).

### 3.4 HIGH — Video head: not attended by actions, not droppable, not ablated (§2.2)

Fix list: `build_model(drop_video=...)` passthrough; one training run without VIDEO_GEN; extend
`structural_attn_bias` so only VIDEO_GEN queries see VIDEO_GEN keys (retrain required for droppability);
inference-time video on/off row only after the mask fix.

### 3.5 HIGH — Evaluation protocol as coded

* `trial_runner.py:67-79`: loop order `system → task → occlusion → seed → trial` — every teacher trial before any
  student trial; no interleaving, no placement schedule. Drift in lighting/object wear/operator fatigue is
  confounded with the system.
* `trial_runner.py:92-93`: success/damage are operator y/n; `run_deploy.py:467-490` prompt is s/f/c with **no
  damage field** even though egg fragility is the one touch-relevant outcome.
* `aggregate.py:32-56` bootstraps over `seed` groups; with one checkpoint per arm this is a per-trial bootstrap
  labelled "over seeds". Report it honestly (Wilson CI) and say "single checkpoint per arm".
* Plan says ≥3 training seeds — impossible (34 h per from-scratch teacher).
* `configs/eval_campaign.example.yaml` tasks do not exist in the data.
* Fix: §5.

### 3.6 MEDIUM — ACC lead-time / anticipation metric (RQ2)

* Gate label = contact within `event_lookahead_s = 0.5 s` (`hardware.nuc.yaml:146`, `windows.py:349-360`); gate is
  computed once per replan (0.9–1.6 s); contact package on a 1.0 s grid (`windows.py:136-137`: `temporal_comp/fps`
  = 4/4). "Transients lasting tens of ms" (`pipeline.md §3`) are two orders of magnitude below the model's temporal
  resolution.
* `metrics.py:100-106`: `before = crossings[crossings <= t_on]; lead = t_on − before[-1]` — no cap; a gate that rose
  once early and stayed up scores an arbitrarily large "lead". `event_f1` (`metrics.py:109-137`) compares the
  next-step prediction to the label **at** the replan time (off by one step).
* No CASA variant: `acc.py:95-102` learns α; no `alpha_fixed`. RQ2's comparison cannot be run.
* Fix: drop RQ2 as a headline; if kept, report gate AUROC / PR at replan times against a proprio-only logistic
  baseline (height, speed, gripper), cap the crossing window at 2×lookahead, add `AccConfig.alpha_fixed`.

### 3.7 MEDIUM — DAgger scaffold is inconsistent and can train the student on its own failures

* `dagger/relabel.py` writes `<ep>/relabels/teacher.pt`; **nothing reads it** (`grep relabel phantom/train` → only
  the `distill_hid.py:17` docstring; `precompute_relabels` / `teacher_mode="cached"` do not exist).
* `distill_step` runs the teacher online on every window (`distill_hid.py:71-72`), which is a valid privileged-
  teacher DAgger — fine. But rollouts merged via `--extra-data` (`distill_hid.py:208-209`) become ordinary windows:
  `action_chunk` = the student's own executed actions, and the grounding term `w_ground·action_v_mse`
  (`distill_hid.py:123-125, 147`) imitates them unless `is_failure_demo(meta)` fires (`schema.py:155-171`:
  `success is False` / tag / `_fail`). A rollout left `success=None` (no label, or `c`) is imitated at full weight.
* Fix: `action_weight = 0` for any window whose episode is rollout-sourced (manifest `source` field,
  `dagger/manifest.py:19-21`), or use the teacher relabel as the grounding target; delete or wire `relabel.py`.
  Budget: one round, ~30 rollouts on one task, reusing the student's eval episodes (they already record full
  sensors, `planner.py:4-9`).

### 3.8 MEDIUM — Speed governor is an unvalidated design element in every rig number

* `governor.py:15-25`: scale ∈ [0.15, 1] driven by **max** σ over the chunk (`scale_profile`); `executor.py:182-183`
  advances `play_time` by `dt·scale`; config `sigma_lo 0.1, sigma_hi 1.0, min_scale 0.15` (`hardware.nuc.yaml:264-267`).
* No `--no-governor` flag (`run_deploy.py:161-237`); the governor scale is not in `planner_trace.json`
  (`planner.py:280-291`), so `tools/rig_trace_decompose.py` cannot separate "policy decelerates" from "governor
  slowed playback" — the "executor exonerated" conclusion compares the governed prefix.
* Fix: log `scale` per replan; run one A/B arm with `safety.governor.min_scale: 1.0` (identity; value-only config
  drift only warns, `common.py:253-255`); ablate on/off in the paper or remove.

### 3.9 MEDIUM — Claims with no artifact behind them

* Contact-play SSL pretraining of the tactile encoder (README, pipeline §6c, playbook (1)): never run on real
  data; `runs/tactile_pretrain/tactile_encoder_pretrain.pt` is the synthetic smoke output; no `--tactile-pretrain`
  in any launch line.
* 5 tasks + occlusion + fragility instrumentation + failure demos "per fragile task": actual = 4 grasp-and-place
  tasks, 45 `_fail` under-grasp demos, no occlusion, no force film/eggshell scoring.
* "Language instruction": a task token through a cached embedding (`inference.md`).
* Fix: rewrite to the actual pipeline before a reviewer diffs README against the method section.

### 3.10 LOW — Reporting hygiene

* `tools/terminal_eval.py` numbers (17.4 mm) are open-loop model-selection metrics on demo states; they sit next to
  0/26 rig in the docs. Never present them as results; at most a figure showing offline ≠ closed-loop.
* v4 val split is per-episode (`manifests/all.jsonl`), the v5 holdout is per-session (`intake_holdout.json`,
  `a8ff9cf`). Report only session-level holdout numbers.
* `prev_chunk` train/test mismatch: training uses recorded previous actions (`windows.py:278-279`), deployment
  uses the previous *proposal* (`planner.py:300-312`, marked DEFERRED). State it or fix it (executor already records
  executed actions, `executor.py:206-212`).
* Student trained without conditioning dropout (`distill_step` never nulls) → `--guidance > 1` invalid for the
  student.
* Behaviour matching at shared (x_t, t) against a tactile-conditioned teacher regresses the student to the
  conditional mean over unobserved tactile states — the standard privileged-distillation caveat; say it.

---

## 4. Experiments that are missing, ordered by value per hour

### 4.1 Offline tactile-counterfactual sensitivity (0 rig hours, ~1 GPU-hour) — RUN FIRST
Extend `tools/terminal_eval.py` with `--null {tactile,wrist,prev_cpk,all}` using the `_null_obs_batch` subsets.
Output per condition: endpoint_err, z_end_err, close_step_err, commit_ratio, gripper aperture, plus the
per-window action Δ vs the real-input sample. Also run it on the 45 `_fail` demos' windows (contact heads saw
them, action head did not): does the predicted chunk differ between "closed on object" and "closed on air" states?
This is the paper's premise test and the rig-fix diagnostic in one.

### 4.2 Teacher that grasps (rig-fix lens; listed here because every claim depends on it)
Candidates already visible from this lens: velocity channels in `ur_state` (qd, tcp_speed; `windows.py:266-270`)
create a self-consistency loop (slow arm → slow chunk); the governor (§3.8); `prev_chunk` mismatch (§3.10).
Whatever the fix, it must be one recipe applied identically to teacher, student and control.

### 4.3 `no_distill` control (from scratch, student layout) — start now, ≈35 h H100
Same data/recipe as the teacher. Needed for any distillation claim.

### 4.4 HID student — after 3.1 is fixed, ≈5k steps initialised from the teacher
Each step = 2–4 teacher forwards + student fwd/bwd ≈ 3× a teacher step (~20 s/step at batch 4×2) → 5k steps ≈ 28 h.
Do **not** run the 30k default. Keep `w_ground` on demos only (§3.7).

### 4.5 Video-head ablation — one run without VIDEO_GEN (≈35 h from scratch, or a 3k fine-tune as a cheap proxy)
Compare terminal_eval + 15 rig episodes. This is the difference between "WAM" and "auxiliary loss" in the paper.

### 4.6 Governor on/off — 15 rig episodes each, teacher only.

### 4.7 DAgger-lite — 1 round on 1 task, reusing the student's eval rollouts (~30), 1k HID steps, 25 re-eval eps.

Explicitly **not** feasible: 3 training seeds, 2 DAgger rounds × 50/task, occlusion variants, 5 tasks, HID-S,
contact-play pretraining, CASA and tactile-as-image variants (each is a from-scratch run plus rig time).

---

## 5. Statistical protocol that survives review (pre-register it in the repo)

* **Arms**: teacher (v_fix), HID student, no-distill control. Optional 4th: teacher without governor.
* **Tasks**: waffles (best-behaved, most demos) + one touch-relevant task: egg (fragility → damage rate) or
  whiteboard (wipe force regulation). Carton only if hours remain.
* **n**: 25 trials per arm per task. Power (two-proportion, α=0.05, 80 %): 0.3 vs 0.7 needs ~23/arm; 0.5 vs 0.8
  needs ~38/arm. Fisher exact at 25/arm: 18/25 vs 10/25 → p=0.045; 20/25 vs 10/25 → p=0.009; 15/25 vs 10/25 →
  p=0.26 (not significant). Wilson 95 % CI half-width at n=25, p=0.5 is ±0.18. Budget ≈ 150–175 valid episodes ≈
  9–11 rig-hours at ~3.5 min/episode incl. resets, leaving 5–8 h for the fix loop and ablations.
* **Design**: interleave arms per placement (A,B,C on the same taped-grid cell + orientation, schedule drawn
  before the session); both policies loaded simultaneously (single-digit GiB each per `inference.md`) so
  interleaving costs no reload; the same `--seed` list across arms.
* **Success**: objective, computed from recorded streams — Robotiq gOBJ==2 at close ∧ tactile contact mask >
  τ for ≥1 s ∧ TCP z rises ≥5 cm after close ∧ (for place tasks) TCP enters the bin envelope; operator label
  only as an override with a note. `rig_trace_decompose.py` already reads OBJ states. Add `[d]amage` to the
  `run_deploy` prompt.
* **Report**: success % with Wilson CI, Fisher exact between arms, miss distance / z-at-close distributions from
  `planner_trace.json`, damage % on egg, peak normal force from tactile (already in `metrics.py:18-28`). State:
  one checkpoint per arm, sampling seeds only, no occlusion, 4-task dataset with 2 evaluated.
* **Retention/recovery ratios**: only if a true vision-only arm exists; otherwise report the three arms' rates and
  the paired per-placement differences.

---

## 6. Minimum credible paper in 3 weeks (calendar)

| Day | Rig | GPU | Paper |
|---|---|---|---|
| 1–2 | — | 4.1 counterfactual on v5_6; launch `no_distill` from scratch (needs 3.2 (i)); fix 3.1 | Decide framing from 4.1 |
| 2–6 | Fix loop for the teacher (≤6 rig-h): velocity-masked / governor-off / prev_chunk-executed variants, 5–8 eps each | 3k-step fine-tunes per variant (~5 h each) | Method section to the real pipeline (§3.9) |
| 6–7 | Freeze teacher recipe | Retrain/refine teacher; start HID student (4.4); start no-VIDEO_GEN ablation (4.5) | |
| 8–12 | Main matrix: 2 tasks × 3 arms × 25 (≈9 h) | DAgger-lite round from the student's eval rollouts | Results tables from the ledger |
| 13–14 | Ablations: governor off, video off (15 eps each, teacher) ; DAgger-lite re-eval (25) | | |
| 15–20 | Buffer / re-runs | | Writing, figures, limitations |

If day 6 arrives without a teacher that grasps ≥50 % on waffles, the paper becomes a negative-result /
analysis paper about commitment failure in flow-matching chunk policies with privileged contact heads — which
the counterfactual study (4.1) and the trace decomposition already support — or it slips to the next deadline.
