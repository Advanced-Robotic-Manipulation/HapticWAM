# PHANTOM review — lens: SELF-IMPROVEMENT / DAGGER READINESS

Repo `/Users/sannikov/GitHub/phantom` @ `3539988` (2026-08-28). Read: `phantom/dagger/*`, `phantom/train/{dagger_driver,distill_hid,finetune_hids,train_teacher,common}.py`, `phantom/config/training.py`, `phantom/recording/recorder.py`, `phantom/data/{windows,schema,episode_store,derived}.py`, `phantom/deploy/{runtime,executor,planner,start_pose,safety}.py`, `phantom/scripts/{run_deploy,run_dagger_round}.py`, `phantom/inference/policy.py`, `phantom/model/rf.py` (sample/training_step), `tools/{intake_recovery,rig_trace_decompose,episode_qc}.py`, docs (rig_session_v5, training_playbook, recovery_demos_protocol, launch_guide).

Checks run: (1) `relabel_root` + `write_manifest` smoke on a tiny synthetic set (works: 4 eps, 3 records/ep, 5 s); (2) `run_dagger_round --tiny` (crashes, see F5); (3) demo grasp/lift statistics over 1115 episodes from the scratchpad `lift_*.jsonl` (derived from the repo's own dataset by the earlier `lift.py` pass; used for the success-rule thresholds below).

Side effect to disclose: check (1) generated a new gitignored synthetic root `data/synthetic/43cec1ae04/` (4 episodes) and wrote `relabels/teacher.pt` into each. `git status` is clean. I did not delete them (policy); `rm -rf data/synthetic/43cec1ae04` is safe.

---

## 0. TL;DR

The repo has *scaffolding* for **student** DAgger (rollout → teacher relabel → distill) and a **student-only, force-penalty** AWR fine-tune. Neither is what the rig problem needs (the *teacher* stops 65–120 mm high and closes on air), and both are broken or dead in ways that would silently corrupt a self-improvement round:

1. **Rollout episodes record the policy's raw chunk proposals as `actions`, not executed motion** (demos record measured TCP deltas). Filtered BC on "successful rollouts" would imitate proposals, on governor-warped timestamps, before clamp/rate-limit. Must be fixed *before* any rollout enters training (F1).
2. **A rollout with no operator verdict (`success=None`) or tagged `contaminated` trains at full action weight** — nothing in the training path excludes either (F2).
3. **Robotiq OBJ is not a grasp-success signal on this rig**: only 5 % (Carton) / 2 % (egg) / 23 % (waffles) of *successful* demos ever show OBJ=2; the over-squeeze failure demos show it 95–100 %. An OBJ-based auto-label (as the literature note proposes) would label ~80–98 % of real successes as failures on 3 of 4 tasks (F3). The usable label is tactile contact-during-lift (rule in §4).
4. **No per-episode weight reaches any loss** — `dagger/manifest.py` weights are never read; `distill_hid --extra-data` just concatenates; `train_teacher` has no rollout path; `action_weight` is binary (F4). AWR exists only in `finetune_hids` (student, force reward, unit-mismatched, unmasked on failure demos — F8).
5. `run_dagger_round` crashes on launch and bypasses every rig safety layer and the label prompt (F5); the DAgger relabel pass writes files nothing reads (F6); `distill_hid`/`finetune_hids` train on the held-out `val` split (F7); rollouts recorded with the z-floor/hitbox on carry a different `config_hash` and hard-fail `train_teacher` (F9).

What *does* work and should be the backbone of a one-week loop: `run_deploy` (seeds, persistent noise, labels, safety, trace), the zarr episode schema (rollouts carry every stream the demos do), `is_failure_demo → action_weight 0` (failures supervise contact heads only), `intake_recovery.py` (normalize/place/manifest append), `train_teacher --init-weights` (the v5 recipe, ~6 h/3 k steps), `terminal_eval`, and the hub/`DEMO.pt` staging flow. §5 gives the concrete design (3 new files, ~1 day of code) and a 2-iteration rig plan with ~100 rollouts.

---

## 1. What exists today, per question

### (a) Run policy rollouts with noise
- `phantom/scripts/run_deploy.py` is the only viable collector. Stochasticity = the flow-matching *initial noise* only (`rf.py:sample`, `torch.randn(x0.shape, generator=self._gen)`), NFE 5. Controls: `--seed S` (per-episode `seed+i`, `run_deploy.py:445-449`), `--persistent-noise` (one draw held across replans, `rf.py` `reuse_noise`), `--guidance`. No action-noise injection, no noise temperature, no DART-style perturbation. Start-pose jitter ≤ 1σ of the demo start distribution (`deploy/start_pose.py:108-119`).
- Every rollout records the full demo schema (`recording/recorder.py:27-46`: tactile fields_ds/wrench/area/keyframes/infer_img, arm q/qd/tcp/speed/ft, gripper [pos, obj], scene camera) plus `planner_trace.json` with every sampled chunk, gate, p_evt, sigma, latency (`deploy/planner.py:262-272`) and condition tags in `meta.json` (`run_deploy.py:324-332`).
- `phantom/scripts/run_dagger_round.py` + `phantom/dagger/rollout.py`: student-mode only, crashes (F5), no safety envelope, no labels, `--max-replans 20`, raw (non-EMA) weights by default.
- The rig-observed bimodality across seeds ("one seed descended at full speed to the right height") means the *seed* is the exploration knob. A one-line `--noise-scale` (multiply the initial draw by 1.0–1.3, tag it) is the only cheap addition worth making.

### (b) Auto-label success — which streams carry what
| stream | rate | content | usable as label? |
|---|---|---|---|
| `gripper.zarr` (T,2) | 100 Hz | `[pos 0..1, obj]`; obj = Robotiq gOBJ raw: 0 moving, 1 stopped-while-opening, 2 stopped-while-closing (stall on object), 3 at requested position (`drivers/base.py:67-82`, `drivers/real/robotiq.py:90-94`) | **pos yes** (close via `common.close_index`, `common.py:300-321`); **obj NO** as success (§4, F3); obj==2 sustained = over-squeeze/stall flag |
| `tactile_{left,right}_fields_ds.zarr` (T,72,96,8) | ~8 Hz | `[disp_x, disp_y, depth, shear_x, shear_y, fx, fy, fz]`; `dist_force_unit_to_N: 0.0` ⇒ force channels uncalibrated, **depth is the contact channel** (`hardware.nuc.yaml:102,136-138`) | **yes** — `dv.derive_timestep` → `mask_frac` (fraction of cells with \|depth\| > 0.05); contact = `mask_frac > tau_contact_area 0.025`, *exactly* the gate-label definition used in training (`windows.py:331, 353-354`) |
| `tactile_*_wrench` (6), `tactile_*_area` (1) | 8 Hz | SDK wrench / contact area | secondary (area cross-checks mask_frac) |
| `arm_tcp_pose.zarr` (T,6) | 125 Hz | measured TCP | **yes** — z at close, lift height |
| `arm_ft.zarr` | 125 Hz | wrist F/T (biased at episode start) | weak (object masses are small) |
| `planner_trace.json` | per replan | gate, p_evt, sigma, chunk | model's own output — **never a label** (circular); fine as a diagnostic |
| `meta.success` | — | operator `s/f/c` prompt (`run_deploy.py:466-489`) | task-level truth; `c` → `success=None` + tag `contaminated` |

Nothing computes an automatic label today. `tools/episode_qc.py:39-43` computes `obj2_frac`/`closes`; `tools/rig_trace_decompose.py` prints z-at-close/plateau/OBJ per episode; the scratchpad `lift.py` computes lift/hold. None writes a label.

### (c) Fold rollouts into training with weights
- Teacher path (`train_teacher.py`): episodes come from `manifests/all.jsonl` rows with `path`/`split` (`common.manifest_split`, `common.py:323-370`); rows are appended by `tools/intake_recovery.py manifest` (symlinks under `tasks/<task>/`). **No per-row weight is read; `WindowSampler` emits only `action_weight ∈ {0,1}`** from `is_failure_demo` (`windows.py:403`; `schema.py:155-171`). Nothing keys on `meta.policy`/`dagger_round`/`contaminated`.
- Student path: `dagger_driver.py:50-52` writes `runs/dagger/round_k.json` with `weight` per entry — **decorative**: `read_manifest` has no caller (grep), and `distill_hid.py:206-208` merges rollouts by `ds.index += sampler_s.build_index(Path(extra))` (uniform). Loss terms on rollouts: `action_v_mse` (masked by `action_weight`), `behavior_match`/`traj_distill`/`event_distill` (unmasked — teacher-relabel semantics, fine).
- AWR: `HIDSConfig.beta_awr/max_weight` (`config/training.py:91-93`) are used only by `finetune_hids.hids_step` (`finetune_hids.py:90-98`): reward = −λ·max(0, peak_force − τ), advantage = r − running mean, w = exp(A/β) clipped at 20, applied to the ACTION RF loss of the **student** with a velocity-MSE leash. Not success-based, not for the teacher, and buggy (F8).

### (d) Retrain and stage
- Works: `train_teacher --init-weights <v5.pt> --data <root>/tasks --hardware configs/hardware.nuc.yaml --split train [--grasp-frac 0.3 --photo-aug 1.0 --event-band-weight 0 --acc-two-pass …]` (must match the checkpoint's model config, `train_teacher.py:205-214`; norm_stats must be byte-equal, `:237-253` ⇒ **never rerun `dump_norm_stats` after adding rollouts**), `tools/terminal_eval.py`, hub upload, `runs/…/DEMO.pt` symlink + `.stage_map` (docs/rig_session_v5.md:42-51).
- Blocker for rollouts: F9 (config-hash drift hard-fail) and F1 (what `actions` means).

---

## 2. Findings (ranked)

### F1 [high] Rollout `actions` stream = raw chunk proposals at governor-warped times, not executed motion
- `deploy/executor.py:205-212`: `record_action(t0, plan.actions[self._last_action_k])` — the *plan* row, written when playback *enters* grid step k under governed play time (`_play_time += dt * scale`, `:178-180`), **before** the workspace/z-floor clamp (`:218-219`) and the kinematic rate limit (`:236-249`), and ignoring chunk blending (`:190-199`).
- Demos: `data_collect/session.py:746-750` record `pose_delta(prev_tcp, tcp)` of consecutive **measured** TCP poses on the 10 Hz grid (same in `drivers/record_episodes.py:335`). The teleop comment even says this is "the ONLY action stream WindowSampler reads".
- `windows.py:276-277` samples `actions` by nearest/future timestamp on a uniform 0.1 s grid; with governor scale < 1 the same row is picked repeatedly and a 0.1 s delta is read as one step (velocity overstated up to 1/min_scale); with the z-floor active the recorded descent continues below where the arm stopped.
- Consequence for filtered BC: "successful rollout" windows would teach the model its own proposals (including the slow-descent mode), not the motion that succeeded; for DAgger relabel the `prev_chunk` fed to the teacher is built from these rows too.
- Fix (offline, no rig code): in the rollout intake pass (§5) **re-derive `actions.zarr` from `arm_tcp_pose` + `gripper` on a 10 Hz grid with `dv.pose_delta`**, identical to the teleop derivation; keep the original as `actions_plan.zarr`. Optionally also fix the executor to emit `pose_delta(prev_cmd, last_cmd)` + the gripper target actually handed to the worker, on the wall clock. Add a parity test: for a demo, re-deriving from `arm_tcp_pose` must reproduce `actions.zarr` within noise.

### F2 [high] Unlabeled and `contaminated` rollouts train at full action weight
- `schema.py:167-171`: `is_failure_demo` is True only for `success is False`, tag `deliberate_failure`, or task `*_fail`. `success=None` ⇒ False ⇒ `windows.py:403` `action_weight = 1.0`. `run_deploy.py:468-489`: Enter=skip leaves `success=None`; `c` sets `success=None` + tag `contaminated`. `episode_store.list_episodes` (`:188-209`) skips only non-finalized; `manifest_split` skips nothing by tag; grep: no consumer of `contaminated` in `phantom/` except `rig_trace_decompose`.
- Scenario: a session of 30 rollouts, operator skips the prompt on 6 (or the process dies before the prompt — `runtime.py:225` finalizes with `success=None`); `intake_recovery.py manifest` appends them as `split: train`; the fine-tune imitates 6 failed descents at weight 1.0 on-policy states — precisely the states where the model is already wrong. The collect app had this exact bug (`session.py:357-377`) and fixed it with `status='aborted' + tag 'unlabeled'`; deploy did not.
- Fix: in `run_episode`/`run_deploy`, when no verdict is given mark `status='aborted'` + tag `unlabeled` (reuse `recorder.relabel(status=…, tags=…)`, `recorder.py:207-241`); in the intake pass refuse `policy != ''` episodes with `success is None` or tag `contaminated`; make `manifest_split` skip rows whose tags contain `contaminated`/`unlabeled`.

### F3 [high] Robotiq OBJ=2 is not a grasp-success signal on this rig (and the literature note has the codes inverted)
- Evidence from 1115 episodes (scratchpad `lift_*.jsonl`, computed from the dataset by the earlier `lift.py`; `obj2_frac_closed` = fraction of the closed phase with obj==2):

| task (success demos, n=250) | obj2>0 | obj2>0.5 | lift mm p5/50/95 | z_close mm p5/50/95 | peak closure p5/50/95 |
|---|---|---|---|---|---|
| Carton | 0.05 | 0.03 | 201/225/270 | 97/118/129 | 0.41/0.48/0.58 |
| egg | 0.02 | 0.02 | 74/121/209 | 70/78/86 | 0.49/0.52/0.56 |
| waffles | 0.23 | 0.13 | 228/311/378 | 53/68/88 | 0.54/0.60/0.64 |
| whiteboard | 0.99 | 0.99 | 86/119/160 | 150/157/166 | 0.47/0.50/0.52 |
| `*_fail` over-squeeze (v4, success=True, n=20/task) | 0.95–1.00 | | | | |
| `*_fail` under-grasp (0822, success=False, n=15/task) | 0.00 | | 105–263 (lifted an empty gripper) | inside the success band | 0.42–0.51 |

  Only whiteboard (the rigid eraser) stalls the fingers; the soft/light objects are reached at the commanded aperture, so gOBJ reports 3 ("at requested position"). OBJ=2 flags *over-squeeze*, not grasp. Aperture does not discriminate either (under-grasp fails close to demo-like apertures at demo-like heights — by construction). `research/06_failure_demos_success_reward.md §3` states "OBJ=2 (stopped, no object) … OBJ=1 (stopped, object detected) is a near-ground-truth success label" — this is inverted vs `drivers/base.py:70-73` (Robotiq: 1 = contact while opening, 2 = contact while closing) and, more importantly, wrong for this rig per the table.
- Consequence: an OBJ-driven auto-label would mark ~95 % of Carton/egg and ~80 % of waffles successes as failures (weight 0) and the training signal collapses to whiteboard; the paper's "self-improvement from tactile reward" claim would rest on a mislabeled set.
- Fix: the label must come from **tactile contact sustained through the lift** (rule in §4), which is (i) the signal the paper is about and (ii) the very definition the ACC gate is trained on. Use OBJ only as a stall/over-squeeze flag. Validate the rule offline on compute3 before the rig: expect ≥ 95 % positive on the 1037 successes, ≈ 0 % on the 45 under-grasp fails and on the 39 rig rollouts (08-20 + 08-28).

### F4 [high] No per-episode weight reaches any training loss; the AWR config never touches the teacher
- `dagger/manifest.py:13-25` writes `weight` per entry; `read_manifest` (`:28-29`) has zero callers. `dagger_driver.py:55-64` passes `--data`/`--extra-data` and the manifest is not consulted. `distill_hid.py:206-208`: `ds.index += sampler_s.build_index(Path(extra))` — uniform. `train_teacher.py` has no rollout/extra-data path and `windows.py:403` emits a binary `action_weight`. `HIDSConfig.beta_awr/max_weight` (`training.py:91-93`) are consumed only by the student HID-S program.
- So the loop the memory file calls "≈80 % there" is, for the teacher: rollouts → (no weighting, no filtering beyond failure-masking) → `train_teacher`. That is plain filtered BC by *inclusion* only, with the F1/F2 problems on top.
- Fix (minimal, ~40 lines): add `weight: float = 1.0` to `EpisodeMeta` (`from_dict` is tolerant); `windows.py:403` → `w["action_weight"] = (0.0 if is_failure_demo(meta) else 1.0) * float(meta.weight)`; `group_velocity_mse` already normalizes by `w.sum()` (`losses.py:49-51`) so weights ≠ 1 are safe. Optional: a per-window terminal-phase multiplier for rollouts (see §5 formula). Keep `dagger/manifest.py` weights or delete the module — but do not leave a config that pretends to weight.

### F5 [medium] `run_dagger_round` crashes at launch and bypasses every rig safety layer and the label prompt
- Verified: `python -m phantom.scripts.run_dagger_round --round 1 --ckpt "" --tasks waffles --tiny --device cpu` → `AttributeError: 'Namespace' object has no attribute 'text'` at `run_deploy.py:89` (`task_text=(args.text or args.task)`); the parser (`run_dagger_round.py:27-38`) defines `--tasks` and no `--text`/`--task`.
- Even if fixed: `dagger/rollout.py:20-31` drives `DeploymentRuntime` directly — no z-floor (`run_deploy.py:262-268`), no STOP hitbox (`:270-279`), no joint-space start gate/homing (`:349-425`), no AE settle, no outcome prompt, `--max-replans 20` (the 08-28 doc raised it to 40), `--ema` default **False** (run_deploy defaults True), `mode="student"` hard-coded. The playbook still points at it (`training_playbook.md:76`, `launch_guide.md:128`).
- Fix: retire `run_dagger_round.py` + `dagger/rollout.py`; collect all rollouts with `run_deploy` and add `--dagger-round k` / `--policy-name` pass-through tags there (both already exist on `run_episode`, `runtime.py:153-155`). Update the docs.

### F6 [medium] The DAgger relabel pass produces files nothing reads; its anchors and `prev_chunk` do not match deploy
- `relabel.py:52-55` writes `<ep>/relabels/teacher.pt`; grep: no reader. `HIDConfig.teacher_mode="cached"`/`relabel_dir` (`training.py:70-71`) are unused; `distill_step` always calls `teacher_rf.sample(batch)` online (`distill_hid.py:75-76`). `dagger_driver` therefore spends a full teacher pass per rollout (bf16, `relabel.py:60-62`) for nothing. Also: anchors every `chunk_horizon/rate = 1.6 s` (`relabel.py:80`) whereas the rig replans every ~0.9 s; and the teacher's `prev_chunk` at relabel = the last 16 *recorded* action rows (`windows.py:277`, executed prefixes of successive plans) whereas at deploy it was the previous full proposal (`policy.py:101-111`; acknowledged as DEFERRED at `planner.py:281-290`).
- Fix: default `--skip-relabel` (or delete `relabel.py`); if a cached teacher is wanted for H100 economy, implement the reader keyed by (episode, t0) and anchor at the recorded replan times from `planner_trace.json`. For the *teacher* self-improvement loop none of this is needed — there is no expert to relabel with; the "expert" is the outcome label + human recovery (§5).

### F7 [medium] `distill_hid` and `finetune_hids` train on the held-out `val` split
- `distill_hid.py:205` `ds = C.WindowDataset(data_root, sampler_s)` and `finetune_hids.py:157` — no `episodes=C.manifest_split(data_root, "train")`, unlike `train_teacher.py:259-260`. `WindowDataset.build_index` falls back to `list_episodes(root)` = every finalized episode incl. the 124 val episodes. Any student/HID-S offline number reported against `val` is contaminated; the DAgger on/off ablation would be too.
- Fix: two lines each: `train_eps = C.manifest_split(data_root, "train")` → `episodes=train_eps`; add the same val loader as the teacher.

### F8 [medium] `finetune_hids` AWR reward mixes units and un-masks failure demos
- τ per task: `calibrate_tau_per_task` uses `dv.peak_normal_force` on **raw** `fields_ds` (`finetune_hids.py:58-63` → `derived.py:216-223`, raw \|depth\|). Window reward: `window_peak_force` cumsums `batch["cpk_d_disp"][..., 2]` (`finetune_hids.py:79-81`) which `windows.py:303-306` has already **normalized** by `norm_stats` — a cumsum of z-scored deltas compared against a raw-depth quantile. `adv = r - running_mean` is then meaningless; `w = exp(adv/β)` saturates at the clip.
- `per_sample` (`:107-109`) ignores `batch["action_weight"]` → the 115 `*_fail` demos (over-squeeze + under-grasp) get AWR-weighted **action** imitation, the one thing the rest of the codebase carefully prevents.
- Also `build_model` twice (student + frozen reference, `:143-146`) = 2 × 2B on one GPU; `taus` from `list_episodes(root)` includes val.
- Fix: compute both sides in the same units (de-normalize `cpk_d_disp` with `norm.std/mean`, or compute the window's peak from the raw field frames the sampler already touched); multiply by `action_weight`; or shelve HID-S (it is "optional row" and student-only — not on the critical path).

### F9 [medium] Rollouts recorded with the safety batch carry a different `config_hash` → `train_teacher` CONFIG DRIFT hard-fail
- `run_deploy.py:262-289` builds `hw` via `apply_z_floor`/`apply_hitbox`/`apply_tcp_speed_limit` (`safety.py:306-319` `hw.model_copy(update=…)`); `DeploymentRuntime`/`EpisodeRecorder`/`EpisodeWriter` stamp `meta.config_hash = hw.config_hash()` (`episode_store.py:45-48`) = sha256 of the **full** `model_dump` (`hardware.py:522-527`), which now includes the raised `safety.workspace_m.z` and hitbox. `WindowSampler._ep` compares against `load_hardware(args.hardware).config_hash()` (`windows.py:145-155`) → every rollout counts as drift → `train_teacher.py:286-296` `SystemExit("CONFIG DRIFT…")`. The documented override `--allow-config-drift` is blanket and would also hide a *real* drift (the wrist-window 125-vs-31 class of bug the check exists for).
- Fix: record the hash of the *pre-override* hardware config for deploy episodes (pass `hw_base.config_hash()` into `EpisodeWriter`, or exclude the `safety` block from the hash *on the recording side only* — do **not** change `config_hash()` itself, or all 1115 stored demo hashes stop matching). For the 39 rollouts already recorded, the intake pass can rewrite `meta.config_hash` after asserting `hardware_shapes` equality.

### F10 [low] Small items
- `run_dagger_round --ema` default False vs `run_deploy` True; `--max-replans 20` vs 40 (`run_dagger_round.py:32-35`).
- `intake_recovery.manifest --val-min-eps` would hold out the *last deploy day* of rollouts as `val` (session = `ep.resolve().parent.name` = `YYYYMMDD`, `:145`); rollouts must always be `split: train` (a rollout-state val set should be built from human *recovery* demos, not from rollouts).
- `WindowDataset.grasp_frac` anchors on `close_index` — for an on-air close this is a bogus terminal window; harmless while the episode is weight 0, wrong if a failed rollout is ever given weight > 0.
- `dump_norm_stats.py:49` uses `list_episodes(root)`: if rollouts live under `tasks/`, a rerun would fold them into the stats and break `--init-weights` (`train_teacher.py:237-253`). Document "never rerun".
- Docs (`training_playbook.md:73-80`, `launch_guide.md:118-131`) describe a DAgger flow that does not run.

---

## 3. What is missing for a 2–3-iteration filtered-BC / advantage-weighted loop (50–150 rollouts)

1. Rollout intake (label, re-derive actions, hash, weight, place, manifest) — **does not exist**.
2. Per-episode weight plumbing — **does not exist** (F4).
3. An on-policy-state validation set — **does not exist**: `terminal_eval` needs a GT close inside the chunk, so it cannot score rollout states (no GT there). The only GT on policy-produced states is a *human recovery from that state*.
4. A source of positive supervision at the failure states. Filtered BC on successes only yields ~5–10 successes per 30 rollouts at the current 0/26 rate (likely 0 for a while) — not enough to move a 2B model in 3 k steps. The literature the team collected agrees ("SFT on successes only stalls"). The cheapest positive signal is **recovery demos started from the policy's own end state** (DAgger-lite without any runtime handoff code): after each failed rollout the arm is exactly at the covariate-shift state; open the gripper (`GRIPPER_OPEN.sh`), start a collect-app episode from there, teleop the correction + grasp + place. This replaces the hand-jogged "recovery protocol" whose start poses QC showed were inside the demo distribution (`intake_recovery.py:20-23`).
5. Nothing exploits the seed multi-modality: no K-sample selection at replan and no offline "which seed succeeded" analysis. Not needed for the loop, but the Q-planning / best-of-K idea in the lit note needs a critic that this repo does not have.
6. No test covers `dagger/*`, `dagger_driver`, rollout intake, or `run_deploy`'s label→meta path (grep of `tests/`).

---

## 4. Success rule (auto-label) with thresholds from the demo statistics

Inputs: `gripper.zarr`, `arm_tcp_pose.zarr`, `tactile_{left,right}_fields_ds.zarr`, `meta.json`. All thresholds are either training's own constants or ≥ 3× margins below demo p5.

```
t_close  = ts_grip[close_index(pos)]                    # common.close_index (0.45 / +0.15 rise, fallback)  -> None => label 0
z_close  = tcp_z(t_close)
t_rel    = first t > t_close with pos(t) < pos(t_close) - 0.10, else episode end   # lift.py rule
hold     = [t_close + 0.5 s, t_rel)
contact(t) = max_f mask_frac_f(t) > tau_contact_area (0.025)   # mask_frac from dv.derive_timestep on fields_ds,
                                                                # = the ACC gate label definition (windows.py:331,353)
c_hold   = mean_{t in hold} contact(t)
lift     = max_{t in hold, contact(t)} tcp_z(t) - z_close
grasp_ok = (t_close is not None)
        and (z_close <= Z_CLOSE_MAX[task])              # demo p95 + 15 mm: waffles 103, Carton 144, egg 101, whiteboard 181 (mm)
        and (len(hold) >= 2.0 s)                        # demo hold p5 = 5-11 s
        and (c_hold >= 0.8)                             # to be validated on compute3: expect ~1.0 on successes, ~0 on under-grasp fails
        and (lift >= 50 mm)                             # demo lift p5: egg 74, whiteboard 86, Carton 201, waffles 228
stall    = mean_{t in hold} (obj(t) == 2) > 0.5         # over-squeeze flag only (95-100 % on the v4 *_fail demos); NOT part of grasp_ok
task_ok  = operator verdict 's'                          # place/wipe completion stays human
label    = 1 if grasp_ok and task_ok; 0 if (not grasp_ok) or verdict 'f'; EXCLUDE if verdict 'c'/None or grasp_ok != task_ok
```

Why these signals: the 08-28 rollouts closed at z = 137–190 mm (waffles; demo p95 = 88) with p_none 0.99 — rule 2 and 4 both fire. The 45 under-grasp fails lift an empty gripper 105–263 mm (rule 5 would pass!) — only rule 4 (tactile contact) separates them, which is why contact must be mandatory and OBJ must not be. Validate the rule on: 1037 successes (expect ≥ 95 % → tune `c_hold` down to 0.6 if tactile dropouts bite), 45 under-grasp fails (expect 0 %), 39 rig rollouts (expect 0 %), 80 over-squeeze fails (expect grasp_ok = 1, stall = 1). Write the validation as a pytest fixture over a small hub subset so the rule is regression-tested.

---

## 5. Concrete design that fits in a week

### Files to add / change (≈ 1 dev-day)
1. **`tools/rollout_intake.py`** (new; mirrors `intake_recovery.py` subcommands):
   - `label <deploy_day_dir>`: computes §4 per episode, writes `meta.auto = {grasp_ok, z_close_mm, lift_mm, c_hold, stall, t_close}`, reconciles with `meta.success`, sets `meta.tags += ['rollout', 'iter:<k>', 'ckpt:<file>']`, sets `meta.weight`, marks `status='aborted'` + `unlabeled`/`contaminated` where §4 says EXCLUDE.
   - `rederive-actions`: renames `actions.zarr → actions_plan.zarr` (never deletes) and writes a new `actions.zarr` = `pose_delta(tcp(t_k−0.1), tcp(t_k))` + gripper pos on a 10 Hz grid from `arm_tcp_pose`/`gripper` (F1). Parity test against a demo.
   - `fix-hash`: asserts `meta.hardware_shapes == hw.shape_relevant_fields()` and rewrites `meta.config_hash` to the demos' hash (F9) — until the recording-side fix lands.
   - `place` + `manifest`: symlink under `tasks/<task>/`, append rows with `split: train`, `source: rollout_r<k>`, `weight`, never val.
2. **`phantom/data/schema.py`**: `EpisodeMeta.weight: float = 1.0`; **`phantom/data/windows.py:403`**: `action_weight = (0 if is_failure_demo else 1) * meta.weight` (F4). Optional per-window terminal multiplier (below).
3. **`phantom/scripts/run_deploy.py`**: no-verdict ⇒ `recorder.relabel(status='aborted', tags=['unlabeled'])` (F2); `--dagger-round`/`--policy-name` pass-through; `--noise-scale` (one multiply in `rf.sample`, tagged). **`phantom/train/common.py:manifest_split`**: skip rows with `contaminated`/`unlabeled` tags.
4. **Recording-side hash** (F9): `EpisodeWriter(..., config_hash=hw_base.config_hash())` from `DeploymentRuntime`.
5. Retire `run_dagger_round.py`, `dagger/rollout.py`; default `--skip-relabel` in `dagger_driver` (F5/F6). Fix `distill_hid`/`finetune_hids` split (F7) if any student run is planned.
6. **`tools/terminal_eval.py`**: accept `--episodes-from-manifest val_recovery` so recovery demos from policy states become the on-policy validation number.

### Weighting formula (no critic; what 3 k fine-tune steps can absorb)
```
w_ep = 1.0                      demos (1037)
w_ep = 0                        failure demos + failed rollouts (action masked; contact/event/gate heads still supervised — unchanged)
w_ep = w_R = 1.0                recovery-from-policy-state demos (human actions from the shifted states)
w_ep = w_S = clip(1 / (p_task + 0.2), 1, 3)   successful rollouts; p_task = the collecting policy's success rate on that task
                                              (AWR with r∈{0,1}, A = r − p, β = 1 gives exp(1−p) ≈ same range; the clip is the max_weight)
window multiplier (rollouts + recoveries only): ×2 for t0 ∈ [t_close − 1.5 s, t_close − 0.2 s]  (the commit phase — reuse WindowDataset's grasp band)
```
Sampling: keep `--grasp-frac 0.3`; the rollout/recovery pool is small (≈ 60–100 episodes vs 1037), so also oversample those episodes ×3 in `build_index` (`windows_per_episode`) rather than pushing `w_S` higher — large weights on few windows destabilize a low-LR LoRA fine-tune.

### Rig + compute plan (2 iterations, ~100 rollouts, fits ~8 rig-hours)
- **D1 (code)**: items 1–5 above; validate §4 on compute3 over the hub data (numbers in §4); relabel the 39 existing rollouts (expected all 0) — they become iteration-0 negatives for the contact heads and the first "on-policy state" pool.
- **D2 (rig 1, ~3 h, waffles + Carton)**: `run_deploy --seed 100 --persistent-noise --episodes 20` per task, v5_6. After *each failure*: gripper open, collect-app recovery demo from the arm's current state (`docs/recovery_demos_protocol.md` steps 3–5 apply verbatim; skip its step 2). Yield ≈ 40 rollouts (expect 0–8 successes) + ≈ 35 recovery demos.
- **D3 (compute)**: intake → `train_teacher --init-weights v5_6 --max-steps 3000 --grasp-frac 0.3 --photo-aug 1.0 --event-band-weight 0 --acc-two-pass --lr … (v5 knobs)` → `terminal_eval` on `val` **and** on 8–10 held-out recovery demos (the on-policy number) → hub → `DEMO.pt`.
- **D4 (rig 2, ~3 h)**: interleaved A/B v6 vs v5_6 on the 3×3 grid, 10/arm/task, same s/f/c labels; again recovery demos after each v6 failure → iteration-2 pool (~40 rollouts + ~25 recoveries).
- **D5–6**: v7 fine-tune from v6 on the union; rig 3 = paper A/B (v4 → v5 → v7 curve + the ablation "recoveries vs successes-only vs demos-only", one flag each).

### What to report in the paper from this
Success rate per iteration (auto-label = tactile contact-through-lift, human verdict for task completion, both logged), z-at-close and miss distance from `planner_trace`, and the ablation that the tactile label — not OBJ — is what makes the loop work (F3 is itself a reportable finding: mechanical object-detect fails on soft objects; the visuotactile gel does not).

---

## 6. Explicit uncertainty
- Certain (read + reproduced): F1, F2, F4, F5 (crash reproduced), F6, F7, F9 (hash is over the full dump; run_deploy records the overridden hw).
- Believe (read, not run): F8 unit mismatch (normalization in `windows.py:303-306` is unambiguous; the tau side is raw by construction).
- Educated guess needing the compute3 check: the `c_hold ≥ 0.8` threshold in §4 — the demo *tactile* contact fraction during the hold was not in any local statistics file (only gripper/TCP were). Everything else in §4 is from the 1115-episode table.
