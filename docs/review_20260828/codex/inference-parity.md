1. Severity: high  
File: [phantom/data/windows.py](/Users/sannikov/GitHub/phantom/phantom/data/windows.py:276), [phantom/deploy/planner.py](/Users/sannikov/GitHub/phantom/phantom/deploy/planner.py:278), [phantom/deploy/executor.py](/Users/sannikov/GitHub/phantom/phantom/deploy/executor.py:182), [tests/test_deploy_parity_fixes.py](/Users/sannikov/GitHub/phantom/tests/test_deploy_parity_fixes.py:436)  
Claim: Deploy feeds `prev_chunk` as the last accepted plan proposal, while training feeds the last executed `STREAM_ACTIONS`, so inference is conditioned on motion that often did not happen.  
Evidence: `WindowSampler` builds `prev_chunk` from recorded action rows before `t0`, but `PlannerLoop` explicitly leaves the executed-history fix “DEFERRED” even though `ChunkExecutor` time-warps, blends, clamps, and rate-limits the accepted plan before the arm sees it.  
Failure scenario: If the executor slows or attenuates a descent chunk, the next replan still believes the full descent-and-close happened and can decide it is already low enough, producing the observed “stop high, close on air” behavior.  
Minimal fix: Feed `prev_chunk` from a 16-step ring of executed post-blend/post-governor/post-safety actions (or measured TCP deltas plus gripper state), then lock it with an end-to-end parity test against `WindowSampler`.

2. Severity: high  
File: [phantom/data/windows.py](/Users/sannikov/GitHub/phantom/phantom/data/windows.py:252), [phantom/deploy/planner.py](/Users/sannikov/GitHub/phantom/phantom/deploy/planner.py:77), [phantom/inference/policy.py](/Users/sannikov/GitHub/phantom/phantom/inference/policy.py:166), [phantom/deploy/executor.py](/Users/sannikov/GitHub/phantom/phantom/deploy/executor.py:96), [phantom/config/hardware.py](/Users/sannikov/GitHub/phantom/phantom/config/hardware.py:315)  
Claim: Training learns “observe at `t0`, then predict actions starting 0.1 s later from that same state,” but deploy executes each chunk from a state roughly one inference latency newer, and `model_tick_hz` is not enforced anywhere.  
Evidence: Training anchors future actions at `t0 + (1..H)/rate`; deploy stamps `Plan.action_times` at `obs.t + latency`, `ChunkExecutor.submit()` rebases the chunk onto the current pose at submit time, and a repo-wide `rg model_tick_hz` finds only the schema definition.  
Failure scenario: With the rig’s reported 0.9-1.3 s steady-state replans, the model acts on stale visual/proprio evidence while the arm has already moved under the previous chunk, so small z errors compound until the close happens well above the object.  
Minimal fix: Train and eval with explicit latency augmentation that shifts the action target start by measured replan latency, and make deploy honor a real replan cadence instead of “as fast as inference finishes.”

3. Severity: high  
File: [phantom/model/rf.py](/Users/sannikov/GitHub/phantom/phantom/model/rf.py:150), [phantom/inference/policy.py](/Users/sannikov/GitHub/phantom/phantom/inference/policy.py:158), [phantom/deploy/planner.py](/Users/sannikov/GitHub/phantom/phantom/deploy/planner.py:278)  
Claim: `prev_cpk` has different semantics in training and deploy: training uses a same-window proxy, while deploy uses the true previous replan’s package from older observation/history.  
Evidence: `_acc_inputs_train()` either injects noised GT or a short same-batch self-sample as the “previous” package, but deploy passes `prev_plan.cpk` from the prior accepted replan and reuses it at the next call.  
Failure scenario: ACC learns on a non-stale proxy and then is asked on-robot to fuse a stale previous-plan contact package with fresh observations, which can mis-bias contact anticipation exactly around the grasp approach.  
Minimal fix: Either ablate `prev_cpk` consistently in train and deploy as an immediate control, or generate real previous-replan packages offline by rolling the policy over recorded episodes at deploy cadence and train on those.

4. Severity: high  
File: [phantom/deploy/runtime.py](/Users/sannikov/GitHub/phantom/phantom/deploy/runtime.py:182), [phantom/deploy/executor.py](/Users/sannikov/GitHub/phantom/phantom/deploy/executor.py:205), [phantom/recording/recorder.py](/Users/sannikov/GitHub/phantom/phantom/recording/recorder.py:103), [phantom/data/windows.py](/Users/sannikov/GitHub/phantom/phantom/data/windows.py:215)  
Claim: Deploy episodes record nominal plan actions, not executed actions, so any rollout data reused for DAgger or offline analysis is mislabeled.  
Evidence: `DeploymentRuntime` wires `executor.record_action` into the recorder, `ChunkExecutor` logs `plan.actions[k]` before safety clamp and without correcting for governor/blend/rate-limit, and `WindowSampler` later trains only from `STREAM_ACTIONS`.  
Failure scenario: A rollout that actually executed a shortened or clamped descent is stored as a full descent chunk, so DAgger retraining and action-gap postmortems learn from fiction rather than what the robot did.  
Minimal fix: Record executed action deltas after final target selection, or derive them from consecutive measured TCP poses in a rollout-only stream and make training consume that stream.

5. Severity: medium  
File: [phantom/data/windows.py](/Users/sannikov/GitHub/phantom/phantom/data/windows.py:379), [phantom/deploy/planner.py](/Users/sannikov/GitHub/phantom/phantom/deploy/planner.py:179), [phantom/data/derived.py](/Users/sannikov/GitHub/phantom/phantom/data/derived.py:226)  
Claim: The teacher-only `reactive` input is computed over one tactile-frame step in training but over one whole replan interval in deploy.  
Evidence: Training compares the current field to its immediate predecessor at `t0`, deploy compares the latest fields to `self._prev_fields` from the previous snapshot, and `reactive_score()` is just mean absolute difference with no dt normalization.  
Failure scenario: On the rig, where replans are about a second apart, the reactive cue lands far outside the training distribution and can distort the teacher gate near first contact.  
Minimal fix: Build `reactive` from the latest two tactile ring frames or from a fixed-rate tactile buffer, and add a parity test that checks deploy snapshots against `WindowSampler` on recorded episodes.

6. Severity: medium  
File: [phantom/data/windows.py](/Users/sannikov/GitHub/phantom/phantom/data/windows.py:202), [phantom/deploy/planner.py](/Users/sannikov/GitHub/phantom/phantom/deploy/planner.py:160), [phantom/data/derived.py](/Users/sannikov/GitHub/phantom/phantom/data/derived.py:122)  
Claim: Deploy computes the `contact_state` slip feature with a hardcoded nominal tactile `dt`, while training uses the actual inter-frame timestamp delta.  
Evidence: `_field_frame()` returns true `dt` for training windows, but `SnapshotBuilder` sets `dt_field = 1.0 / field_ds_rate_hz` before calling `derive_timestep()`, whose slip statistic divides by `dt`.  
Failure scenario: Any tactile jitter or dropped sample rescales deploy slip and therefore shifts the teacher’s `OBS_MECH` input distribution relative to training.  
Minimal fix: Use the two latest tactile timestamps from the ring to compute `dt`, falling back to the nominal rate only during single-sample warmup.

7. Severity: medium  
File: [configs/paths.yaml](/Users/sannikov/GitHub/phantom/configs/paths.yaml:24), [phantom/backbone/text_embedding.py](/Users/sannikov/GitHub/phantom/phantom/backbone/text_embedding.py:51), [phantom/data/windows.py](/Users/sannikov/GitHub/phantom/phantom/data/windows.py:390), [phantom/scripts/run_deploy.py](/Users/sannikov/GitHub/phantom/phantom/scripts/run_deploy.py:88)  
Claim: In this repo configuration, text conditioning is effectively disabled even though both training and deploy pass task strings.  
Evidence: `configs/paths.yaml` leaves `cosmos_text_embedding_cache` empty, `TextEmbeddingProvider` then falls back to the empty-string embedding for non-empty text, `WindowSampler` fills `w["text"]`, `run_deploy` passes `args.text or args.task`, and a local `load_paths()` check returned `cache=''`.  
Failure scenario: The code appears to use task/instruction text, but all tasks collapse to the same embedding, so any instruction-conditioning or text-based task-disambiguation claim is unsupported.  
Minimal fix: Require a verified text-embedding cache whenever non-empty text is present, or remove text from the claimed input set until that cache is populated and parity-tested.