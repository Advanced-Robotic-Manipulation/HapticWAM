**Recommended next move: share the evidence, repair execution, then run one decisive lab pilot.**

This is the action plan from the [project review](../project_review_20260905.md) and [remote-artifact supplement](../project_review_20260905_remote.md). GitHub already has the [current carry investigation, #4](https://github.com/Advanced-Robotic-Manipulation/phantom/issues/4), and the [older investigation, #3](https://github.com/Advanced-Robotic-Manipulation/phantom/issues/3). Keep their historical context; extract bounded current work rather than opening another broad audit issue.

| Option | Recommendation | Concrete output |
|---|---|---|
| Share on GitHub | Now, as a review PR through the ordinary team workflow. | One entry point to conclusions, raw-artifact limitations, the episode ledger and this plan. |
| Open code issues | Four focused implementation issues, linked to #3/#4. | Acceptance criteria and regression cases; no assignment to unconfirmed owners. |
| Go to the lab | Book the next slot now. Run the hardware checks after the execution prerequisites pass; outcome review can start immediately. | Known-demonstration execution gate, then a frozen checkpoint pilot. |
| Work on the model | Defer another broad fine-tune. Allow targeted offline diagnostics tied to a specific failure. | A declared intervention supported by the pilot, rather than another bundle of changed factors. |
| Collect more training data | Only when the pilot identifies missing learned behavior. | Expert corrections from actual encountered states, plus session-separated evaluation. |
| Change strategy | Narrow the active question now. | One waffles grasp-and-carry task; reliable teacher first; privileged-transfer comparison afterward. |
| Build a simulator | Build a small offline execution harness alongside the fixes; defer a general tactile/contact simulator. | Reproducible driver/executor/session regressions without occupying the robot. |

**GitHub handoff, prepared locally.**

The [PR body](github/pr.md) and the four issue bodies below are drafts. This document does not indicate that anything has been published. Publish the review through a branch/PR; do not push directly to main, mass-assign teammates or send separate notifications. Link the resulting PR and focused issues to existing #4 when publishing is requested. #3 contains superseded diagnoses and already completed work; cite its context without reopening every old checklist item.

| Priority | Suggested issue title | Draft |
|---|---|---|
| P0 | Keep executor state aligned with the command actually sent to the arm | [Command acceptance](github/01-command-acceptance.md) |
| P0 | Distinguish busy policy servers and enforce exclusive rig ownership | [Session ownership](github/02-session-ownership.md) |
| P0 | Persist complete trial identity, stage outcomes and trigger telemetry | [Trial records](github/03-trial-records.md) |
| P1 | Correct evaluation clocks, paired close statistics and binary intervals | [Evaluation validity](github/04-evaluation-validity.md) |

The first three gate a counted lab experiment. The fourth gates use of the affected automated research metrics; a simple explicit human outcome table may support the engineering pilot before every optional anticipation metric is repaired. Keep the limiter disabled until its separate blocking cases in #4 pass. The team reports it ran in the last six September 4 episodes; that is evidence of a test, not a controlled demonstration of benefit.

**Agent work: a bounded execution harness, approximately one to two engineering days initially.**

Reuse the actual `ChunkExecutor`, `URArm` and localhost policy-server implementation. Inject scripted RTDE responses, measured-state lag, a controllable clock, recorded action proposals and delayed plan arrivals. Existing fake-controller and server tests supply much of the structure. Start with short regression fixtures representing silent IK rejection, empty/nonfinite IK, rejected servo commands, near-boundary incremental motion, a late replan and a second client probing a busy server.

The deliverable is a repeatable command that reproduces the relevant bug before its fix and passes afterward, with an event trace separating proposed, streamed and measured state. Assert that rejected commands cannot advance the executor anchor; invalid joints are never streamed; playback under approximately 0.8-second recorded planner latency and injected overruns obeys the existing cap/stop semantics; and a busy probe cannot initiate competing control ownership. Stop-state evidence must capture the trigger independently of the settled arm.

Existing `tools/replay_rig.py` and `tools/replay_deploy_path.py` replay policy predictions on recorded observations; they do not simulate how new actions change the world or exercise the entire actuator chain. The current mock arm also omits realistic IK/q/qd behavior. Extend the missing interfaces, rather than relying on those tools as if they already proved closed-loop task success.

Timebox the first harness to the defects above. Exact RTDE timing, real six-axis IK behavior, friction, slip, object retention and visual changes remain lab questions. A photorealistic or tactile simulator is a separate research investment. Consider it later only if a named, recurring physical failure is expensive to test and a simulator can reproduce and predict that failure on held-out real trials.

**Human lab work: one task and two blocks.**

Before running, choose a marked reachable destination from the demonstrated workspace and write the outcome definition. Recommended engineering task: acquire waffles, lift clear, transport to that destination and retain the object there for one second without manual task assistance or an interrupting guard. Score acquisition, retention and transport separately. Record release as a separate diagnostic until intentional release through the latch is implemented and validated. Call this grasp-and-carry; do not label it full pick-and-place.

Review the 29 September 4 recordings against that definition where the camera and streams permit. Preserve unknown outcomes and distinguish interrupted runs from completed attempts. The handoff does not relabel the underlying recordings automatically.

Block A is roughly 5–10 supervised executions of representative successful demonstrations through the actual delta-TCP/IK deployment path, using predefined tracking tolerances derived from physical clearances. A direct joint-space teleoperation replay does not answer the same question. If the runtime cannot execute known-good motion, use the remaining session to diagnose execution; do not interpret subsequent learned-policy failures as a checkpoint ranking. This is an engineering gate, not a statistical reliability claim.

Block B, once A passes, is 16 matched v5_6/ftA_1500 pairs, extendable to 20 pairs if the session allows. Freeze controller/settings and task definition. Match object placement, realized starting joints/pose and sampling seed, and alternate which checkpoint goes first. Use independent episode records for each attempt. Any code/control change starts a new evaluation block; do not pool it silently with the old one.

Primary outcome is the predeclared grasp-and-carry completion. Also report phase failures, intervention/stop causes, duration and planner latency. Include allocated attempts and pre-episode launch failures in the accounting, while distinguishing equipment-invalid attempts from task failures under an explicit rule. A guard-stopped attempted task does not disappear from the denominator. Sixteen pairs are a large-effect diagnostic pilot, not a promise of statistical significance or a complete paper experiment.

The human operator owns physical setup, supervision, resets and judgments requiring observation. The agent can own code fixes, offline checks, immutable manifests, checkpoint/recipe verification, pairing schedules and readout. There is no unattended hardware-testing requirement.

**Let the pilot decide subsequent investment.**

| Observation | Next action |
|---|---|
| Known demonstrations fail through the deployed controller | Fix geometry, timing, acceptance and task-phase semantics; do not retrain to compensate for an execution defect. |
| Demonstrations execute, but the policy repeatedly chooses wrong actions from particular states | Collect successful expert corrections from those states. Separate outcome provenance from normal task instructions, and retain an independent holdout. |
| Acquisition works but the policy opens or over-reaches during transport | Inspect carry/release supervision and controller semantics; change one intervention, checkpoint fixed, before bundling a new training recipe. |
| Model objectives or sampler settings are implicated by a reproducible offline ablation | Run that bounded intervention with unchanged data/runtime and declared checkpoint selection. Finite loss alone is not the success criterion. |
| One teacher completes the task repeatably | Establish the teacher–student–direct-supervision controls, including a no-tactile-target baseline for a privileged-training claim; then evaluate HID. |
| A simpler action-only baseline succeeds while PHANTOM fails under the same runtime | Reconsider model/objective complexity. |
| The controlled pilot still gives no stable completion | Replan the research scope and submission instead of adding more unrelated modules. |

The strategy change is a narrower sequence of falsifiable experiments. It does not require discarding the useful infrastructure or assuming tactile distillation cannot work. Defer HID-S, the large occlusion study and broad anticipation/uncertainty claims until the central transfer benefit and the relevant measurement tools are established.

**Exit condition for this work block:** one synchronized review handoff, four bounded issues, passing execution prerequisites, and a labeled pilot table that tells the team whether to invest next in control, correction data or the model. A new training run or a new simulator is not itself that exit condition.
