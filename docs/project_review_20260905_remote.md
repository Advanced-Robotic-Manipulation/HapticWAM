**PHANTOM review supplement: actual compute3 artifacts**

September 5, 2026. Companion to the [project review](project_review_20260905.md) and [history ledger](project_review_20260905_history.md). This follow-up inspects retained experiment artifacts over SSH, rather than relying only on repository accounts. Compute3's tracked checkout matches `f64aea8`; the outer launch scripts also match their tracked copies.

**The remote evidence makes the diagnosis more specific.** The teacher has learned something measurable, and the latest recorded session contains none of the large joint-speed spikes recorded on September 1. The immediate gap is converting that progress into stable task execution and a complete, controlled outcome table. The logs do not support blaming the present blockage on NaNs, an unfinished fine-tune, or a universally broken validation split. They do expose a real launch-ownership failure and another statistical correction.

**What was inspected.** The project root is `compute3:/home/physicalai/phantom-icra-2027`; the Python repository is its `phantom/` child. I read all 117 retained deployment metadata files and available planner traces under `data/episodes/deploy/`: 116 are marked real and one is marked mock. I read all available stop records and the low-dimensional arm/gripper/wrench streams for all 51 September episodes, which are all marked real. I also inspected checkpoint metadata, resident-server and launch logs, archived E9/E13 results, the ftA checkpoint sweep and preserved training logs. No GPU training/sampling, robot control, process management or remote file changes were performed. Reading artifacts and recomputing statistics is not rerunning the policies.

The [episode ledger](review_20260905/episodes.csv) preserves all 117 records, their raw checkpoint/revision tags, recorded labels/stops, selected telemetry summaries, original paths and metadata/trace SHA-256 digests. Unknown labels remain unknown. Missing stop reasons remain unrecorded. The [read-only inspection script](review_20260905/inspect_remote_episodes.py) records how the telemetry summaries were computed. This is the retained recording inventory, not necessarily every launch ever attempted: the logs contain at least one failed launch that never created an episode.

**1. There is stronger evidence of a modest offline improvement than the prose alone conveyed.**

The raw E13 files contain exactly matched episode/window/seed keys, four seeds per episode, and no duplicate keys. Recomputing v5 minus v4 while keeping the four seeds together gives:

| Held-out set | v4 endpoint error | v5_6 endpoint error | Mean improvement | Episode-bootstrap 95% interval for improvement |
|---|---:|---:|---:|---:|
| Older 78 episodes | 22.17 mm | 20.86 mm | 1.31 mm | 0.12–2.98 mm |
| Expanded 124 episodes | 20.82 mm | 18.23 mm | 2.59 mm | 1.29–4.11 mm |

V5 improves the seed-averaged endpoint in 54/78 and 90/124 episodes respectively. A collection-session bootstrap sensitivity check also yields positive intervals: 0.17–2.63 mm for the eight older sessions and 0.75–4.95 mm for the fifteen expanded sessions. These are descriptive, post-selection validation intervals from 10,000 bootstrap draws, not preregistered claims or robot-success estimates. The small session counts limit precision. Individual-window seed variation is not a reason to dismiss the aggregate improvement.

The original 790-row manifest has 81 collection sessions: 73 training and eight validation, with **zero sessions crossing the split**. That separation is directly verified. The extra 46 validation episodes belong to seven collection sessions and have consistent raw directory counts, but the complete expanded training manifest is not available here; their separation from the expanded training pool cannot be independently certified from these artifacts alone.

Sources: remote `/tmp/e13_val78_{v4,v5_6}.json`, `/tmp/e13_val124_{v4,v5_6}.json`; manifests under project `data/val_eval/` and `data/val124/`. The [recomputed results](review_20260905/e13_recomputed.json) include source digests and exact statistics; the [recomputation script](review_20260905/recompute_e13.py) documents the matching, missing-value handling and bootstrap units.

**2. The corrected close-height claim still contained a subset mismatch.**

The earlier E13 correction described +4.79 mm for v4 and +6.00 mm for v5 as a comparison on identical rows. The remote data shows these averages use **197 and 250 different finite-close rows**, respectively. On the 194 rows with finite predicted and ground-truth close heights for both models, the means are **+4.82 mm for v4 and +7.21 mm for v5**.

Missing predicted closes are material: among 460 rows with a ground-truth close, the evaluator finds a predicted close in 197 v4 rows (42.8%) and 250 v5 rows (54.3%). Thus v5 crosses the evaluator's closing threshold more often, while closing higher on the shared finite subset. Report both; averaging only observed closes hides this tradeoff. Here “no predicted close” means the tool's threshold was not crossed within its 16-step terminal window. It is not an adjudicated robot grasp failure.

This amends the close-height paragraph in the first review. It does not revive the old 17–19 mm explanation for 30–60 mm closed-loop misses. The E9 tactile-null and contact-pinning numbers otherwise match the surviving raw files: tactile-null worsens endpoint error from 20.86 to 23.24 mm, and both zero-pinned and GT-pinned future contact perform worse than ordinary co-denoising. Their interpretation remains input sensitivity, not proof of distillation benefit.

**3. Training ran, but the selection metrics disagree.**

The preserved logs contain 24,000 optimizer-step records: 18,000 from v4 steps 2001–20000, and 3,000 each for v5 and ftA. All reported metrics in those records are finite; no traceback was found. Both fine-tunes reached step 3000. This does not prove every gradient or update was correct, but it rules out the simple explanation that these runs never completed or visibly collapsed into NaNs. Sources are the immutable archived [v4 log](https://huggingface.co/armteam/phantom-checkpoints/blob/241af105223b319a34486672f8ccad3b61368d49/teacher_v4_790eps/train_v4.log), [v5 log](https://huggingface.co/armteam/phantom-checkpoints/blob/241af105223b319a34486672f8ccad3b61368d49/teacher_v5_batch0822/train_v5.log) and [ftA log](https://huggingface.co/armteam/phantom-checkpoints/blob/241af105223b319a34486672f8ccad3b61368d49/teacher_v5_ftA/train_ftA.log).

V5's sampled action cosine is approximately 0.807–0.819 from step 500 onward, while sampled MSE varies from 0.566–0.670. FtA's action training loss falls sharply, but sampled MSE moves from 0.6755 at step 500 to 0.8468 at step 3000; cosine improves from approximately 0.724 to 0.773 and remains below v5's range. FtA step 1500, the deployed build, has sampled MSE 0.9761, the worst of its six sampled-validation checkpoints. It was chosen using a separate terminal evaluation. Raw training-loss magnitudes must not be compared across these runs as if the noise and objectives were unchanged.

Those training-loop sampled diagnostics use a small fixed window set and **raw weights**, while the deployed artifacts use **EMA weights**. The disagreement identifies different selection criteria, not a direct ranking of deployed policies or proof that step 1500 was incorrectly selected. The separate EMA terminal sweep below is the closer artifact comparison.

The actual terminal sweep provides a more precise comparison than the generic “ftA” label:

| Build | Overall mean endpoint | Overall median endpoint | Waffles mean endpoint | Waffles signed end-height error |
|---|---:|---:|---:|---:|
| v5_6 | 18.23 mm | 15.74 mm | 26.18 mm | +8.36 mm |
| ftA_1500 | 19.60 mm | 13.67 mm | 24.42 mm | +13.54 mm |

FtA_1500 improves the median and waffles endpoint, but worsens overall mean endpoint and ends higher. FtA_2500 is the build with approximately 13.4 mm median and 21.0 mm mean; those figures should not be attached to step 1500. This is a selection tradeoff that requires the frozen waffles pilot, not evidence that either metric guarantees closed-loop superiority.

Actual checkpoint metadata also narrows some implementation concerns. Both teachers used microbatch four and accumulation two, so the previously reproduced cancellation of positive weights at microbatch one does **not** apply to these training configurations. They share normalization and include text conditioning; the inspected server logs explicitly confirm EMA application. V5 and ftA change several model/training factors, so their comparison cannot isolate one loss or inference idea. The logs/configs record a compute profile but not a reliable physical accelerator/world-size audit; they do not prove that either run used the historically problematic multi-rank path.

Source checkpoints are `phantom/runs/teacher_v5_batch0822/v5_6.pt` and `phantom/runs/teacher_v5_ftA/teacher_001500.pt`; their current SHA-256 digests and the archive revision are preserved in [provenance.json](review_20260905/provenance.json). These are current file identities, not retrospectively recovered per-episode hashes. Training logs were recovered from the project's referenced model-artifact archive after no corresponding files were found on compute3. Both inspected teachers have 27,889,205 trainable parameter values and no tactile-SSL checkpoint initialization. Their sampled label checks include both gate classes and all five contact-event types, so the earlier collapsed-label diagnosis does not describe these checks. Distributed-force calibration remains disabled; native resultant-wrench calibration is separate. These artifacts do not establish a calibrated distributed-force safety predictor.

**4. The deployment inventory establishes progress and missing outcomes simultaneously.**

| Recorded session | Episodes | Stored success=true | Stored success=false | Unknown |
|---|---:|---:|---:|---:|
| August 11 | 17 | 0 | 0 | 17 |
| August 13 | 1 | 0 | 0 | 1 |
| August 14 | 4 | 0 | 0 | 4 |
| August 18 | 7 | 0 | 7 | 0 |
| August 20 | 11 | 0 | 11 | 0 |
| August 28 | 26 | 0 | 15 | 11 |
| September 1 | 22 | 0 | 12 | 10 |
| September 4 | 29 | 0 | 0 | 29 |

These columns count retained records and literal stored labels, **not a success-rate estimate**. August 11 includes the one mock episode (`ep_teacher_waffles_1786474335`); the ledger explicitly records driver mode. “Zero stored successes” cannot establish that no grasp ever succeeded; all September 4 episodes are aborted/unlabeled despite evidence of contact and lifting. The earlier 0/26 August summary is also not the same as 26 operator-labeled failures.

Four September 1 recordings contain joint-speed maxima of 3.36–6.93 rad/s. None of September 4's 29 recordings exceeds 3 rad/s; their maximum is 1.302 rad/s. That is direct evidence that the later session did not reproduce the earlier large spikes. It is not a controlled estimate of a particular guard's effect, since code and starting conditions changed.

September 4's recorded termination reasons are:

| Reason | Episodes |
|---|---:|
| Operator stop | 13 |
| Safety stop | 10 |
| Lift-complete detector | 2 |
| Executor crash | 1 |
| Unrecorded | 3 |

Seven of the ten safety stops have detailed event records: five wrist-extension stops and two joint-speed stops. The remaining three have only the generic reason. The detailed reach-related stops occur with minimum absolute elbow angle approximately 9.9–20.6 degrees; the two speed stops record 1.243 and 1.302 rad/s. One later wrist event reaches 0.46823 m, so the raised margin was exercised in an uncontrolled trial; it has not been established as a validated operating envelope.

**Team account added when preparing the GitHub handoff.** [Issue #4](https://github.com/Advanced-Robotic-Manipulation/phantom/issues/4) explicitly reports that the limiter was enabled at elbow minimum 0.40 rad and joint-speed cap 1.0 rad/s for the last six September 4 episodes; counter persistence was added afterward. This corrects the earlier historical assertion that the limiter never ran on the arm. The measured final-six maxima of 0.25–0.41 rad/s and all-operator stops are consistent with that account, but do not isolate its effect from starts/seeds or prove task completion. The retrospective activation account should be cited; missing counters must not be treated as proof of nonactivation. The issue's count of 27 episodes differs from the 29 retained metadata records, so the ledger preserves all 29 rather than silently removing two.

The recorded terminal state is indeed post-stop: for example `1788538626_001` records the triggering speed as 1.243 rad/s but `stop_state.qd_max` as 0.0115 rad/s. Trigger telemetry and settled terminal state need separate names and storage.

Median episode-level planner latency for September 4 is 0.813 seconds; the median left tactile update rate is 8.016 Hz. These are measured planner/wrench-stream timings, not the raw camera acquisition rate or the arm servo rate. Any next latency intervention should be sized against these measurements.

**5. September 4 was not a verified v5-versus-ftA comparison.**

The first 23 episode tags say `BEST.pt`; the last six say `teacher_001500.pt`. The current `teacher_v5_ftA/BEST.pt` symlink resolves to `teacher_001500.pt` and dates from September 1. The launch log also resolves this alias to step 1500. FtA's server has sustained client sessions; v5's logged connections last approximately 40 milliseconds, consistent with identity probes. Together, these sources strongly support the interpretation that September 4 exercised ftA through two loading paths, rather than two different models.

Per-episode weight digests were not recorded, so retrospective identity cannot be made cryptographically certain. Do not treat the two basename tags as an A/B. All 29 episodes claim git `fd4a032` despite three hardware config hashes and evolving stop behavior. September 1's full 22-episode inventory contains four revision tags, refining the three-revision subset described in the repository account. These findings turn the provenance concern into an observed limitation.

**6. A real launch failure is explained by the current busy-server design.**

The ftA server retained a client connection from 19:40:52 to 19:47:19 on September 4. During that interval, the terminal transcript shows a new ftA/LEVERS launch reporting only a v5 server, loading ftA locally, then failing all three RTDE initialization attempts because input registers were already in use. It did not reach episode recording.

The implementation explains how this can happen: [the server](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/phantom/inference/remote.py#L225) services one connected client until disconnect before accepting another. [PICK](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/tools/rig/PICK.sh#L49) times out its information probe after ten seconds and interprets the lack of a response as no matching server, enabling local fallback. A busy server and an absent server therefore look the same. The logs strongly support this failure chain; they do not independently identify the owner of the robot's occupied registers.

Fix status reporting to distinguish idle, busy and unreachable, and acquire a cross-process rig ownership lease before any control/preflight action. The per-process driver counter is insufficient. This is a concrete prerequisite for reliable trials. The subsequent uncaught server EOF is visible in the same logs but already repaired in the current source; it should not be counted again as an unfixed defect.

Sources: project `logs/serve_ftA.log:61`, `logs/serve_v5_6.log`, and `tmp-terminal.txt:16,46,64`. Session-working copies are retained locally as `/tmp/research-review-remote-serve-fta.log`, `/tmp/research-review-remote-serve-v5.log` and `/tmp/research-review-remote-terminal.txt`. Server connection durations are not episode counts or inference latencies.

**What changes in the next-step recommendation.**

The first review's execution-first, one-task plan remains appropriate, now with firmer evidence. Add busy-server detection and cross-process ownership to the bounded execution prerequisites. Repair command acceptance and preserve trigger-state telemetry. Establish grasp/hold/transport/release labels for the retained September recordings where synchronized evidence permits; leave ambiguous outcomes unknown.

Then prove representative successful demonstrations can execute through the deployment controller and run a fixed, interleaved v5_6 versus ftA_1500 waffles pilot. Record exact weights and effective settings, actual starts, placements and every attempted launch. The remote evidence supports keeping both checkpoints as plausible candidates; it does not justify choosing a winner by one offline metric or interpreting aliases as different models.

Training loss reduction is already demonstrated. The next useful outcome is a reliably executed, clearly labeled task and an interpretable comparison. Student distillation and its required controls remain the subsequent research stage; the inspected artifacts do not establish a deployed student result.
