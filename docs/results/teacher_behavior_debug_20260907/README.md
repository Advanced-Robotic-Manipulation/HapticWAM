# Teacher behaviour: diagnosis and proposed fixes

The teacher can complete pick and place, but its current execution is unreliable for two distinct reasons: some approaches touch the packet without securing a lift, and some retained grasps follow a carry trajectory into the arm's reach boundary before requesting release. There is also a reproducible closure/cadence problem in the faster K1 setting and a separate constrained-hold defect in the experimental limiter.

This is a **read-only, post-experiment diagnosis**, not a new policy experiment or a claim that the proposed fixes work. It examines all 12 ftA1500 V5 confirmation trials, four V7 timing/K diagnostics, and ten complete timestamped real demonstrations. V6 limiter conclusions below cite the already completed two-case audit. Source recordings, scored trials, controller settings and weights remain unchanged. Development cases are never pooled into confirmation.

The practical candidate remains **ftA1500 EMA, NFE1, K4, guidance 1**, persistent noise, parity, and the recorded maximum-ten-step/minimal_v5 execution profile. Its reserved confirmation result was **12/12 contact acquisition, 7/12 sustained lift, 6/12 carry and 1/12 strict box placement**, with one drop. That supports continuing diagnosis with this candidate; it does not establish reliable task completion or a statistically clear model winner. [Existing model decision](../../isaac_teacher_decision_20260907.md).

![Measured simulator trajectories and reach margins](kinematics/trajectory_comparison.png)

## Findings and their limits

### 1. Contact is being mistaken for a secure pick in informal interpretation

The frozen acquisition score requires bilateral packet contact above 0.1 N for 0.15 s. All twelve V5 trials meet it, but five never sustain a lift. In those five the loaded-grasp latch never arms, and the saved tactile observations frequently show weak or missing active-gel contact even while both physical pad bodies touch the packet. The teacher does issue upward motion and the controller preserves it. This is consistent with peripheral contact or insufficient retention, not a controller deleting the lift command.

The distinction matters because body contact, active-gel contact and stable retention are different measurements. Geometry, contact projection and the selected grasp pose could each contribute; saved observations alone do not establish their causal shares. Several failed retained carries have strong tactile inputs and high model hold forecasts, so missing tactile contact cannot explain every failure. The model forecast is anticipatory, not a calibrated measurement of grasp security. [Sensor audit](sensors/report.md), [proposal/execution audit](policy/README.md).

**Proposed fix:** first calibrate TCP-to-pad geometry, measured aperture versus closure command, active contact faces, and force/field transfer. Then test a bounded contact-and-closure phase before substantial ascent, using observable measured aperture/load/retention signals. Any added phase gate is a controller intervention and must be compared separately. Preserve force limits; do not fabricate gel observations from backing contact or lower success thresholds.

### 2. Carry orientation consumes reach margin before release

Across twelve V5 and two V7 K4 trajectories, there are no IK rejections or abrupt elbow-branch changes. Peak requested-to-measured TCP tracking errors are about 12–15 mm; orientation errors peak around 1.6–2.1 degrees. Failed retained carries generally move toward the box, but their combined translation and rotation approach the 0.468 m wrist-radius stop. In several failed carries, a local Jacobian decomposition finds that rotation increases radius while translation would reduce it. Height alone is not a sufficient diagnosis: successful and failed peak heights overlap, and some failures are already descending when stopped.

The successful release poses have nominal IK solutions, and local geometric paths from pre-stop configurations to those poses exist in the audited model. These calculations do not certify obstacle clearance, contact stability, dynamics or hardware tracking. They show that the box is not intrinsically unreachable in the nominal reconstruction. Even the V7 successful trajectory comes within **0.60 mm** of the stop, so its success does not establish a robust hardware margin. The six failed V5 lifted trials and failed V7 K4 carry do not propose release-sized opening in their post-lift executable prefixes; the latch is not suppressing an attempted placement release. [Kinematic analysis](kinematics/README.md), [release-command audit](policy/README.md).

**Proposed fix:** preview all K candidate prefixes from the actual accepted joint/pose reference, checking accumulated translation **and** orientation against IK branch, reach margin, joint-speed and workspace constraints. Log every candidate and separate translation/rotation costs. The current selector screens low-contact descent and then minimizes a mixed metre/radian continuity distance; rotation contributes about 81% of the selected-reference squared distance numerically. It does not check full-motion feasibility. Use explicitly justified physical or normalized weights, keeping the final measured safety guards. Existing logs omit unselected trajectories, so there is no evidence yet that reranking would have saved these trials. Do not substitute a blind height cap, locked orientation or enlarged reach limit.

### 3. Faster K1 often discards the closure it predicts

Median consumed action steps per plan were 2.48/1.76 in the two K1 trials, versus 6.69/7.36 in K4. For K1 seed 904302, plan 33 predicts closure increasing from about 0.485 to 0.592 over its first ten samples. Replanning plus governed playback executes only 1.31 samples: command 0.485–0.488, with measured closure 0.385–0.389. The close is allowed by the veto. Stronger later closure is repeatedly replaced before execution. Both K1 trials contact the packet but fail to lift.

**Proposed fix:** keep K4 for the current candidate. Test execution-prefix commitment or a predeclared replan cadence independently of K, preserving measured latency and sensor freshness. First replay saved commands through the CPU executor to verify which closure samples the intervention exposes, then run a small matched closed-loop test. Slower replanning is not already proven to produce a safe grasp; it also changes observation and previous-contact conditioning. If the effect persists, training should supervise the short prefix actually executed at deployment cadence. [Exact plans and timing analysis](policy/README.md).

### 4. The experimental limiter stops before a fresh response can arrive

In both V6 cases, valid IK candidates violate the configured elbow envelope while the stationary anchor remains valid. Twenty-five holds trigger a stop after about 0.2 s. In seed 904302 a post-hold observation is captured at 13.996 s, its response is due at 14.810 s, and the controller stops at 14.188 s. That response never gets an execution opportunity. This defect belongs to the optional limiter; it is not the cause of the limiter-disabled V5/V7 stops.

**Proposed fix:** distinguish a verified constrained hold from invalid IK, wrong branch or an invalid anchor. Explicitly stream the verified stationary joint setpoint at native servo cadence, keep measured safety checks active, and use a finite elapsed-time budget that permits a fresh observation-based plan. Do not reset the deadline indefinitely. Extending the timeout alone does not establish native servo parity, and a fresh response may still fail to recover. Keep this intervention separate from candidate filtering. [Completed V6 evidence](../teacher_success_anchor_v6/diagnostic_results/recommendation.md), [native hold requirements](../teacher_success_anchor_v6/hold_stop_review.md).

### 5. The current session transfer needs measurement before model blame

Ten complete successful August demonstrations include pick, carry and release. Their maximum wrist radius is 405–429 mm, leaving at least 39 mm below the simulator's current stop. Their carry peak TCP height is 302–358 mm. The fixed simulator start comes from a September deployment: it differs from those recorded demonstration starts in pose and has closure 0.078 versus 0.255–0.427. The reconstructed packet position and appearance also transfer between sessions. These are documented differences, not proof of a frame bug or an unseen training distribution.

The current window loader excludes the initial history interval from training anchors, but the checkpoint does not preserve an expanded historical episode manifest. Another 240 retained inspection mirrors lack enough stored data/timestamps for trajectory analysis and are excluded. Their incompleteness is **not evidence that the optimizer trained on corrupt data**. Real SDK area can be zero while the field mask indicates substantial contact, so that disagreement alone is not a simulator defect either. [Demonstration and provenance audit](data/README.md).

**Proposed fix:** measure base/table, TCP/pads/aperture, packet and box poses, camera calibration and sensor response, then replay a complete measured trajectory through the calibrated simulator. For distribution testing, keep the waffle/box fixed and use authentic measured arm and gripper starts, as in collection. Study measured object-reset jitter as a separate factor. If phase-matched observations and execution are correct but carry rotation still drifts, collect corrective close/lift/carry/release demonstrations and fine-tune specifically on those transitions rather than starting a broad checkpoint sweep.

## Next experiment, in order

1. **CPU controller check:** implement and test the hold-versus-fault distinction with native/simulator feedback parity, timeout, sensor freshness and safety preemption. This checks the controller mechanism, not task success.
2. **Small isolated simulator test:** compare that hold change alone on the two existing limiter failures and preserved successful anchors; retain every stop/drop and the full post-place horizon. In a separate cell, test candidate feasibility with all K trajectories recorded. Test K1 cadence only as its own intervention.
3. **Fresh confirmation after a mechanism improves:** use a fixed, measured waffle/box setup and declared recorded arm/gripper starts, with matched sampling seeds. Freeze success thresholds and controller settings beforehand. Keep development seeds out of the confirmation denominator. Random placement comes afterward as a separate robustness test.

No step above was executed in this audit. A successful legacy video remains a success; proposed interventions must preserve it while improving the full denominator.

## Reproducibility and validation

The policy audit verifies 552 plans and 537 deliveries. All pose channels are unchanged in 536 deliveries; the one exception is post-placement historical recovery in an already successful trial. Independent reconstruction matches 39,264 active requested XYZ targets within 1.24e-8 m, and gripper commands exactly. Three CPU selector checks reproduce the source behaviour. The sensor audit records hashes for all 552 saved observation inputs. The kinematics and demonstration audits retain numerical outputs, source/episode provenance and CPU extraction helpers; their subreports explain sampling and approximation limits.

The raw studies remain on compute3 under `/home/physicalai/phantom-icra-2027/sim/waffles/runs/`. Newly extracted evidence uses `teacher_behavior_debug_20260907/`. This directory contains analysis artifacts only; no inference, new physics rollout, model training or hardware launch occurred during this diagnosis.
