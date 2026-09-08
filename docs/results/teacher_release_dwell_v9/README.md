# Teacher release-dwell diagnostic

**Launched 2026-09-08 at 06:37:40 UTC after declaration and CPU preflight; outcomes pending.** The controller code and ftA1500 weights are unchanged from the completed V8 bounded-hold experiment. This follow-up changes only the opening-intent dwell in the existing release gate, from 0.200 to 0.100 seconds. The final frozen binding SHA256 is `26b1689eae3f8c8acf1d97938a61cb5257a81599a2a0f26bb539b49bc4efb418`; all 5,106 pinned files passed the final check. The owned supervisor records both campaign and final launcher exit codes.

## Reason for the intervention

In [V8 bounded-hold seed 904301](../teacher_carry_hotfix_v8/README.md), the arm recovers from a constrained hold and carries the packet inside the box, but retains it through 60 seconds. The teacher supplies eligible original opening commands in two executed bursts lasting 0.120 and 0.080 seconds. The existing gate requires 0.200 continuous seconds, so both bursts are discarded and the load latch stays engaged. Additional stronger opening samples occur in planned tails that are superseded before execution.

The 0.100-second treatment tests whether admitting the longer observed opening burst allows the teacher's opening to execute. It does not force an aperture or infer success from the release-controller flag. After commitment, all aperture commands still originate in the policy or the already frozen historical veto/recovery behavior. Retention, release, support, drops and later safety stops must be independently scored from physical state.

## Frozen comparison

Four new development trials: seeds 904301 and 904302 under each of the 0.200-second control and 0.100-second treatment. These are reused development seeds, not held-out model confirmation. The frozen execution order is **control/904301, control/904302, treatment/904301, treatment/904302**. The unchanged driver requires at least two distinct seeds and complete seed-grid coverage per campaign, so the originally preferred ABBA arrangement was rejected in CPU preflight before any simulation. This AABB amendment was declared before launch and preserves the existing driver. Condition order and changing shared-compute latency remain explicit comparison limitations.

Both settings use the same teacher checkpoint `teacher_v5_ftA/teacher_001500.pt`, SHA256 `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e`; EMA, NFE1, K4, guidance 1, persistent noise/parity and maximum ten played steps. Keep the identical measured arm/gripper start, fixed waffle/box/camera, frozen V8 runtime/driver/inference trees, full-client `rpc_wall` delivery timing, tactile mapping, minimal_v5 veto and release/FINISH behavior.

The bounded-hold budget remains 2.5 seconds; elbow and command-speed bounds remain 0.40 rad and 1.0 rad/s. The original-policy closure threshold remains 0.45, the loaded-grasp requirement and release volume remain unchanged, and every measured workspace/reach/speed/load/freshness guard remains active. The new dwell is a controller intent filter, not an increase in a physical safety limit.

Each run observes 60 seconds, including any FINISH, with the existing two-second physics tail after an actual stop. Use the same strict physical stage/support scorer and retain all four trials, including failures. No automatic retries, parameter sweep or post-hoc denominator changes. A missing or invalid trial is reported explicitly rather than replaced.

## Validation and decision rule

Before execution, replay the observed V8 gate inputs on CPU at both dwell values to verify the proposed mechanism. This counterfactual permission check cannot establish the resulting physical outcome because observations and subsequent plans change when the gripper moves.

The completed CPU check reproduces the original 0.200-second gate phase and opening-start state on every observed tick in all six V8 cases. At 0.100 seconds it commits permission only in bounded-hold seed 904301, at 25.708 s after opening begins at 25.604 s. The original and delivered teacher closure is 0.443204701 (plan 29, index 5), whereas the recorded latched command was 0.64738059. No other V8 case commits in this counterfactual. This verifies a gate effect, not object release. In particular, the qualifying burst has only about 16 ms remaining after commitment; the subsequent policy can still command closing, so physical completion is uncertain.

Launch only after frozen source/input/configuration checks show the dwell as the sole treatment difference. After completion, audit actual original versus accepted opening commands, eligible dwell duration, latch commitment, measured aperture/load, object release/support, FINISH, stops and inference latency. Preserve synchronized tactile side-by-side videos and all raw timestamps.

Promote neither setting as a reliable winner from two trials. If the shorter dwell exercises policy-authorized release and improves physical completion without an added drop or safety issue in the pair, retain it as a development candidate for a separately declared fresh-start confirmation. If it merely changes a gate flag without physical placement, report that failure and diagnose the next mechanism; do not relabel it as task success.

No hardware is started or changed. Raw V8 trials and their 0/6 strict-placement outcome remain unchanged.
