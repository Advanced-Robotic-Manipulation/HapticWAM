# Teacher release-dwell diagnostic

**Completed: the original 0.200-second release dwell achieves 2/2 strict placements; the 0.100-second treatment achieves 1/2. Keep the original dwell with the new bounded-hold controller as the working simulator configuration.** This is a development choice, not a statistically reliable winner or hardware qualification. The treatment's failed case never sustains a lift; that failure precedes any possible effect of the release dwell.

The controller code and ftA1500 weights are unchanged from the completed V8 bounded-hold experiment. This follow-up changes only the opening-intent dwell in the existing release gate, from 0.200 to 0.100 seconds. It launched on 2026-09-08 at 06:37:40 UTC after declaration and CPU preflight and finished with launcher exit code zero at 06:47:36 UTC. The final frozen binding SHA256 is `26b1689eae3f8c8acf1d97938a61cb5257a81599a2a0f26bb539b49bc4efb418`; all 5,106 pinned files passed both the final preflight and post-run check. Both campaign controllers also exited zero, owned processes ended, and port 7799 was free.

## Physical outcomes

| Release dwell | Seed | Acquired | Sustained lift | Carry | Strict placement | Placement time | Stop/drop |
|---|---:|:---:|:---:|:---:|:---:|---:|---|
| 0.200 s control |904301|yes|yes|yes|yes|24.668 s|none|
| 0.200 s control |904302|yes|yes|yes|yes|24.336 s|none|
| 0.100 s treatment |904301|yes|yes|yes|yes|25.936 s|none|
| 0.100 s treatment |904302|yes|no|no|no|—|measured joint-speed stop at 33.108 s; no scored drop|

All four trials are valid. Each successful packet is released, settles with bin support and unloaded robot contact, and remains there through the full 60-second horizon. These are object-state successes, not merely release-gate or FINISH events. The failed pickup remains a failure in the planned denominator; its absence of a scored drop does not establish retention because the sustained-lift gate was never met.

The same bounded-hold/original-dwell configuration scored **0/2 in V8 and 2/2 in these fresh V9 controls**. Both results remain visible and separate. Identical sampling seeds and scene inputs do not guarantee identical closed-loop rollouts with measured RPC timing and physics. The shorter-dwell comparison therefore does not establish that lowering the dwell improves release, or that the dwell caused the failed pickup. No model weights were changed between these results.

The independent [analysis](analysis/README.md) checks raw hashes, actual control/observation timing, divergence before release commitment, original versus accepted opening commands, and physical support. The [tactile side-by-side reviews](video_review/README.md) retain all four outcomes. [Completion evidence](completion/README.md) records the denominator, process cleanup and source/input verification.

Both matched pairs already diverge at first policy delivery: 1.676 versus 1.444 s for seed 904301, and 1.660 versus 1.468 s for seed 904302. This precedes the earliest eligible release opening by more than 20 seconds. The failed treatment never arms the loaded-grasp latch or enters its release window; the shorter dwell is behaviorally inactive. Its measured joint speed reaches 1.22944 rad/s despite the submitted-command step bound of 1.0 rad/s, and the existing 1.2 rad/s measured guard stops it. Command bounds do not certify measured dynamics. The three successful packets end with zero recorded robot contact force and approximately 0.343 N of bin support.

## Working configuration and limits

Use ftA1500 EMA, NFE1, K4, guidance 1, persistent noise/parity, maximum ten played steps and the same minimal_v5 profile. Enable the opt-in command limiter with elbow minimum 0.40 rad, command joint-speed maximum 1.0 rad/s and **2.5-second verified hold budget**. Retain the original **0.200-second release dwell** and all other release and measured safety settings.

This fixes a proven controller abort and now has demonstrated full-task executions. It does not remove the remaining pickup and timing variability or validate the estimated contact/tactile model. There is no demonstrated benefit from the 0.100-second change, so it is not promoted. For the next confirmation, keep the waffle/box fixed, measure the setup, and vary authentic measured arm/gripper starts with fresh matched seeds. Native execution needs an attended qualification of streamed holds, measured aperture and release timing; this study did not start or modify hardware.

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
