# Teacher carry hotfix: bounded, verified constraint holds

The premature constrained-hold abort is fixed. The six-case development experiment demonstrates recovery from that hold, but **does not demonstrate reliable pick and place: all six cases fail strict placement**. One hotfix trial carries the packet into the box and retains it through the full 60-second observation horizon. The other never establishes a sustained lift and eventually hits the existing upper workspace boundary.

This is a controller fix with unchanged ftA1500 teacher weights. It is not a new model winner or a completed hardware qualification. The previous successful V5/V7 recordings remain successful; these fresh trials do not overwrite them or enter their confirmation denominators.

## What was wrong, and what changed

The original reach problem occurs when the teacher's combined translation and wrist rotation move the arm toward extension during lift/carry. The measured wrist-radius stop responds correctly. Earlier recorded commands show that this is not generally an IK failure, elbow-branch jump, or deletion of the teacher's translation by the executor. The reconstructed observations and cross-session setup remain possible causes of the policy's trajectory; the evidence does not isolate a defect in the weights. See the [prior diagnosis](../teacher_behavior_debug_20260907/README.md).

The optional command limiter adds a separate defect: valid, stationary constraint holds share the 25-reject counter with failed motion. At 125 Hz that allows only about 0.2 seconds, shorter than a fresh teacher response. Both new legacy-limiter controls reproduce this stop.

The new opt-in `servo_constraint_hold_s` setting separates a verified constraint hold from invalid IK and branch faults. During a hold the native driver continues streaming the verified stationary joint target, and simulation reports its actual held FK. A fixed elapsed-time deadline bounds the wait. A new plan, repeated target, or identical joint solution cannot renew it. At least 0.001 rad cumulative accepted joint progress clears a live hold; expiry produces the ordinary stop `servo_constraint_hold_timeout`. Existing measured safety checks remain active throughout.

The tested hold budget is **2.5 seconds**, with the existing optional elbow bound of 0.40 rad and command joint-speed bound of 1.0 rad/s. The measured wrist-radius stop remains **0.468 m**. No force, workspace, speed or reach safety threshold was raised. No object-state oracle, forced release, motion-component projection, physics change or retraining was added. Default hold behavior remains disabled.

## Declared six-case experiment

All cases use one identical measured arm/gripper start and the same fixed successful-anchor scene. Seeds 904301 and 904302 vary policy sampling, **not arm position**. The checkpoint is `teacher_v5_ftA/teacher_001500.pt`, SHA256 `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e`: EMA, NFE1, K4, guidance 1, persistent noise, parity and at most ten played steps. All cases retain full-client `rpc_wall` delivery timing, the minimal_v5 veto/release behavior and the same approximate tactile mapping.

| Setting | Seed | Acquired | Sustained lift | Carry | Strict placement | Stop |
|---|---:|:---:|:---:|:---:|:---:|---|
| Limiter disabled |904301|yes|yes|yes|no|wrist extension, 15.620 s|
| Limiter disabled |904302|yes|no|no|no|wrist extension, 15.676 s|
| Legacy limiter |904301|yes|yes|yes|no|25 constrained ticks, about 0.2 s|
| Legacy limiter |904302|yes|no|no|no|25 constrained ticks, about 0.2 s|
| Bounded hold |904301|yes|yes|yes|no|none; full 59.996 s trace|
| Bounded hold |904302|yes|no|no|no|upper workspace boundary, 42.484 s|

All six are valid trials, with no drops or inference failures. The two stopped legacy cases retain their actual failure in the denominator. The second bounded-hold trace continues to 44.480 s for post-stop physics observation. Peak lift alone is insufficient: the second bounded-hold packet rises 31.7 mm briefly but fails the sustained-lift criterion.

The source/configuration conditions ran sequentially, with two reused development seeds per condition. Actual inference durations remain part of the closed-loop dynamics and are not numerically forced to match. This small experiment establishes a hold-recovery mechanism, not a reliable success-rate improvement. The fresh limiter-disabled seed 904301 failing while its historical V7 run succeeded also shows why a seed alone does not guarantee an identical real-time closed-loop rollout.

## Exercised recovery and remaining failure

In bounded-hold seed 904301, the main hold starts at 12.980 s. A fresh plan activates at 13.644 s; accepted joint progress of 0.006514 rad clears the hold at 14.252 s. The controller therefore gives the teacher the recovery opportunity that the legacy 0.2-second abort removed. The run subsequently achieves a 342.3 mm lift and 413.8 mm carry. The packet ends inside the bin but remains pad-loaded, so it is not scored as released or placed.

The two bounded-hold runs contain 169 and 276 accepted held rows, respectively: approximately 1.352 and 2.208 seconds accumulated across distinct holds. Neither exhausts the 2.5-second deadline of an individual hold. Accumulated hold time is not a single continuously renewed timeout, and accepted constraint holds must be read from their explicit telemetry rather than the older rejected/stale-hold metric.

The independent [analysis](analysis/README.md) finds a second execution mismatch: bounded-hold seed 904301 does request original openings at or below 0.45 closure, unchanged by the veto, while inside the eligible release window. Only two contiguous bursts reach execution: 25.604–25.724 s (0.120 s elapsed) and 45.020–45.100 s (0.080 s elapsed). Neither meets the configured 0.200-second continuous dwell. Stronger opening samples also occur in planned tails that are replaced before execution. The latch therefore remains engaged. This trial must not be described as a teacher that never asks to release.

The [tactile side-by-side reviews](video_review/README.md) use saved simulated tactile observations. Physical placement is determined from the packet and contact/support traces, not from appearance or the controller's FINISH flag. The carried packet stays gripped inside the bin volume without bin contact or unloaded support.

## Decision and next intervention

Keep this opt-in implementation as a verified controller correction. Do not describe it as an end-to-end task fix or silently promote the diagnostic 2.5-second limiter setting to the lab handoff. The teacher still needs a reliable grasp and a feasible carry/release sequence. Increasing the measured reach limit or globally disabling the gripper latch would not address those causes.

The completed, separately declared [V9 release-dwell diagnostic](../teacher_release_dwell_v9/README.md) compares the current 0.200-second opening dwell with 0.100 seconds while retaining the bounded-hold controller and every other behavior setting. Its original-dwell controls place2/2 packets; the shorter-dwell treatment places1/2. The failed treatment never sustains lift or activates the release gate, so the shorter dwell cannot explain that pickup failure. Retain the original0.200s dwell with the bounded-hold fix as the working simulator candidate. V8's0/2 and V9's2/2 for the same settings remain separate evidence of timing/repeatability sensitivity, not a reliable-winner claim.

A later controller experiment can preview complete translation-and-rotation prefixes of all four teacher candidates from the actual accepted state, retain candidates with reach and workspace margin, and log every candidate. Existing logs do not prove that another unselected candidate would have succeeded. Keep the waffle/box fixed and use authentic measured arm/gripper starts for subsequent distribution testing, matching collection practice; vary measured object-reset jitter separately.

## Validation and reproduction

The implementation passed 228 focused local checks across the available simulator and PyTorch environments. After the identical-joint Cartesian-anchor correction, all 32 affected integration/shared-selector checks passed. The assembled frozen runtime passed **100 focused CPU tests**, with its retained logs in [preflight](preflight/). No hardware was started or updated.

The scored runtime preserves the prior immutable V7 simulator with only the declared hold changes. Native dependencies are pinned separately; all three runtime/driver/inference trees use the live hardware schema plus only the optional hold field. Model/preprocessing code and checkpoint bytes remain unchanged. The [launch record](launchplan/README.md) and [completion record](completion/README.md) preserve the six-case declaration, frozen input/source hashes, actual outcomes and cleanup checks. All 5,098 pinned files passed the post-inference check. The three campaign controllers exited successfully, their policy servers stopped, and port 7799 was free. The detached launcher's operating-system exit status was not retained; its final `six_valid_trials_completed` marker was retained.

Authoritative raw runs are on compute3 at `/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_carry_hotfix_v8/paired/`. Local review artifacts were recovered after an overnight session reset; raw trials were not rerun. Source recordings, the live rig checkout and older scored studies remain unchanged.
