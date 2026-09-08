# V8 completion and release audit

The bounded hold fixes the premature carry stop in the first seed, but **none of the six trials completes placement**. The remaining failure in that seed is a release timing mismatch: the teacher supplies opening commands, but each played opening is shorter than the release gate's required continuous dwell. The other seed still fails to sustain a lift.

These are six contemporaneous development trials with one fixed measured arm start, one fixed waffle pose and two sampling seeds per setting. The inference delivery clock uses actual client RPC wall time. Sequential blocks therefore have different latency histories despite matched settings; these results do not establish a reliable model winner or a causal success-rate improvement.

| Setting | Seed | Acquisition / sustained lift / carry / placement | Stop | Last physics sample |
|---|---:|---|---|---:|
| Plain K4 | 904301 | yes / yes / yes / no | wrist extension, 15.620 s | 17.616 s |
| Plain K4 | 904302 | yes / no / no / no | wrist extension, 15.676 s | 17.672 s |
| Legacy limiter | 904301 | yes / yes / yes / no | 25 rejected holds, 15.108 s | 17.104 s |
| Legacy limiter | 904302 | yes / no / no / no | 25 rejected holds, 15.364 s | 17.360 s |
| Bounded hold, 2.5 s | 904301 | yes / yes / yes / no | none; no FINISH | 59.996 s |
| Bounded hold, 2.5 s | 904302 | yes / no / no / no | top workspace, 42.484 s | 44.480 s |

All six remain valid under the unchanged authoritative scoring thresholds. No trial records a scored drop. Acquisition without a sustained lift is not a completed pick. Raw trajectories and independent robot/bin contact support are retained; no controller FINISH is interpreted as task completion.

## What the hold fixed

The legacy limiter stops after exactly 25 held ticks, about 0.2 seconds, with no plan captured after hold onset delivered before stopping. In bounded seed 904301, a post-hold observation is captured at 13.652 s, delivered and activated at 14.452 s, and its first nonheld accepted execution is at 14.460 s. The policy subsequently carries toward the bin and remains active through 60 seconds. Bounded seed 904302 also executes a post-hold plan: capture 13.540 s, delivery/activation 14.860 s, first nonheld accepted execution 14.964 s. It later stops at the top workspace bound.

The two bounded runs contain 169 and 276 explicitly accepted held ticks. Independent replay of the hold budget from accepted joint targets agrees with recorded state; the longest active budgets are 1.264 and 1.704 seconds, below 2.5 seconds. Holds submit exactly the previous joint target and do not increase the IK fault count. Both bounded runs stay below 461.60 mm measured wrist radius against the unchanged 468 mm stop. Submitted joint steps satisfy the declared 1 rad/s bound and the elbow remains at or above 0.4 rad. These submitted-command checks do not certify hardware stopping distances or contact realism.

## Why seed 904301 never releases

The release gate is armed and the measured TCP is inside its declared release volume during both opening bursts below. The original policy gripper values pass the terminal veto unchanged, are eligible for release, and are selected by the executor. However, the gate requires an uninterrupted 0.200 s at closure ≤ 0.45 before it may suppress the loaded grip latch.

| Original opening selected by executor | Policy closure | Ticks | Elapsed time checked by gate | Actually commanded closure |
|---|---:|---:|---:|---:|
| 25.604–25.724 s, plan 29, index 5 | 0.443205 | 16 | 0.120 s | 0.647381 |
| 45.020–45.100 s, plan 53, index 6 | 0.435846 | 11 | 0.080 s | 0.647381 |

The corresponding command intervals last 0.128 and 0.088 seconds. Neither interval reaches the 0.200 s dwell. The opening timer resets when subsequent selected commands return above 0.45. The loaded latch therefore continues to command closure 0.647381.

This is also affected by chunk replacement. Plan 29 predicts stronger opening at indices 7–9 (closures about 0.436, 0.430 and 0.404), but only indices 0–5 execute before replacement. Plan 28's opening at indices 8–9 also never executes. The selected candidate includes release intention; this trace does not support the claim that the model never proposes opening. Arrays for the other K4 candidates were not recorded, so no unselected-candidate claim is made.

The waffle remains gripped: bin contact is zero throughout the run, and total robot normal contact is 12.707 N at the final sample. At 24 s the object is over the bin region at approximately `[-0.4049, 0.0555, 0.1121]` m, but still supported by the gripper. This is a carry into the placement region, not a physical placement.

## Bounded permission counterfactual

[release_audit.py](release_audit.py) replays the frozen release controller on the recorded TCP, selected policy closure, eligibility and observed latch state. It stops as soon as release permission commits. The unmodified 0.200 s gate exactly reproduces its observed phase and opening timer on every replayed active tick across all six cases.

Changing only opening dwell to 0.100 s would commit permission in bounded seed 904301 at **25.708 s**, using the same original policy command 0.443205 from plan 29, index 5. No other case commits in this observed-input replay. The holding branch does not read pad loads before first commit; no tactile replacement is introduced or used to justify this result.

This predicts a gate transition only. Once the gripper opens, object motion, tactile observations, subsequent policy outputs and safety behavior would diverge. Retention, release, placement and task success require a new closed-loop experiment. A separately declared 0.200-versus-0.100 s dwell comparison is therefore justified; V8 scores and its six-case denominator must remain unchanged.

## Audit correction and reproducibility

The [original audit](previous/completed_audit.json) flagged `accepted_pose_is_not_target_fk` in five stopped cases. Each finding is confined to the single `safety_hold` row. Frozen runner code explicitly reports measured TCP and submits measured q on that path; the reported FK discrepancy is only 1.56–3.21e−7 in a pose component. The [revised audit](completed_audit.json) keeps the 1e−10 nominal-FK equality requirement for active accepted commands and separately requires exact measured q and measured TCP equality for stopped holds. All six pass. No runtime code, raw trace, scoring threshold or saved primary score was changed to resolve the audit.

The numerical outputs pin the launch/completion markers, campaign inputs, checkpoint, source, primary scores and raw trace hashes. Raw data lives on compute3 under `/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_carry_hotfix_v8/paired`. Outputs can be reproduced with a new output directory/file:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python audit_completed.py --out /tmp/v8_audit_new
PYTHONDONTWRITEBYTECODE=1 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python release_audit.py --root /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_carry_hotfix_v8/paired --out /tmp/v8_release_new.json
```

Validation: all six authoritative input hash sets verified; all six independent completion/command/budget checks passed; baseline release gate replay exactly matched observed state in all six cases; both audit scripts passed Ruff. Source and raw data are read-only. This is still an exploratory simulation with unvalidated tactile transfer and estimated scene/physics parameters.
