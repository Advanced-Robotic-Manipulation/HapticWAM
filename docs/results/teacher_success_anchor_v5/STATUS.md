# Teacher comparison at the successful scene

This is an active experiment, not a winner announcement. The original v1
pick-and-place remains a physical success at 25.2 s; its later safety stop is
reported separately.

The scene and measured start are fixed to that successful cell. The comparison
uses teachers only. Waffle placement is held fixed because collection varied
arm starts and trajectories much more than waffle position. Arm-start testing
is a separate extension after confirmation.

## Development evidence

| Profile | Acquired | Lifted | Carried | Placed | Trials |
|---|---:|---:|---:|---:|---:|
| Historical controller baseline | 2 | 2 | 2 | 0 | 2 |
| FINISH only | 2 | 1 | 1 | 0 | 2 |
| Gel mapping correction only | 2 | 2 | 2 | 2 | 2 |
| Distal gripper wrist proxy only | 2 | 2 | 2 | 0 | 2 |
| Current veto only | 2 | 1 | 1 | 0 | 2 |
| Delivery feedback only | 2 | 1 | 1 | 1 | 2 |
| All corrections together | 2 | 1 | 1 | 0 | 2 |
| Gel correction plus FINISH | 2 | 1 | 1 | 0 | 2 |

These are previously observed development seeds 904301 and 904302. They do not
count toward model selection or confirmation. The all-corrections v4 gate
failed and that plan remains unexecuted. The v5 amendment is explicit and
prospective: validated mechanics and inputs admit the environment; a particular
teacher's success is not a prerequisite for comparing teachers.

Accepted-command replay reproduced all 21 state/contact arrays bitwise at all
501 timestamps shared with the original successful run. Native inference on
saved first observations reproduced the corresponding saved proposals bitwise.
Fresh closed-loop runs are still sensitive to RGB and timing differences.

## Active profile and execution

The minimal v5 source is frozen v1 plus the gel mapping correction and the
post-release FINISH hold. It retains the historical wrist, veto and request-time
feedback. The physical scene, safety limits and strict object/support scoring
are unchanged. Source composition is documented in
[the source audit](minimal_profile_audit.md).

The two minimal-profile development trials completed with wrist-extension
stops at 15.988 and 15.868 s. Neither reached release or FINISH. This does not
establish a regression caused by FINISH: the source preserves pre-completion
commands, while closed-loop image and timing histories differ between runs.
It does establish that the earlier two gel-only successes were insufficient
evidence of repeatable placement.

The four-recipe screen launched at 09:11:14 UTC on 2026-09-07 under
[the frozen protocol](protocol.json), campaign SHA256
`3392e9f703ef008f2a16d6c133870192ddb1642526ef7adb3702b2e8a6c9b9c5`.
Twenty-four screening cells precede the top-two paired confirmation. No
prospective model result is available yet. Raw recordings and earlier outcomes
are retained on compute3.

Simulated tactile inputs are uncalibrated; this selected historical controller
profile is not the current live rig software. A simulator ranking requires a
separate lab qualification and an explicit software/configuration handoff.
