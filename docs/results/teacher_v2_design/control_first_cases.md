# First frozen v2 cases: controller feedback audit

The opt-in simulator/native terminal-veto feedback contract had a real mismatch. The frozen simulator used the captured model-request TCP and aperture; the new native release bridge used measured delivery TCP and aperture. This audit identifies that code gap without attributing the two observed failures to it: **none of their 85 delivered plans had any action rewritten by the veto, and replacing its feedback with the measured delivery state changes no action arrays in the recorded-trajectory replay.**

These are superseded diagnostic runs from `runs/teacher_robustness_v2/screen/rollouts`, not results from the amended delivery-feedback study. Their files remain unchanged. Full measurements, per-plan decisions, source hashes and input hashes are in [control_first_cases.json](control_first_cases.json); the read-only CPU reconstruction is [control_first_cases.py](control_first_cases.py).

| ftA1500 NFE1/K4, Aug22 start 5928 | Seed 903101 | Seed 903102 |
|---|---:|---:|
| Requested / delivered plans | 75 / 74 | 12 / 11 |
| Actual delivered actions differing from raw proposal | 0 / 74 | 0 / 11 |
| Delivery-feedback counterfactual action changes | 0 / 74 | 0 / 11 |
| Largest request→delivery TCP Z difference | 44.490 mm | 53.055 mm |
| Largest request→delivery measured closure difference | 0.06451 | 0.02613 |
| Veto decisions | 17 `none`, 57 `close_allowed` | 11 `none` |
| Largest requested→measured TCP translation gap | 3.118 mm | 212.113 mm |
| Maximum saved gel normal force, left/right | 0 / 0 N proxy | 0 / 0 N proxy |
| Release-controller phase | Always `unarmed` | Always `unarmed` |
| First additional commanded closure | 10.916 s | None |
| Safety stop | None in 60 s horizon | 10.020 s, `joint_speed` and `workspace_clamp` |

“Additional closure” is the first issued command greater than initial measured closure plus `2/255`; it does not mean contact. At 10.916 s in seed 903101, the current measured TCP was `[-0.360000, -0.172538, 0.076549]` m and commanded closure was 0.446978. The preceding 10.868 s physical trace sample put the pad midpoint 57.613 mm outside the packet oriented bounding box. No loaded latch or release gate was armed. In seed 903102, the stop row had requested TCP `[-0.380804, -0.104911, 0.031500]` m versus measured `[-0.504574, -0.105400, 0.203758]` m. Its largest joint speed was 1.338116 rad/s. The logs establish poor physical tracking despite converged geometric IK before the stop; this audit does not identify the collision or drive cause.

The first ten raw action increments sum to XYZ `[-20.021, +41.357, -29.878]` mm for seed 903101 and `[-22.284, +77.542, -39.443]` mm for seed 903102. These are proposed chunk displacements, not fully executed trajectories. The filter left those proposals unchanged. Later tail closures likewise cannot be treated as executed: the first `close_allowed` plan in seed 903101 had a raw maximum closure of 0.504031, but its executed command maximum was 0.446978 before replacement.

## Geometry and interpretation

The retained task settings are `z_ref = 0.0415`, `z_margin = 0.0615`, `z_floor = 0.0315` m, so the native close hatch is TCP Z ≤ 0.103 m. The estimated scene table top is 0.053 m and initial packet center is 0.070 m. These are different geometric quantities: the veto threshold is TCP height, not pad clearance or object distance, and the safety floor is not a tabletop collision test. The first additional closure was already permitted by this hatch. Therefore, removing its mask or moving its close threshold cannot be credited with correcting these particular recorded failures. Contact geometry and pressure inputs still require separate validation.

The saved gel values are simulator proxy outputs, with declared Newton units for the PhysX-derived increment. Their zero values do not prove that every part of the gripper or robot had no environmental contact. Static measured sensor residuals and the mapping law remain uncalibrated.

## Reconstruction and amendment

The helper loads each original request observation, raw proposal, delivery timestamp, causal previously issued gripper history, and saved tactile baseline/gel loads. A replay using captured feedback reproduces every recorded action array and veto action label. A second evolving veto instance substitutes only measured delivery TCP/aperture while retaining the same recorded trajectory and proposals. Only seed 903101 replan 18 changes its decision label (`none` → `close_allowed`); its actions remain identical. This fixed-trajectory counterfactual is not a prediction of a rerun's closed-loop result.

The local amendment changes only the opt-in release profile to current measured arm/gripper rings. Default operation retains captured request feedback. Neither changes the original teacher observation or raw proposal. Missing measured feedback raises an error instead of silently falling back. Per-plan `diag.terminal_veto.feedback_source` and `policy_info.json:terminal_veto_feedback_source` distinguish `current_delivery` from `request_snapshot_historical`; the former also records the arm and gripper sample timestamps. Existing safety checks retain their priority.

Four delayed actual-adapter regression cases cover both directions across the close-height threshold with and without the release profile, verify measured aperture masking, and prove captured inputs/proposals remain unchanged. A missing-feedback case verifies explicit failure. The focused terminal-veto, placement-release, release-FINISH and policy-adapter suite passed **84 tests**; scoped Ruff checks passed. Frozen `source_teacher_v2` and its scored/diagnostic runs were not edited.
