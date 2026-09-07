# Gripper-contact wrist proxy: preflight passed

**The new proxy includes the housing collision while preserving the original physical replay exactly.** All 21 original state/contact arrays and all 182 scene frames are bitwise identical to the original seed903102 trajectory. The scene configuration is unchanged. The added arrays are only cached `wrist_ft` and `wrist_capture_t`.

The [numeric audit](gripper_wrist_preflight_audit.json) verifies 1,503 wrist samples on the exact 125 Hz grid from 0 to 12.016 s. Every scene-frame wrist value equals the latest causal wrist sample, with zero value/timestamp error and maximum age 4 ms. No future sample is used. The initial contact wrench is zero and the initial wrist vector exactly equals the declared native recorded bias; that same bias is added unchanged at every sample.

Only the gripper housing and two pad rigid bodies contribute. Their explicit external contacts retain the raw signed PhysX normal impulses, normals and world points. The CPU audit independently recomputes each force as `signed_impulse / .004 s × world_normal`, then each moment as `(contact_point − measured_TCP_position) × force`. Every per-actor force, per-actor moment, total wrench and bias addition has zero arithmetic error against the saved records. There is no force clipping or absolute-value replacement in the vector wrench calculation. At 122 matching timestamps, normal-force magnitudes also agree exactly with the separate whole-robot diagnostic observer.

## Collision signal and timing

| Event | Simulation time | Signal |
|---|---:|---|
| Initial sample | 0 s | Contact wrench zero; recorded bias unchanged |
| First force change above 60 N | 5.096 s | Housing force `[0, 0, 76.086]` N |
| Peak contact force | 5.192 s | Housing force `[0, 0, 176.401]` N |
| Original tracking error first above 20 mm | 5.300 s | Accepted versus measured TCP divergence |

At the force peak, the loaded point is approximately `[-.295756, -.080545, .194000]` m, with world normal `[0, 0, 1]`, and positive raw impulse `0.705606` N·s. This produces an upward force opposing the commanded descent. Relative to the measured TCP, its contact moment is `[-1.209, -17.287, 0]` N·m. Adding the unchanged recorded bias gives raw wrist Fz `157.400` N. The direction and lever arm follow the actual contact buffers; the proxy does not rotate these vectors into tool axes or move the torque origin to world zero.

The force threshold crossing establishes that the previously omitted housing load is now available before the large mechanical tracking error. **This command replay does not execute a new safety decision or establish a closed-loop stop time.** The safety supervisor's baseline adaptation, debounce and changed commands must be assessed in the separately declared policy run.

## Scope and limits

This is an idealized contact-normal wrench about the measured TCP, expressed in world/UR-base axes. It is **not a calibrated model of the UR3's current-based wrench estimate**. Tangential/friction forces, gravity, inertia, self-contact and proximal-arm contacts are deliberately omitted. True hardware torque-origin conventions, tare, sensor transfer and contact geometry still require validation. Including all upstream arm contacts by simple summation would not be a justified extension.

The proxy changes the declared observation model for the replacement study; it does not change any recorded result or retroactively alter the halted screen. Its physics-invariance and arithmetic checks pass for this preflight. Reproduce with the [read-only CPU auditor](audit_gripper_wrist.py) on compute3; source and input hashes are included in the JSON. No simulator launch or source edit was performed by this audit.
