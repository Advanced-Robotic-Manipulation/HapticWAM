# Prospective shared servo limiter

The local branch now shares the existing optional native elbow/joint-speed limiter with an explicit simulator diagnostic. It is **not enabled in the running v5 screen**, and no frozen source or recording was changed. No new simulator, model or hardware run has been launched for this change.

The fixed settings are the existing historical values: elbow magnitude at least 0.40 rad and commanded joint speed at most 1.0 rad/s. Nominal UR3 geometry puts the wrist center at 0.461711 m at this elbow angle, leaving 6.289 mm below the unchanged 0.468 m measured stop. These values were not tuned to the observed trial outcomes. The measured stop, wrist/tactile force limits, release gates, gripper commands, scene and policy remain unchanged.

## Implementation and activation

- `phantom/drivers/servo_limiter.py` extracts the existing finite-IK, raw-branch, elbow, speed, three-iteration bisection and optional slide search. It imports no simulator or device connection.
- `phantom/drivers/real/ur.py` delegates its existing optional helpers to this module. The native executor already receives the shortened commanded pose through `ServoResult`; the disabled path still uses the original command-selection flow.
- `tools/sim/run_waffles.py --mode policy --servo-reach-limiter` explicitly selects the historical 0.40 rad / 1.0 rad/s settings. The disabled path retains its original nominal IK and independent per-joint speed clipping. The enabled path uses the shared selector instead of that independent clipping, then reports nominal FK of the selected joints as the accepted setpoint. It never reports that setpoint as a measurement.
- The enabled path records parameters in `policy_info.json` and `run.json`; execution rows record mode, violation, interpolation fraction, IK-call count, consecutive rejects and accepted joints/pose. Twenty-five consecutive rejects request `servo_limiter_stall`, preserving the commanded grasp and entering the ordinary stopped-tick observation tail. This is an executed controller failure, not retryable infrastructure failure.

The native driver already obtains `getInverseKinematics(target, qnear)` before calling `servoJ`. Reuse needs no new RTDE API. Each enabled tick performs at most seven IK calls total: the initial query and at most six candidate solves. Their real controller latency still needs measurement against the 125 Hz budget. The simulator uses nominal CPU IK and cannot establish that timing or serial calibration.

## Saved-command CPU evidence

[The reproducible audit](audit_existing_wrist_limiter.py) evaluated 143 bounded single-step samples from five immutable trajectories using the unchanged pre-extraction native implementation, pinned at driver SHA `b24cbdb9c21b6a5b113d59145e08b716655d9036f9bc2bc371e614f4879587a0`. [Numeric results](existing_native_trace_check.json) preserve source and execution hashes.

| Saved case | First command inside 0.40 rad margin | First command beyond 0.468 m | Actual measured stop |
|---|---:|---:|---:|
| Baseline failed carry, 904301 | 12.980 s | 14.716 s | 14.780 s |
| Combined bridge failed carry, 904301 | 13.388 s | 14.764 s | 14.828 s |
| Minimal failed carry, 904301 | 14.828 s | 15.924 s | 15.988 s |
| Gel-only placement, 904301 | 13.484 s | 54.108 s | 54.156 s |
| Original placement, 4242 | 16.076 s | 31.316 s | 31.380 s |

Thus the three failed carries offered 1.16–1.80 seconds of command-level warning before the measured wrist stop; their final command-radius crossings preceded the stop by 64 ms. The successful placements also entered the margin. The guard will alter successful trajectories too: this is not evidence that clamping produces placement.

All returned sample candidates satisfied the existing branch/speed checks and either stayed within the command margin or made a bounded inward escape from an already violating anchor. Most later failed-carry samples yielded a hold because their *original* preceding anchor was already outside the proposed command region. Every sample was independently reanchored to that original preceding command. Those holds do not predict what a sequential guarded rollout would do after the first intervention.

Observed measured-minus-commanded wrist-radius error ranged from −5.981 to +1.780 mm over these recordings. This describes recorded tracking, not a bound on future overshoot. The existing 1 rad/s command limit does not constrain measured acceleration, contact impulses or braking distance. Neither the margin nor the pure helper guarantees that the measured stop cannot still fire.

## Validation and remaining gates

Captured full-UR3 IK responses check the native extraction's exact query sequence, streamed joints and returned poses for both disabled and enabled cases. The enabled simulator selector matches those native decisions. Independent full-DH tests cover positive/negative elbow branches, inward escape, measured geometric radius, speed limits and retention of multi-turn wrist-joint branches; malformed clocks, missing/NaN IK, branch flips and bounded holds fail closed. A real `SimulationPolicyAdapter` test confirms that the 25th reject leads to an ordinary `servo_limiter_stall` and preserves grip. Local results: **76 tests passed** in the simulator environment; the one PyTorch-dependent native executor test also passed separately in the review environment (**77 total**).

The historical `abs(elbow)` rule is not a general periodic wrist-radius predicate. Its intended principal elbow branch must be verified at startup; the inspected command traces use that branch. Do not normalize all raw joints or accept an IK branch jump to satisfy this rule. The shoulder-to-TCP tangent is only a candidate direction: it is not the wrist-center constraint, and every returned IK solution remains checked. Bisection does not prove global reachability or find every feasible route.

Before any comparison, freeze the enabled setting as a separate profile, verify ordinary stop scoring and source hashes, and run an exact-command/initialization diagnostic with explicit runtime records. Then use matched checkpoints, seeds, starts and timing policy for a separate closed-loop comparison. Inspect actual q/qd, wrist-radius overshoot, accepted-pose feedback, rejects and full release/support gates. Keep the current v5 results separate. Hardware use additionally requires measured controller IK latency and cautious unloaded-pose qualification; nominal simulation geometry is not a hardware calibration.
