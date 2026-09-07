# Limiter hold and replanning clocks: prospective review

**The existing 25-reject stop can end a constrained hold before a plan based on the held observation arrives.** This is a code-level finding, not a prediction of either pending v6 result. No runtime, frozen source, threshold, or recording was changed.

## Exact current behavior

The frozen v6 runner and shared selector were inspected read-only on compute3. The current live native UR driver was separately checked at SHA `b24cbdb9c21b6a5b113d59145e08b716655d9036f9bc2bc371e614f4879587a0`; the older UR file carried inside the simulator source is not that live driver and is not used for simulated actuation. [Source hashes and the small CPU checks](hold_stop_review.json) distinguish them.

| Layer | Current intervention |
|---|---|
| Shared selector | A finite target with an elbow/speed violation gets at most six additional candidate IK solves. Failure returns `limiter_hold`; a missing target solution can also reach this reason with `violation=no_solution`. An initial branch jump is rejected before shortening. |
| Native UR driver | `limiter_hold`, invalid IK, branch rejects and missing anchors enter the same consecutive reject counter. Reject 25 raises `RuntimeError`; the executor crash handler reports `executor_crash`. The streak resets only after a successful `servoJ` send. |
| Sim runner | Every unsuccessful selector result enters its own 25-reject counter. Reject 25 calls `request_stop("servo_limiter_stall")`, then reports rejected execution. The adapter also has a 25-reject counter; first stop reason wins. A selected step resets both counters. |
| Sim feedback | Rejected arm targets do not advance the accepted TCP/joint anchor. Gripper commands execute independently and enter history even when arm IK holds. Plan playback continues; a newly delivered plan rebases at the last confirmed command. |
| Sim stop | `request_stop` clears a pending plan. At the next executor tick, the ordinary stopped path holds measured arm joints, retains the last gripper command unless another safety event requires release, and starts the observation tail of up to 2 s within the trial horizon. |

At the configured 125 Hz, successive rejects are 8 ms apart: if the first is at `t_hold`, the 25th requests the stop at **`t_hold + 0.192 s`**, and the next execution row reports stopped at **`t_hold + 0.200 s`**. This is a tick count, not a native wall-time deadline; real IK/RTDE delay can lengthen the native interval. The simulator's modeled inference delivery clock continues to advance through these executor ticks.

`ready_for_replan` admits no new inference while another plan is pending. A post-hold capture followed by 0.8–1.2 s inference/delivery cannot arrive within this 0.192 s streak. An already pending result can arrive sooner, but may use observations preceding the hold. An accepted plan alone does not clear the reject count: its next arm command must actually be selected. There is no current allowance for one fresh observation-based replan and no rejection-counter reset at submission. The 0.8–1.2 s range is a motivating observed planning scale, not a guaranteed latency bound.

## A hold is not necessarily an IK fault

Three bisection iterations test fractions down to 1/8; they do not test zero motion or prove that no smaller feasible step exists. A CPU counterexample with a smooth linear IK mapping, a feasible anchor, and a 1 rad/s limit returned `limiter_hold / joint_speed` after seven valid solves, although the untested 1/16 step was feasible at 0.625 rad/s. This is explicitly a synthetic algorithm counterexample, not a UR3 trajectory or simulator result.

For an actual v6 stall, inspect `violation`, valid/invalid IK outcomes, the preceding accepted anchor, measured q/qd and tracking, and the first hold's observation/request/delivery timestamps. `limiter_hold` alone cannot distinguish a kinematically valid constrained anchor, insufficient search resolution, unavailable IK, a previously violating anchor, or a physical tracking/contact problem. No arbitrary shorter-step search or timeout was introduced here.

## Minimal separate option if the diagnostic warrants it

Retain the existing default and all measured safety limits. A future opt-in native/sim setting could distinguish **verified constraint holds** from **IK faults**:

1. Permit the longer class only for an elbow/speed restriction with finite, branch-consistent joints and a verified current anchor inside the configured command envelope. Missing/invalid IK, branch jumps, invalid clocks/anchors, and an already violating anchor keep the current fault treatment. Measured safety checks remain active every executor tick; a valid command anchor does not certify safe contact forces or measured tracking.
2. Keep the arm anchor stationary, explicitly record a held outcome, and preserve existing gripper/latch/release behavior. Do not fabricate executed motion or reset the IK fault counter by reporting the rejected proposal as sent. Native code currently returns before calling `servoJ` on a reject; extending that gap is not a tested hardware hold. An opt-in implementation must deliberately stream the verified held joint target at the native servo cadence, distinguish it from accepted progress, and qualify control-script/watchdog behavior.
3. Use a separately declared elapsed-time deadline shared between native monotonic time and simulator time. Do not extend it indefinitely on every old plan, rejected plan, or zero-progress send. One illustrative budget is **2.5 s** for up to 1.2 s remaining on an older pending request, 1.2 s for a fresh request and 0.1 s scheduling; this is an unvalidated design budget, not a recommended hardware setting or a latency guarantee. A timeout remains an explicit controller stop, with no retry/reclassification as infrastructure.
4. Log hold onset, elapsed time, hold/fault counts, pending request/capture/delivery times, whether any post-onset observation produced a delivered plan, accepted motion and stop cause. CPU tests must cover no-plan timeout, a fresh plan arriving within budget, stale-only plans, invalid-IK escalation, measured safety preemption, and native/sim feedback parity. Hardware qualification additionally needs servo timing and unloaded holding evidence.

Changing only the number 25 would mix mechanical faults with intentional holds and could silently extend periods without native servo sends. The proposed distinction must therefore be declared and tested separately after the unchanged v6 diagnostic. It does not imply that a longer hold produces pickup, placement, lower forces, or safer hardware operation. The earlier rolling-wrist/contact-load limitation also remains unresolved.

Validation performed here: the existing real-adapter stall/grip test passed (`1 passed, 22 deselected`), confirming the 25th-reject/next-tick behavior; the synthetic search-resolution counterexample passed. These are CPU checks only.
