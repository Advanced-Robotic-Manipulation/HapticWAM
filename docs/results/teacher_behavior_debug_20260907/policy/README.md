# Teacher proposal versus controller audit, 2026-09-07

The failed pickups contain upward commands; the failed carries do not contain
a release-sized opening in the executable prefix. Those are different failure modes. The historical veto
does not explain away the missing lifts or the arm's excessive carry reach.

This post-experiment, CPU-only audit reads **all 12 ftA1500 V5 confirmation
trials** and **all four V7 K1/K4 diagnostic trials**. It changes no source,
policy, recording, score, or safety threshold and performs no inference.
The two V7 seeds are development diagnostics, not an independent model ranking.
[Numeric evidence](audit.json) retains each input/primary-score SHA256, selected
proposal, delivery transform, replay exposure, release transition, and stop.

## What the controller actually changed

There are 552 original plans and 537 delivered plans. In **536/537 deliveries**,
all six raw pose-increment channels are bitwise identical after the terminal
veto. All lateral channels are unchanged in all 537. The sole pose rewrite is
the successful V5 seed 904510's `recovery_open` plan 29, delivered at 25.004 s:
it removes upward increments after physical placement was already confirmed at
24.268 s. Its full proposed chunk contains 120.456 mm positive Z increments;
only 2.517 mm of that rewritten ascent reaches the played part before FINISH.
It cannot explain earlier failures.

Gripper transforms are more frequent: V5 has 39 `close_masked` and one
`recovery_open` deliveries; V7 has 26 `close_masked` deliveries. A mask can
change an opening prefix merely because the same chunk has a later close.
Actual played opening at or below 0.45 is masked above 0.45 for 0.976 s in V5
904512 and 0.528 s in V7 K1 904302. Neither trial achieves a lift or arms the
loaded aperture latch. These changes are real, but they are not blocked
placement attempts by a carried object.

The audit independently reconstructs elapsed-head skipping, accepted-target
rebasing, pose integration, previous-chunk blending and the translation speed
clamp. It matches **39,264 active, non-stop, pre-FINISH requested XYZ targets**
with maximum error **1.24e-8 m**. Gripper commands reconstructed from delivered
action, physical closure bounds and logged latch have zero error. This checks
the action interpretation against the recorded execution rather than treating
a complete proposed chunk as an executed trajectory. Rotational channels are
checked for raw/delivered identity; their physical tracking is covered by the
separate kinematics audit.

| Cohort | Acquired | Lifted | Carried | Strict place | Loaded latch in every non-lift trial |
|---|---:|---:|---:|---:|---|
| V5 ftA1500 K4, 12 reserved seeds | 12 | 7 | 6 | 1 | Never armed |
| V7 K4, 2 development seeds | 2 | 2 | 2 | 1 | No non-lift trial |
| V7 K1, 2 development seeds | 2 | 0 | 0 | 0 | Never armed |

Here acquisition means the frozen scorer's bilateral packet contact above
0.1 N for 0.15 s. It is not the controller's loaded-grasp criterion: that uses
each pad's baseline-relative tactile load above 2.5 in the configured physical
force convention, taking the trailing 0.6 s maximum. The proxy's transfer to
real fingertip measurements remains unvalidated.

The five V5 non-lift seeds 904503/504/508/509/512 have maximum packet rises of
1.86/19.12/4.60/8.36/4.61 mm. Their active raw upward exposure after acquisition
is 426–660 mm, unchanged by the veto. V7 K1 seeds 904301/302 similarly expose
397/539 mm upward increments while the packet rises only 0.21/1.08 mm.
Requested TCP positive-Z travel, including blending and rate caps, is
492/613 mm in those two K1 cases. These are **cumulative positive travel**, not
net displacement, object rise, or an assertion that the entire chunk executes.
Missing lift therefore requires a contact/retention explanation; it is not a
missing upward command. The sensor audit can test whether the measured proxy
and the selected action's closure describe an adequately retained grasp.

## Why faster K1 does not provide an equivalent execution test

Median played action indices per plan are **2.48/1.76 for K1**, versus
**6.69/7.36 for K4**. Both use a ten-step cap, the same 10-Hz action grid,
uncertainty governor, and a 4-Hz replan request ceiling. Fast K1 returns before
much of a slowly played chunk executes. Stronger closure is often in its tail.

For K1 seed 904302, plan 33 requests at 10.236 s and proposes closure
`[.485, .488, .525, .535, .545, .541, .578, .569, .592, .588]`.
The veto explicitly allows the close. With maximum sigma 0.637, the governor
and next replan leave only 1.31 actions played: command .485–.488, measured
closure .385–.389. Plan 39 likewise proposes a .603 tail but executes 1.16
actions at approximately .504. By plan 55 it reaches about 2.25 actions and
commands up to .555. This is **observed repeated truncation of stronger closure**,
not evidence that replaying the tail would produce a safe successful grasp.

The native CPK branch also changes the effective cadence. After logged CPK
invalidation, K1 median native latency is .604/.605 s; with a retained package
it is .214/.215 s. `rf.py:506–541` skips the two-pass anticipation when a prior
package exists; invalidation deliberately removes that conditioning. It
therefore changes both computation and input state. The sampler's persistent
outer noise does not mean that a newly computed inner anticipation uses a
fixed draw. None of the V7 cases has a stale hold or playback-cap dwell.

## K4 selection: measured facts and unresolved weaknesses

Native [policy selection](../../../../phantom/inference/policy.py) uses
`_select_seed` at lines 196–234. It first screens nine-step descent when ACC
contact probability is below .5, then minimizes previous-accepted-plan L2
distance over six **metre and rotation-vector-radian** channels. It neither
scores candidate grip retention nor checks joint reach, obstacle clearance or
placement feasibility. K4 is already one batch of four denoising trajectories
(`phantom/model/rf.py:496–576`), not four sequential RPC calls.

The descent screen rejects candidates in 24/256 V5 calls and 2/47 V7 K4 calls.
The selected row is nonzero in 166/256 and 41/47 calls, respectively. For the
selected candidate against its previous reference, rotation contributes median
81.1%/81.8% of the unweighted mixed-unit squared distance. This demonstrates
the numerical weighting actually used; it does not prove another candidate
was physically better. Full nonselected pose/gripper arrays were not saved.

The previous-plan reference offset is clipped to 0–15 even when a future
chunk has no true temporal overlap. The [three CPU source checks](selector_cpu_checks.json)
reproduce this on a synthetic expired plan, together with descent screening
and unit weighting. **There are no no-overlap occurrences in these 16 NFE1
trials**; maximum raw offset is 12. It is a real code limitation, not an
explanation of these observed failures.

CPK index is reconstructed, not logged: backbone temporal compression 4 and
fps 4 give a 1-s latent period. The native index is
`round(previous_accepted_plan.latency_s / 1 s)`: K4 uses 1; K1 uses 0 or 1.
An invalidated package makes that index inoperative. Native ACC inputs are
identical across K rows, so using row-zero ACC contact probability is not a
demonstrated per-candidate probability bug. Candidate contact readouts are
different tensors. Same sampling seed across K1/K4 also does not imply the
same complete noise tensors or previous CPK history.

## Release and the smallest useful next tests

All six failed V5 lifted trials and the failed V7 K4 carry have **zero played
raw opening commands <=.45 after lift**. The latch does retain slightly lower
still-closed commands, as designed, but it does not suppress an attempted
placement release in this group. Preserve the guards; relaxing the latch or
FINISH criteria would not address those stops. The successful V5/V7 cases do
command sustained opening, commit release at 23.580/23.300 s, satisfy physical
placement at 24.268/24.000 s, and FINISH at 25.076/24.172 s. Safety continues
through the full 60-s observation horizon.

This is also true of their unplayed but executable first ten samples in every
post-lift proposed plan: minimum closure is .469–.564 across the seven failed
lifted cases. Partial opening toward those values occurs; calling it a complete
release intent would require a different, explicitly validated aperture rule.

1. **Address executed closure, with moderate causal confidence.** First replay
   a few saved K1 pickup plans through the CPU executor at the logged governor
   and a fixed, predeclared slower replacement cadence; report the closure
   samples reached without invoking a model. If that confirms the intended
   extra closure, a small matched closed-loop cadence test with measured full
   client timing can distinguish truncation from contact geometry. Keep safety,
   force limits and actual sensor freshness unchanged. Native CPK indexing
   depends on latency rather than the new request interval; record that
   limitation instead of calling the test input-identical. For a durable model
   fix, train/evaluate the executable prefix at deployment cadence, including
   closure and failed-retention examples, rather than relying on unplayed tails.
2. **Improve candidate feasibility before changing model weights.** On saved
   inputs, log all K pose/gripper trajectories and separate translation,
   rotation, descent and true-overlap scores. Evaluate accumulated candidate
   motion from the actual accepted reference with UR3 reach/joint guards;
   compare explicit physical or training-normalized metric weights. Rejecting
   an unsafe candidate must preserve the final safety veto and must not depend
   on simulator object truth. The current logs cannot identify a winning
   counterfactual candidate, so benefit is unproven. Fix the expired-overlap
   fallback separately, with the included CPU case, without claiming it fixes
   the present NFE1 failures.
3. **Treat transport/release as a policy-phase problem after confirming reach.**
   Use the independent kinematic audit to determine whether the intended bin
   approach is feasible at the current orientation. The teacher currently
   reaches a safety boundary before proposing release in the failed carries.
   A bounded saved-input test with verified feasible demonstrations and matched
   contact observations can distinguish unsafe carry continuation from poor
   scene/contact transfer. Do not force an opening, add an object oracle, or
   label successful FINISH alone as successful placement.

## Reproduce without model or hardware access

Run from the repository on compute3 with NumPy available. Both helpers only
read existing data/source and print new diagnostic JSON:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 docs/results/teacher_behavior_debug_20260907/policy/audit_policy.py > /tmp/policy_audit_recheck.json
PYTHONDONTWRITEBYTECODE=1 python3 docs/results/teacher_behavior_debug_20260907/policy/audit_selector.py --source /home/physicalai/phantom-icra-2027/phantom/phantom/inference/policy.py > /tmp/selector_recheck.json
```

The selector helper enforces the server-recorded source hash. The trace helper
checks primary input hashes before analysis, preserves every denominator, and
fails if reconstructed execution does not agree with the logged commands.
`played_*` values integrate the active new plan's consumed samples before
blend weighting; `requested_*` values integrate actual consecutive requested
targets. Neither is a sum of all overlapping proposed horizons. All units
remain simulator/declared controller units, without a force-calibration claim.
