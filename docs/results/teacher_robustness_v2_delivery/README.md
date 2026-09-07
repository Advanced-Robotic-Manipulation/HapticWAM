# Fixed-waffle teacher comparison — completed corrected study

**All 56 planned trials completed and passed the frozen validity checks. No teacher qualified for final pick-and-place experiments.** Reserved-start confirmation gives both selected teachers 3/12 acquisitions, 1/12 sustained lifts and carries, and 0/12 placements. The declared conclusion is **inconclusive; pickup candidates only**. These are exploratory simulator results: camera/geometry and tactile/wrist transfer remain uncalibrated, so they do not rank physical-hardware performance.

## Reserved confirmation: the decision set

| Teacher / recipe | Trials | Acquisition | Sustained lift | Carry | Strict / clean placement | Safety stops | Wrench stops |
|---|---:|---:|---:|---:|---:|---:|---:|
| ftA3000, NFE 1 / K 4 | 12 | 3 | 1 | 1 | 0 / 0 | 9 | 7 |
| v5_6, NFE 1 / K 4 | 12 | 3 | 1 | 1 | 0 / 0 | 12 | 5 |

There were no scored drops or tactile force/depth stops in confirmation. Fewer total safety stops for ftA3000 do not establish a safer winner: it had more wrist-limit stops. Both fail the required 8/12 clean-placement gate. The paired clean-placement difference is zero; its 95% start-cluster bootstrap interval is the degenerate [0, 0] because every outcome is zero. **That interval does not establish equivalence or bound hardware success.** Six starts are resampled with both matched seeds intact (10,000 draws, seed 20260906).

The two lift/carry events occur at different starts: ftA3000 at 6314/seed903201, v5_6 at 6060/seed903201. Both terminate at the unchanged wrist-extension limit while still holding. They remain 92–122 mm above the release ceiling; neither enters the release volume after latching or plays an eligible opening. Nominal in-volume IK poses exist within the extension limit, so the evidence does not establish a globally unreachable box. Their actual transfer paths and release timing fail; the IK examples do not validate collision-free paths. See the [terminal geometry audit](failure_summary.md#the-two-carries-stop-before-entering-the-release-volume). A successful low-level stage is not an end-to-end success. See [confirmation selection and exact gates](confirmation/selection.json), [all24 trial rows](confirmation/trials.csv), [confirmation report](confirmation/report.md), and the [independent final audit](final_audit.md).

## Discovery screen: selection only

| Teacher / recipe | Trials | Acquisition | Lift / strict / clean placement | Safety stops |
|---|---:|---:|---:|---:|
| ftA1500, NFE 1 / K 4 | 8 | 0 | 0 / 0 / 0 | 4 |
| ftA3000, NFE 1 / K 4 | 8 | 1 | 0 / 0 / 0 | 6 |
| v5_6, NFE 1 / K 4 | 8 | 2 | 0 / 0 / 0 | 8 |
| ftA1500, NFE 5 / K 1 | 8 | 1 | 0 / 0 / 0 | 7 |

The NFE5/K1 recipe also has one non-safety `veto_retry_cap` stop. Its comparison with NFE1/K4 changes two settings; it cannot identify an NFE effect. The frozen ranking advanced v5_6 and ftA3000. Discovery counts are selection-biased and are not pooled with confirmation to declare a winner. The [screen rows](screen/trials.csv), [selection](screen/selection.json), and [independent screen/design audit](screen_confirmation_audit.md) preserve every case.

## What this design tests

Collection kept the waffle approximately fixed while arm starts and trajectories varied. This experiment therefore fixes one canonical green packet, camera, materials and lighting, and uses ten authentic synchronized **arm/gripper/wrist starts**. Four starts and two seeds per candidate form the 32-trial screen; six reserved starts and two fresh seeds form the 24-trial confirmation. Recorded trajectories are not imposed on policy control. Gripper aperture and wrist bias covary with arm pose, so this is not an isolated joint-position perturbation.

Only ten recordings had complete native timestamped robot, camera and tactile evidence; the other240 mirrors could not support equivalent initialization. This cohort covers a limited part of the collection distribution. Reserved starts are simulator-study holdouts, not proven training-unseen examples. Initial measured velocities are recorded but deployment starts are physically settled; all ten initialization checks passed.

All candidates use their own saved teacher architecture and EMA, audited normalization, task text `waffles`, guidance 1, persistent noise, parity, live veto and 10-step playback. The controller and success thresholds are fixed before scoring. The [protocol](../../../configs/sim/teacher_v2_delivery_protocol.json), [execution guide](../../isaac_teacher_v2_experiment.md) and exact [derived confirmation configuration](teacher_v2_delivery_confirmation.json) define the complete design. Native inference latency is measured under concurrent Isaac/teacher GPU use: confirmation event-weighted means are 0.851 s for ftA3000 and 0.809 s for v5_6. These are distinct from wall-call and activation latency and are not a hardware-only benchmark; [all latency summaries](final_summary.json) retain those distinctions.

## Diagnosed weaknesses and fixes

The [four screening acquisition traces](acquisition_diagnostic/README.md) separate weak retention before the main upward command from continued pressing with backing/table contact. Three do not arm the native bilateral 2.5 N latch; one arms it late but still does not lift. Maximum screening packet-center rise is 17.52 mm, below the unchanged 30 mm/0.5 s lift gate. Lowering the latch threshold alone would not fix the pressing cases. Pad/TCP geometry, aperture and calibrated load transfer are needed to distinguish placement errors from misleading proxy loads.

The [first-start audit](first_start_diagnostic/README.md) reconstructs housing/box, pad/environment and pad/packet contacts and two horizon misses. The [all 56 failure classification](failure_summary.md) retains contact, tracking, reach, latch, veto and stop evidence for the whole study. Some wrist stops follow unloading from a shifted baseline; they must not all be described as collisions at the stop instant.

Corrections implemented **before** this study include gel manifold coverage, external wrist contacts on the housing and both pads, current measured delivery feedback for the release-enabled terminal veto, and shared policy-commanded release/FINISH behavior. Physical task success remains independently scored from object state. The ten earlier v2 trials remain superseded diagnostics and are excluded. The several simultaneous changes prevent a paired estimate of any one fix's effect.

A further rolling-baseline flaw was found during diagnosis: gradually increasing load can enter its reference. The separately published [experimental fixed-reference patch, PR13](https://github.com/Advanced-Robotic-Manipulation/phantom/pull/13) detects the saved gradual overload at 10.460 s instead of 17.972 s and preserves default behavior. It is **not in this frozen study**. Its saved-input replay is not a new pickup result; one additional later trigger in ten native demonstrations remains unadjudicated. Physical wrist-bias qualification is required before enabling that mode.

The separate [six-call RGB diagnostic](../teacher_v2_design/rgb_transfer_diagnostic.md) completed with ftA1500. Swapping only the initial rendered RGB for an aligned real frame changes the proposed 10-step endpoint by 2.206/4.953 mm across two seeds; repeated rendered controls are exactly equal. This tests one input, not later perception or task performance. It does not isolate the cause of the closed-loop failures. Two failed diagnostic launches produced no valid proposals and remain preserved separately; neither affects the 56 scored trials.

## Reproduction, video and laboratory use

[Watch the synchronized 60-second tactile comparison](media/paired_pickup_attempts_60s.mp4) · [14-second still](media/paired_pickup_14s.png) · [video audit](video_audit.json) · [all 56 video locations and hashes](video_manifest.json).

The four-panel comparison shows ftA3000 on the left and v5_6 on the right, with start 6060 on the top row and start 6314 below, all seed 903201. These are both confirmation lift cases plus their matched counterparts. Each panel shows saved supplied tactile pixels and force/state traces; stopped traces explicitly freeze at their own end time. The choice illustrates all observed lifts, not a success-rate sample. All 56 individual videos passed full-frame decoding and causal-map checks (23,006 decoded frames); the montage has 901 frames at 15 fps, including t=60s.

Executed source: `/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_v2_delivery`. Raw results: sibling `runs/teacher_robustness_v2_delivery/{screen,confirmation}/rollouts/`. [Execution arguments](execution_plan.json), [completion marker](study_complete.json), [step exit ledger](study_ledger.jsonl), [frozen inputs](prelaunch_frozen_inputs.json), [compact final results](final_summary.json), and [published-source audit](published_source_audit.json) are retained. The source archive SHA256 is `5f4a0cf7027d00a1463c53e61e17e73919bc9453a5be1b733554672692be4f07`. All 105 executed files match the published executable code; one differs only in a corrected historical comment. Original recordings remain unchanged.

At the corrected freeze, 366 simulator tests passed; eight RGB-helper and 12 existing FakeRTDE tests also passed. Exact-command replay was bitwise invariant with added sensing, and force/moment/cache audits reconstructed exactly. Three teacher payloads passed [preprocessing checks](../teacher_v2_design/preprocessing_audit.md). These validate implementation, not physical force/contact calibration.

For the lab, retain **ftA3000 and v5_6, EMA, NFE1/K4 as paired diagnostic pickup candidates**. Neither is approved by these results as a final placement configuration. Begin with the [one-page lab card](../../isaac_teacher_lab_card_20260907.md), [teacher handoff](../../isaac_lab_handoff_20260907.md) and [measurement sheet](../../isaac_teacher_measurements_20260907.md). Measure base/table, pad/TCP and backing geometry, actual aperture, packet/box dimensions, camera alignment and unloaded/known-load sensor response. Validate measured demonstration motion in the resulting new scene revision before another ranking experiment.

Keep waffle placement marked and fixed for that next comparison. Reuse these cases only as regression checks after corrections, and reserve new measured starts/seeds for fresh confirmation. Measure actual reset jitter before making packet placement a separate sensitivity block; arbitrary joint-plus-object randomization would obscure the current failure causes.
