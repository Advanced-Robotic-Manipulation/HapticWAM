# Completed-study failure diagnosis

All **56 corrected trials are valid**. The independent classifier waited for `study_complete.json`, checked the exact 32 screen and 24 confirmation identities, and then read completed traces on CPU. Object-stage counts agree with the authoritative [final summary](final_summary.json). No scores, thresholds, source, scene or controller were changed; superseded trials are excluded. The phases describe different starts and candidates, so the combined counts below describe failure patterns, not a model ranking or an independent 56-trial success estimate.

| Observed outcome | Screen, 32 | Confirmation, 24 |
| --- | ---: | ---: |
| Acquisition | 4 | 6 |
| Sustained lift / carry | 0 / 0 | 2 / 2 |
| Release in bin / full task | 0 / 0 | 0 / 0 |
| Observed native load latch | 1 | 2 |
| Safety stop | 25 | 21 |
| Veto retry cap | 1 | 0 |
| Reached 60 s horizon without a stop | 6 | 3 |

No scored drop occurred. That does not establish reliable retention: only two trials reached sustained lift, and both stopped while holding the packet.

## Stops and contact evidence

The 46 safety stops comprise 34 `wrench_limit`, four `wrist_extension`, four `hitbox_exit`, three `joint_speed`, and one `tactile_depth`. One joint-speed stop also logs `workspace_clamp`; it is a co-occurring event, not another termination. The veto retry cap is a separate controller termination.

The unchanged rolling wrist subguard was reconstructed from actual executor wrist inputs: 60 N / 15 Nm, 0.3 s debounce, and a 2 s baseline time constant. Wrench-stop presence and first stop time match the logs in **all 56 cases**. Of 34 wrench stops, 33 have substantial instrumented distal contact at the stop sample: summed positive pair magnitudes range from 76.86 to 214.77 N. Their largest contact pairs are housing–bin front (14), right pad–mat (8), pad–bin front (5), right pad–packet (5), and housing–mat (1). These are observed associations, not proof that the largest pair alone caused the stop; the guard acts on the net six-component wrench relative to its evolving baseline.

One wrench stop is distinct: screen `v5_6_nfe1_k4__start_1787396273__seed 903101` stops at 17.972 s after unloading. Only 0.094 N of distal contact remains, while the baseline retains the preceding load. Earlier contacts exceeded 100 N. The [force-and-baseline plot](acquisition_diagnostic/wrist_baseline_drift.png) and [acquisition audit](acquisition_diagnostic/README.md) show gradual load absorption followed by a large deviation on unloading. This is neither a benign-contact finding nor a revised safety verdict. The conservative flag requires a wrench stop, current summed contact below 0.1 N, and baseline force displacement above 60 N; exactly one of 56 cases meets it. Other baseline effects may exist below that diagnostic threshold.

## Reach, tracking and acquisition

Every initial closure already exceeds the frozen absolute reach threshold of 0.25. Consequently, the absolute reach-before-closure outcome is false throughout and cannot establish that the robot never approached the packet. The separate first-additional-closing diagnostic uses a command exceeding initial measured closure by `2/255`. It exists in 43 trials: midpoint-to-object OBB distance at the preceding scene sample has median 216.1 mm; four are within 20 mm. Thirteen have no such additional closing command. This measures the first increase beyond the initial closure, including any early command before a later reopening; it is not the distance at every subsequent grasp attempt. Twenty of 56 trials eventually bring the modeled pad midpoint within 20 mm of the object.

Before the first instrumented distal actor/environment pair exceeds 0.1 N, maximum accepted-versus-measured TCP position error is at most 15.54 mm in every trial (median of per-trial maxima 3.66 mm). Over the whole active execution, 15 trials exceed 20 mm and the largest error is 105.90 mm. No IK rejection is logged. These observations distinguish tracking before observed contact from later blocking, but do not establish collision-free proximal-arm motion or diagnose policy perception. The [four screen acquisitions](acquisition_diagnostic/README.md) also distinguish brief physical bilateral contact from sufficient active-gel loading and later upward commands.

## The two carries stop before entering the release volume

The release TCP box is `[-0.542928,-0.068538,0.058]` to `[-0.240206,0.141937,0.274]` m. Opening requires an eligible policy closure at or below 0.45 for 0.2 s after the native load latch is armed. **Neither carried trial has any post-latch sample inside this box, an active release window, or an eligible played opening at or below 0.45.** Both remain in `holding`; no release is committed or finished.

| Terminal quantity | v5_6, start6060, seed903201 | ftA3000, start6314, seed903201 |
| --- | --- | --- |
| Acquisition / lift / carry, s | 10.600 / 12.600 / 14.268 | 26.868 / 31.736 / 34.536 |
| First wrist-extension stop, s | 16.428 | 36.124 |
| Reconstructed wrist radius, m | 0.468013613 | 0.468028429 |
| Guard limit, m | 0.468 | 0.468 |
| TCP above release ceiling, mm | 122.48 | 92.12 |
| Other TCP exclusion | Y is 23.72 mm below box | XY is inside box |
| Measured / commanded closure | 0.546721 / 0.608958 | 0.520767 / 0.610785 |
| Minimum eligible played closure after latch | 0.506209 | 0.520352 |
| Maximum packet rise / carry distance, m | 0.346873 / 0.230868 | 0.300462 / 0.264157 |

Exact measured/requested TCP poses are xyz metres followed by rotation-vector radians:

```text
v5_6 measured  [-0.489757031,-0.092260672,0.396476749,-1.312486946,-1.169064734,0.839907482]
v5_6 requested [-0.489583239,-0.086559812,0.396557334,-1.327162850,-1.148627057,0.827080431]
ftA3000 measured  [-0.423034469,-0.010170853,0.366119637,-1.615108491,-1.231261649,0.685042690]
ftA3000 requested [-0.422522121,-0.005075934,0.366530392,-1.627329127,-1.216992767,0.679929686]
```

The guard radius is computed from the recorded elbow angle and the frozen UR3 dimensions; it independently matches the shoulder-to-DH-frame-4 distance. The physical bin rim is at z=0.194 m. The nearest scene samples show both packets above the rim; only the second packet has all projected corners inside the bin interior. Sampling offsets are 28 ms before and 12 ms after the respective executor stops.

Nominal CPU IK, holding each terminal gripper orientation fixed, converges at the nearest release-box point with wrist radii 0.371987 and 0.393877 m. Examples over the bin center at z=0.274 m also converge below the radius limit. Thus these traces do **not** establish that the box is globally unreachable under this guard: the actual paths reach extension before returning into the release volume, and no in-volume opening is observed. These isolated pose solutions do not validate a safe trajectory, collision clearance, IK branch transition or retained grasp. The nearest box point for the first carry would still leave part of the packet outside the bin footprint; TCP eligibility alone is not placement success.

The [additional raw-proposal audit](carried_raw_opening_audit.json) checks the distinction between model intent and played commands. In all archived heads captured after the first latch, v5_6 has 0/80 and ftA3000 has 0/112 original predicted closures at or below 0.45, including 16-row tails and one final unactivated head per case. Minima are 0.49818 and 0.53016. Delivered post-latch heads show no closure changes from veto filtering. Thus these archived post-latch predictions contain no qualifying opening hidden only by veto or playback truncation; behavior after termination is unknown.

## Reproducibility and limits

[Per-case classification](final_failure_classification.json) contains all 56 identities, 11 input hashes per case, source/hardware hashes, timings, contact pairs and guard reconstructions. Its SHA256 is `3b0b55eb80b14c599f842eececb8715237bd7dabad9bee97f73ab910c04c6cad`. The guarded [classifier](first_start_diagnostic/classify_completed_study.py) ran once after completion. [Terminal geometry JSON](carried_terminal_geometry.json) and its [CPU helper](audit_carried_terminal_geometry.py) provide exact poses, dimensions, played-opening checks, IK residuals and input hashes.

The wrist proxy covers housing and two pads using normal-contact forces and point moments. It omits friction, inertia, gravity, proximal-arm/self contacts and the CB3 estimator's calibration. Rigid packet/contact geometry and force magnitudes remain uncertain and timestep-sensitive. The evidence supports specific simulated failure mechanisms and measurements to collect next; it does not establish real force accuracy, a hardware success rate or a safe replacement trajectory.
