# Corrected study: first recorded arm start

All eight completed trials for `start_1787395928` are valid under the frozen object-state scorer. Six end with a native `wrench_limit` stop; two reach the 60-second horizon. None acquires, lifts, carries, or releases the packet. These are descriptive results for one start, **not a ranking or a substitute for the full corrected study**. No superseded screen runs are pooled here.

[Machine-readable audit](audit.json) · [rendered snapshots](contact_sheet.jpg) · [frame provenance](frames_manifest.json)

## What contacted what

Actor identity comes from explicit PhysX actor/filter buffers. The force column below is the largest sum of contact normal-force magnitudes for that actor/body pair before the stop, sampled at 125 Hz. It is not a calibrated real wrist measurement, active gel pressure, or necessarily the force at the stop.

| Teacher runtime | Seed | Outcome/time | Loaded modeled pair | Pair peak (N) | First additional close: time / reach error |
|---|---:|---|---|---:|---|
| ftA1500 NFE1/K4 | 903101 | Horizon, 60 s | Right finger / mat; later packet | 31.94 mat; 1.09 packet | 12.428 s / 56.2 mm |
| ftA1500 NFE1/K4 | 903102 | Wrench stop, 5.380 s | Housing / bin front | 174.32 | None |
| ftA3000 NFE1/K4 | 903101 | Horizon, 60 s | Right finger / mat; later packet | 67.01 mat; 0.71 packet | 12.548 s / 64.1 mm |
| ftA3000 NFE1/K4 | 903102 | Wrench stop, 5.836 s | Housing / bin front | 126.29 | None |
| v5_6 NFE1/K4 | 903101 | Wrench stop, 7.372 s | Right finger / mat and bench | 117.69 mat + 77.47 bench | None |
| v5_6 NFE1/K4 | 903102 | Wrench stop, 5.636 s | Left finger / bin front | 128.28 | None |
| ftA1500 NFE5/K1 | 903101 | Wrench stop, 5.436 s | Left finger / bin front | 158.49 | 4.316 s / 216.7 mm |
| ftA1500 NFE5/K1 | 903102 | Wrench stop, 7.092 s | Right finger / packet | 129.21 | None |

The housing contact in ftA1500/903102 acts upward on the bin rim: at the stop, the loaded point is approximately `[-0.30330, -0.08054, 0.19400] m`, normal `[0,0,1]`. For ftA3000/903102 the contact is against the front wall near its upper edge, with normal `[0,1,0]`. Thus these are distinguishable modeled rim/wall collisions, not inference from a net force vector. Left-finger/bin contacts have oblique upward normals and occur on the finger edge rather than a valid inward gel face.

The v5_6/903101 load is shared by the mat and underlying bench. Logged penetrations reach about 5.7 mm against the mat and 2.7 mm against the bench. For ftA1500 NFE5/903102, the right finger presses the packet downward: the reaction on the finger is nearly upward, and loaded signed separations reach −13.4 mm at the stop. These large penetrations expose the limits of the compliant contact model; they do not establish realistic packet deformation or physical force magnitude. The earlier [time-step sensitivity audit](../../teacher_v2_design/command_force_convergence_audit.md) remains relevant.

## Causal timing and safety input

The independently reconstructed native wrench subguard matches all six stop times exactly and predicts no wrench stop for the two horizon cases. It subtracts a calm-only rolling baseline with a 2-second time constant, freezes that baseline while over limit, and requires 0.3 seconds above 60 N or 15 N·m. The debounce is three **10 Hz action-rate ticks**, even though inputs are checked at 125 Hz.

| Teacher / seed | First threshold crossing (s) | Logged stop (s) | Force deviation at stop (N) | First tracking error >20 mm (s) | Maximum tracking error before stop (mm) |
|---|---:|---:|---:|---:|---:|
| ftA1500 K4 / 903102 | 5.076 | 5.380 | 168.63 | 5.284 | 28.74 |
| ftA3000 K4 / 903102 | 5.532 | 5.836 | 125.68 | — | 9.00 |
| v5_6 K4 / 903101 | 7.068 | 7.372 | 191.99 | — | 9.90 |
| v5_6 K4 / 903102 | 5.332 | 5.636 | 125.23 | 5.548 | 26.57 |
| ftA1500 NFE5 / 903101 | 5.132 | 5.436 | 154.19 | 5.380 | 23.83 |
| ftA1500 NFE5 / 903102 | 6.788 | 7.092 | 125.51 | — | 8.02 |

Tracking error is accepted-target versus measured TCP position, excluding stopped hold rows. Every accepted/requested positional difference is below 0.101 mm and no IK rejects are reported. The two horizon cases remain within 3.13/4.54 mm tracking error. Therefore a failed reach is not generally an IK rejection or large drive-tracking failure. Where tracking exceeds 20 mm, the recorded collision and threshold crossing precede it. This temporal association supports physical obstruction in the simulated geometry without proving how the real apparatus would respond.

All recorded executor wrist values exactly match their causal cached contact samples. Recomputed signed impulse divided by physics step, summed contact forces, and point-to-TCP moments have zero numerical discrepancy in this audit. Initial wrist bias is exactly `[8.276634, 13.145563, -19.001765, -4.314573, 4.556973, 1.136910]` in N/N·m, with force norm **24.543 N**. The policy receives this raw bias-added input; safety evaluates deviation from its rolling baseline. The ftA3000/903101 mat peak of 67 N does not itself imply a missed 60 N trip: its largest baseline-relative deviation is only 42.78 N.

## Reach, gel contact, and limits

All eight start at measured closure 0.4274, already above the frozen absolute reach threshold of 0.25. The alternative diagnostic uses the first command exceeding initial measured closure by `2/255` and the last prior 15 Hz pad-midpoint distance to the packet's oriented box. This is not fingertip clearance or the first increase after any later reopening. A null value means that additional-close threshold was never crossed, not that the arm never moved or the fingers never opened.

The two horizon cases close at 56.2/64.1 mm under this diagnostic, later push the packet by 24.9/27.6 mm, and never achieve bilateral acquisition. Their minimum midpoint distances are 31.0/33.4 mm. Across the block the largest sampled packet height rise is below 0.05 mm, far below the fixed 30 mm sustained-lift gate.

All logged active-gel compression is zero and no grip latch is observed. The loaded finger contacts are rejected because they are outside the inward face and/or have non-compressive face normals. In the packet-loading NFE5/903102 case, local contact X is approximately −10.84/−5.84 mm, while the right inner face is +6 mm; normal X is only about +0.125. This is a toe/backing contact, not the sparse manifold interior-pressure issue addressed by mapper v2. The mapper records the ignored load; it does not turn arbitrary finger-body force into gel compression.

The wrist proxy covers the housing and two finger bodies against the eight explicit environment bodies. It omits proximal-arm and self contacts, friction, gravity, inertia, and a calibrated current-based UR3 transfer model. Collision geometry, compliance and force magnitude remain hypotheses. Correct sensor arithmetic is narrower evidence than physical sensor calibration. No thresholds, geometry, policy settings or source files were changed by this audit.

## Reproduce

Run the CPU audit against the completed immutable source/data on compute3:

```bash
ssh compute3 'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python -' \
  < audit_first_start.py > audit.json
```

The JSON records the corrected campaign SHA256 `5ea36cfb5404b6e20461a251beb1173c9e0ee809e279dfca4733e6acc502bda0`, frozen source/hardware hashes, each input hash, exact contact records, native-subguard reconstruction and unchanged scorer outputs. The [extractor](extract_frames.py) reads only the eight completed existing videos; [montage builder](make_contact_sheet.py) preserves the extracted JPEGs and adds labels to the overview. Frame timestamps are the last rendered samples at or before the stated event; force identification comes from logs, including when visual geometry is occluded.

The separate [completed-study classifier](classify_completed_study.py) is prepared for the final 32 screening plus 24 confirmation cases. It accepts `--study-root`, `--screen-root`, and `--confirmation-root`, and refuses to analyze before `study_complete.json`, both `all_trials_completed` states, and the exact 32/24 identity sets exist. Keep it beside `audit_first_start.py` on compute3 and redirect JSON to a new diagnostic artifact outside raw trial folders. It adds contact identity at the first stop, first instrumented contact, precontact tracking and stage/closure/latch summaries without ranking candidates or changing scores. It has not been run against the unfinished study, and no automatic waiter is running.
