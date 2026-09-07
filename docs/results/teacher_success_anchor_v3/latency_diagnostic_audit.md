# Historical adapter-delay diagnostic

The completed seed-4242 diagnostic acquired, lifted and carried the free packet,
then stopped on `wrist_extension` at **17.348 s** while holding it. It did not
release or place the packet and did not drop it. The independent frozen scorer
accepts the trace as valid. This single controlled diagnostic is excluded from
all model-selection and confirmation denominators.

| Observation | Historical anchor | Recorded adapter-delay run |
|---|---:|---:|
| Acquisition | 8.000 s | 8.000 s |
| Lift | 12.668 s | 12.668 s |
| Carry | 15.336 s | 15.336 s |
| Strict placement | 25.200 s | None |
| Maximum lift | 341.763 mm | 333.034 mm |
| Actual final sample | 33.376 s | 19.344 s |

The requested horizon was 33.38 s; the actual safety stop ended it after the
unchanged 1.996 s sampled observation tail. There were no IK rejects, stale-plan
holds or inference errors. The final packet remained loaded and outside the bin;
maximum positive bin contact was zero. Contact magnitudes remain simulator
quantities, not calibrated physical measurements.

Exactly **21 of 38** historical per-request delays were consumed before that
stop. Every consumed delay, request timestamp and full 16-step action grid
matches the original exactly. All 20 activated plans also activated at exactly
the original times. Plan 20 remained pending at the stop. No schedule exhaustion,
synthetic tail, delay padding or extra request occurred. The authoritative
`rollout/anchor_latency_schedule.json` contains the actual used rows;
`managed.json.schedule.used=[]` is the preserved prelaunch template, not missing
runtime evidence.

K4 first selects a different candidate at replan **14**, request **12.156 s**:
historical K0 versus diagnostic K3. The requested CPK offset is unchanged at
every matched request. The native selection-reference index does differ at
replan 8 (8 to 9), because native K4 still computes that
index from fresh inference latency before the adapter override. That request retains
the historical K0 choice. Requested CPK offset alone does
not prove that a package survived a veto.

The initial observation has eight bitwise-identical non-RGB fields. RGB differs
by RMS 0.917832 uint8 levels (maximum 31). The first selected K is still K0, while
the first action array differs: the 10-step XYZ endpoint changes 0.146709 mm and
closure RMSE is 0.000397276. These proposals precede later contact and selection
differences. The separate saved-input experiment establishes first-proposal
RGB sensitivity; this latency diagnostic does not isolate RGB throughout a
closed-loop episode.

Effective scene and initialization records match the historical run. Recorded
inputs, ftA1500 EMA identity, inference settings and frozen source hashes are
retained in [latency_diagnostic_audit.json](latency_diagnostic_audit.json). The
reviewed overlay changes only the runner timing hook and adds the schedule
reader. Owned server PID 2250771 and Isaac PID 2252121 are absent; both saved
exit codes are zero, server status is `stopped`, and cleanup reports no errors
or foreign process signals.

This result demonstrates that matching adapter request/delivery/action-grid
timing alone did not restore historical placement. It cannot exclude native
K4 timing or subsequent observation differences, and it does not compare model
quality. Recompute the analysis without GPU or runtime writes by piping
[audit_latency_diagnostic.py](audit_latency_diagnostic.py) into the live repo's
Python on compute3; the script reads the completed raw directories and prints
JSON to stdout. Its frozen scorer SHA256 is
`b8460fc530acd6f11ea166f3fbff0ddc2ee74f4c73d956752abab0435975969f`.
