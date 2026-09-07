# Independent RPC-overlay review

Read-only review found no material timing-port defect in
`source_teacher_anchor_rpc_v7`. The actual adapter and runner hashes match the
[assembly manifest](source_manifest.json). The transplanted `replan` and
`_read_replan_clock` methods are AST-identical to the reviewed RPC donor. The
actual-output [CPU audit](cpu_delivery_audit.json) already verifies default
native trace parity, delayed head skipping/rebasing, native latency/grid/CPK
preservation, immutable proposals, expired chunks, stop cancellation, invalid
clocks and explicit CLI behavior. No additional model, GPU or hardware test was
performed for this review.

The first active v7 request independently confirms explicit `rpc_wall` mode,
limiter disabled, and correctly pinned external-driver files. Native latency
remains 1.263436583 s, the client call measures 1.269322420 s and the outer runner
measures 1.455924870 s. At request time 0.260 s, the returned native action grid
still begins at 1.523436583 s. Delivery instead waits until approximately
1.529322420 s, followed by executor quantization. This is a **partial first-plan
check**, not a result for the four-trial diagnostic. Exact identities and capture
time are in [initial_runtime_rpc_audit.json](initial_runtime_rpc_audit.json).

The directly measured client-minus-native duration is only 5.886 ms in that
sample. The 186.602 ms outer-minus-client duration includes work outside the
policy call, including snapshot/observation saving and adapter bookkeeping.
Earlier outer-minus-native trace estimates do not isolate RPC overhead and
must not be used as a transport measurement. This one sample also does not
establish typical overhead or explain a physical outcome.

Limits relevant to the four-case K1/K4 diagnostic:

- The timer measures the complete synchronous client policy call. Physics
  advances after that call and schedules the corresponding simulated delay;
  inference and physics do not advance concurrently in real time. Observation
  callbacks and post-call runner bookkeeping are excluded deliberately.
- Native action-grid, seed-selection and CPK timing still use native latency.
  The correction changes delivery only, with existing elapsed-head skipping.
  It does not retrospectively change selection to anticipate network delay.
- K4 is already batched. K1 removes beam selection and changes random-number
  draw shapes; a shared integer seed does not establish identical model noise.
  These two development seeds and reversed recipe order diagnose settings;
  they do not establish a model ranking or independent confirmation.
- Historical request-time terminal-veto feedback, the pad-only wrist proxy,
  uncalibrated gel mapping and the declared safety configuration remain.
  Longer delivery may make request-time veto evidence older. This is not a
  claim of current native hardware-controller equivalence.
- The v7 runtime intentionally changes only adapter and runner. Its copied old
  campaign/scorer scripts do not forward or audit RPC mode. The actual launch
  resolves this with the separate, pinned `source_teacher_anchor_driver_v7`:
  campaign SHA `235db3d485412da413c9abf866d22c0ff63522fdba70de65b6f7d5234b79aefa`,
  analyzer SHA `11b88baedfa6bed8696abb87fb30a51677be56538267c9fef3ddc60743d307a4`.
  The old v6 external driver must not be substituted.

The declared diagnostic has four ftA1500-only trials: K4 then K1 for seed
904301, followed by K1 then K4 for seed 904302. Both use NFE 1, EMA, guidance 1,
parity and persistent noise, task `waffles`, max-play 10, actual client-call
delivery, no latency override, and the standard v5 hardware limits with the
optional limiter off. No winner or pickup benefit is inferred here.
