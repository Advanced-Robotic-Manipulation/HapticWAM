# Two-student nominal simulation check

Both students loaded and executed through the deployment interface, but **neither acquired, lifted or placed the packet in either seed**. All four trials are valid recorded policy failures: no missing cases, infrastructure retries, IK rejects, stale holds or inference errors. This is a small compatibility check, not a reliability estimate or a ranking between students.

The frozen design tests revised ftA `student_ftA_r1/student_001200.pt` and `student_v5_6/student_001200.pt` at the same nominal arm/object configuration with seeds 4242/4243. Each uses EMA, NFE 1/K 4, guidance 1, persistent noise, parity, task text `waffles`, the existing safety/latch/placement-release controller and strict object-state/support criteria. The planned horizon is 60 s; actual safety stops end action execution, followed by up to 2 s of physics observation. The previous teacher campaign remains separate.

This smoke uses the frozen historical `fd4a032` veto profile plus the simulation placement-release extension; tomorrow's operator-run native pilot uses the live veto, so this is not a test of an identical current hardware executor.

| Student / seed | Reach diagnostic | Acquisition / lift / place | Terminal cause | Stop time | Mean native inference | Peak pad–packet normal force |
|---|---|---|---|---:|---:|---:|
| Revised ftA /4242 | Yes | No / No / No | Wrench limit |7.596s|0.672s|109.4N|
| Revised ftA /4243 | No | No / No / No | Wrench limit |8.740s|0.715s|118.0N|
| v5_6 /4242 | Yes | No / No / No | Top workspace exit |13.180s|0.650s|60.8N|
| v5_6 /4243 | No | No / No / No | Wrist extension |15.196s|0.658s|0N|

Large contact forces did not produce sustained bilateral acquisition. Packet vertical motion was zero apart from numerical noise. Force magnitudes remain sensitive to estimated collision geometry and a rigid packet model; neither the synthetic tactile/wrist mapping nor hardware transfer is calibrated. Reach is the frozen diagnostic based on an absolute closure threshold, not a claim that the first incremental closing command was well timed. The CSV retains that separate distance diagnostic.

These models are **fingertip-tactile-free but retain wrist force/torque**. Their saved `mask_wrist=False`, ACC mode, RoPE and checkpoint-specific noise configuration were preserved. The tactile panels below show stored simulator/controller proxy observations used by shared safety and the grip latch; **the students do not consume these observed fingertip pixels**. They are not real tactile footage.

[Four-case contact sheet](results/student_nominal_smoke_v1/review_contact_sheet.jpg) · [Per-trial CSV](results/student_nominal_smoke_v1/report/per_trial.csv) · [Summary](results/student_nominal_smoke_v1/report/summary.json)

| Trial | Scene and controller-tactile review |
|---|---|
| Revised ftA /4242 | [Video](results/student_nominal_smoke_v1/video_reviews/student_ftA_r1__nominal__seed4242/policy_review.mp4) |
| Revised ftA /4243 | [Video](results/student_nominal_smoke_v1/video_reviews/student_ftA_r1__nominal__seed4243/policy_review.mp4) |
| v5_6 /4242 | [Video](results/student_nominal_smoke_v1/video_reviews/student_v5_6__nominal__seed4242/policy_review.mp4) |
| v5_6 /4243 | [Video](results/student_nominal_smoke_v1/video_reviews/student_v5_6__nominal__seed4243/policy_review.mp4) |

All four H.264 videos preserve the actual recorded duration and native 15 Hz causal scene/tactile mappings. [Video verification](results/student_nominal_smoke_v1/video_reviews/manifest.json) checks stored gel pixels, strict score agreement and decoded frame counts. [Runtime audit](results/student_nominal_smoke_v1/runtime_audit.json) verifies checkpoint/EMA/NFE 1/K 4 identities, normalization, native source hashes, and that the owned controller and both servers exited with endpoint 7797 clear. Existing resident servers were not signaled by this audit.

The exact [design](results/student_nominal_smoke_v1/campaign/campaign_snapshot.json) has SHA256 `dc02a6ea05cbfa3c9b3bb410a731f77aa7e8fff74352e3c16548346b5efa307a`. The [checkpoint readiness audit](results/student_readiness_20260906/student_inventory.md) records complete current weight hashes and input semantics. Shared GPU timing was not isolated. No teacher–student confidence interval or broader success claim is inferred from these two seeds.

Authoritative raw outputs remain on `compute3:/home/physicalai/phantom-icra-2027/sim/waffles/runs/student_nominal_smoke_v1/`. The publication copy includes compact numeric traces, command logs, scores and all four review videos; bulky observation/tactile arrays and duplicate raw scene videos remain remote.
