# Paired RGB diagnostic: six valid first-plan proposals

Changing rendered RGB to the paired real camera frame changes this teacher’s first proposal, while repeated rendered inputs reproduce exactly the same stored actions. The head-10 endpoint shifts by **2.206 mm** and **4.953 mm** for the two seeds. All proposals retain their initial **+Y motion and downward Z motion**; this RGB swap does not reverse that direction. These are unexecuted proposals from one saved input, not a pickup test or a causal explanation of hardware failures.

The explicit v3 run completed at **2026-09-07 01:48:34 UTC**. The [independent audit](audit.json) passes all 36 checks. [Raw results](execution/results.json), [input/identity manifest](execution/manifest.json), [owned-server exit metadata](server/server.json), [ownership acknowledgment](execution/ownership.json) and all six saved proposals are included as small JSON files; observation tensors and model weights are not copied.

## Paired effect and repeat control

Each entry compares real RGB against rendered RGB for the same reset seed. XYZ step RMSE is across the 16 × 3 translational action values. Closure is the normalized 0-open / 1-closed command. Endpoint differences use native cumulative deltas without an additional time multiplier.

| Seed | XYZ step RMSE (mm) | Head-10 endpoint difference (mm) | Full-16 endpoint difference (mm) | Closure RMSE | Rendered-repeat difference |
|---|---:|---:|---:|---:|---|
| 903101 | 0.1741 | 2.206 | 2.601 | 0.007215 | Exactly zero in all stored action values |
| 903102 | 0.3098 | 4.953 | 5.196 | 0.003173 | Exactly zero in all stored action values |

The same candidate index (`k_pick = 0`) is selected in all six calls. Native inference latency spans 1.139–1.185 s; action timestamps shift with latency even when the repeated action values are identical. The repeat control concerns action values, not exact wall-clock timing.

| Seed / RGB | Head-10 ΔX (mm) | Head-10 ΔY (mm) | Head-10 ΔZ (mm) | Mean head-10 closure | Proposal |
|---|---:|---:|---:|---:|---|
| 903101 / rendered | -20.021 | 41.357 | -29.878 | 0.353057 | [JSON](execution/seed903101_rendered.json) |
| 903101 / real_rgb | -20.509 | 41.798 | -27.773 | 0.359483 | [JSON](execution/seed903101_real_rgb.json) |
| 903102 / real_rgb | -23.798 | 76.138 | -34.879 | 0.397372 | [JSON](execution/seed903102_real_rgb.json) |
| 903102 / rendered | -22.334 | 77.116 | -39.508 | 0.399300 | [JSON](execution/seed903102_rendered.json) |

The real-image swap slightly reduces descent in both seeds (about 2.11 mm and 4.63 mm over ten steps). Its closure shift has opposite signs across the two seeds. These observations describe this input only; two seeds do not establish a general visual-transfer error or model ranking.

## Input and loading checks

The request is original halted-study `fta1500_nfe1_k4__start_1787395928__seed903101/observations/0000.npz`, at simulation time 0.26 s. Every saved non-RGB field matches the original source by dtype, shape and value hash: wrist history, UR state, tactile gel, fields, contact state, reactive input, previous chunk and request time. `t` and `reactive` are restored from NPZ zero-dimensional arrays to their original Python-float wire types without numerical changes; the remaining seven fields, including RGB, retain their NumPy dtypes and shapes.

The real image independently matches canonical demo5928 camera frame 3 at native time 5581.950773053104. It precedes the request by 49.852 ms and is 10.148 ms after the requested camera timestamp. The robot moved 1.032 mm from its real initial TCP; the image’s measured pose differs from the simulator request by 5.130 mm and 0.799°. Full-image RGB MAE is 36.383 / 255. This is appearance difference, not a camera-calibration metric, and the replacement includes a small physical pose/timing mismatch.

All six calls use **ftA1500 teacher EMA**, SHA-256 `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e`, with NFE 1, K 4, guidance 1, parity and persistent noise enabled, task text `waffles`, and no previous plan/CPK. The seed/noise state is reset before every request. Call order is rendered → real → rendered-repeat for seed 903101 and real → rendered → rendered-repeat for seed 903102. The checkpoint’s current file hash, hardware file and six native inference-source hashes were recomputed; saved-request and dedicated-server EMA/normalizer/backbone/text-cache identities match. Backbone, text-cache and normalizer agreement is based on the server’s audited metadata; this independent check did not reload a model.

Both primary stages were complete (32 screening and 24 confirmation trials), with their controller processes gone before these calls. Their selection/progress hashes still match the pre-inference completion guard. Dedicated server PID 1576780 and orchestrator PID 1576778 are gone; port 7798 has no listening socket at audit time. No hardware, physics, safety filtering or object scoring was invoked for these proposals.

## Preserved failed attempts and reproduction

[V1](../teacher_rgb_transfer_v1/README.md) incorrectly treated its own ownership acknowledgment as foreign busy and sent no diagnostic replan. [V2](../teacher_rgb_transfer_v2/README.md) submitted one request that failed in native preprocessing before model sampling because the NPZ-restored reactive value had the wrong scalar container type. Neither produced a valid proposal. Both attempts and their stopped-server records are retained. V3 changes only those client ownership/serialization errors; inference settings, seeds, source input and paired RGB order remain fixed. The 18 focused CPU protocol/preprocessing tests pass, including the actual native teacher batch-preprocessing path.

The [v3 orchestrator](run_rgb_after_study.py), [actual server command](server/command.json) and [actual client command](orchestration/client_command.json) identify the executed paths and recipe. They refuse an existing output; rerunning needs a new explicitly owned diagnostic version. The [read-only audit helper](audit_rgb.py) can be rerun via Python stdin on compute3 and emits small JSON only. Source helper SHA-256 is `2aee711780e3e2dbecec9c928e46efcdf8486b9d5e05a24ae42a1b03bc32b5a0`.

The original saved request retains the halted study’s earlier non-RGB proxy/state limitations. This check covers one initial image and two seeds. It neither updates the corrected 56-trial study nor tests whether corrected appearance improves closed-loop pickup or placement.
