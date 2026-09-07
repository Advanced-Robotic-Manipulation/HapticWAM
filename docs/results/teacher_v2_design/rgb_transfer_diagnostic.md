# Paired RGB transfer diagnostic — completed

The six native first-plan proposals completed in **teacher_rgb_transfer_v3** after the 56-trial primary study. [Results, independent audit and exact saved proposals](../teacher_rgb_transfer_v3/README.md) preserve all six calls. No proposal was executed through a controller or scored as a task success. The diagnostic uses ftA1500 EMA/NFE1/K4 as a historical input reference, not a comparison of the two selected teachers.

| Seed | Real-vs-rendered 10-step endpoint difference | Closure RMSE | Repeated-render action difference |
|---|---:|---:|---:|
| 903101 | 2.206 mm | 0.007215 | Exactly zero |
| 903102 | 4.953 mm | 0.003173 | Exactly zero |

This input is RGB-sensitive. Replacing its initial image does not reverse the proposed initial direction, establish a pickup benefit, or isolate the cause of later grasp/transfer failures. Both image variants retain the original request's non-RGB proxy limitations, and the real image includes a small pose mismatch. The diagnostic neither establishes camera calibration nor rules out consequential visual differences during contact and transport.

The [client](../../../tools/sim/diagnose_rgb_transfer.py) uses the first actual request from halted original case `fta1500_nfe1_k4__start_1787395928__seed 903101`: one latest RGB frame and eight other saved fields. Joint/TCP state, wrist history, fingertip arrays, contact state, reactive input, previous chunk and request time retain their exact saved numerical values. All arrays preserve dtype/shape/bytes. Original Python scalar fields `t` and `reactive` are restored from their NPZ scalar arrays for the native wire interface; no observation value is changed by that restoration.

The real replacement is demo 5928 frame 3 at native time 5581.950773053104. It is 10.148 ms after the requested camera timestamp and 49.852 ms before the model request. Real q/TCP are joined by their shared timestamp, not row number. The real arm moved 1.032 mm from its recorded start and differs from the saved simulated request by 5.130 mm/0.799°. Full-image RGB MAE is 36.383/255; this is an appearance diagnostic, not an optical calibration error. The [original CPU preparation manifest](rgb_transfer_preparation.json) and v3 execution manifest preserve the source evidence.

The fixed call order is rendered/real/rendered-repeat for seed 903101 and real/rendered/rendered-repeat for seed 903102. Every call resets native episode/noise state, uses no previous plan, and retains task `waffles`, EMA, NFE1, K4, guidance 1, parity and persistent noise. The audit verifies checkpoint, normalizers, backbone/text cache, hardware and six native inference-source hashes. Proposed deltas are accumulated without a second dt multiplication; action timestamps and inference latency remain distinct from action values.

Two earlier attempts are preserved: [v1](../teacher_rgb_transfer_v1/README.md) rejected its own newly claimed server connection as foreign ownership; [v2](../teacher_rgb_transfer_v2/README.md) passed ownership but failed native preprocessing on a NumPy scalar where the original interface requires a Python float. Neither produced a valid proposal; v2 failed before model sampling. The final client adds a bounded info-only idle probe, retains native ownership refusal for races, and restores exact scalar wire types. Eighteen CPU tests include the actual native observation preprocessing path.

The [recorded v3 orchestrator](../teacher_rgb_transfer_v3/run_rgb_after_study.py) owns a dedicated server on 7798, refuses existing output directories and checks completed study/controller cleanup before launching. It completed six calls and cleaned up its owned server. The [read-only audit helper](../teacher_rgb_transfer_v3/audit_rgb.py) recomputes saved comparisons and identities without inference. Exact commands, ready metadata, ownership records and outputs are under `/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_rgb_transfer_v3/`.

A new execution must use a fresh output/helper location and an explicitly owned unused port; the preserved orchestrator intentionally refuses to overwrite these results. The original failed outputs, the successful six calls, all primary scores and source recordings remain unchanged. This helper is outside the corrected primary study's frozen 105 files and its 56 scored rollouts.
