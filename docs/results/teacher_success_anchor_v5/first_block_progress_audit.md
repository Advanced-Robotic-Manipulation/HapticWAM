# Independent first-block integrity audit

All four completed seed 904401 cases pass the frozen external analyzer and the independent provenance checks. This is an integrity audit, not a model ranking.

| Candidate | Actual NFE / K | EMA | Strict valid | Recorded coverage |
|---|---:|---|---|---|
| fta1500_nfe1_k4 | 1 / 4 | EMA | yes | safety at 18.748 s; trace through 20.744 s |
| fta3000_nfe1_k4 | 1 / 4 | EMA | yes | safety at 7.820 s; trace through 9.816 s |
| v5_6_nfe1_k4 | 1 / 4 | EMA | yes | safety at 8.556 s; trace through 10.556 s |
| fta1500_nfe5_k4 | 5 / 4 | EMA | yes | full horizon through 59.996 s |

Verified actual checkpoint bytes, EMA selection, server and policy recipes, and every saved raw plan’s NFE/K. All four effective scene files and initialization files match byte for byte; all eight non-RGB first-observation arrays match. The measured initial state and tactile baseline have the same pinned hashes. Rendered RGB is not asserted identical.

The preflight checked 94 runtime source/assets files, 46 external-driver files, seven live-core files and two prepared-episode files (149 total), plus the distinct checkpoint files, hardware, initial-state, tactile-baseline and robot-USD hashes. Physics and reported clocks remain aligned within 2.85 microseconds. The three shortened recordings have actual safety termination and observation tails; the fourth reaches the declared 60-second horizon. No case completed execution FINISH.

The robot is UR3 CB3, supported by the recorded dashboard identity. Existing held-out nominal joint-to-TCP validation reports 1.340 mm translation RMSE and 1.500 mm maximum error. Separately, the driven measured replay reports joint RMSE 0.009425 rad and TCP RMSE 6.347 mm / 1.025 degrees. These checks cover different error sources; neither certifies the custom pad/table calibration. See [robot evidence](robot_model_evidence.json).

Files: [full compact audit](first_block_progress_audit.json), [reproducible CPU audit helper](audit_screen_first_block.py). The independent per-trial score records stay under compute3 `sim/waffles/runs/teacher_success_anchor_v5/screen/live_audit/first_block/trials/`; the primary controller analysis and raw recordings are unchanged. Watcher PID 2319150 completed normally after all four cases.
