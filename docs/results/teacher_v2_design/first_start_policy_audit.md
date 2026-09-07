# First-start eight-trial diagnostic

All eight completed first-start trials passed independent raw-metric, runtime-recipe and selector-metadata checks. Initial q/TCP/packet/closure values match exactly across candidates and seeds. This is a single-start diagnostic, not the32-case screen selection or a winner claim.

The initial pad-midpoint distance to the estimated packet OBB was 276.28mm in every case; initial measured closure was 0.427403. All starts already exceeded the absolute.25 closure threshold.

| Candidate | Seed | Acquire/lift/carry/place/clean | Stop and time | Additional closing onset | Error before additional closing | Native mean / delivery mean |
|---|---:|---|---|---:|---:|---:|
| fta1500_nfe1_k4 | 903101 | 0/0/0/0/0 | horizon; no stop | 10.916s | 57.61mm | 0.787/0.787s |
| fta1500_nfe1_k4 | 903102 | 0/0/0/0/0 | joint_speed @10.020s | NA | NA | 0.813/0.813s |
| fta3000_nfe1_k4 | 903101 | 0/0/0/0/0 | wrist_extension @22.660s | NA | NA | 0.796/0.796s |
| fta3000_nfe1_k4 | 903102 | 0/0/0/0/0 | joint_speed @10.540s | NA | NA | 0.810/0.810s |
| v5_6_nfe1_k4 | 903101 | 0/0/0/0/0 | wrench_limit @7.516s | NA | NA | 0.825/0.825s |
| v5_6_nfe1_k4 | 903102 | 0/0/0/0/0 | wrench_limit @8.004s | NA | NA | 0.823/0.823s |
| fta1500_nfe5_k1 | 903101 | 0/0/0/0/0 | joint_speed @10.860s | 4.340s | 214.99mm | 0.941/0.941s |
| fta1500_nfe5_k1 | 903102 | 0/0/0/0/0 | wrench_limit @7.068s | NA | NA | 0.951/0.951s |

No trial acquired a sustained bilateral grasp or triggered controller completion. This does not mean zero contact: peak packet-filtered pad normal force reached.760N for ftA1500 NFE1/K4 seed903101,34.44N for v5_6 seed903101, and132.88N for ftA1500 NFE5/K1 seed903102. These are uncalibrated simulated contact magnitudes. There were no IK rejects in these eight cases.

The stopped-tick event list can include nonterminal `workspace_clamp` alongside a true stop such as `joint_speed`; the table above names the actual terminal cause. No numerical clean-score mismatch was found. No source, thresholds, mapping or controller settings were changed after these outcomes.

[Detailed machine-readable audit](first_start_policy_audit.json) includes per-trial settings, clocks, reach diagnostics, hold/reject/completion details and native/delivery/wall latency summaries. [CPU read-only helper](first_start_policy_audit.py).
