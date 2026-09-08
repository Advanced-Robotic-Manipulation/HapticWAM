# V9 release dwell audit

Four trials on two reused development seeds. One measured arm start and one waffle pose; actual RPC delivery timing can differ across repeats.

| Dwell / seed | Acquire / lift / carry / place / drop | Release permission | Command open | Stop | End |
|---|---|---:|---:|---|---:|
| 0.2 s / 904301 | 1/1/1/1/0 | 23.948 s | 23.948 s | none | 59.996 s |
| 0.2 s / 904302 | 1/1/1/1/0 | 23.620 s | 23.620 s | none | 59.996 s |
| 0.1 s / 904301 | 1/1/1/1/0 | 25.204 s | 25.204 s | none | 59.996 s |
| 0.1 s / 904302 | 1/0/0/0/0 | — | — | joint_speed | 35.108 s |

Audit: **audit_pass**. [Numerical audit](audit_v9.json) pins raw and frozen inputs and reports matched per-seed outcomes, actual latency, accepted command bounds, hold budgets, release proposals, latch transitions and physical robot/bin support.

No historical pooling or reliable-winner inference. A release permission or controller FINISH alone is not placement. Physical task success and any subsequent safety stop are reported separately.
