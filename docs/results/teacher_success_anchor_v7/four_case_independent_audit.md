# Independent four-case RPC diagnostic audit

**Passed: four original cases rescored, zero discrepancies, 4,859 frozen file/input checks.** The pinned v7 analyzer, recipe/EMA/configuration checks, clock instrumentation, raw physical scores and legacy execution-only FINISH evidence agree. The complete [primary summary](diagnostic/summary.json) and [progress ledger](diagnostic/progress.json) are mirrored alongside this audit.

All cases use ftA1500 teacher EMA, NFE 1, guidance 1, persistent noise, parity, the same fixed physical anchor, historical request-time minimal-v5 release/FINISH profile and full-client RPC delivery. The limiter is explicitly off. Only K differs (1 versus 4). The two seeds 904301/904302 are reused development seeds. This is not teacher selection, independent confirmation, a reliable-win claim or grounds to expand automatically. V5 reserved-selection results remain unchanged.

| K | Seed | Acquired/lifted/carried | Strict place / clean FINISH | Actual stop | Sampled peak pad→packet normal sum |
|---:|---:|---|---|---|---:|
|4|904301|yes/yes/yes|24.000s / 24.172s|none through 59.996 s|5.945 N|
|4|904302|yes/yes/yes|no / no|wrist_extension at 16.236s|3.810 N|
|1|904301|yes/no/no|no / no|wrist_extension at 13.516s|14.277 N|
|1|904302|yes/no/no|no / no|none through 59.996 s|32.346 N|

K4 acquired/lifted/carried 2/2 and physically placed with clean FINISH 1/2; the other case stopped at wrist extension while carrying. K1 acquired 2/2 but lifted/carried/placed 0/2; one case stopped at wrist extension and one reached the full horizon. All four have zero scored drops, zero IK rejects and zero recorded stale/rejected hold duration. Force magnitudes are uncalibrated simulated contact diagnostics, not safe hardware load claims.

| K | Seed | Native model duration, mean | Full client call, mean | Client minus native, mean | Outer runner wall, mean |
|---:|---:|---:|---:|---:|---:|
|4|904301|0.816683s|0.819721s|3.038ms|0.980758s|
|4|904302|0.872616s|0.875802s|3.187ms|1.063016s|
|1|904301|0.303406s|0.306575s|3.169ms|0.505482s|
|1|904302|0.248991s|0.252076s|3.085ms|0.447348s|

The complete client call adds about 3.1 ms on average here. The much larger outer-minus-client duration includes runner observation/audit work; it is not pure RPC transport overhead. Native Plan latency and action-time grids remain unchanged for CPK conditioning, while actual delivery follows the measured client call. The independent audit found no early activation; maximum recorded physics-clock deviation is 2.85 µs. Per-plan counts, tails and timing differences are in the [numeric audit](four_case_independent_audit.json).

The clean K4 development success reaches strict physical placement at 24.000 s and FINISH at 24.172 s, with no later stop through 59.996 s. This supports that physical placement remains possible with full-client delivery in this instance. It does not establish that switching the clock caused the success, nor does it turn the two-seed K comparison into a reliable policy ranking.

The frozen campaign retains stale inherited `comparability_requirements` prose referring to four prospective teachers and v3 winner criteria. The audit preserves those strings verbatim in its erratum field, verifies the actual executable grid as two recipes × two seeds, and applies no such ranking or winner test. No frozen file was rewritten.

Reproduce using the existing compute3 Python (choose a new output path):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v7/audit_four_case_diagnostic.py --out /tmp/v7_independent_four_case_audit.json
```

[Read-only helper](audit_four_case_diagnostic.py) · [Source assembly audit](source_audit.md). No model, GPU, hardware or running process was controlled by this audit.
