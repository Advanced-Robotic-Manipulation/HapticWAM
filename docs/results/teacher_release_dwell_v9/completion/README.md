# Completed four-case release-dwell diagnostic

All four declared trials were valid. The owned launcher ran once, from 06:37:40 through 06:47:36 UTC on 2026-09-08, and exited zero. Both controller children exited zero. The post-run audit passed at 06:48:50 UTC: all **5,106 pinned files** still matched, all owned supervisor/launcher/controller/server processes were absent, and port 7799 was free. `completion_audit.json` is the machine-readable evidence and rendering gate.

| Setting | Seed | Acquisition | Sustained lift/carry | Strict placement | Placement time | Stop |
|---|---:|---|---|---|---:|---|
| Opening dwell 0.2 s | 904301 | yes | yes | yes | 24.668 s | none through 60 s |
| Opening dwell 0.2 s | 904302 | yes | yes | yes | 24.336 s | none through 60 s |
| Opening dwell 0.1 s | 904301 | yes | yes | yes | 25.936 s | none through 60 s |
| Opening dwell 0.1 s | 904302 | yes | no | no | — | measured joint speed, 33.108 s |

No trial met the scorer's drop definition. The treatment failure occurred before sustained lift or any release commit, so the 2/2 versus 1/2 result does not establish a causal regression from shortening the release dwell. The 0.2 s controller completed the whole task in both repetitions. Neither this small experiment nor a selected video establishes reliability: both seeds were reused, only one measured arm start and object pose were tested, and two sequential setting blocks retained actual shared-GPU/RPC timing variation. Keep V8 and V9 denominators separate.

`trial_table.json` retains all stage timestamps and observed controller events. Primary score JSONs live under `paired/*/analysis/trials/`; the summary files are indices. A controller FINISH event is recorded separately from supported physical placement. The three successful runs were observed for the full 60 s horizon; the failure retained the configured approximately 2 s physics tail after its safety stop.

`launcher_completed.json` preserves the actual launcher return code; `paired/*_completed.json` preserve controller return codes. `recovery_manifest.json` records byte-for-byte copies of 57 final score/config/progress/metadata/log files from their absolute compute3 paths. Raw simulator states, execution streams, policy observations and videos remain unchanged on compute3; the independent analysis and video review use those original files. `audit_completion.py` reads existing scores and verifies pins/cleanup without rescoring or launching models.
