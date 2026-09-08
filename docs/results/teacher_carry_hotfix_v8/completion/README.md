# Completed six-case record

All six planned cases completed validly. The original `completion_audit.json` was produced at 21:37:50 UTC on 2026-09-07 after verifying all 5,098 pinned inputs. All three controller children exited zero, owned server processes were absent and port 7799 was free. The detached launcher's final OS wait code was no longer available after reap; its six-valid-trials marker and each child completion were verified. `recovery_verification.json` independently checks the unchanged pins and free port on September 8 without relying on stale PID identities.

| Setting | Seed | Acquisition | Sustained lift/carry | Strict placement | Stop |
|---|---:|---|---|---|---|
| Plain K4 | 904301 | yes | yes | no | wrist extension, 15.620 s |
| Plain K4 | 904302 | yes | no | no | wrist extension, 15.676 s |
| Legacy limiter | 904301 | yes | yes | no | limiter stall, 15.108 s |
| Legacy limiter | 904302 | yes | no | no | limiter stall, 15.364 s |
| Bounded hold | 904301 | yes | yes | no | none through 60 s |
| Bounded hold | 904302 | yes | no | no | top workspace exit, 42.484 s |

No case met the scored drop definition. Sequential settings shared actual GPU/RPC timing; same seed does not guarantee an identical closed-loop trajectory. This is a two-seed development diagnostic, not a reliable winner or hardware-validation study. The primary scorer JSONs are preserved under `paired/*/analysis/trials/`; `summary.json` is an index, not embedded trial metrics. Accepted constraint holds are distinct from the primary scorer's rejected/stale hold-duration metric.

`monitor_latest.json` and `bounded_seed904301_hold_audit.json` are preserved original monitoring aids; final analysis should use the completed audit and primary scores. `recovery_manifest.json` and `source_recovery_manifest.json` map copied bytes to their original paths and SHA-256 digests. The local temporary worktree had disappeared before September 8, so uncommitted local-only draft notes and their exact bytes could not be recovered. This report uses surviving remote artifacts and committed prior reports and does not reconstruct or invent missing preflight evidence. The original final six-case audit, configs, manifests, source changes, CPU suite/log and scorer outputs all survived.
