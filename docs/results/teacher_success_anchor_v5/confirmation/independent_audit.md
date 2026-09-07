# Independent reserved-confirmation audit

**Passed: all 24 reserved cases are valid and match the authoritative scores exactly.** The separate screen audit passed its 24 cases, giving 48/48 matched model-comparison cases across the two stages. No missing, retried, reused, or invalid case was substituted. Each returned raw score, case/input hash, physical/stop/completion field, ranking tier, paired statistic, and final winner gate agrees with the frozen review layer.

| Candidate | Acquired | Lifted | Carried | Strict place | Clean finish | Drops | Actual safety stops |
|---|---:|---:|---:|---:|---:|---:|---:|
| fta1500_nfe1_k4 | 12/12 | 7/12 | 6/12 | 1/12 | 1/12 | 1/12 | 11/12 |
| fta3000_nfe1_k4 | 8/12 | 4/12 | 4/12 | 0/12 | 0/12 | 0/12 | 12/12 |

ftA1500 remains the conditional ranked leader. It is not a validated clear winner: physical-placement difference is +8.33 percentage points, paired bootstrap 95% interval [0, +25] points, exact two-sided McNemar p=1. The declared ≥8/12 physical placements, strictly positive interval lower bound, p≤0.05, and no additional drops gates fail. Only the no-increase-in-pre-placement-force/depth-stops gate passes. The interval does not establish equivalence or hardware reliability.

The ftA1500 success is seed904510: release-in-bin at23.600s, strict supported physical completion at24.268s, controller FINISH at25.076s, and no actual stop through the60s horizon. Final packet-to-bin support is0.34355N, robot support0N, inside-bin and unloaded checks pass. Physical completion was established independently of FINISH; the older runner's FINISH evidence comes from execution diagnostics, not a missing run.json field. Pad/packet peak is6.384N, an uncalibrated simulated contact diagnostic.

ftA1500 seed904502 lifted and then dropped at13.936s, before its15.716s wrist-extension stop. ftA3000 seed904509 ends inside the bin, unloaded and bin-supported, but does not meet the frozen ordered release/full-task criteria; it stopped for wrench_limit at18.244s. A supported final object state is therefore reported separately and is not silently promoted to strict success.

Terminal safety events: ftA1500 has10 wrist_extension and1 hitbox_exit_top; ftA3000 has5 wrist_extension,3 wrench_limit,3 hitbox_exit_top, and1 joint_speed. All23 actual stops precede strict physical completion; no post-completion stop occurred. Tactile-force-limit labels include depth, but neither candidate had such a stop here. Event tables may contain overlapping coevents in general; these are terminal causes, not nonterminal workspace clamps.

Mean of per-episode native inference durations is0.86463s for ftA1500 and0.87539s for ftA3000. Recorded delivery-inclusive duration equals native duration in this declared v5 profile; full RPC wall time is logged separately and is not substituted into that delivery clock. These comparisons concern matched sampling seeds at one fixed simulator anchor with shared compute, approximate rigid packet/contact and pad-only wrist sensing. They do not establish varied-start robustness, calibrated force transfer, or physical-lab success probability. The separate RPC/limiter development work does not alter these scores.

- [Authoritative selection](selection.json) and [report](report.md).
- [Independent full audit](confirmation_independent_audit.json); 24 raw cases rescored using pinned review modules and unchanged thresholds.
- [Final transition ledger](final_status.json); study_complete, no automatic arm extension.
- [Completed-screen independent audit](../screen/independent_audit.md).
- [Read-only audit helper](../audit_completed_confirmation.py), alongside the unchanged pinned screen helper.

Campaign SHA256 `fe62f45fcaf2e71e4df890ff7303bf70a2b49e35fc2a745f6f7492a5f200b855`; parent protocol SHA256 `f67fe79a5267c6d253d7d90f67f2a367c8899f030959c8f42fa714d072ab7ac2`. Primary selection SHA256 `464d7d354d334232ab4861e0e15faf6e118e1f46e8f2882df0cd6248289d0d60`.

The audit changed no primary source, raw record, process, model, threshold, or score. It recomputed the paired12-seed bootstrap (10,000 draws, fixed20260907 seed) and exact McNemar arithmetic independently of the primary selection implementation.
