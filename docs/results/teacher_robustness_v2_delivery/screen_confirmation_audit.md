# Independent screen and confirmation-design audit

**Passed.** All 32 completed screen cases were rescored from their raw numeric, execution and planner traces using the frozen evaluator. The recomputed physical stages, stop/completion fields and final support conditions match every selector row. Runtime, scene and case/server identity checks pass. The published per-trial CSV matches those rows, and the local selection/confirmation files match compute3 byte hashes. See the [audit JSON](screen_confirmation_audit.json) and [read-only CPU helper](audit_screen_confirmation.py).

| Candidate | Trials | Acquisition | Sustained lift | Strict/clean placement | Safety stops |
|---|---:|---:|---:|---:|---:|
| v5_6, NFE1/K4 | 8 | 2 | 0 | 0 / 0 | 8 |
| ftA3000, NFE1/K4 | 8 | 1 | 0 | 0 / 0 | 6 |
| ftA1500, NFE5/K1 | 8 | 1 | 0 | 0 / 0 | 7 |
| ftA1500, NFE1/K4 | 8 | 0 | 0 | 0 / 0 | 4 |

There are four bilateral acquisitions in total, with no sustained lift or placement. Reapplying the frozen ranking selects **v5_6 NFE1/K4 and ftA3000 NFE1/K4**. ftA3000 wins the one-acquisition tie because it has six safety stops versus seven for ftA1500 NFE5/K1. The latter also has one separate, non-safety `veto_retry_cap` stop. A selected configuration is not a demonstrated pickup candidate or final winner.

The derived confirmation JSON is exactly reproduced apart from its recorded creation timestamp. Its hash is `8242935967b8d16616759fd3282834c346f16f51f779f2a2dab64ecdb20298fe`, and the active confirmation snapshot has the same bytes. All shared parameters, scene, physical thresholds, release/veto profile, measured sensor baseline and selected checkpoint recipes remain unchanged. All ten recorded-start file hashes and the baseline hash match.

Confirmation contains the six reserved starts ending **6060, 6094, 6128, 6314, 6346 and 6461**, with fresh seeds **903201 and 903202**, twelve trials per selected candidate and 24 total. Candidate block order alternates AB/BA across the six starts. Neither a discovery start payload nor a discovery seed is reused. This audit reads no confirmation outcomes.

The JSON/CSV field `tactile_force_limit_stop` includes either `tactile_fz` or `tactile_depth`; human-facing reports should label it **tactile force/depth stops**. Logged event lists can include nonterminal workspace clamps, while actual terminal controller causes are reported separately. Zero lift/place counts, or a later degenerate confidence interval, do not establish equivalence or hardware performance. Confirmation retains six paired start clusters with both seeds together; no training-unseen claim is supported.
