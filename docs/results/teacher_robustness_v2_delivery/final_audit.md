# Independent final audit

**Passed: 56 valid trials, zero missing or invalid cases. No clear simulator winner.** The [audit JSON](final_audit.json) contains raw-file hashes, verified counts and independently reconstructed paired gates; [the CPU helper](audit_final.py) reproduces the checks without inference or physics.

All 24 confirmation trials were rescored from their raw physical, planner and execution traces. Their stage outcomes, final object support, stop/completion fields and runtime checks match the authoritative selector rows exactly. The prior 32-trial screen audit remains valid: all 192 underlying trace/log/config hashes are unchanged. The copied confirmation [selection](confirmation/selection.json), [CSV](confirmation/trials.csv) and [report](confirmation/report.md) match compute3 byte hashes.

| Confirmation candidate | Valid trials | Acquired | Sustained lift / carry | Strict / clean placement | Safety stops | Wrench stops |
|---|---:|---:|---:|---:|---:|---:|
| ftA3000, NFE1/K4 | 12 | 3 | 1 / 1 | 0 / 0 | 9 | 7 |
| v5_6, NFE1/K4 | 12 | 3 | 1 / 1 | 0 / 0 | 12 | 5 |

Both carry trials ended with a wrist-extension stop while holding the packet. Across screen and confirmation together, there were **10 acquisitions, two sustained lifts/carries, no releases into the bin, no full placements and no scored drops**. There were 46 safety stops, one separate `veto_retry_cap` controller stop and nine horizon endings without a stop. Safety causes were 34 wrench, four wrist-extension, four hitbox, three joint-speed and one tactile-depth event. One joint-speed stop also logged a nonterminal workspace clamp; it is not an extra terminal stop.

The frozen ranking puts ftA3000 first on fewer safety stops after tied placement, lift and acquisition counts. This ordering does not satisfy the winner rule. Independently resampling the six paired start clusters, with both seeds kept together, reproduces the frozen 10,000-replicate bootstrap exactly: clean-placement difference **0**, descriptive 95% interval **[0, 0]**. Both input matrices are entirely zero. This degenerate interval does **not** establish equivalence or a hardware success probability.

| Predeclared winner gate for ranked ftA3000 versus v5_6 | Result |
|---|---|
| At least 8/12 clean placements | Fails: 0/12 |
| Paired 95% lower bound above zero | Fails: zero |
| No increase in drops | Passes: 0 versus 0 |
| No increase in wrench stops | Fails: 7 versus 5 |
| No increase in tactile force/depth stops | Passes: 0 versus 0 |

The verified conclusion is **inconclusive; pickup candidates only**. Confirmation uses six reserved simulator-study starts and fresh seeds, not established training-unseen data. The field `tactile_force_limit_stop` includes both force and depth limits. Contact geometry, sensor transfer and shared-compute latency remain limitations; these conditional simulator results do not validate a physical pick-and-place policy.
