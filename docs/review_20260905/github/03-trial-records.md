The retained September 4 directory has 29 episodes, all with unknown outcomes. Twenty-three carry `ckpt:BEST.pt` and six `ckpt:teacher_001500.pt`, apparently the same ftA build under different loading paths. All share one git tag despite three hardware hashes and evolving controls. Stop/state records also mix triggering telemetry with state measured after stopping.

Related: #3's provenance work and #4's trial protocol/stop-state items. Preserve existing recordings; do not auto-convert guard stops into success/failure labels or overwrite historical metadata based on current configurations.

Acceptance criteria:

- Each allocated attempt has a stable ID before launch, with a result even if initialization fails and no episode is recorded.
- Save an immutable effective manifest: loaded weight digest, client/server revisions and dirty-diff identities, normalization/data identities, all effective model/controller/lift/latch/limiter settings, seed, actual starting state, placement/pair ID and declared outcome definition.
- Loaded server identity comes from the server's actual artifact. Mutable basenames/symlinks are display names, not experiment identity.
- Preserve acquisition, retention, transport and release outcomes separately from termination cause. Unknown remains a valid explicit outcome; interventions and invalid attempts have explicit accounting rules.
- Store trigger timestamps and synchronized trigger state separately from settled terminal state, including relevant safety values and accepted-command diagnostics.
- The launcher blocks completion of a session readout when counted attempts have unexplained missing records; it must not invent labels or silently drop guard-stopped trials.
- Tests cover changed effective overrides, resident-server/client mismatch, pre-episode failure, unknown outcome and a stop where trigger speed differs from settled speed.

For the first pilot, use a simple reviewed human outcome ledger for the declared grasp-and-carry task. Full pick-and-place requires validated intentional release semantics; a gripper latch must not silently redefine completion.

Sources: [launch metadata](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/phantom/scripts/run_deploy.py#L735), [post-stop state capture](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/phantom/deploy/runtime.py#L329). Episode inventory: `compute3:~/phantom-icra-2027/data/episodes/deploy/20260904/`.
