# V9 independent audit

All four trials completed and passed the independent audit. **Keep the 0.200 s release dwell with the 2.5 s arm hold for the next pilot.** The contemporaneous 0.200 s controls both completed strict placement; the 0.100 s diagnostic completed one of two. This does not establish that a shorter dwell causes worse performance: its failed trial never reached an eligible release, and the paired trajectories already differed during approach.

| Release dwell | Seed | Sustained lift / carry | Release permission and commanded opening | Strict physical placement | Stop / duration |
|---|---:|---|---:|---:|---|
| 0.200 s | 904301 | yes / yes | 23.948 s | 24.668 s | none / 59.996 s |
| 0.200 s | 904302 | yes / yes | 23.620 s | 24.336 s | none / 59.996 s |
| 0.100 s | 904301 | yes / yes | 25.204 s | 25.936 s | none / 59.996 s |
| 0.100 s | 904302 | no / no | never | never | measured joint speed, 33.108 s; physics tail to 35.108 s |

All four acquired contact; none recorded a scored drop. Each of the three placements has zero final robot contact and approximately 0.343 N final bin support, and remains observed through the full 60 s horizon. The physical result does not depend on the controller FINISH flag. The failure never arms the loaded grasp latch or reaches an active release window, so changing release dwell never changes its gripper behavior. Its maximum measured joint speed is 1.22944 rad/s against the unchanged 1.2 rad/s stop; submitted joint-step speeds remain at or below 1.0 rad/s. Command bounds do not guarantee measured tracking limits.

## What can and cannot be attributed to the dwell

Configured and settled initial joint states are exactly matched. However, measured RPC timing differs immediately:

| Seed | First control delivery | First treatment delivery | First differing joint/gripper command | Earliest eligible opening in either run |
|---|---:|---:|---:|---:|
| 904301 | 1.676 s | 1.444 s | 1.444 s | 23.748 s |
| 904302 | 1.660 s | 1.468 s | 1.468 s | 23.420 s |

Approach and pick therefore diverge more than 20 seconds before the release setting can intervene. This is variation in real-time delivery, inference, observations or physics, not evidence that a shorter release dwell harms grasp acquisition. The audit observes different delivery timing but does not isolate it as the sole cause. Mean full-client RPC latencies for control/treatment are 0.797/0.790 s for seed 904301 and 0.813/0.870 s for seed 904302; individual maxima span 1.181–1.410 s. Median latency alone hides this variation.

The 0.200 s control openings persist long enough this time: the first opening spells last 1.048 s and 0.776 s across selected executor ticks. They pass the existing release gate. The successful 0.100 s trial has a 0.936 s opening spell, so it does not demonstrate a need for the shorter threshold. That latter statement concerns observed duration only; it is not a replay of alternate physics after a different release time.

The previous [V8 bounded-hold runs](../../teacher_carry_hotfix_v8/analysis/README.md) with the same 0.200 s recipe scored **0/2**, while these V9 repeats score **2/2**. Both results remain visible. V8 seed 904301 had only 0.120 s and 0.080 s played opening bursts, exposing a real release persistence weakness; the V9 repeats show the teacher can also supply sustained opening. This small experiment supports a working pilot configuration, not a reliable winner or robust success rate. Improving repeatability is still necessary.

## Audit scope and reproduction

The completed blocks are `control_dwell200` and `treatment_dwell100`, each with sampling seeds 904301 and 904302. The unchanged campaign driver requires two seeds per campaign, so execution uses two blocks (AABB). The rejected single-seed ABBA preflight does not contribute a simulated trial. No retries or extra trials were added.

The audit ran after all four valid trials and both complete primary score indices existed, plus confirmation that all owned processes stopped and all 5,106 frozen pins remained unchanged. It consumes saved strict physical scores without rescoring. Runtime configuration, source/checkpoint/input hashes, fixed scoring thresholds, full horizon or actual stop tail, initial configurations and the sole 0.200-versus-0.100 s opening-dwell difference are checked independently.

The command/FK, hold-budget and fresh-observation checks reuse the pinned V8 independent helper (`2219a5258f5914ae391b6181195affb50c918187efcd38e1ce8fb40c842e1cd2`). Active accepted commands must equal nominal FK; stopped safety holds must equal measured q/TCP. The release audit reconstructs original and delivered gripper samples at actual executor indices, verifies release eligibility, and independently replays the release gate up to its first commit. Subsequent opening, unloading and placement are reported from actual command/state/support telemetry and authoritative physical scores.

Every arm hold uses exactly the previous submitted joint target, independently replays its fixed budget without restarting on a no-op, and preserves the real IK-fault counter. Maximum active hold budgets are 1.376, 1.128, 1.160 and 1.456 s, all below 2.5 s. Every trial activates a post-hold observation's plan and resumes nonheld accepted execution. All measured wrist radii remain below 461.60 mm against the unchanged 468 mm stop.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python audit_v9.py --out /tmp/v9_independent_audit_new
```

The default inputs point to immutable V8/V9 paths on compute3. Choose a new output directory; the helper refuses an existing output. It launches no simulation or inference and changes no raw inputs.

Results: [numerical audit](audit_v9.json), [compact table](audit_v9.md), [audit run output](audit_run.log), [preparation checks](preparation_checks.json). Both block source/input manifests equal V8's frozen bounded-hold manifest except campaign identity. All primary score/raw hashes pass; all four command/budget/horizon audits pass; reconstructed release eligibility and gate state/timer/first commit match observed execution exactly. No audit error was suppressed or threshold changed.

Preparation guards also pass: actual frozen dwell-only designs accepted; undeclared opening-threshold and object-mass changes rejected; incomplete campaigns and wrong block counts rejected before raw trace analysis. Ruff, JSON and helper-hash checks pass.

The three successes establish simulated pick, carry and physical placement for these particular executions. The teacher, tactile transfer and reconstructed geometry/physics remain unvalidated for reliable hardware transfer.
