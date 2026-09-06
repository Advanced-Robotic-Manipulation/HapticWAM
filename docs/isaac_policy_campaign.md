For the current results and hardware handoff, start with the [Isaac Sim overview](isaac_sim.md). This document retains the earlier reconstruction/protocol history; local artifact paths refer to files outside Git, with retrieval locations explained in the overview.

This campaign evaluates waffle policies in closed loop using matched initial conditions. The design, thresholds, policy identities, and comparisons are frozen in [policy_campaign.json](../configs/sim/policy_campaign.json) before scoring. It uses 20 configurations and two sampling seeds per policy: 40 rollouts for each of three primary policies, followed by 40 for the original student, for 160 planned rollouts. The nominal configuration runs first.

| Policy ID | Checkpoint relative to the live PHANTOM repository | Architecture and interpretation |
| --- | --- | --- |
| `teacher` | `runs/teacher_v5_ftA/teacher_001500.pt` | Teacher; exploratory because simulated tactile inputs are uncalibrated proxies |
| `revised_student` | `runs/student_ftA_r1/student_001200.pt` | Revised student; primary method |
| `control` | `runs/control_ftA/teacher_001200.pt` | **Student architecture despite filename**; no HID loss, retains teacher initialization and auxiliary tactile training |
| `original_student` | `runs/student_ftA/student_001200.pt` | Original student; secondary comparison after the primary cohort |

The frozen file records verified checkpoint SHA256 hashes, the entire nominal scene, its source hash, and a pre-scoring scheduling amendment. Seven model loads execute nominal two-seed blocks for each primary policy (six rollouts), the remaining 38 rollouts for each primary checkpoint (114), and then all 40 original-student rollouts. This avoids repeated model reloads when only one additional checkpoint fits on the GPU. Checkpoint blocks introduce a time/order confound; matched states and seeds do not eliminate that confound.

| Family | Configurations | Frozen variation from nominal |
| --- | ---: | --- |
| Nominal | 1 | Full Aug 22 pick/place initial scene |
| Object placement | 8 | Nonzero points of the ±10 mm world X/Y grid; no yaw or friction changes |
| Packet friction | 3 | Static and dynamic packet coefficients ×0.70, ×0.85, or ×1.15; pad friction unchanged |
| Camera | 4 | ±2° about local camera X or Y, composed with the nominal camera rotation; camera position and intrinsics fixed |
| Observation transport delay | 2 | Sensor histories delayed by 80 or 160 ms; request clock, accepted-command history, executor, and safety remain current |
| Added inference delay | 2 | 150 or 300 ms response-delivery delay after measured native inference; native action grid retained |

These are separate perturbation families, with one family changed per condition. They are not a factorial design or an estimate of natural laboratory variation. Each configuration uses sampling seeds 4242 and 4243 across all policies. Full generated camera overrides and intended parameters are stored in the frozen design, and each rollout records effective parameters. Material perturbations act on packet coefficients; PhysX material combination changes the effective pad–packet coefficient by a smaller amount.

Observation delay cuts camera/state/tactile/wrist sensor histories at an older causal time, while `snapshot.t`, the request/action grid, and locally known accepted-gripper-command history remain current, matching native SnapshotBuilder semantics. Added inference delay is post-inference transport time: native action times and selection do not anticipate delivery; late delivery skips the expired action-grid head and rebases. The final pre-scoring amendment also replaces the forearm's inaccurate whole-mesh convex hull with convex decomposition, preserving collisions, after original-CAD and dynamics checks. Policy initialization must settle for two seconds with joint error below 0.02 rad and joint speed below 0.05 rad/s. The current launch configurations include this validated collision correction; the original reconstruction remains archived.

All policies receive the same 30 s simulated horizon. After an actual safety stop, free physics remains observable for up to two seconds, capped by the same 30 s horizon, with no further policy actions or inference. Actual safety termination can end a rollout; task completion alone cannot end it early, so release, settling, and subsequent loss remain observable. The nominal scene and measured initial arm/gripper state come from the prepared Aug 22 demonstration. Subsequent actions come from policy inference rather than the recorded action or joint trajectory. Student checkpoints share the same executor safety and latch semantics. The declared common inference settings are EMA, NFE 5, guidance 1, persistent noise, parity enabled, one inference candidate (K=1), max-play 10, and task text `waffles`. Dedicated server metadata records the actual checkpoint architecture and resolved optional runtime filters.

The selected deployment recipe keeps the grip latch and mandatory motion-stall watchdog enabled, with optional terminal veto/recovery disabled and lift-success auto-stop disabled. All audited students and the control also consume the unvalidated wrist-wrench proxy (`mask_wrist=False`); their results therefore retain force-observation uncertainty even without teacher gel inputs. The preserved latch may prevent release/placement, and that outcome must be reported under the unchanged deployment logic.

Success is determined from object and contact state by [policy_metrics.py](../phantom/sim/policy_metrics.py). Planner flags, gripper OBJ estimates, `lift_complete`, and an asserted `achieved` flag cannot supply task success. The ordered stages are:

1. Reach before closure is a diagnostic: pad midpoint is within 20 mm of the object box while normalized closure is below 0.25. It is not a full-task gate.
2. Acquisition requires packet-filtered bilateral pad force above 0.1 N for at least 0.15 s.
3. Lift requires the free packet to rise at least 30 mm for at least 0.5 s.
4. Carry requires at least 50 mm displacement over at least 0.25 s, with retention evaluated using the declared 0.15 s contact-gap allowance.
5. Release and full task require the oriented packet box to be inside the bin with the declared 2 mm geometric tolerance, unloaded pads, and at least 0.5 s of settling below 0.03 m/s and 0.5 rad/s.

The evaluator also reports drop events, per-stage event times, maximum object lift, carry distance, final object state, stops, rejected IK commands, held-command duration, replans, inference/activation latency, and reported collisions. The exact drop and all other thresholds are in the frozen file. A later safety stop does not erase physical task completion that already occurred; it remains a separate reported outcome. Whole-robot collision-free behavior is not certified by pad contact forces. Non-packet pad forces and reported collision events are measured diagnostics with explicit observability limits.

Policy scoring requires a policy-mode run, finite trace data, an independent physics clock, at most 0.2 s between state samples, and audited metadata declaring a dynamic, nonkinematic free packet with no attachments or post-initialization pose writes. The clock tolerance is 1 ms. Metadata is an audit claim, not a mathematical proof that source code never changed object state. The analyzer additionally checks checkpoint identity, the frozen design hash, effective physical/sensor parameters, all effective inference settings including EMA, the complete frozen hardware model, and raw trace start/end coverage against the declared horizon. Execution/planner logs and an immutable server-ready snapshot are required even when there are no plans. The final pre-scoring v4 recipe corrects the unscored preflight hybrid K=4 setting to the source-audited GO_ANY baseline K=1; the two matched episode sampling seeds remain unchanged. A policy failure or safety stop remains scored; missing provenance or an incomplete run is marked invalid instead of silently treated as success or dropped from the report.

The primary endpoint is `full_task`, and the predeclared primary contrast is revised student minus control. Teacher contrasts are exploratory. Revised-versus-original and control-versus-original contrasts are secondary. Rates weight each configuration equally and each of its two seeds equally; perturbation families are not reweighted to equal sizes after observing outcomes. The report includes every stage, family, seed, seed-disagreement count, and scheduled-denominator bounds for missing or invalid trials.

Paired uncertainty intervals use 10,000 percentile bootstrap replicates with analysis seed 20260906. Each bootstrap resamples the 20 matched configuration clusters while retaining both seed outcomes and the policy pairing within each cluster. The 95% interval is a descriptive scenario-resampling interval for this fixed sensitivity suite, not a hardware or deployment-population confidence bound. Two sampling seeds provide limited information about sampling variability. A degenerate interval, including `[0, 0]`, does not establish policy equivalence. Incomplete pairs are reported explicitly; only configurations with both seeds valid for both compared policies enter a paired interval. No acceptance thresholds, cases, or comparisons may be tuned after observing outcomes.

Each rollout directory contains `case.json` with the frozen campaign hash, policy ID, condition ID, sampling seed, checkpoint hash, and intended/effective parameters. It also retains `run.json`, `effective_config.json`, `sim_trace.npz`, `execution_trace.jsonl`, `planner_trace.json`, `policy_info.json`, and an immutable `server_ready.json` snapshot. Duplicate trial keys are rejected. An executed failure must not be replaced with a more favorable attempt. A demonstrable infrastructure failure before any policy command may be retried only with explicit linked attempt provenance.

Analyze completed or partial campaign output with ordinary CPU Python:

```bash
python tools/sim/analyze_policy_campaign.py \
  --campaign configs/sim/policy_campaign.json \
  --runs artifacts/isaac_waffles/policy_campaign_v1/rollouts \
  --out artifacts/isaac_waffles/policy_campaign_v1/analysis
```

The analyzer recomputes physical metrics from raw traces and writes separate per-trial scores and `summary.json`; source run files are not changed. An incomplete report retains the full scheduled denominator and identifies missing/invalid trials. Focused statistics tests verify matched-ID joins, preservation of seed pairing within clusters, deterministic interval signs, missing-trial bounds, frozen-design enforcement, duplicate rejection, and effective-scene audits.

The [campaign controller](../tools/sim/run_policy_campaign.py) writes a plan by default. On compute3, use the dedicated simulator source and audited runtime hardware file:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python ../sim/waffles/source/tools/sim/run_policy_campaign.py \
  --campaign ../sim/waffles/source/configs/sim/policy_campaign.json \
  --source ../sim/waffles/source \
  --live-repo /home/physicalai/phantom-icra-2027/phantom \
  --evidence ../sim/waffles/evidence \
  --hardware-config ../sim/waffles/runs/policy_campaign_v1/runtime/hardware_campaign.yaml \
  --output ../sim/waffles/runs/policy_campaign_v1 --port 7792
```

Adding `--execute` starts the seven checkpoint blocks. Each block owns one dedicated inference-server process and launches Isaac trials sequentially. The same audited hardware file is supplied to the server and simulator; its frozen SHA256 must match. The controller hashes simulator tools/modules, imported core modules, assets, scene configurations, and read-only episode inputs once before execution. It records every case, command, server metadata, and completion status. Only subprocess groups created by that controller are terminated. Completed trials, including physical failures, are reused on restart after identity checks; incomplete attempts are preserved and require explicit provenance review rather than automatic replacement. No hardware launcher is called.
