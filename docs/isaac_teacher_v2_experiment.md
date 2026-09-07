**Completed result:** all 56 cases are valid; both reserved teachers lift/carry 1/12 and place 0/12. No clear winner; pickup candidates only. See the [final report, audits and tactile comparison video](results/teacher_robustness_v2_delivery/README.md).

# Teacher v2: fixed waffle, varied measured starts

**Corrected profile frozen at 2026-09-07T00:02:21.650286+00:00.** The active [parent protocol](../configs/sim/teacher_v2_delivery_protocol.json) and [32-trial screen](../configs/sim/teacher_v2_delivery_screen.json) retain the fixed waffle, ten measured starts, four teacher configurations and reserved confirmation design. They add current-delivery guard feedback and an explicit gripper contact wrench proxy. All 366 simulator tests available at freeze and exact-command physics invariance passed.

The [original v2 protocol](../configs/sim/teacher_v2_protocol.json) and its ten completed trials remain preserved as superseded diagnostics. No confirmation case ran before the correction. The request-versus-delivery mismatch did not rewrite the first two trials' action arrays; it is not their demonstrated failure cause. A separate invariant replay identified a housing/box collision omitted by the pad-only wrist proxy.

The previous teacher study achieved some strict simulated placements but nominal success did not reproduce reliably; later stops, incomplete controller release handling and uncertain tactile contact mapping limited interpretation. V2 declares the controller/mapping changes before scoring: live native terminal veto using current delivery TCP/gripper, explicit `gripper_contact_proxy` wrist feedback, `manifold_patch_v2` gel coverage and the existing policy-commanded release controller with `finish_after_release=true`, `finish_observation_s=2`. The controller observes robot feedback, not simulated object truth. Completing its release sequence cannot by itself earn task success.

## Conditions and candidates

Every trial uses the canonical August scene from [waffles_pick_place.json](../configs/sim/waffles_pick_place.json), including green packet center `[-.366715,-.259618,.07]` m and yaw `.44229` rad. Camera, object pose, friction and physical parameters remain common. The packet pose and dimensions are image estimates, not newly measured geometry.

The [cohort evidence](results/teacher_v2_design/README.md) identifies ten complete timestamped teleoperation recordings. Their q/TCP samples share exact native timestamps; gripper/wrist values are causal samples with their own recorded times. No later grasp/lift pose is substituted for recording start. The files preserve measured gripper closure and wrist bias as well as arm pose, so this is a **recorded-start cohort**, not an isolated joint-position perturbation.

| Stage | Starts, identified by recording timestamp suffix | Episode seeds | Candidates | Trials |
|---|---|---|---:|---:|
| Discovery screen | 5928, 5963, 6028, 6273 | 903101, 903102 | 4 | 32 |
| Reserved confirmation | 6060, 6094, 6128, 6314, 6346, 6461 | 903201, 903202 | selected 2 | 24 |

The full timestamp and hash for every initial state are recorded in the protocol. The already fitted5928 example is confined to discovery. Discovery spans the observed Y extremes and lowest Z; remaining starts were reserved using source inputs before new policy outcomes. One recording reverses the wrapper; only its robot/gripper/wrist start transfers into the fixed canonical object scene.

| Frozen candidate order | Teacher checkpoint | NFE | K |
|---|---|---:|---:|
| 1 | ftA step1500, deployed reference | 1 | 4 |
| 2 | ftA step3000, newly staged complete checkpoint | 1 | 4 |
| 3 | v5_6, earlier complete step3000 baseline | 1 | 4 |
| 4 | ftA step1500, inference-recipe comparator | 5 | 1 |

All three unique payloads passed [CPU EMA and normalization checks](results/teacher_v2_design/teacher_payloads.json). Each uses its own saved architecture configuration, EMA, task text exactly `waffles`, guidance 1, persistent noise, parity and ten-step playback cap. The two recipes differ in both NFE and K; they do not isolate an NFE effect. “Later checkpoint” does not imply “better checkpoint.” No student trial belongs to this experiment.

The screen has four start blocks, rotating candidate order left once per block: ABCD, BCDA, CDAB, DABC. Both episode seeds run inside each candidate block. Every candidate therefore occupies each block position once. Confirmation alternates AB/BA across its six starts. This balances coarse order, while native inference latency and shared GPU scheduling still vary; matching seeds does not make the closed loop bitwise deterministic.

## Controller, clock and physical outcomes

The maximum horizon is60 simulation seconds, with native measured inference latency and no added observation/delivery delay or fixed latency override. Actual controller/safety stops retain at most2 seconds of free physics, capped by that horizon. After successful release-controller completion the adapter holds the accepted arm/gripper state and stops replanning; simulation and safety observation continue through the horizon. The2-second finish observation belongs to the release-controller configuration, not a claim that object placement is already successful.

Initial settling retains the existing2-second free hold and limits of .02rad position error and .05rad/s joint speed. The [startup audit](results/teacher_v2_design/start_preflight_audit.json) passed all ten starts without exclusions: maximum joint error .003675rad, initial joint speed .029972rad/s, packet displacement 1.4323e−7m and packet-to-robot contact0N. These preflight results establish initialization eligibility, not policy success. No later measured arm state may replace a failed initial state silently. Infrastructure failures, ineligible initial states and executed policy failures stay explicitly distinct; thresholds cannot be relaxed after viewing scored outcomes.

Physical full task requires ordered acquisition, sustained lift, carry and a released packet settled within the bin, using the unchanged [support-verified thresholds](../configs/sim/teacher_v2_delivery_protocol.json). Positive packet-to-bin normal force and low packet-to-all-robot force must support the settled placement. Reach and first-additional-closure diagnostics remain separate; all ten starts already have closure greater than .25.

The operational **clean placement** criterion additionally requires final oriented-bin containment, final unloaded pads, final robot contact≤.1N, final bin contact>.1N, adapter `completed_reason=placement_release_finished`, and no actual controller stop. A later safety stop preserves an already achieved physical full-task outcome but removes clean-placement status. A controller completion without physical placement earns neither physical full task nor clean placement.

## Frozen selection and confirmation

The [CPU selector](../tools/sim/select_teacher_candidate.py) refuses selection until all32 planned screen cases are present, valid and matched. It orders candidates lexicographically by clean-placement count, strict physical full-placement count, sustained-lift count, acquisition count, fewer actual safety stops, then the frozen candidate order. It returns the top two for confirmation; this discovery ranking does not establish a winner.

The two selected IDs populate the numeric confirmation configuration by reference to the already frozen parent protocol and hash-linked screen result. Its starts, fresh seeds, controller and criteria cannot change in response to screen behavior. A **clear simulator winner** is declared only when the leading confirmed candidate satisfies all of these predeclared operational gates:

- At least8 clean placements among its12 confirmation trials.
- The lower bound of the paired95% start-cluster bootstrap difference in clean-placement rate is strictly positive against the other candidate.
- No increase in drops, wrench-limit stops or tactile force/depth-limit stops, checked separately. The frozen machine field `tactile_force_limit_stop` includes both `tactile_fz` and `tactile_depth` events.

The bootstrap uses10,000 draws and seed20260906, resampling the six recorded starts while retaining both matched episode seeds and both candidates within each draw. Otherwise the result is inconclusive, with at most a pickup candidate if sustained lifts occurred. These are simulator decision rules, not calibrated hardware probabilities. Six clusters provide limited precision, and zero or degenerate intervals do not establish equivalence.

## Run and inspect

The final remote source directory is `/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_v2_delivery`; prepared recording evidence remains `/home/physicalai/phantom-icra-2027/sim/waffles/evidence/fit/ep_waffles_1787395928_000`. The campaign launcher defaults to a plan and requires explicit `--execute` to run. Draft designs are rejected by the frozen-design loader. The parent owns simulator/model execution and final input paths; no hardware launch script is part of this workflow.

On compute3, define paths to the archived source and a **new** output directory. Keep the live virtual-environment Python path as written; resolving its symlink loses the environment. This launches simulator/model processes only:

```sh
SIM_BASE=/home/physicalai/phantom-icra-2027/sim/waffles
SIM_SOURCE="$SIM_BASE/source_teacher_v2_delivery"
SIM_PYTHON=/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python
SIM_OUTPUT="$SIM_BASE/runs/teacher_robustness_v2_delivery_repeat"
"$SIM_PYTHON" "$SIM_SOURCE/tools/sim/run_policy_campaign.py" \
  --campaign "$SIM_SOURCE/configs/sim/teacher_v2_delivery_screen.json" \
  --source "$SIM_SOURCE" \
  --live-repo /home/physicalai/phantom-icra-2027/phantom \
  --evidence "$SIM_BASE/evidence" \
  --hardware-config "$SIM_BASE/runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml" \
  --robot-usd "$SIM_BASE/runs/validation_v2/pick_place/robot_asset/waffles/waffles.usda" \
  --port 7799 --output "$SIM_OUTPUT/screen" --execute
```

The port must be free before launch. Do not run this while the original study is active. After all 32 cases complete, derive selection and the reserved confirmation configuration without modifying the archived source:

```sh
"$SIM_PYTHON" "$SIM_SOURCE/tools/sim/select_teacher_candidate.py" \
  --campaign "$SIM_SOURCE/configs/sim/teacher_v2_delivery_screen.json" \
  --runs "$SIM_OUTPUT/screen/rollouts" \
  --out "$SIM_OUTPUT/screen/selection" --stage screen
"$SIM_PYTHON" "$SIM_SOURCE/tools/sim/freeze_teacher_confirmation.py" \
  --protocol "$SIM_SOURCE/configs/sim/teacher_v2_delivery_protocol.json" \
  --screen "$SIM_SOURCE/configs/sim/teacher_v2_delivery_screen.json" \
  --selection "$SIM_OUTPUT/screen/selection/selection.json" \
  --out "$SIM_OUTPUT/teacher_v2_delivery_confirmation.json"
```

Run the campaign command again with `--campaign "$SIM_OUTPUT/teacher_v2_delivery_confirmation.json"` and `--output "$SIM_OUTPUT/confirmation"`; all other arguments stay the same. After its 24 cases finish, run the selector with that confirmation configuration, `--runs "$SIM_OUTPUT/confirmation/rollouts"`, `--out "$SIM_OUTPUT/confirmation/selection"` and `--stage confirmation`. Stop on any failed command and preserve partial outputs.

The original execution is stored at `$SIM_BASE/runs/teacher_robustness_v2_delivery`. Its `execution_plan.json` contains the exact five argument arrays, source archive hash and frozen protocol hashes; `study_ledger.jsonl` records subprocess exits. The source archive SHA256 is `5f4a0cf7027d00a1463c53e61e17e73919bc9453a5be1b733554672692be4f07`. Raw observations, measured states, proposed plans, accepted commands and per-trial review videos remain under the stage's `rollouts/` directory.

`selection.json`, `trials.csv` and `report.md` retain all stage outcomes, final support, terminal versus nonterminal safety events, completion reasons, holds/rejects and native/delivery/wall/activation latencies. All representative video choices follow scoring; no outcome is inferred from a selected video.

## Limits and exclusions

Only10 of250 available waffles demonstrations retain native timestamps, wrist data and images; the remaining240 are thin mirrors. These complete starts occupy a narrower workspace than the broader first-row distribution. All ten appear in expanded `val124`, but the complete expanded training manifest is absent: **do not claim training-unseen evaluation**. Nine starts share one August session and one comes from a nearby session, so confirmation does not demonstrate broad session or camera generalization.

The September deployment/post-hand packet fit, prior teacher-debug trajectories, prior40-trial teacher study, all student trials and every v2 preflight are excluded from the 56 scored cases. Camera extrinsics, packet/pad geometry, rigid-packet compliance and tactile/wrist force calibration remain unresolved physical-transfer limits. The August baseline uses a separate coherent static calibration capture from5928: left0.3052s and right0.4127s after first arm sample, with measured areas0. It is shared by all cases and is not causal q0 sensor history. Residual depth/wrench values are retained. This baseline is evidence for unloaded sensor appearance, not a validated tactile simulator. Physical measurements and sensor calibration in the lab remain necessary before interpreting these results as hardware performance.
