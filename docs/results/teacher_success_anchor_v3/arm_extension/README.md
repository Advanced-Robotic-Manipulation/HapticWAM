# Conditional arm-start robustness extension

This is a frozen **planning specification**, not an executed campaign. Run one
teacher only if the completed reserved confirmation returns a non-null
`clear_simulator_winner` under the existing selection rules. Otherwise leave
this extension `not_run`; do not substitute a ranked leader, add seeds to find
wins or relax the placement/safety criteria.

There are **six total measured starts, including the anchor control**, each
paired with fresh sampling seeds **904601 and 904602**: 12 episodes if enabled.
The two anchor episodes are reported separately from the ten varied-start
episodes. Keep the winner's confirmed checkpoint/EMA, inference recipe, source,
controller, safety limits and 60 s horizon unchanged. This extension cannot
rerank the core study or become another model-selection round.

| Measured start | Role | TCP XYZ (m) | Recorded closure | Worst native σ |
|---|---|---|---:|---:|
| Sept4 teacher 5016 | Anchor control | −0.3540, −0.3031, 0.3381 | 0.0784 | 1.440 |
| Aug22 5928 | Varied start | −0.3354, −0.2998, 0.3563 | 0.4275 | 1.432 |
| Aug22 6273 | Varied start | −0.3370, −0.3021, 0.2970 | 0.3725 | 1.388 |
| Aug22 6128 | Varied start | −0.3455, −0.3093, 0.3290 | 0.4275 | 1.286 |
| Aug22 5963 | Varied start | −0.3496, −0.2751, 0.3504 | 0.3137 | 1.107 |
| Aug22 6461 | Varied start | −0.3293, −0.3036, 0.3246 | 0.2549 | 1.333 |

The anchor is mandatory. The other five are chosen deterministically from the
ten existing authentic Aug22 initial-state files by greedy maximin distance in
the native gate's standardized TCP, raw joints and gripper coordinates. The
complete candidate pool, source hashes, causal sample checks, gate values and
each selection step are saved in [start_pool_audit.json](start_pool_audit.json).
No new core outcome files were read to choose starts. This freeze occurs while
the core study runs; some recordings were already used in earlier v2 studies.

All 11 available measured starts pass the unchanged native **2.5σ** start gate.
The gate handles equivalent rotation vectors and uses unwrapped raw joints with
the existing 5° joint-standard-deviation floor. It is a distribution check,
not proof of clearance or physical readiness. The selected set spans
24.65/34.15/59.37 mm in TCP X/Y/Z and up to 0.2501 rad on an individual joint.

Each case retains its **own recorded q, gripper and raw wrist bias**. No random
joint jitter, IK-branch swap, forced common aperture or scripted approach is
allowed. This varies a measured initialization bundle, so changes cannot be
attributed to arm pose alone. The simulator still performs its normal settling;
source qd is evidence rather than an imposed starting velocity.

The successful anchor packet pose/yaw/texture, table, mat, bin, camera, lighting
and materials stay fixed. All six cases use the same Sept4 no-contact tactile
baseline (`80be5522…afdfce`) and the winner's confirmed sensor mappings. Moving
the robot changes its rendered image naturally. Aug22 raw wrist biases combined
with a shared Sept4 tactile baseline are an explicit approximation, not a
fully synchronized replay of each recording's sensors.

Before enabling inference, verify each exact input hash and unchanged-scene
initialization, retain native gate and runtime settling limits, and save the
initial q/gripper/packet state and image. A blocked or invalid start stays in
the planned denominator; preserve its evidence and do not replace its pose.
The launch manifest must bind the selected teacher/source only after reserved
confirmation, without editing the frozen start list or seeds.

Report frozen strict physical placement and final support, all intermediate
stages, drops, pre/post-placement safety stops, timing and normal FINISH
separately. Show both seed outcomes for every start. Aggregate the five varied
starts separately from the anchor control; any optional interval must preserve
the two-seed start clusters and disclose the small five-cluster sample. These
are fixed-suite simulator results, not hardware success probabilities or
unseen-session validation.

[protocol.json](protocol.json) contains the complete 12-case plan, source/input
identities, winner gate, execution order, inherited physical/scoring settings
and failure handling. [manifest.json](manifest.json) pins this specification.
No GPU, inference server or physical hardware is launched by this preparation.
