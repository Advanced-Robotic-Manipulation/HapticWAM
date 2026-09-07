# Isolated limiter source audit

Prepared source: `/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_anchor_limiter_v6`. No simulator, inference or hardware run was launched during this preparation. The separate diagnostic remains conditional on the running v5 confirmation finishing.

[The source manifest](source_manifest.json) records4350 original non-cache files and4351 output files. Exactly two paths differ: the new `phantom/drivers/servo_limiter.py` and a bounded backport into `tools/sim/run_waffles.py`. The [runner patch](runner.patch) retains the minimal profile's gel mapping, FINISH implementation, historical request-time veto, pad-only wrist proxy, scene and physics. All original files were rehashed after assembly and remained unchanged. The copy uses independent files rather than hardlinks.

| Pin | SHA256 |
|---|---|
| Source manifest | `df54897c11d4e1b6774c76e07382f67b0e9e31c0724d43a3deff50b2b13fc9b6` |
| Canonical output-file digest | `88dce60048c385d087bce1a73c6db7c0ef8c62783d95f48962210136c40e6527` |
| Backported runner | `903eae23fe5f09e3b1b5fc6c2153edc3de1d43619d0f2719201eb5f373443be2` |
| Shared limiter | `41c0c6beebd805dda4d9bf7c66e507491eb1121cdb5cf70a701ce845b37e027f` |

After removing only the declared disabled hooks, the entire output runner AST equals the original minimal runner AST. The original IK path remains intact when the flag is absent. This is source-path equivalence, not a claim of bitwise repeatability for new rendered policy rollouts.

The independent review found no additional defect for the declared principal elbow branches. The shared helper retains the audited native branch rejection, finite IK checks, three-step bisection, optional tangent search and commanded-speed predicate. The intended values are0.40rad minimum elbow magnitude and1.0rad/s commanded joint speed. The tangent is a candidate direction; it is not a projection of the wrist center. This constrains proposed joint targets, not measured contact motion, braking distance or tracking error. Existing77 CPU checks cover native extraction and full UR3 geometry; [their source report](../teacher_success_anchor_v5/prospective_servo_limiter.md) records the evidence and limitations.

The native UR driver was deliberately preserved. Its frozen minimal SHA`2ee975a3cce284d6197199d5d07f0b440c7ae196f464be5ec3982cd028b0f731` differs from the audited extraction base`b24cbdb9c21b6a5b113d59145e08b716655d9036f9bc2bc371e614f4879587a0`. The simulator calls the shared helper directly; this source copy does not claim a backported native-driver delegation.

[The additional CPU feedback audit](cpu_feedback_audit.json) imported the actual assembled runner and minimal adapter without constructing Isaac or a model. The flag defaults off and parses when explicit. Twenty-five consecutive rejected commands request`servo_limiter_stall` at0.192s; the next stopped tick at0.2s preserves the0.61 commanded closure. The accepted stop hold resets the consecutive counter, while the stop reason remains recorded. The runner uses its unchanged`min(t + 2.0, duration)` observation tail. No exception was used to classify the rejection as infrastructure failure. A complete Isaac trial and strict runtime scoring remain separate validation steps.

Activation adds only`--servo-reach-limiter` to the policy launcher using this new source. The diagnostic must pin and pass the same hardware YAML to server and simulator with exactly`elbow_min_rad: 0.40` and`servo_joint_speed_max_rad_s: 1.0`; the hook reports those effective fields. Measured wrist-extension, force, tactile and all other thresholds remain unchanged. The existing campaign continues using its frozen source and hardware.

Reproduction helpers are [assemble_limiter_source.py](assemble_limiter_source.py) and [audit_limiter_cpu.py](audit_limiter_cpu.py). Their remote copies and pinned donor inputs reside under`runs/teacher_success_anchor_v6/source_inputs/`; the manifest and CPU output reside under`source_audit/`. The assembler refuses an existing output/audit directory and refuses writes through inherited symlinks.
