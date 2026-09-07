# Completed screen audit

**Passed: all24 raw trials rescored with zero discrepancies.** Every selected trial field, input hash, complete score hash and aggregate ranking matched the authoritative [selection](selection.json). All24 are valid, original attempts under the unchanged thresholds. [The machine-readable audit](screen_independent_audit.json) retains per-trial evidence.

| Teacher/recipe | Acquired | Lifted/carried | Physical place | Clean FINISH | Controller stops | Mean native latency |
|---|---:|---:|---:|---:|---:|---:|
| ftA1500 NFE1/K4 | 5/6 | 4/6 | 0/6 | 0/6 | 6/6 | 0.853s |
| ftA3000 NFE1/K4 | 5/6 | 3/6 | 0/6 | 0/6 | 6/6 | 0.850s |
| v5_6 NFE1/K4 | 1/6 | 1/6 | 0/6 | 0/6 | 6/6 | 0.872s |
| ftA1500 NFE5/K4 | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 | 3.911s |

All eight lifts also achieved the declared carry criterion. No trial achieved release/placement or recorded a drop. The latency column is the mean of six per-episode native latency means. Delivery-inclusive values equal native values because no additional delay was injected; measured wall-call means were1.039,1.043,1.077 and4.123s respectively.

The exact first terminal safety events were:

- ftA1500 NFE1/K4: five`wrist_extension`, one`hitbox_exit_top`.
- ftA3000 NFE1/K4: four`wrist_extension`, two`wrench_limit`.
- v5_6 NFE1/K4: three`wrench_limit`, one each`wrist_extension`, `joint_speed` and`hitbox_exit_top`.
- ftA1500 NFE5/K4: six horizon endings without a controller stop.

There were no tactile force/depth limit stops. The schema's tactile-force category includes both force and depth; it does not silently exclude depth. All18 safety stops occurred before placement. Absence of stops in the slower NFE5 recipe accompanies zero lifts and does not establish safety equivalence or a task winner.

The independently reconstructed frozen ranking correctly advances **ftA1500 NFE1/K4 and ftA3000 NFE1/K4**. Both have zero placements in this screen; the decision is a candidate selection, not a winner claim. The [derived confirmation](confirmation_campaign.json), SHA`fe62f45fcaf2e71e4df890ff7303bf70a2b49e35fc2a745f6f7492a5f200b855`, preserves all shared scene/profile/input values and uses the twelve untouched paired seeds904501–904512.

The automatic transition passed its source and matching checks and started confirmation controller PID2415593. [The launch snapshot](transition_launch_binding_snapshot.json), [plan](confirmation_launch/plan.json) and [frozen inputs](confirmation_launch/frozen_inputs.json) preserve that handoff. This snapshot is not a live progress file. Final conclusions await the complete reserved confirmation and its unchanged paired winner gates.

The read-only audit helper is [audit_completed_screen.py](../audit_completed_screen.py), SHA`d8fc6a03617ac43f9d22c41be9f4f92334966c15231923c52c278a25703e916f`. Its temporary rescore files were discarded. No primary scores, frozen source, runtime process, hardware or video were changed by this audit. The separately generated24 video reviews are indexed in [the video manifest](../screen_video_manifest.json).
