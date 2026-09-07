# Teacher decision after restoring the successful waffles experiment

**Use `teacher_v5_ftA/teacher_001500.pt`, EMA, NFE1, K4, guidance1 as the next teacher candidate.** Keep persistent noise, parity, task `waffles`, maximum10 played actions, and the explicit minimal_v5 release/FINISH profile. This is a concrete candidate recommendation; full-task reliability has not been established.

The earlier change of scene and starting-state distribution was a confound. We returned to the original successful packet pose, geometry, camera and measured start, reproduced the old accepted commands, then compared checkpoint/recipe choices with those conditions held fixed. The original successful video remains a physical success. A subsequent stop is reported separately and does not erase a completed placement.

## Completed comparison

All24 screen cases and all24 reserved confirmation cases passed independent raw-data, source and scoring audits. The four screen recipes were ftA1500/NFE1/K4, ftA3000/NFE1/K4, v5_6/NFE1/K4 and ftA1500/NFE5/K4. The two selected checkpoints then received12 matched fresh sampling seeds each.

| Confirmation only | Acquired | Sustained lift | Carry | Full box placement | Drop |
|---|---:|---:|---:|---:|---:|
| ftA1500 / NFE1 / K4 |12/12|7/12|6/12|1/12|1/12|
| ftA3000 / NFE1 / K4 |8/12|4/12|4/12|0/12|0/12|

The placement difference is +8.33 percentage points, paired95% interval [0,+25], exact McNemar p=1. The frozen reliable-winner gates fail. The recommendation follows the observed ranking and stronger pickup progression, with that uncertainty disclosed. See the [complete confirmation audit](results/teacher_success_anchor_v5/confirmation/independent_audit.md); screen and development outcomes are never pooled into this denominator.

Seed904510 supplies a new closed-loop physical success: placement at24.268s, FINISH at25.076s, unloaded packet supported in the box through60s, no drop or stop. The teacher initiated opening; the historical recovery rule contributed additional opening after physical placement. [Exact event and contact audit](results/teacher_success_anchor_v5/selected_success_audit.md). All48 comparison videos include the saved tactile inputs and passed full decoding and hash checks; the successful clip is an illustration of one trial.

## Changes to keep and changes to withhold

Keep the sparse-contact gel-coverage correction and post-release FINISH hold. The mapping correction is supported by contact-conservation tests; FINISH is exercised by the new full-horizon successful run. They do not make the simulated tactile response calibrated to real gel.

Do not promote the current reach-limiter profile. Both separate development cases acquired, lifted and carried with its command bounds respected, but both stopped before release. Exactly25 constraint holds consumed roughly0.2s, cancelling pending teacher responses before they could be evaluated. The IK solutions existed and accepted-command feedback was correct: this is a hold-handling limitation. A longer bounded hold is a future intervention to test, not a demonstrated recovery. [V6 evidence and decision](results/teacher_success_anchor_v6/diagnostic_results/recommendation.md).

The explicit complete-client timing option and K1/K4 diagnostic are documented separately in the [V7 record](results/teacher_success_anchor_v7/README.md). They do not rewrite V5 results or change the recommended hardware settings automatically. The older runner's elapsed inference field includes observation callbacks/bookkeeping; its difference from native inference cannot be attributed wholly to RPC transport.

The completed four-case diagnostic favors retaining K4: it lifted/carried2/2 and placed1/2, while K1 acquired2/2 but never lifted. K1's mean full-client duration was0.279s versus0.848s for K4, so faster responses alone did not improve these trajectories. Both recipes had one wrist-extension stop and no drops. These reused development seeds are excluded from the confirmation denominator.

## Remaining weaknesses and next experiment

Ten of the twelve ftA1500 confirmation trials stopped on wrist extension. Acquisition alone is therefore insufficient; the approach/lift/carry trajectory must remain feasible through release. Contact geometry and force transfer also need calibration: one failed simulated approach developed sustained high peripheral pad loads, while the rolling wrist reference absorbed part of the gradual load. Successful trajectories had much smaller loads. Do not infer a safe hardware force envelope from these simulated values.

Measure table/base and TCP/pad geometry, aperture, packet and box dimensions/poses, camera calibration, and unloaded/known-load sensor behavior at the next bench session. Then qualify one attended fixed-start pickup and release using the recommended teacher. For the next declared pilot, hold the waffle/box/camera fixed and vary authentic measured arm/gripper starts; this matches the collection practice. Test measured object-reset jitter afterward as a separate factor. Arbitrary joint randomization and simultaneous object randomization would obscure the current failure mechanism.

The [operator handoff](isaac_lab_handoff_20260907.md) includes checkpoint/hash, source/CLI settings, measurement sheet, release configuration boundary and a proposed six-start/two-seed pilot. The native minimal_v5 port has CPU parity evidence and requires explicit opt-in; no hardware was started or updated. The gated simulator arm-start extension remains unexecuted because its declared winner requirement was not met.

Source, audits and handoff are on [PR14](https://github.com/Advanced-Robotic-Manipulation/phantom/pull/14). Frozen mixed-source archives reproduce the scored simulator; a repository commit alone does not substitute for those manifests. Raw recordings and older studies remain unchanged.
