# Minimal gel-v2 plus FINISH source audit

The new `source_teacher_anchor_minimal_v5` is an independent full copy of the
immutable successful v1 source with exactly four reviewed replacements:

| Module | Donor | SHA256 |
|---|---|---|
| `tools/sim/run_waffles.py` | `source_teacher_anchor_gel_v3` | `b5184eb802ed2d99e1ac2f678fe485b810c72d878b0a52f330d4fec9423ca923` |
| `tools/sim/gel_contact.py` | `source_teacher_anchor_gel_v3` | `d7f4cb1d4606aa0ced6d360be6073498e5505460248d75cf6d04c7cf52853d6e` |
| `phantom/sim/policy_adapter.py` | `source_teacher_anchor_finish_v3` | `97adde55eb1b6e619054c45426e571ab08fe39f5f2da16e51b3b5a02022eae88` |
| `phantom/sim/release_controller.py` | `source_teacher_anchor_finish_v3` | `9719c8fc98d8eb390c67e81462fbf48f926d017b27bfe62f5eedeb51162c0284` |

The runner change only permits `manifold_patch_v2` in its argument choices. Its
runtime mechanics remain the original code. The separate campaign must select
that coverage mode, retain `contact_proxy` wrist feedback and historical
`fd4a032` veto/request feedback, and set `finish_after_release: true` in the
otherwise unchanged release configuration. The full-copy file hash ledger is
[minimal_profile_source_manifest.json](minimal_profile_source_manifest.json).

The FINISH overlay preserves the v1 constructor, observation/inference path,
original teacher-opening eligibility, terminal veto and latch. Before FINISH,
target integration, blend, accepted-command history, safety and rate-limiting
arithmetic remain the same. New diagnostic fields and a pure measured-pose
clamp eligibility check do not alter those commands. Gel-v2 does change tactile
feedback and can therefore change policy behavior before release; it is not a
claim of pre-release parity with the uncorrected gel model.

The only new terminal branch follows the existing policy-commanded release,
a previously loaded latch, and measured open/unloaded dwell. It additionally
requires an accepted open command and a measured TCP inside the unchanged
clamp. It then records `placement_release_finished`, freezes the achieved
measured pose and accepted opening, clears pending work and disables replanning.
Safety is checked on every subsequent control tick and can still stop the hold.
There is no object pose, support score or success label input to the controller.

The old runner keeps physics and sensors active through its declared 60 s
horizon after normal FINISH. Only an actual stopped command creates its bounded
two-second safety tail. Completion is logged in `execution_trace.jsonl` under
`diagnostics.completed_reason`, `completed_at_s` and `completion_hold`, plus the
release diagnostics. This runner has no top-level completion field in `run.json`;
review tools must use those saved execution fields.

Six focused CPU checks passed: unchanged two-module base/overlay hashing,
bitwise commands with FINISH disabled through release and rearm, bitwise
commands with FINISH enabled until its terminal transition, achieved-pose hold
and subsequent safety preemption, no completion without loaded policy release,
and unchanged gel-runner mechanics. These tests use fake observations, not GPU
or hardware. The hash-pinned assembly helper copies and verifies source only;
it does not launch a runtime.

The profile excludes the gripper assembly wrist proxy, current-delivery veto,
live veto recovery, native shared-release hooks, fixed wrist baseline and
recorded latency overrides. It retains the old pad-only wrist observability
limitation and uncalibrated pressure/contact transfer. It is a deliberately
isolated simulator diagnostic profile, not a claim that every latest native
hardware correction is installed or validated. Physical release support and
hardware qualification remain separate.
