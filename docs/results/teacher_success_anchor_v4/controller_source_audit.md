# Combined controller and source audit

The declared bridge keeps the successful v1 scene, measured initial robot state,
Sept4 tactile baseline, hardware limits and scoring thresholds. It changes the
contact coverage to `manifold_patch_v2`, includes housing contact in the wrist
proxy, uses live terminal veto with current delivery feedback, and enables
release FINISH. This audit found no source-level obstruction to that profile.
It does not claim a bridge outcome or physical deployment readiness.

The exhaustive remote file scan found 219 shared files: 207 byte-identical and
12 changed. There are 21 added files. All 4131 v1-only files are copied artifacts,
tests or documentation; no runtime module or asset is missing. Scene creation,
kinematics, contact assets, SafetyMonitor, governor and tactile image proxy are
unchanged. The [JSON audit](controller_source_audit.json) records the full changed
file ledger, source hashes and input checks. The original and bridge hardware
manifests have different shapes but pin the same unchanged file hash.

Native `_note_executed_close` and `_pad_loads` match the live methods at AST level.
The only `_apply_veto` difference is the intentional substitution of simulated
delivery time for `time.perf_counter()`. Model observations retain their request
snapshot; enabled veto reads current measured TCP/aperture at delivery. Original
teacher openings can be restored only inside the configured release eligibility
window. Release and FINISH read robot TCP, gripper, tactile and latch state;
object pose and the independent success scorer never enter this controller.

FINISH **holds through the full requested 60 s** in the frozen simulator.
`run_waffles.py:1253` stops replanning on completion, whereas only
`command.stopped` at line 1301 sets the bounded two-second stop tail at line 1318.
The normal final tick remains the requested duration at line 1377. The adapter
holds the achieved measured TCP and continues live safety checks. Native
planner termination after its separate two-second `finish_observation_s` is a
different operational mode, not the scored simulator horizon. The bridge gate
therefore requires both development seeds to reach strict placement, normal
FINISH, no actual stop and actual 60 s trace coverage.

Compatibility is with the reviewed **proposed native opt-in source**. The live
compute3 checkout still lacks the shared release module and its executor,
planner, runtime and CLI hooks; this audit did not install them or run hardware.
The current asynchronous driver/servo path still needs physical qualification.
The rolling-calm wrench baseline remains unchanged; the separate fixed-reference
patch is not part of this bridge.

The gripper wrist proxy sums signed normal forces from housing and two pads,
with contact-point moments about measured TCP and the recorded initial bias.
It omits tangential force, gravity/inertia and UR3 current-based transfer. Both
that mapping and corrected gel coverage remain uncalibrated approximations.

This review binds [bridge_campaign.json](bridge_campaign.json), SHA256
`3445e7e051a2a1c14c9494035bc51d63f06948b62972617d9151b88fb45c8e71`,
and the prospective [v4 protocol](protocol.json), SHA256
`29d921b476807e68e129472d4090f47920af52ce60b6dc999d00f7fef820fe36`.
