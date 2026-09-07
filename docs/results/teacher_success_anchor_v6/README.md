# Command-limiter development diagnostic

Two previously observed development seeds (904301 and 904302) test ftA1500 EMA, NFE1, K4 in the unchanged successful physical scene with the existing native command limiter enabled. This is separate from the frozen v5 model screen and confirmation. No v6 GPU trial has started.

The new hardware file changes exactly `safety.elbow_min_rad` from null to 0.40 and `safety.servo_joint_speed_max_rad_s` from null to 1.0. The measured wrist stop remains 0.468 m; force, speed-stop, workspace, tactile and release gates stay fixed. The campaign explicitly enables `adapter_profile.servo_reach_limiter`; the runtime records actual parameters and accepted commands.

[The two-cell campaign](limiter_diagnostic_campaign.json) preserves the parent nominal scene plus its single −10 mm X/Y condition, yielding the same already-verified effective scene. [Hardware provenance](hardware_limiter_manifest.json) records the exact two-field change. The source overlay and command plan are bound in [the CPU preflight](diagnostic_preflight.json), [full runtime source audit](source_manifest.json), and [external driver manifest](external_driver_manifest.json). The independent driver uses commit `08baa87` campaign logic, with its complete mixed-snapshot inventory pinned. The runtime is the prior minimal source with exactly two file changes (the shared helper and its default-off runner hook); disabled-path AST equality passed.

Report both outcomes, including failures. Inspect acquisition, sustained lift, carry, physical release and bin support; measured wrist radius; commanded elbow margin and joint steps; all limiter holds/rejects and final stop reasons. These development outcomes cannot establish a model winner. Promotion to a broader comparison requires a separate prospective plan.

The v5 screen and confirmation finish first; they use their original immutable source and configuration. No hardware control is authorized by this simulator experiment.

Compact runtime and driver tar archives are available on compute3 under `runs/teacher_success_anchor_v6/reproduction_archives/`. [The archive manifest](reproduction_archives_manifest.json) includes archive hashes and every file hash; both archives were reopened and verified. Checkpoints, recordings and Isaac remain external hash-pinned dependencies.
