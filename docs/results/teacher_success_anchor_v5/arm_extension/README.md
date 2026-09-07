# Conditional v5 measured-start extension

This [prospective amendment](amendment_v5.json) binds the unchanged measured-start
suite to v5 protocol SHA256
`f67fe79a5267c6d253d7d90f67f2a367c8899f030959c8f42fa714d072ab7ac2`.
It remains **not run**, with no selected model. Execute only if the completed,
valid 24-trial reserved v5 confirmation returns a non-null
`clear_simulator_winner` under every original winner gate. A ranked leader alone
does not qualify. These outcomes cannot rerank models or supply missing evidence
for the original winner declaration.

The original [v3 specification](../../teacher_success_anchor_v3/arm_extension/protocol.json),
candidate-pool audit and v4 amendment are unchanged. There are **six starts total**:
the Sept4 anchor control, followed by measured Aug22 starts 5928, 6273, 6128,
5963 and 6461. Use seed **904601** in that order, then **904602** in reverse
order. This is exactly 12 trials: two anchor controls and ten varied starts.
No new pose selection, random joint jitter, joint wrapping, common forced
aperture or scripted approach is allowed.

Each start retains its original measured q, closure and raw wrist-bias bundle.
The selected spread is unchanged: TCP X/Y/Z spans 24.65/34.15/59.37 mm;
maximum per-joint spread is 0.2501 rad; closure spans 0.0784–0.4275. All selected
states pass the original 2.5-sigma native gate (worst 1.43975 sigma). This is a
distribution check, not collision clearance or physical qualification.

Keep the exact successful Sept4 packet pose, scene, camera, lighting, materials
and shared Sept4 tactile baseline. Use the winning checkpoint and every confirmed
inference setting with immutable `source_teacher_anchor_minimal_v5`: gel coverage
`manifold_patch_v2`, original `contact_proxy` wrist, historical `fd4a032` veto
with request-snapshot feedback, and policy release/FINISH. Do not substitute the
new local native port, live/current-delivery veto, gripper-assembly wrist,
fixed wrist reference or a reach-limiter variant. FINISH holds through the common
60-second simulator horizon unless an actual safety stop occurs.

Combining an Aug22 measured wrist bias with the common Sept4 tactile baseline
is an explicit approximation. This measures sensitivity to the recorded robot,
gripper and wrist initialization bundle, not arm pose alone or unseen-session
sensor transfer.

[Command derivation](command_derivation.md) describes the mechanical post-selection
binding and plan-only review. Input hashes and all inherited field digests are
in the amendment; the [manifest](manifest.json) pins this separate addition.
Preserve all 12 scheduled keys, including rejected or invalid initializations.
Report the ten varied-start trials separately from the two controls, including
both seeds, strict placement/support, FINISH, stage times, drops and all stops.
No GPU, model server or hardware was launched to prepare this amendment.
