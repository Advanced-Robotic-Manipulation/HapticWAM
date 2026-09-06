# Post-experiment tactile force diagnostic

The nominal video's low gel force is produced by the frozen mapper's **sparse-manifold fallback**, not by shear, force-vector cancellation, a normal-axis sign error, or the 124 ms sample age. The successful case uses its area-overlap branch. The implementation and saved arrays agree exactly, but that agreement does **not** validate the physical pressure distribution. This diagnostic changes no source, rollout, or score; the campaign remains 5/40 strict full-task successes.

| Case | Video/scene time | Causal tactile capture | Packet normal force in scene trace, left/right | Packet normal force at tactile capture, left/right | Saved gel normal force, left/right |
|---|---:|---:|---:|---:|---:|
| Nominal, seed 4242 |17.000s|16.876s|5.211847/5.214882N|5.222712/5.192275N|0/0.245378N|
| Placement −10 mm X/−10 mm Y, seed 4242 |18.000s|17.876s|6.194927/6.336052N|6.135059/6.281293N|4.685056/5.037976N|

At 16.876 s, both nominal pads have four recorded packet manifold vertices, but only two have strictly positive constraint force. The mapper drops the two zero-force vertices before forming the support hull. Two remaining points cannot form a nonzero-area polygon, so the explicitly documented `point_fallback` selects individual points inside the active ellipse.

- Left loaded points in pad(Y,Z) coordinates are (−8.402,+20.785) mm carrying 3.669394N and(+10.230,−15.558) mm carrying 1.553318N. Both are outside the ellipse, giving zero gel force.
- Right loaded points are(+8.951,+19.280) mm carrying 4.946168N, outside the ellipse, and(+12.772,−0.163) mm carrying 0.246107N, inside. The accepted point's normal alignment is 0.997040, giving0.245378N.
- All loaded contacts pass the inner-face and normal-alignment tests. Normal projection would remove only 0.000006 N from the entire left force and 0.015369 N from the entire right force before area filtering. It cannot explain the missing ≈5 N.
- The neighboring tactile samples 16.628, 16.756, and 17.004 s show the same two-point fallback and low gel load. The difference therefore persists across the cited scene/tactile timestamp gap.

At 17.876 s, the successful placement case has four strictly positive-force vertices per pad. Its convex support polygons have areas 400.995/156.147 mm²; their overlaps with the active ellipse are 306.221/125.239 mm², fractions 0.763653/0.802061. Multiplying each pad's total projected normal force by these fractions reproduces 4.685056/5.037976 N. Projection losses are below 0.000003 N per pad.

Forces here are **simulated newtons**, from PhysX normal-contact impulses divided by physics timestep. The physical packet trace sums populated pad/packet constraint magnitudes, including any backing/linkage contacts. The mapper instead computes `Σ(abs(force) × abs(unit_normal_pad_X) × area_fraction)`. Pad X is the closing axis; the inner face is X = −6 mm (left) or +6 mm (right), with 2 mm tolerance. The active ellipse satisfies `(padY/13.5mm)²+(padZ/18mm)²≤1`. Image columns follow +padZ and rows +padY; the proxy adds compression as negative SDK wrench Z. SDK image-axis orientation and pressure transfer remain uncalibrated. Tangential force is not observed by this normal-contact API and is not inferred from world net force.

The remote raw `policy_tactile.npz` files were checked directly: all 144 nominal and 252 placement timestamps and gel-force rows exactly match `gel_contact_trace.json` (maximum error 0). Re-running the frozen mapper on every saved contact record in pad coordinates also gives maximum force error 0. Both videos use saved uint8 gel images of shape [2, 288, 384]. The JSON includes input and frozen-source SHA256 hashes, exact per-contact points/normals/forces, filtering decisions, nearby samples, and selected gel-frame hashes. No hardware or simulator/model execution was used for this audit.

A bounded hypothesis for a **future isolated mapper version** is to test whether valid zero-force geometric manifold vertices should define support when the same body's total normal load is positive. This may remove the two-point branch discontinuity, but zero-force vertices do not prove that the enclosed region bears pressure; including them may overestimate active gel load. Compare any such change with measured contact/pressure evidence or a calibrated compliant-contact model. Do not substitute whole-pad net force for gel load, apply this hypothesis to frozen scores, or attribute the nominal outcome to this mechanism alone.

Reproduction helper: [audit_gel_force_projection.py](audit_gel_force_projection.py). Full machine-readable evidence: [post_experiment_tactile_diagnostic.json](post_experiment_tactile_diagnostic.json). Run the helper with `--root` pointing to the campaign parent retaining raw tactile NPZ and `--source` pointing to its frozen source.
