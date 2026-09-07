# Fresh September 4 mapper validation

**The fresh measured-motion replay passes observer invariance, contact provenance and force-accounting checks.** All 19 state/contact arrays from the previously validated `validation_v2/lift` replay are bitwise identical across 221 frames spanning 0–14.664 s. Two independently observed support arrays are newly present. Scene configuration, camera and robot asset match; no motion or collision behavior changed.

The [numeric audit](mapper_fresh_sept4_audit.json) reconstructs V1 and V2 from the same actual populated contact records. Its 442 pad readouts contain 853 unique contacts, including 181 zero-impulse entries. All separations are supplied and finite, from −11.481 to +1.996 mm. No vertex exceeds the fixed 2 mm contact-generation envelope, and no duplicate filter indices occur.

There are 134 eligible Waffle patch readouts, 20 using zero-impulse geometry. Sixty-three readouts fall back because no eligible positive load exists; 23 mat readouts use the undeclared-body point fallback. Every recomputed V2 gel force and UV exactly equals the saved value; all force-accounting residuals are below 9×10⁻¹⁶ N.

Twenty pad readouts differ between V1 and V2, with maximum absolute difference 1.348 N. Per-pad peak gel force is unchanged at `[4.266, 4.881]` N. Both versions have 44 scene samples with bilateral gel force at least 2.5 N. These are scene-rate diagnostics, not the live policy's 8 Hz latch stream and not evidence of improved closed-loop success.

The deepest contact at 9.336 s is a left-pad/Waffle point with −11.481 mm signed separation and 101.515 N load. Although its X position is near the inner face, its normal is mostly tangent to the gel. It remains excluded from gel pressure, and the physical load is retained. This deep penetration is a contact-model limitation; a correct mapper does not make it physically calibrated.

Uniform support-patch pressure, speculative-gap interpretation, active gel dimensions and image orientation remain explicit assumptions. See the [August audit](mapper_fresh_replay_audit.md) for the fixed contact-offset gate, API sources and force-conservation semantics. The same [read-only CPU helper](audit_mapper_fresh_replay.py) generated this result using fresh directory `teacher_v2_preflights/dynamics_sept4_mapper_v2`, reference `validation_v2/lift`, and frozen source `source_teacher_v2_mechanics1` under the compute3 simulator workspace. No GPU work was launched by this audit.
