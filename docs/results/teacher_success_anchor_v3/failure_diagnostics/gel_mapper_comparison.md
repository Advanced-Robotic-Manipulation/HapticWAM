# Gel v2 component: same-contact mechanism and live outcomes

Both completed gel-only trials achieved support-verified placement with unchanged task thresholds and no detected drop. The matched-seed baselines carried but stopped before release. This is a diagnostic component result; differing RGB and measured inference timing prevent attributing the entire live trajectory difference to the mapper.

| Seed | Profile | First latch s | Lift / carry s | Release / full s | First executor guard s |
|---|---|---:|---|---|---|
| 904301 | baseline | 9.380 | 10.068 / 12.668 | — / — | wrist_extension 14.780 |
| 904301 | gel v2 | 8.252 | 9.868 / 12.400 | 23.068 / 23.736 | wrist_extension 54.156 |
| 904302 | baseline | 9.500 | 11.200 / 13.668 | — / — | wrist_extension 15.692 |
| 904302 | gel v2 | 11.252 | 11.668 / 13.736 | 21.400 / 22.136 | wrist_extension 27.900 |

The gel cases first entered the release window at 16.036/16.948 s and first played original-eligible opening commands at 22.164/21.172 s. Both later wrist-extension stops occur after completed placement. FINISH is absent from these gel-only cases; adding that terminal behavior would require a separately declared profile/validation.

## What is established on identical contacts

The [read-only CPU reconstruction](audit_gel_same_contacts.py) feeds exactly the same saved force, local point, local normal, populated counts and real separation values to the frozen v1 mapper and the preserved v1/v2 branches. Across 434 and 224 tactile samples before stop, the reconstructed v2 loads equal the recorded loads exactly. The preserved v1 branch also equals the actual frozen v1 implementation exactly. Maximum force-partition residual is below 1e-15 N. No new force is assigned to zero-impulse support points.

At seed 904301 t8.252 s, each pad has four populated geometric vertices. Only one left and two right vertices carry positive solver force. Their support hulls overlap the declared active ellipse, although the loaded points individually lie outside it:

| Pad | Packet normal total N | Loaded / geometric vertices | Active overlap fraction | Frozen v1 gel N | V2 gel N |
|---|---:|---:|---:|---:|---:|
| Left | 3.766494 | 1 / 4 | .786180 | 0 | 2.943483 |
| Right | 3.555467 | 2 / 4 | .856047 | 0 | 3.025496 |

V1 loses the hull when zero-impulse vertices are omitted and falls back to outside-ellipse points. V2 keeps those vertices as geometry, with all separations inside the fixed 2 mm contact-generation envelope. Their saved separations range from −.525 to +1.632 mm on the left and −.297 to +1.379 mm on the right. Positive separation is speculative contact geometry, **not measured gel indentation**. The loaded forces remain the only source of pressure. Uniform pressure over the convex support patch is still an uncalibrated assumption.

On this same recorded trajectory, v2 first exceeds 2.5 N bilaterally at 8.252 s; frozen v1 would first do so at 9.252 s. This is a sensor-only counterfactual on a fixed trace, not a prediction of the trajectory under v1. Four samples exceed the bilateral criterion under v2 but not v1.

Seed 904302 has the same first bilateral criterion time under both mappers (11.252 s), but nine later samples cross it only under v2. At18.132 s v1 reports zero on both pads, while v2 reports 3.435/4.615 N from packet totals 4.045/5.373 N. Here each side again has only two loaded vertices among four support vertices. This demonstrates a later sensor-history difference even when initial latch timing is unchanged.

## What cannot be concluded

The original baseline 904301 already latched at 9.380 s and retained 4.59/4.80 N physical packet load just before its wrist-extension stop. Baseline 904302 also already latched and retained its object. Their failures therefore cannot be summarized as “old mapper never latched.” The revised mapping can affect model tactile observations and controller load history, but this audit does not isolate which changed the transport trajectory.

Startup confounds are measured, not hypothetical: both pairs have byte-identical scene configurations and identical initialization JSON; all eight non-RGB arrays in the first observation are bitwise equal. RGB differs by RMS .923/1.004 uint8 levels (maximum 27/38). First raw translation rows already differ by up to 14.9/22.4 µm before contact; first plan activation is 1.452→1.460 s and 1.468→1.436 s. Native timing and subsequent image histories are therefore unmatched before the mapper becomes active.

The defensible result is a reproduced sparse-support defect in v1, a conservative force-preserving v2 response on the same saved geometry, and two successful fresh v2 placements under the declared simulator. It is not a calibrated tactile model or proof of reliable hardware transfer. Preserve baseline results and finish the frozen study; separately validate any selected gel-plus-FINISH profile with fresh trials and unchanged safety.

[Live case/stage audit](gel_v2_cases.json), [same-contact numeric audit](gel_same_contacts.json), and [initial eight-case report](eight_cases.md) include source/input hashes. The mapper hashes are d7f4cb1d…853d6e (v2) and 99791534…a66f300 (frozen v1). All analysis was CPU-only, read-only against completed raw trials.
