# First divergence in the two seed4242 repeats

The two completed repeats use the exact original scene and settled state, but do not reproduce its placement. **The earliest measured input difference is RGB at the first observation, before any policy command.** All eight other saved input arrays are bitwise equal. Raw policy outputs already differ slightly; delivery timing then differs by −24/+16 ms and changes subsequent observations. These runs cannot isolate native model nondeterminism, renderer variability or timing as the sole cause of the final divergence.

This is a read-only CPU comparison of `teacher__anchor_repeat1__seed4242` and `teacher__anchor_repeat2__seed4242` against the original `teacher__placement_xm10_ym10mm__seed4242`. It excludes the seed4243 cases and changes no scores or frozen inputs. [Detailed differences and hashes](reproduction_first_divergence.json) and [contact/latch details](reproduction_contact_detail.json) preserve the evidence.

| Quantity | Original | Repeat1 | Repeat2 |
| --- | ---: | ---: | ---: |
| First observation, s | 0.260 | 0.260 | 0.260 |
| First activation, s | 1.484 | 1.460 | 1.500 |
| Acquisition, s | 8.000 | 8.936 | 8.068 |
| Sustained lift, s | 12.668 | 12.536 | 12.200 |
| Carry, s | 15.336 | 15.000 | None |
| Drop, s | None | None | 13.200 |
| Full placement, s | 25.200 | None | None |
| First safety stop, s | 31.380 | 17.044 | 16.892 |
| Stop event | Wrist extension | Wrist extension | Wrist extension |

The repeats' output durations19.040/18.888 s include the two-second observation period after stopping. They are not the first safety-stop times. Both repeats pass the frozen validity checks.

## State, image and action differences

Effective scene dictionaries and file hashes match exactly. Configured and settled q, settled qd, packet position and packet velocity are identical. At t=0.260, `t`, `ur_state`, `wrist_window`, `gel`, `fields`, `contact_state`, `reactive` and `prev_chunk` are bitwise identical to the original. Camera and tactile sample timestamps also match.

RGB is different: original-versus-repeat RMS error is0.9218/0.9211 levels on the0–255 scale, maximum30/25. Between the new repeats it is0.9090 RMS, maximum21. These are small pixel differences with otherwise matching scene state; their source has not been isolated. They must not be silently replaced or dismissed as an identical model input.

The first16×7 raw policy heads differ before veto or timing can affect the next observation. Maximum per-row translation differences are35.4/25.3 micrometres; maximum rotation-vector differences are0.0001730/0.0000837 rad; maximum normalized closure differences are0.0006945/0.0019793. These small initial differences do not by themselves identify the mechanism of the later failure.

In125 Hz execution logs, requested TCP/joint/finger commands first differ at1.460/1.484 s and measured state first differs at1.468/1.492 s. Physics agrees before the differing commands. The15 Hz scene trace first shows packet pose differences at7.468/7.536 s, around contact, rather than at initialization. Subsequent heads are conditioned on different capture times, images, proprioception, contact states and previous chunks; comparing them by replan index is descriptive, not an equal-input test.

## Different contact outcomes, same terminal guard

Repeat1 lifts the packet316 mm and carries it311 mm. Its load latch arms at11.380 s; gel-force peaks are4.97/4.40 N. At17.044 s the gripper remains closed and physically loaded, with roughly5.47/5.59 N pad–packet forces in the nearest scene sample. The release controller remains `holding`, outside its volume. The stop is wrist extension, not a high-wrench event.

Repeat2 lifts109 mm but drops at13.200 s and never arms the native load latch. Its physical pad–packet force peak is4.30 N, while active-gel estimates peak at2.243/2.047 N; the strongest concurrent weaker-pad estimate is1.896 N. No tactile sample has both gel loads at or above the frozen2.5 N latch threshold. At the16.892 s wrist-extension stop the robot is empty, the wrist input equals its initial bias and the packet is back on the table. This distinguishes loss of retention from the later empty-arm stop. The observation is consistent with the known v1 contact-mapping sensitivity; it does not establish calibrated physical gel loading or justify lowering the latch threshold.

## Next controlled tests

1. **Six first-observation inference calls, separate diagnostic only:** use the exact original first NPZ twice, repeat1's first NPZ twice, and repeat2's first NPZ twice. Reset the complete policy/sampler state to seed4242 for each call, retain identical checkpoint/EMA/NFE1/K4 configuration, and log applied server overrides and noise state. Non-RGB inputs are already equal, so between-pair differences isolate the RGB input change; within-pair differences test exact-input repeatability. First verify that the reset actually clears persistent noise and any previous-chunk state. Do not call unequal-input differences native nondeterminism.
2. **Timing control if needed:** on an isolated diagnostic copy, retain fresh inference but impose the original per-call delivery schedule, including its slow first call. A fixed average latency is insufficient. Compare first observations/heads before proceeding; if rendered RGB still differs, this is a timing control with residual image differences, not an exact replay. Keep the primary result under actual measured timing separate.
3. **Mechanical control:** replay the original exact125 Hz target-q and target-finger commands, with the original startup and scene, without policy inference. Compare q/TCP/gripper/contact/object trajectories to the original. This can test mechanically reproducible pickup and placement; it is not evidence that a policy reproduced success. Logging both mapper versions on the same saved contacts is a separate sensor diagnostic, not a reason to overwrite the v1 reference.

The earliest test should be the six-call input comparison. There is no evidence here of changed initial mechanics, and raw policy heads differ before timing causes state divergence. A successful reference exists, but these repeats show that seed4242 plus a matching frozen configuration is not yet a robust full-placement result.

The [main CPU audit](audit_reproduction.py) and [contact audit](audit_reproduction_contact_detail.py) read only completed remote files. They ran at low CPU priority with `OMP_NUM_THREADS=1`; no simulator, model or hardware was launched by this audit.
