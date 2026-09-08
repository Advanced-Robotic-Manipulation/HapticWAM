# V9 tactile side-by-side video review

All four V9 trials have verified scene/tactile side-by-side videos. Both
0.200-second control trials physically release and settle the packet on the box
floor; one 0.100-second treatment trial does the same. The other treatment trial
never achieves sustained lift and ends with the packet on the mat.

These are the four declared development trials, using the same teacher and the
bounded 2.5-second constraint hold. The original one-trial ABBA blocks failed the
frozen driver's minimum-two-seeds and complete-grid checks; the preauthorized
fallback executed `control_dwell200` and then `treatment_dwell100`, each with
seeds 904301 and 904302 (AABB). No extra trial was rendered or launched.

| Setting | Seed | Strict placement | Frames at 15 Hz | Visual review |
| --- | --- | ---: | ---: | --- |
| control_dwell200 | 904301 | 24.668 s | 900 | At 22 s the packet is still held. By 30 s it rests on the box floor with open, unloaded pads, and remains there through the full horizon. [Before release](control_dwell200_904301_frame_22.00s.jpg), [after release](control_dwell200_904301_frame_30.00s.jpg), [final](control_dwell200_904301_frame_final.jpg) |
| control_dwell200 | 904302 | 24.336 s | 900 | The same held-to-released transition, with zero tactile load after release and the packet settled in the box. [Before release](control_dwell200_904302_frame_22.00s.jpg), [after release](control_dwell200_904302_frame_30.00s.jpg), [final](control_dwell200_904302_frame_final.jpg) |
| treatment_dwell100 | 904301 | 25.936 s | 900 | Packet settles on the box floor with open, unloaded pads. [After release](treatment_dwell100_904301_frame_30.00s.jpg), [final](treatment_dwell100_904301_frame_final.jpg) |
| treatment_dwell100 | 904302 | Failed | 527 | Packet remains on the mat. The empty gripper moves through the task and returns before the safety stop; no sustained lift. [At 12 s](treatment_dwell100_904302_frame_12.00s.jpg), [final](treatment_dwell100_904302_frame_final.jpg) |

The three successes satisfy the frozen physical scorer, including release,
settling, and box support. They are not inferred from the renderer's separate
“RELEASE COMPLETE / HOLD” controller label. All three continue through
59.996 simulated seconds without a later stop or drop. The failure stops at
33.108 s and has the existing physics observation tail through 35.108 s.
Its historical “Object: acquired” stage label does not assert a current grasp;
likewise, `dropped=false` does not imply retention because this case never met the
sustained-lift prerequisite of the drop metric.

The two successful control videos demonstrate full pick-to-place with the
original 0.200-second release dwell and the new bounded hold. This small video
set does not establish a reliable success rate or prove that the shorter dwell
caused its failed trial. See the [experiment report](../README.md) for the paired
analysis and decision.

## Files and verification

Remote videos:
`/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_release_dwell_v9/video_reviews/`.
Verified local mirror:
`/home/agent/skoltech/research/artifacts/isaac_waffles/teacher_release_dwell_v9/video_reviews/`.
Each contains
`{setting}/fta1500_nfe1_k4__successful_anchor__seed{seed}/policy_review.mp4`
and its complete alignment JSON. The four final MP4 files total 8,904,790 bytes.

[`render_review.py`](render_review.py) did not start rendering until the four-trial
completion marker and independent completion audit passed, all owned processes
were stopped, the dedicated port was free, and the review agent received
completion confirmation. All 5,106 source/input pins were independently verified
before rendering. The helper also checked both completed blocks, all four valid
scores, declared block order, and effective release dwell values.

The current presentation tool used the unchanged frozen V8 runtime source via
`PHANTOM_REVIEW_SOURCE`. It displays the saved left/right native grayscale
policy-input pixels selected causally. The panels explicitly retain the
uncalibrated static-measured-baseline-plus-physics-deformation label; they are
neither real tactile footage nor validated sensor predictions.

- All four video metric dictionaries exactly equal their frozen authoritative
  scores. Scene/tactile sample ages are nonnegative.
- Every MP4 passed a complete `ffmpeg -xerror` decode and a separate OpenCV
  decode with exact frame counts. All final local MP4 and alignment-JSON hashes
  match the decoded remote files.
- The videos retain original simulation time, including the full observation
  horizon or stopped-trial physics tail. Last-frame times precede final trace
  samples by less than one video frame interval. Stopped trials show the age of
  the last sampled tactile observation.
- Ten compact JPEG screenshots preserve the verified frame content. Their
  source PNG hashes, JPEG hashes, and simulation times are recorded.
- Initial presentation drafts omitted the seed overlay after the AABB layout
  change. They are retained separately in `video_reviews_without_seed_labels/`;
  the final manifest and links refer only to the corrected, fully decoded
  videos. This was a presentation regeneration, not a simulation rerun.

[`manifest.json`](manifest.json) contains the final complete video, metadata,
source-input, score, and frame hashes, plus runtime provenance and object/support
outcomes. [`local_verification.json`](local_verification.json) contains local
paths and screenshot verification. The rendering helper passes Ruff.
