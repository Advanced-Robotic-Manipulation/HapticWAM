# V8 tactile side-by-side video review

All six completed V8 trials have a scene video with the saved left/right tactile
input pixels beside it. The panels retain the **uncalibrated simulated input**
label: a static measured baseline plus physics-driven deformation. They are not
real tactile footage or validated sensor predictions.

The videos are available on compute3 under
`/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_carry_hotfix_v8/video_reviews/`
and mirrored locally under
`/home/agent/skoltech/research/artifacts/isaac_waffles/teacher_carry_hotfix_v8/video_reviews/`.
Each contains `{setting}/fta1500_nfe1_k4__successful_anchor__seed{seed}/policy_review.mp4`
and its complete frame/tactile alignment JSON. The six MP4 files total 9,241,758 bytes.

| Setting | Seed | Video frames at 15 Hz | Visual review and frozen object-state result |
| --- | --- | ---: | --- |
| plain_k4 | 904301 | 265 | Stops with the packet still gripped above the box; carried, no release. [Final frame](plain_k4_904301_frame_final.jpg) |
| plain_k4 | 904302 | 266 | Empty gripper above the box; packet remains on the mat. No sustained lift. [Final frame](plain_k4_904302_frame_final.jpg) |
| legacy_limiter | 904301 | 257 | Stops with the packet gripped during carry, short of placement. [Final frame](legacy_limiter_904301_frame_final.jpg) |
| legacy_limiter | 904302 | 261 | Empty gripper above the box; packet remains on the mat. No sustained lift. [Final frame](legacy_limiter_904302_frame_final.jpg) |
| bounded_hold | 904301 | 900 | Carries and lowers into the box volume, then keeps gripping through 59.93 s. No release or box support. [At 22 s](bounded_hold_904301_frame_22.00s.jpg), [final frame](bounded_hold_904301_frame_final.jpg) |
| bounded_hold | 904302 | 668 | Leaves packet on the mat, moves empty gripper into the box, then returns toward the packet before a safety stop. No sustained lift. [At 22 s](bounded_hold_904302_frame_22.00s.jpg), [final frame](bounded_hold_904302_frame_final.jpg) |

The bounded-hold 904301 video shows a useful carry recovery, **not a completed
placement**. The packet is visibly between the pads at the final frame, with
saved tactile input normal forces about 5.51 N and 5.18 N. The authoritative
score confirms `final_inside_bin=true`, `final_contacts_unloaded=false`, zero
recorded bin contact force, and `released_in_bin=false`/`full_task=false` at
59.996 s. The 22 s image also shows the packet still held; the appearance of
being inside the box cannot substitute for release and support verification.

For seed 904302, the renderer's “Object: acquired” label reports the highest
historical stage reached. It does not mean the packet remains acquired in that
frame. Likewise, the frozen `dropped=false` score does not mean grasp retention
succeeded: this trial never met the sustained-lift gate used by that drop metric.

## Verification and reproduction

[`render_review.py`](render_review.py) renders only the existing six completed
trials. The current `tools/sim/make_policy_video.py` presentation tool was copied
into the remote `review_tools/` directory and used the frozen V8 runtime source
through `PHANTOM_REVIEW_SOURCE`. No source recording, rollout, scoring threshold,
or controller was changed. No simulator trial was started.

- All six rendered metric dictionaries exactly match their frozen authoritative
  trial scores.
- Every frame selects scene and tactile samples causally; sample ages were
  checked as nonnegative.
- Every MP4 passed a complete `ffmpeg -xerror` decode and a separate OpenCV
  decode with exact frame-count agreement. Local MP4 and metadata hashes match
  these decoded remote files.
- The final video frame precedes the last trace sample by less than one 15 Hz
  frame interval. Stopped runs preserve the approximately two-second physics
  tail and explicitly display the increasing age of their last tactile input.
- Ten compact JPEG screenshots were extracted from the verified PNG frames;
  only image encoding changed. Their source PNG and JPEG hashes are recorded.

[`manifest.json`](manifest.json) records complete video, metadata, source-input,
score and screenshot hashes, rendering provenance, outcomes, and tactile mapping.
[`local_verification.json`](local_verification.json) records all six verified
local video paths and the compact screenshots. The rendering helper passes
Ruff. This is visual and artifact validation, not a new policy experiment.
