# Minimal gel v2 + FINISH: completed two-trial diagnostic

Both trials completed with valid strict scores. Neither placed or entered the controller FINISH state. These two same-seed live runs are a diagnostic, not a model ranking or a controlled causal estimate of FINISH.

| Seed | Frozen stages | First safety stop | Measured TCP XYZ at stop, m | Retention |
|---|---|---|---|---|
| 904301 | acquisition 8.068 s; lift 11.600 s; carry 14.000 s | wrist_extension 15.988 s | [−.46449, −.02486, .38414] | Held through stop; pre-stop packet forces 5.71/5.85 N |
| 904302 | acquisition 8.400 s; no sustained lift/carry | wrist_extension 15.868 s | [−.47194, −.11536, .40214] | Packet returned to mat; maximum rise 21.33 mm |

Seed 904301 has a loaded latch and no eligible post-latch opening. Its TCP is 110.1 mm above the release ceiling when measured wrist radius reaches .468094 m. Over the previous half-second it advances 62.5 mm in +Y while descending 19.9 mm. It remains outside the release volume and never executes the new terminal behavior.

Seed 904302 never arms the latch: maximum simultaneous weaker gel load is 2.337 N, below the unchanged 2.5 N bilateral criterion. Its last bilateral packet contact is sampled at 11.400 s; the packet rises less than the 30 mm lift threshold and is back on the mat before the wrist-extension stop. The frozen dropped=false outcome is conditional on the scorer’s prior-lift gate and should not be interpreted as successful retention.

The decisive FINISH check reads **execution_trace.jsonl**, not the old runner’s run.json. Both traces have zero nonempty `diagnostics.completed_reason` rows and zero `completion_hold` rows. Release phases at stop are `holding` and `unarmed`; both have `controller_finished:false`, no committed opening and no finished timestamp. The absence of a top-level completion field in run.json was not used as evidence.

The prior gel-only cases placed at 23.736/22.136 s. Here, failures occur before FINISH can act; their difference does not establish a detrimental terminal-hold effect. Native delivery timing and rendered observations remain live and unmatched, and no new source/controller modification was made for this audit.

[Detailed stage/contact audit](minimal_diagnostic_cases.json) includes both exact stop poses, reach/closure/latch diagnostics and frozen input hashes. [Video manifest](minimal_video_manifest.json) records movie/mapping hashes, force-array hashes, source metrics hashes, exact-score equality, full decoding and causal timestamp checks. The [CPU helper](render_minimal_reviews.py) independently checks saved execution completion and calls the frozen presentation tool.

| Seed | Decoded frames | Presentation horizon | MP4 bytes |
|---|---:|---:|
| 904301 | 270 | 17.984 s | 1,080,266 |
| 904302 | 268 | 17.864 s | 1,007,797 |

Both reviews are 15 Hz and use saved native runtime gel pixels with explicit uncalibrated labels. Total MP4 bytes: 2,088,063. Bulk videos remain remotely under:

```text
/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v3/video_reviews/minimal_profile_diagnostic/teacher__fixed_anchor__seed904301/policy_review.mp4
/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v3/video_reviews/minimal_profile_diagnostic/teacher__fixed_anchor__seed904302/policy_review.mp4
```

Each directory also contains the per-frame mapping JSON and first-frame PNG. CPU rendering PID 2305250 ran sequentially at nice 19 on CPUs 30–31 and exited normally. All score inputs were hash-verified; no source, raw trial, safety threshold, simulator, model or hardware operation was changed.
