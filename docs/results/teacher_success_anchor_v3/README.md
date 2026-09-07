# Recover and compare the successful teacher anchor

The [frozen protocol](protocol.json) keeps the exact v1 scene and measured start that produced a verified placement. Physical placement is the primary outcome; a later safety stop is reported separately. The [launch dependencies](anchor_launch_dependencies.json) retain the original command and input paths. The [dependency audit](dependency_audit.json), reproduced by [this CPU helper](audit_anchor_dependencies.py), passes all **120 original source/core/input checks**, all three checkpoint hashes, and every launch-file existence check. It does not run a model or hardware.

The anchor is `teacher__placement_xm10_ym10mm__seed4242`: ftA1500 EMA, NFE1/K4, task `waffles`; beige packet center `[-0.3937067185, -0.2930692770, 0.07]` m, yaw `0.3165660141` rad; the measured September 4 teacher5016 initial arm/wrist state and gripper closure `0.0784313753`. Camera, box, packet dimensions/materials and robot geometry remain fixed. Its historical physical placement at 25.2 s remains valid under the current physical evaluator; the later 31.38 s wrist-extension stop did not remove the packet from the bin.

| Stage | Frozen trials | Purpose |
|---|---:|---|
| Historical reproduction | 4 | Two repeats each of seeds4242/4243, retaining the historical positive and negative controls |
| Component diagnostics | 12 | Baseline plus five isolated component changes, each at matched fresh seeds904301/904302 |
| Teacher/recipe screen | 24 | Four candidates, each using fresh seeds904401–904406 |
| Reserved confirmation | 24 | The two screen leaders, each using fresh seeds904501–904512 |
| Core total | **64** | Arm-start robustness is a separate later study |

The five diagnostic changes are FINISH only, gel mapper v2 only, distal-wrist proxy only, live veto algorithm only while retaining request-time feedback, and current-delivery feedback only while retaining the historical veto algorithm. Each starts from the exact legacy baseline. Their success rates do not select the main comparison profile. **All checkpoint-screen and confirmation trials retain the frozen v1 controller/sensor profile.** FINISH stays a separate diagnostic; a controller-completion flag never substitutes for physical placement.

The four candidates are ftA1500 NFE1/K4, ftA3000 NFE1/K4, v5_6 NFE1/K4, and ftA1500 NFE5/K4. All use EMA, guidance1, parity, persistent noise and the same task text. NFE5/K4 changes NFE alone relative to the reference recipe; its measured native latency is part of the resulting closed-loop behavior. It is supported by the native API and cannot be silently excluded because it is slower. Exact weights and input hashes are in the protocol.

Screen ranking begins with strict support-verified physical placements, then final supported retention, clean finish, lift, acquisition, fewer pre-placement safety stops and frozen candidate order. All 24 matched trials must be valid before advancing two candidates. Normal FINISH is secondary and may remain absent under the historical main profile. The ordered physical stages and all-body support telemetry remain the task truth.

Confirmation can declare a **physical-placement winner at this fixed simulator anchor** only if the leader achieves at least 8/12 placements, its paired 95% interval is strictly positive against the runner-up, the exact two-sided discordant-pair test is at most0.05, and drops and pre-placement wrench/tactile force/depth stops do not increase. The bootstrap resamples the12 matched seed pairs, not individual candidates. The exact test avoids overstating evidence from a few one-direction discordant outcomes. No thresholds, seed counts or exclusion rules may change to obtain significance. Later safety stops, final object support and clean finish are reported separately.

The runtime source for reproduction and the main comparison is `/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_pick_place_v1`. Current orchestration may manage the candidate recipe and fresh output locations, but must launch that original source/core. The corrected v2 source differs in ten executed files; the audit lists them. Restoring only the scene JSON while using v2 runtime would not reproduce this anchor. Source/input equality still cannot promise bitwise closed-loop repeats because physics/rendering and native inference timing vary.

This scene was selected because it previously succeeded. New reserved seeds evaluate prospective performance at that particular anchor; they do not establish unseen arm-start, session or hardware success. Both successful and failed runs remain in the record. Calibration of geometry and sensor transfer is still required for physical deployment.

The [executable comparison and gate](execution.md) preserve the accepted protocol while separating command-replay mechanics evidence from policy outcomes.
