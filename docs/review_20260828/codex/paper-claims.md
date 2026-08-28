1. `high`  
File: [phantom/model/hht/hht.py](/Users/sannikov/GitHub/phantom/phantom/model/hht/hht.py:42), [phantom/model/hht/hht.py](/Users/sannikov/GitHub/phantom/phantom/model/hht/hht.py:76), [phantom/inference/policy.py](/Users/sannikov/GitHub/phantom/phantom/inference/policy.py:90), [phantom/config/training.py](/Users/sannikov/GitHub/phantom/phantom/config/training.py:66)  
Claim: The repo’s “sensor-free student” claim is false as implemented because the student still consumes wrist F/T, which is a contact-bearing sensor.  
Evidence: `HHT.obs_frames()` always builds `OBS_PROPRIO` from `wrist` and `ur_state` for both teacher and student, deploy always feeds a normalized wrist window, and the training config still calls HID “the sensor-free student.”  
Failure scenario: A reviewer can reasonably say any student gain comes from the surviving force signal rather than from tactile imagination/distillation.  
Minimal fix: Rename the claim everywhere to “fingertip-sensor-free,” or add a true no-wrist student and report both variants.

2. `high`  
File: [docs/training_playbook.md](/Users/sannikov/GitHub/phantom/docs/training_playbook.md:111), [docs/launch_guide.md](/Users/sannikov/GitHub/phantom/docs/launch_guide.md:153), [phantom/train/train_teacher.py](/Users/sannikov/GitHub/phantom/phantom/train/train_teacher.py:145), [phantom/train/train_teacher.py](/Users/sannikov/GitHub/phantom/phantom/train/train_teacher.py:189), [phantom/scripts/run_deploy.py](/Users/sannikov/GitHub/phantom/phantom/scripts/run_deploy.py:34), [phantom/deploy/planner.py](/Users/sannikov/GitHub/phantom/phantom/deploy/planner.py:28)  
Claim: The key paper baselines/ablations (`vision_only`, `no_distill`, `drop_tactile`, CASA `alpha=0`) are not actually implemented as distinct train/deploy paths.  
Evidence: Docs promise a vision-only model without tactile+F/T, a no-distill student-layout control, a wrist-removed drop-tactile variant, and an `alpha=0` CASA toggle, but `train_teacher` hardcodes `student=False`, `run_deploy` maps every non-`teacher` system to the same student layout, `TACTILE_INPUT_MODES` only toggles fingertip streams, and `AccGate` exposes only the learned `W_alpha` path.  
Failure scenario: The central comparisons can only be missing, mislabeled, or hand-edited, which invalidates the paper’s main causal claims.  
Minimal fix: Add explicit end-to-end flags for `--student`, `--no-wrist-ft`, and `--acc-alpha {0,1,learned}`, then retrain only the baselines you can honestly instantiate.

3. `high`  
File: [phantom/data/windows.py](/Users/sannikov/GitHub/phantom/phantom/data/windows.py:395), [phantom/data/windows.py](/Users/sannikov/GitHub/phantom/phantom/data/windows.py:403), [phantom/model/rf.py](/Users/sannikov/GitHub/phantom/phantom/model/rf.py:308), [phantom/model/rf.py](/Users/sannikov/GitHub/phantom/phantom/model/rf.py:316), [docs/training_playbook.md](/Users/sannikov/GitHub/phantom/docs/training_playbook.md:138)  
Claim: The training code explicitly removes action supervision from failure demos, so the policy has no way to learn corrective behavior after a failed close.  
Evidence: `WindowSampler` sets `action_weight=0.0` for failure demos, and `training_step()` uses that weight only on `action_v_mse` while still supervising contact/event/gate heads.  
Failure scenario: Exactly the observed rig behavior is allowed: the model can correctly predict “no contact” yet still execute the successful-demo lift/transport action because only success trajectories teach the action head.  
Minimal fix: Collect failed-close recovery demos and keep nonzero action supervision on post-close windows, or relabel failed-close states with teacher/recovery actions instead of zeroing them out.

4. `high`  
File: [docs/rig_session_v5.md](/Users/sannikov/GitHub/phantom/docs/rig_session_v5.md:101), [docs/STATUS.md](/Users/sannikov/GitHub/phantom/docs/STATUS.md:74), [docs/STATUS.md](/Users/sannikov/GitHub/phantom/docs/STATUS.md:79), [configs/eval_campaign.example.yaml](/Users/sannikov/GitHub/phantom/configs/eval_campaign.example.yaml:10), [README.md](/Users/sannikov/GitHub/phantom/README.md:104)  
Claim: The repo currently supports only an offline teacher-improvement story; the student, baseline, and real-robot paper claims are experimentally unsupported in committed evidence.  
Evidence: `rig_session_v5.md` reports only teacher offline endpoint/miss metrics and explicitly says “Offline != rig,” `STATUS.md` still lists contact-play, teleop dataset, DAgger, and eval as open, and the example eval campaign has blank baseline checkpoints.  
Failure scenario: In 3 weeks this becomes an architecture paper with anecdotes, not a credible manipulation-results paper.  
Minimal fix: Minimum credible set now is `teacher`, `no_distill`, true fingertip-free `student`, and true `vision_only` (or rename it), with 3 offline seeds each plus 10–15 interleaved real episodes/task on `waffles`, `Carton`, `egg`, and `whiteboard`; add DAgger only if the offline student already clears teacher-adjacent behavior.

5. `high`  
File: [README.md](/Users/sannikov/GitHub/phantom/README.md:6), [phantom/model/sequence.py](/Users/sannikov/GitHub/phantom/phantom/model/sequence.py:190), [phantom/model/rf.py](/Users/sannikov/GitHub/phantom/phantom/model/rf.py:373), [phantom/config/model.py](/Users/sannikov/GitHub/phantom/phantom/config/model.py:32)  
Claim: As implemented, this is not yet a convincing WAM in the accepted sense because predicted future video is structurally auxiliary to control and can be dropped at inference.  
Evidence: `SequenceLayout.structural_attn_bias()` forbids `CONTACT` and `ACTION` queries from attending to `VIDEO_GEN`, `sample()` can rebuild the layout with `drop_video=True`, and the video loss is downweighted to `0.1`.  
Failure scenario: A hostile reviewer will say the “world” branch is decorative regularization, not a world model that actually informs action.  
Minimal fix: Run the promised `--drop-video` ablation on teacher and student; if control does not degrade materially, stop centering the paper on the video-generation head and present the model as action+contact prediction with an auxiliary visual head.

6. `high`  
File: [phantom/train/train_teacher.py](/Users/sannikov/GitHub/phantom/phantom/train/train_teacher.py:145), [phantom/train/train_teacher.py](/Users/sannikov/GitHub/phantom/phantom/train/train_teacher.py:182), [phantom/train/train_teacher.py](/Users/sannikov/GitHub/phantom/phantom/train/train_teacher.py:184), [docs/launch_guide.md](/Users/sannikov/GitHub/phantom/docs/launch_guide.md:153), [phantom/model/acc.py](/Users/sannikov/GitHub/phantom/phantom/model/acc.py:95)  
Claim: The anticipatory-contact claim is one accidental flag away from being invalid, and the advertised reactive/CASA control is not a real evaluated path yet.  
Evidence: `train_teacher` defaults to `gt_noised` and explicitly warns not to report lead-time from that run, while the docs describe `alpha=0/1` as a “small code toggle” rather than an exposed train/eval configuration.  
Failure scenario: Reporting ACC lead-time from the wrong checkpoint or without a sealed CASA baseline gives reviewers an easy fatal objection.  
Minimal fix: Verify every headline checkpoint’s saved model config before use, expose `alpha` clamping as a real flag, and run a paired ACC-vs-CASA teacher ablation on the same data/seed.

7. `medium`  
File: [phantom/eval/trial_runner.py](/Users/sannikov/GitHub/phantom/phantom/eval/trial_runner.py:67), [docs/rig_session_v5.md](/Users/sannikov/GitHub/phantom/docs/rig_session_v5.md:34)  
Claim: The automated eval protocol is experimentally confounded because it runs all trials grouped by system instead of the interleaved A/B procedure the rig doc itself prescribes.  
Evidence: `run_campaign()` loops `system -> task -> occlusion -> seed -> trial`, while `rig_session_v5.md` explicitly says comparable A/Bs should be interleaved per grid cell.  
Failure scenario: Later systems inherit object wear, camera drift, operator fatigue, and any latent calibration shift, so “recovery ratio” can move without the model changing.  
Minimal fix: Precompute a randomized paired schedule that interleaves systems within task and placement cell, and log placement/object instance IDs in the ledger.

8. `medium`  
File: [phantom/eval/aggregate.py](/Users/sannikov/GitHub/phantom/phantom/eval/aggregate.py:32), [phantom/eval/aggregate.py](/Users/sannikov/GitHub/phantom/phantom/eval/aggregate.py:68), [phantom/eval/trial_runner.py](/Users/sannikov/GitHub/phantom/phantom/eval/trial_runner.py:92)  
Claim: The current statistical protocol is too weak for ICRA because it gives per-cell CIs only, no uncertainty on the headline ratios, and uses manual binary success labels as the primary outcome.  
Evidence: `aggregate.py` bootstraps success within cells but computes `recovery_ratio` and `retention` as bare point estimates, and `trial_runner.py` asks the operator for success/damage after each trial.  
Failure scenario: Reviewers can reject the significance story and question independence/objectivity even if the raw win looks real.  
Minimal fix: Use paired block bootstrap or mixed-effects logistic regression over interleaved trials, and make miss-distance/z-at-close plus tactile-confirmed grasp success the primary scripted metrics.

9. `medium`  
File: [phantom/train/finetune_hids.py](/Users/sannikov/GitHub/phantom/phantom/train/finetune_hids.py:49), [phantom/train/finetune_hids.py](/Users/sannikov/GitHub/phantom/phantom/train/finetune_hids.py:71)  
Claim: The HID-S reward is scientifically mismatched because `tau_obj` is calibrated on absolute episode peak force, but training penalizes zero-based cumulative sums of per-window force deltas.  
Evidence: `calibrate_tau_per_task()` measures absolute peak normal force from recorded episodes, while `window_peak_force()` reconstructs force as `cpk_d_fz.cumsum(dim=1)` from window-local deltas with no initial absolute force offset.  
Failure scenario: Windows that start already in contact look artificially safe, so any HID-S row can under-penalize the very grasps it claims to make gentler.  
Minimal fix: Add absolute current/future normal-force targets to the window format, or seed the cumulative sum with the current absolute `f_z` before comparing against `tau_obj`.

10. `medium`  
File: [phantom/eval/metrics.py](/Users/sannikov/GitHub/phantom/phantom/eval/metrics.py:95), [phantom/eval/metrics.py](/Users/sannikov/GitHub/phantom/phantom/eval/metrics.py:118)  
Claim: The ACC evaluation is finger-asymmetric and can bias the contact-anticipation story.  
Evidence: `acc_lead_times()` appends onset lead-times for every tactile sensor independently, but `event_f1()` hardcodes only `hw.tactile.sensors[0]`.  
Failure scenario: A two-finger grasp can count twice in the lead-time plot while only one finger’s mistakes count in F1, making anticipatory contact look cleaner than it is.  
Minimal fix: Use one symmetric event definition for both metrics: either merge per-finger onsets into physical episode-level contact events or macro-average over both fingers.