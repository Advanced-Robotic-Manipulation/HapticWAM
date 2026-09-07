# First two reference-teacher trials: descriptive diagnosis

Both completed ftA1500 trials fail before acquisition. They belong to the **subsequently halted original screen**, preserved as diagnostic evidence; they are not a new teacher ranking or the replacement study's scored results. The [numeric audit](audit.json), [contact sheet](contact_sheet.jpg) and [read-only helper](audit_first_two.py) retain exact case/input hashes, physical metrics, body-labeled contacts and sampled proposed/accepted/measured paths.

| Seed | Observed result | First additional close | Closest pad-midpoint/packet distance | Main contact evidence |
|---|---|---|---:|---|
| 903101 | 60 s horizon, no safety stop, no acquisition/lift | 10.916 s, distance 57.6 mm | 31.7 mm | Right backing on mat; later outer backing brushes packet |
| 903102 | Joint-speed stop at 10.020 s, no acquisition/lift | Never exceeds initial closure + 2/255 | 182.8 mm | Bin-front contact; large accepted/measured path divergence |

Both start with measured closure `.427403`, already above the frozen absolute `.25` reach-before-close threshold. The useful additional diagnostic checks the first executed command greater than initial closure plus the two-count gripper deadband. Its distance is measured at the preceding scene frame: the midpoint of the two actual pad origins is transformed into the actual packet's oriented frame, and its Euclidean distance outside the packet box is computed. This is a midpoint proxy, not minimum finger-surface distance. It also does not identify every closing motion after an intervening opening.

## Seed 903101: misses the grasp region and contacts the mat

The arm first moves toward positive world Y while descending, stopping its early approach roughly 80–90 mm beyond the packet center in Y. Requested and accepted Cartesian positions differ by at most 0.100 mm, and accepted-to-measured position error stays below 3.12 mm. There are no IK rejects, stale-plan holds or inference errors. The accepted approach was therefore executed closely; its initial spatial miss is not explained by a large actuator tracking error.

By the first additional close at 10.916 s, the pad midpoint remains 57.6 mm from the packet box. The arm then makes small movements near the mat until the horizon. Explicit right-pad/mat contacts reach 26.894 N at tactile sample 33.380 s (scene-rate net peak 27.399 N). Their local X coordinates are approximately −5 to −11 mm; the right inner gel face is +6 mm. These are backing contacts and correctly remain excluded from gel pressure.

At 54.004 s the right backing brushes the packet with 0.733 N in the tactile record (scene peak 0.760 N). Local X is approximately −10.95 mm, again the outer side. The packet moves 23.6 mm on the mat, with no meaningful rise or bilateral contact. Gel force and grip latch remain zero. The placement-release controller remains unarmed; it is not responsible for this failure to acquire.

## Seed 903102: bin-directed approach followed by physical blockage

The early proposed and accepted path moves toward the bin front while remaining far from the packet. Requested and accepted positions stay within 0.093 mm. At about 5.3 s, however, accepted-to-measured error exceeds 20 mm and eventually reaches 211.5 mm. For example, near 6 s the accepted TCP is approximately `[-.389, -.069, .108]` m, while the measured TCP is `[-.429, -.067, .183]` m. The increasing error cannot be described as successful execution of the accepted Cartesian path.

Explicit contact filters identify `/World/Bin/Front`; a left-pad sample reaches 2.525 N at 9.876 s. Other robot-body/bin loads were not observed in this run, and wrist feedback sums only pad net forces. The images and tracking divergence support physical blockage near the bin, but do not identify every blocking link or certify that the collision geometry and force response are realistic. The optional whole-robot/environment observer was subsequently implemented for a **separate diagnostic command replay**, without modifying this run.

The stop at 10.020 s is caused by `joint_speed`: wrist joint 1 reaches −1.338 rad/s, above the existing 1.2 rad/s guard. `workspace_clamp` is also reported. It is not a wrist-force stop, a grasp-complete flag or an infrastructure timeout. The post-stop two-second observation contains no task completion. No safety relaxation is justified by this result.

Bin contacts use V2's explicit point fallback because the frozen convex-patch whitelist contains only the packet. The loaded inner-face manifold vertices lie outside the active ellipse, producing zero gel pressure. That is the declared conservative environment-contact approximation; it does not prove that a real gel would feel zero pressure across the intervening contact area. Body/patch attribution and pressure calibration remain separate issues.

## Wrist baseline and interpretation

Both trials use the same measured initial wrist bias:

`[8.277, 13.146, −19.002]` N and `[-4.315, 4.557, 1.137]` N·m.

Its force norm is **24.543 N**, not 47 N. Maximum raw wrist-force norms are 24.703 N and 29.147 N. The policy receives this raw fixed bias plus simulated whole-pad net force and estimated pad moments. The native safety supervisor initializes an independent baseline from the first wrist sample, subtracts a calm-only moving average, and debounces deviations. Thus a constant measured bias does not by itself trip the wrench guard. The proxy does not include all robot-body/environment loads or calibrated inertia/gravity effects.

These results support two distinct conditional failure descriptions: a closely executed approach that misses the grasp region, and a bin-directed proposal followed by large physical tracking divergence and a valid joint-speed safety stop. They do not isolate whether the initial miss comes from visual registration, policy conditioning, sample variability or real-model differences; they do not establish that either trajectory would behave identically in the laboratory. The source parity issue that halted the screen is tracked separately. No scene, controller, threshold, checkpoint or sampling setting was tuned from these two outcomes.

A subsequent [observation-only command replay](observer_replay_audit.md) identifies the missing body load: gripper housing on the bin front peaks at 175.196 N at 5.200 s, before the tracking divergence. All 21 original trace arrays remain bitwise identical with the observer enabled. This strengthens the physical-blockage diagnosis while leaving the recorded policy outcome and its original sensor inputs unchanged.

The subsequent [gripper-contact wrist preflight](gripper_wrist_preflight_audit.md) includes housing and pad normal-contact wrenches, with contact-point moments. It preserves all original physics arrays, has exact causal 125 Hz caching, and exposes the housing load before the large tracking error. It is an explicitly idealized replacement observation model, not calibrated UR3 sensor transfer or a revised outcome for these cases.
