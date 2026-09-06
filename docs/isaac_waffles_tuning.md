For the current results and hardware handoff, start with the [Isaac Sim overview](isaac_sim.md). This document retains the earlier reconstruction/protocol history; local artifact paths refer to files outside Git, with retrieval locations explained in the overview.

The waffle reconstruction now reproduces a recorded lift and a separate recorded pick, transport, and release using one shared set of contact and gripper parameters. These are driven replays of measured robot feedback with a free rigid packet. The result supports further policy testing, but does not establish a calibrated one-to-one digital twin or validated tactile transfer.

The complete tuning inventory is summary.json (local artifact `artifacts/isaac_waffles/tuning/summary.json`). It retains failed trials, initialization exclusions, configuration values, source hashes, sampled contact spans, packet motion, and non-packet contact measurements. The original reconstruction and launcher documentation remain in [isaac_waffles.md](isaac_waffles.md).

The principal deliverables are the final 15 Hz comparisons: pick and place (local artifact `artifacts/isaac_waffles/validation_v2/pick_place/comparison_tactile/side_by_side.mp4`), fit lift (local artifact `artifacts/isaac_waffles/validation_v2/lift/comparison_tactile/side_by_side.mp4`), and held-out lift 5066 (local artifact `artifacts/isaac_waffles/validation_v2/heldout5066/comparison_tactile/side_by_side.mp4`). Each preserves both 640×480 scene views and adds recorded left/right tactile images alongside explicitly labeled simulated force proxies. Each output directory also contains a separate tactile video, frame-alignment metadata, quantitative motion/image metrics, and the simulator trace. The portable scene is waffles_scene.usdz (local artifact `artifacts/isaac_waffles/validation_v2/pick_place/waffles_scene.usdz`).

The final assessment is **task sequences reproduced, with partial timing and appearance validation**. All three final runs achieve their simulated task outcomes. The fit lift passes the declared physical and image-proxy gates (wrapper-landmark RMSE 11.17 px, p95 18.90 px). The final pick/place run carries, releases, and settles the packet, but simulated contact begins at 6.868 s versus the real tactile threshold crossing at 7.48 s: 0.612 s early, outside the unchanged 0.5 s tolerance. Its wrapper-landmark RMSE improves to 9.79 px, while p95 remains 24.30 px against a 20 px tolerance. Closure-based release is 13.868 s versus approximately 13.8 s in the recording; packet contact unloads at 13.6 s. The held-out `5066` run retains the packet but fails contact/lift timing gates (9.868 versus 11.64 s contact; 12.668 versus 11.96 s lift). Higher sampling exposes the pick/place timing failure that the 5 Hz trial did not reveal; acceptance thresholds were not relaxed. See the final pick/place (local artifact `artifacts/isaac_waffles/validation_v2/pick_place/pick_place_evaluation.json`), lift (local artifact `artifacts/isaac_waffles/validation_v2/lift/pick_place_evaluation.json`), and held-out (local artifact `artifacts/isaac_waffles/validation_v2/heldout5066/pick_place_evaluation.json`) evaluations.

| Final 15 Hz run | Joint RMSE, rad | TCP position RMSE, mm | TCP geodesic rotation RMSE, degrees | Blue-bin silhouette IoU |
| --- | ---: | ---: | ---: | ---: |
| Pick/place | 0.01819 | 12.00 | 2.371 | 0.749 |
| Fit lift | 0.00625 | 5.23 | 0.677 | 0.857 |
| Held-out 5066 | 0.00624 | 4.56 | 0.690 | 0.860 |

These state errors compare actual simulated articulation/tool poses with native measured feedback, without smoothing through the 15 Hz reference grid. The larger pick/place tracking error remains relevant to policy transfer: its maximum TCP position error is 41.57 mm. Blue-bin IoU is an uncalibrated static silhouette proxy, not full-scene geometric accuracy. Independent PhysX and episode clocks agree within 0.94 microseconds. All final video frames decode successfully: raw simulator videos have 297/221/250 frames, while common-support comparison videos have 296/220/250 frames for pick/place/lift/held-out respectively. The final off-grid state is not padded into a fictitious reference frame. Media integrity (local artifact `artifacts/isaac_waffles/validation_v2/media_integrity.json`) records hashes, dimensions, and decoded frame counts.

The following results describe the completed 5 Hz tuning/sensitivity traces. Onsets have up to 0.2 s sampling uncertainty. The final 15 Hz runs use the same physics with the refined camera; their separate summary (local artifact `artifacts/isaac_waffles/validation_v2/summary.json`) and per-run metrics are authoritative for the final videos.

| Recording or trial | Split/purpose | Simulated peak packet rise | Observed simulated result |
| --- | --- | ---: | --- |
| Sept 4 teacher `1788535016_005`, shared 70 mm stroke | Fit lift | 326.67 mm | Bilateral grasp retained through clip end |
| Aug 22 demonstration `1787395928_000`, shared 70 mm stroke | Fit pick/place | 235.76 mm | Carried, released, and settled inside bin |
| Same demonstration, 2 ms physics step | Timestep sensitivity | 239.33 mm | Released and settled inside bin |
| Same demonstration, packet friction ×0.85 | Friction sensitivity | 234.81 mm | Released and settled inside bin |
| Sept 4 teacher `1788535066_006` | Frozen held-out | 328.35 mm | Retained through end; real footage also lifts/retains |
| Sept 1 teacher `1788262056_000` | Frozen held-out | 398.91 mm | Retained through end; real footage also lifts/retains |
| Sept 4 teacher `1788538100_000` | Frozen held-out | 1.91 mm | Failed pickup; real footage also retreats without the packet |

The Aug 22 shared-stroke trial has bilateral packet contact from the sampled 7.0–13.4 s interval. Its next sample at 13.6 s has zero packet force. A separate closure-drop detector identifies release at 14.0 s, compared with the visually identified real release near 13.8 s. These are distinct event definitions. The final oriented packet box lies inside the bin and rests on its floor with zero pad packet force. The event evaluator (local artifact `artifacts/isaac_waffles/tuning/teleop_ellipse75_stroke70/pick_place_evaluation.json`) passes the declared physical gates. It does not pass the preliminary image-proxy tolerances: projected wrapper-landmark RMSE is 16.32 px and p95 is 25.35 px before the final camera refinement. This proxy is conditional on an assumed object-top landmark; it is not a calibrated 3D pose error or direct rendered segmentation.

The held-out failure needs a qualification: a person moves the real packet around 4–5 s in episode `1788538100_000`. That perturbation is not replayed in the simulator. The failed-pick outcome agrees qualitatively, but the contact trajectory and dynamic disturbance are not validated. The held-out adjudication (local artifact `artifacts/isaac_waffles/evidence/pick_place_analysis/heldout_adjudication/adjudication.json`) records source video hashes and links dense frame montages. No held-out geometry was fitted to these outcomes.

The held-out `5066` lift outcome agrees, but its stricter timing gates do not all pass: persistent simulated contact starts near 10.0 s versus a native tactile threshold crossing near 11.64 s, and 20 mm packet lift begins near 12.8 s versus a measured TCP lift proxy near 11.96 s. Contact thresholds compare different, uncalibrated units, and TCP lift is not a measured real packet pose. These timing discrepancies remain visible in the held-out evaluator (local artifact `artifacts/isaac_waffles/tuning/heldout5066_frozen/pick_place_evaluation.json`). The Sept 1 timing and carry gates pass under their declared conventions.

The successful candidate uses a 70 mm gripper stroke parameter, the visually estimated 165.56 mm pad center offset and −0.377 rad yaw, rounded elliptical pad collision hulls, a 35 g rigid packet, and a 4 ms physics step. The table top is at 53 mm in the robot base frame, with the packet initially centered at 70 mm. These dimensions and the effective aperture remain estimates, not ruler measurements. The full settings are in [waffles.json](../configs/sim/waffles.json) and [waffles_pick_place.json](../configs/sim/waffles_pick_place.json); the earlier reconstruction is preserved in [waffles_reconstruction_v1.json](../configs/sim/waffles_reconstruction_v1.json).

Packet friction defaults to static 0.65 and dynamic 0.50; pad friction is 1.0. The sensitivity trial changes the packet coefficients to 0.5525/0.425 while retaining pad friction 1.0. With the configured average combine mode, pad–packet static friction changes from 0.825 to 0.77625 and dynamic friction from 0.75 to 0.7125. It therefore tests a modest effective contact-friction change, not a 15% change to the combined pad coefficient. The retained placement at a 2 ms step is a useful numerical sensitivity check, not timestep convergence certification.

The initialization fix assigns the measured starting arm pose, resets packet pose/velocity once, and then settles the scene before t=0. Driven replay does not attach, teleport, or overwrite the packet state afterward. `raised55_toe185`, `raised75_toe185`, and `teleop75_toe185_gap82` had 186–282 mm initial packet displacement from the fresh-asset startup bug; they are excluded from pickup evaluation. `raised75` and `raised90` remain geometry sensitivity trials with explicit 29–34 mm initial-position warnings. The earlier 76.5 mm stroke ellipse trial lifted only 52.31 mm before losing its grasp; it is preserved as a retention failure.

Contact-force realism remains a material limitation. The shared-stroke Sept 4 trial includes a left-pad transient of 118.74 N, while its late grasp is approximately 5 N per pad. The failed held-out `8100` trial has a right-only transient of 132.14 N. These peaks are sampled at render rate, so faster transients may be missed. Subtracting packet-filtered force from net pad force identifies non-packet contacts; overlap between pad bounds and the support plane makes table contact plausible, but no separately filtered floor-force measurement is available. These force traces need calibration before force-sensitive policy conclusions.

Real tactile images are the native grayscale `infer_img` streams, preserved losslessly with their own MasterClock timestamps. All five episodes have separate exports in evidence_tactile_v1 (local artifact `artifacts/isaac_waffles/evidence_tactile_v1`); existing replay exports and source recordings are unchanged. The video uses the previous available tactile sample, never a future observation, and marks missing or stale samples. Native tactile image rate is approximately 4–6 Hz, so a 15 Hz video does not imply 15 Hz tactile sensing. Wrench and SDK depth streams are also retained at native timestamps. Their units are not assumed to be calibrated Newtons or metres.

Simulated tactile panels use packet-filtered PhysX normal force and a force-weighted contact centroid in normalized pad coordinates. The Gaussian rendering is an uncalibrated force proxy. Contacts may include pad backing or linkage geometry, and the centroid is not a calibrated gel-image projection. Optical gel appearance, distributed indentation, shear, slip, wrapper flex, and wafer fracture are not reproduced. No pixel similarity score between the recorded gel images and these proxies is claimed.

Reproduce the demonstration on compute3 from the durable scene source:

```bash
cd /home/physicalai/phantom-icra-2027/sim/waffles
source/tools/sim/launch_waffles.sh \
  --mode dynamics \
  --episode evidence/fit/ep_waffles_1787395928_000 \
  --config source/configs/sim/waffles_pick_place.json \
  --output runs/reproduced_pick_place
```

Headless operation is the default. For an interactive view, run `source/tools/sim/view_waffles.sh` from the same compute3 directory; it defaults to the complete Aug 22 demonstration and leaves the GUI open for inspection after the replay. The launcher uses Isaac Sim 6.0 at `/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0` and preloads its matching bundled NCCL; bypassing it with the supplied `python.sh` can load an incompatible system NCCL. No hardware launcher is involved. These replay results do not constitute a new closed-loop evaluation of the resident policy; the existing deployment adapter and its earlier policy smoke test are documented separately.

The latest policy interface smoke test (local artifact `artifacts/isaac_waffles/validation_v2/policy_interface_smoke/run.json`) runs for 1.996 simulated seconds at 125 Hz, activates one existing v5_6 plan with 1.378 s inference latency, and ends without stop/errors; it does not assess pick success.

Generate the synchronized comparison using ordinary CPU Python from the PHANTOM repository:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python ../sim/waffles/source/tools/sim/compare_replay.py \
  --reference ../sim/waffles/evidence/fit/ep_waffles_1787395928_000 \
  --sim-trace ../sim/waffles/runs/reproduced_pick_place/sim_trace.npz \
  --sim-video ../sim/waffles/runs/reproduced_pick_place/sim.mp4 \
  --tactile-reference ../sim/waffles/evidence_tactile_v1/fit/ep_waffles_1787395928_000 \
  --out ../sim/waffles/runs/reproduced_pick_place/comparison_tactile --fps 15
```

`prepare_waffles.py --tactile-only --prepared-root ../sim/waffles/evidence --out NEW_OUTPUT_ROOT --data-root ../data` exports the two fit and three held-out tactile sidecars without replacing existing evidence. Refresh the local tuning inventory with `python artifacts/isaac_waffles/tuning/summarize.py` after syncing small trace/config files. The complete verification suite passes 85 tests: 68 across six existing test modules and 17 geometry tests. This includes the 19 focused replay/tactile tests for irregular native timestamps, SO(3) interpolation, independent physics-clock errors, causal tactile sampling, contact-UV rendering, and synchronized video frame dimensions. Repository Ruff checks, import sorting, and formatting pass on the owned tools and tests. Documented CLI options were checked against each tool's `--help` output.
