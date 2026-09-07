# Restoring the successful v1 reference case

The exact reference is `teacher__placement_xm10_ym10mm__seed4242` from `teacher_pick_place_v1`. Its frozen score records acquisition at 8.000 s, sustained lift at 12.668 s, carry at 15.336 s, release in the bin at 24.400 s and support-verified full placement at 25.200 s. A later wrist-extension stop at 31.380 s remains part of this outcome. It is a successful placement reference, not a clean episode finish or evidence of a reliable model winner.

At **2026-09-07 07:50:55 UTC**, independent read-only checks found no hash mismatch in 90 frozen source/assets, seven live deployment/core files, 17 inference files, two prepared replay files, five runtime inputs or seven original score inputs. [Audit JSON](initial_audit.json) preserves counts, original states, first proposals and observation hashes. The original recordings and frozen outputs were not changed.

## Exact setup to restore

Remote base: `/home/physicalai/phantom-icra-2027/sim/waffles`.

| Component | Frozen reference |
| --- | --- |
| Source | `source_teacher_pick_place_v1` |
| Original rollout | `runs/teacher_pick_place_v1/campaign/rollouts/teacher__placement_xm10_ym10mm__seed4242` |
| Campaign SHA256 | `325f6144bf605c14800bed56e7eed322632a00f3dba491d26b819abd79049f5c` |
| Effective scene SHA256 | `2aae7d4d1e2d2f95131aa1ad50a7907c54c86726a6c39e2756167cc709f69cb9` |
| Checkpoint | `phantom/runs/teacher_v5_ftA/teacher_001500.pt`, SHA256 `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e` under project root |
| Runtime sampling | Teacher, EMA, NFE1/K4, guidance1, persistent noise, parity fixes, seed4242, max-play10; task `waffles`, video enabled |
| Initial state | `runs/teacher_pick_place_v1/sept4_policy_initial_state.json` |
| Tactile baseline | `runs/teacher_pick_place_v1/sept4_no_contact_sensor_baseline.npz` |
| Hardware limits | `runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml` |
| Robot asset | `runs/validation_v2/pick_place/robot_asset/waffles/waffles.usda` |

Use the original case's `effective_config.json` directly. Its packet center is `[-0.3937067184864266,-0.2930692769792478,0.07000000000000008]` m and yaw is `0.31656601407527596` rad. The condition's minus-10-mm XY displacement is already included; applying it again changes the reference. Packet mass is 35 g with dimensions 154.495 × 27.214 × 28 mm, friction 0.65/0.5. This scene uses the original packet texture default, not the later August green texture.

The robot is the existing nominal UR3 CB3 asset, with UR-base xyz/rotation-vector TCP and a 180 mm tool offset. Initial recorded q is `[0.234745130,-1.519814793,1.075420380,0.723345399,1.174745798,-3.004658286]` rad; gripper closure is **0.078431375**, rather than the later August start's approximately 0.427. Initial wrist bias is `[-6.679699239,-3.775478906,3.547405498,1.352852400,-1.737826255,0.458645563]` N/Nm. These are requested initial values; original post-settle maximum q error is 0.003725354 rad and maximum |qd| is 0.029448494 rad/s. Compare the fresh settled state, not just the initial JSON.

Physics is dt=4 ms, position/velocity solver iterations16/4, joint stiffness/damping3500/180, finger stiffness/damping1800/25 and drive force15 N. The rounded pads use 70 mm effective stroke, center z=0.16556 m, yaw−0.377 rad, and compliance12000 N/m with damping30. Forearm collision uses convex decomposition. The initial free hold lasts2 s with frozen q-error0.02 rad and speed0.05 rad/s gates. Camera/rendering is 15 Hz with the archived balanced camera; control is125 Hz and tactile8 Hz. These parameters are estimated simulator settings, not calibrated robot/gel dynamics.

Restore the complete **v1 sensor/controller profile** when testing this reference:

- Tactile is `measured_baseline_proxy` using the September4 baseline and **`manifold_patch` v1**, including its documented sparse-contact limitation. It is not `manifold_patch_v2`.
- Wrist is **`contact_proxy`**: measured initial bias plus each pad rigid body's net contact force, with moments computed at pad origins about the TCP. Housing contacts are absent from this model. The later gripper-wide wrist observer changes the observation and safety feedback.
- Terminal veto is historical **`fd4a032`**, not the newer live veto/current-delivery variant. Its archived config has z_ref0.0415, z_floor0.0315, z_margin0.0615, open_aperture0.232 and retry cap3.
- Placement release is the original opt-in controller, without `finish_after_release`. It permits policy opening inside the declared TCP release volume after load latching; it does not halt the policy after placement. The later stop must remain visible.
- Safety remains the original rolling baseline, force60 N / torque15 Nm with0.3 s debounce, wrist radius0.468 m, joint speed1.2 rad/s and the archived tactile/workspace limits. Restoring an old feedback model is a reproduction condition, not validation of its safety coverage.

## Locate the first divergence before interpreting the outcome

Compare in this order, preserving both simulation timestamps and raw array shapes/dtypes:

1. **Initialization:** effective scene, actual settled q/qd and finger gap, packet pose/velocity, robot asset and camera intrinsics. Check t=0 and the pre-inference segment of `sim_trace.npz` and `robot_settling.npz`. These comparisons isolate mechanics/start differences before feedback from inference.
2. **First observation:** original `observations/0000.npz` is captured at t=0.260 s. Arm/gripper timestamps are0.260, camera0.200, tactile0.252. Compare `rgb`, `ur_state`, `wrist_window`, `gel`, `fields`, `contact_state`, `reactive` and `prev_chunk` separately. Exact array hashes for the first three observations are in the JSON; if unequal, report RMS/max differences rather than treating all mismatch as policy randomness. This is the raw input passed to native preprocessing, not an already-normalized tensor.
3. **First proposal and sampler state:** compare all16×7 raw actions and selected K head diagnostics before delivery veto, plus checkpoint EMA selection, seed reset, persistent-noise state and per-strip setting. The reference first action is `[-0.0003408713,0.0009347936,-0.0056119980,-0.0096893245,-0.0065288674,-0.0123330494,0.1670730710]`. Equal checkpoint and seed are insufficient if input/history or sampler call sequence differs.
4. **Delivery and execution:** original first raw inference latency is1.220687008 s; plan0 activates at1.484 s. Next snapshots are1.492 and2.300 s, with plan activations2.292 and3.076 s. Compare captured snapshot, ready/delivery/activation time, veto result, governed play time, requested/accepted TCP, target q/fingers and measured response. Use exact common timestamps before divergence; thereafter interpolation can describe trajectory error but cannot establish identical policy inputs.
5. **Contact and task stage:** compare physical pad–packet forces, active-gel forces/coverage, net wrist input, latch, opening permission, and independent packet-to-bin/robot support. Keep physical acquisition/lift/release/full placement separate from later stops. A successful image or the release controller's state alone is insufficient.

The reference has38 inference calls with latencies0.769626–1.220687 s, median0.773281 s. The fresh primary repeat should keep actual measured inference latency and report it explicitly. A changed first-call latency shifts subsequent scene/sensor samples and persistent-noise history, so even an unchanged seed does **not** promise a bitwise repeat of a closed loop.

If the fresh result diverges, use separate, labelled diagnostic controls before modifying the reference: feed the saved raw observation sequence to the model to isolate preprocessing/sampling; impose the original per-call delivery schedule to isolate timing while retaining fresh inference; or replay exact recorded125 Hz target-q/finger commands to test mechanics and sensing. A single fixed mean latency does not reproduce the original schedule. The frozen launcher already exposes scalar `--policy-latency`; it does not expose a per-call recorded schedule or command replay. Those two controls require an isolated diagnostic copy and must not be reported as fresh primary policy outcomes.

At the process check immediately before the integrity audit, compute3's RTX5090 reported about11.5 GiB allocated and2% utilization; the long-running native server on port7778 remained present. No simulator was running in that snapshot. This is a timestamped availability observation, not a reservation. The parent owns the new port7799 process and all GPU launches; this audit starts/stops no processes.
