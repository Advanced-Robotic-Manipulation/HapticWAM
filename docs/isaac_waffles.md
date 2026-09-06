For the current results and hardware handoff, start with the [Isaac Sim overview](isaac_sim.md). This document retains the earlier reconstruction/protocol history; local artifact paths refer to files outside Git, with retrieval locations explained in the overview.

This environment reconstructs the PHANTOM **packaged wafer-bar experiment** in Isaac Sim 6.0. It includes the UR3, tactile gripper, optical table, gridded mat, blue collection bin and packet, together with measured-motion replay and a bridge to the existing policy interface. The updated scene reproduces the recorded pick/carry/place through physical contacts. See the [current tuning results and tactile videos](isaac_waffles_tuning.md) for revision2 launch commands and limits. Exact physical correspondence and calibrated sensor/policy transfer remain unestablished.

This guide retains the simulator architecture and **original revision1 validation** below. Its original settings are preserved in [waffles_reconstruction_v1.json](../configs/sim/waffles_reconstruction_v1.json); current defaults use the tuned revision2 geometry and camera.

The scene uses 3D geometry and rendered lighting. Recorded scene footage is used as the comparison reference, not as a camera backdrop. The packet's printed top is a rectified patch from an actual fit-episode image; [texture provenance](../assets/sim/waffles/packet_top.provenance.json) records its source pixels. This preserves observed appearance but also carries the source image's illumination and limited resolution.

The executable defaults are in [waffles.json](../configs/sim/waffles.json). The accompanying [evidence manifest](../configs/sim/waffles_evidence.json) distinguishes observed measurements, configured hardware values, nominal CAD, visual estimates and unknown quantities. Each run writes its own `effective_config.json`; that file defines the settings actually used, including object displacement and friction changes.

| Component | Evidence and current representation | Limit on interpretation |
|---|---|---|
| Robot | Actual hardware is **UR3 CB3** according to the dashboard-read note in [hardware.nuc.yaml](../configs/hardware.nuc.yaml). The six-joint arm uses pinned official Universal Robots visual/collision meshes and nominal kinematics/inertias, with [source hashes](../assets/sim/ur3/PROVENANCE.json). | The configuration's `generation: e-series` is a documented schema workaround. This is not a UR3e model or serial-specific factory calibration. |
| TCP and frames | UR controller base frame; metres, radians and rotation vectors. Configured TCP offset is 180 mm along tool Z. Actual simulated tool pose is compared independently with nominal FK. | The TCP offset is configured evidence, not a newly measured pad-tip location. |
| Gripper | Robotiq 2F-85, with a detailed [historical ROS-Industrial CAD visual model](../assets/sim/robotiq/PROVENANCE.json) and custom pad geometry. Contacts use two driven prismatic pads. A fit-only shell-centroid estimate (local artifact `artifacts/isaac_waffles/evidence/gripper_centroid_estimate.json`) refines gripper yaw and pad location. | The visual linkage and closure mapping are estimates. Historic CAD lacks the rig's DM-Tac mounts; visible white-shell centers do not identify the physical contact faces. |
| Gripper feedback | Recorded position is `POS/255`; object status uses the Robotiq `0..3` convention. Configured nominal bare-gripper stroke is 85 mm, with a rig note identifying approximately `0.9` as pad contact. | Feedback fraction is not a measured gap. The simulation's gap conversion and object-status detector are proxies. |
| Camera | Recorded D435 RGB is 640×480, configured at 15 Hz. Camera pose/focal length are estimated from manually annotated fit-episode landmarks. | No recovered intrinsic, distortion or extrinsic calibration. Depth and wrist cameras are disabled in the inspected rig configuration. |
| Table, mat, bin and packet | Geometry follows the fit footage and layout estimates (local artifact `artifacts/isaac_waffles/evidence/layout_estimates.json`). | Physical dimensions, mounting heights, grid pitch and packet pose are image estimates. Table holes are visual details over a continuous collision slab. |
| Contact and motors | PhysX rigid-body packet, friction, pad compliance and bounded joint/finger drives; configurable disturbances. | Packet mass, wrapper friction/flex, wafer fracture, gel compliance, actuator response and tactile transfer are uncalibrated. |

Original recordings remain under `/home/physicalai/phantom-icra-2027/data` on `compute3`; simulator code is installed separately at `/home/physicalai/phantom-icra-2027/sim/waffles/source`. The launcher is `/home/physicalai/phantom-icra-2027/sim/waffles/source/tools/sim/launch_waffles.sh`. It constructs no real robot or sensor driver. Preparing evidence reads Zarr groups with `mode="r"` and writes separate exports. Do not point an output argument into a source episode directory.

To inspect the scene from a graphical desktop on `compute3`, run the convenience launcher below. It uses the prepared evidence, creates a new timestamped output directory, and leaves the scene open after replay. Headless execution has been tested; interactive desktop display has not been manually verified.

```bash
/home/physicalai/phantom-icra-2027/sim/waffles/source/tools/sim/view_waffles.sh
```

The transferable scene package (local artifact `artifacts/isaac_waffles/validation/replay_fit/waffles_scene.usdz`) contains its mesh and texture dependencies. The synchronized real/sim video (local artifact `artifacts/isaac_waffles/validation/replay_fit/comparison/side_by_side.mp4`) shows the reference footage alongside the rendered reconstruction.

To prepare recordings available on the same machine, run [prepare_waffles.py](../tools/sim/prepare_waffles.py) from a Python environment with PHANTOM's recording dependencies, NumPy, SciPy and OpenCV. A fresh output directory is required for each complete export; existing completed episode exports are deliberately not overwritten.

```bash
python tools/sim/prepare_waffles.py \
  --data-root /path/to/phantom-data \
  --out /path/to/waffles-evidence
```

The corresponding command on `compute3`, using the existing project environment, is:

```bash
ssh compute3
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python /home/physicalai/phantom-icra-2027/sim/waffles/source/tools/sim/prepare_waffles.py \
  --data-root /home/physicalai/phantom-icra-2027/data \
  --out /home/physicalai/phantom-icra-2027/sim/waffles/evidence
```

Durable prepared evidence uses the latter output path. The workspace copy is artifacts/isaac_waffles/evidence (local artifact `artifacts/isaac_waffles/evidence/`). Each episode contains `replay.npz`, `reference.mp4`, sample PNGs, a contact sheet and a metadata/provenance manifest. The inventory (local artifact `artifacts/isaac_waffles/evidence/inventory.json`) records current stream availability and metadata hashes. Older full-dataset exports often retain only reduced training streams; the selected demonstration has complete motion and RGB.

All Zarr `ts` values already use the recorder's RTDE **MasterClock** domain. The exporter subtracts one shared `t0_master`; it does not apply `meta.clock_calibration.offset` again. Native measurements are preserved as `native_<stream>` and `native_<stream>_t`. The 15 Hz `t/q/qd/tcp/gripper` grid is for synchronized media and convenience, not the highest-rate motion truth. Joint positions are interpolated without wrapping their recorded branch, TCP orientations use SO(3) SLERP, and exported gripper feedback uses the preceding recorded sample. RGB uses nearest frames and records signed camera snap errors. Replay and quantitative state comparison prefer the native joint/TCP samples.

The split was fixed before scene fitting and is recorded in split.json (local artifact `artifacts/isaac_waffles/evidence/split.json`):

| Role | Episode | Use |
|---|---|---|
| Fit | `ep_teacher_waffles_1788535016_005` — September 4 | Camera/layout/packet appearance and baseline motion. |
| Fit | `ep_waffles_1787395928_000` — August 22 demonstration | Additional motion evidence; its green wrapper differs from September's cream/red wrapper. |
| Held out | `ep_teacher_waffles_1788535066_006` — September 4 | Recorded lift-complete stop with a current rule-derived failure label. |
| Held out | `ep_teacher_waffles_1788538100_000` — September 4 | Reach/wrist-extension failure evidence. |
| Held out | `ep_teacher_waffles_1788262056_000` — September 1 | Cross-session motion and appearance check; current rule-derived grasp label. |

“Held out” means excluded from fitting the scene; it does not establish exclusion from policy training. September 1 and August 22 show scene/appearance differences, so a cross-session image error includes those differences. Manual camera annotations have uncertainty, and their in-sample reprojection error is not a held-out calibration result. Current remote rule-derived labels differ from the September 5 review snapshot; the exports preserve current metadata hashes and do not promote those labels to independent operator verdicts.

Use the supplied launcher for this installation. Its `python.sh` can preload an incompatible NCCL library before bundled Torch initializes. [launch_waffles.sh](../tools/sim/launch_waffles.sh) sources `setup_python_env.sh`, preloads `kit/libcarb.so` and the matching bundled `extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib/libnccl.so.2`, then starts Kit Python. This workaround does not modify the shared Isaac installation. `ISAAC_SIM_ROOT` can override `/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0`; the bundled paths must exist in that installation.

The following commands run on `compute3`. The canonical output paths shown already contain the completed validation runs; choose new output names when experimenting to preserve those results. Add `--gui` to display the scene and leave it open for inspection after the run, or `--save-stage-only` to export its USD stage without running an episode.

```bash
cd /home/physicalai/phantom-icra-2027/sim/waffles/source
bash tools/sim/launch_waffles.sh \
  --episode ../evidence/fit/ep_teacher_waffles_1788535016_005 \
  --output ../runs/replay_fit --mode replay

bash tools/sim/launch_waffles.sh \
  --episode ../evidence/fit/ep_teacher_waffles_1788535016_005 \
  --output ../runs/dynamics_fit --mode dynamics

bash tools/sim/launch_waffles.sh \
  --episode ../evidence/fit/ep_teacher_waffles_1788535016_005 \
  --output ../runs/dynamics_push --mode dynamics \
  --push-at 8 --push-velocity 0.15 0 0 --friction-scale 0.7
```

| Mode | Behavior | Appropriate interpretation |
|---|---|---|
| `replay` | Assigns measured joint/gripper poses directly to reproduce the recorded rig trajectory. | Geometry, camera registration and kinematic comparison. Joint teleportation/zeroed velocities make contact and retention outcomes unsuitable for physics validation. |
| `dynamics` | Uses measured motion as targets for force-limited drives; reads actual joint/tool/object feedback. | Tracking, collision and sensitivity experiments under stated physical assumptions. This does not prove calibrated grasp retention. |
| `policy` | Feeds rendered/measured simulation observations to an existing policy server and executes returned actions through the deployment adapter, IK and drives. | Closed-loop integration and hypothesis testing. Sensor models and controller parity must be audited before comparing lab success rates. |
| `contact_probe` | Initializes a free packet once between the pads, then attempts pinch, lift, hold and release through drives. No attachment or subsequent object repositioning is used. | Synthetic contact/control diagnostic, not recorded-pickup parity. Check IK failures and commanded/measured tracking before interpreting contact retention. |

The probe can be initialized at a lower recorded arm pose to keep the requested lift inside the workspace. This changes its synthetic initial condition, not the contact parameters:

```bash
bash tools/sim/launch_waffles.sh \
  --episode ../evidence/fit/ep_teacher_waffles_1788535016_005 \
  --output ../runs/contact_probe_lower --mode contact_probe \
  --probe-start-time 6 --duration 10
```

Policy mode connects to an already running inference server; it does not launch hardware. Select a server that is dedicated to simulation and inspect the saved `policy_info.json` for the actual model identity. The address below is an example; use the resident server's real address and matching policy mode.

```bash
bash tools/sim/launch_waffles.sh \
  --episode ../evidence/fit/ep_teacher_waffles_1788535016_005 \
  --output ../runs/policy_v5_6 --mode policy \
  --policy-server 127.0.0.1:7777 --policy-mode teacher \
  --tactile contact_proxy --wrist contact_proxy --seed 4242
```

The policy adapter uses simulation time, the configured 125 Hz executor period and episode-recorded safety overrides. Inference latency delays plan activation on simulation time; `--policy-latency` explicitly overrides that delay for a controlled latency experiment. The tactile proxy samples at the configured tactile rate, deriving an approximate pressure field from contact forces. It does not render a calibrated DM-Tac gel image, shear/slip field or distributed force. The wrist proxy combines pad contact forces/torques with recorded initial bias, not the CB3's actual current-based estimation process. `--tactile zero_ablation` and `--wrist zero_ablation` are explicit ablations, not physical sensor models. The completed resident `v5_6` run below exercised actual plan inference and activation; this establishes interface operation, not grasp success.

Run outputs include the USD stage, a flattened USD and a self-contained USDZ package, `sim.mp4`, first/last images, `sim_trace.npz`, `effective_config.json` and `run.json`. The trace contains measured joint positions/velocities, actual tool-derived TCP, independent nominal FK, packet position, pad forces and separate reported/PhysX clocks. Policy outputs additionally retain model/settings information and planner/execution audit records. A failed run writes `FAILED.txt`; do not treat its partial video as a completed trial. The layered USD can reference imported assets; use the packaged USDZ for transfer, and retain the source and run configuration for regeneration.

Use [compare_replay.py](../tools/sim/compare_replay.py) in the recording/CPU environment to produce `metrics.json`, a synchronized `side_by_side.mp4` and `video_alignment.npz`:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python ../sim/waffles/source/tools/sim/compare_replay.py \
  --reference ../sim/waffles/evidence/fit/ep_teacher_waffles_1788535016_005 \
  --sim-trace ../sim/waffles/runs/dynamics_fit/sim_trace.npz \
  --sim-video ../sim/waffles/runs/dynamics_fit/sim.mp4 \
  --out ../sim/waffles/runs/dynamics_fit/comparison
```

State comparisons use each native stream's timestamps and only their common observed support. Joint errors preserve raw multi-turn branches. TCP translation uses Euclidean distance and orientation uses geodesic SO(3) distance. An independent asset-versus-nominal-FK result detects import/frame disagreement. Reported versus PhysX time must agree within **0.5 ms**; a larger mismatch leaves an explicit failed report and makes the comparator exit with status `2`. Missing independent clocks are marked unevaluated. Full-frame RGB MAE is photometric appearance error; the largest-blue-component IoU/centroid distance is only an uncalibrated static-bin silhouette proxy. Neither is a calibrated robot/object alignment measurement.

The original revision1 seven-run campaign completed; summary.json (local artifact `artifacts/isaac_waffles/validation/summary.json`) records native-sample metrics, independent clock checks, packet trajectories and policy audit counts. All seven independent physics clocks passed the 0.5 ms check, and all local simulation videos matched their trace frame counts. The largest observed clock discrepancy was below 0.9 microseconds. Per-run metrics and comparison videos are retained in the validation artifacts (local artifact `artifacts/isaac_waffles/validation/`).

| Run | TCP translation RMSE (mm) | Joint RMSE (rad) | Blue-bin silhouette IoU |
|---|---:|---:|---:|
| Replay fit (local artifact `artifacts/isaac_waffles/validation/replay_fit/comparison/metrics.json`) | 1.379 | 3.69e-8 | 0.908 |
| Held-out September 1 (local artifact `artifacts/isaac_waffles/validation/replay_heldout_1788262056_000/comparison/metrics.json`) | 1.392 | 3.75e-8 | 0.833 |
| Held-out September 4, 5066 (local artifact `artifacts/isaac_waffles/validation/replay_heldout_1788535066_006/comparison/metrics.json`) | 1.294 | 3.79e-8 | 0.909 |
| Held-out September 4, 8100 (local artifact `artifacts/isaac_waffles/validation/replay_heldout_1788538100_000/comparison/metrics.json`) | 1.330 | 3.78e-8 | 0.909 |
| Driven fit (local artifact `artifacts/isaac_waffles/validation/dynamics_fit/comparison/metrics.json`) | 4.580 | 0.00602 | 0.909 |
| Driven disturbance (local artifact `artifacts/isaac_waffles/validation/dynamics_push/comparison/metrics.json`) | 4.580 | 0.00602 | 0.908 |

The three held-out replay episodes have a sample-weighted TCP translation RMSE of **1.340 mm**, with a maximum error of **1.500 mm**. Near-zero replay joint errors are expected because those modes explicitly assign recorded joints; the residual TCP error reflects nominal-model/calibration differences. The blue-bin scores describe one static silhouette, not calibrated whole-scene alignment. The baseline driven run did not lift the packet. The velocity disturbance moved it **12.785 mm** across the mat, with no meaningful rise. These results demonstrate driven tracking and a response to disturbance, while leaving the recorded pickup unreproduced.

The full v5_6 policy run (local artifact `artifacts/isaac_waffles/validation/policy_v5_6/run.json`) received 16 plans and activated 15 at the configured 125 Hz executor rate. Median inference latency was **0.913 s**; there were no IK rejections or policy stop, but the packet did not lift. The policy comparison (local artifact `artifacts/isaac_waffles/validation/policy_v5_6/comparison/metrics.json`) measures divergence from the original recording, not tracking an intended recorded trajectory. The checkpoint was independently hashed on `compute3`: `7edcb8335681e19bead5ad39a2a80fe3fd8c5893b1761d2dd812c304881fe65c` for `runs/teacher_v5_batch0822/v5_6.pt`.

An additional first synthetic contact probe (local artifact `artifacts/isaac_waffles/validation/contact_probe/run.json`) produced positive pad contact forces and observed retention/release, but **failed the intended reach/control test**: it accumulated 1,918 IK rejections, with approximately 75 mm final target-position error. Its observed packet rise is not evidence that the requested 100 mm lift was reproduced.

The lower-pose repeat (local artifact `artifacts/isaac_waffles/validation/contact_probe_lower/run.json`), using the same contact parameters and the six-second recorded arm pose as its initial seed, completed the synthetic pinch/lift/hold/release sequence with **zero IK rejections**. The commanded lift was **99.90 mm**, the retained packet's reported rise was **96.51 mm**, and peak pad contact force was **3.39 N**. After opening, the packet fell to the table. Its video (local artifact `artifacts/isaac_waffles/validation/contact_probe_lower/sim.mp4`) and trace show a free body retained through contact, without an attachment or repeated object repositioning. Both probe clocks and finalized video/trace frame counts passed their checks. This verifies the operation of this synthetic contact diagnostic under the chosen approximate parameters; the recorded table pickup remains unreproduced, and contact/tactile transfer remains uncalibrated.

The CPU checks cover measured UR3 pairs/frame conventions/IK, policy adapter and remote interface behavior, replay clocks/native interpolation/geodesic errors, and the visual gripper loader. All **58 tests** in the five-module suite passed; these tests do not calibrate the physical scene. Run the relevant suite in an environment with the repository's test dependencies:

```bash
python -m pytest -q \
  tests/test_sim_kinematics.py \
  tests/test_sim_policy_adapter.py \
  tests/test_sim_remote_policy.py \
  tests/test_sim_replay_metrics.py \
  tests/test_sim_gripper_visual.py
```

To establish a usable lab-correlated simulator, freeze the scene/configuration and complete the following measured calibration work before tuning on the held-out episodes:

1. Recover serial-specific UR kinematics and measure the tool/pad transforms, gripper aperture versus feedback, mounting plate/table dimensions and packet dimensions/mass. Keep nominal and measured versions identifiable.
2. Calibrate camera intrinsics/distortion and camera-to-base pose with measured fiducials across the workspace. Evaluate withheld landmark observations and report uncertainty; do not choose acceptance thresholds after seeing the evaluation errors.
3. Identify arm/finger drive response and contact parameters using controlled motion, closure-force, slide, squeeze, lift and release trials. Check all relevant collision surfaces and preserve observed failure cases, including reach/IK, closure and retention failures.
4. Match RGB/tactile/wrench sampling, delays, noise and observation preprocessing. Teacher-policy comparisons need a calibrated tactile observation model or an explicitly bounded sensor ablation; proxy images alone cannot establish tactile transfer.
5. Replay the untouched held-out episodes, then run matched closed-loop perturbations with fixed seeds, model hashes, recorded effective settings and explicit lab outcome judgments. Treat simulated success as a hypothesis until those trials reproduce both successes and failures.
