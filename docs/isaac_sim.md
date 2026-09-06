The Isaac Sim reconstruction supports measured-motion replay and closed-loop PHANTOM policy execution for the packaged waffle experiment. The scene includes the UR3 CB3, Robotiq gripper with custom pads, table, mat, packet, box and rendered camera. Packet motion comes from physics; policy rollouts do not attach it to the gripper or replay an object trajectory.

The completed teacher experiment achieved **5/40 physically verified pick-to-box placements**, with nominal **0/2**. All five placements used one sampling seed and were followed by arm-extension stops. There is no reliable winning configuration. The [published results](results/teacher_pick_place_v1/README.md) include every trial, plots, hashes, the tactile diagnosis and two explicitly selected full scene/tactile videos. All 40 videos and raw observations remain on compute3.

Use these documents for the current work:

| Purpose | Document |
|---|---|
| Experiment results, controller changes and reproduction command | [Teacher pick-to-box](isaac_teacher_pick_place.md) |
| Tomorrow's attended hardware pilot and pinned student models | [7 September lab handoff](isaac_lab_handoff_20260907.md) |
| Four nominal student compatibility trials and their failure outcomes | [Student smoke results](isaac_student_smoke.md) |
| Ready student identities, EMA, normalization and modality audit | [Student checkpoint inventory](results/student_readiness_20260906/student_inventory.md) |
| Proposed mapper validation, placement randomization and later arm perturbations | [Next experiments](isaac_next_experiments.md) |
| Scene architecture, evidence and original reconstruction | [Scene guide](isaac_waffles.md) |
| Measured pick/place replay and tactile side-by-side validation | [Replay tuning](isaac_waffles_tuning.md) |
| Earlier pickup debugging and separate benchmark protocol | [Pickup diagnostics](isaac_teacher_debug.md), [matched campaign](isaac_policy_campaign.md) |

The separate four-trial student smoke completed with no acquisitions, lifts or placements for either revised ftA or v5_6 (two nominal seeds each). Revised ftA stopped on wrench limits; v5_6 stopped on workspace or arm-extension limits. This verifies execution and exposes failures; it does not rank students or establish physical pickup readiness.

The simulation release controller is an explicit extension to the deployment adapter. It has **not** been ported to the physical executor. The students omit fingertip tactile model tokens, retain wrist force/torque, and still share tactile-dependent safety and latch logic. The current tactile mapper has a documented sparse-contact discontinuity; material properties, camera calibration and sensor transfer remain estimates. The handoff therefore starts with matched physical pickup checks and places a separate validation gate before autonomous placement.

Simulation source is under `phantom/sim/` and `tools/sim/`; frozen scene/campaign settings are under `configs/sim/`; mesh licenses and asset provenance are under `assets/sim/`. The launcher uses the installed Isaac Sim 6.0 at `/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0`. See the scene guide for environment setup and the teacher protocol for the exact frozen campaign invocation. Default campaign invocation writes a plan; `--execute` starts owned inference and simulator processes. It does not start hardware drivers.

For CPU validation in an environment with the project dependencies and `sim` extras:

```bash
PYTHONPATH=. python -m pytest -q tests/test_sim_*.py
ruff check phantom/sim tools/sim tests/test_sim_*.py
```

The publication branch was based on native commit `0259ed3355f2ad3078778e7cc347b100775e06d3`; all **244 simulation tests** and Ruff passed on that base. These checks validate implementation behavior and regression cases, not physical calibration or hardware operation. No hardware action was executed for this handoff.

Large artifacts are deliberately outside Git. The original recording store is `/home/physicalai/phantom-icra-2027/data`; prepared simulator evidence and runs are under `compute3:/home/physicalai/phantom-icra-2027/sim/waffles/`. Historical guides identify local artifact paths under `artifacts/isaac_waffles/`. To retrieve a specific artifact, use its relative suffix: `evidence/` maps directly to remote `evidence/`, while experiment directories such as `teacher_debug_v1/` map to remote `runs/teacher_debug_v1/`. Some historical local `validation/` mirrors map individual cases into remote `runs/`; inspect the recorded run manifests before copying. The teacher results README gives the exact command for its complete video set. Preserve source recordings and existing frozen output directories; use a new output directory for every new experiment.
