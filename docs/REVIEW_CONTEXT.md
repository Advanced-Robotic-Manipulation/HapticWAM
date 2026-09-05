# Review context — PHANTOM robot deployment stack (read first)

This PR is a **review scaffold**: the base branch is `main` with the deployment
stack removed, so the diff shows the complete files that move a real robot. Nothing
here is being merged; the goal is an adversarial correctness review of code that runs
a physical arm next to people.

## What runs on the rig
- **Robot**: UR3 CB3 (Polyscope 3.14) driven over ur_rtde at **125 Hz servoJ**
  (`control.executor_rate_hz = arm.rtde_control_hz = 125`, one tick = 8 ms; CB3 cannot
  go faster). Every `getInverseKinematics` call is an RTDE round trip on the robot's own
  8 ms control cycle. Robotiq 2F-85 gripper over RS-485/USB. Two Daimon DM-Tac tactile
  pads on the fingertips (SDK on CPU, ~8 Hz fields; per-pad force `fz` in N). RealSense
  D435 camera at ~15 Hz. No wrist F/T sensor: `wrist_ft` is a current-based estimate
  with a large pose-dependent bias, hence the rolling-baseline wrench guard.
- **Policy**: a Cosmos-Predict2.5-2B world-action model with LoRA (the "teacher", uses
  tactile input) served by `phantom/scripts/policy_server.py` (one process per model,
  localhost port 7777/7778, `multiprocessing.connection` with an authkey). The robot
  process `phantom/scripts/run_deploy.py --policy-server auto` attaches as
  `RemotePolicy`. A replan takes ~0.8–1.2 s on the RTX 5090 and returns a **chunk** of
  Cartesian TCP poses + gripper apertures (0 = open, 1 = closed) at 10 Hz plus a
  "contact package" (`cpk`, GPU tensors that never cross the wire — replaced by an
  integer token).
- **Process layout** (`phantom/deploy/runtime.py`): one `DeploymentRuntime` per
  process builds the rig session (sensor rings + recorder), a `PlannerLoop`
  (`phantom/deploy/planner.py`, replans and applies the "terminal veto" state machine
  to the gripper channel), a `ChunkExecutor` (`phantom/deploy/executor.py`, its own
  thread: streams the current chunk at 125 Hz with a Cartesian rate limit and a
  cross-fade between chunks, sends gripper commands, records what was actually sent),
  and a `SafetyMonitor` (`phantom/deploy/safety.py`, checked every executor tick).
  Threads: executor thread, planner loop (main), recorder drain threads, gripper
  worker, sensor workers. `RTDEControlInterface` is NOT thread-safe: `URArm` wraps it
  in `_ctrl_lock`.

## Safety stack (all thresholds tuned from recorded rig data; see docs/rig_session_v5.md)
Layered, in order of how early they act:
1. **Driver IK branch guard** (`phantom/drivers/real/ur.py::servo_l`): IK is seeded with
   the previous solution; a solution that jumps more than `IK_BRANCH_TOL_RAD` (0.35) is
   refused (hold); 25 consecutive refusals raise and the executor's crash net stops the
   arm. Root cause of every 09-01 "whip" was an unseeded IK returning the other elbow
   branch at the elbow-straight boundary (wrist-centre distance 470.5 mm).
2. **Servo-level limiter** (same file, `_limited_step`): **disabled by default**
   (`elbow_min_rad`/`servo_joint_speed_max_rad_s` = None). Known-defective, see
   docs/review_servo_limiter_0905.md; do not spend effort re-finding those items.
3. **SafetyMonitor.check()** per tick: wrist-extension stop (`wrist_extension_stop_m`,
   pure elbow geometry from DH constants `ur_dh_a2_a3_d4_m`), measured joint speed stop
   (`joint_speed_stop_rad_s` 1.2), workspace box clamp, hitbox (only the TOP face exit
   is a stop), wrench limit with rolling baseline + debounce, tactile force/depth limits,
   stale-plan timeout, and the `lift_complete` success event (trailing-window pad load
   AND TCP height, held; disabled when `lift_complete_z_m == 0`, which is the CLI
   default since 09-04).
4. **Executor**: chunk-tail cap (`--max-play-steps 10`: steps past the validated head of
   a chunk are never executed, the arm holds until the next plan), Cartesian rate
   limit, **aperture latch** (`_latched_grip`: once both pads carry > `grip_latch_fz_n`
   the commanded closure can only increase until an intended release clears it — 4 of
   5 "lost" objects on 09-04 were policy-commanded releases mid-carry), `stopJ` on
   every halt, `LETGO_STOP_REASONS` decide whether a stop opens the gripper.
5. **Planner veto** (`_apply_veto`): rewrites the gripper channel — masks closes the
   model proposes before the demo close band, recovery rules (`recovery_open`,
   `recovery_tactile`: closed on air + risen 30 mm + both pads quiet 1 s → open and
   re-descend), a per-episode retry cap that ends the episode.
6. **Episode termination** (`PlannerLoop.run`): operator Enter (`operator_stop`),
   `lift_complete`, guard stops, replan budget, 150 s wall budget. `run_deploy`
   persists the reason, safety events and arm state to `stop.json` and meta tags.

## Config semantics
- `configs/hardware.nuc.yaml` is the rig config (NUC/compute3). Sections: `arm`
  (IP, RTDE rates, servoJ lookahead 0.1 s / gain), `gripper` (force clamped to the
  pad ceiling: 30 N total per pad max; `default_force` 0.04), `tactile` (serials,
  SDK backend, rates), `cameras`, `derived`, `recording` (zarr streams, ring sizes),
  `control` (executor rate), `teleop`, `safety` (this is the section that matters;
  yaml overrides the `SafetyConfig` pydantic defaults in `phantom/config/hardware.py`;
  `run_deploy` further overrides some fields from CLI flags via `model_copy`).
- `SafetyConfig` is a frozen pydantic model; field comments carry the data that
  justified each default. Units: metres, newtons, rad/s, seconds.
- Training configs (`configs/compute.yaml`, `configs/train*.yaml`) are NOT part of this
  review; the deployment code only consumes a checkpoint via `PhantomPolicy`.

## What we want found
- Any path by which an unsafe or discontinuous pose can be streamed to `servoJ`
  (IK seeding/branch logic, cross-fade math, rate limit, `_pose_at` indexing, tail cap
  edge cases, what happens on the first tick of a chunk, on a replan that arrives
  mid-chunk, on a stale plan, on reconnect).
- Races between executor thread, planner, gripper worker, recorder and safety
  (stop/halt ordering, `_grip_latch` clearing, `stopped_reason` overwrites, lock
  coverage of RTDE calls, join timeouts).
- Gripper release paths: every place the gripper can be commanded open while an object
  may be held, and whether that is intended.
- Correctness of what gets recorded (`record_action`, gripper `g_sent`, latched value,
  stop.json), since recorded rollouts feed the next training round.
- Policy-server protocol: token/cpk lifetime, reconfigure semantics, one-client-at-a-
  time behaviour, connect timeout, failure modes that leave the robot process hanging.
- Safety monitor arithmetic: baselines, trailing windows, debounce, `recovered()`
  livelock, unit mistakes, off-by-one on tick counts, NaN handling on sensor dropouts
  (tactile frames can go stale; camera can drop).
- Anything in `run_deploy` start gate / auto-home that can move the arm unexpectedly.

## What not to flag
Style, naming, docstring length, type-hint completeness, "consider adding tests"
without a concrete failing scenario, and the limiter items already listed in
docs/review_servo_limiter_0905.md.
