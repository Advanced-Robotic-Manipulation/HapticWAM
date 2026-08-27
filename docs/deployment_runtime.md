# Deployment runtime

The receding-horizon loop of pipeline.md §6e, as implemented in
`phantom/deploy/`.

## Architecture (single host, the 5090 box)

```
tactile L  ──proc──▶ ring: fields_ds / keyframes / img ─┐
tactile R  ──proc──▶ ring: ...                          │
arm (RTDE) ─thread─▶ ring: q qd tcp ft pstop            ├─▶ SnapshotBuilder ─▶ PhantomPolicy.replan (GPU)
gripper    ─thread─▶ ring: pos cur                      │        │ Plan(actions, σ, gate, ĉ)
camera     ─thread─▶ ring: color                        │        ▼
                                                        │   ChunkExecutor (thread @ executor_rate_hz)
EpisodeRecorder ◀── drains all rings every 0.25 s ──────┘        │ servoL / gripper
                                                            SafetyMonitor (every tick, before send)
```

- Tactile sensors run in **their own processes** (spawn-safe; in-child
  downsample + f16; full-res frames cross IPC only as decimated keyframes).
- The planner runs in the calling thread (GPU-bound; torch releases the GIL),
  the executor in a dedicated thread. This is the pragmatic single-host
  layout; a zmq 3-process split is the documented upgrade path if executor
  jitter under planner load proves too high (measure first).
- **AR grounding is structural**: every replan reads real sensors from the
  rings; imagination re-enters only through `prev_plan.cpk` — ACC's designed
  one-step staleness.
- **Tactile always records, in every system mode** — student modes merely
  exclude it from the model input. Load-bearing for DAgger relabeling and for
  peak-force metrics on student trials.

## Per-replan cycle

1. `SnapshotBuilder.build()` — latest camera frame, wrist-F/T window, UR
   state (+ tactile keyframe/gel/contact-state in teacher mode) from the rings.
2. `PhantomPolicy.replan(obs, prev_plan, tcp_pose)` — HHT encode → ACC gate
   (consuming the PREVIOUS replan's generated contact package) → `nfe`-step
   joint denoise → unpack: denormalized action chunk on the action-rate grid,
   σ profile, gate/event diagnostics, new contact package.
3. `ChunkExecutor.submit(plan)` — accepted only if the plan still covers
   `now + control.replan_min_lead_s`.
4. Executor plays the chunk: cumulative-delta pose interpolation from the
   plan's reference TCP pose, **playback speed scaled by the governor**
   (σ → `g(σ)` time reparametrization — the path is preserved, the robot just
   slows where its own hallucination is uncertain).
5. Repeat. Stale plan (no fresh chunk within `safety.stale_plan_timeout_s`
   past the chunk end): decelerate to hold, servo session kept alive.

## Safety behavior matrix (checked EVERY executor tick, before the command)

| Condition | Source | Action |
|---|---|---|
| wrist |F| > `safety.wrench_limit_N` or |T| > `wrench_limit_Nm` | arm ring | controlled stop, episode aborted (`safety_stop`) |
| peak fingertip f_z > `tactile_fz_limit_N` (calibrated) / indentation > `tactile_depth_limit` (fallback) | tactile rings | controlled stop, episode aborted |
| tactile ring stale (sensor stall) | ring freshness | controlled stop, episode aborted |
| `camera_scene` ring stale > `safety.camera_stale_s(hw)` (wedged RealSense) | ring freshness | controlled stop, episode aborted (`safety_stop`) |
| `arm` ring stale > `safety.arm_stale_s(hw)` (dead RTDE-receive worker / stalled stream) | ring freshness | controlled stop, episode aborted (`safety_stop`); `tcp_pose`, `tcp_speed`, the F/T window and the protective-stop flag all freeze together, so nothing below this row is trustworthy |
| TCP target outside `safety.workspace_m` | commanded target | clamp to the box, continue (logged) |
| protective stop | RTDE flag | executor exits (`protective_stop`); see recovery below |

All events are t_master-stamped in `SafetyMonitor.log_events` and counted in
the episode result; the eval harness reads them. A *sustained* condition (a
clamp, a stale ring) is retained and logged ONCE per condition — the rising
edge, then at most 1 Hz while it holds, with the tick count on the event —
so `EpisodeResult.safety_events` counts problems, not executor ticks, and the
2 ms servo loop is not doing 500 stderr writes a second.

`SnapshotBuilder.build()` carries the same camera- and arm-freshness thresholds
as hard requirements (like the gripper-state one), so a frozen scene stream or a
frozen robot state can never condition a replan even with no SafetyMonitor in
the loop. They raise `planner.StaleStreamError` (an `AssertionError` subclass,
so ring warm-up still treats it as "not ready yet"), carrying the stop reason
the planner/runtime convert it into — `camera_scene_stale`, `arm_stale`,
`gripper_stale`. The episode then ends through the NORMAL stop path: the
executor is halted, the planner trace is written, and the operator still gets
the success/notes prompt. Before 2026-08-27 these escaped `run_episode` as an
rc-1 traceback and left an empty episode directory with no label.

`EpisodeResult.stopped_reason` values and what run_deploy does with each:

| reason | meaning | run_deploy |
|---|---|---|
| `protective_stop` | UR protective stop (control script dead) | rebuild control before the next episode |
| `safety_stop` | SafetyMonitor STOP_EPISODE (wrench, tactile, `arm_stale`, `camera_scene_stale`) | rebuild control before the next episode |
| `motion_stall` / `executor_crash` | arm not following / servo thread died | rebuild control before the next episode |
| `servo_stop_failed` | `servoStop()` failed against a script that is still playing — `_servo_active` is stuck, homing `move_l` would be refused | rebuild control before the next episode (the rebuild clears the guard) |
| `camera_scene_stale` / `arm_stale` / `gripper_stale` / `snapshot_invalid` | planner-side snapshot rejection | episode ends cleanly, labelled as usual |
| `worker_died` | a sensor/arm worker thread or process is gone (includes a RealSense whose bounded rebuild gave up: its poller exits) | **session-fatal**, exit code 5 |
| `executor_join_timeout` | a stale executor/gripper thread still owns the device | **session-fatal**, exit code 5 |

Ring warm-up at episode start (`runtime.warmup_timeout_s()`) is derived from
`drivers/real/realsense.first_frame_budget_s()` — the driver tolerates three
5 s `wait_for_frames()` timeouts before rebuilding the pipeline, so a camera
that is healing itself needs ~17 s to produce its first frame. The old flat
10 s budget aborted the episode mid-recovery.

## Protective-stop recovery

1. Clear the fault on the pendant (Enable robot).
2. The RTDE control session is dead. `run_deploy` handles this between
   episodes: when the previous episode ended with any of
   `_CONTROL_DEAD_REASONS` (`protective_stop`, `executor_crash`,
   `motion_stall`, `servo_stop_failed`, `safety_stop`) it blocks on
   `URArm.is_ready_for_control()` (prompting until the pendant is clear),
   calls `URArm.reconnect_control()`, and verifies the new control script is
   actually **running** before homing the next episode. If it cannot recover
   in 3 attempts the campaign stops with exit code 4 rather than running
   zero-motion episodes. Mid-episode, the trial is finished as failed.
   If the start-pose homing `move_l` then fails anyway — right after the script
   was rebuilt and verified playing — the campaign also stops with exit code 4:
   the arm is not controllable from this process (Local mode, sticky servo
   guard, IK reject), and continuing is what produced whole campaigns of
   hand-jogged OOD starts. A homing failure with no preceding recovery keeps
   the old tolerant behaviour (jog by hand; the start gate re-checks).
3. Re-zero F/T (`0` in teleop, automatic at episode start otherwise).

`URArm` invariant: `_servo_active` is true only while a servo session may be
live on the control script running *right now*. It is cleared on a successful
`servo_stop()`, on `_teardown_ctrl()`/`reconnect_control()` (a fresh script has
no servo stream), and on a failed `servoStop()` whose receive-side check shows
the script is not running. It stays set only when `servoStop()` failed against
a script that is still playing. A sticky true would refuse every subsequent
`move_l`, i.e. disable start-pose homing for the rest of the session.

## System modes (`--system`)

`teacher` (full tactile input) · `student` · `vision_only` · `no_distill` ·
`drop_tactile` — modes select which streams the model's snapshot exposes
(`deploy/planner.py::TACTILE_INPUT_MODES`); pair each with the matching
checkpoint (see the eval campaign YAML).

## Latency budget

Fill the measured column after the week-1 5090 smoke
(`python -m phantom.scripts.smoke_test --synthetic --device cuda` for the
denoise; a mock `run_deploy --tiny` for the loop overheads).

| Stage | Budgeted | Measured (5090) |
|---|---|---|
| Snapshot build + HHT encode | 50–150 ms | (inside replan total) |
| `nfe`-step denoise (2B, extended layout) | 1–3 s | 1.03–1.41 s whole replan (v3 ckpt, mock drivers, 2026-08-11) |
| Plan transport + swap | ~ms | |
| Executor tick jitter (std) | < 2 ms | |
| Chunk duration (`H / action_rate_hz`) | 1.6 s @ H=16 | 1.6 s — replan fits inside it |

If measured denoise exceeds the chunk duration, raise
`control.chunk_horizon` (config only — the layout adapts; keep it a multiple
of 4) or lower `nfe`; the stale-plan hold covers transients either way.

## Dry run

```bash
# no hardware, no checkpoint, mock everything:
python -m phantom.scripts.run_deploy --system teacher --tiny --task smoke --max-replans 3
```

For a dry run with the REAL model and checkpoint (mock drivers, any CUDA
box), see [inference.md](inference.md).
