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
| TCP target outside `safety.workspace_m` | commanded target | clamp to the box, continue (logged) |
| protective stop | RTDE flag | executor exits (`protective_stop`); see recovery below |

All events are t_master-stamped in `SafetyMonitor.log_events` and counted in
the episode result; the eval harness reads them. A *sustained* condition (a
clamp, a stale ring) is retained and logged ONCE per condition — the rising
edge, then at most 1 Hz while it holds, with the tick count on the event —
so `EpisodeResult.safety_events` counts problems, not executor ticks, and the
2 ms servo loop is not doing 500 stderr writes a second.

`SnapshotBuilder.build()` carries the same camera-freshness threshold as a
hard assert (like the gripper-state assert), so a frozen scene stream can
never condition a replan even with no SafetyMonitor in the loop.

## Protective-stop recovery

1. Clear the fault on the pendant (Enable robot).
2. The RTDE control session is dead. `run_deploy` handles this between
   episodes: when the previous episode ended with `protective_stop`,
   `executor_crash` or `motion_stall` it blocks on
   `URArm.is_ready_for_control()` (prompting until the pendant is clear),
   calls `URArm.reconnect_control()`, and verifies the new control script is
   actually **running** before homing the next episode. If it cannot recover
   in 3 attempts the campaign stops with exit code 4 rather than running
   zero-motion episodes. Mid-episode, the trial is finished as failed.
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
