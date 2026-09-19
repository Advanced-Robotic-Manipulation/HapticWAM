# `phantom.data_collect` — Echo teleop + data collection app

```
python -m phantom.scripts.collect                 # panel at http://<host>:8899
python -m phantom.scripts.collect --config configs/data_collect.yaml --host 0.0.0.0
```

A ground-up operator application for collecting HapticWAM demonstrations on
the UR3 + Echo exoskeleton rig: device-rate teleop, continuous gripper, a
latched DM-Tac force safeguard, a live rerun view, a web control panel, and
session-end offload to an external drive. It sits alongside the reference
stack (`phantom.scripts.panel` / `record_episodes.py`, documented in
[data_collection_sop.md](data_collection_sop.md)) — same rig, same reused
drivers/recording layers, different application layer.

## Why this exists

The reference stack's web page showed live DM-Tac data (depth, gel, 6-axis
force) fine, but **teleop itself felt awful**. The root cause: the 125 Hz
servo streamer was fed by the **10 Hz record loop** — `EchoTeleop.
latest_q_target()` was built for a high-rate path but had zero callers
anywhere in that stack. The arm chased a 100 ms staircase through an
acceleration tracker capped at 1 rad/s (well under human joint speed) behind
a ~160 ms one-euro filter — 200–400 ms of stacked lag plus a 10 Hz
surge/brake lurch. The known-good vendored `third_party/echo_teleop/
main11.py` sends every fresh leader sample straight to
`servoJ(lookahead=0.1, gain=200)` and lets the UR controller do the
smoothing — no software lag added at all.

`phantom.data_collect.teleop.DirectServoStreamer` restores that architecture:

- a thread at `control_rate_hz` (125 Hz, CB3-safe) pulls the leader's cached
  target **directly** every cycle — the record/session loop is never in the
  command path, it only reads `last_cmd` for logging;
- **TRACK is a pure per-cycle slew limit** (`|Δq| ≤ v_max·period`): zero
  added lag below `v_max` (default 3 rad/s, above human motion — main11 has
  no software cap at all), not the sqrt-braking acceleration law (which adds
  50–150 ms of speed-dependent lag and is only appropriate for a slow,
  bounded glide);
- the acceleration-limited tracker is kept for **ENGAGE only** — session
  start, safeguard resume, and post-protective-stop re-entry all glide from
  the measured arm pose to the leader pose at a slow, explicit rate
  (`engage_v_max_rad_s`) instead of jumping;
- the one-euro filter on the leader signal is retuned from a 1 Hz cutoff
  (~160 ms lag) to 5 Hz (~8 ms — just enough to kill encoder quantization).

## Architecture

```
panel HTTP threads     buttons -> queue; POST /api/safeguard -> live TactileSafeguard
runner thread          run_session(): rig up -> MAIN LOOP (bookkeeping only,
                        10-50 Hz) -> rig down -> offload
echo reader thread     ~100 Hz serial poll -> cached filtered joint target + gripper
streamer thread        125 Hz: leader target -> slew limit / engage glide -> servo_j
gripper pilot thread   <=50 Hz: pulls the leader's freshest squeeze -> gripper.move
safeguard thread       50 Hz DM-Tac scan -> on trip: streamer.hold() + gripper open
sensor workers/pollers phantom.recording.workers (one process per DM-Tac sensor)
recorder drain thread  rings -> zarr episode every 0.25 s
rerun thread           15 Hz live view (mode-gated)
```

The main loop is deliberately dumb: episode start/stop, absolute-action
logging, safeguard/arm-guard bookkeeping, panel state. It was the reference
stack's single point of failure (feeding the servo loop) — here it never
touches motion at all.

All phase transitions inside `DirectServoStreamer` are **compare-and-set**
under a lock, so a safeguard `hold()` fired from another thread can never be
silently overwritten by an in-flight ENGAGE→TRACK transition — an earlier
draft of this had exactly that race (an adversarial code review caught it,
along with 24 other findings, all fixed before this app was pushed).

## What gets recorded

All timestamps share one master clock (RTDE-derived).

| stream | content | rate |
|---|---|---|
| `arm_q` `arm_qd` `arm_tcp_pose` `arm_tcp_speed` `arm_ft` | **absolute** UR3 state | RTDE rate (125 Hz, CB3) |
| `actions_abs` | **absolute** commanded action `[q_target(6), gripper(1)]` | control rate |
| `gripper` | opening + gOBJ status (0..3) | ~100 Hz |
| `camera_scene_color` | RealSense RGB 640×480 | 30 Hz |
| `tactile_{left,right}_wrench` / `_area` | 6-axis force (N) + contact area | tactile rate |
| `tactile_{left,right}_fields_ds` / `_keyframes` / `_infer_img` | field stack + gel image | **full mode only** |

`actions_abs` is the absolute joint target actually commanded plus the
absolute gripper position sent — not the Δ-EE actions the reference stack
derives from consecutive TCP poses (both are legitimate; this app records
the raw teleop command directly since the streamer already has it).

## Collection modes

Chosen per session in the panel wizard:

- **full** — every DM-Tac modality HapticWAM needs (field stack, keyframes,
  gel image, wrench, area), scene RGB, and full UR3/gripper state; the rerun
  view shows depth/deformation/shear/force-z maps and the gel image per
  finger.
- **lite** — UR3 + RealSense RGB + per-sensor **6-axis force only**; the
  heavy field/keyframe/gel streams are neither recorded nor shown (rerun
  displays only the wrench plots). Useful for fast iteration or when only
  the resultant force matters.

The mode is applied via `EpisodeRecorder`'s `stream_filter` hook
(`phantom.data_collect.session.lite_stream_filter`) and mirrored in
`CollectRerun`'s blueprint.

## The DM-Tac safeguard

`phantom.data_collect.safeguard.TactileSafeguard` watches both sensors on
its own 50 Hz thread and trips when, for either sensor:

- resultant fingertip force ‖F‖ (`getForce`, calibrated Newtons) exceeds
  `safeguard.force_limit_n` (default 4 N — the pad crushes at 30 N), **or**
- peak indentation depth exceeds `safeguard.depth_limit` (the uncalibrated
  fallback, always active), **or**
- the sensor's stream has gone stale (dead/hung worker — a frozen ring must
  not hide an overload behind its last benign sample).

On trip:

1. teleop **freezes** (`DirectServoStreamer.hold()` — takes priority over
   any concurrent phase transition) and the gripper is **commanded fully
   open** (`GripperPilot.open_now()`);
2. the running episode is saved as a failure
   (`safeguard:<sensor>:<force|depth|stale>`);
3. the panel shows a red banner with the trip reason;
4. **nothing auto-resumes.** Collection continues only when the operator
   clicks **RESUME COLLECTION** in the panel — no process restart, no
   config reload. Resume re-arms motion *before* clearing the latch (so a
   fresh trip during the resume window still freezes things, rather than
   landing on a rig that already thinks it's clear) and the gripper follows
   the leader's **current** squeeze value, never replaying the stale
   pre-trip one.

`enabled`, `force_limit_n` and `depth_limit` are all editable live from the
panel (`POST /api/safeguard`) — disabling the safeguard while tripped also
releases the frozen motion, so there's always a way out of a stuck rig.
Grip force and close position are additionally hard-clamped at the
`RobotiqGripper` driver boundary (`cmd_force_limit_N`, `max_close_cmd`)
regardless of what this app ever sends.

## `configs/data_collect.yaml` — the operator's file

Everything that differs between rigs/machines:

- `ports` — DM-Tac left/right `dev_id` (yellow-cable serial), RealSense
  serial, Echo USB VID/PID;
- `storage` — local staging root, `external_drive` path, minimum free space,
  verification mode (`size`/`sha256`), `require_separate_device` (refuses to
  offload onto an unmounted mountpoint that's really the local disk);
- `teleop` — tracking bounds (`v_max_rad_s`, `a_max_rad_s2`), engage glide
  speed, one-euro filter tuning;
- `gripper` — tick calibration (or use the panel's Calibrate button), EMA,
  deadband, speed, command rate;
- `safeguard` — enabled + both thresholds (session-start defaults; editable
  live afterward);
- `default_mode` — `full` or `lite`;
- `panel` / `rerun` — host/port.

Per-machine values that must not be committed (drive letters, real serials)
go in the gitignored `configs/data_collect.local.yaml`, deep-merged on top.
`load_collect()` patches these ports/tuning values into the named
`hardware.yaml` **before** validation, so `HardwareConfig` stays the single
source of truth for shapes/rates.

## Episode playback + recording verification

`phantom.data_collect.playback` closes the loop on "did everything actually
get recorded?" — directly inside the same web GUI:

- the **Episodes** card lists every episode found on the **external drive**
  and in local staging (grouped by session, newest first; the ↻ button
  rescans — plug the drive into the workstation and refresh);
- **Verify recording** runs a read-only integrity check and renders a
  per-stream report table under the rerun view: rows, duration, achieved vs
  expected rate (from `hardware.yaml`), shape, and flags. Hard failures
  (missing expected stream for the episode's mode, empty stream,
  non-monotonic timestamps, ts/data length mismatch, NaN/Inf values,
  non-finalized meta) make the verdict red; soft findings (rate low, >5×
  timestamp gap, a stream covering only part of the episode) are yellow
  warnings;
- **▶ play** streams the whole episode — every recorded modality — into the
  embedded rerun viewer: scene RGB, per-finger depth/deformation/shear/Fz
  maps and gel image (full mode), 6-axis wrenches, joints, TCP, wrist F/T,
  gripper, and the absolute actions. Pause / resume / stop / 0.25–4×
  speed from the panel; playback is paced by wall clock × speed and drops
  stale image frames rather than falling behind.

Playback logs on a separate rerun timeline (`t_episode`, one segment per
playback) so it never mixes with the live `t_host` data, and it is mutually
exclusive with sessions and offloads (both sides check under one shared
gate lock): you cannot start a session while replaying, and you cannot
replay or offload over each other.

The same verifier + player exists as a CLI for scripting / quick checks
without the panel:

```
python -m phantom.scripts.play_episode D:/phantom_episodes/<session>/ep_... 
python -m phantom.scripts.play_episode <ep_dir> --verify-only   # exit 1 if incomplete
python -m phantom.scripts.play_episode <ep_dir> --speed 2
```

## Running a session

1. Edit `configs/data_collect.yaml` (or a `.local.yaml` overlay) for this
   rig: sensor serials, RealSense serial, `storage.external_drive`.
2. `python -m phantom.scripts.collect` (add `--host 0.0.0.0` for a rig
   tablet — trusted LAN only, the buttons move a robot).
3. Open the printed panel URL. Fill the session wizard: task, operator,
   instruction text, **mode** (full/lite), target episode count. **Start
   session.**
4. Don the Echo exoskeleton; the arm glides to meet it (ENGAGE), then
   tracks directly (TRACK). Use the Echo's own start/stop switch or the
   panel's start/stop button to record episodes; success/fail/discard from
   the panel.
5. If the safeguard trips: relax the grasp, click **RESUME COLLECTION**.
6. **End session** — episodes are automatically offloaded to the external
   drive with verification; retry manually from the panel ("Offload now")
   if the drive was unplugged or full.

## Testing

`tests/test_data_collect_*.py` (config, teleop, gripper, safeguard, session,
panel, offload, playback) run entirely on mocks — no hardware, no rig — via
`phantom_test_utils.make_small_hw()` and stand-in rings/drivers. They cover
the safeguard latch/resume/runtime-edit semantics, the streamer's
engage/track/hold/resume state machine (including the HOLD-cannot-be-stomped
guarantee), the gripper pilot's device-rate pull and stale-value handling on
release, and the offload verification/same-device guard.

This app was built by reading the vendored `main11.py` (the known-good
control feel) and the reference GitHub stack (to diagnose exactly why its
teleop felt bad), then hardened by an adversarial multi-reviewer code review
that confirmed 25 concrete defects (6 critical, all in the safeguard/resume
race paths) — every one fixed with a regression test before this landed.
