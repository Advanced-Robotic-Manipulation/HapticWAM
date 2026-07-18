# Data collection SOP

Target (pipeline.md §7): **150 teleop episodes × 5 tasks = 750**, ~20–40 s
each, all tasks co-trained; **+20–30 deliberate-failure episodes per fragile
task**; **2–4 h of contact play** for the tactile SSL pretrain; later
**2 DAgger rounds × ~50 student rollouts/task**.

Task suite: fragile grasp · textured/slippery grasp-and-place · insertion ·
surface wipe with force regulation · in-hand regrasp.

## Session start checklist

1. `configs/hardware.yaml`: `mode.drivers: real`, `meta.bench_verified: true`.
2. Sensors on separate USB root hubs; RealSense mounted and aimed; scene lit.
3. Robot in Remote Control; workspace clear.
4. **The operator path (recommended): `python -m phantom.scripts.panel`** —
   starts ONLY the web panel at http://127.0.0.1:8788 (no robot yet). The
   operator opens it in a browser: session wizard (rig summary + pre-flight
   checklist, task picker, operator, teleop device, episode target) →
   "Запустить сессию" brings up the whole rig; recording runs from the
   browser (progress toward the target, episode log with outcomes, embedded
   rerun view of every stream, safety banners with recovery instructions);
   "Завершить сессию" returns to the wizard with a summary — ready for the
   next session without restarting anything. `--host 0.0.0.0` for a rig
   tablet (trusted LAN only — the buttons move the robot). Rerun needs
   `pip install 'phantom[viz]'`; the panel works without it.
   CLI alternative: `record_episodes --task <task> --teleop echo --panel`
   (same live view, session config from the command line).

## Echo exoskeleton teleop (`--teleop echo` — the lab's leader device)

- Joint-space leader: the arm follows `base_pose + exo_offset / divisor`
  (sensitivity from the device's 3-level switch); config in
  `configs/hardware.yaml` `teleop.echo`.
- Recording writes TWO action streams: canonical Δ-EE `actions (T,7)` derived
  from measured TCP poses, plus raw joint targets `actions_qtarget (T,6)`.
- The device's own start/stop flag toggles episodes exactly like SPACE
  (both work; keyboard hotkeys stay active for tags/abort/quit).
- Gripper is CONTINUOUS 0..1 from the exo gripper ticks — calibrate
  `gripper_open_tick`/`gripper_closed_tick` on the rig before the first
  session (read raw ticks at full open / full close).
- Motion is SMOOTH by construction (`teleop/filters.py` + `teleop/streamer.py`):
  one-euro filter on the leader signal at device rate (tremor + encoder
  quantization removed), then a high-rate servo streamer (`control_rate_hz`,
  default 125 Hz — no 10 Hz step-and-hold) tracking through a bounded
  velocity+acceleration profile with exact no-overshoot landing. The engage
  jump when donning the exoskeleton becomes an S-curve from rest.
- Safety in the joint-space branch (now enforced at control rate in the
  streamer): velocity bound from `arm.limits.joint_speed_rad_s`, acceleration
  bound `teleop.echo.joint_accel_rad_s2`, reactive workspace hold (arm
  freezes at the last in-bounds pose if the TCP leaves `safety.workspace_m`,
  resumes when steered back), parks on protective stop.
- Feel tuning: `filter_min_cutoff` down = calmer at rest; `filter_beta` up =
  less lag in fast motion; `joint_accel_rad_s2` up = snappier tracking.

## Safety circuit during collection (automatic — know what it does)

The deployment `SafetyMonitor` runs every teleop tick. On a wrist-wrench or
fingertip force/indentation limit (or a stale tactile stream) the episode is
**saved as a failure** (`notes: safety_stop:<kind>`), the arm freezes, and
teleop auto-resumes once forces drop below 0.8× the limits. On a protective
stop the episode is discarded; clear it on the pendant and teleop resumes by
itself. Commanded gripper force is hard-clamped at the driver boundary to
`gripper.cmd_force_limit_N` (30 N DM-Tac pad ceiling) — asking for more in
code or config cannot crush the pads.

## Hotkeys (keyboard is always active, also alongside the SpaceMouse/Echo)

| Key | Action |
|---|---|
| SPACE | start episode / stop-and-save |
| g | mark SUCCESS and stop |
| b | mark FAILURE and stop |
| f | tag deliberate-failure (cycles subtype: over_squeeze → induced_slip → near_limit) |
| ESC | abort episode (discard entirely) |
| 0 | re-zero wrist F/T (do this with the tool free of contact) |
| ~ | quit after the current episode |

Motion (keyboard teleop): `w/s a/d q/e` translate, `i/k j/l z/x` rotate,
`o/p` gripper open/close.

## Per-episode discipline

- Reset the scene the same way every time (marked object poses).
- F/T is re-zeroed automatically at episode start (`wrist_ft.bias_on_episode_start`).
- 20–40 s per episode; end with `g`/`b` (explicit outcome), not SPACE, whenever
  the outcome is defined.
- Vary grasp points, approach directions, object poses across episodes — the
  bottleneck is task-level action diversity, not frame count.

## Deliberate-failure episodes (fragile tasks)

HID-S's auto-threshold and the slip ontology are only as informative as the
boundary examples: on **sacrificial objects**, record controlled over-squeezes
and induced slips. Press `f` during the episode (tags
`deliberate_failure` + subtype), end with `b`. These episodes are excluded
from the τ_obj calibration automatically (it uses successes only) but teach
the event ontology what the boundary looks like.

## Contact play (SSL pretrain data)

No task, no teleop skill: random pokes, grasps, slides over many objects and
textures with the sensors mounted. Record as normal episodes under
`--task contact_play`. An afternoon (~2–4 h total sensor-on time) suffices —
at the native rate one 20 s episode is thousands of field frames.

## After every session (10 min)

```bash
python -m phantom.scripts.postprocess_episodes --data data/episodes/<date>
python -m phantom.scripts.dump_norm_stats --data data/episodes/<date>
```

**QC — replay 1 in 20 episodes:** open the derived streams
(`derived_<sensor>_events`) and sanity-check the event timeline against what
you remember doing (onset when you touched, slip when it slipped). A silent
sensor (all `none`) or a saturated one (all `slip`) means a threshold or a
mount problem — fix before the next session.

Disk check: each episode dir should be ~tens of MB at the default plan. The
recorder aborts loudly if the disk cannot keep up.

## Backup / transfer to the training cluster

Episodes are plain directories of zarr chunks — rsync-friendly:

```bash
rsync -a --info=progress2 data/episodes/ user@cluster:/data/phantom/episodes/
```

Verify after transfer: `python -m phantom.scripts.postprocess_episodes --data ...`
on the cluster side runs the same derived pass and fails on truncated chunks.

## Provenance rules

- Every episode stores a byte-exact `hardware.yaml` copy + config hash — never
  edit those.
- If the hardware config changes mid-collection (e.g. rate drop), just keep
  recording: loaders are timestamp-driven and warn on shape-relevant drift.
- Keep `norm_stats.json` versioned with the dataset it was computed from.
