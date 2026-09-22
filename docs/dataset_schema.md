# HapticWAM episode schema

The authoritative description of one recorded episode: what is on disk, what each
number means, how the streams line up in time, and how to get one episode out of the
published datasets without downloading 100 GB.

Everything here is derived from the code in this repository — `phantom/data/schema.py`
(stream names, `EpisodeMeta`, trainability rules), `phantom/data/episode_store.py`
(zarr layout, JPEG encoding, chunking), `phantom/recording/{recorder,workers}.py`
(which streams exist and at what rate), `phantom/data/derived.py` (channel semantics
and the derived contact channels), `phantom/config/hardware.py` + `configs/hardware*.yaml`
(shapes, rates, dtypes, unit scales) and `phantom/data/windows.py` (how training windows
are drawn) — and cross-checked against a real published episode
(`waffles/ep_waffles_1785592739_002`, `config_hash b2504b6d5ff0c29d`).

## Contents

1. [One episode on disk](#1-one-episode-on-disk)
2. [Time base and stream alignment](#2-time-base-and-stream-alignment)
3. [Episode id naming](#3-episode-id-naming)
4. [Streams](#4-streams)
   - [4.1 Tactile (per sensor)](#41-tactile-per-sensor)
   - [4.2 The 8-channel tactile field stack](#42-the-8-channel-tactile-field-stack)
   - [4.3 Arm](#43-arm)
   - [4.4 Gripper](#44-gripper)
   - [4.5 Camera](#45-camera)
   - [4.6 Actions](#46-actions)
   - [4.7 Derived streams (offline, optional)](#47-derived-streams-offline-optional)
   - [4.8 Simulation-only streams](#48-simulation-only-streams)
5. [`meta.json`](#5-metajson)
6. [Trainability semantics](#6-trainability-semantics)
7. [How training windows are drawn](#7-how-training-windows-are-drawn)
8. [How to read one episode](#8-how-to-read-one-episode)
9. [How the datasets are packaged](#9-how-the-datasets-are-packaged)
10. [Sample episodes on the hub](#10-sample-episodes-on-the-hub)
11. [Fetching a single episode](#11-fetching-a-single-episode)

---

## 1. One episode on disk

One directory per episode; the directory name is the episode id and always starts with
`ep_`:

```
ep_waffles_1785592739_002/
├── meta.json                      # EpisodeMeta (phantom/data/schema.py)
├── actions.zarr/                  # one zarr group per stream …
│   ├── data                       #   (T, …) the samples
│   └── ts                         #   (T,)   float64 master-clock seconds
├── actions_abs.zarr/
├── arm_q.zarr/  arm_qd.zarr/  arm_tcp_pose.zarr/  arm_tcp_speed.zarr/  arm_ft.zarr/
├── gripper.zarr/
├── camera_scene_color.zarr/
├── tactile_left_{fields_ds,keyframes,infer_img,wrench,area}.zarr/
└── tactile_right_{fields_ds,keyframes,infer_img,wrench,area}.zarr/
```

* zarr v2 groups (DirectoryStore), one group per stream, exactly two arrays per group:
  `data` with shape `(T, …)` and `ts` with shape `(T,)`.
* Compression: `Blosc(cname="zstd", clevel=3, shuffle=BITSHUFFLE)` for every dense array
  (`phantom/data/episode_store.py`).
* Chunking: `rows_per_chunk = min(recording.zarr_chunk_frames, recording.zarr_chunk_mb / row_bytes)`
  — 120 rows or 8 MB, whichever is smaller. zarr rewrites a partially filled chunk on
  every append, so the byte bound is what keeps the recorder's drain fast.
* Every stream is written independently and has **its own `T` and its own timestamps**.
  Streams are never resampled onto a common grid at record time.
* Streams are created lazily on first append: a stream that produced no samples (a
  disabled camera, a pad-free deploy) has **no directory at all**. Check for existence,
  do not assume.

## 2. Time base and stream alignment

`ts` is `t_master` in **seconds, float64**, on the `MasterClock` domain
(`phantom/timesync/clock.py`):

* the master clock is the **UR controller's RTDE timestamp**;
* every worker timestamps its samples with `time.perf_counter()` (`t_host`, one
  machine-wide monotonic clock shared by the parent process and all worker processes);
* the recorder converts on drain: `t_master = t_host + offset`, where `offset` is the
  median of `t_rtde - t_host` over the lowest-latency decile of 200 calibration pairs.
  The calibration is stored per episode in `meta.clock_calibration`
  (`{"offset", "n_samples", "spread_s"}`; the reference episode has
  `spread_s = 6.3e-4`, i.e. sub-millisecond).
* `t_master` is therefore **not** a wall-clock epoch. It is monotonic within an episode
  and comparable **across streams of the same episode** — which is the only thing any
  consumer needs. Use `meta` / the episode id for absolute time.

Alignment recipe (what `phantom/data/windows.py` does, and what you should do):

| Purpose | Rule | Helper |
|---|---|---|
| observation at `t` | nearest sample by `|ts - t|` | `_EpisodeCache.nearest_idx` |
| future target at `t` | first sample with `ts >= t` (never resolve backwards) | `_EpisodeCache.future_idx` |
| wrist F/T window | linear interpolation onto a uniform grid `(t0 - window_s, t0]` | `np.interp` |
| usable anchor range | `[max(first ts) + past, min(last ts) - future]` over camera, `arm_ft`, `actions` and every `fields_ds` | `WindowSampler.valid_range` |

## 3. Episode id naming

Two generators, and the presence of a policy segment is the difference:

| Source | Pattern | Example |
|---|---|---|
| Teleoperation (`phantom/data_collect/session.py`, `phantom/scripts/record_episodes.py`) | `ep_<task>_<epoch>_<idx>` | `ep_waffles_1785592739_002` |
| Deployment / policy rollouts (`phantom/deploy/runtime.py`) | `ep_<policy>_<task>_<epoch>_<idx>` | `ep_student_Carton_1789493413_000` |
| Sim export (`tools/sim/export_expert_episode.py`) | `ep_sim_<task>_<trial>__<source-episode>` | `ep_sim_egg_egg2_ep0001__ep_student_egg_1788970209_002` |
| Synthetic test fixtures (`phantom/data/synthetic.py`) | `ep_synth_<seed>_<n>` | `ep_synth_000_0001` |

`<epoch>` is `int(time.time())` at episode start, `<idx>` a zero-padded per-session
sequence number. `<policy>` is the deploy mode (`teacher`, `student`, …) and is mirrored
in `meta.policy`. A few early deploy takes have no trailing index at all
(`ep_teacher_waffles_1786654841`). **`meta.json` is authoritative for task and policy** —
directory names carry the label typed at the rig.

## 4. Streams

Shapes and rates below are the values the published corpus was recorded with
(`configs/hardware.nuc.yaml`, `config_hash b2504b6d5ff0c29d`); the schema itself fixes
only the *ranks*. "nominal" is the configured rate, "measured" the rate actually
achieved in the reference episode (`(T-1) / (ts[-1] - ts[0])`).

### 4.1 Tactile (per sensor)

Two sensors, `left` and `right` (Daimon DM-Tac W2L fingertip pads, one per Robotiq 2F-85
finger). Stream name = `tactile_<sensor>_<kind>` (`phantom.data.schema.tactile_stream`).

| Stream | Shape | dtype | Nominal | Measured | Units / meaning |
|---|---|---|---|---|---|
| `tactile_<s>_fields_ds` | `(T, 72, 96, 8)` | float16 | 8 Hz | ~5.7 Hz | Mean-pooled tactile field stack, the full-rate tactile signal. Native pad field is 288×384; `recording.field_ds` = 72×96 is a 4×4 mean pool. Channels: §4.2. |
| `tactile_<s>_keyframes` | `(T, 144, 192, 8)` | float16 | 3 Hz | ~2.5 Hz | Same 8 channels at higher spatial resolution (2×2 mean pool of the native field), time-decimated. Read once per training window (teacher-only input `fields`). |
| `tactile_<s>_infer_img` | `(T, 288, 384)` | uint8 | 8 Hz | ~5.7 Hz | The SDK's gel image (`getInferImg`), grayscale, sensing-area crop. Teacher-only input `gel`; expanded to 3 channels at load. |
| `tactile_<s>_wrench` | `(T, 6)` | float32 | 8 Hz | ~5.7 Hz | Resultant pad wrench `[Fx, Fy, Fz, Mx, My, Mz]`, **SI**: forces in N, torques in N·m. The SDK reports torques in 1e-2 N·m; `phantom/drivers/real/dmtac.py` applies `force_unit_to_N=1.0` / `torque_unit_to_Nm=0.01` at the driver boundary. |
| `tactile_<s>_area` | `(T,)` | float32 | 8 Hz | ~5.7 Hz | Contact area in mm² (`getContactArea`). |
| `tactile_<s>_raw_img` | `(T, 480, 640)` | uint8 | — | — | Raw grayscale sensor image. Only written when `recording.archive_raw_img: true`; **off for the published corpus**, so this stream is absent. |

Notes.

* The declared 8 Hz is a **cap** passed to the SDK, not a guarantee: the SDK's host-side
  reconstruction is the bottleneck and the loop free-runs at ~5.5–6.5 Hz per sensor with
  both sensors plus camera and recorder running. Use `ts`, never an assumed rate.
* `keyframes` and `infer_img` are wall-clock gated (decimate by time, not by frame
  index), so their rates hold even as the source rate drifts.
* A pad-free deploy (`run_deploy --pad-free`) allocates no tactile rings at all: those
  episodes have **no `tactile_*` streams** and carry the `padfree:on` tag.
* Left and right have independent timestamps; do not assume row `i` of the two sensors is
  the same instant.

### 4.2 The 8-channel tactile field stack

`fields_ds` and `keyframes` share one canonical channel order
(`TactileFrame.field_stack`, `phantom/drivers/base.py`); widths come from
`tactile.field_channels` and are validated to sum to 8:

| Index | Group (config key) | Channel | SDK source | Units |
|---|---|---|---|---|
| 0, 1 | `deformation2d` (2) | in-plane displacement x, y | `getDeformation2D` | SDK deformation units |
| 2 | `depth` (1) | normal indentation | `getDepth` | SDK depth units (**uncalibrated**; `derived.tau_contact_depth = 0.05` is the contact threshold on \|depth\|) |
| 3, 4 | `shear` (2) | shear x, y | `getShear` | SDK shear units |
| 5, 6, 7 | `dist_force` (3) | distributed force fx, fy, fz | `getDistributeForce` | **UNCALIBRATED** — `tactile.dist_force_unit_to_N = 0.0` is the sentinel for "units unverified", so the whole corpus runs depth-based contact detection, not fz-based |

`phantom/data/derived.channel_slices()` returns these slices from the config; use it
rather than hardcoding indices. Everything the model is supervised on
(contact mask, CoP, slip, events, gate) is **derived from these channels on the fly**
with the thresholds in `configs/hardware.yaml → derived:` — see §4.7.

### 4.3 Arm

UR3 over RTDE. One process polls `RTDEReceiveInterface` and pushes into the `arm` ring;
the recorder splits the ring fields into five streams.

| Stream | Shape | dtype | Nominal | Measured | Units / meaning |
|---|---|---|---|---|---|
| `arm_q` | `(T, 6)` | float64 | 125 Hz | 125.0 Hz | Joint positions, rad (`getActualQ`) |
| `arm_qd` | `(T, 6)` | float64 | 125 Hz | 125.0 Hz | Joint velocities, rad/s (`getActualQd`) |
| `arm_tcp_pose` | `(T, 6)` | float64 | 125 Hz | 125.0 Hz | TCP pose `[x, y, z, rx, ry, rz]` in the robot base frame: metres + **axis-angle (rotation vector)**, radians (`getActualTCPPose`) |
| `arm_tcp_speed` | `(T, 6)` | float64 | 125 Hz | 125.0 Hz | TCP twist, m/s and rad/s (`getActualTCPSpeed`) |
| `arm_ft` | `(T, 6)` | float64 | 125 Hz | 125.0 Hz | Wrist wrench `[Fx, Fy, Fz, Mx, My, Mz]`, N and N·m (`getActualTCPForce`) |

Caveats that matter for modelling:

* The rig is a **CB3-generation UR3**. `getActualTCPForce` there is a *current-based
  estimate*, not a real F/T sensor: it carries a large pose-dependent static bias
  (~18 N / ~5 N·m at rest, up to 36–45 N in some poses). The safety layer treats it as a
  deviation from a rolling baseline, never as an absolute. Bias it per episode before use.
* Rotation vectors are sign-ambiguous (`r` and `r·(1 − 2π/‖r‖)` are the same rotation) and
  the controller does flip between them. Use `phantom.data.derived.rotvec_nearest` /
  `pose_delta` for any pose difference.
* The `arm` ring also carries `t_rtde` and `protective_stop`, but `_stream_map()` does not
  map them to episode streams — they are **not** in the recorded episodes.

### 4.4 Gripper

| Stream | Shape | dtype | Nominal | Measured | Units / meaning |
|---|---|---|---|---|---|
| `gripper` | `(T, 2)` | float32 | 100 Hz | ~94 Hz | `[position, obj]` |

* `position` — normalised closure, `0.0` fully open … `1.0` fully closed. Commanded
  closure is clamped at the driver boundary to `gripper.max_close_cmd = 0.9` on this rig
  (pad-into-pad contact above that), so recorded positions never approach 1.0.
* `obj` — the **Robotiq `gOBJ` object-detection status**, raw `0..3` stored as a float:
  `0` = fingers moving, `1` = stopped by contact while opening, `2` = stopped by contact
  while closing (**holding an object**), `3` = at the requested position. It is *not* a
  motor current: no current/force channel is obtainable on this rig (`GET CUR` answers
  `?`, RTDE `tool_output_current` stays at 0.0 mA through a genuine motor stall), and
  `gOBJ` is the one contact-bearing gripper channel that responds.
* Gripper state is recorded from inside `GripperPilot`'s own loop
  (`phantom/data_collect/gripper.py`), not a separate poller, because a second thread
  raced the single Robotiq TCP socket.

### 4.5 Camera

| Stream | Shape | dtype | Nominal | Measured | Units / meaning |
|---|---|---|---|---|---|
| `camera_scene_color` | `(T, 480, 640, 3)` | uint8 (JPEG on disk) | 15 Hz | ~15.0 Hz | RealSense D435 scene RGB |
| `camera_wrist_color` | `(T, 480, 640, 3)` | uint8 | — | — | Wrist camera — **disabled on this rig**, stream absent |

**On-disk encoding.** With `recording.scene_jpeg_quality > 0` (92 for this corpus) each
frame is JPEG-encoded and the `data` array is a *1-D zarr object array* of variable-length
byte strings (`numcodecs.VLenBytes`), not a dense `(T, H, W, 3)` array. The group is
self-describing through its attributes:

```json
{"encoding": "jpeg", "frame_shape": [480, 640, 3], "frame_dtype": "uint8", "jpeg_quality": 92}
```

`EpisodeReader.data("camera_scene_color")` detects `encoding == "jpeg"` and returns a
`JpegFrameArray` that decodes lazily and presents `shape == (T, 480, 640, 3)`, so every
consumer sees raw RGB frames. Reading the group directly gives you the JPEG bytes — decode
with `phantom.data.jpeg_codec.decode_jpeg` (cv2 or Pillow, RGB order). Raw uint8 frames
are still supported (`scene_jpeg_quality: 0`); older episodes without the `encoding`
attribute are dense arrays.

### 4.6 Actions

| Stream | Shape | dtype | Nominal | Measured | Units / meaning |
|---|---|---|---|---|---|
| `actions` | `(T, 7)` | float32 | 10 Hz | ~9.9 Hz | **Canonical action.** `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`: Δ-EE pose step between consecutive measured TCP poses (metres, rotation-vector radians, `derived.pose_delta` with the rotvec continuity guard) plus the commanded gripper closure `0..1`. This is the only action stream `WindowSampler` reads. |
| `actions_abs` | `(T, 7)` | float32 | 10 Hz | ~9.9 Hz | Absolute command: `[q_target(6) (rad), gripper(1)]` — what the Echo exoskeleton leader actually commanded. Present in teleop and rig episodes. |
| `actions_qtarget` | `(T, dof)` | float32 | 10 Hz | — | Raw joint-space teleop targets; written by some teleop loops only. |
| `actions_plan` | `(T, 7)` | float32 | 10 Hz | — | **Deploy only.** The policy's raw chunk proposal at governor-warped times, before clamp / rate limiting. Created by `tools/rederive_rollout_actions.py`, which moves the executor's proposal aside and rebuilds `actions` as the *measured* Δ-EE on the action grid. Its presence (or the `actions_rederived` tag) is how you know a rollout's `actions` are measured and not proposed. |

Action-grid conventions:

* `control.action_dim = 7` and `control.action_rate_hz = 10.0` are schema constants; a
  policy chunk is `chunk_horizon = 16` actions, i.e. 1.6 s.
* Δ-EE is a per-tick delta at 10 Hz, composed by cumulative sum in rotation-vector space
  (`phantom/deploy/executor.py::_pose_at`). The reference episode's per-axis extremes are
  ±0.03 m and ±0.07 rad per tick.
* In teleop the Δ is **one tick late by construction** (it is computed from the two most
  recent measured poses). Actions and states carry independent timestamps and windows
  sample by nearest `ts`, so this is a latency property, not a misalignment bug.
* The gripper channel of `actions` is the command *that actually went out* — the aperture
  latch's floor is applied before recording, so it never shows an "open" the executor
  refused to send.

### 4.7 Derived streams (offline, optional)

`phantom/recording/postprocess.py` can write four more streams per sensor from
`fields_ds`: `tactile_<s>_mask_frac` `(T,)`, `tactile_<s>_cop` `(T, 2)`,
`tactile_<s>_slip` `(T,)`, `tactile_<s>_events` `(T,)` int64.

They are **not present in the published episodes** and are not needed: `WindowSampler`
derives the same quantities on the fly from `fields_ds`, so a change to a threshold in
`configs/hardware.yaml → derived:` does not invalidate any recorded data. Definitions
(`phantom/data/derived.py`):

| Quantity | Definition | Config |
|---|---|---|
| contact mask | per-cell `abs(depth) > tau_contact_depth` (or `abs(fz)·unit > tau_contact_fz` when the distributed force is calibrated — it is not, here) | `contact_source: depth`, `tau_contact_depth: 0.05` |
| `mask_frac` | fraction of pad cells in contact. Frame-level contact is `mask_frac > tau_contact_area`, **not** a single hot cell | `tau_contact_area: 0.025` |
| `cop` | force/depth-weighted centroid of the contact cells in normalised `[-1, 1]²`, `(x, y) = (columns, rows)`; **NaN** when fewer than `cop_min_contact_cells` cells are in contact | `cop_min_contact_cells: 10` |
| `slip` | background-subtracted mean tangential flow of the `deformation2d` channels per second, blended with the friction-cone ratio when forces are calibrated | `tau_slip: 0.5`, `slip_flow_weight: 0.5`, `friction_cone_mu: 0.8` |
| `events` | `{none, onset, hold, slip, release}` over debounced contact runs | `event_min_hold_ticks: 2` |
| ACC gate label | "contact within `(t, t + event_lookahead_s]`" | `event_lookahead_s: 0.5` |

### 4.8 Simulation-only streams

Isaac Sim exports (`tools/sim/export_expert_episode.py`) reproduce the same layout so the
sampler reads them like a teleop demo, with two differences:

* `contact_gt` `(T, 3)` float32 — `[any pad in contact, left N, right N]`, the *physical*
  pad-object contact from the simulator.
* The pads are not simulated: the `tactile_*` streams are **idle rows copied from a real
  episode's untouched start** and tiled at the recorded rates. Such episodes carry the
  `pads_masked` tag, and `WindowSampler` then takes contact events and the gate from
  `contact_gt` and zeroes the tactile reconstruction losses (`contact_weight = 0`).

## 5. `meta.json`

`EpisodeMeta` (`phantom/data/schema.py`), written with `json.dumps(..., indent=1)`. The
loader is tolerant: unknown keys are ignored, missing keys take the default, so old files
keep working.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `task` | str | *required* | Task key: `Carton`, `egg`, `waffles`, `whiteboard`, or `<task>_fail` for a deliberate failure demonstration. Also the language-conditioning fallback. |
| `text` | str | `""` | Optional natural-language instruction. Empty falls back to `task` (mirrors `embed_task_texts.collect_texts`), so an episode is never silently trained unconditioned. |
| `operator` | str | `""` | Teleoperator initial. |
| `tags` | list[str] | `[]` | See the tag table below. |
| `policy` | str | `""` | `teleop` for human demos; the policy name (`teacher`, `student`, …) for rollouts. `is_policy_rollout` = non-empty and `!= "teleop"`. |
| `dagger_round` | int | `-1` | DAgger round for on-policy rollouts; `-1` for teleop. |
| `success` | bool \| null | `null` | Operator verdict. `null` = never judged. For the v4 `*_fail` demos `true` means "the episode successfully captured the intended failure". |
| `damage` | bool | `false` | Collateral damage (object broken, table/gripper hit), independent of `success`; also mirrored as the `damaged` tag. |
| `notes` | str | `""` | Free-text operator note. |
| `driver_modes` | dict | `{}` | `{"drivers": "real"\|"mock", <device>: <mode>}` — which devices were real. Filled by `EpisodeRecorder.start`. |
| `clock_calibration` | dict | `{}` | `{"offset", "n_samples", "spread_s"}` of the `MasterClock` fit used to convert `t_host → t_master` (§2). |
| `config_hash` | str | `""` | First 16 hex chars of sha256 over the canonical YAML dump of the whole `HardwareConfig`. Identifies the rig configuration the episode was recorded under; `WindowSampler` warns on drift. Deploy pre-sets the **base** (as-loaded) hash so per-task safety overrides do not make every rollout look like another rig. |
| `hardware_shapes` | dict | `{}` | The shape-relevant subset of the config (`field`, `field_ch`, `keyframe_ds`, `n_fingers`, `wrench_dim`, `wrist_ft_dim`, `wrist_window_len`, `ur_state_dim`, `contact_state_dim`, `action_dim`, `chunk_horizon`, `cpk_shape`). Checkpoint compatibility is asserted on these. |
| `deploy_overrides` | dict | `{}` | Deploy-only run-time hardware overrides applied on top of `config_hash`: `z_floor_m`, `hitbox_m`, `tcp_speed_m_s`. Keeps the safety envelope an episode actually ran under auditable. |
| `status` | str | `"recording"` | `recording` (crashed or still in flight) → `finalized` → or `aborted`. Only `finalized` episodes are listed by `list_episodes()` or trained on. Data is never deleted on abort. |
| `weight` | float | `1.0` | Per-episode multiplier on the **action** loss. `1.0` = ordinary demo; intake writes `>1` to oversample a small on-policy rollout pool (`clip(1/(p_task+0.2), 1, 3)`). A failure demo is forced to `0` regardless. |

Tags in use:

| Tag | Effect |
|---|---|
| `full` | Complete teleop take (recorder default). No training effect. |
| `batch_<YYYYMMDD>` | Intake batch provenance. No training effect. |
| `deliberate_failure`, `undergrasp` | Marks a staged failure demonstration (`is_deliberate_failure_demo`). |
| `damaged` | Collateral damage; training-inert. |
| `pads_masked` | Sim episode with idle (unmodelled) pads — contact comes from `contact_gt`. |
| `actions_rederived` | `actions` has been rebuilt as the measured Δ-EE by `tools/rederive_rollout_actions.py`. |
| `contaminated`, `unlabeled`, `redo`, `crushed`, **`padfree:on`** | `NON_TRAINING_TAGS` — the episode must not be indexed at all. `padfree:on` marks a pad-free deploy: **no `tactile_*` streams exist**, so it can never supervise a contact target. |

## 6. Trainability semantics

Three predicates in `phantom/data/schema.py` decide what an episode may supervise. They
are the gate `WindowSampler.build_index` applies, and any re-implementation should apply
the same ones.

`is_trainable_episode(meta)` — may this episode be indexed at all?

* `False` if any tag is in `NON_TRAINING_TAGS`;
* `False` if `status != "finalized"`;
* `False` for a **policy rollout with no verdict** (`policy` set and not `teleop`, and
  `success is None`) — an unjudged on-policy rollout is exactly the state where the model
  is already wrong;
* otherwise `True`. Deliberate failure demos stay trainable on purpose: they carry real
  supervision for the contact / event / slip heads.

`is_failure_demo(meta)` — must this episode's actions never be imitated?
Any of: `success is False`; the `deliberate_failure` tag; a task name ending in `_fail`.
`WindowSampler` sets `action_weight = 0` for these (the tactile, contact, event and gate
targets still train).

`is_deliberate_failure_demo(meta)` — the same minus the verdict rule: only the two
encodings that say the failure was *staged*. It exists because `success is False` cannot
distinguish "recorded to show a failure" from "the policy tried and missed"; the opt-in
`rollout_action_weight="judged_rollouts"` mode uses it to re-admit judged on-policy
rollouts whose actions have been re-derived.

`needs_rederive(ep_dir, meta)` — `True` for a policy rollout whose `actions` is still the
executor **proposal** (no `actions_plan.zarr`, no `actions_rederived` tag). Training on it
imitates commands the safety layer refused, at a cadence the sampler misreads as the 10 Hz
grid.

## 7. How training windows are drawn

`phantom/data/windows.py`, for reference when comparing against your own loader. A window
is anchored at `t0`:

* **conditioning** — `frames_pix` camera frames from `t0` forward on the backbone's pixel
  grid (bilinear-resized, scaled to `[-1, 1]`); the wrist F/T window `(t0 − 0.25 s, t0]`
  resampled to `wrist_window_len = 31` samples; the UR state at `t0` as
  `[q(6), qd(6), tcp_pose(6), tcp_speed(6), gripper(2)] = 26` values; the previous action
  chunk `(t0 − H/rate, t0]`.
* **targets** — future frames; the contact package on the latent grid
  `u_k = t0 + k·temporal_comp/fps` (Δ-field, Δfz, mask, CoP, slip, wrench, wrist, events,
  gate), all derived on the fly from `fields_ds`; and the action chunk `(t0, t0 + H/rate]`
  drawn with the strict-future rule.
* **teacher-only inputs** — `fields` (the `keyframes` stream at `t0`), `gel`
  (`infer_img` at `t0`), raw `contact_state` (per finger:
  `wrench(6) + area(1) + CoP(2) + slip(1) + mask_frac(1) = 11`), and `reactive`.
  The student drops these; the tactile *targets* stay.
* Pad wrench is optionally **baselined per episode** (median of the first 8 rows, the pads
  idle untouched at the start pose) on both the input and the target, because the idle
  offset drifts per session and is otherwise a session id the model can fit.
* `valid_range()` refuses an episode too short for one window.

## 8. How to read one episode

Needs `zarr>=2.16,<3`, `numcodecs`, `numpy` and (for scene frames) `Pillow` or `opencv`.
No part of this repository is required.

```python
"""Read one HapticWAM episode directory: streams, shapes, rates, one frame."""
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import zarr

EP = Path("ep_waffles_1785592739_002")          # <- an episode directory

meta = json.loads((EP / "meta.json").read_text())
print(f"{EP.name}: task={meta['task']} policy={meta['policy']} "
      f"success={meta['success']} status={meta['status']} tags={meta['tags']}")

for zp in sorted(p for p in EP.iterdir() if p.suffix == ".zarr"):
    g = zarr.open_group(str(zp), mode="r")
    ts = np.asarray(g["ts"][:], dtype=np.float64)          # master-clock seconds
    data, enc = g["data"], g.attrs.get("encoding")
    shape = ((len(ts), *g.attrs["frame_shape"]) if enc == "jpeg" else data.shape)
    dtype = (g.attrs.get("frame_dtype", "uint8") if enc == "jpeg" else data.dtype)
    hz = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else float("nan")
    print(f"  {zp.stem:<24} {str(shape):<20} {str(dtype):<8} {hz:6.2f} Hz"
          f"  {ts[0]:.3f}..{ts[-1]:.3f}s{'  [jpeg]' if enc == 'jpeg' else ''}")

# --- align streams: nearest sample to an anchor time t0 --------------------
def nearest(g, t):
    ts = np.asarray(g["ts"][:], dtype=np.float64)
    i = int(np.searchsorted(ts, t))
    if i <= 0:
        return 0
    if i >= len(ts):
        return len(ts) - 1
    return i if (ts[i] - t) < (t - ts[i - 1]) else i - 1

cam = zarr.open_group(str(EP / "camera_scene_color.zarr"), mode="r")
t0 = float(np.asarray(cam["ts"][:])[len(cam["ts"]) // 2])   # mid-episode anchor

# scene frame at t0 (JPEG-encoded: decode the bytes; raw episodes index directly)
if cam.attrs.get("encoding") == "jpeg":
    from PIL import Image
    frame = np.asarray(Image.open(BytesIO(bytes(cam["data"][nearest(cam, t0)]))).convert("RGB"))
else:
    frame = np.asarray(cam["data"][nearest(cam, t0)])
print("scene frame", frame.shape, frame.dtype)

# tactile field stack at t0: (72, 96, 8) = [disp_x, disp_y, depth, shear_x, shear_y, fx, fy, fz]
fld = zarr.open_group(str(EP / "tactile_left_fields_ds.zarr"), mode="r")
stack = np.asarray(fld["data"][nearest(fld, t0)], dtype=np.float32)
depth = np.abs(stack[..., 2])                     # channel 2 = normal indentation
print("fields", stack.shape, "mask_frac", float((depth > 0.05).mean()))

# proprioception + the canonical action at t0
for name in ("arm_q", "arm_tcp_pose", "arm_ft", "gripper", "actions"):
    g = zarr.open_group(str(EP / f"{name}.zarr"), mode="r")
    print(f"{name:<14}", np.asarray(g["data"][nearest(g, t0)]).round(4))
```

Inside this repository the same thing is three lines:

```python
from phantom.data.episode_store import EpisodeReader
r = EpisodeReader("ep_waffles_1785592739_002")
r.meta, r.streams(), r.ts("actions"), r.data("camera_scene_color")[0]  # JPEG decoded for you
```

## 9. How the datasets are packaged

Two of the five episode repositories are **loose** (one directory per episode, browsable
in the hub file viewer); three are **packed** into tar archives. Packing was forced by
scale, not preference: the loose teleop corpus is ~865,000 files, and
`snapshot_download` of a repo that size spends 1–3 hours just enumerating before a byte of
training data lands. Nothing is transformed by packing — a tar member is the byte-identical
episode file.

| Repository | Form | Size | Unit of packing |
|---|---|---|---|
| [`armteam/hapticwam-teleop-raw`](https://huggingface.co/datasets/armteam/hapticwam-teleop-raw) | **loose** | ~112 GB | `tasks/<task>/ep_*/…` (canonical view), `archive/`, `collect/` (as-recorded sessions), `manifests/` |
| [`armteam/hapticwam-rig-episodes`](https://huggingface.co/datasets/armteam/hapticwam-rig-episodes) | **loose** | 24 GB | `20260915_experiment{,_extra,_superseded}/ep_*/…` |
| [`armteam/hapticwam-teleop-dataset`](https://huggingface.co/datasets/armteam/hapticwam-teleop-dataset) | packed | 102.7 GB | one `.tar.zst` per task |
| [`armteam/hapticwam-sim-episodes`](https://huggingface.co/datasets/armteam/hapticwam-sim-episodes) | packed | 82.7 GB | one `.tar` per episode (and per campaign under `raw_trials/`) |
| [`armteam/hapticwam-rollouts`](https://huggingface.co/datasets/armteam/hapticwam-rollouts) | packed | ~50 GB | one `.tar`/`.tar.zst` per deploy day |

**The same episodes exist loose.** `hapticwam-teleop-raw` holds the teleop corpus one
directory per episode under `tasks/<task>/`, and `hapticwam-rig-episodes` holds the
closed-loop evaluation takes loose. If you want to *look* at an episode, use those, or the
`samples/` folders described in §10 — you never need a 15 GB shard to see the format.

### `hapticwam-teleop-dataset` (packed)

```
Carton.tar.zst  Carton_fail.tar.zst  egg.tar.zst  egg_fail.tar.zst
waffles.tar.zst waffles_fail.tar.zst whiteboard.tar.zst whiteboard_fail.tar.zst
batch_20260822.tar.zst   manifests.tar   manifests_v6.tar   norm_stats.json
index.jsonl              MANIFEST_OF_RECORD.md
samples/<task>/ep_*/…                       # §10
```

* A `<task>.tar.zst` is a plain tar, zstd-compressed, whose members are
  `<task>/ep_<task>_<epoch>_<idx>/…` — i.e. it unpacks straight into the canonical
  `tasks/` view. 180 episodes per success task, 10–20 per `*_fail` task.
* `batch_20260822.tar.zst` unpacks to the **raw session layout**
  `20260822_<time>_<label>/ep_*/…` (34 sessions, 325 episodes), normalised into the task
  view at provision time by `tools/intake_recovery.py`.
* `manifests_v6.tar` → `manifests/all.jsonl`, one row per episode over all 1,115
  (`manifests.tar` covers only the 790-episode v4 view). Rows carry `path`, `task`,
  `success`, `split`, `failure_demo`, `tactile_contact`, `duration_s`, `peak_force_N`,
  `max_contact_mm2`, `session`, `operator`, `text`. **The manifest is the episode → task
  (→ shard) map**; read `MANIFEST_OF_RECORD.md` first.
* `index.jsonl` here is a **file** index, one row per repository file:
  `{"path", "size", "sha256", "blob_id", "source"}` — `sha256` is the LFS hash (null for
  small non-LFS files) and `source` the path the file was copied from during the
  2026-09 repository restructure. Use it to verify a download, not to find an episode.

```bash
# fetch and extract ONE task shard (the smallest success task is ~14.5 GB)
hf download armteam/hapticwam-teleop-dataset waffles.tar.zst --repo-type dataset --local-dir .
mkdir -p tasks && tar -I zstd -xf waffles.tar.zst -C tasks     # -> tasks/waffles/ep_*/
# just one episode out of the shard, without writing the rest to disk:
tar -I zstd -xf waffles.tar.zst -C tasks waffles/ep_waffles_1785592739_002
```

### `hapticwam-sim-episodes` (packed, one tar per episode)

```
sim_expert_20260912/{index.jsonl, norm_stats_v6.json, raw_trials/*.tar, tasks/waffles/*.tar}
sim_expert_20260914/{index.jsonl, raw_trials/*.tar, tasks/{Carton,egg}/*.tar, aside_duplicates/*.tar}
samples/<unit>/…                            # §10
```

* Each `tasks/<task>/<unit>.tar` is **one episode** (~100–200 MB uncompressed tar,
  429 members), members rooted at the episode directory name.
* `raw_trials/*.tar` are whole campaigns / raw trial exports;
  `_canonical_assets.tar` is the shared scene asset bundle.
* Each set's `index.jsonl` is a genuine **unit → tar** index, one row per unit:
  `{"set", "unit", "tar", "bytes", "members", "sha256", "task", "success", "status"}`
  (the last three only for episode units). This is the map to use.

```bash
hf download armteam/hapticwam-sim-episodes sim_expert_20260914/index.jsonl \
    --repo-type dataset --local-dir .
# pick a unit, then fetch exactly its tar:
hf download armteam/hapticwam-sim-episodes \
    sim_expert_20260914/tasks/egg/ep_sim_egg_egg2_ep0001__ep_student_egg_1788970209_002.tar \
    --repo-type dataset --local-dir .
tar -xf sim_expert_20260914/tasks/egg/ep_sim_egg_egg2_*.tar -C .
```

### `hapticwam-rollouts` (packed, one tar per deploy day)

```
deploy_<YYYYMMDD>.tar.zst         # the archive the rig box packed that day
rollouts_0901_0904.tar.zst  rollouts_0908_0909.tar.zst
zarr_rollouts/deploy_<YYYYMMDD>.tar   # repack of the loose per-chunk zarr tree
rig_deploy/deploy_<YYYYMMDD>.tar      # per-take deploy packs  + rig_deploy/index.jsonl
rig_sessions/rig_<YYYYMMDD>.tar.gz    # whole-session archives incl. operator notes
manifests_r3.tar   index.jsonl   samples/<take>/…        # §10
```

* `zarr_rollouts/deploy_<day>.tar` and `rig_deploy/deploy_<day>.tar` are **uncompressed**
  tars with members rooted at `deploy_<YYYYMMDD>/ep_<policy>_<task>_<epoch>_<idx>/…`.
* `rig_deploy/index.jsonl` is a **day → tar** index:
  `{"day", "tar", "bytes", "takes", "members", "sha256"}`. The root `index.jsonl` is the
  same per-file index as in the teleop repo.
* `manifests_r3.tar` → `all_r3.jsonl`, the DAgger round-3 manifest (1,238 rows; only the
  `*_rollout` rows resolve inside this repo).

```bash
hf download armteam/hapticwam-rollouts rig_deploy/index.jsonl --repo-type dataset --local-dir .
hf download armteam/hapticwam-rollouts zarr_rollouts/deploy_20260813.tar \
    --repo-type dataset --local-dir .            # 79 MB, 1 take — the cheapest whole unit
tar -xf zarr_rollouts/deploy_20260813.tar        # -> deploy_20260813/ep_*/
```

### The two loose repositories

```bash
# one teleop episode, loose (no tar, ~96 MB)
hf download armteam/hapticwam-teleop-raw --repo-type dataset --local-dir . \
    --include "tasks/waffles/ep_waffles_1785592739_002/*"
# one closed-loop rig take, loose
hf download armteam/hapticwam-rig-episodes --repo-type dataset --local-dir . \
    --include "20260915_experiment/ep_student_Carton_1789493413_000/*"
```

## 10. Sample episodes on the hub

Each packed repository carries a `samples/` folder holding complete, **loose, verbatim**
episodes — byte-for-byte the members of the shard they were taken from — so the format can
be browsed and read without downloading a shard. `samples/README.md` in each repository
names the exact source archive of every sample.

| Repository | Sample | Size | Source archive |
|---|---|---|---|
| `hapticwam-teleop-dataset` | `samples/waffles/ep_waffles_1785592739_002/` | 96 MB | `waffles.tar.zst` |
| | `samples/Carton/ep_Carton_1785597992_007/` | 130 MB | `Carton.tar.zst` |
| | `samples/egg/ep_egg_1785776148_006/` | 117 MB | `egg.tar.zst` |
| | `samples/whiteboard/ep_whiteboard_1785783007_008/` | 95 MB | `whiteboard.tar.zst` |
| `hapticwam-sim-episodes` | `samples/egg/ep_sim_egg_campaign_egg3_ep0049__ep_teacher_egg_1788969327_004/` | 59 MB | `sim_expert_20260914/tasks/egg/<unit>.tar` |
| `hapticwam-rollouts` | `samples/deploy_20260909/ep_student_Carton_1788973160_000/` | 55 MB | `zarr_rollouts/deploy_20260909.tar` |
| | `samples/deploy_20260813/ep_teacher_waffles_1786654841/` | 79 MB | `zarr_rollouts/deploy_20260813.tar` |

Each teleop sample is the first episode of its shard; the sim sample keeps the episode
directory but drops the tar's two leading `tasks/<task>/` components; the rollout samples
keep the tar's own `deploy_<day>/<take>/` layout. The 2026-09-09 student take is the
complete rollout example; the 2026-08-13 one is the whole content of that single-take
pack, kept as it is — an early, unjudged DAgger round-0 teacher take whose `gripper`
group was created but never written.

## 11. Fetching a single episode

`tools/hub/fetch_episode.py` does the resolution, the download and the extraction:

```bash
python tools/hub/fetch_episode.py --dataset teleop --episode first --out /tmp/ep
python tools/hub/fetch_episode.py --dataset teleop --episode ep_waffles_1785592739_002
python tools/hub/fetch_episode.py --dataset sim    --episode first
python tools/hub/fetch_episode.py --dataset rig    --episode ep_student_Carton_1789493413_000
python tools/hub/fetch_episode.py --dataset rollouts --episode deploy_20260813
```

It resolves the episode to a single archive (via each repository's `index.jsonl` /
manifest / the `<task>` shard rule) or to its loose path, downloads **only that one file**,
extracts just that episode, and prints the stream inventory of what it got. `--samples`
takes the published sample instead — a few tens of MB rather than a shard. See
`python tools/hub/fetch_episode.py --help`.

For bulk provisioning of a whole training corpus use `tools/provision_*.sh`, which pull
and unpack every shard; this script is deliberately the single-episode path.
