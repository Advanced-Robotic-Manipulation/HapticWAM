# DM-Tac W2L / `dmrobotics` SDK — verified facts

Distilled from the vendor package (`SDK_Publish_1.2.10` source + the official
manuals v2.0, 2026-04/05) — **not** from marketing pages. This is the ground
truth `phantom/drivers/real/dmtac.py` is written against. Items marked BENCH
still need one measurement on the physical unit (see hardware_bench_day1.md).

## Sensor (W2L)

| Property | Value |
|---|---|
| Sensing area | 36 × 27 mm |
| Internal camera | 640 × 480, **grayscale** out of the SDK |
| Perceptual grid | 384 × 288 → numpy arrays are **(H=288, W=384)** |
| Max rate | 120 Hz (fields are computed **host-side**; vendor table: i7 CPU ~30 Hz, RTX 3050 ~90 Hz, RTX 4060 ~120 Hz with TensorRT) |
| USB | USB 2.0, ~4.5 MB/s per sensor — up to 5 sensors per port |
| **Force ceiling** | **30 N total** on the pad — the Robotiq (max 235 N) MUST be force-clamped in software (`safety.py` + gripper command path) |
| Gel layer | replaceable; no solvents (alcohol/acetone), no sharp edges, −10…50 °C |

## Python SDK (`dmrobotics`, pip install from the vendor package)

- **Python 3.8–3.11 ONLY** (compiled per-version bundles; PyArmor-encrypted
  core), **linux_x86_64 + windows_x86_64 only — no macOS**. Needs numpy<2 →
  the recording/deploy box venv is **py3.11 with numpy<2 overall** (the
  tactile workers are spawned from the phantom process and share its
  interpreter; torch coexists with numpy 1.26 fine).
- Backends: `"cpu" | "cuda" | "flux"` (lowercase). GPU path: `pip install
  .[gpu]` then build TensorRT engines once per machine: `dmrobotics trt rebuild`.
- ⚠ **Every `SensorOptions` channel enable defaults to `False`**
  (`enable_raw/deformation/depth/shear/force`) — the manual wrongly says True.
  A driver that doesn't set them reads nothing.
- ⚠ `setEnableFlags(raw, deformation, depth, shear)` has **no force argument**
  (manual shows one — doc drift).
- Connect by **serial string** (`dev_id="X26040565"`, yellow cable label) for
  multi-sensor rigs; int index only for quick single-sensor tests.

### Data API — every field getter returns `(fid, data)`

| Call | Returns | Units |
|---|---|---|
| `getRawImg()` / `getInferImg()` | `(fid, DMTacImage)` — pixels at `.img` (uint8), serial at `.serial` | — |
| `getDeformation2D()` | `(fid, ndarray (H,W,2) f32)` | mm |
| `getDepth()` | `(fid, ndarray (H,W) f32)` | mm |
| `getShear()` | `(fid, ndarray (H,W,2) f32)` | mm |
| `getDistributeForce()` | `(fid, ndarray (H,W,3) f32)` per manual; `(fid, fx, fy, fz)` in some snippets — **driver handles both** | BENCH (unverified) |
| `getForce()` | `(fid, ndarray (1,6) f32)` | **Fx,Fy,Fz in N; Mx,My,Mz in 10⁻² N·m** (confirmed) |
| `getContactArea()` | bare `float` | mm² |

Sync / lifecycle: `wait_for_new(last_fid, timeout_ms)` (blocks for a new
frame), `getDevStatus()` (0 OK / 1 RESETTING / 2 DISCONNECTED),
`reset()` (re-zero reference — keep the pad untouched), `disconnect()`,
`listConnectedDevIDs()`, `getEvents()` (device log).

### Offline recompute (the storage plan)

`sensor.process(raw_img, getdepth=, getshear=, getforce=, getdistforce=)` +
`setBaseFrame(first_frame)` recompute all fields from recorded **raw
grayscale** frames offline (vendor example `gen_feat_hdf5.py`; HDF5 helpers
`init_h5/append_h5/read_h5` in `dmrobotics.utils`). So the recording plan can
archive raw @ ~37 MB/s/sensor instead of the full field stack (~62 MB/s in
fp16 at ds-resolution, ~425 MB/s at full res/fp32) and derive fields offline.
BENCH item (e): verify bit-parity on the rig, then set
`recording.archive_raw_img: true` + `offline_recompute_ok: true`.

### Extras in the vendor package

- `TactileSlipDetector` (`dmrobotics.extensions`, `slip_threshold=0.35`,
  needs `enable_deformation`) → SLIP/SAFE + coherence + valid-vector count —
  a vendor baseline for our derived slip score.
- Reference scripts: `demo.py` (viewer), `main.py` / `main_mp.py`
  (single/multi-sensor high-rate capture), `save_img.py`, `gen_feat_hdf5.py`,
  `get_force.py`, `slip_detection.py`, `gen_trt.py`.
- CAD in the full zip: sensor STEP models **and ready Robotiq 2F finger
  adapters per sensor size** (also Franka Panda) — the W2L one is in the
  "大号连接件" (Large) folder. No custom mount design needed.
- Visualization UI builds for Linux + Windows (device connect by serial,
  calibrate button = keep pad untouched).

## Remaining bench items (half a day with the unit)

(a) `getDistributeForce` units → `tactile.dist_force_unit_to_N`;
(b) true concurrent dual-sensor rates + `getInferImg` size/channels;
(c) numpy (H, W) ordering of the 384×288 grid → `tactile.hw_order`;
(d) sustained multi-stream disk throughput;
(e) offline-recompute bit-parity (then flip the two config flags).

## Field notes from the Denmark rig (2026-07, incoming/DM-Tac-SDK)

Facts observed on the OTHER team's working setup with our two units
(their scripts are vendored read-only at `../incoming/DM-Tac-SDK/`; the most
complete recording reference is
`scripts/recording/get_force_record_with_gripper_npz_two.py`).

- **Our unit serials: `L26050098` (their idx 0) and `L26190169` (their idx 2)**
  — now set as `tactile.sensors.dev_id` in hardware.yaml. BENCH: confirm which
  is mounted left vs right. USB identity: product token `N160MU2`, serial
  regex `[LMSX]\d+` (2nd-gen gate: numeric part ≥ 2548).
- **Network options are real and used** even with cpu/cuda backends:
  `SensorOptions(remote_addr="192.168.127.10:50051" [2nd sensor :50052],
  pc_host="192.168.127.100", pc_port=60001)`. The sensors live on their own
  subnet `192.168.127.x` (robot subnet is `192.168.88.x`; one older script
  used `10.42.0.x` — a different, earlier network regime). Threaded through
  `TactileSensorEntry.remote_addr/pc_port` + `TactileConfig.pc_host`.
- **Measured rate: ~10 Hz** (Δt ≈ 0.099 s) — but that run had `max_fps=10`
  set, so it is a floor, not the ceiling; bench item (b) stands.
- Their recordings enable only deformation+depth+force (`enable_shear=False`
  everywhere; `getDistributeForce`/`getContactArea` never called) — our
  driver reads the full set.
- Their npz schema (for cross-checks / conversion): `time (N,)`,
  `force (N,6)`, `depth (N,288,384)`, `deformation (N,288,384,2)`
  [+ `force_i/depth_i/deformation_i` per sensor + `gripper_target/pos/force`
  0-255 + row-aligned `cam_<id>.mp4` in the two-sensor gripper variant].
- Gripper endpoint confirmed working: URCap socket `192.168.88.56:63352`
  (right UR3 = ours) — matches `drivers/real/robotiq.py` + `arm.ip`.
- SDK `sensor.process()` offline replay demo: `scripts/offline/gen_feat_hdf5.py`
  (visual only); TensorRT engine rebuild one-liner: `scripts/offline/gen_trt.py`.
