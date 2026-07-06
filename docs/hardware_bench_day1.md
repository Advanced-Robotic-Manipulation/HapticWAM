# Day-1 hardware bench

Blocking items from pipeline.md's verification checklist. Each procedure names
the **exact `configs/hardware.yaml` field(s) it writes**. Finish by setting
`meta.bench_verified: true` and committing the YAML; then rerun
`pytest tests/` and `python -m phantom.scripts.mock_smoke`.

The semi-automated runner (`python -m phantom.scripts.bench_day1 --items ...`)
executes the measurable parts and prints the values; it never edits the YAML —
you review and edit.

---

## ARM — generation + wrist F/T (pipeline.md checklist item 2)

1. On the pendant: PolyScope version. 5.x/e-Series has a built-in 6-axis
   F/T sensor; 3.x/CB3 only estimates TCP force from joint currents.
2. `python -m phantom.scripts.bench_day1 --items arm` — verifies RTDE
   connectivity and the sustained receive rate.

| Finding | Fields to set |
|---|---|
| e-Series | `arm.generation: e-series`, `arm.rtde_receive_hz/rtde_control_hz: 500`, `wrist_ft.source: ur_internal`, `wrist_ft.rate_hz: 500`, `control.executor_rate_hz: 500` |
| CB3 | `arm.generation: cb3`, rates `125`, **order a Robotiq FT-300S now** (ACC depends on a clean wrist wrench), `wrist_ft.source: ft300s`, `wrist_ft.rate_hz: 100`, `control.executor_rate_hz: 125` |
| Always | `arm.ip`, `arm.model`, `arm.tcp_offset_m` (measure with gripper + sensors mounted), `arm.payload_kg`, `safety.workspace_m` (jog the arm to the safe extremes and read TCP) |

The config validators enforce the CB3 forks — an inconsistent combination
refuses to load.

## (b) Tactile shapes + true concurrent rates

`python -m phantom.scripts.bench_day1 --items b`

NOTE (SDK verified, docs/sensor_sdk.md): every field getter returns
`(fid, data)`, and the image getters return `(fid, DMTacImage)` with pixels at
`.img` (uint8 grayscale) — the driver unpacks these; the bench script reads
through the driver, so shapes below are already canonical.

- `getInferImg()` `.img` shape → `tactile.infer_img {h, w, c}`
- `getRawImg()` `.img` shape → `tactile.raw_img` (expected 480×640 gray)
- field getters' shapes → `tactile.field {h, w}` (interpret with item (c)!)
- sustained dual-sensor concurrent rate over 5 s → `tactile.rate_hz`; if it
  lands below 120, also lower `recording.field_ds_rate_hz` accordingly.
  Put the two sensors on separate USB root hubs before concluding.

## (c) numpy (H, W) ordering

`python -m phantom.scripts.bench_day1 --items c`

Press a fingertip near the corner that is TOP-LEFT in the sensor's mounted
orientation; the script prints the argmax index of the depth map.

- axis 0 counts along the configured `field.h` side → `tactile.hw_order: hw`
- axes swapped vs the physical press → `tactile.hw_order: wh`

The real driver transposes at the boundary; nothing downstream ever sees the
SDK ordering.

## (a) Force units

`python -m phantom.scripts.bench_day1 --items a`

The resultant wrench units are ALREADY DOCUMENTED (dev manual §1.2.2.8):
Fx,Fy,Fz in N (`force_unit_to_N: 1.0`), Mx,My,Mz in 1e-2 N*m
(`torque_unit_to_Nm: 0.01`) — this item only SANITY-CHECKS them and measures
the undocumented distributed-force units.

Place a known mass (e.g. 100 g calibration weight → 0.981 N) flat on the gel:

- `expected_N / median(getForce z)` ≈ 1.0 confirms the documented scale
- `expected_N / median(sum getDistributeForce z)` → `tactile.dist_force_unit_to_N`

Repeat with 2–3 masses; accept only if roughly linear. **If inconclusive,
leave both at `0.0`** — the UNCALIBRATED sentinel automatically keeps the
whole stack on depth-based contact + the deformation fallback (pipeline.md §5):
`derived.contact_source` must stay `depth`, HID-S penalizes peak indentation,
and the tactile-force e-stop uses `safety.tactile_depth_limit`.

If calibrated: set `derived.contact_source: fz`, tune `derived.tau_contact_fz`
(a light touch should just cross it), set `safety.tactile_fz_limit_N`, and
**rerun `dump_norm_stats`** on any existing data.

## (d) Sustained recording throughput

`python -m phantom.scripts.bench_day1 --items d` prints the config-derived
MB/s estimate; then run a 60 s real recording:

```bash
python -m phantom.scripts.record_episodes --task bench_throughput
```

Watch for `recorder drain took ...s > interval` warnings — sustained warnings
mean the disk cannot keep the configured plan; lower `recording.field_ds`
resolution or `field_ds_rate_hz`, or move `data_root` to NVMe.

## (e) Offline field recompute

Archive a short raw-image stream (`recording.archive_raw_img: true`,
temporarily), power-cycle the sensor, and check whether the SDK can rebuild
`getBaseFrame`-referenced fields from the archived raws. If yes →
`tactile.offline_recompute_ok: true` (enables the raw-archive recording plan
as a disk-space fallback).

## Derived-threshold calibration (after a–c)

With one sensor live, run light/firm presses and slides while watching the
derived channels (record + `postprocess_episodes`, or a small notebook over
`phantom.data.derived`):

- `derived.tau_contact_depth` — a light touch crosses it, sensor noise does not
- `derived.tau_slip` + `derived.friction_cone_mu` — deliberate slides flag
  `slip`, firm holds do not
- cross-check our mask-derived area against native `getContactArea` within
  `derived.area_crosscheck_tol`

## Finish

```yaml
meta:
  bench_verified: true
```

```bash
pytest tests/ && python -m phantom.scripts.mock_smoke
python -m phantom.scripts.dump_norm_stats --data <any recorded data>   # if units changed
```
