# PROJECT STATUS — living document

> Update this file whenever a milestone lands. A fresh Claude/Opus session (or
> a new teammate) should be able to read THIS + README.md and know exactly
> where the project stands and what to do next. Last update: **2026-07-17**.

## Machines

| Host | What | State |
|---|---|---|
| **compute3** (`ssh compute3`, user physicalai) | **RTX 5090 32 GB — THE paper box** (train + deploy) | **PROVISIONED 2026-07-10**: `~/phantom-icra-2027/` has repo+submodule, py3.11 venv (numpy<2, torch 2.13+cu130), dmrobotics SDK[gpu], Cosmos weights (4.6 GB), paths.local.yaml. 67/67 tests; `smoke_test --synthetic --device cuda` on REAL weights: **ALL STAGES PASSED, peak VRAM 20.32 GiB @ batch 1**. Shared lab box (root uvicorn:8000 + k3s — don't touch); 291 GB free. |
| **compute2** (`ssh compute2`, user isr-lab-4) | RTX 4090 24 GB — robot/teleop box | Arm rig lives here: 2× UR3 (right 192.168.88.56 = ours, left .40), Robotiq 2F-85, RealSense, Echo exo leader. Their teleop = `Echo-Yolo/Echo/main11.py` (conda env `echo`) — vendored minimal copy in `third_party/echo_teleop/` (was untracked on their disk!). Root disk FULL — use /media/isr-lab-4/juniors (285 GB free) for anything new. |
| **nuc** (`ssh nuc@100.64.0.18`, tailscale) | Intel NUC, 8-core, **no NVIDIA GPU** — recording box | **FIELD-TESTED 2026-07-17**: `~/phantom-icra-2027` (py3.10 venv, torch CPU, rerun, dmrobotics SDK `--no-deps`). **Both DM-Tac sensors on local USB** (video6=L26050098, video8=L26190169); RealSense D435 (944523029596); WD Passport (data disk). Config `configs/hardware.nuc.yaml` (tactile:real, `sdk_backend: cpu`, 10 Hz — measured concurrent dual-sensor rate on CPU). `paths.local.yaml` → `/home/nuc/phantom-data`. Real episode recorded end-to-end via the web panel over tailscale (both sensors, real infer_img, no false safety trip). Robot NOT wired here (arm/gripper stay mock on this box). `nuc` user added to `video` group. |

## Where we are, one paragraph

The codebase is **complete and train-ready**: all pipeline components are
implemented and executing (67/67 tests, zero skips; `mock_smoke` end-to-end
recording pipeline passes; `smoke_test --tiny --synthetic` runs the full ML
path — teacher step → 5-NFE sample → HID → HID-S → checkpoint round-trip).
Nothing has trained on real data yet because the physical rig hasn't been
benched and the Cosmos weights aren't downloaded on a GPU box. Everything
left is hardware-side or a run, not code.

## DONE (with evidence)

| What | Evidence |
|---|---|
| `phantom/data/` layer (schema, zarr store, derived channels, windows, synthetic, contact-play) | `tests/test_derived_closure.py`, `tests/test_windows.py` |
| Real DM-Tac driver vs verified SDK v1.2.10 (fid/DMTacImage unpacking, enable flags, serial connect, SI wrench scaling) | `phantom/drivers/real/dmtac.py`; facts in `docs/sensor_sdk.md` |
| Recording pipeline incl. raw-frame archive path (`archive_raw_img`) + offline recompute script | `mock_smoke` passes; raw path verified on mocks (273 frames); `scripts/recompute_fields.py` |
| Model core: HHT / ACC (true two-pass implemented) / ACE (per-group σ NLL) / LFA / rectified flow | `tests/test_model_tiny.py`, `smoke_test` ALL STAGES |
| Training programs: pretrain_tactile, train_teacher (`--acc-two-pass`), distill_hid (+DAgger), finetune_hids | `smoke_test`; pretrain ran to ckpt on synthetic |
| Deploy runtime: planner/executor (chunk blending, action recording)/σ-governor/safety | code-audited; imports clean; mock-tested via runtime path |
| Eval: metrics (lead-time, event-F1), recovery/retention aggregation with 95% seed-bootstrap CIs | `phantom/eval/` |
| Cosmos submodule integration (imports through compat shims; `[cosmos]` extra) | 92/92 modules import with submodule present |
| Docs: sensor SDK truth, bench plan, SOPs, playbooks, launch guide | `docs/` |
| **Echo teleop integrated** (2026-07-17): `teleop/echo.py` clean-room device, joint-space branch in `record_episodes`, Δ-EE derived from measured TCP (`data/derived.py pose_delta`), raw joint targets to new `actions_qtarget` stream, continuous gripper | `tests/test_echo.py`; scripted mock rehearsal: both streams written, Δ-EE nonzero |
| **Teleop smoothing (SOTA pass, 2026-07-17)**: one-euro filter on the leader signal at device rate + `JointServoStreamer` @ `control_rate_hz` (125 Hz, kills the 10 Hz step-and-hold) + discrete-correct bounded vel/accel tracker with exact no-overshoot landing (`teleop/filters.py`, `teleop/streamer.py`); workspace hold + p-stop park moved into the streamer at control rate; gripper EMA + command deadband | 20 tests in `tests/test_echo.py` incl. noise attenuation, accel bounds, no-overshoot, streamer-on-mock; echo rehearsal green through the streamer |
| **First real-hardware run (NUC, 2026-07-17)** — fixed 3 latent bugs that only surface with real single-open sensors: **(1) double-open** — `Rig.connect_all` opened tactile in the parent AND the SensorSession workers re-opened the same physical device → "occupied"; added `Rig.worker_owned_tactile` (recording path skips parent tactile open). **(2) reset transient** — first frame(s) after SDK reset spike to depth peak ~2.0 (vs settled ~0.09) and tripped the tactile e-stop; `DmTacSensor._warmup()` discards them. **(3) orphaned workers** — a hard-killed parent (kill -9/crash) left `daemon` tactile workers holding the device; added Linux `PR_SET_PDEATHSIG(SIGKILL)`. | `tests/test_rig.py` + updated suite (120 passed); real dual-sensor episode on the NUC verified clean (both wrench+fields+infer_img streams, safety silent), clean stop self-releases devices |
| **Web-first collection (2026-07-17)**: `python -m phantom.scripts.panel` — the operator's entry point: session wizard in the browser (rig summary, pre-flight checklist, task/operator/device/target form, RU labels) → `SessionRunner` brings up the rig and runs the same `run_collection` path as the CLI; progress bar + episode log with outcomes, safety banners with recovery hints, session restart without process restart; `--teleop none` (NullTeleop) for contact-play/monitoring | 7 more panel tests (118 total); e2e wizard rehearsal: 2 full sessions incl. episode recording driven entirely over HTTP; both views screenshot-verified |
| **Live viz + web control panel (2026-07-17)**: `phantom/viz/` — `RerunLogger` streams every session ring (cameras, tactile fields/wrench, arm, gripper, events) to an embedded rerun.io web viewer (verified on rerun 0.34, legacy fallbacks kept); stdlib-only dark-UI control panel (`--panel`, :8788) with SSE live status, wrench sparkline, and the full episode-control button set wired into the record loop as a third button source | `tests/test_panel.py` (6 tests); e2e: episode recorded start→success entirely via the web API; panel screenshot verified in a browser |
| **Safety circuit closed over teleop (2026-07-17)**: `SafetyMonitor` (was deploy-only) now runs every teleop tick — wrist wrench, fingertip force/indentation e-stop, tactile staleness, workspace; STOP saves the episode as failure + parks the streamer; auto-resume via new `recovered()` hysteresis gate (0.8×limits + fresh tactile). **Gripper pad ceiling enforced**: `force_range_N`→N mapping + `cmd_force_limit_N: 30` in config, every `move()` clamps at the driver boundary; fixed real config bug — `default_force: 0.3` was ≈85 N on the 2F-85 (pad crush), now 0.04 ≈ 28.6 N, validator rejects regressions | `tests/test_safety.py` (11 tests — SafetyMonitor previously untested); e2e trip rehearsal: p-stop mid-episode → abort+hold → clear → auto-resume |
| **DM-Tac Denmark-rig facts integrated** (2026-07-17): serials `L26050098`/`L26190169` + network fields (`remote_addr/pc_host/pc_port`) in config schema+yaml+driver; arm.ip = .56 confirmed; CLI `tactile_monitor`; latent bug fixed — `Rig.connect_all` never called `gripper.activate()` (real URCap rejects move() unactivated) | 89 passed / 6 skipped (skips = no cosmos submodule on the dev Mac); `mock_smoke` green; raw dump vendored read-only at `../incoming/DM-Tac-SDK/` |

## NOT DONE (everything below needs the rig, a GPU box, or humans)

1. ~~Weights download~~ **DONE on compute3** (anonymous HF download worked;
   4.6 GB at `~/phantom-icra-2027/cosmos-predict2.5-2b`).
2. ~~5090 machine setup~~ **DONE 2026-07-10 (compute3)** — real-weights GPU
   smoke green, peak VRAM 20.32 GiB @ batch 1 (≈11 GiB headroom on 32 GB).
   Still open: `verify_backbone --save-ref` parity ref + `dmrobotics trt
   rebuild` (defer to when sensors arrive). compute2 (recording rig) still
   needs its venv+SDK when sensors are mounted there.
3. **Day-1 sensor bench** — `docs/hardware_bench_day1.md`. Reduced to 5 items:
   dist-force units, concurrent dual-sensor rates + infer_img size, (H,W)
   ordering, disk throughput, offline-recompute bit-parity (then flip
   `archive_raw_img` + `offline_recompute_ok`). ~~Put real sensor SERIALS into
   `tactile.sensors` dev_ids~~ **DONE 2026-07-17** (`L26050098`/`L26190169`
   from the Denmark rig) — still confirm left-vs-right mounting. NEW small
   item: calibrate `teleop.echo.gripper_open/closed_tick`.
4. **Arm confirmation** — the arms are **UR3s** (two, right=.56 is ours).
   Generation (UR3e vs CB3) still unknown → read the pendant/plate; CB3 ⇒
   order the Robotiq FT-300S (ACC needs a real wrist wrench). Update
   `arm.model/generation/ip` + wrist_ft in configs/hardware.yaml.
5. **Gripper mounts** — fabricate the vendor Robotiq-2F adapters from the CAD
   zip (Large folder for W2L); clamp gripper force ≤ pad's 30 N in config.
6. ~~**Teleop swap**~~ **DONE 2026-07-17** — `phantom/teleop/echo.py`
   (clean-room, NOT importing third_party) + `record_episodes --teleop echo`.
   Semantics settled: control stays joint-space (the lab's proven path),
   dataset actions stay Δ-EE derived from measured TCP, raw joint targets
   additionally recorded to `actions_qtarget`. Remaining: first dry run on
   the rig + gripper-tick calibration (bench doc).
7. **Contact-play collection** (2–4 h) → `pretrain_tactile`.
8. **Teleop dataset** (150 eps × 5 tasks + 20–30 failure eps per fragile task)
   → `postprocess_episodes` → `dump_norm_stats` — `docs/data_collection_sop.md`.
9. **Training runs** — `docs/training_playbook.md`. The FINAL teacher (the one
   whose gate lead-time is reported, RQ2) MUST use `--acc-two-pass`.
10. **2 DAgger rounds** on the rig; optional HID-S; **eval campaign**
    (5 systems × 5 tasks × occlusion, ≥3 seeds × 20 trials) → `run_eval`.

## Open decisions (Mikhail + Kostya)

- **Name**: PHANTOM stays; expansion should encode distillation. Candidates on
  the table: "Predictive Haptic ANTicipation withOut Measurement"; or DISTAL
  (distal = the fingertip end anatomically, contains "distill"). No deadline
  until paper writing.
- **Compute**: single RTX 5090 remains the default (`configs/compute.yaml`,
  `target: rtx5090` — bit-identical legacy path). **NEW**: 8×H100 / 8×A100
  training is now a config flip + torchrun (`h100x8`/`a100x8` profiles;
  DDP-corrected loaders/sharding/device placement; `requirements-a100.txt`
  added) — see docs/training_playbook.md. The remaining decision is only
  whether to rent a node (full-FT Cosmos-Policy recipe would then be an
  option).
- ICRA 2027 deadline ~Sep 15, 2026 → data collection must start ASAP after
  the bench.

## Known honest limitations (by design, documented)

- Language conditioning is inert (cached empty-string Reason1 embedding) —
  upgrade path documented in `backbone/text_embedding.py`.
- `verify_backbone` shim-parity has never run on a real-TE Linux box (needs
  H100/NGC container) — the SDPA path is what we train with regardless.
- `scripts/recompute_fields.py` is written against the documented SDK but
  needs its `process()` return arity pinned on the rig (bench item (e)).
- GUI-less: all monitoring is logs + tensorboard.
