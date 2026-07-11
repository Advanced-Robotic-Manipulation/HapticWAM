# PROJECT STATUS — living document

> Update this file whenever a milestone lands. A fresh Claude/Opus session (or
> a new teammate) should be able to read THIS + README.md and know exactly
> where the project stands and what to do next. Last update: **2026-07-10**.

## Machines

| Host | What | State |
|---|---|---|
| **compute3** (`ssh compute3`, user physicalai) | **RTX 5090 32 GB — THE paper box** (train + deploy) | **PROVISIONED 2026-07-10**: `~/phantom-icra-2027/` has repo+submodule, py3.11 venv (numpy<2, torch 2.13+cu130), dmrobotics SDK[gpu], Cosmos weights (4.6 GB), paths.local.yaml. 67/67 tests; `smoke_test --synthetic --device cuda` on REAL weights: **ALL STAGES PASSED, peak VRAM 20.32 GiB @ batch 1**. Shared lab box (root uvicorn:8000 + k3s — don't touch); 291 GB free. |
| **compute2** (`ssh compute2`, user isr-lab-4) | RTX 4090 24 GB — robot/teleop box | Arm rig lives here: 2× UR3 (right 192.168.88.56 = ours, left .40), Robotiq 2F-85, RealSense, Echo exo leader. Their teleop = `Echo-Yolo/Echo/main11.py` (conda env `echo`) — vendored minimal copy in `third_party/echo_teleop/` (was untracked on their disk!). Root disk FULL — use /media/isr-lab-4/juniors (285 GB free) for anything new. |

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
   `archive_raw_img` + `offline_recompute_ok`). Put real sensor SERIALS into
   `tactile.sensors` dev_ids.
4. **Arm confirmation** — the arms are **UR3s** (two, right=.56 is ours).
   Generation (UR3e vs CB3) still unknown → read the pendant/plate; CB3 ⇒
   order the Robotiq FT-300S (ACC needs a real wrist wrench). Update
   `arm.model/generation/ip` + wrist_ft in configs/hardware.yaml.
5. **Gripper mounts** — fabricate the vendor Robotiq-2F adapters from the CAD
   zip (Large folder for W2L); clamp gripper force ≤ pad's 30 N in config.
6. **Teleop swap** — the lab teleop is now KNOWN and vendored:
   `third_party/echo_teleop/` (Echo exo leader → right UR3, see PROVENANCE.md).
   Integration = write `phantom/teleop/echo.py` TeleopDevice wrapping
   `echo_teleoperation.Echo` + settle joint-space vs Δ-EE action semantics
   (notes in PROVENANCE.md §Integration).
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
