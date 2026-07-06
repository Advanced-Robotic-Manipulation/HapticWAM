# PROJECT STATUS — living document

> Update this file whenever a milestone lands. A fresh Claude/Opus session (or
> a new teammate) should be able to read THIS + README.md and know exactly
> where the project stands and what to do next. Last update: **2026-07-06**.

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

1. **Weights download** (any GPU box, ~5 GB total — paths already match
   `configs/paths.yaml`):
   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/Cosmos-Predict2.5-2B \
       robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt \
       robot/action-cond/cr1_empty_string_text_embeddings.pt \
       tokenizer.pth --local-dir /path/to/cosmos-predict2.5-2b
   # then point configs/paths.local.yaml cosmos_weights_root at it
   ```
2. **5090/compute2 machine setup** — `requirements/README.md` (venv rule:
   py3.11 + numpy<2, tactile workers share the interpreter). Verify with
   `pytest`, `mock_smoke`, `smoke_test --synthetic --device cuda` (this also
   measures real VRAM — decisive for batch/res choices), then
   `verify_backbone --save-ref` (backbone parity, never yet run on GPU).
3. **Day-1 sensor bench** — `docs/hardware_bench_day1.md`. Reduced to 5 items:
   dist-force units, concurrent dual-sensor rates + infer_img size, (H,W)
   ordering, disk throughput, offline-recompute bit-parity (then flip
   `archive_raw_img` + `offline_recompute_ok`). Put real sensor SERIALS into
   `tactile.sensors` dev_ids.
4. **Arm confirmation** — model plate: UR5e (built-in 500 Hz F/T) vs CB3 UR5
   (buy Robotiq FT-300S). Set `arm.generation` + wrist_ft accordingly.
5. **Gripper mounts** — fabricate the vendor Robotiq-2F adapters from the CAD
   zip (Large folder for W2L); clamp gripper force ≤ pad's 30 N in config.
6. **Teleop swap** — lab's own teleop replaces `phantom/teleop/*` (interface:
   emit `TeleopCommand`s / call `recorder.record_action`).
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
- **Compute**: assume single RTX 5090 (LoRA, 256–480p). If an 8×H100 window
  materializes, the full-FT Cosmos-Policy recipe becomes an option — decide
  only then (`requirements-h100.txt` is ready).
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
