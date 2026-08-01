# Code structure

One line per module, plus the design rules that keep the codebase coherent.
The authoritative research spec is [../pipeline.md](../pipeline.md); this file
maps spec → code.

## Design rules (enforced, not aspirational)

1. **No hardware literal outside `configs/hardware.yaml`.** Every tensor
   shape, rate, unit scale and threshold derives from `HardwareConfig`. Ranks
   are fixed by the schema; values are expected to change after the bench.
   Checkpoints snapshot the config and refuse to load if *shape-relevant*
   fields changed (value-only changes warn).
2. **Import discipline.** `phantom/config`, `data`, `drivers`, `recording`,
   `timesync`, `teleop`, `eval` and `tests` never import `cosmos_predict2` —
   they run on any machine. Only `phantom/backbone` and `phantom/model`
   (and `train`/`inference`/`deploy` through them) touch cosmos, always via
   `phantom.backbone.loader.setup_cosmos()` (installs shims + sys.path).
3. **One prefix separates ours from theirs.** Every new trainable parameter
   name contains `phantom_` (LoRA params contain `lora_`). Base-checkpoint
   loading, freezing, checkpoint save/load and teacher→student init are all a
   single prefix test (`backbone/loader.py`, `train/common.py`).
4. **One contact ontology.** `phantom/config/model.py::EVENTS`
   (`none/onset/hold/slip/release`) is shared by derived labels, ACC, ACE and
   eval. `phantom/data/derived.py` is the single implementation of every
   derived quantity — recording post-pass, dataloaders, ACC labels, HID-S
   reward and eval metrics all call the same functions.
5. **RF targets are deterministic.** Generated groups (contact package,
   actions) are packed into latent frames by parameter-free codecs
   (`model/ace/packing.py`); trainable capacity lives on the input side (HHT)
   and in the readout heads.

## configs/

- `hardware.yaml` — the hardware value store; `BENCH:` marks unverified defaults.
- `paths.yaml` (+ optional `paths.local.yaml`) — cosmos repo/weights, data/run roots.
- `eval_campaign.example.yaml` — template for `run_eval`.

## phantom/config/

- `hardware.py` — pydantic schema + cross-field validators (CB3⇒FT-300S+125 Hz,
  divisibility, channel sum = 8, executor==RTDE rate, force-unit sentinel …)
  and derived quantities (`cpk_shape`, `ur_state_dim`, `field_bytes_per_s`,
  `config_hash`, `shape_relevant_fields`).
- `paths.py` — paths schema, local-override merge.
- `backbone.py` — transcribed frozen Cosmos net kwargs + latent geometry +
  `tiny()` CPU preset + `verify_against_state_dict()` (transcription-drift guard).
- `model.py` — PHANTOM-specific knobs: layout counts, loss weights λ, LoRA r/α,
  ACC config, `rope_time_mode`, `student` flag; the EVENTS ontology.
- `training.py` — dataclass configs for the four training programs.

## phantom/data/

- `derived.py` — pure numpy: contact mask, CoP (f_z-weighted centroid), slip
  score (flow ⊕ friction-cone blend), event labels (debounced transitions),
  gate label, reactive (CASA) statistic, τ_obj auto-calibration, peak force.
- `schema.py` — canonical stream names, `EpisodeMeta`, `NormStats`.
- `episode_store.py` — zarr v2 episodes (`data`+`ts` per stream), `EpisodeReader`
  with nearest/window/interp queries on the master clock.
- `windows.py` — `WindowSampler`: episode → model-tick training window
  (video, gel, fields, contact state, wrist window, UR state, action chunks,
  contact-package targets, event/gate labels). Timestamp-driven — degraded
  rates load identically. Warns on provenance (episode recorded under
  different shape-relevant config).
- `synthetic.py` — schema-conforming episodes from the mock signal model on a
  virtual clock (`--synthetic` smoke data). Regenerated per config hash.
- `contact_play.py` — full-res keyframe dataset for the tactile SSL pretrain.

## phantom/drivers/

- `base.py` — `TactileSensor`/`Arm`/`Gripper`/`Camera` ABCs, frame dataclasses,
  `Rig` aggregate; first-frame shape checks against the config.
- `mock/scenario.py` — `ContactScenario`: the deterministic phase machine
  (approach→onset→hold→slip→release) ALL mocks share, with the wrist-F/T ramp
  leading tactile onset (ACC lead-time structure).
- `mock/{dmtac,ur,robotiq,realsense}.py` — co-varying synthetic signals; the
  tactile mock emits SDK-unit forces consistent with the configured unit scale.
- `real/dmtac.py` — `dmrobotics` wrapper; hw_order transpose and force-unit
  scaling happen at this boundary, nowhere else.
- `real/ur.py` — `ur_rtde`; Receive always, Control only on `connect(control=True)`
  (exclusive ownership), protective-stop reconnect.
- `real/robotiq.py` — URCap socket (port 63352) ASCII protocol.
- `real/realsense.py` — pyrealsense2.
- `factory.py` — `make_rig(hw)`: real/mock per config `mode` section.

## phantom/recording/ + timesync/ + teleop/

- `ringbuffer.py` — single-writer shared-memory rings (lock-free readers).
- `workers.py` — per-tactile-sensor **spawn-safe processes** (in-child
  downsample + f16 + keyframe decimation; full-res frames never cross IPC
  except keyframes); arm/gripper/camera thread pollers; `SensorSession`.
- `recorder.py` — drains rings → zarr episode; t_host→t_master mapping;
  throughput watchdog vs the config-derived estimate.
- `postprocess.py` — offline derived-channel pass (recomputable when τ change).
- `timesync/clock.py` — RTDE master clock (lowest-latency-decile calibration,
  EWMA drift); identity in mock mode.
- `teleop/` — keyboard (works everywhere) + SpaceMouse; episode-control button
  semantics (start/stop, success/fail, failure tagging, F/T zero).

## phantom/backbone/

- `compat.py` — `sys.modules` shims for `transformer_engine` (RMSNorm + RoPE,
  numerics-matched), `megatron`, `multistorageclient`; valid `__spec__` for
  peft probing. **Shim parity must be verified on Linux before trusting
  full-checkpoint numbers** (`scripts/verify_backbone.py`).
- `loader.py` — `setup_cosmos` (shims + sys.path + `cosmos_cuda` stub),
  `build_phantom_net`, `load_base_weights` (strip `net.`/`accum_*`/
  `_extra_state`; assert missing ⊂ `phantom_*`), `inject_lora` (peft, after
  base load), `set_trainable`.
- `vae.py` — frozen Wan2.1 VAE wrapper + geometry-exact `FakeVAE` for smoke.
- `text_embedding.py` — cached empty-string Reason1 embedding (what the
  action-cond checkpoint was post-trained with).

## phantom/model/

- `sequence.py` — `SequenceLayout`: THE single source of truth for the
  extended frame axis (video | obs | contact | action), cond mask, frame-type
  ids, RoPE positions, structural attention mask, haptic-group token mask.
  Student/drop-video variants are layout changes, not new architectures.
- `phantom_dit.py` — `PhantomDiT(ActionChunkConditionedMinimalV1LVGDiT)`:
  re-implemented forward on the same submodules; frame-type embeddings
  (zero-init), custom per-frame RoPE, intent chunk through the pretrained
  action-AdaLN path, ACC gate + attention bias, CONTACT hidden taps.
- `attention_bias.py` — `BiasHolder`/`BiasedSDPAOp` wrap of each block's
  `self_attn.attn_op` (torch-SDPA backend only, hard-asserted). The holder is
  cleared at the START of the next forward — activation checkpointing re-runs
  blocks during backward and must see the same bias.
- `acc.py` — the anticipatory gate: φ over [wrist TCN ‖ z_vis ‖ intent ‖
  prev-ĉ summary]; g/p_evt/α heads; CASA reactive term; α≡0 ⇒ CASA,
  student ⇒ α→1.
- `hht/` — tactile field conv encoder (+ SSL pretrain heads), contact-state
  MLP, wrist TCN, UR-state MLP, observation-frame assembly.
- `ace/packing.py` — deterministic ContactPackage/action ⇄ latent-frame codecs
  (per-finger Δ-fields, event bands, CoP bumps in per-finger local coords,
  slip tiles, wrench/wrist block grids; action value strips).
- `ace/heads.py` — EventReadout + SigmaHead (heteroscedastic σ → HID
  confidence + speed governor).
- `ace/losses.py` — grouped RF objective (λ_a/λ_v/λ_c/λ_w + ACC auxiliaries).
- `rf.py` — `PhantomRectifiedFlow`: build_x0 (FRAME_REPLACE), training_step,
  `prepare_denoise` (shared-(x_t,t) hook for HID/HID-S), few-NFE Euler
  `sample()` with drop-video.

## phantom/train/

- `builder.py` — `build_model()`: layout → net (+base weights, +LoRA) → HHT →
  RF wrapper; `tiny=True` CPU preset.
- `common.py` — DDP, optimizer/scheduler, EMA, **checkpoint format** (never
  re-saves the 2B base; lora + phantom + EMA + config snapshots + norm stats),
  `WindowDataset`, synthetic-dataset provisioning, generic train loop.
- `pretrain_tactile.py` (1) — masked recon + cross-channel SSL.
- `train_teacher.py` (2) — teacher fine-tune, joint objective.
- `distill_hid.py` (3) — saliency×confidence trajectory distillation + soft
  event KL + shared-noise action-velocity behavior matching + GT grounding.
- `dagger_driver.py` (3b) — relabel rollouts → manifest → re-distill.
- `finetune_hids.py` (4) — AWR with auto-τ_obj + KL leash to frozen reference.

## phantom/inference/ + deploy/

- `inference/policy.py` — `PhantomPolicy.replan(obs, prev_plan)` → `Plan`
  (denormalized chunk on the action grid, σ profile, gate, contact package =
  next replan's ACC input).
- `deploy/planner.py` — ObsSnapshot from rings (system modes select model
  inputs; tactile always recorded), replan loop, planner trace.
- `deploy/executor.py` — executor-rate servo streaming, cumulative-delta pose
  interp, governed playback time, chunk swap + blend, stale-plan hold.
- `deploy/governor.py` — σ → playback speed (path-preserving).
- `deploy/safety.py` — wrench/tactile/workspace/protective-stop checks every
  tick, priority over the governor, t_master-logged events.
- `deploy/runtime.py` — wires one episode/trial end to end.

## phantom/data_collect/

Ground-up operator app for Echo teleop + data collection — see
[data_collect_app.md](data_collect_app.md) for the why/how; entry point
`phantom/scripts/collect.py`. Reuses `phantom.drivers`/`recording`/
`teleop.echo`/`viz.rerun_logger` underneath; replaces only the application
layer that drives them.

- `config.py` — `configs/data_collect.yaml` schema (ports/serials, external
  drive, teleop feel, safeguard defaults, mode); `load_collect()` patches the
  hardware config's ports section before validation.
- `teleop.py` — `DirectServoStreamer`: pulls the Echo reader's cached target
  every `control_rate_hz` cycle (no 10 Hz hop), TRACK is a pure per-cycle
  slew limit (transparent below `v_max`), `AccelLimitedTracker` glide kept
  only for ENGAGE/resume; all phase transitions are compare-and-set so a
  concurrent safeguard `hold()` always wins; workspace hold retreats to the
  last safely-inside pose; protective-stop recovery reconnects the control
  interface.
- `gripper.py` — `GripperPilot`: continuous 0..1 position on its own thread,
  pulling the leader's freshest squeeze value (device-rate, not the session
  loop); pad-force/close clamps stay at the driver boundary.
- `safeguard.py` — `TactileSafeguard` (latched trip on fingertip force /
  indentation / stream staleness; runtime-editable thresholds; resume only
  via an explicit call) + `ArmGuard` (wrench/protective-stop, auto-resume).
- `session.py` — `CollectApp`: the main loop (episode lifecycle, absolute
  `actions_abs` logging, safeguard/arm-guard bookkeeping) and `CollectEcho`
  (adds live gripper-tick calibration to `phantom.teleop.echo.EchoTeleop`).
- `panel.py` — stdlib SSE control panel (safeguard card, session wizard with
  the full/lite mode switch, gripper calibration, offload card).
- `rerun_view.py` — `CollectRerun(RerunLogger)`: mode-gated live view (full =
  everything; lite = scene RGB + arm + gripper + tactile wrench only).
- `offload.py` — verified move of staged episodes to the external drive
  (per-file size/sha256 check before local delete; same-filesystem guard).
- `playback.py` — episode playback + recording verification from the panel:
  `verify_episode` (per-stream integrity report: presence vs the mode's
  expected set, rates, monotonic ts, finite values), `EpisodePlayer`
  (wall-clock-paced replay of every modality into the embedded rerun viewer
  on a separate `t_episode` timeline; pause/stop/speed), and
  `PlaybackController` (drive/staging scan, path confinement, mutual
  exclusion with sessions/offloads). CLI: `phantom/scripts/play_episode.py`.

## phantom/dagger/ + eval/

- `dagger/rollout.py` — student rollouts with full-sensor recording.
- `dagger/relabel.py` — offline teacher relabeling from recorded streams.
- `dagger/manifest.py` — round-k training mixes (DAgger on/off ablation).
- `eval/trial_runner.py` — campaign YAML → resumable ledger CSV.
- `eval/metrics.py` — peak force, threshold violations, slip recovery, **ACC
  lead time** (gate crossing vs derived onset), event F1, latency.
- `eval/aggregate.py` — recovery ratio + retention tables, markdown report,
  lead-time distribution export.

## phantom/scripts/

`mock_smoke` · `smoke_test` · `verify_backbone` · `bench_day1` ·
`record_episodes` · `postprocess_episodes` · `dump_norm_stats` · `run_deploy` ·
`run_dagger_round` · `run_eval` · `collect` (`phantom.data_collect`'s entry
point — [data_collect_app.md](data_collect_app.md)) · `play_episode`
(verify/replay a recorded episode without the panel) — see
[launch_guide.md](launch_guide.md).
