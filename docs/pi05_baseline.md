# pi0.5 baseline

An external VLA baseline (Physical Intelligence pi0.5, via LeRobot) run on the
HapticWAM rig through OUR deploy loop, so its numbers are comparable with the
student's. Two halves, built independently against one shared contract:

* **Export** — HapticWAM episodes to a LeRobot dataset, and the fine-tune.
* **Deploy adapter** — this document's second half: a policy server that makes
  a LeRobot policy answer `replan()` the way `phantom/deploy/` expects.

## Shared observation / action contract

Both halves must agree exactly; a mismatch here is invisible at runtime and
shows up only as a policy that behaves worse than it should.

| key | shape | content |
| --- | --- | --- |
| `observation.images.scene` | `(1, 3, 224, 224)` float32 | scene camera, RGB, CHW, `[0, 1]`, `bilinear_resize` of the full frame to a **square** (640x480 is squashed, not cropped) |
| `observation.state` | `(1, 7)` float32 | `[tcp x, y, z, rx, ry, rz, gripper aperture]`, rotvec, BASE frame |
| `task` | `[str]` | the episode instruction |
| action | `(T, 7)` | `[dx, dy, dz, drx, dry, drz, grip]` per 10 Hz step, base frame, metres / radians / aperture |

The state fields are read out of `ObsSnapshot.ur_state` at the same offsets
`PlannerLoop.run` uses (`ur_state[2*dof : 2*dof+6]` for the pose,
`ur_state[-2]` for the aperture), so the deployed state vector is by
construction the one the recorder wrote.

### The processor pipelines are load-bearing

Verified against lerobot 0.4.4. A LeRobot policy is not called on raw
observations. Normalisation, pi05's discretisation of the state into the
PaliGemma prompt and the tokenisation all live in a `PolicyProcessorPipeline`
that is saved **with the checkpoint**, and the matching unnormalisation lives
in the post-processor. `PI05Policy.predict_action_chunk` reads only the image
keys and the tokenised prompt: the state reaches the model inside that prompt.

So the adapter runs LeRobot's own chain, the one
`lerobot/utils/control_utils.py:predict_action` and
`lerobot/async_inference/policy_server.py:_predict_action_chunk` run:

```
raw numpy -> torch -> preprocessor -> predict_action_chunk
          -> postprocessor (once per chunk STEP) -> numpy
```

Two consequences for the exporter. The dataset statistics written at export
time become the normalisation the deployed policy uses, so they must come from
the same data the model was fine-tuned on. And the camera key must be the one
in `config.image_features`: pi05 pads an ABSENT camera with a blank image and a
zero mask, so a misnamed key runs happily and produces nonsense. The server
refuses that at load (`check_checkpoint_contract`), as it refuses a checkpoint
whose action dimension is not 7.

`--no-processors` exists only for a policy that genuinely ships no pipeline.
Using it on pi05 feeds the model normalised-space garbage.

## Export

`tools/export_lerobot.py` writes a LeRobot v3.0 dataset straight from the
HapticWAM episode store. It does not invent field names: the frame dict is
validated by `lerobot.datasets.utils.validate_frame` against the feature spec
handed to `LeRobotDataset.create`, so the export is whatever the installed
lerobot (0.4.4) writes.

```
python tools/export_lerobot.py \
    --episodes-root data/phantom-episodes \
    --out-root ~/data2/lerobot/phantom_pi05 \
    --repo-id phantom/scene_pi05 \
    --splits train val --recon-checks 5 --overwrite
```

Run on `compute` (the episode store lives there), 36 min for both splits.
`--limit 10` gives a smoke export.

### What came out

| | train | val |
| --- | --- | --- |
| manifest rows | 991 | 124 |
| episodes exported | 876 | 124 |
| skipped | 115 | 0 |
| frames | 177,193 | 24,652 |
| duration | 17,938 s | 2,496 s |

**Every one of the 115 skips is the same reason: `deliberate failure demo
(action_weight 0)`.** Not one episode was lost to a missing stream, a short
action grid or a camera that does not overlap the action grid, and none were
dropped as untrainable or as a policy rollout needing re-derivation. The
exporter writes the full per-episode list to `phantom_skipped.json` next to the
split, so this is checkable rather than asserted. Failure demos are dropped
because LeRobot has no per-sample action weight, and our own training path
gives them weight 0 -- keeping them would train pi0.5 on demonstrations we
deliberately teach our students to ignore. `--include-failure-demos` exports
them if a later ablation wants them.

### Validation

`tools/validate_lerobot_export.py` re-loads the written dataset through
`LeRobotDataset` -- exercising the parquet and the encoded AV1 video, not the
writer's buffers -- and checks feature shapes, dtypes, fps, per-episode counts,
a non-empty `task` on every sampled frame, images in `[0, 1]`, and the
integration identity the executor relies on.

```
python tools/validate_lerobot_export.py \
    --root ~/data2/lerobot/phantom_pi05/train \
    --episodes-root data/phantom-episodes --n-recon 8
```

Both splits pass (`OK: export is loadable and pi0.5-shaped`), 10 fps, robot
`ur5e_phantom`, one video key `observation.images.scene` at 224x224x3, state
and action both `(7,)` with the names in the contract above.

The reconstruction check integrates the LOADED actions the way
`deploy/executor.py:_pose_at` does -- `t0_pose + cumsum(action[:, :6])` -- and
compares against the recorded 125 Hz TCP stream:

| split | windows | mean err | worst peak | worst rot peak |
| --- | --- | --- | --- | --- |
| train | 8 episodes | 0.27 - 0.37 mm | 4.23 mm | 0.63 deg |
| val | 8 episodes | 0.30 - 0.42 mm | 5.46 mm | 0.85 deg |

The residual is not an export artefact. `actions` rows are deltas between TCP
poses the recorder read at its own tick, while the reference is the 125 Hz
stream sampled nearest to the nominal grid time; sub-tick lag at ~100 mm/s is a
few mm at the peak of a fast reach. The validator therefore gates on the mean
(< 1 mm) and only warns on the peak (< 10 mm).

## Fine-tune

Runs on `compute2` (RTX 4090, 24 GB). `compute`'s GPU is busy with another
training run and must not be touched.

### Three things the published checkpoint needs first

`lerobot/pi05_base` is published against a lerobot that is not 0.4.4, and
against a gated tokenizer. `tools/patch_pi05_processors.py` fixes all three,
idempotently, keeping a `.orig` beside every file it edits:

```
python tools/patch_pi05_processors.py \
    --policy-dir ~/lerobot/pi05_base \
    --tokenizer ~/lerobot/paligemma_tokenizer \
    --single-camera observation.images.scene
```

1. **Two processor steps 0.4.4 does not register.**
   `relative_actions_processor` in the preprocessor and
   `absolute_actions_processor` in the postprocessor.
   `PolicyProcessorPipeline.from_pretrained` resolves every `registry_name`
   through `ProcessorStepRegistry` before reading its config, so an unknown
   name is a hard `ImportError` -- even though both steps are published with
   `"enabled": false` and are therefore no-ops. The tool drops them and
   REFUSES to drop either if it is ever found enabled, because an enabled
   relative-action step would silently re-differentiate an action column that
   is already a pose delta.

2. **A gated tokenizer.** The preprocessor names
   `google/paligemma-3b-pt-224`, which needs manual approval, and there is no
   `HF_TOKEN` on the box. `leo009/paligemma-3b-pt-224` carries byte-identical
   tokenizer files; they are downloaded and checked against the sha256 and byte
   sizes Google publishes for its own blobs before anything is rewritten, so a
   tampered mirror cannot pass:
   `tokenizer.json` `ef6773c1...` 17,549,604 B, `tokenizer.model` `8986bb4f...`
   4,264,023 B. The step is then pointed at the local directory.

3. **Three camera slots for a one-camera rig.** `config.json` declares
   `base_0_rgb`, `left_wrist_0_rgb` and `right_wrist_0_rgb`. Left alone,
   `modeling_pi05` appends the two absent cameras as -1 images with a ZERO
   `img_mask` -- and that is a pure memory tax, because those masks become the
   2-D attention mask (nothing attends to a blank camera) and
   `position_ids = cumsum(pad_masks) - 1`, so a masked token does not advance
   the rotary position of anything after it. Measured on this GPU: the
   three-slot layout OOMs at batch 4 (20.7 GiB), one slot fits batch 6 in
   17.5 GiB. Nothing in `lerobot.policies.pi0*` matches the literal camera
   names, so the slots are collapsed onto OUR key,
   `observation.images.scene` -- which also lets the adapter's checkpoint
   contract match `config.image_features` exactly, with no `--rename_map` in
   the picture.

**A benign warning to expect.** Loading pi05_base prints `Could not remap state
dict keys: Missing key(s): ...paligemma.model.language_model.embed_tokens.weight`.
The embedding is not missing: Gemma ties it to `lm_head`, which the checkpoint
does carry. Verified directly -- the live `embed_tokens` and the checkpoint's
`lm_head` are the same tensor object (`data_ptr` equal) with matching values
(std 0.1856), and an ordinary expert weight loads as a control.

### The run

```
lerobot-train \
  --policy.path=~/lerobot/pi05_base \
  --policy.train_expert_only=true \
  --policy.dtype=bfloat16 --policy.device=cuda \
  --policy.chunk_size=50 --policy.n_action_steps=16 \
  --policy.push_to_hub=false \
  --dataset.repo_id=phantom/scene_pi05_train \
  --dataset.root=~/lerobot/data/phantom_pi05/train \
  --batch_size=6 --steps=20000 --save_freq=2500 --log_freq=50 \
  --num_workers=6 \
  --output_dir=~/lerobot/runs/pi05_phantom_expert_v1 \
  --job_name=pi05_phantom_expert_v1 --wandb.enable=false --seed=1000
```

Wrapped by `~/lerobot/launch_pi05.sh` under `setsid nohup`, log at
`~/lerobot/runs/pi05_phantom_expert_v1/train.log`. Two wrapper details are not
cosmetic: `lerobot-train` REFUSES to start when `--output_dir` already exists,
and it creates that directory only at its first checkpoint -- so the log starts
at a staging path and is renamed into the run dir once it appears (same
filesystem, the writer's descriptor follows the rename).

**Expert-only** means the whole VLM is frozen and only the action expert and
the projections train: 693.4 M trainable of 3,616.8 M. Measured, on the exported
data, against the alternatives on this 24 GB card:

| config | trainable | batch | peak reserved | s/step |
| --- | --- | --- | --- | --- |
| expert_only | 693.4 M | 6 | ~17.5 GiB | 0.262 |
| expert_only | 693.4 M | 8 | 20.0 GiB | 0.335 |
| expert_only (3 cam) | 693.4 M | 4 | OOM | - |
| lora16 | 1.3 M | 4 | 12.4 GiB | 0.159 |
| vision_frozen / full | 3,616.8 M | 1 | OOM | - |

Batch is **6, not 8**, because another user's process holds 2.3 GB on this
shared GPU: batch 8 would leave ~1.2 GB, and batch 6 leaves 4.2 GB. Full
fine-tuning and vision-frozen both OOM at batch 1, so expert-only is not a
preference here, it is the only configuration above LoRA that fits.

Defaults inherited from `PI05Config`, all of them openpi's: AdamW at peak lr
2.5e-5, betas (0.9, 0.95), weight decay 0.01, 1,000 warm-up steps then cosine
decay to 2.5e-6 (auto-scaled from 30k to the 20k we run). 20,000 steps at batch
6 is 120,000 samples over 177,193 frames, i.e. ~0.68 epochs.

**Checkpoints.** `runs/pi05_phantom_expert_v1/checkpoints/NNNNNN/`, with
`last` a symlink to the newest. Each holds `pretrained_model/` (weights,
policy config, train config **and the saved pre/post-processor pipeline** --
this is what the deploy adapter loads) plus `training_state/` for resuming.

Measured at step 2500: **9.1 GB per checkpoint**. lerobot 0.4.4 prunes nothing
and `/` had 45 GB free after that first one, so the eight checkpoints this run
wants (73 GB) do not fit -- untouched, it hits ENOSPC around step 15000.

Two scripts sit on compute2, and only one of them is running:

* `~/lerobot/archive_checkpoints.sh <train_pid> <min_free_GB>` -- ARMED. When
  `/` drops below 14 GB it RELOCATES the oldest checkpoint to the box's second
  disk (`/media/isr-lab-4/Main/pi05_phantom_expert_v1_checkpoints`, 307 GB
  free) and nothing else. Nothing is deleted; `mv` unlinks the source only
  after the copy lands, so every checkpoint stays readable, just not on `/`.
  It never touches the newest checkpoint or whatever `last` resolves to, and
  it exits when the training pid does.
* `~/lerobot/prune_checkpoints.sh <run_dir> <keep_n> [--apply]` -- NOT armed,
  and deliberately so. It genuinely deletes, which needs the operator's
  explicit say-so; without `--apply` it only prints what it would remove. Same guards.

### What the deploy adapter gets, and the one thing it must fix

Read off the step-2500 checkpoint, so this is what it will see:

| | value |
| --- | --- |
| `config.image_features` | exactly `['observation.images.scene']` -- no `--rename_map`, the adapter's key matches the checkpoint's |
| saved `rename_observations_processor` | empty map, so nothing is silently renamed under the adapter |
| action feature | `(7,)` -- passes the adapter's action-dim contract |
| `chunk_size` / `n_action_steps` / `num_inference_steps` | 50 / 16 / 10 |
| normalisation | `policy_preprocessor_step_2_normalizer_processor.safetensors` and the matching unnormalizer, both written from THIS dataset's stats |

**The tokenizer path is absolute and local to compute2.** The saved
preprocessor carries
`tokenizer_processor.tokenizer_name = /home/isr-lab-4/lerobot/paligemma_tokenizer`,
which will not exist on the deploy box. Either ship that directory to the same
path beside the checkpoint, or override the step when the adapter builds its
pipelines -- `make_pre_post_processors` already honours
`preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": <path>}}`
next to the `device_processor` override it passes today.

**Do not gate on the declared state shape.** `config.json` still says
`observation.state` is `(32,)`, inherited from pi05_base: pi0.5 pads state to
`max_state_dim` 32 inside `Pi05PrepareStateTokenizerProcessorStep`. The real
input is our `(7,)` vector and the saved normalizer stats are 7-dimensional.
The action feature was rewritten to 7 by training; the state feature was not.

### Offline evaluation

`tools/pi05_offline_eval.py` scores a checkpoint on the val split with the
same `endpoint_err_mm` `tools/terminal_eval.py` reports for our students --
the distance between predicted and demonstrated TCP after integrating 16 pose
deltas -- so a pi0.5 row can sit next to a student row without a footnote.

```
python tools/pi05_offline_eval.py \
    --ckpt ~/lerobot/runs/pi05_phantom_expert_v1/checkpoints/last/pretrained_model \
    --data ~/lerobot/data/phantom_pi05/val \
    --horizon 16 --anchor close --max-windows 200 --out pi05_eval.json
```

It runs the DEPLOY path, not a shortcut: preprocessor,
`predict_action_chunk(num_steps=nfe)`, then the postprocessor once per chunk
step exactly as `phantom/inference/lerobot_policy.py` applies it. Windows are
anchored in the 1.5 s before each episode's first gripper close (`--anchor
uniform` spreads them over the whole episode and is the easier metric -- quote
which one). Every row also carries `zero_endpoint_err_mm`, the same metric for
a policy that proposes no motion at all: that is the scale reference a
fine-tune has to beat before any of the numbers mean anything.

Exercised end to end against the step-2500 checkpoint (CPU, 2 windows, while
the GPU stayed on training). It loads the checkpoint and its saved pipelines,
runs the chunk, and reports 50.2 mm endpoint against 53.7 mm for the no-motion
reference -- i.e. 0.94x, a model that has learnt essentially nothing yet at
step 2500 out of 20000. That is the expected reading this early and it is the
point of quoting the reference: the tooling is verified, the number is not yet
a result. Re-run it on the finished checkpoint with the GPU free.

## Deploy adapter

`phantom/inference/lerobot_policy.py` (`LeRobotPolicy`) duck-types
`PhantomPolicy`; `phantom/scripts/lerobot_server.py` serves it over the same
protocol, ownership rules and `--probe` contract as
`phantom.scripts.policy_server` — it reuses `PolicyServer` itself, only the
policy object differs. Everything from `SnapshotBuilder` down (executor,
safety monitor, recorder, governor) is unchanged.

### Launch

Terminal A, once per model:

```
.venv/bin/python -m phantom.scripts.lerobot_server \
    --ckpt runs/pi05_phantom_ft --port 7790 \
    --hardware configs/hardware.nuc.yaml \
    --policy-type pi05 --device cuda \
    --task "pick up the egg" --action-space delta --image-size 224
```

It prints `READY ckpt=... port=7790` when warm. `--probe --port 7790` prints
`<ckpt> <idle|busy> <sha12>` and exits 0/1/2 exactly like the phantom server,
so `PICK.sh` detects it with no change.

Terminal B, per episode batch:

```
run_deploy --system student --policy-server 127.0.0.1:7790 \
    --task egg --text "pick up the egg" \
    --hardware configs/hardware.nuc.yaml \
    --nfe 10 --max-play-steps 9 --grip-play-steps 9
```

`--nfe` is live: it is forwarded to pi05's flow-matching sampler as
`num_steps` (the checkpoint's default is `config.num_inference_steps`, 10), so
it is the latency lever if a replan is too slow. A policy whose chunk call does
not accept it logs a warning once and records `nfe_applied: false` in the
trace, so a run can never claim a denoise budget the model ignored.

`--system student` because this policy reads no tactile stream: it selects the
sensor-free `SnapshotBuilder` path. The server's checkpoint is the one that
runs; `--ckpt` on the `run_deploy` line only warns when it disagrees.

### Flags that MUST be off

| flag | why | enforced |
| --- | --- | --- |
| `--parity-fixes` | there is no `prev_chunk` conditioning and no ACC package to realign. | yes |
| `--k-seeds > 1` | one chunk per replan; there is nothing to select between. | yes |
| `--drop-video` | the scene image is this policy's only observation. | yes |
| `--terminal-veto` | no ACC head. `p_evt` is `zeros(5)`, so `p_contact = 1.0` and the close-mask allows every close while logging `close_allowed` — an arm tagged `veto:on` that vetoes nothing. | **no** |
| `--persistent-noise` | inert (no held sampling noise); leave at the default. | n/a |

The three enforced flags travel to the server inside `configure`, which does
`setattr` on the policy, so the adapter refuses them on the attribute itself:
the attach fails and `run_deploy` exits 3 before the arm is touched. Sending
the defaults stays silent, so an ordinary launch is unaffected.

`--terminal-veto` is not in `remote.CONFIGURABLE` and never reaches the server,
so it stays operator discipline. Leave it off.

`--max-play-steps` / `--grip-play-steps` stay ON: the chunk-tail caps are
executor-side and protect this policy exactly as they protect ours.

### What the adapter fills into `Plan`

| field | value |
| --- | --- |
| `actions` | `(16, 7)` denormalised base-frame deltas, truncated or padded from the policy chunk |
| `action_times` | `obs.t + latency + k/10 s` |
| `t0_pose` | the measured TCP pose passed to `replan` |
| `sigma` | `zeros(16)`. `GovernorConfig.sigma_lo >= 0` is validated, so zero is provably full playback speed for any legal governor config |
| `gate`, `p_evt` | `0.0`, `zeros(5)` — logged only, and the veto they feed is refused above |
| `cpk` | `None` (no contact package; `_invalidate_cpk` is a no-op) |
| `diag` | `policy`, `policy_class`, `action_space`, `chunk_steps`, `padded_steps`, `nfe`, `nfe_applied` |

**Horizon.** pi05's native chunk (50) is truncated to `chunk_horizon` (16). A
SHORTER chunk is padded: `--pad-mode hold` (default) pads with zero pose-delta
and the last commanded aperture; `--pad-mode repeat` repeats the last row.
Hold is the default because a repeated delta extrapolates motion the policy
never proposed, into the chunk tail the executor is most likely still playing
when the next replan lands.

**Absolute-pose policies.** `--action-space absolute` reads the six pose
channels as absolute base-frame poses and differences them with
`phantom.data.derived.pose_delta`, chaining from the MEASURED pose, so the
executor's `t0_pose + cumsum(deltas)` playback reproduces the predicted
trajectory with the rotvec continuity guard applied. The gripper column is an
absolute aperture command in both modes.

### MODELS.tsv row

`tools/rig/MODELS.tsv` is the rig menu (`label`, `ckpt`, `note`, optional
`system`). One tab-separated row per model; the pi0.5 baseline is a
sensor-free arm:

```
pi05_ft	runs/pi05_phantom_ft	pi0.5 BASELINE (LeRobot) - serve with lerobot_server, port 7790; NO --terminal-veto/--parity-fixes/--k-seeds	student
```

Do not add it until the fine-tune exists; `SERVE.sh` starts
`phantom.scripts.policy_server` per slot and would need the lerobot server
started by hand on that slot's port (`7776 + slot`).

### Verified vs. unverified

`tests/test_lerobot_adapter.py` pins the whole adapter against a stub policy,
with no `lerobot` import and no GPU: observation mapping, delta and absolute
decoding, rotvec wrap, truncation and both padding modes, the pre/post
processor pipelines being applied around inference (post-processor once per
chunk step), the `--nfe` passthrough and its loud degradation, zero-sigma full
speed, refused flags, the checkpoint contract check, the reset/seed path, and
an `info` + `replan` round trip over a real localhost policy-server wire whose
plans the real `ChunkExecutor` accepts.

Every LeRobot name the adapter binds to was checked against an installed
`lerobot 0.4.4`, by introspection and not from memory:
`get_policy_class("pi05")` resolves `PI05Policy`, which has `from_pretrained`,
`reset`, `select_action` and `predict_action_chunk`; the chunk call takes
`**kwargs` and so accepts `num_steps`, while `select_action` does not;
`make_pre_post_processors(policy_cfg, pretrained_path=...)` is the pipeline
factory. `PI05Config` defaults: `num_inference_steps` 10, `chunk_size` 50,
`n_action_steps` 50, `image_resolution` (224, 224), `max_state_dim` 32. The
224 default here therefore matches pi05's own resolution exactly, so its
internal `resize_with_pad` is a no-op on our frames.

Still needs the real checkpoint: that the fine-tune's saved processor
statistics are the ones we expect, the exported camera key matching
`config.image_features`, and the per-replan latency on the 5090.
