# pi0.5 baseline

An external VLA baseline (Physical Intelligence pi0.5, via LeRobot) run on the
PHANTOM rig through OUR deploy loop, so its numbers are comparable with the
student's. Two halves, built independently against one shared contract:

* **Export** — PHANTOM episodes to a LeRobot dataset, and the fine-tune.
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

The LeRobot names the adapter binds to were read from the real
`lerobot 0.4.4` wheel: `get_policy_class("pi05")`, `PI05Policy.from_pretrained`
/ `reset` / `select_action` / `predict_action_chunk`,
`make_pre_post_processors`, and `PI05Config.num_inference_steps`.

Still needs the real checkpoint: that the fine-tune's saved processor
statistics are the ones we expect, the exported camera key matching
`config.image_features`, and the per-replan latency on the 5090.
