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
    --max-play-steps 9 --grip-play-steps 9
```

`--system student` because this policy reads no tactile stream: it selects the
sensor-free `SnapshotBuilder` path. The server's checkpoint is the one that
runs; `--ckpt` on the `run_deploy` line only warns when it disagrees.

### Flags that MUST be off

| flag | why |
| --- | --- |
| `--terminal-veto` | no ACC head. `p_evt` is `zeros(5)`, so `p_contact = 1.0` and the close-mask allows every close while logging `close_allowed` — an arm tagged `veto:on` that vetoes nothing. `run_deploy.assert_acc_head` refuses this for phantom checkpoints; `LeRobotPolicy.assert_deploy_flags` refuses it here. |
| `--parity-fixes` | there is no `prev_chunk` conditioning and no ACC package to realign. |
| `--k-seeds > 1` | one chunk per replan; there is nothing to select between. |
| `--drop-video` | the scene image is this policy's only observation. |
| `--persistent-noise` | inert (no held sampling noise); leave at the default. |

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
| `diag` | `policy`, `policy_class`, `action_space`, `chunk_steps`, `padded_steps` |

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

`tests/test_lerobot_adapter.py` pins the whole adapter against a stub policy —
observation mapping, delta and absolute decoding, rotvec wrap, truncation and
both padding modes, zero-sigma full speed, refused flags, the reset/seed path,
and an `info` + `replan` round trip over a real localhost policy-server wire
whose plans the real `ChunkExecutor` accepts. What still needs the real
checkpoint: the LeRobot entry point actually present on the installed version
(`predict_action_chunk`, with a per-step `select_action` fallback), the
normalisation statistics baked into the checkpoint, and the per-replan latency
on the 5090.
