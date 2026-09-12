"""LeRobotPolicy: an external LeRobot policy (target: pi05) behind the exact
`replan(obs, prev_plan, tcp_pose) -> Plan` surface `phantom/deploy/` drives.

Why an adapter and not a second deploy loop: the baseline is only a baseline if
it runs through OUR executor, OUR safety monitor and OUR recorder, so the
20-30 mm numbers it produces are comparable with the student's. Everything from
`SnapshotBuilder` down is therefore unchanged; only the object that turns an
`ObsSnapshot` into a `Plan` differs.

Observation contract (SHARED with the dataset exporter — the two must agree
byte-for-byte or the policy sees a distribution it never trained on):

    observation.images.scene  (1, 3, S, S) float32 in [0, 1], RGB, CHW,
                              `phantom.data.windows.bilinear_resize` of the
                              scene frame to S x S (S = 224 by default).
                              NOTE this SQUASHES 640x480 to square; the
                              exporter must squash identically, not crop.
                              (pi05 then `resize_with_pad`s to its own
                              `config.image_resolution` and rescales to
                              [-1, 1] internally — `_preprocess_images`.)
    observation.state         (1, 7) float32
                              [tcp x, y, z, rx, ry, rz (rotvec, BASE frame),
                               gripper aperture]
                              read out of `ObsSnapshot.ur_state` at the same
                              offsets `PlannerLoop.run` uses.
    task                      str, the episode instruction.

VERIFIED against lerobot 0.4.4 (`lerobot/utils/control_utils.py:predict_action`,
`lerobot/async_inference/policy_server.py:_predict_action_chunk`): a LeRobot
policy is NOT called on raw observations. Normalisation, pi05's state
DISCRETISATION into the PaliGemma prompt (`Pi05PrepareStateTokenizerProcessorStep`)
and tokenisation all live in a `PolicyProcessorPipeline` saved WITH the
checkpoint, and unnormalisation lives in the matching post-processor. So the
call chain here is exactly LeRobot's:

    raw numpy -> torch -> preprocessor -> predict_action_chunk
              -> postprocessor (per chunk step) -> numpy

`predict_action_chunk` for pi05 reads only the image keys and the tokenised
prompt; the state reaches the model INSIDE that prompt, which is why the
pipelines are load-bearing rather than a convenience. Skipping them
(`--no-processors`) feeds the model normalised-space garbage and is offered
only for a policy that genuinely ships no pipeline.

Observation-QUEUE policies (Diffusion Policy, ACT with n_obs_steps > 1)
-----------------------------------------------------------------------
pi05 and X-VLA are single-frame: `n_obs_steps = 1`, and their `_queues` hold
only ACTION, so one snapshot is one call. Diffusion Policy is not — it
conditions on the last `n_obs_steps` observations, and `predict_action_chunk`
does NOT build that history: it STACKS whatever is already in
`policy._queues` (lerobot 0.4.4 `modeling_diffusion.py:94`). Its camera
features are also stacked into the single key `observation.images` BEFORE
queueing, by `select_action`, not by the model.

So for such a policy this adapter keeps its OWN per-episode history of the
last `n_obs_steps` post-processor batches and, every replan, clears the
policy's observation queues and refills them oldest-first — repeating the
first frame while the history is short, which is exactly lerobot's
`populate_queues` cold start. `reset_episode` drops the history, so an
episode never conditions on the previous episode's frames.

CAVEAT, and it is a real one: at training time those `n_obs_steps` frames are
CONSECUTIVE dataset frames, 1/fps apart (0.1 s here). On the rig the adapter
only sees an observation when `PlannerLoop` replans, so the older frame is one
REPLAN old, not one control step old. `Plan.diag["obs_dt_s"]` records the
actual spacing so a trace never hides it.

The ground-truth ACTION key is dropped from every batch before inference: a
queue policy has an ACTION queue too, and a batch carrying `action` would
prime it and make the policy replay the demonstration instead of predicting.

Action contract: the post-processed chunk is (T, 7) of
`[dx, dy, dz, drx, dry, drz, grip]` at `hw.control.action_rate_hz` (10 Hz), in
metres / radians / aperture units. `--action-space absolute` instead reads the
six pose channels as ABSOLUTE base-frame poses and differences them with
`phantom.data.derived.pose_delta`, the same rotvec-continuity convention the
executor's cumulative-sum playback assumes.

What this policy does NOT produce, and what that forbids:

  * no ACC head -> `gate = 0.0`, `p_evt = zeros(5)`. `p_evt[none] = 0` makes
    `p_contact = 1.0`, so the terminal veto's close-mask would allow EVERY
    close while logging `close_allowed` — a veto that measures nothing
    (`run_deploy.assert_acc_head` refuses exactly this). `--terminal-veto`
    MUST be off; `LeRobotPolicy.assert_deploy_flags` enforces it server-side.
  * no contact package -> `cpk = None`; `_invalidate_cpk` is a no-op.
  * no predictive uncertainty -> `sigma = zeros(H)`. `GovernorConfig.sigma_lo`
    is validated `>= 0`, so a zero sigma is PROVABLY full playback speed for
    any legal governor config (`SpeedGovernor.scale`).
  * no K-seed sampling, no ACC self-anticipation, no `prev_chunk`
    conditioning -> `--k-seeds > 1`, `--parity-fixes` and `--drop-video` are
    inert here and are rejected rather than silently ignored.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from phantom.data import derived as dv
from phantom.data.windows import bilinear_resize
from phantom.inference.policy import ObsSnapshot, Plan

log = logging.getLogger(__name__)

DEFAULT_IMAGE_KEY = "observation.images.scene"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_IMAGE_SIZE = 224
ACTION_SPACES = ("delta", "absolute")
PAD_MODES = ("hold", "repeat")

# lerobot.utils.constants, inlined so this module stays importable without
# lerobot (verified against lerobot 0.4.4: ACTION = "action",
# OBS_IMAGES = "observation.images").
LR_ACTION_KEY = "action"
LR_IMAGES_KEY = "observation.images"

# Deploy levers this policy structurally cannot honour. Passing them would
# label an arm with a condition it never ran (the failure mode `cond_tags`
# and `assert_acc_head` exist to prevent).
UNSUPPORTED_FLAGS = {
    "terminal_veto": "no ACC head: p_evt is zeros(5), p_contact 1.0, so every "
                     "close would be allowed while the trace logged "
                     "`close_allowed`",
    "parity_fixes": "no prev_chunk / ACC conditioning to realign",
    "drop_video": "the scene image is this policy's only observation",
}


def _denoise_steps(cfg) -> int:
    """The denoise/integration steps the policy's own config asks for.

    pi05 / Diffusion Policy: `num_inference_steps`, and for Diffusion Policy a
    None there means `num_train_timesteps` (lerobot 0.4.4
    `DiffusionModel.__init__`). X-VLA: `num_denoising_steps`. 0 when the policy
    declares none, so a trace never invents a number."""
    if cfg is None:
        return 0
    n = getattr(cfg, "num_inference_steps", None)
    if n is None:
        n = getattr(cfg, "num_denoising_steps", None)
    if n is None:
        n = getattr(cfg, "num_train_timesteps", None)
    try:
        return int(n)
    except (TypeError, ValueError):
        return 0


def _accepts_kwarg(fn, name: str) -> bool:
    """True when `fn` takes `name` by keyword, explicitly or via **kwargs.
    An unreadable signature (a C extension) counts as NOT accepting: better a
    lever that visibly does nothing than a call that dies mid-episode."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD
                                 for p in params.values())


class _EpisodeReset:
    """The `policy.rf` shim `PolicyServer` reaches through.

    `PolicyServer._handle_owned` seeds with
    `self.policy.rf._gen = torch.Generator().manual_seed(seed)` and then calls
    `self.policy.rf.reset_episode_noise()`. LeRobot policies keep an internal
    action QUEUE rather than held noise, so `reset_episode_noise` clears that
    queue, and the seed is applied to the process RNG the policy samples from
    (pi0/pi05 flow-matching draws its prior from the global torch generator).

    It also drops the adapter's observation history, and that placement is
    load-bearing: `PolicyServer._handle_owned` serves the wire's
    `("reset_episode", seed)` by calling `self.policy.rf.reset_episode_noise()`
    directly, NOT `LeRobotPolicy.reset_episode`. Clearing the history only in
    the latter would leave a queue policy conditioning the first chunk of
    episode N+1 on the LAST frame of episode N.
    """

    def __init__(self, policy, owner=None):
        self._policy = policy
        self._owner = owner
        self._gen_value = None
        self.resets = 0

    @property
    def _gen(self):
        return self._gen_value

    @_gen.setter
    def _gen(self, gen) -> None:
        self._gen_value = gen
        seed = None
        try:
            seed = int(gen.initial_seed())
        except Exception:            # noqa: BLE001 — a non-Generator is not fatal
            log.warning("episode seed: %r is not a torch.Generator — not seeding", gen)
        if seed is not None:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            log.info("episode seed %d applied to the torch RNG", seed)

    def reset_episode_noise(self) -> None:
        self.resets += 1
        clear = getattr(self._owner, "_clear_obs_history", None)
        if callable(clear):
            clear()
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()


class LeRobotPolicy:
    """Duck-types `PhantomPolicy` for `PlannerLoop` / `PolicyServer`."""
    policy_kind = "lerobot"

    def __init__(self, policy, hw, *, task_text: str = "",
                 action_space: str = "delta",
                 image_size: int = DEFAULT_IMAGE_SIZE,
                 image_key: str = DEFAULT_IMAGE_KEY,
                 state_key: str = DEFAULT_STATE_KEY,
                 pad_mode: str = "hold", device: str = "cpu",
                 use_task_key: bool = True,
                 preprocessor=None, postprocessor=None,
                 rename_map: dict | None = None):
        assert action_space in ACTION_SPACES, \
            f"action_space must be one of {ACTION_SPACES}, got {action_space!r}"
        assert pad_mode in PAD_MODES, \
            f"pad_mode must be one of {PAD_MODES}, got {pad_mode!r}"
        self.policy = policy
        # LeRobot's PolicyProcessorPipelines, loaded from the checkpoint.
        # `preprocessor` carries the dataset normalisation stats, pi05's
        # state->prompt discretisation and the PaliGemma tokeniser;
        # `postprocessor` carries the matching unnormalisation.
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.hw = hw
        self.action_space = action_space
        self.image_size = int(image_size)
        self.image_key = image_key
        self.state_key = state_key
        self.pad_mode = pad_mode
        self.device = device
        self.use_task_key = bool(use_task_key)
        # observation-key rename applied to OUR batch before the checkpoint's
        # own pipeline sees it. Needed only when the pipeline does NOT already
        # carry the rename it was trained with (`--no-processors`), because
        # `make_pre_post_processors(pretrained_path=...)` restores the saved
        # `rename_observations_processor` step verbatim.
        self.rename_map = dict(rename_map or {})
        self.rf = _EpisodeReset(policy, owner=self)
        # SnapshotBuilder mirrors this off the server's `info`; a LeRobot
        # policy never saw a wrench baseline, so the snapshot must not
        # subtract one.
        self.wrench_baseline_rows = 0

        self.H = int(hw.control.chunk_horizon)
        self.A = int(hw.control.action_dim)
        self.rate = float(hw.control.action_rate_hz)
        self.dof = int(hw.arm.dof)

        cfg = getattr(policy, "config", None)
        # how many past observations this policy conditions on. 1 for pi05 and
        # X-VLA; 2 for our Diffusion Policy checkpoint.
        self.n_obs_steps = max(1, int(getattr(cfg, "n_obs_steps", 1) or 1))
        # the camera keys the checkpoint declares, in the order the model
        # stacks them — this order IS part of the trained contract.
        self.image_features = list(getattr(cfg, "image_features", None) or [])
        # the policy's own denoise budget, for the trace. NOT the same as
        # `--nfe`: Diffusion Policy exposes no per-call override, and a None
        # `num_inference_steps` means it runs `num_train_timesteps` DDPM steps
        # (lerobot 0.4.4 `modeling_diffusion.py:204`).
        self.denoise_steps = _denoise_steps(cfg)
        # per-episode observation history for a queue policy, oldest first
        self._obs_hist: list[dict] = []
        self._last_obs_t: float | None = None
        self._obs_dt: float = 0.0

        # --- the PhantomPolicy attribute surface `remote.CONFIGURABLE`
        # reconfigures and `run_deploy`'s cond_tags read. Only `task_text` is
        # live; the rest exist so `configure` and the tag line do not crash,
        # and `assert_deploy_flags` refuses the ones that would LIE.
        self.task_text = task_text
        # pi05's flow-matching denoise steps are the direct analogue of our
        # `nfe`, and `predict_action_chunk` forwards `num_steps` down to
        # `sample_actions` — so --nfe stays a real latency lever and the
        # `nfe{...}` cond_tag stays honest. `_nfe_kwarg` records whether this
        # policy actually accepts it (probed on the first replan).
        self.nfe = getattr(getattr(policy, "config", None),
                           "num_inference_steps", None)
        self._nfe_kwarg: bool | None = None
        self.guidance = 1.0
        self.k_seeds = 1
        self.parity_fixes = False
        self.persistent_noise = False
        self.drop_video = False
        self.close_p = 0.5

    # -- live refusal of the levers this policy cannot honour -----------
    # `RemotePolicy.__init__` sends run_deploy's whole lever set through
    # `configure`, which does `setattr(policy, k, v)` server-side. Three of
    # them are visible there, so they are refused at ATTACH rather than
    # silently ignored: the error travels back as ("err", traceback),
    # `RemotePolicy._call` raises, and `run_deploy --policy-server <addr>`
    # exits 3 before the arm is touched. run_deploy always sends the DEFAULTS
    # too, so only the unsupported VALUE raises.
    #
    # `--terminal-veto` is NOT in `remote.CONFIGURABLE` and never reaches the
    # server, so it stays operator discipline (docs/pi05_baseline.md) plus
    # `assert_deploy_flags` for any caller that does see the flags.

    def _refuse(self, flag: str) -> None:
        raise RuntimeError(
            f"a LeRobot baseline cannot honour --{flag.replace('_', '-')}: "
            f"{UNSUPPORTED_FLAGS[flag]}. Drop it from the run_deploy line — "
            "running with it set would tag the arm with a condition it never "
            "ran.")

    @property
    def parity_fixes(self) -> bool:
        return False

    @parity_fixes.setter
    def parity_fixes(self, value) -> None:
        if value:
            self._refuse("parity_fixes")

    @property
    def drop_video(self) -> bool:
        return False

    @drop_video.setter
    def drop_video(self, value) -> None:
        if value:
            self._refuse("drop_video")

    @property
    def k_seeds(self) -> int:
        return 1

    @k_seeds.setter
    def k_seeds(self, value) -> None:
        if int(value or 1) > 1:
            raise RuntimeError(
                "a LeRobot baseline cannot honour --k-seeds > 1: this policy "
                "samples one chunk per replan; there is nothing to select "
                "between. Drop it from the run_deploy line.")

    def assert_deploy_flags(self, **flags) -> None:
        """Refuse a whole flag set at once, for a caller that can see it all
        (including `--terminal-veto`, which never reaches the server)."""
        bad = [f"--{k.replace('_', '-')}: {why}"
               for k, why in UNSUPPORTED_FLAGS.items() if flags.get(k)]
        if int(flags.get("k_seeds") or 1) > 1:
            bad.append("--k-seeds > 1: this policy samples one chunk per "
                       "replan; there is nothing to select between")
        if bad:
            raise RuntimeError(
                "a LeRobot baseline cannot honour these deploy flags:\n  "
                + "\n  ".join(bad)
                + "\nDrop them from the run_deploy line — running with them "
                  "set would tag the arm with a condition it never ran.")

    def _clear_obs_history(self) -> None:
        """Forget every frame of the finished episode. Called from
        `_EpisodeReset.reset_episode_noise`, which is the ONE path both the
        local runtime and the policy-server wire go through."""
        self._obs_hist = []
        self._last_obs_t = None
        self._obs_dt = 0.0

    def reset_episode(self) -> None:
        self.rf.reset_episode_noise()

    # -- observation ---------------------------------------------------
    def observation(self, obs: ObsSnapshot) -> dict:
        """The (numpy) LeRobot batch for one snapshot — the exporter's target.

        Kept public and torch-free so `tests/` and the exporter can assert the
        SAME mapping without importing lerobot or touching a GPU."""
        ur = np.asarray(obs.ur_state, dtype=np.float32).ravel()
        need = 2 * self.dof + 6 + 2
        if ur.size < need:
            raise ValueError(
                f"ur_state has {ur.size} entries, need >= {need} "
                f"(q, qd, tcp_pose, tcp_speed, gripper) for a {self.dof}-dof arm")
        tcp = ur[2 * self.dof:2 * self.dof + 6]      # same slice as PlannerLoop.run
        grip = ur[-2]                                # same index as `grip_now`
        state = np.concatenate([tcp, [grip]]).astype(np.float32)

        rgb = np.asarray(obs.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"scene frame must be (H, W, 3), got {rgb.shape}")
        S = self.image_size
        img = bilinear_resize(rgb.astype(np.float32), (S, S)) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
        batch = {self.image_key: img[None],         # (1, 3, S, S)
                 self.state_key: state[None]}       # (1, 7)
        # `--rename-map`: land the scene camera under the key THIS checkpoint
        # was trained on (X-VLA: observation.images.image). Applied to
        # observation keys only, never to `task`.
        if self.rename_map:
            batch = {self.rename_map.get(k, k): v for k, v in batch.items()}
        # a plain str, exactly as LeRobot's own `predict_action` passes it;
        # `AddBatchDimensionComplementaryDataStep` wraps it into a
        # one-element list inside the pipeline
        batch["task"] = self.task_text
        return batch

    def _to_torch(self, batch: dict) -> dict:
        import torch
        out = {}
        for k, v in batch.items():
            if k == "task":
                if self.use_task_key:
                    out[k] = v
                continue
            out[k] = torch.from_numpy(np.asarray(v)).to(self.device)
        return out

    # -- observation history (queue policies only) ---------------------
    def _obs_queues(self) -> dict | None:
        """The policy's OBSERVATION queues, or None when it keeps none.

        lerobot policies all expose `_queues`, but pi05's and X-VLA's hold the
        ACTION deque ALONE — they are single-frame models and manage nothing we
        must prime. Only a policy with a non-ACTION queue (Diffusion Policy:
        `observation.state` + `observation.images`) needs the history dance, so
        the presence of such a queue IS the test."""
        q = getattr(self.policy, "_queues", None)
        if not isinstance(q, dict):
            return None
        obs = {k: v for k, v in q.items() if k != LR_ACTION_KEY}
        return obs or None

    def _stack_cameras(self, batch: dict) -> dict:
        """Add the single `observation.images` key a Diffusion-Policy queue is
        named after: its cameras stacked on a new axis at dim=-4.

        `DiffusionPolicy.select_action` does this BEFORE queueing and
        `predict_action_chunk` only stacks the queue, so an adapter that skips
        it hands `generate_actions` a batch with no image key at all. The
        stacking order is `config.image_features`, which is the order the
        encoder was trained with."""
        import torch
        keys = [k for k in self.image_features if k in batch]
        if not keys:
            keys = sorted(k for k in batch
                          if isinstance(k, str)
                          and k.startswith(LR_IMAGES_KEY + "."))
            if keys:
                log.warning("policy declares no image_features; stacking %s "
                            "into %s in SORTED order — verify it matches the "
                            "order the checkpoint was trained with",
                            keys, LR_IMAGES_KEY)
        if not keys:
            return batch
        out = dict(batch)
        out[LR_IMAGES_KEY] = torch.stack([out[k] for k in keys], dim=-4)
        return out

    def _push_obs(self, batch: dict) -> None:
        """Remember this (post-processor) batch, keeping n_obs_steps frames."""
        self._obs_hist.append(batch)
        if len(self._obs_hist) > self.n_obs_steps:
            del self._obs_hist[:-self.n_obs_steps]

    def _prime_queues(self, queues: dict) -> None:
        """Refill the policy's observation queues from OUR history.

        Deliberately a REFILL from a cleared deque rather than an append, so a
        replan's conditioning depends only on `self._obs_hist` and never on how
        many times the policy happened to be called before. The short-history
        rule is lerobot 0.4.4 `populate_queues`': repeat the oldest frame until
        the queue is full, which is how `select_action` starts an episode."""
        for key, q in queues.items():
            frames = [f[key] for f in self._obs_hist if key in f]
            if not frames:
                log.warning("policy queue %r has no matching batch key — it "
                            "will be left as the policy built it", key)
                continue
            maxlen = q.maxlen or len(frames)
            frames = frames[-maxlen:]
            q.clear()
            for _ in range(maxlen - len(frames)):
                q.append(frames[0])
            for f in frames:
                q.append(f)

    # -- action --------------------------------------------------------
    def _call_chunk(self, fn, batch):
        """`predict_action_chunk`, passing `--nfe` through as `num_steps` when
        the policy's signature accepts it.

        Decided by inspecting the signature, not by catching TypeError: a
        TypeError raised INSIDE the model is a real bug and must not be
        silently retried as "this policy takes no kwargs". pi05's chunk call is
        `(batch, **kwargs)` and forwards straight into `sample_actions`, whose
        `num_steps` defaults to `config.num_inference_steps`."""
        want = int(self.nfe) if self.nfe else None
        if want is None:
            return fn(batch)
        if self._nfe_kwarg is None:
            self._nfe_kwarg = _accepts_kwarg(fn, "num_steps")
            if not self._nfe_kwarg:
                log.warning("%s.predict_action_chunk takes no num_steps — "
                            "--nfe %s is IGNORED and the checkpoint's own "
                            "denoise step count runs",
                            type(self.policy).__name__, want)
        return fn(batch, num_steps=want) if self._nfe_kwarg else fn(batch)

    def _raw_chunk(self, batch: dict) -> np.ndarray:
        """(T, A) numpy chunk, through LeRobot's own pre/post pipelines.

        Mirrors `lerobot.async_inference.policy_server._predict_action_chunk`:
        the pre-processor normalises / tokenises, `predict_action_chunk` is the
        chunk-native call every LeRobot policy exposes, and the post-processor
        unnormalises ONE chunk step at a time — it is written for a
        `(B, action_dim)` action, not for a `(B, T, action_dim)` chunk.

        `select_action` is the fallback for a policy without the chunk call: it
        pops one action per call off the policy's internal queue, so H calls on
        the same observation reproduce the head of the chunk.
        """
        import torch
        with torch.no_grad():
            if self.preprocessor is not None:
                batch = self.preprocessor(batch)
            # The ground-truth ACTION key must never reach the policy: a queue
            # policy would prime its ACTION deque from it and replay the
            # demonstration. Our own `observation()` never emits it, so on the
            # pi05 path this is a no-op copy.
            batch = {k: v for k, v in batch.items() if k != LR_ACTION_KEY}
            fn = getattr(self.policy, "predict_action_chunk", None)
            if callable(fn):
                queues = self._obs_queues()
                if queues is not None:
                    if LR_IMAGES_KEY in queues:
                        batch = self._stack_cameras(batch)
                    self._push_obs(batch)
                    # `reset()` rebuilds the deques, so re-read them and prime
                    # from scratch — the conditioning is then a pure function
                    # of the history.
                    reset = getattr(self.policy, "reset", None)
                    if callable(reset):
                        reset()
                        queues = self._obs_queues() or queues
                    self._prime_queues(queues)
                out = self._call_chunk(fn, batch)
                if out.ndim == 1:            # (A,)
                    out = out[None, None]
                elif out.ndim == 2:          # (T, A) — no batch dim
                    out = out[None]
            else:
                sel = getattr(self.policy, "select_action", None)
                if not callable(sel):
                    raise AttributeError(
                        f"{type(self.policy).__name__} exposes neither "
                        "`predict_action_chunk` nor `select_action` — this is "
                        "not a LeRobot PreTrainedPolicy")
                rows = [sel(batch) for _ in range(self.H)]
                rows = [r[None] if r.ndim == 1 else r for r in rows]   # (B, A)
                out = torch.stack(rows, dim=1)                          # (B,T,A)
            if self.postprocessor is not None:
                out = torch.stack([self.postprocessor(out[:, i, :])
                                   for i in range(out.shape[1])], dim=1)
            return np.asarray(out[0].detach().float().cpu().numpy(),
                              dtype=np.float64)

    def _to_deltas(self, chunk: np.ndarray, tcp_pose: np.ndarray) -> np.ndarray:
        """(T, 7) policy output -> (T, 7) base-frame deltas + gripper."""
        if chunk.ndim != 2:
            raise ValueError(f"policy chunk must be 2-D (T, A), got {chunk.shape}")
        if chunk.shape[1] < self.A:
            raise ValueError(
                f"policy chunk has {chunk.shape[1]} action dims, need >= {self.A} "
                f"([dx dy dz drx dry drz grip]) — the exported dataset's action "
                "layout does not match the deploy contract")
        if chunk.shape[1] > self.A:
            log.warning("policy chunk has %d action dims; using the first %d",
                        chunk.shape[1], self.A)
        chunk = chunk[:, :self.A]
        if self.action_space == "delta":
            return chunk
        # absolute: chain pose_delta from the MEASURED pose, exactly the
        # convention `SnapshotBuilder.prev_chunk_from_history` and the
        # recorder use. The executor then replays
        # t0_pose + cumsum(deltas) == the absolute trajectory, rotvec
        # continuity guard included.
        poses = np.vstack([np.asarray(tcp_pose, dtype=np.float64).ravel()[:6],
                           chunk[:, :6]])
        deltas = np.stack([dv.pose_delta(poses[k], poses[k + 1])
                           for k in range(chunk.shape[0])])
        return np.concatenate([deltas, chunk[:, 6:7]], axis=1)

    def _fit_horizon(self, actions: np.ndarray) -> tuple[np.ndarray, int]:
        """Truncate to H, or pad up to H. Returns (actions, n_padded)."""
        T = actions.shape[0]
        if T == 0:
            raise ValueError("policy returned an empty action chunk")
        if T >= self.H:
            return np.array(actions[:self.H], dtype=np.float64), 0
        pad = self.H - T
        last = actions[-1]
        if self.pad_mode == "repeat":
            rows = np.repeat(last[None], pad, axis=0)
        else:
            # HOLD: zero pose-delta (stay put) with the last commanded
            # aperture. Repeating the last DELTA would extrapolate motion the
            # policy never proposed, straight down the chunk tail the
            # executor is most likely to still be playing at the next replan.
            rows = np.zeros((pad, self.A), dtype=np.float64)
            rows[:, 6] = last[6]
        return np.concatenate([actions, rows], axis=0).astype(np.float64), pad

    # ------------------------------------------------------------------
    def replan(self, obs: ObsSnapshot, prev_plan: Plan | None,
               tcp_pose: np.ndarray) -> Plan:
        t_start = time.perf_counter()
        # spacing between the frames a queue policy conditions on. At training
        # time it is 1/fps; here it is one REPLAN, which is longer.
        self._obs_dt = (0.0 if self._last_obs_t is None
                        else float(obs.t - self._last_obs_t))
        self._last_obs_t = float(obs.t)
        np_batch = self.observation(obs)
        chunk = self._raw_chunk(self._to_torch(np_batch))
        actions = self._to_deltas(chunk, tcp_pose)
        actions, padded = self._fit_horizon(actions)
        if not np.isfinite(actions).all():
            raise ValueError("policy produced a non-finite action chunk")
        latency = time.perf_counter() - t_start
        t_exec0 = obs.t + latency
        return Plan(
            t_created=obs.t,
            t0_pose=np.asarray(tcp_pose, dtype=np.float64).copy(),
            actions=actions,
            action_times=t_exec0 + np.arange(self.H) / self.rate,
            # zeros => SpeedGovernor.scale() == 1.0 for every legal config
            sigma=np.zeros(self.H, dtype=np.float64),
            gate=0.0,
            p_evt=np.zeros(5, dtype=np.float64),
            cpk=None,
            latency_s=latency,
            diag={"policy": "lerobot",
                  "policy_class": type(self.policy).__name__,
                  "action_space": self.action_space,
                  "chunk_steps": int(chunk.shape[0]),
                  "padded_steps": int(padded),
                  "nfe": int(self.nfe) if self.nfe else 0,
                  # whether --nfe actually reached the policy, so a trace
                  # never claims a denoise budget the model ignored
                  "nfe_applied": bool(self._nfe_kwarg),
                  # the policy's OWN denoise budget (Diffusion Policy takes no
                  # per-call override, so this is the only honest number)
                  "denoise_steps": int(self.denoise_steps),
                  # queue policies: how many past frames conditioned this
                  # chunk, and how far apart the newest two actually were
                  "obs_hist": len(self._obs_hist),
                  "obs_dt_s": round(float(self._obs_dt), 4)},
        )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_lerobot_policy(ckpt: str, *, policy_type: str = "pi05",
                        device: str = "cuda", with_processors: bool = True):
    """Load the policy AND the processor pipelines saved with the checkpoint.

    Returns `(policy, preprocessor, postprocessor)`; the pipelines are None
    when `with_processors` is False or the checkpoint ships none. lerobot is
    imported lazily and only here: `phantom` must stay installable (and its
    tests runnable) without lerobot on the path."""
    cls = _policy_class(policy_type)
    log.info("loading %s from %s (%s)", cls.__name__, ckpt, device)
    policy = cls.from_pretrained(ckpt)
    policy = policy.to(device)
    policy.eval()
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    pre = post = None
    if with_processors:
        # exactly the async policy server's construction (lerobot 0.4.4)
        from lerobot.policies.factory import make_pre_post_processors
        dev = {"device": device}
        pre, post = make_pre_post_processors(
            policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": dev},
            postprocessor_overrides={"device_processor": dev})
    return policy, pre, post


def pipeline_rename_map(pipeline) -> dict:
    """The observation rename the checkpoint's OWN preprocessor performs.

    `make_pre_post_processors(pretrained_path=...)` rebuilds the pipeline from
    the saved `policy_preprocessor.json`, so a checkpoint trained with
    lerobot's `--rename_map` (our X-VLA run: scene -> image) restores that step
    verbatim and the adapter needs no rename of its own. Reading it back is
    what lets `check_checkpoint_contract` compare the POST-rename key against
    `config.image_features` instead of failing on a key the pipeline was about
    to fix."""
    out: dict = {}
    for step in getattr(pipeline, "steps", None) or []:
        rm = getattr(step, "rename_map", None)
        if isinstance(rm, dict):
            out.update(rm)
    return out


def widen_action_steps(policy, chunk_horizon: int) -> int | None:
    """Raise a queue policy's `n_action_steps` to the most steps ONE chunk can
    execute, so the deploy horizon is filled with PREDICTED steps.

    Diffusion Policy returns `n_action_steps` rows (8 by default) sliced out of
    a `horizon`-long sample at `start = n_obs_steps - 1`, so a single chunk can
    legally yield `horizon - n_obs_steps + 1` = 15 steps. Left at 8, every
    replan would hand the executor 8 real steps and 8 HELD ones — half the
    deploy horizon frozen. `tools/pi05_offline_eval.py` raises it the same way
    for scoring, so the rig and the offline table stay comparable.

    A no-op for pi05 (n_action_steps 50) and X-VLA (no `horizon` attribute).
    Returns the new value, or None when nothing changed."""
    cfg = getattr(policy, "config", None)
    vals = [getattr(cfg, a, None) for a in ("n_action_steps", "horizon",
                                            "n_obs_steps")]
    if cfg is None or any(v is None for v in vals):
        return None
    try:
        n_act, horizon, n_obs = (int(v) for v in vals)
    except (TypeError, ValueError):
        return None
    if n_act >= chunk_horizon:
        return None
    new = min(int(chunk_horizon), horizon - n_obs + 1)
    if new <= n_act:
        return None
    cfg.n_action_steps = new
    reset = getattr(policy, "reset", None)
    if callable(reset):
        # the ACTION deque's maxlen is n_action_steps; rebuild it
        reset()
    log.warning("n_action_steps raised %d -> %d (horizon %d, n_obs_steps %d) "
                "so one chunk covers the %d-step deploy horizon instead of "
                "padding %d steps every replan",
                n_act, new, horizon, n_obs, chunk_horizon, chunk_horizon - n_act)
    return new


def check_checkpoint_contract(policy, *, image_key: str, action_dim: int,
                              chunk_horizon: int,
                              rename_map: dict | None = None,
                              image_size: int | None = None) -> None:
    """Loud, load-time check that the checkpoint matches the deploy contract.

    Every one of these is a silent-wrong-answer at runtime otherwise: a
    missing camera key is padded with a BLANK image (`_preprocess_images`
    fills absent `config.image_features` with -1s and a zero mask), and a
    different action dimension means the six pose channels are not the pose
    channels."""
    cfg = getattr(policy, "config", None)
    if cfg is None:
        log.warning("policy has no `config` — skipping the checkpoint contract "
                    "check; verify the camera key and action layout by hand")
        return
    feats = getattr(cfg, "image_features", None)
    # the key the MODEL sees, after the adapter's `--rename-map` and the
    # checkpoint's own saved rename step
    sent = (rename_map or {}).get(image_key, image_key)
    if feats is not None and sent not in feats:
        raise RuntimeError(
            f"checkpoint expects image keys {sorted(feats)} but the adapter "
            f"sends {sent!r}. A missing camera is silently padded with a "
            "BLANK image, so this would run and produce nonsense. Fix the "
            "exporter's camera name, pass --image-key, or map ours onto the "
            f"checkpoint's with --rename-map "
            f"'{{\"{image_key}\": \"{sorted(feats)[0] if feats else '...'}\"}}'.")
    if feats is not None and len(feats) > 1:
        # X-VLA declares 3 image slots (it was fine-tuned from lerobot/xvla-base,
        # whose config carries them) but our export has ONE camera. lerobot's
        # `XVLAPolicy._prepare_images` fills the absent slots with zeros AND a
        # zero attention mask, which is exactly what training did, so this is a
        # warning rather than a refusal — but a policy that masks nothing would
        # be reading blank frames as real ones.
        log.warning("checkpoint declares cameras %s; the adapter supplies only "
                    "%r. Verify the policy MASKS the absent views (X-VLA does; "
                    "Diffusion Policy does not and would raise instead)",
                    sorted(feats), sent)
    if image_size is not None and isinstance(feats, dict) and sent in feats:
        shape = tuple(getattr(feats.get(sent), "shape", ()) or ())
        internal = getattr(cfg, "resize_imgs_with_padding", None) \
            or getattr(cfg, "resize_shape", None) \
            or getattr(cfg, "image_resolution", None)
        if len(shape) == 3 and tuple(shape[1:]) != (image_size, image_size):
            msg = ("checkpoint declares %s as %s but --image-size is %d"
                   % (sent, shape, image_size))
            if internal:
                log.warning("%s. The policy resizes internally (%s), so this "
                            "is only a mismatch if the TRAINING export used a "
                            "different size than %d", msg, internal, image_size)
            else:
                raise RuntimeError(
                    msg + ". This policy does NOT resize internally "
                    "(resize_shape / crop_shape are unset), so the vision "
                    "encoder would see the wrong input shape. Pass "
                    f"--image-size {shape[1]}.")
    out = getattr(cfg, "output_features", None)
    try:
        n = int(out["action"].shape[0])
    except Exception:                                # noqa: BLE001
        n = None
    if n is not None and n != action_dim:
        raise RuntimeError(
            f"checkpoint predicts {n} action dims, the deploy contract needs "
            f"{action_dim} ([dx dy dz drx dry drz grip]). Re-export or remap.")
    n_steps = getattr(cfg, "n_action_steps", None)
    if n_steps is not None and int(n_steps) < chunk_horizon:
        log.warning("checkpoint returns %d action steps but the deploy horizon "
                    "is %d — the tail will be PADDED every replan",
                    int(n_steps), chunk_horizon)


def _policy_class(policy_type: str):
    """Resolve a LeRobot policy class by its registered type name.

    `lerobot.policies.factory.get_policy_class` is the supported lookup; the
    direct-module import is the fallback for layouts where the factory moved."""
    try:
        from lerobot.policies.factory import get_policy_class
        return get_policy_class(policy_type)
    except Exception as e:                       # noqa: BLE001
        log.warning("lerobot factory lookup for %r failed (%s) — trying the "
                    "module path", policy_type, e)
    import importlib
    # class names are not a single convention: pi05 -> PI05Policy,
    # xvla -> XVLAPolicy, diffusion -> DiffusionPolicy, act -> ACTPolicy.
    attrs = (f"{policy_type.upper()}Policy", f"{policy_type.capitalize()}Policy",
             f"{policy_type.title().replace('_', '')}Policy")
    mods = (f"lerobot.policies.{policy_type}.modeling_{policy_type}",
            f"lerobot.common.policies.{policy_type}.modeling_{policy_type}")
    for mod in mods:
        try:
            m = importlib.import_module(mod)
        except Exception:                        # noqa: BLE001
            continue
        for attr in attrs:
            cls = getattr(m, attr, None)
            if cls is not None:
                return cls
    raise ImportError(
        f"could not resolve a LeRobot policy class for type {policy_type!r}. "
        "Check the installed lerobot version and pass --policy-type.")
