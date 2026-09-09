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
    observation.state         (1, 7) float32
                              [tcp x, y, z, rx, ry, rz (rotvec, BASE frame),
                               gripper aperture]
                              read out of `ObsSnapshot.ur_state` at the same
                              offsets `PlannerLoop.run` uses.
    task                      [str] the episode instruction.

Action contract: the policy returns a chunk (T, 7) of
`[dx, dy, dz, drx, dry, drz, grip]` at `hw.control.action_rate_hz` (10 Hz),
already in metres / radians / aperture units (LeRobot policies unnormalise
their own output). `--action-space absolute` instead reads the six pose
channels as ABSOLUTE base-frame poses and differences them with
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


class _EpisodeReset:
    """The `policy.rf` shim `PolicyServer` reaches through.

    `PolicyServer._handle_owned` seeds with
    `self.policy.rf._gen = torch.Generator().manual_seed(seed)` and then calls
    `self.policy.rf.reset_episode_noise()`. LeRobot policies keep an internal
    action QUEUE rather than held noise, so `reset_episode_noise` clears that
    queue, and the seed is applied to the process RNG the policy samples from
    (pi0/pi05 flow-matching draws its prior from the global torch generator).
    """

    def __init__(self, policy):
        self._policy = policy
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
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()


class LeRobotPolicy:
    """Duck-types `PhantomPolicy` for `PlannerLoop` / `PolicyServer`."""

    def __init__(self, policy, hw, *, task_text: str = "",
                 action_space: str = "delta",
                 image_size: int = DEFAULT_IMAGE_SIZE,
                 image_key: str = DEFAULT_IMAGE_KEY,
                 state_key: str = DEFAULT_STATE_KEY,
                 pad_mode: str = "hold", device: str = "cpu",
                 use_task_key: bool = True):
        assert action_space in ACTION_SPACES, \
            f"action_space must be one of {ACTION_SPACES}, got {action_space!r}"
        assert pad_mode in PAD_MODES, \
            f"pad_mode must be one of {PAD_MODES}, got {pad_mode!r}"
        self.policy = policy
        self.hw = hw
        self.action_space = action_space
        self.image_size = int(image_size)
        self.image_key = image_key
        self.state_key = state_key
        self.pad_mode = pad_mode
        self.device = device
        self.use_task_key = bool(use_task_key)
        self.rf = _EpisodeReset(policy)
        # SnapshotBuilder mirrors this off the server's `info`; a LeRobot
        # policy never saw a wrench baseline, so the snapshot must not
        # subtract one.
        self.wrench_baseline_rows = 0

        self.H = int(hw.control.chunk_horizon)
        self.A = int(hw.control.action_dim)
        self.rate = float(hw.control.action_rate_hz)
        self.dof = int(hw.arm.dof)

        # --- the PhantomPolicy attribute surface `remote.CONFIGURABLE`
        # reconfigures and `run_deploy`'s cond_tags read. Only `task_text` is
        # live; the rest exist so `configure` and the tag line do not crash,
        # and `assert_deploy_flags` refuses the ones that would LIE.
        self.task_text = task_text
        self.nfe = getattr(getattr(policy, "config", None), "num_steps", None)
        self.guidance = 1.0
        self.k_seeds = 1
        self.parity_fixes = False
        self.persistent_noise = False
        self.drop_video = False
        self.close_p = 0.5

    # ------------------------------------------------------------------
    def assert_deploy_flags(self, **flags) -> None:
        """Refuse a launch that asked for a lever this policy cannot honour.

        Called by `lerobot_server` at load and by `configure` at attach: a
        silently-ignored `--terminal-veto` produces an arm tagged `veto:on`
        that vetoed nothing (validation 2026-08-30 §9)."""
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
        return {self.image_key: img[None],          # (1, 3, S, S)
                self.state_key: state[None],        # (1, 7)
                "task": [self.task_text]}

    def _to_torch(self, batch: dict) -> dict:
        import torch
        out = {}
        for k, v in batch.items():
            if k == "task":
                if self.use_task_key:
                    out[k] = list(v)
                continue
            out[k] = torch.from_numpy(np.asarray(v)).to(self.device)
        return out

    # -- action --------------------------------------------------------
    def _raw_chunk(self, batch: dict) -> np.ndarray:
        """(T, A) numpy chunk out of whichever LeRobot entry point exists.

        `predict_action_chunk` is the chunk-native call (pi0 / pi05 / ACT /
        diffusion all expose it). `select_action` is the fallback: it returns
        ONE action per call off the policy's internal queue, so H calls on the
        same observation reproduce the chunk — correct, but it re-runs any
        policy whose queue is shorter than H.
        """
        import torch
        with torch.no_grad():
            fn = getattr(self.policy, "predict_action_chunk", None)
            if callable(fn):
                out = fn(batch)
                arr = np.asarray(out.detach().float().cpu().numpy(), dtype=np.float64)
                if arr.ndim == 3:
                    arr = arr[0]
                elif arr.ndim == 1:
                    arr = arr[None]
                return arr
            sel = getattr(self.policy, "select_action", None)
            if not callable(sel):
                raise AttributeError(
                    f"{type(self.policy).__name__} exposes neither "
                    "`predict_action_chunk` nor `select_action` — this is not "
                    "a LeRobot PreTrainedPolicy")
            rows = []
            for _ in range(self.H):
                a = sel(batch)
                a = np.asarray(a.detach().float().cpu().numpy(), dtype=np.float64)
                rows.append(a.reshape(-1)[:])
            return np.stack(rows)

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
                  "padded_steps": int(padded)},
        )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_lerobot_policy(ckpt: str, *, policy_type: str = "pi05",
                        device: str = "cuda"):
    """`from_pretrained` on the LeRobot policy class for `policy_type`.

    Imported lazily and only here: `phantom` must stay installable (and its
    tests runnable) without lerobot on the path."""
    cls = _policy_class(policy_type)
    log.info("loading %s from %s (%s)", cls.__name__, ckpt, device)
    policy = cls.from_pretrained(ckpt)
    policy = policy.to(device)
    policy.eval()
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    return policy


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
    for mod, attr in ((f"lerobot.policies.{policy_type}.modeling_{policy_type}",
                       f"{policy_type.upper()}Policy"),
                      (f"lerobot.common.policies.{policy_type}."
                       f"modeling_{policy_type}", f"{policy_type.upper()}Policy")):
        try:
            return getattr(importlib.import_module(mod), attr)
        except Exception:                        # noqa: BLE001
            continue
    raise ImportError(
        f"could not resolve a LeRobot policy class for type {policy_type!r}. "
        "Check the installed lerobot version and pass --policy-type.")
