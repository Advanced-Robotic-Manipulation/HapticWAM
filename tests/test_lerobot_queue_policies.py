"""The Diffusion-Policy / X-VLA half of the LeRobot deploy adapter.

Everything here runs against STUB policies — no `lerobot` import, no
checkpoint, no GPU. The stubs are not convenient fakes: each one reproduces the
part of lerobot 0.4.4's real contract the adapter has to satisfy, because that
contract is the whole reason this code exists.

  * `_FakeQueuePolicy` mirrors `DiffusionPolicy`: `predict_action_chunk` stacks
    the policy's OBSERVATION QUEUES (`modeling_diffusion.py:94`), not the batch
    it is handed, and the cameras live under the single stacked key
    `observation.images`. An adapter that only passes a batch gets a KeyError
    or a one-frame history, so the stub asserts on the queue contents.
  * `_FakeSingleFramePolicy` mirrors pi05 and X-VLA: `_queues` holds the ACTION
    deque ALONE. The adapter must leave such a policy completely alone — that
    is the pi05-path-unchanged guard.
"""
from __future__ import annotations

import sys
import types
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phantom.config.hardware import load_hardware
from phantom.data import derived as dv
from phantom.inference.lerobot_policy import (DEFAULT_IMAGE_KEY,
                                              DEFAULT_STATE_KEY,
                                              LR_ACTION_KEY, LR_IMAGES_KEY,
                                              LeRobotPolicy, _policy_class,
                                              check_checkpoint_contract,
                                              pipeline_rename_map,
                                              widen_action_steps)
from phantom.inference.policy import ObsSnapshot

XVLA_IMAGE_KEY = "observation.images.image"
TCP = np.array([0.35, -0.12, 0.28, 0.10, -2.90, 0.05])


def _hw():
    return load_hardware(None)              # configs/hardware.yaml (mock rig)


def _snap(hw, *, z=0.28, grip=0.37, t=0.0):
    """A student-shaped snapshot whose z and aperture identify the frame."""
    ur = np.zeros(hw.ur_state_dim, np.float32)
    dof = hw.arm.dof
    ur[2 * dof:2 * dof + 6] = TCP
    ur[2 * dof + 2] = z
    ur[-2] = grip
    return ObsSnapshot(t=t, rgb=np.full((480, 640, 3), 17, np.uint8),
                       wrist_window=np.zeros((hw.wrist_ft.window_len, 6), np.float32),
                       ur_state=ur)


def _delta_chunk(T, dz=-0.004, grip=0.6):
    a = np.zeros((T, 7), np.float32)
    a[:, 0] = 0.003
    a[:, 2] = dz
    a[:, 6] = grip
    return a


class _FakeQueuePolicy:
    """Diffusion Policy's contract, reproduced."""

    def __init__(self, chunk, *, n_obs_steps=2, cameras=(DEFAULT_IMAGE_KEY,),
                 n_action_steps=8, horizon=16):
        self.chunk = np.asarray(chunk, np.float32)
        self.config = SimpleNamespace(
            n_obs_steps=n_obs_steps, horizon=horizon,
            n_action_steps=n_action_steps,
            image_features={k: SimpleNamespace(shape=(3, 224, 224))
                            for k in cameras},
            output_features={"action": SimpleNamespace(shape=(7,))},
            num_inference_steps=None, num_train_timesteps=100)
        self.resets = 0
        self.stacked: list[dict] = []        # queue contents, per call
        self.batches: list[dict] = []         # batch keys, per call
        self.reset()

    def reset(self):
        self.resets += 1
        self._queues = {
            DEFAULT_STATE_KEY: deque(maxlen=self.config.n_obs_steps),
            LR_ACTION_KEY: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[LR_IMAGES_KEY] = deque(maxlen=self.config.n_obs_steps)

    def predict_action_chunk(self, batch, noise=None):
        self.batches.append(dict(batch))
        # lerobot 0.4.4 `DiffusionPolicy.predict_action_chunk`, verbatim
        stacked = {k: torch.stack(list(self._queues[k]), dim=1)
                   for k in batch if k in self._queues}
        assert LR_IMAGES_KEY in stacked, (
            f"the adapter must stack the cameras into {LR_IMAGES_KEY!r} and "
            f"keep it in the batch; got {sorted(batch)}")
        assert stacked[DEFAULT_STATE_KEY].shape[1] == self.config.n_obs_steps
        self.stacked.append(stacked)
        return torch.from_numpy(self.chunk)[None]


def _pi05_config():
    """The REAL pi05 checkpoint config (armteam/phantom-checkpoints,
    pi05_20k/pretrained_model/config.json), so the "pi05 path unchanged"
    claim is a test and not an assertion."""
    return SimpleNamespace(
        n_obs_steps=1, chunk_size=50, n_action_steps=16,
        image_resolution=(224, 224), num_inference_steps=10,
        empty_cameras=0, max_state_dim=32,
        image_features={DEFAULT_IMAGE_KEY: SimpleNamespace(shape=(3, 224, 224))},
        output_features={"action": SimpleNamespace(shape=(7,))})


class _FakeSingleFramePolicy:
    """pi05 / X-VLA: `_queues` holds the ACTION deque and nothing else."""

    def __init__(self, chunk, *, cameras=(XVLA_IMAGE_KEY,), n_action_steps=30,
                 config=None):
        self.chunk = np.asarray(chunk, np.float32)
        self.config = config or SimpleNamespace(
            n_obs_steps=1, n_action_steps=n_action_steps,
            image_features={k: SimpleNamespace(shape=(3, 256, 256))
                            for k in cameras},
            output_features={"action": SimpleNamespace(shape=(7,))},
            num_denoising_steps=10, resize_imgs_with_padding=(224, 224))
        self.resets = 0
        self.batches: list[dict] = []
        self.num_steps: list = []
        self.reset()

    def reset(self):
        self.resets += 1
        self._queues = {LR_ACTION_KEY: deque(maxlen=self.config.n_action_steps)}

    def predict_action_chunk(self, batch, noise=None):
        """X-VLA's real signature: `(batch, noise=None)`, no `num_steps` — its
        denoise budget is `config.num_denoising_steps` and nothing else."""
        self.batches.append(dict(batch))
        self.num_steps.append(None)
        return torch.from_numpy(self.chunk)[None]


class _FakePi05Policy(_FakeSingleFramePolicy):
    """pi05: single-frame like X-VLA, but its chunk call is `(batch, **kwargs)`
    and forwards `num_steps` into `sample_actions`, so `--nfe` is a real
    lever."""

    def predict_action_chunk(self, batch, **kwargs):
        self.batches.append(dict(batch))
        self.num_steps.append(kwargs.get("num_steps"))
        return torch.from_numpy(self.chunk)[None]


def _adapter(hw, policy, **kw):
    kw.setdefault("task_text", "pick up the egg")
    return LeRobotPolicy(policy, hw, device="cpu", **kw)


# --------------------------------------------------------- history priming
def test_cold_start_repeats_the_first_frame_to_fill_the_history():
    """lerobot's `populate_queues` cold start: one observation, n_obs_steps
    identical frames. Anything else feeds the model a zero or a garbage frame
    on the first replan of every episode."""
    hw = _hw()
    pol = _FakeQueuePolicy(_delta_chunk(15))
    ad = _adapter(hw, pol)
    ad.replan(_snap(hw, z=0.30, t=1.0), None, TCP)

    st = pol.stacked[0][DEFAULT_STATE_KEY]
    assert st.shape == (1, 2, 7)
    torch.testing.assert_close(st[:, 0], st[:, 1])
    assert st[0, 0, 2] == pytest.approx(0.30, abs=1e-6)
    # (B, n_obs_steps, num_cameras, C, H, W) — the shape
    # `DiffusionModel.generate_actions` documents for its batch
    assert pol.stacked[0][LR_IMAGES_KEY].shape == (1, 2, 1, 3, 224, 224)


def test_history_rolls_so_the_second_replan_sees_two_distinct_frames():
    hw = _hw()
    pol = _FakeQueuePolicy(_delta_chunk(15))
    ad = _adapter(hw, pol)
    ad.replan(_snap(hw, z=0.30, t=1.0), None, TCP)
    ad.replan(_snap(hw, z=0.20, t=1.8), None, TCP)
    ad.replan(_snap(hw, z=0.10, t=2.6), None, TCP)

    second = pol.stacked[1][DEFAULT_STATE_KEY]
    assert second[0, 0, 2] == pytest.approx(0.30, abs=1e-6)   # older
    assert second[0, 1, 2] == pytest.approx(0.20, abs=1e-6)   # newest
    third = pol.stacked[2][DEFAULT_STATE_KEY]
    # only n_obs_steps frames are kept — the first observation is gone
    assert third[0, 0, 2] == pytest.approx(0.20, abs=1e-6)
    assert third[0, 1, 2] == pytest.approx(0.10, abs=1e-6)


def test_reset_episode_drops_the_history_so_the_next_episode_cold_starts():
    hw = _hw()
    pol = _FakeQueuePolicy(_delta_chunk(15))
    ad = _adapter(hw, pol)
    ad.replan(_snap(hw, z=0.30, t=1.0), None, TCP)
    ad.replan(_snap(hw, z=0.20, t=1.8), None, TCP)
    ad.reset_episode()
    assert ad._obs_hist == []
    ad.replan(_snap(hw, z=0.05, t=9.0), None, TCP)

    st = pol.stacked[-1][DEFAULT_STATE_KEY]
    torch.testing.assert_close(st[:, 0], st[:, 1])
    assert st[0, 0, 2] == pytest.approx(0.05, abs=1e-6)
    # and the replan spacing restarts, so a trace cannot report a gap that
    # spans the episode boundary
    assert ad._obs_dt == 0.0


def test_the_wire_reset_path_also_drops_the_history():
    """`PolicyServer._handle_owned` serves ("reset_episode", seed) by calling
    `policy.rf.reset_episode_noise()`, NOT `policy.reset_episode()`. If only
    the latter cleared the history, every episode after the first would
    condition its opening chunk on the previous episode's last frame."""
    hw = _hw()
    pol = _FakeQueuePolicy(_delta_chunk(15))
    ad = _adapter(hw, pol)
    ad.replan(_snap(hw, z=0.30, t=1.0), None, TCP)
    ad.replan(_snap(hw, z=0.20, t=1.8), None, TCP)

    ad.rf.reset_episode_noise()                  # exactly what the wire does
    assert ad._obs_hist == []
    ad.replan(_snap(hw, z=0.05, t=9.0), None, TCP)
    st = pol.stacked[-1][DEFAULT_STATE_KEY]
    torch.testing.assert_close(st[:, 0], st[:, 1])
    assert st[0, 0, 2] == pytest.approx(0.05, abs=1e-6)


def test_obs_spacing_is_recorded_because_it_is_not_the_training_spacing():
    hw = _hw()
    ad = _adapter(hw, _FakeQueuePolicy(_delta_chunk(15)))
    p0 = ad.replan(_snap(hw, t=1.0), None, TCP)
    p1 = ad.replan(_snap(hw, t=1.8), None, TCP)
    assert p0.diag["obs_dt_s"] == 0.0 and p0.diag["obs_hist"] == 1
    assert p1.diag["obs_dt_s"] == pytest.approx(0.8, abs=1e-6)
    assert p1.diag["obs_hist"] == 2
    # Diffusion Policy's own budget: num_inference_steps None -> 100 DDPM steps
    assert p1.diag["denoise_steps"] == 100


# ------------------------------------------------------------ ACTION key
def test_the_ground_truth_action_key_never_reaches_the_policy():
    """A batch carrying `action` would prime the policy's ACTION queue, and it
    would then replay the demonstration instead of predicting."""
    hw = _hw()
    pol = _FakeQueuePolicy(_delta_chunk(15))

    def pre(batch):
        out = dict(batch)
        out[LR_ACTION_KEY] = torch.zeros(1, 16, 7)   # what a dataset batch has
        return out

    ad = _adapter(hw, pol, preprocessor=pre)
    ad.replan(_snap(hw, t=1.0), None, TCP)
    assert LR_ACTION_KEY not in pol.batches[0]
    assert len(pol._queues[LR_ACTION_KEY]) == 0


# ---------------------------------------------------------------- rename
def test_rename_map_lands_the_scene_camera_under_the_checkpoint_key():
    hw = _hw()
    ad = _adapter(hw, _FakeSingleFramePolicy(_delta_chunk(30)),
                  rename_map={DEFAULT_IMAGE_KEY: XVLA_IMAGE_KEY})
    batch = ad.observation(_snap(hw))
    assert XVLA_IMAGE_KEY in batch and DEFAULT_IMAGE_KEY not in batch
    assert batch[XVLA_IMAGE_KEY].shape == (1, 3, 224, 224)
    # the state key and the language key are untouched
    assert batch[DEFAULT_STATE_KEY].shape == (1, 7)
    assert batch["task"] == "pick up the egg"


def test_no_rename_map_leaves_the_batch_byte_for_byte_as_the_pi05_path():
    hw = _hw()
    ad = _adapter(hw, _FakeSingleFramePolicy(_delta_chunk(30)))
    assert set(ad.observation(_snap(hw))) == {DEFAULT_IMAGE_KEY,
                                              DEFAULT_STATE_KEY, "task"}


def test_pipeline_rename_map_reads_the_checkpoints_own_saved_step():
    pre = SimpleNamespace(steps=[
        SimpleNamespace(rename_map={DEFAULT_IMAGE_KEY: XVLA_IMAGE_KEY}),
        SimpleNamespace(),                       # a step with no rename
    ])
    assert pipeline_rename_map(pre) == {DEFAULT_IMAGE_KEY: XVLA_IMAGE_KEY}
    assert pipeline_rename_map(SimpleNamespace(steps=[])) == {}
    assert pipeline_rename_map(None) == {}


def test_contract_check_accepts_the_key_the_rename_produces():
    hw = _hw()
    pol = _FakeSingleFramePolicy(_delta_chunk(30))
    with pytest.raises(RuntimeError, match="BLANK image"):
        check_checkpoint_contract(pol, image_key=DEFAULT_IMAGE_KEY,
                                  action_dim=7,
                                  chunk_horizon=hw.control.chunk_horizon)
    check_checkpoint_contract(
        pol, image_key=DEFAULT_IMAGE_KEY, action_dim=7,
        chunk_horizon=hw.control.chunk_horizon,
        rename_map={DEFAULT_IMAGE_KEY: XVLA_IMAGE_KEY}, image_size=224)


def test_a_policy_that_does_not_resize_refuses_a_wrong_image_size():
    """Diffusion Policy has resize_shape / crop_shape unset, so a 256 px batch
    into a 224 px checkpoint is a wrong-shape vision encoder, not a resize."""
    hw = _hw()
    pol = _FakeQueuePolicy(_delta_chunk(15))
    check_checkpoint_contract(pol, image_key=DEFAULT_IMAGE_KEY, action_dim=7,
                              chunk_horizon=hw.control.chunk_horizon,
                              image_size=224)
    with pytest.raises(RuntimeError, match="does NOT resize internally"):
        check_checkpoint_contract(pol, image_key=DEFAULT_IMAGE_KEY,
                                  action_dim=7,
                                  chunk_horizon=hw.control.chunk_horizon,
                                  image_size=256)


# ------------------------------------------------- single-frame policies
def test_a_single_frame_policy_is_left_completely_alone():
    """The pi05/X-VLA guard: no camera stacking, no queue priming, and no
    per-replan `reset()` — resetting one of these would clear its ACTION
    queue for nothing."""
    hw = _hw()
    pol = _FakeSingleFramePolicy(_delta_chunk(30))
    ad = _adapter(hw, pol, rename_map={DEFAULT_IMAGE_KEY: XVLA_IMAGE_KEY})
    ad.replan(_snap(hw, t=1.0), None, TCP)
    ad.replan(_snap(hw, t=1.8), None, TCP)

    assert pol.resets == 1                       # construction only
    assert ad._obs_hist == []
    assert set(pol.batches[0]) == {XVLA_IMAGE_KEY, DEFAULT_STATE_KEY, "task"}
    assert LR_IMAGES_KEY not in pol.batches[0]


def test_the_real_pi05_config_is_untouched_by_every_new_lever():
    """Regression guard for the checkpoint actually on the rig: the new
    image-size check, the multi-camera warning and the chunk widening must all
    be no-ops for pi05, and it must never be treated as a queue policy."""
    hw = _hw()
    pol = _FakePi05Policy(_delta_chunk(50), config=_pi05_config())

    # widening: 16 executable steps already cover the 16-step deploy horizon
    assert widen_action_steps(pol, hw.control.chunk_horizon) is None
    assert pol.config.n_action_steps == 16
    # contract check: no rename, and 224 px matches the declared feature
    check_checkpoint_contract(pol, image_key=DEFAULT_IMAGE_KEY, action_dim=7,
                              chunk_horizon=hw.control.chunk_horizon,
                              rename_map={}, image_size=224)

    ad = _adapter(hw, pol)
    assert ad.n_obs_steps == 1
    plan = ad.replan(_snap(hw, t=1.0), None, TCP)
    assert ad._obs_queues() is None and ad._obs_hist == []
    assert pol.resets == 1                        # never reset per replan
    assert set(pol.batches[0]) == {DEFAULT_IMAGE_KEY, DEFAULT_STATE_KEY, "task"}
    assert plan.actions.shape == (hw.control.chunk_horizon, 7)
    assert plan.diag["denoise_steps"] == 10       # pi05's num_inference_steps
    # --nfe still reaches pi05 as num_steps, exactly as before this change
    assert pol.num_steps == [10] and plan.diag["nfe_applied"] is True


# ------------------------------------------------------- chunk handling
def test_a_widened_diffusion_chunk_fills_the_horizon_and_pads_the_rest():
    hw = _hw()
    H = hw.control.chunk_horizon
    pol = _FakeQueuePolicy(_delta_chunk(15, dz=-0.004, grip=0.6))
    plan = _adapter(hw, pol).replan(_snap(hw, t=1.0), None, TCP)

    assert plan.actions.shape == (H, 7)
    assert plan.diag["chunk_steps"] == 15
    assert plan.diag["padded_steps"] == H - 15
    np.testing.assert_allclose(plan.actions[:15, 2], -0.004, atol=1e-7)
    # HOLD padding: zero pose delta, last aperture carried
    np.testing.assert_allclose(plan.actions[15, :6], 0.0, atol=1e-12)
    assert plan.actions[15, 6] == pytest.approx(0.6, abs=1e-6)


def test_a_long_xvla_chunk_is_truncated_to_the_deploy_horizon():
    hw = _hw()
    H = hw.control.chunk_horizon
    pol = _FakeSingleFramePolicy(_delta_chunk(30))
    plan = _adapter(hw, pol).replan(_snap(hw, t=1.0), None, TCP)
    assert plan.actions.shape == (H, 7)
    assert plan.diag["chunk_steps"] == 30 and plan.diag["padded_steps"] == 0
    assert plan.diag["denoise_steps"] == 10


def test_absolute_action_space_still_differences_on_the_queue_path():
    hw = _hw()
    H = hw.control.chunk_horizon
    poses = np.tile(TCP, (15, 1)).astype(np.float32)
    poses[:, 2] = TCP[2] - 0.002 * np.arange(1, 16)
    chunk = np.concatenate([poses, np.full((15, 1), 0.6, np.float32)], axis=1)
    pol = _FakeQueuePolicy(chunk)
    plan = _adapter(hw, pol, action_space="absolute").replan(
        _snap(hw, t=1.0), None, TCP)

    expect = dv.pose_delta(TCP, poses[0])
    np.testing.assert_allclose(plan.actions[0, :6], expect, atol=1e-9)
    np.testing.assert_allclose(plan.actions[:15, 2], -0.002, atol=1e-7)
    assert plan.actions.shape == (H, 7)


# ------------------------------------------------------ chunk widening
def test_widen_action_steps_only_touches_a_short_executable_chunk():
    pol = _FakeQueuePolicy(_delta_chunk(8))
    assert widen_action_steps(pol, 16) == 15          # horizon 16 - n_obs 2 + 1
    assert pol.config.n_action_steps == 15
    assert widen_action_steps(pol, 16) is None        # idempotent

    # pi05: 50 executable steps already cover the horizon
    pi05 = SimpleNamespace(config=SimpleNamespace(n_action_steps=50, horizon=50,
                                                  n_obs_steps=1))
    assert widen_action_steps(pi05, 16) is None
    assert pi05.config.n_action_steps == 50
    # X-VLA declares no `horizon` — nothing to widen towards
    assert widen_action_steps(_FakeSingleFramePolicy(_delta_chunk(30)), 16) is None
    assert widen_action_steps(SimpleNamespace(), 16) is None


# ------------------------------------------------- policy class resolution
def _fake_lerobot(monkeypatch, *, factory=None, modules=None):
    """Install a minimal fake `lerobot` package tree in sys.modules."""
    pkg = types.ModuleType("lerobot")
    pkg.__path__ = []                                        # type: ignore[attr-defined]
    pol = types.ModuleType("lerobot.policies")
    pol.__path__ = []                                        # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot", pkg)
    monkeypatch.setitem(sys.modules, "lerobot.policies", pol)
    if factory is not None:
        mod = types.ModuleType("lerobot.policies.factory")
        mod.get_policy_class = factory
        monkeypatch.setitem(sys.modules, "lerobot.policies.factory", mod)
    for name, attrs in (modules or {}).items():
        mod = types.ModuleType(name)
        for a, v in attrs.items():
            setattr(mod, a, v)
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.mark.parametrize("ptype", ["diffusion", "xvla"])
def test_policy_type_resolves_through_the_lerobot_factory(monkeypatch, ptype):
    seen: list[str] = []

    def factory(name):
        seen.append(name)
        return f"{name}-class"

    _fake_lerobot(monkeypatch, factory=factory)
    assert _policy_class(ptype) == f"{ptype}-class"
    assert seen == [ptype]


@pytest.mark.parametrize("ptype,attr", [("diffusion", "DiffusionPolicy"),
                                        ("xvla", "XVLAPolicy")])
def test_policy_type_falls_back_to_the_module_path(monkeypatch, ptype, attr):
    """The factory lookup is the supported path; the module path is the
    fallback, and the class names follow no single convention
    (DiffusionPolicy, XVLAPolicy, PI05Policy)."""
    sentinel = object()
    mod = f"lerobot.policies.{ptype}.modeling_{ptype}"
    _fake_lerobot(monkeypatch, modules={mod: {attr: sentinel}})
    assert _policy_class(ptype) is sentinel


def test_an_unresolvable_policy_type_raises_import_error(monkeypatch):
    _fake_lerobot(monkeypatch)
    with pytest.raises(ImportError, match="could not resolve"):
        _policy_class("not_a_policy")
