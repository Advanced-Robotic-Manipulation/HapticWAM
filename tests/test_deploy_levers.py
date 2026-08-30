"""Deploy levers from the 2026-08-28 review (docs/review_20260828), all behind
flags that default OFF so the rig A/B can attribute each effect:

P2 `--parity-fixes` — prev_chunk built from the MEASURED arm history + the
    gripper commands actually sent (as WindowSampler builds it in training)
    instead of the policy's own last proposal; contact-state dt from the
    tactile ring timestamps; `reactive` from the two consecutive fields_ds
    frames rather than the last two replans; ACC prev_cpk summarised at step
    round(latency / latent_dt).
P3 `--terminal-veto` — close-mask + phantom-grasp open/re-descend recovery with
    a per-episode retry cap, the first thing at deploy that lets the ACC gate
    modify a chunk instead of only being logged.
P6 `--k-seeds K` — K chunks per replan in one batched denoise, then
    contact-consistent selection.
"""

from __future__ import annotations

import time
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phantom.data import derived as dv
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.planner import PlannerLoop, SnapshotBuilder, TerminalVeto
from phantom.inference.policy import ObsSnapshot, PhantomPolicy, Plan, _cpk_row
from phantom.model.ace.packing import ContactPackage
from phantom.model.acc import AccInputs
from phantom.model.rf import _expand_acc
from phantom_test_utils import make_small_hw


# ---------------------------------------------------------------------------
# ring / executor stubs
# ---------------------------------------------------------------------------

class _PoseRing:
    """Arm ring whose rows carry explicit timestamps AND tcp_pose."""

    def __init__(self, ts, poses, dof=6):
        self.ts = np.asarray(ts, dtype=np.float64)
        self.poses = np.asarray(poses, dtype=np.float64)
        assert len(self.ts) == len(self.poses)
        self.dof = dof
        self.asked = []

    def latest(self, k=1):
        self.asked.append(k)
        m = min(k, len(self.ts))
        sl = slice(len(self.ts) - m, len(self.ts))
        return self.ts[sl].copy(), {
            "q": np.zeros((m, self.dof)), "qd": np.zeros((m, self.dof)),
            "tcp_pose": self.poses[sl].copy(), "tcp_speed": np.zeros((m, 6)),
            "ft": np.zeros((m, 6)),
        }

    def latest_ts(self):
        return float(self.ts[-1])


class _FixedRing:
    def __init__(self, fields, age=0.0, n=1):
        self.fields, self.age, self.n = fields, age, n

    def latest(self, k=1):
        m = min(k, self.n)
        ts = np.array([time.perf_counter() - self.age] * m)
        return ts, {f: np.stack([np.asarray(v)] * m) for f, v in self.fields.items()}

    def latest_ts(self):
        return time.perf_counter() - self.age


def _rings(hw, arm_ring, grip_pos=0.4):
    return {
        "camera_scene": _FixedRing({"color": np.zeros(hw.cameras.scene.color.hwc,
                                                      np.uint8)}),
        "arm": arm_ring,
        "gripper": _FixedRing({"state": np.array([grip_pos, 3.0])}),
    }


def _builder(hw, arm_ring, *, parity=False, executor=None, grip_pos=0.4,
             mode="vision_only", rings=None):
    rings = _rings(hw, arm_ring, grip_pos) if rings is None else rings
    return SnapshotBuilder(hw, types.SimpleNamespace(rings=rings), mode,
                           parity_fixes=parity, executor=executor)


def _descending_ring(hw, t_now, *, span_s=2.5, rate_hz=None, v_z=-0.05):
    """Arm ring descending at v_z m/s, sampled at the RTDE rate."""
    rate_hz = rate_hz or hw.arm.rtde_receive_hz
    n = int(span_s * rate_hz)
    ts = t_now - np.arange(n)[::-1] / rate_hz
    poses = np.zeros((n, 6))
    poses[:, 0] = 0.1                      # x constant
    poses[:, 2] = 0.3 + v_z * (ts - ts[0])  # z ramps down
    return _PoseRing(ts, poses)


# ---------------------------------------------------------------------------
# P2 — prev_chunk from the executed past
# ---------------------------------------------------------------------------

def test_prev_chunk_is_absent_unless_the_flag_is_on():
    """Default OFF: the snapshot must not carry prev_chunk, so PhantomPolicy
    keeps conditioning on prev_plan exactly as before."""
    hw = make_small_hw()
    t = time.perf_counter()
    assert _builder(hw, _descending_ring(hw, t)).build().prev_chunk is None


def test_parity_prev_chunk_is_the_measured_pose_delta_chain():
    """The 16 rows must equal `derived.pose_delta` chained over the measured
    poses nearest t_now - (H - k)/rate — the grid windows.py:277 samples."""
    hw = make_small_hw()
    H, rate = hw.control.chunk_horizon, hw.control.action_rate_hz
    t = time.perf_counter()
    ring = _descending_ring(hw, t, v_z=-0.05)
    snap = _builder(hw, ring, parity=True).build()
    assert snap.prev_chunk.shape == (H, hw.control.action_dim)

    grid = snap.t - (H - np.arange(-1, H)) / rate
    idx = np.abs(ring.ts[None, :] - grid[:, None]).argmin(axis=1)
    p = ring.poses[idx]
    want = np.stack([dv.pose_delta(p[k], p[k + 1]) for k in range(H)])
    assert np.allclose(snap.prev_chunk[:, :6], want, atol=1e-9)
    # a 0.05 m/s descent over a 0.1 s grid step is -5 mm per row
    assert np.allclose(snap.prev_chunk[:, 2], -0.005, atol=2e-4)


def test_parity_prev_chunk_reaches_back_a_full_horizon():
    """The ring read must SPAN (H+1)/rate seconds, not the wrist window: with
    the old row count the oldest grid points all collapsed onto one sample."""
    hw = make_small_hw()
    t = time.perf_counter()
    ring = _descending_ring(hw, t)
    _builder(hw, ring, parity=True).build()
    span_s = (hw.control.chunk_horizon + 1) / hw.control.action_rate_hz
    assert max(ring.asked) >= span_s * hw.arm.rtde_receive_hz


def test_parity_prev_chunk_gripper_is_the_executed_command():
    """Training's prev_chunk gripper channel is the command that was SENT;
    the executor's entered-step history is the deploy-side equivalent."""
    hw = make_small_hw()
    H = hw.control.chunk_horizon
    t = time.perf_counter()
    ex = SimpleNamespace(gripper_cmd_at=lambda ts: np.full(len(ts), 0.77))
    snap = _builder(hw, _descending_ring(hw, t), parity=True,
                    executor=ex, grip_pos=0.4).build()
    assert np.allclose(snap.prev_chunk[:, 6], 0.77)
    assert snap.prev_chunk.shape[0] == H


def test_parity_prev_chunk_gripper_falls_back_to_the_measured_aperture():
    """Before the first executed step (first replan / warm-up) the executor has
    nothing to report; the aperture the gripper is holding is the honest
    stand-in, never a hardcoded 0 (a frozen off-distribution gripper input is
    what collapsed the action magnitude ~3x in the 08-14 field root-cause)."""
    hw = make_small_hw()
    t = time.perf_counter()
    partial = np.full(hw.control.chunk_horizon, np.nan)
    partial[-3:] = 0.9
    for ex in (None,
               SimpleNamespace(gripper_cmd_at=lambda ts: None),
               SimpleNamespace(gripper_cmd_at=lambda ts: partial)):
        snap = _builder(hw, _descending_ring(hw, t), parity=True,
                        executor=ex, grip_pos=0.42).build()
        g = snap.prev_chunk[:, 6]
        assert np.allclose(g[:3], 0.42)
        if ex is not None and ex.gripper_cmd_at(np.zeros(1)) is not None:
            assert np.allclose(g[-3:], 0.9)


def test_parity_first_replan_is_still_the_zero_action_chunk():
    """An arm holding the start pose yields all-zero pose deltas — the same
    'no previous chunk == zero PHYSICAL action' conditioning the fallback
    branch constructs, and NOT raw zeros in normalized space."""
    hw = make_small_hw()
    t = time.perf_counter()
    n = int(2.5 * hw.arm.rtde_receive_hz)
    ts = t - np.arange(n)[::-1] / hw.arm.rtde_receive_hz
    still = np.tile(np.array([0.1, -0.4, 0.25, 0.0, 3.1, 0.0]), (n, 1))
    snap = _builder(hw, _PoseRing(ts, still), parity=True).build()
    assert np.allclose(snap.prev_chunk[:, :6], 0.0, atol=1e-12)


def test_parity_prev_chunk_needs_two_rows():
    hw = make_small_hw()
    t = time.perf_counter()
    sb = _builder(hw, _PoseRing([t], np.zeros((1, 6))), parity=True)
    assert sb.prev_chunk_from_history(t, np.array([t]), {"tcp_pose": np.zeros((1, 6))},
                                      0.4) is None


# ---------------------------------------------------------------------------
# P2 — tactile parity: measured dt and the consecutive-frame reactive score
# ---------------------------------------------------------------------------

def _tactile_rings(hw, arm_ring, *, dt_ring: float, grip_pos=0.4):
    """Teacher-mode rings whose fields_ds ring holds two frames dt_ring apart."""
    rings = _rings(hw, arm_ring, grip_pos)
    h, w = hw.recording.field_ds.h, hw.recording.field_ds.w
    now = time.perf_counter()

    class _Tac:
        def __init__(self, base):
            self.base = base

        def latest(self, k=1):
            ts = np.array([now - dt_ring, now])[-min(k, 2):]
            m = len(ts)
            cur = np.full((h, w, 8), self.base, dtype=np.float32)
            prev = np.full((h, w, 8), self.base - 1.0, dtype=np.float32)
            frames = np.stack([prev, cur])[-m:]
            return ts, {"fields_ds": frames,
                        "wrench": np.zeros((m, 6), np.float32),
                        "area": np.zeros(m, np.float32)}

        def latest_ts(self):
            return now

    for i, s in enumerate(hw.tactile.sensors):
        rings[f"tactile_{s.name}"] = _Tac(float(i + 2))
        rings[f"tactile_{s.name}_kf"] = _FixedRing(
            {"keyframe": np.zeros((hw.recording.keyframe_ds.h,
                                   hw.recording.keyframe_ds.w, 8), np.float32)})
    return rings


def test_parity_contact_state_dt_comes_from_the_ring_timestamps(monkeypatch):
    """Training reads the MEASURED inter-frame dt (windows.py `_field_frame`);
    deploy used the nominal 1/field_ds_rate_hz, and derive_timestep divides the
    tangential flow by it, so a jittery ring rescaled `slip` at deploy only."""
    hw = make_small_hw()
    t = time.perf_counter()
    dt_ring = 3.7 / hw.recording.field_ds_rate_hz          # ring running slow
    seen: list[float] = []
    real = dv.derive_timestep
    monkeypatch.setattr(dv, "derive_timestep",
                        lambda cur, prev, dt, hw_: (seen.append(dt),
                                                    real(cur, prev, dt, hw_))[1])
    rings = _tactile_rings(hw, _descending_ring(hw, t), dt_ring=dt_ring)
    _builder(hw, None, mode="teacher", rings=rings).build()
    assert seen and all(abs(d - 1.0 / hw.recording.field_ds_rate_hz) < 1e-12
                        for d in seen)

    seen.clear()
    rings = _tactile_rings(hw, _descending_ring(hw, t), dt_ring=dt_ring)
    _builder(hw, None, mode="teacher", rings=rings, parity=True).build()
    assert seen and all(abs(d - dt_ring) < 1e-6 for d in seen)


def test_parity_reactive_is_the_consecutive_frame_difference():
    """Training differences the two consecutive fields_ds frames at t0 (~1/30 s
    apart); deploy differenced the last two REPLANS (~1 s apart on the rig),
    i.e. a ~30x larger dt into the same CASA psi_react MLP."""
    hw = make_small_hw()
    t = time.perf_counter()
    dt_ring = 1.0 / hw.recording.field_ds_rate_hz

    sb = _builder(hw, None, mode="teacher", parity=True,
                  rings=_tactile_rings(hw, _descending_ring(hw, t), dt_ring=dt_ring))
    # the stub's consecutive frames differ by exactly 1.0 in every cell
    assert sb.build().reactive == pytest.approx(1.0)

    legacy = _builder(hw, None, mode="teacher",
                      rings=_tactile_rings(hw, _descending_ring(hw, t), dt_ring=dt_ring))
    assert legacy.build().reactive == 0.0          # no previous replan yet
    assert legacy.build().reactive == 0.0          # frames are static replan-to-replan


# ---------------------------------------------------------------------------
# P2 — ACC prev_cpk step alignment
# ---------------------------------------------------------------------------

def _cpk(B=1, Tc=3, F=2, cph=3, cpw=4):
    return ContactPackage(
        event=torch.zeros(B, Tc, 5), d_disp=torch.zeros(B, Tc, F, 3, cph, cpw),
        d_fz=torch.zeros(B, Tc, F, cph, cpw), mask=torch.zeros(B, Tc, F, cph, cpw),
        cop=torch.zeros(B, Tc, F, 2), slip=torch.zeros(B, Tc, F),
        wrench=torch.zeros(B, Tc, F, 6), wrist=torch.zeros(B, Tc, 6))


def _fake_policy(hw, sample_fn, **kw):
    """PhantomPolicy with a stub rf/backbone — enough for _batch_from_obs and
    replan without the 2B model (tests/test_model_tiny.py covers the real one)."""
    pol = object.__new__(PhantomPolicy)
    pol.hw = hw
    pol.bb = SimpleNamespace(res_h=16, res_w=16, frames_pix=2, t_video=3,
                             temporal_comp=8, fps=8.0)
    pol.pm = SimpleNamespace(layout=SimpleNamespace(student=True))
    from phantom.data.schema import NormStats
    pol.norm = NormStats.identity()
    pol.task_text = "t"
    pol.nfe, pol.guidance = 5, 1.0
    pol.persistent_noise, pol.drop_video = False, False
    pol.parity_fixes = kw.get("parity_fixes", False)
    pol.latent_dt = pol.bb.temporal_comp / pol.bb.fps
    pol.k_seeds = kw.get("k_seeds", 1)
    pol.close_p = kw.get("close_p", 0.5)
    pol.rf = SimpleNamespace(sample=sample_fn, device=torch.device("cpu"),
                             dtype=torch.float32)
    return pol


def _pred(K, H, A, *, dz=None):
    from phantom.model.rf import PhantomPrediction
    actions = torch.zeros(K, H, A)
    if dz is not None:
        for j, v in enumerate(dz):
            actions[j, :, 2] = v
    return PhantomPrediction(
        actions_B_H_A=actions, cpk=_cpk(K), event_logits_B_Tc_E=torch.zeros(K, 3, 5),
        log_sigma_B_Tc_K=torch.zeros(K, 3, 1),
        governor_sigma_B_Tc=torch.zeros(K, 3),
        acc=SimpleNamespace(g=torch.zeros(K),
                            p_evt=torch.tensor([[1.0, 0, 0, 0, 0]] * K)),
        x_final_B_C_T_H_W=torch.zeros(K, 1, 1, 1, 1))


def _snap(hw):
    return ObsSnapshot(t=time.perf_counter(),
                       rgb=np.zeros((4, 4, 3), np.uint8),
                       wrist_window=np.zeros((hw.wrist_ft.window_len, 6), np.float32),
                       ur_state=np.zeros(2 * hw.arm.dof + 14, np.float32))


def _prev_plan(hw, latency=0.96):
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    return Plan(t_created=0.0, t0_pose=np.zeros(6), actions=np.zeros((H, A)),
                action_times=np.arange(H) / hw.control.action_rate_hz,
                sigma=np.zeros(3), gate=0.0, p_evt=np.zeros(5), cpk=_cpk(),
                latency_s=latency)


def test_prev_cpk_step_is_the_latency_aligned_index():
    """flatten_summary(step=0) means 'contact at t0 + latent_dt', which is what
    training's two-pass proxy predicts at the SAME t0. The deploy package is
    one replan (latency L) older, so the aligned index is round(L/latent_dt)."""
    hw = make_small_hw()
    seen = {}
    H, A = hw.control.chunk_horizon, hw.control.action_dim

    def sample(batch, **kw):
        seen.update(kw)
        return _pred(1, H, A)

    pol = _fake_policy(hw, sample, parity_fixes=True)
    latent_dt = pol.latent_dt                            # 1.0 s here
    pol.replan(_snap(hw), _prev_plan(hw, latency=0.96 * latent_dt), np.zeros(6))
    assert seen["prev_cpk_step"] == 1
    pol.replan(_snap(hw), _prev_plan(hw, latency=0.2 * latent_dt), np.zeros(6))
    assert seen["prev_cpk_step"] == 0
    # ... and OFF by default: the historical step 0
    off = _fake_policy(hw, sample)
    off.replan(_snap(hw), _prev_plan(hw, latency=0.96 * latent_dt), np.zeros(6))
    assert seen["prev_cpk_step"] == 0


def test_policy_prefers_the_snapshot_prev_chunk_over_its_own_proposal():
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    seen = {}

    def sample(batch, **kw):
        seen["prev_chunk"] = batch["prev_chunk"].clone()
        return _pred(1, H, A)

    pol = _fake_policy(hw, sample, parity_fixes=True)
    prev = _prev_plan(hw)
    prev.actions = np.full((H, A), 9.0)              # the PROPOSAL
    snap = _snap(hw)
    snap.prev_chunk = np.full((H, A), 3.0, np.float32)   # the EXECUTED past
    pol.replan(snap, prev, np.zeros(6))
    assert torch.allclose(seen["prev_chunk"], torch.full((1, H, A), 3.0))
    # without the snapshot channel the legacy proposal path is untouched
    pol.replan(_snap(hw), prev, np.zeros(6))
    assert torch.allclose(seen["prev_chunk"], torch.full((1, H, A), 9.0))


# ---------------------------------------------------------------------------
# executor: the executed-gripper history
# ---------------------------------------------------------------------------

def _executor(hw):
    return ChunkExecutor(hw, SimpleNamespace(servo_stop=lambda: None),
                         SimpleNamespace(), SimpleNamespace())


def test_gripper_cmd_at_zero_order_holds_the_entered_steps():
    hw = make_small_hw()
    ex = _executor(hw)
    assert ex.gripper_cmd_at([1.0]) is None            # nothing executed yet
    ex._grip_hist.extend([(10.0, 0.1), (10.5, 0.6), (11.0, 0.9)])
    got = ex.gripper_cmd_at([9.9, 10.0, 10.2, 10.7, 99.0])
    assert np.isnan(got[0])                            # before the first step
    assert list(got[1:]) == [0.1, 0.1, 0.6, 0.9]


def test_only_entered_steps_are_recorded_and_they_are_clipped():
    """Chunk TAILS that never execute must never enter the history — that is
    the whole difference between the executed past and a proposal."""
    hw = make_small_hw()
    ex = _executor(hw)
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    actions = np.zeros((H, A))
    actions[:, 6] = np.linspace(0.0, 2.0, H)           # deliberately out of range
    plan = Plan(t_created=0.0, t0_pose=np.zeros(6), actions=actions,
                action_times=np.zeros(H), sigma=np.zeros(3), gate=0.0,
                p_evt=np.zeros(5), cpk=None)
    ex._plan, ex._swap_t = plan, time.perf_counter()
    ex._play_time = 2.5 / hw.control.action_rate_hz    # entered steps 0..2
    ex._last_action_k = -1
    # replicate the loop's step-entry block
    k = min(int(ex._play_time * hw.control.action_rate_hz), H - 1)
    while ex._last_action_k < k:
        ex._last_action_k += 1
        ex._grip_hist.append((float(ex._last_action_k),
                              float(np.clip(actions[ex._last_action_k, 6], 0, 1))))
    assert len(ex._grip_hist) == 3
    assert all(0.0 <= g <= 1.0 for _, g in ex._grip_hist)


def test_executor_start_clears_the_gripper_history():
    hw = make_small_hw()
    ex = _executor(hw)
    ex._grip_hist.append((1.0, 0.5))
    ex._thread = ex._grip_thread = None
    ex.start()
    try:
        assert ex.gripper_cmd_at([2.0]) is None
    finally:
        ex._stop.set()


# ---------------------------------------------------------------------------
# P3 — terminal veto
# ---------------------------------------------------------------------------

class _Ex:
    """Ideal executor: an accepted chunk is played to the end, so every one of
    its action-grid steps is ENTERED. `entered_grip_after` is what the terminal
    veto latches its recovery on (F3, 2026-08-30) — a rejected plan, or a plan
    whose close lives in a tail playback never reached, arms nothing."""

    def __init__(self, accepts=None, enter=True):
        self.stopped_reason = None
        self.submitted = []
        self.accepts = accepts
        self.enter = enter
        self._hist: list[tuple[float, float]] = []
        self._t = 0.0

    def last_cmd(self):
        return None

    def request_stop(self, reason):
        if self.stopped_reason is None:
            self.stopped_reason = reason

    def submit(self, plan):
        ok = True if self.accepts is None else self.accepts[
            min(len(self.submitted), len(self.accepts) - 1)]
        self.submitted.append(np.array(plan.actions, copy=True))
        if ok and self.enter:
            for g in plan.actions[:, 6]:
                self._t += 0.1
                self._hist.append((self._t, float(g)))
        return ok

    def entered_grip_after(self, t):
        return [(ts, g) for ts, g in self._hist if ts > t]


class _Snaps:
    def __init__(self, hw, z=0.25, grip=0.30):
        self.hw, self.z, self.grip = hw, z, grip

    def build(self):
        snap = _snap(self.hw)
        snap.ur_state[2 * self.hw.arm.dof + 2] = self.z          # tcp z
        snap.ur_state[-2] = self.grip
        return snap


class _Pol:
    """Emits a closing, lifting chunk; p_evt[none] scripted per replan."""

    def __init__(self, hw, p_none, grip_cmd=0.8, dz=+0.01):
        self.hw, self.p_none, self.grip_cmd, self.dz = hw, list(p_none), grip_cmd, dz
        self.n = 0

    def replan(self, snap, prev_plan, tcp_pose):
        H, A = self.hw.control.chunk_horizon, self.hw.control.action_dim
        a = np.zeros((H, A))
        a[:, 2] = self.dz                              # rising = lifting
        a[:, 6] = self.grip_cmd
        pn = self.p_none[min(self.n, len(self.p_none) - 1)]
        self.n += 1
        return Plan(t_created=float(self.n), t0_pose=np.asarray(tcp_pose, float).copy(),
                    actions=a, action_times=np.arange(H) / 10.0, sigma=np.zeros(3),
                    gate=0.0, p_evt=np.array([pn, 1 - pn, 0, 0, 0]), cpk=None,
                    latency_s=0.9)


def _loop(hw, pol, ex, veto, z=0.25, grip=0.30):
    return PlannerLoop(hw, pol, _Snaps(hw, z=z, grip=grip), ex, veto=veto)


VETO = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=3, z_floor=0.10,
                    z_margin=0.015, open_aperture=0.25)


def test_veto_off_leaves_the_chunk_and_the_trace_untouched():
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.99]), ex, None)
    loop.run(max_replans=2)
    assert all(t["terminal_veto"] is None for t in loop.trace)
    assert np.allclose(ex.submitted[0][:, 6], 0.8)     # close still commanded


def test_close_is_masked_when_contact_is_not_anticipated():
    """p_evt[none]=0.99 => p_contact 0.01 <= 0.5 and the TCP is 150 mm above
    the floor: the commanded close is rewritten to hold the aperture."""
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.99]), ex, VETO, z=0.25, grip=0.30)
    loop.run(max_replans=1)
    assert np.allclose(ex.submitted[0][:, 6], 0.30)
    assert loop.trace[0]["terminal_veto"]["action"] == "close_masked"
    # only the gripper channel is rewritten
    assert np.allclose(ex.submitted[0][:, 2], 0.01)


def test_close_is_allowed_when_p_contact_clears_the_threshold():
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.2]), ex, VETO, z=0.25)
    loop.run(max_replans=1)
    assert np.allclose(ex.submitted[0][:, 6], 0.8)
    assert loop.trace[0]["terminal_veto"]["action"] == "close_allowed"


def test_close_is_allowed_within_15mm_of_the_task_z_floor():
    """At the floor, closing is what a demo does — the mask must not turn the
    one deep descent we are trying to produce into a no-op."""
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.99]), ex, VETO, z=VETO.z_floor + 0.010)
    loop.run(max_replans=1)
    rec = loop.trace[0]["terminal_veto"]
    assert rec["action"] == "close_allowed" and rec["at_floor"] is True
    ex2 = _Ex()
    loop2 = _loop(hw, _Pol(hw, [0.99]), ex2, VETO, z=VETO.z_floor + 0.030)
    loop2.run(max_replans=1)
    assert loop2.trace[0]["terminal_veto"]["action"] == "close_masked"


def test_a_non_closing_chunk_is_never_touched():
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.99], grip_cmd=0.35), ex, VETO, grip=0.30)
    loop.run(max_replans=1)
    assert loop.trace[0]["terminal_veto"]["action"] == "none"
    assert np.allclose(ex.submitted[0][:, 6], 0.35)


def test_phantom_grasp_recovery_opens_and_forbids_the_lift():
    """Replan 0 closes with p_contact 0.8; replan 1 reports p_none 0.95 =>
    the grasp is phantom, so open to the task aperture and clamp away every
    upward z (the chunk was a pure +10 mm/step lift)."""
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.2, 0.95]), ex, VETO)
    loop.run(max_replans=2)
    assert loop.trace[0]["terminal_veto"]["action"] == "close_allowed"
    rec = loop.trace[1]["terminal_veto"]
    assert rec["action"] == "recovery_open" and rec["retries"] == 1
    a = ex.submitted[1]
    assert np.allclose(a[:, 6], VETO.open_aperture)
    assert np.all(np.cumsum(a[:, 2]) <= 1e-12)         # never above z_now
    assert np.allclose(a[:, 0], 0.0)                   # x/y untouched


def test_recovery_preserves_a_descent():
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.2, 0.95], dz=-0.004), ex, VETO)
    loop.run(max_replans=2)
    assert np.allclose(ex.submitted[1][:, 2], -0.004)   # descent survives intact


def test_the_retry_cap_ends_the_episode():
    """An uncapped open/re-descend loop is the safety gap the review's
    completeness critic flagged: cap it and stop with a named reason."""
    hw = make_small_hw()
    ex = _Ex()
    v = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=2, z_floor=0.10,
                     open_aperture=0.25)
    # close, phantom, close, phantom, close, phantom -> 3rd exceeds the cap
    loop = _loop(hw, _Pol(hw, [0.2, 0.95, 0.2, 0.95, 0.2, 0.95]), ex, v)
    loop.run(max_replans=20)
    assert ex.stopped_reason == "veto_retry_cap"
    assert loop.trace[-1]["terminal_veto"]["action"] == "retry_cap"
    assert loop.trace[-1]["accepted"] is False
    assert len(ex.submitted) == len(loop.trace) - 1     # the capped plan is not sent


def test_a_rejected_close_is_not_the_close_recovery_reacts_to():
    """Same rule as prev_chunk/prev_cpk: a plan the executor rejected was never
    commanded, so it cannot be the close that a phantom grasp follows."""
    hw = make_small_hw()
    ex = _Ex(accepts=[False, True])
    loop = _loop(hw, _Pol(hw, [0.2, 0.95]), ex, VETO)
    loop.run(max_replans=2)
    assert loop.trace[1]["terminal_veto"]["action"] != "recovery_open"


# ---------------------------------------------------------------------------
# P6 — K-seed sampling and selection
# ---------------------------------------------------------------------------

def test_expand_acc_replicates_every_conditioning_tensor():
    acc = AccInputs(wrist_feat_B_D=torch.arange(3.0).reshape(1, 3),
                    intent_B_H_A=torch.ones(1, 2, 7),
                    prev_cpk_summary_B_S=torch.full((1, 4), 2.0),
                    react_score_B=torch.tensor([0.5]))
    out = _expand_acc(acc, 4)
    assert out.wrist_feat_B_D.shape == (4, 3)
    assert out.intent_B_H_A.shape == (4, 2, 7)
    assert out.prev_cpk_summary_B_S.shape == (4, 4)
    assert out.react_score_B.shape == (4,)
    assert torch.allclose(out.wrist_feat_B_D, torch.arange(3.0).expand(4, 3))
    assert _expand_acc(AccInputs(torch.zeros(1, 2), torch.zeros(1, 1, 1),
                                 torch.zeros(1, 1), None), 2).react_score_B is None


def test_cpk_row_keeps_the_selected_batch_row():
    cpk = _cpk(B=3)
    cpk.wrist[1] = 7.0
    row = _cpk_row(cpk, 1)
    assert row.wrist.shape[0] == 1 and float(row.wrist.max()) == 7.0


def _sel(hw, actions, p_contact, prev_plan=None, close_p=0.5):
    pol = _fake_policy(hw, lambda *a, **k: None, k_seeds=actions.shape[0],
                       close_p=close_p)
    return pol._select_seed(actions, p_contact, prev_plan, 0.0,
                            hw.control.action_rate_hz)


def test_kseed_rejects_the_timid_head_while_contact_is_unlikely():
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    acts = np.zeros((4, H, A))
    acts[0, :, 2] = -0.005          # 45 mm of head descent
    acts[1, :, 2] = -0.001          # 9 mm — under 50% of the K-max
    acts[2, :, 2] = -0.004
    acts[3, :, 2] = +0.002          # lifting
    j, rec = _sel(hw, acts, p_contact=0.05)
    assert j in (0, 2) and rec["k_rejected"] == 2
    assert rec["head_dz_spread_mm"] == pytest.approx(63.0, abs=0.1)


def test_kseed_rejection_is_disabled_once_contact_is_anticipated():
    """The rule exists to break the under-commit mode BEFORE contact; once
    p_contact is high, a shallow chunk is the correct terminal behaviour."""
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    acts = np.zeros((4, H, A))
    acts[:, :, 2] = np.array([-0.005, -0.0001, -0.004, 0.002])[:, None]
    _, rec = _sel(hw, acts, p_contact=0.9)
    assert rec["k_rejected"] == 0


def test_kseed_picks_the_chunk_nearest_the_previous_plan():
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    acts = np.zeros((3, H, A))
    acts[:, :, 2] = -0.005                        # all survive the descent gate
    acts[0, :, 0] = 0.010
    acts[1, :, 0] = 0.001
    acts[2, :, 0] = -0.010
    prev = _prev_plan(hw)
    prev.actions = np.zeros((H, A))
    prev.actions[:, 0] = 0.001
    prev.action_times = np.zeros(H)               # zero offset
    j, rec = _sel(hw, acts, p_contact=0.05, prev_plan=prev)
    assert j == 1 and rec["k_pick"] == 1


def test_kseed_falls_back_to_the_first_survivor():
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    acts = np.zeros((2, H, A))
    acts[0, :, 2] = -0.001
    acts[1, :, 2] = -0.005
    j, _ = _sel(hw, acts, p_contact=0.05, prev_plan=None)
    assert j == 1                                  # 0 rejected, 1 is first survivor


def test_kseed_spread_reaches_the_trace():
    """The K-seed head_dz spread is the P6 diagnostic — it must land in the
    planner trace entry, not only in the log."""
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    pol = _fake_policy(hw, lambda batch, **k: _pred(3, H, A, dz=[-0.005, -0.001, -0.004]),
                       k_seeds=3)
    ex = _Ex()
    loop = PlannerLoop(hw, pol, _Snaps(hw), ex)
    loop.run(max_replans=1)
    d = loop.trace[0]["diag"]
    assert d["k_seeds"] == 3 and "head_dz_spread_mm" in d


def test_single_seed_replan_is_bit_identical_to_the_old_path():
    hw = make_small_hw()
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    seen = {}

    def sample(batch, **kw):
        seen.update(kw)
        return _pred(1, H, A, dz=[-0.003])

    pol = _fake_policy(hw, sample)
    plan = pol.replan(_snap(hw), None, np.zeros(6))
    assert seen["k_seeds"] == 1
    assert plan.actions.shape == (H, A)
    assert "k_seeds" not in plan.diag           # no K bookkeeping when K == 1


# ---------------------------------------------------------------------------
# run_deploy wiring: flags, provenance tags, veto construction
# ---------------------------------------------------------------------------

def _parse(*argv):
    from phantom.scripts.run_deploy import build_parser
    return build_parser().parse_args(["--system", "teacher", "--task", "x", *argv])


def test_every_lever_defaults_off():
    a = _parse()
    assert (a.parity_fixes, a.terminal_veto, a.k_seeds) == (False, False, 1)
    assert (a.veto_p_close, a.veto_p_none, a.veto_max_retries) == (0.5, 0.9, 3)


def test_build_veto_is_none_without_the_flag():
    from phantom.scripts.run_deploy import build_veto
    assert build_veto(_parse(), None, 0.1) is None


def test_build_veto_uses_the_executor_z_floor_and_the_demo_aperture():
    from phantom.scripts.run_deploy import build_veto
    from phantom.train.common import CLOSE_ABS_POS, CLOSE_ABS_RISE
    stats = SimpleNamespace(gripper_mean=0.22)
    v = build_veto(_parse("--terminal-veto", "--veto-p-close", "0.7",
                          "--veto-max-retries", "5"), stats, 0.123)
    assert (v.p_close, v.max_retries, v.z_floor, v.open_aperture) == (0.7, 5, 0.123, 0.22)
    assert (v.close_pos, v.close_rise) == (CLOSE_ABS_POS, CLOSE_ABS_RISE)


def test_condition_tags_record_every_lever_on_or_off():
    """A/B attribution requires the arm to be reconstructable from the episode
    alone (audit 2026-08-20: the condition lived in the operator's memory)."""
    import inspect
    from phantom.scripts import run_deploy as RD
    src = inspect.getsource(RD.main)
    for frag in ("parity:", "veto:", "kseeds:"):
        assert frag in src
    assert "parity_fixes=args.parity_fixes" in src and "veto=veto" in src


def test_nfe_and_compile_document_the_latency_levers():
    from phantom.scripts.run_deploy import build_parser
    helps = {a.dest: (a.help or "") for a in build_parser()._actions}
    assert "--nfe 3" in helps["nfe"] and "bench_inference" in helps["nfe"]
    assert "bench_inference" in helps["compile"] and "GO_ANY" in helps["compile"]
    assert "PROFILE FIRST" in helps["k_seeds"]
