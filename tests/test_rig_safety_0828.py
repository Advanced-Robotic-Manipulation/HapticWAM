"""Rig 2026-08-28 safety batch: joint-space start gate (wrapped wrist / flipped
IK branch), per-task z no-go floor (clamp), STOP hitbox (demo envelope), TCP
speed cap override, and the shell gripper tool."""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager

import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.deploy import start_pose as sp
from phantom.deploy.safety import (SafetyAction, SafetyMonitor, apply_hitbox,
                                   apply_tcp_speed_limit, apply_z_floor)
from phantom.recording.ringbuffer import SharedRingBuffer
from phantom_test_utils import make_small_hw


# ---------------------------------------------------------------- stats file
def test_shipped_start_stats_carry_joint_and_envelope_fields():
    stats = sp.load_start_stats()
    for task in ("Carton", "egg", "waffles", "whiteboard"):
        st = stats[task]
        assert st.q_mean is not None and st.q_mean.shape == (6,)
        assert st.q_std is not None and np.all(st.q_std >= 0)
        assert st.tcp_z_min is not None and 0.03 < st.tcp_z_min < 0.12
        assert st.tcp_min is not None and st.tcp_max is not None
        assert np.all(st.tcp_max > st.tcp_min)
        # the floor sits at the bottom of the envelope, the start pose inside it
        assert abs(st.tcp_min[2] - st.tcp_z_min) < 1e-6
        assert np.all(st.tcp_min <= st.tcp_mean[:3]) and np.all(st.tcp_mean[:3] <= st.tcp_max)
        # wrist 3 of every task's demos sits near -pi: the wrapped +pi reading
        # seen on 08-28 must be a full turn away, not a rounding away
        assert st.q_mean[5] < -2.9


def test_loader_rejects_half_specified_joint_stats(tmp_path):
    y = tmp_path / "sp.yaml"
    y.write_text("tasks:\n  t:\n    n: 30\n    tcp_mean: [0,0,0.3,0,3.1,0]\n    tcp_std: [0.01,0.01,0.01,0.1,0.1,0.1]\n"
                 "    gripper_mean: 0.2\n    gripper_std: 0.1\n    q_mean: [0,0,0,0,0,0]\n")
    with pytest.raises(AssertionError):
        sp.load_start_stats(y)


# ---------------------------------------------------------------- joint gate
def _stats():
    return sp.load_start_stats()["waffles"]


def test_joint_gate_passes_the_demo_configuration():
    st = _stats()
    sig, table = sp.start_sigma_report(st, st.tcp_mean, st.gripper_mean, q=st.q_mean)
    assert sig.shape == (6 + 1 + 6,)
    assert np.all(sig[7:] == 0.0)
    assert "q6:" in table and "FULL-TURN" not in table


def test_joint_gate_refuses_a_wrist_wrapped_by_a_full_turn():
    """Same flange orientation, joint reading +2pi: the TCP pose gates fine,
    the RAW joint vector the policy consumes is ~150 sigma out."""
    st = _stats()
    q = st.q_mean.copy()
    q[5] += 2 * np.pi
    sig, table = sp.start_sigma_report(st, st.tcp_mean, st.gripper_mean, q=q)
    assert sig[:7].max() == 0.0                    # pose + gripper: perfect
    assert sig[7 + 5] > 2.5 and np.isfinite(sig[7 + 5])
    assert "FULL-TURN" in table and "q6" in table


def test_joint_gate_refuses_a_flipped_ik_branch():
    st = _stats()
    q = st.q_mean.copy()
    q[0] -= np.radians(127)                        # base rotated (08-28 Carton eps)
    q[2] = -q[2]                                   # elbow flipped
    sig, _ = sp.start_sigma_report(st, st.tcp_mean, st.gripper_mean, q=q)
    assert sig[7:].max() > 2.5


def test_joint_gate_tolerates_small_offsets_via_std_floor():
    """A 0.5 deg demo std must not turn a 4 deg offset into a refusal."""
    st = _stats()
    assert np.degrees(st.q_std[1]) < 1.0            # shoulder is repeated tightly
    q = st.q_mean.copy()
    q[1] += np.radians(4.0)
    sig, _ = sp.start_sigma_report(st, st.tcp_mean, st.gripper_mean, q=q)
    assert sig[7 + 1] < 1.0


def test_joint_gate_non_finite_reading_gates_out():
    st = _stats()
    q = st.q_mean.copy(); q[3] = np.nan
    sig, _ = sp.start_sigma_report(st, st.tcp_mean, st.gripper_mean, q=q)
    assert np.isinf(sig[7 + 3])


def test_joint_gate_absent_without_stats_or_q():
    st = _stats()
    sig, _ = sp.start_sigma_report(st, st.tcp_mean, st.gripper_mean)
    assert sig.shape == (7,)
    from dataclasses import replace
    st2 = replace(st, q_mean=None, q_std=None)
    sig, _ = sp.start_sigma_report(st2, st.tcp_mean, st.gripper_mean, q=st.q_mean)
    assert sig.shape == (7,)


def test_home_joints_moves_j_before_l():
    class Arm:
        calls = []
        def move_j(self, q, s, a, blocking=True): self.calls.append(("j", np.asarray(q).copy()))
        def move_l(self, p, s, a, blocking=True): self.calls.append(("l", np.asarray(p).copy()))
    class Grip:
        def move(self, *a): pass
        def get_state(self):
            from types import SimpleNamespace
            return SimpleNamespace(obj=3.0, moving=False, position=0.2)
    hw = make_small_hw()
    st = _stats()
    arm = Arm(); arm.calls = []
    sp.move_to_start(arm, Grip(), hw, st, rng=np.random.default_rng(0), home_joints=True)
    assert [c[0] for c in arm.calls] == ["j", "l"]
    assert np.allclose(arm.calls[0][1], st.q_mean)
    arm2 = Arm(); arm2.calls = []
    sp.move_to_start(arm2, Grip(), hw, st, rng=np.random.default_rng(0))
    assert [c[0] for c in arm2.calls] == ["l"]


# ---------------------------------------------------------------- z floor / hitbox / speed
@contextmanager
def _rings(hw):
    uid = uuid.uuid4().hex[:8]
    h, w, c = hw.cameras.scene.color.hwc
    rings = {
        "arm": SharedRingBuffer(f"s_arm_{uid}", 16, {
            "ft": ((6,), "float64"), "protective_stop": ((), "uint8")}, create=True),
        "camera_scene": SharedRingBuffer(f"s_cam_{uid}", 4, {
            "color": ((h, w, c), "uint8")}, create=True),
    }
    try:
        yield rings
    finally:
        for r in rings.values():
            r.close()


def _fresh(rings, hw, t):
    rings["arm"].push(t, ft=np.zeros(6), protective_stop=np.uint8(0))
    rings["camera_scene"].push(t, color=np.zeros(hw.cameras.scene.color.hwc, dtype=np.uint8))


def test_z_floor_raises_never_lowers_and_clamps():
    hw = make_small_hw()
    ws = hw.safety.workspace_m
    hw2 = apply_z_floor(hw, ws.z[0] + 0.05)
    assert hw2.safety.workspace_m.z[0] == pytest.approx(ws.z[0] + 0.05)
    assert hw2.safety.workspace_m.z[1] == ws.z[1]
    assert hw2.safety.workspace_m.x == ws.x and hw2.safety.workspace_m.y == ws.y
    assert hw.safety.workspace_m.z[0] == ws.z[0]          # original untouched (frozen copy)
    hw3 = apply_z_floor(hw, ws.z[0] - 0.05)               # cannot go below the yaml bound
    assert hw3.safety.workspace_m.z[0] == ws.z[0]
    with pytest.raises(ValueError):
        apply_z_floor(hw, ws.z[1] + 0.1)
    # shape-relevant fields (checkpoint compat) untouched
    assert hw2.shape_relevant_fields() == hw.shape_relevant_fields()
    with _rings(hw2) as rings:
        mon = SafetyMonitor(hw2, rings)
        t = time.perf_counter(); _fresh(rings, hw2, t)
        mid = np.array([np.mean(ws.x), np.mean(ws.y), ws.z[0] + 0.01, 0, 3.14, 0])
        v = mon.check(t, mid)
        assert v.action == SafetyAction.CLAMP
        assert mon.clamp_target(mid)[2] == pytest.approx(ws.z[0] + 0.05)


def test_hitbox_exit_stops_the_episode_inside_it_is_ok():
    hw = make_small_hw()
    ws = hw.safety.workspace_m
    lo = np.array([np.mean(ws.x) - 0.05, np.mean(ws.y) - 0.05, ws.z[0] + 0.05])
    hi = lo + 0.10
    hw2 = apply_hitbox(hw, lo, hi, margin_m=0.01)
    hb = hw2.safety.hitbox_m
    assert hb is not None and hb.x[0] == pytest.approx(lo[0] - 0.01) and hb.z[1] == pytest.approx(hi[2] + 0.01)
    assert hw.safety.hitbox_m is None
    with _rings(hw2) as rings:
        mon = SafetyMonitor(hw2, rings)
        t = time.perf_counter(); _fresh(rings, hw2, t)
        inside = np.array([*(lo + 0.05), 0, 3.14, 0])
        assert mon.check(t, inside).action == SafetyAction.OK
        outside = inside.copy(); outside[1] = hi[1] + 0.02          # 2 cm past the margin, inside the workspace
        assert ws.contains(outside[:3])
        v = mon.check(t, outside)
        assert v.action == SafetyAction.STOP_EPISODE
        assert any(e.kind == "hitbox_exit" for e in v.events)


def test_hitbox_is_intersected_with_the_workspace():
    hw = make_small_hw()
    ws = hw.safety.workspace_m
    hw2 = apply_hitbox(hw, [ws.x[0] - 1, ws.y[0] - 1, ws.z[0] - 1], [ws.x[1] + 1, ws.y[1] + 1, ws.z[1] + 1], 0.0)
    hb = hw2.safety.hitbox_m
    assert hb.x == ws.x and hb.y == ws.y and hb.z == ws.z
    with pytest.raises(ValueError):
        apply_hitbox(hw, [ws.x[1] + 1, 0, 0], [ws.x[1] + 2, 1, 1], 0.0)


def test_tcp_speed_limit_lowers_never_raises():
    hw = make_small_hw()
    v0 = hw.arm.limits.tcp_speed_m_s
    assert apply_tcp_speed_limit(hw, v0 / 2).arm.limits.tcp_speed_m_s == pytest.approx(v0 / 2)
    assert apply_tcp_speed_limit(hw, v0 * 2).arm.limits.tcp_speed_m_s == pytest.approx(v0)
    with pytest.raises(ValueError):
        apply_tcp_speed_limit(hw, 0.0)


def test_resolve_z_floor_rules():
    from types import SimpleNamespace as NS
    from phantom.scripts.run_deploy import resolve_z_floor
    st = _stats()
    assert resolve_z_floor(NS(no_z_floor=False, z_floor=None, z_floor_margin=0.01), st) == pytest.approx(st.tcp_z_min - 0.01)
    assert resolve_z_floor(NS(no_z_floor=False, z_floor=0.07, z_floor_margin=0.01), st) == 0.07
    assert resolve_z_floor(NS(no_z_floor=True, z_floor=0.07, z_floor_margin=0.01), st) is None
    assert resolve_z_floor(NS(no_z_floor=False, z_floor=None, z_floor_margin=0.01), None) is None


def test_run_deploy_parser_defaults():
    from phantom.scripts.run_deploy import (DEFAULT_MAX_REPLANS, build_parser,
                                            resolve_max_replans)
    a = build_parser().parse_args(["--system", "teacher", "--task", "waffles"])
    # --max-replans parses to None so main can tell "40" from "unspoken"
    # (revalidation 2026-08-31 #1); 40 is still the cap with no wall clock.
    assert a.max_replans is None and DEFAULT_MAX_REPLANS == 40
    a.max_episode_s = 0
    assert resolve_max_replans(a) == 40
    assert a.hitbox_margin == 0.03 and a.z_floor_margin == 0.01
    assert not a.home_joints and not a.no_hitbox and not a.no_z_floor and a.max_tcp_speed is None


# ---------------------------------------------------------------- gripper tool
def test_gripper_ctl_open_close_status_on_mock():
    from phantom.scripts import gripper_ctl as G
    hw = load_hardware(None)
    assert hw.mode.resolve("gripper") == "mock"
    assert G.main(["open"]) == 0
    assert G.main(["close", "--pos", "0.5"]) == 0
    assert G.main(["status"]) == 0
    assert G.main(["reset"]) == 0


def test_floor_is_a_clamp_even_with_the_hitbox_armed():
    """Review 2026-08-28: with the hitbox intersected against the already-raised
    floor, a target below the floor tripped hitbox_exit (STOP) instead of being
    clamped. A deep descent must be pinned at the floor, never stopped."""
    from phantom.scripts.run_deploy import build_parser, resolve_z_floor
    hw = make_small_hw()
    ws = hw.safety.workspace_m
    lo = np.array([np.mean(ws.x) - 0.05, np.mean(ws.y) - 0.05, ws.z[0] + 0.03])
    hi = lo + 0.10
    hw2 = apply_hitbox(hw, lo, hi, margin_m=0.03)          # hitbox first (as run_deploy does now)
    floor = lo[2] + 0.02
    hw2 = apply_z_floor(hw2, floor)
    assert hw2.safety.hitbox_m.z[0] < hw2.safety.workspace_m.z[0]
    with _rings(hw2) as rings:
        mon = SafetyMonitor(hw2, rings)
        t = time.perf_counter(); _fresh(rings, hw2, t)
        deep = np.array([lo[0] + 0.05, lo[1] + 0.05, ws.z[0] - 0.5, 0, 3.14, 0])   # far below everything
        v = mon.check(t, deep)
        assert v.action == SafetyAction.CLAMP, [e.kind for e in v.events]
        assert not any(e.kind == "hitbox_exit" for e in v.events)
        assert mon.clamp_target(deep)[2] == pytest.approx(floor)
        side = np.array([hi[0] + 0.1, lo[1] + 0.05, lo[2] + 0.05, 0, 3.14, 0])     # lateral exit still stops
        assert mon.check(t, side).action == SafetyAction.STOP_EPISODE
    # run_deploy order: hitbox applied before the floor
    src = open("phantom/scripts/run_deploy.py").read()
    assert src.index("apply_hitbox(") < src.index("apply_z_floor(")


def test_episode_seed_is_random_without_seed_and_reproducible_with():
    """rf.py seeds its generator to a constant; run_deploy must not let a rig
    session sample the same noise as every other session (2026-08-29)."""
    from phantom.scripts.run_deploy import episode_seed
    a, b = episode_seed(None, 0), episode_seed(None, 0)
    assert a != b and 0 <= a < 2**32
    assert episode_seed(7, 0) == 7 and episode_seed(7, 3) == 10
