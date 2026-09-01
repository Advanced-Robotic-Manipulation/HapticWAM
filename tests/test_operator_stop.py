"""Operator stop (rig session 2026-09-01): pressing Enter during an episode
must end the replan loop cleanly through the same path as the caps —
stop_reason set, no gripper release, no executor safety stop — so a manual
"the grasp is done" never drops a held object the way a let-go stop would.
"""
from __future__ import annotations

import numpy as np

from phantom.deploy.planner import PlannerLoop, TerminalVeto
from phantom_test_utils import make_small_hw

from test_veto_guards_0831 import _Ex, _Pol, _Snaps

VETO = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=3, z_floor=0.10,
                    z_margin=0.015, open_aperture=0.25)


def test_operator_stop_ends_episode_with_reason():
    hw = make_small_hw()
    ex = _Ex()
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    lp = PlannerLoop(hw, _Pol(hw, p_none=[0.05], grip_cmd=[0.2]),
                     _Snaps(hw, z=0.25, grips=(0.30,)), ex, veto=VETO)
    lp.run(max_replans=50, stop_check=stop_after_two)
    assert lp.stop_reason == "operator_stop"
    # stopped well before the replan cap: the check ended it
    assert len(lp.trace) <= 4


def test_operator_stop_does_not_touch_the_gripper():
    hw = make_small_hw()
    ex = _Ex()
    lp = PlannerLoop(hw, _Pol(hw, p_none=[0.05], grip_cmd=[0.2]),
                     _Snaps(hw, z=0.25, grips=(0.30,)), ex, veto=VETO)
    lp.run(max_replans=50, stop_check=lambda: True)
    assert lp.stop_reason == "operator_stop"
    # clean exit: no safety stop reached the executor, so no let-go/release
    assert ex.stopped_reason is None


def test_no_stop_check_keeps_cap_semantics():
    hw = make_small_hw()
    ex = _Ex()
    lp = PlannerLoop(hw, _Pol(hw, p_none=[0.05], grip_cmd=[0.2]),
                     _Snaps(hw, z=0.25, grips=(0.30,)), ex, veto=VETO)
    lp.run(max_replans=3)
    assert lp.stop_reason == "replan_cap"


def test_make_operator_stop_disabled_off_tty():
    # the entry point: run_deploy's poller must arm only on a real tty —
    # piped/test stdin returns None and the episode keeps cap semantics
    from phantom.scripts.run_deploy import make_operator_stop
    assert make_operator_stop() is None


def test_hitbox_top_exit_is_not_a_letgo():
    """Rig 2026-09-01: the first tactile-confirmed grasp lifted past the
    hitbox ceiling and 'hitbox_exit' (a let-go reason) opened the fingers at
    z=0.49 m — the object was dropped. A pure top-face exit must stop the
    episode WITHOUT releasing; every other face keeps let-go semantics."""
    import time as _time

    import pytest
    from phantom.deploy.executor import is_letgo_reason
    from phantom.deploy.safety import SafetyAction, SafetyMonitor, apply_hitbox
    from test_rig_safety_0828 import _fresh, _rings

    assert not is_letgo_reason("hitbox_exit_top")
    assert is_letgo_reason("hitbox_exit")

    hw = make_small_hw()
    ws = hw.safety.workspace_m
    lo = np.array([np.mean(ws.x) - 0.05, np.mean(ws.y) - 0.05, ws.z[0] + 0.02])
    hi = lo + np.array([0.10, 0.10, 0.10])
    hw2 = apply_hitbox(hw, lo, hi, margin_m=0.01)
    hb = hw2.safety.hitbox_m
    with _rings(hw2) as rings:
        mon = SafetyMonitor(hw2, rings)
        t = _time.perf_counter(); _fresh(rings, hw2, t)
        centre = (lo + hi) / 2
        top = np.array([*centre[:2], hb.z[1] + 0.02, 0, 3.14, 0])
        v = mon.check(t, top)
        assert v.action == SafetyAction.STOP_EPISODE
        kinds = [e.kind for e in v.events]
        assert "hitbox_exit_top" in kinds and "hitbox_exit" not in kinds, kinds
        side = np.array([hb.x[1] + 0.02, centre[1], centre[2], 0, 3.14, 0])
        v2 = mon.check(t + 1.0, side)
        assert any(e.kind == "hitbox_exit" for e in v2.events)


def test_joint_speed_whip_stops_without_letgo():
    """Rig 2026-09-01 #2: carrying the grasp toward full extension, servoL
    whipped the wrists to 5-7 rad/s on a legal Cartesian step (elbow
    singularity) — the Cartesian rate limiter cannot see it. The measured-qd
    guard must stop the episode, and the reason must NOT be a let-go."""
    import contextlib
    import time as _time
    import uuid as _uuid

    from phantom.deploy.executor import is_letgo_reason
    from phantom.deploy.safety import SafetyAction, SafetyMonitor
    from phantom.recording.ringbuffer import SharedRingBuffer

    assert not is_letgo_reason("joint_speed")
    hw = make_small_hw()
    ws = hw.safety.workspace_m
    mid = np.array([np.mean(ws.x), np.mean(ws.y), np.mean(ws.z), 0, 3.14, 0])
    uid = _uuid.uuid4().hex[:8]
    h, w, c = hw.cameras.scene.color.hwc
    rings = {
        "arm": SharedRingBuffer(f"s_jsa_{uid}", 16, {
            "ft": ((6,), "float64"), "protective_stop": ((), "uint8"),
            "qd": ((hw.arm.dof,), "float64")}, create=True),
        "camera_scene": SharedRingBuffer(f"s_jsc_{uid}", 4, {
            "color": ((h, w, c), "uint8")}, create=True),
    }
    with contextlib.ExitStack() as stack:
        for r in rings.values():
            stack.callback(r.close)
        mon = SafetyMonitor(hw, rings)
        t = _time.perf_counter()
        calm = np.zeros(hw.arm.dof)
        rings["arm"].push(t, ft=np.zeros(6), protective_stop=np.uint8(0), qd=calm)
        rings["camera_scene"].push(t, color=np.zeros((h, w, c), dtype=np.uint8))
        assert mon.check(t, mid).action == SafetyAction.OK
        whip = calm.copy(); whip[-1] = 6.9
        rings["arm"].push(t + 0.01, ft=np.zeros(6), protective_stop=np.uint8(0), qd=whip)
        v = mon.check(t + 0.01, mid)
        assert v.action == SafetyAction.STOP_EPISODE
        assert any(e.kind == "joint_speed" for e in v.events)


def test_reach_clamp_caps_commanded_radius():
    """Predictive layer: a command past the singular radius is scaled back
    (direction preserved) and reported as a CLAMP, never a stop — the policy
    just cannot extend the arm into IK-degenerate territory."""
    import time as _time

    from phantom.deploy.safety import SafetyAction, SafetyMonitor
    from test_rig_safety_0828 import _fresh, _rings

    hw = make_small_hw()
    r_max = hw.safety.reach_clamp_m
    assert r_max is not None and r_max > 0
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        t = _time.perf_counter(); _fresh(rings, hw, t)
        far = np.array([2.0, 2.0, 2.0, 0, 3.14, 0])
        out = mon.clamp_target(far)
        # radial scaling happened before the box clamp
        direction = far[:3] / np.linalg.norm(far[:3])
        assert np.linalg.norm(far[:3]) > r_max
        scaled = direction * r_max
        ws = hw.safety.workspace_m
        expect = np.array([np.clip(scaled[0], *ws.x), np.clip(scaled[1], *ws.y),
                           np.clip(scaled[2], *ws.z)])
        assert np.allclose(out[:3], expect)
        v = mon.check(t, far)
        assert any(e.kind == "reach_clamp" for e in v.events)
        assert v.action in (SafetyAction.CLAMP, SafetyAction.STOP_EPISODE)
        near = np.array([np.mean(ws.x), np.mean(ws.y), np.mean(ws.z), 0, 3.14, 0])
        if np.linalg.norm(near[:3]) < r_max:
            assert np.allclose(mon.clamp_target(near)[:3], near[:3])


def test_lift_complete_ends_episode_holding():
    """Success auto-stop through the real planner loop: tactile-confirmed
    grasp above the lift height -> stop_reason 'lift_complete', no executor
    safety stop (gripper held)."""
    hw = make_small_hw()
    ex = _Ex()
    lp = PlannerLoop(hw, _Pol(hw, p_none=[0.05], grip_cmd=[0.8]),
                     _Snaps(hw, z=0.40, grips=(0.30,)), ex, veto=VETO)
    lp.run(max_replans=50, success_check=lambda: True)
    assert lp.stop_reason == "lift_complete"
    assert ex.stopped_reason is None


def test_make_lift_complete_closure_thresholds():
    """The run_deploy closure against fake rings: fires only when BOTH pads
    are loaded AND z is above the lift height, sustained past hold_s."""
    import time as _time
    from types import SimpleNamespace

    from phantom.scripts.run_deploy import make_lift_complete

    class _Ring:
        def __init__(self):
            self.sample = {}

        def latest(self, n):
            return np.array([_time.time()]), self.sample

    rings = {"arm": _Ring(), "tactile_left": _Ring(), "tactile_right": _Ring()}
    rt = SimpleNamespace(session=SimpleNamespace(rings=rings))

    def set_state(z, fl, fr):
        rings["arm"].sample = {"tcp_pose": [np.array([0, 0, z, 0, 3.14, 0])]}
        rings["tactile_left"].sample = {"wrench": [np.array([0, 0, fl, 0, 0, 0])]}
        rings["tactile_right"].sample = {"wrench": [np.array([0, 0, fr, 0, 0, 0])]}

    assert make_lift_complete(rt, 0.0, 3.0, 0.5) is None      # disabled
    chk = make_lift_complete(rt, 0.35, 3.0, hold_s=0.05)
    set_state(z=0.40, fl=8.0, fr=0.1)          # one pad only -> never
    assert not chk()
    set_state(z=0.20, fl=8.0, fr=6.0)          # grasped but not lifted
    assert not chk()
    set_state(z=0.40, fl=8.0, fr=6.0)          # good: arms the hold window
    assert not chk()
    _time.sleep(0.06)
    assert chk()                                # sustained -> fire
    set_state(z=0.40, fl=0.0, fr=6.0)          # condition drops -> re-arm
    assert not chk()
