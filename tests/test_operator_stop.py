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
