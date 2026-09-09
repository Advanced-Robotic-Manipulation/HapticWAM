"""Revalidation 2026-08-31 §2 #3/#5/#7 guards.

#3  the veto must not self-cancel: a MASKED close must never arm the
    phantom-grasp recovery, and a close allowed ONLY by the at_floor hatch
    (the one door open on the 08-28 gate statistics) must never be reopened
    by a high p_none — on that data the recovery would have ended 5/18 real
    grasp attempts in `veto_retry_cap`.
#5  the gripper release must win the race against a latched close (shared
    I/O lock), and a protective stop that co-occurs with a let-go event
    must release too.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np

from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.planner import PlannerLoop, TerminalVeto
from phantom.inference.policy import ObsSnapshot, Plan
from phantom_test_utils import make_small_hw


class _Ex:
    def __init__(self, enter=True):
        self.stopped_reason = None
        self.submitted = []
        self.enter = enter
        self._hist = []
        self._t = 0.0

    def last_cmd(self):
        return None

    def request_stop(self, reason):
        if self.stopped_reason is None:
            self.stopped_reason = reason

    def submit(self, plan):
        self.submitted.append(np.array(plan.actions, copy=True))
        if self.enter:
            for g in plan.actions[:, 6]:
                self._t += 0.1
                self._hist.append((self._t, float(g)))
        return True

    def entered_grip_after(self, t):
        return [(ts, g) for ts, g in self._hist if ts > t]


def _snap(hw, z, grip):
    s = ObsSnapshot(t=time.perf_counter(), rgb=np.zeros((4, 4, 3), np.uint8),
                    wrist_window=np.zeros((hw.wrist_ft.window_len, 6), np.float32),
                    ur_state=np.zeros(2 * hw.arm.dof + 14, np.float32))
    s.ur_state[2 * hw.arm.dof + 2] = z
    s.ur_state[-2] = grip
    return s


class _Snaps:
    def __init__(self, hw, z=0.25, grips=(0.30,)):
        self.hw, self.z, self.grips, self.n = hw, z, list(grips), 0

    def build(self):
        g = self.grips[min(self.n, len(self.grips) - 1)]
        self.n += 1
        return _snap(self.hw, self.z, g)


class _Pol:
    def __init__(self, hw, p_none, grip_cmd=(0.8,), dz=+0.01):
        self.hw, self.p_none, self.grip_cmd, self.dz = hw, list(p_none), list(grip_cmd), dz
        self.n = 0

    def replan(self, snap, prev_plan, tcp_pose):
        H, A = self.hw.control.chunk_horizon, self.hw.control.action_dim
        a = np.zeros((H, A)); a[:, 2] = self.dz
        a[:, 6] = self.grip_cmd[min(self.n, len(self.grip_cmd) - 1)]
        pn = self.p_none[min(self.n, len(self.p_none) - 1)]
        self.n += 1
        p = Plan(t_created=float(self.n), t0_pose=np.asarray(tcp_pose, float).copy(),
                 actions=a, action_times=np.arange(H) / 10.0, sigma=np.zeros(3),
                 gate=0.0, p_evt=np.array([pn, 1 - pn, 0, 0, 0]), cpk="CPK",
                 latency_s=0.9)
        p.diag = {}
        return p


VETO = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=3, z_floor=0.10,
                    z_margin=0.015, open_aperture=0.25)


def _acts(trace):
    return [t["terminal_veto"]["action"] for t in trace]


def test_a_masked_close_never_arms_the_recovery():
    """The rig-shaped case: every close is masked (p_contact 0.05), yet the
    MEASURED aperture ramps past the close rule anyway (open-loop ramp).
    Pre-guard, _note_executed_close read that ramp back as an executed close
    and the recovery reopened + counted retries on closes the veto prevented."""
    hw = make_small_hw()
    ex = _Ex()
    lp = PlannerLoop(hw, _Pol(hw, p_none=[0.95], grip_cmd=[0.8]), 
                     _Snaps(hw, z=0.25, grips=(0.20, 0.30, 0.50, 0.65, 0.70, 0.72)),
                     ex, veto=VETO)
    lp.run(max_replans=6)
    acts = _acts(lp.trace)
    assert set(acts) <= {"close_masked", "none"}, acts
    assert all(t["terminal_veto"]["retries"] == 0 for t in lp.trace)
    assert ex.stopped_reason != "veto_retry_cap"


def test_a_floor_only_close_is_never_reopened_by_high_p_none():
    """z at the floor band, gate says nothing (p_contact 0.05): the close is
    allowed ONLY by the at_floor hatch. On the 08-28 statistics p_none > 0.9
    follows on 80% of replans — the recovery must NOT fire on that latch."""
    hw = make_small_hw()
    ex = _Ex()
    lp = PlannerLoop(hw, _Pol(hw, p_none=[0.95], grip_cmd=[0.8]),
                     _Snaps(hw, z=0.10, grips=(0.30, 0.30, 0.60, 0.62, 0.63, 0.63)),
                     ex, veto=VETO)
    lp.run(max_replans=6)
    acts = _acts(lp.trace)
    assert "close_allowed" in acts
    assert "recovery_open" not in acts and "retry_cap" not in acts
    assert "recovery_skipped_floor_close" in acts, acts
    assert all(t["terminal_veto"]["retries"] == 0 for t in lp.trace)
    # the chunk's gripper command was left alone on the skip replans
    assert all(np.all(a[:, 6] >= 0.79) for a in ex.submitted[2:3])


def test_a_gate_allowed_close_still_recovers_on_phantom_grasp():
    """Control: when the GATE allowed the close (p_contact 0.9), a later
    p_none > 0.9 still triggers exactly one reopen — the guard must not
    disable the recovery it was built to protect."""
    hw = make_small_hw()
    ex = _Ex()
    lp = PlannerLoop(hw, _Pol(hw, p_none=[0.05, 0.05, 0.95, 0.95, 0.2], grip_cmd=[0.8]),
                     _Snaps(hw, z=0.25, grips=(0.30, 0.30, 0.60, 0.62, 0.62)),
                     ex, veto=VETO)
    lp.run(max_replans=5)
    acts = _acts(lp.trace)
    assert "close_allowed" in acts and "recovery_open" in acts, acts


# --------------------------------------------------------------------------
# #5 — release ordering, lock, protective stop
# --------------------------------------------------------------------------

class _SlowGrip:
    def __init__(self, rtt_s=0.015):
        self.moves = []
        self.rtt = rtt_s
        self._lockcheck = threading.Lock()

    def move(self, pos, speed, force):
        with self._lockcheck:                       # detects overlapping I/O
            time.sleep(self.rtt)
            self.moves.append(float(pos))

    def get_state(self):
        return SimpleNamespace(t_host=time.perf_counter(),
                               position=self.moves[-1] if self.moves else 0.3,
                               obj=3.0, moving=False)


def _executor(hw, grip):
    return ChunkExecutor(hw, SimpleNamespace(servo_stop=lambda: None,
                                             stop=lambda d: None),
                         grip, SimpleNamespace(check=None), gripper_ring=None)


def test_the_release_wins_the_race_with_a_latched_close():
    hw = make_small_hw()
    for _ in range(10):
        g = _SlowGrip(rtt_s=0.005)
        ex = _executor(hw, g)
        worker = threading.Thread(target=ex._grip_worker, daemon=True)
        worker.start()
        ex._grip_target = 0.9                       # latched close
        time.sleep(0.002)                           # let the worker pick it up
        ex._halt("wrench_limit")
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert g.moves, "release never ran"
        assert g.moves[-1] == ex.open_aperture, g.moves


def test_protective_stop_with_a_letgo_event_releases():
    hw = make_small_hw()
    g = _SlowGrip(rtt_s=0.0)
    ex = _executor(hw, g)
    ex._halt("protective_stop",
             events=[SimpleNamespace(kind="tactile_fz")])
    assert g.moves and g.moves[-1] == ex.open_aperture


def test_a_bare_protective_stop_does_not_release():
    hw = make_small_hw()
    g = _SlowGrip(rtt_s=0.0)
    ex = _executor(hw, g)
    ex._halt("protective_stop")
    assert g.moves == []
