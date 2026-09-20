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
    episode WITHOUT releasing. Side-face exits now retain the accepted
    gripper target too; all faces remain episode stops."""
    import time as _time

    import pytest
    from phantom.deploy.executor import is_letgo_reason
    from phantom.deploy.safety import SafetyAction, SafetyMonitor, apply_hitbox
    from test_rig_safety_0828 import _fresh, _rings

    assert not is_letgo_reason("hitbox_exit_top")
    assert not is_letgo_reason("hitbox_exit")

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
    # retired as a DEFAULT (0/4 whips; fought one lift) but the mechanism
    # stays available — enable it explicitly here
    assert hw.safety.reach_clamp_m is None
    hw = hw.model_copy(update={"safety": hw.safety.model_copy(
        update={"reach_clamp_m": 0.62})})
    r_max = hw.safety.reach_clamp_m
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
    # first call captures the per-episode tactile BASELINE (rig 09-01: the
    # left pad carried a drifting 0.7-2.2 N zero offset) — simulate a biased
    # untouched pad and verify later readings are judged vs that baseline
    set_state(z=0.20, fl=1.5, fr=0.05)
    assert not chk()                            # baseline capture, never fires
    set_state(z=0.40, fl=9.5, fr=0.15)          # one pad only (vs baseline)
    assert not chk()
    set_state(z=0.20, fl=9.5, fr=6.1)           # grasped but not lifted
    assert not chk()
    set_state(z=0.40, fl=3.5, fr=2.1)           # only 2.0 N over baseline -> no
    assert not chk()
    set_state(z=0.40, fl=9.5, fr=6.1)           # good: arms the hold window
    assert not chk()
    _time.sleep(0.06)
    assert chk()                                # sustained -> fire
    set_state(z=0.40, fl=1.5, fr=6.1)           # condition drops -> re-arm
    assert not chk()


def test_servo_l_rejects_ik_branch_flip():
    """Rig 2026-09-01 root cause: host-side IK with no qnear
    seed returned a different solution branch at the elbow-straight boundary
    and servoJ swept a ballistic arc to it. servo_l must (a) seed IK with the
    previous solution, (b) refuse to stream a flipped solution, (c) give up
    loudly after sustained rejects — and never send the flipped q."""
    from types import SimpleNamespace

    from phantom.drivers.real.ur import URArm

    hw = make_small_hw()
    arm = URArm.__new__(URArm)          # no hardware: wire the internals
    arm._servo_active = False
    arm._recv = None
    arm._seq = 0
    arm._want_control = False
    arm._last_qsol = [0.0] * 6
    arm._ik_rejects = 0
    arm._branch_rejects = 0
    arm._hold_since = None
    arm._ik_dev_max = 0.0
    arm._ik_rejects_total = 0
    import threading as _th
    arm._ctrl_lock = _th.Lock()
    sent = []
    flipped = [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]      # wrist flipped a branch

    class _Ctrl:
        def __init__(self, sol):
            self.sol = sol
            self.qnear_seen = []

        def getInverseKinematics(self, pose, qnear=None):
            self.qnear_seen.append(qnear)
            return list(self.sol)

        def servoJ(self, q, *a):
            sent.append(list(q))
            return True

    ctrl = _Ctrl(flipped)
    arm._require_ctrl = lambda: ctrl
    # flipped solution: never streamed, seed passed, reject counted
    arm.servo_l(np.zeros(6), 0.008, 0.03, 300)
    assert sent == [] and arm._ik_rejects == 1
    assert ctrl.qnear_seen[-1] == [0.0] * 6
    # sustained rejects end in a loud error, still nothing streamed
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="branch"):
        for _ in range(URArm.IK_REJECT_LIMIT + 1):
            arm.servo_l(np.zeros(6), 0.008, 0.03, 300)
    assert sent == []
    # an on-branch solution streams normally and re-seeds
    ctrl.sol = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
    arm.servo_l(np.zeros(6), 0.008, 0.03, 300)
    assert len(sent) == 1 and arm._ik_rejects == 0
    assert arm._last_qsol == ctrl.sol


def test_wrist_extension_guard_stops_before_the_boundary():
    """The guard with real lead (run analysis 09-01): wrist-centre distance
    from the shoulder is pure elbow geometry; > 0.45 m stops the episode
    0.3-1.4 s before an IK branch flip is even reachable. Non-letgo."""
    import contextlib
    import time as _time
    import uuid as _uuid

    from phantom.deploy.executor import is_letgo_reason
    from phantom.deploy.safety import SafetyAction, SafetyMonitor
    from phantom.recording.ringbuffer import SharedRingBuffer

    assert not is_letgo_reason("wrist_extension")
    hw = make_small_hw()
    assert hw.safety.wrist_extension_stop_m == 0.462
    ws = hw.safety.workspace_m
    mid = np.array([np.mean(ws.x), np.mean(ws.y), np.mean(ws.z), 0, 3.14, 0])
    uid = _uuid.uuid4().hex[:8]
    h, w, c = hw.cameras.scene.color.hwc
    rings = {
        "arm": SharedRingBuffer(f"s_wea_{uid}", 16, {
            "ft": ((6,), "float64"), "protective_stop": ((), "uint8"),
            "q": ((hw.arm.dof,), "float64"),
            "qd": ((hw.arm.dof,), "float64")}, create=True),
        "camera_scene": SharedRingBuffer(f"s_wec_{uid}", 4, {
            "color": ((h, w, c), "uint8")}, create=True),
    }
    with contextlib.ExitStack() as stack:
        for r in rings.values():
            stack.callback(r.close)
        mon = SafetyMonitor(hw, rings)
        t = _time.perf_counter()
        bent = np.zeros(hw.arm.dof); bent[2] = np.radians(77)   # wd ~375 mm
        rings["arm"].push(t, ft=np.zeros(6), protective_stop=np.uint8(0),
                          q=bent, qd=np.zeros(hw.arm.dof))
        rings["camera_scene"].push(t, color=np.zeros((h, w, c), dtype=np.uint8))
        assert mon.check(t, mid).action == SafetyAction.OK
        near_straight = np.zeros(hw.arm.dof); near_straight[2] = np.radians(15)  # wd ~467 mm
        rings["arm"].push(t + 0.01, ft=np.zeros(6), protective_stop=np.uint8(0),
                          q=near_straight, qd=np.zeros(hw.arm.dof))
        v = mon.check(t + 0.01, mid)
        assert v.action == SafetyAction.STOP_EPISODE
        assert any(e.kind == "wrist_extension" for e in v.events)


def test_chunk_tail_cap_holds_at_the_cap_point():
    """P1 #5 (run analysis 09-01): all four whips began in the unsupervised
    chunk tail (steps >= HEAD_STEPS, played while the planner was blocked in
    inference). With max_play_steps the playback pose FREEZES at the cap and
    the gripper command freezes with it; uncapped keeps advancing (legacy)."""
    import time as _time

    from phantom.deploy.executor import ChunkExecutor
    from phantom.inference.policy import Plan

    hw = make_small_hw()
    H = hw.control.chunk_horizon
    rate = hw.control.action_rate_hz
    acts = np.zeros((H, 7)); acts[:, 2] = 0.01; acts[:, 6] = np.linspace(0, 1, H)
    plan = Plan(t_created=0.0, t0_pose=np.zeros(6), actions=acts,
                action_times=np.arange(H) / rate, sigma=np.zeros(3), gate=0.0,
                p_evt=np.zeros(5), cpk=None, latency_s=0.0)
    cap = 4
    ex_capped = ChunkExecutor.__new__(ChunkExecutor)
    ex_capped.hw = hw; ex_capped.max_play_steps = cap
    ex_free = ChunkExecutor.__new__(ChunkExecutor)
    ex_free.hw = hw; ex_free.max_play_steps = None
    late = (H - 1) / rate                       # deep in the tail
    pose_c, grip_c = ex_capped._pose_at(plan, late)
    pose_f, grip_f = ex_free._pose_at(plan, late)
    assert pose_c[2] < pose_f[2]                # capped stopped climbing
    assert abs(pose_c[2] - cap * 0.01) < 1e-6   # exactly at the cap point
    assert grip_c == acts[cap - 1, 6]           # gripper frozen with it
    # before the cap, identical
    early = 2.0 / rate
    pc, _ = ex_capped._pose_at(plan, early)
    pf, _ = ex_free._pose_at(plan, early)
    assert np.allclose(pc, pf)


def test_invalidate_cpk_clears_the_remote_token():
    """Verification 09-01: with a policy server the package lives server-side
    behind _cpk_token; _invalidate_cpk must clear the token too or the veto's
    invalidation is a silent no-op on the remote path."""
    from phantom.deploy.planner import PlannerLoop
    from phantom.inference.policy import Plan

    plan = Plan(t_created=0.0, t0_pose=np.zeros(6), actions=np.zeros((2, 7)),
                action_times=np.zeros(2), sigma=np.zeros(3), gate=0.0,
                p_evt=np.zeros(5), cpk="PKG", latency_s=0.0)
    plan._cpk_token = 7
    PlannerLoop._invalidate_cpk(plan)
    assert plan.cpk is None and plan._cpk_token is None


def test_remote_connect_times_out_on_busy_server():
    """Verification 09-01: a server busy with another client accepts the TCP
    connect but never completes the authkey handshake — the client must give
    up with a clear error instead of hanging PICK.sh forever."""
    import socket
    import time as _time

    import pytest as _pytest
    from phantom.inference.remote import RemotePolicy

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)                      # backlog accepts, nobody answers
    port = srv.getsockname()[1]
    old = RemotePolicy.CONNECT_TIMEOUT_S
    RemotePolicy.CONNECT_TIMEOUT_S = 0.5
    try:
        t0 = _time.perf_counter()
        with _pytest.raises(ConnectionError, match="handshake|did not answer|another"):
            RemotePolicy(("127.0.0.1", port))
        assert _time.perf_counter() - t0 < 3.0
    finally:
        RemotePolicy.CONNECT_TIMEOUT_S = old
        srv.close()


def test_tail_cap_does_not_report_uncommanded_steps():
    """Verification 09-01 must-fix: the executed-step index must freeze at
    the cap with the pose — steps the arm never got must not reach
    record_action/_grip_hist (they armed the veto's phantom-grasp latch)."""
    from phantom.deploy.executor import ChunkExecutor
    from phantom.inference.policy import Plan

    hw = make_small_hw()
    H, rate = hw.control.chunk_horizon, hw.control.action_rate_hz
    cap = 4
    ex = ChunkExecutor.__new__(ChunkExecutor)
    ex.hw = hw
    ex.max_play_steps = cap
    # replicate _run's reporting index computation at a play_time deep in
    # the tail, with and without the cap
    play_time = (H - 1) / rate
    _cap = (min(H, ex.max_play_steps) if ex.max_play_steps else H)
    k = min(int(play_time * rate), _cap - 1)
    assert k == cap - 1
    ex.max_play_steps = None
    _cap = (min(H, ex.max_play_steps) if ex.max_play_steps else H)
    k = min(int(play_time * rate), _cap - 1)
    assert k == H - 1


def test_recovered_clears_wrist_and_joint_speed_stops():
    """Verification 09-01: recovered() must clear the two new STOP kinds or
    the shared teleop/collection loop livelocks in safety_hold after a lift
    near full extension."""
    import contextlib
    import time as _time
    import uuid as _uuid

    from phantom.deploy.safety import SafetyMonitor
    from phantom.recording.ringbuffer import SharedRingBuffer

    hw = make_small_hw()
    uid = _uuid.uuid4().hex[:8]
    rings = {"arm": SharedRingBuffer(f"s_rca_{uid}", 16, {
        "ft": ((6,), "float64"), "protective_stop": ((), "uint8"),
        "q": ((hw.arm.dof,), "float64"),
        "qd": ((hw.arm.dof,), "float64")}, create=True)}
    with contextlib.ExitStack() as stack:
        stack.callback(rings["arm"].close)
        mon = SafetyMonitor(hw, rings)
        t = _time.perf_counter()
        straight = np.zeros(hw.arm.dof); straight[2] = np.radians(10)
        rings["arm"].push(t, ft=np.zeros(6), protective_stop=np.uint8(0),
                          q=straight, qd=np.zeros(hw.arm.dof))
        assert not mon.recovered()          # still past the wrist guard
        bent = np.zeros(hw.arm.dof); bent[2] = np.radians(77)
        rings["arm"].push(t + 0.01, ft=np.zeros(6), protective_stop=np.uint8(0),
                          q=bent, qd=np.full(hw.arm.dof, 1.5))
        assert not mon.recovered()          # joints still fast
        rings["arm"].push(t + 0.02, ft=np.zeros(6), protective_stop=np.uint8(0),
                          q=bent, qd=np.zeros(hw.arm.dof))
        assert mon.recovered()              # bent + calm -> clear


def test_tick_rate_lift_complete_in_safety_monitor():
    """Rig 09-04: 4 real grasps were ended by the 500 Hz wrist guard because
    the per-replan lift detector lost the race. The detector now lives in
    SafetyMonitor.check(): baseline-corrected pad loads + tcp z, sustained
    -> a STOP whose executor reason is 'lift_complete' (never a let-go)."""
    import contextlib
    import time as _time
    import uuid as _uuid

    from phantom.deploy.executor import halt_reason_for, is_letgo_reason
    from phantom.deploy.safety import SafetyAction, SafetyEvent, SafetyMonitor
    from phantom.recording.ringbuffer import SharedRingBuffer

    assert not is_letgo_reason("lift_complete")
    assert halt_reason_for([SafetyEvent(0, "lift_complete", 5, SafetyAction.STOP_EPISODE)]) == "lift_complete"
    assert halt_reason_for([SafetyEvent(0, "lift_complete", 5, SafetyAction.STOP_EPISODE),
                            SafetyEvent(0, "wrench_limit", 90, SafetyAction.STOP_EPISODE)]) == "safety_stop"
    assert halt_reason_for([]) == "safety_stop"

    hw = make_small_hw()
    hw = hw.model_copy(update={"safety": hw.safety.model_copy(update={
        "lift_complete_z_m": 0.30, "lift_complete_fz_n": 2.5,
        "lift_complete_hold_s": 0.05})})
    ws = hw.safety.workspace_m
    mid = np.array([np.mean(ws.x), np.mean(ws.y), np.mean(ws.z), 0, 3.14, 0])
    uid = _uuid.uuid4().hex[:8]
    h, w, c = hw.cameras.scene.color.hwc
    names = [s.name for s in hw.tactile.sensors]
    assert len(names) >= 2
    rings = {
        "arm": SharedRingBuffer(f"s_lca_{uid}", 16, {
            "ft": ((6,), "float64"), "protective_stop": ((), "uint8"),
            "tcp_pose": ((6,), "float64")}, create=True),
        "camera_scene": SharedRingBuffer(f"s_lcc_{uid}", 4, {
            "color": ((h, w, c), "uint8")}, create=True),
    }
    for n in names:
        rings[f"tactile_{n}"] = SharedRingBuffer(f"s_lct_{uid}_{n[:3]}", 16, {
            "wrench": ((6,), "float32")}, create=True)
    with contextlib.ExitStack() as stack:
        for r in rings.values():
            stack.callback(r.close)
        mon = SafetyMonitor(hw, rings)
        t = _time.perf_counter()

        def push(z, loads, tt):
            rings["arm"].push(tt, ft=np.zeros(6), protective_stop=np.uint8(0),
                              tcp_pose=np.array([0, 0, z, 0, 3.14, 0.0]))
            rings["camera_scene"].push(tt, color=np.zeros((h, w, c), dtype=np.uint8))
            for n, fz in zip(names, loads):
                rings[f"tactile_{n}"].push(tt, wrench=np.array([0, 0, fz, 0, 0, 0], np.float32))

        # first tick: untouched pads carry a BIAS -> captured as baseline
        push(0.10, [1.5] * len(names), t)
        assert mon.check(t, mid).action == SafetyAction.OK
        # grasped at the floor: loaded but low -> no
        push(0.10, [9.0] * len(names), t + 0.01)
        assert mon.check(t + 0.01, mid).action == SafetyAction.OK
        # lifted + loaded: arms the hold window
        push(0.35, [9.0] * len(names), t + 0.02)
        assert mon.check(t + 0.02, mid).action == SafetyAction.OK
        # sustained past hold -> SUCCESS stop
        push(0.36, [9.0] * len(names), t + 0.10)
        v = mon.check(t + 0.10, mid)
        assert v.action == SafetyAction.STOP_EPISODE
        assert [e.kind for e in v.events] == ["lift_complete"]
        assert halt_reason_for(v.events) == "lift_complete"
        # one pad only 2.0 N over its 1.5 N baseline (3.5 raw) -> not enough,
        # once the 0.6 s trailing window has forgotten the 9 N sample
        push(0.36, [3.5] + [9.0] * (len(names) - 1), t + 0.90)
        assert mon.check(t + 0.90, mid).action == SafetyAction.OK


def test_lift_complete_trailing_window_survives_dropout():
    """09-04: one carried grasp had NO 0.4 s window with both pads
    continuously above threshold (sensor dropout). The trailing-window max
    must ride through a dropout tick; and 4.0 N must reject a weak
    (touch-lost) hold."""
    import contextlib
    import time as _time
    import uuid as _uuid

    from phantom.deploy.safety import SafetyAction, SafetyMonitor
    from phantom.recording.ringbuffer import SharedRingBuffer

    hw = make_small_hw()
    assert hw.safety.lift_complete_fz_n == 4.0 and hw.safety.lift_complete_z_m == 0.32
    hw = hw.model_copy(update={"safety": hw.safety.model_copy(update={
        "lift_complete_hold_s": 0.05, "lift_complete_window_s": 0.6})})
    ws = hw.safety.workspace_m
    mid = np.array([np.mean(ws.x), np.mean(ws.y), np.mean(ws.z), 0, 3.14, 0])
    uid = _uuid.uuid4().hex[:8]
    h, w, c = hw.cameras.scene.color.hwc
    names = [s.name for s in hw.tactile.sensors]
    rings = {"arm": SharedRingBuffer(f"s_twa_{uid}", 16, {
        "ft": ((6,), "float64"), "protective_stop": ((), "uint8"),
        "tcp_pose": ((6,), "float64")}, create=True),
        "camera_scene": SharedRingBuffer(f"s_twc_{uid}", 4, {
            "color": ((h, w, c), "uint8")}, create=True)}
    for n in names:
        rings[f"tactile_{n}"] = SharedRingBuffer(f"s_twt_{uid}_{n[:3]}", 16, {
            "wrench": ((6,), "float32")}, create=True)
    with contextlib.ExitStack() as stack:
        for r in rings.values():
            stack.callback(r.close)
        mon = SafetyMonitor(hw, rings)
        t0 = _time.perf_counter()

        def push(z, loads, tt):
            rings["arm"].push(tt, ft=np.zeros(6), protective_stop=np.uint8(0),
                              tcp_pose=np.array([0, 0, z, 0, 3.14, 0.0]))
            rings["camera_scene"].push(tt, color=np.zeros((h, w, c), dtype=np.uint8))
            for n, fz in zip(names, loads):
                rings[f"tactile_{n}"].push(tt, wrench=np.array([0, 0, fz, 0, 0, 0], np.float32))
            return mon.check(tt, mid)

        push(0.10, [0.0] * len(names), t0)                    # baseline
        push(0.40, [9.0] * len(names), t0 + 0.01)             # lifted + loaded
        r = push(0.40, [0.0] * len(names), t0 + 0.02)         # DROPOUT tick
        assert r.action == SafetyAction.OK                     # window still 9 N -> keeps counting
        r = push(0.40, [9.0] * len(names), t0 + 0.08)
        assert r.action == SafetyAction.STOP_EPISODE and r.events[0].kind == "lift_complete"
        assert mon.contact_load and all(v >= 9.0 for v in mon.contact_load.values())
        # a 3.5 N hold (touch-lost class) never qualifies at 4.0 N
        mon2 = SafetyMonitor(hw, rings)
        rings["arm"].push(t0 + 1.0, ft=np.zeros(6), protective_stop=np.uint8(0),
                          tcp_pose=np.array([0, 0, 0.4, 0, 3.14, 0.0]))
        for n in names:
            rings[f"tactile_{n}"].push(t0 + 1.0, wrench=np.zeros(6, np.float32))
        mon2.check(t0 + 1.0, mid)
        for k in range(5):
            tt = t0 + 1.1 + 0.02 * k
            rings["arm"].push(tt, ft=np.zeros(6), protective_stop=np.uint8(0),
                              tcp_pose=np.array([0, 0, 0.4, 0, 3.14, 0.0]))
            rings["camera_scene"].push(tt, color=np.zeros((h, w, c), dtype=np.uint8))
            for n in names:
                rings[f"tactile_{n}"].push(tt, wrench=np.array([0, 0, 3.5, 0, 0, 0], np.float32))
            assert mon2.check(tt, mid).action == SafetyAction.OK


def test_aperture_latch_never_lets_a_held_grasp_open():
    """09-04 touch-lost mechanism: 4/5 lost objects were RELEASED by the
    policy mid-carry. Once both pads carry load the commanded closure is
    floored; it can close further, never open, until clear_grip_latch()."""
    from types import SimpleNamespace

    from phantom.deploy.executor import ChunkExecutor

    hw = make_small_hw()
    assert hw.safety.grip_latch_fz_n == 2.5
    ex = ChunkExecutor.__new__(ChunkExecutor)
    ex.hw = hw
    ex.safety = SimpleNamespace(contact_load={})
    ex._grip_latch = None
    assert ex._latched_grip(0.30) == 0.30            # no contact: pass-through
    ex.safety.contact_load = {"left": 6.0, "right": 0.5}
    assert ex._latched_grip(0.55) == 0.55            # one pad: still free
    ex.safety.contact_load = {"left": 6.0, "right": 5.0}
    assert ex._latched_grip(0.60) == 0.60            # latch armed at 0.60
    assert ex._grip_latch == 0.60
    assert ex._latched_grip(0.52) == 0.60            # policy tries to open -> floored
    assert ex._latched_grip(0.70) == 0.70            # closing further allowed ...
    assert ex._latched_grip(0.60) == 0.70            # ... and becomes the new floor (running max)
    ex.clear_grip_latch()
    assert ex._latched_grip(0.40) == 0.40            # intended release
    hw0 = hw.model_copy(update={"safety": hw.safety.model_copy(update={"grip_latch_fz_n": 0.0})})
    ex.hw = hw0; ex._grip_latch = None
    assert ex._latched_grip(0.10) == 0.10            # disabled


def test_tactile_phantom_recovery_opens_and_redescends():
    """09-04: 6 phantom lifts (closed on air, lifted 40 cm). Rule evaluated on
    all 17 closes of the session: closed + lifted > 30 mm above the close
    height + trailing-1 s pad load < 2.5 N on both pads, sustained 0.3 s ->
    open + forbid lift. A loaded (carried) gripper is never touched."""
    import time as _time
    from types import SimpleNamespace

    from phantom.deploy.planner import PlannerLoop, TerminalVeto

    hw = make_small_hw()
    names = [s.name for s in hw.tactile.sensors]

    class _Ring:
        def __init__(self):
            self.fz = 0.0

        def latest(self, k):
            return np.full(k, _time.perf_counter()), {"wrench": np.tile(
                np.array([0, 0, self.fz, 0, 0, 0], np.float32), (k, 1))}

    rings = {f"tactile_{n}": _Ring() for n in names}
    veto = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=3, z_floor=0.05,
                        z_margin=0.015, open_aperture=0.20, phantom_t=0.0)

    class _ExL(_Ex):
        cleared = 0

        def clear_grip_latch(self):
            self.cleared += 1

    def run_case(loads, z_seq):
        ex = _ExL()
        snaps = _Snaps(hw, z=0.25, grips=(0.30,))
        lp = PlannerLoop(hw, _Pol(hw, p_none=[0.05], grip_cmd=[0.8]), snaps, ex, veto=veto,
                         session=SimpleNamespace(rings=rings))
        # the FIRST replan captures the per-episode baseline with the pads
        # untouched (as at the real start pose); loads apply from then on
        for r in rings.values():
            r.fz = 0.0
        lp._pad_loads({}, 1.0)          # warm nothing — baseline lives in `state`
        state = {"closed_idx": None, "retries": 0, "g_min": None, "in_close": False,
                 "grip_seen_t": float("-inf"), "close_permitted": False,
                 "allowed_floor_only": False, "closed_floor_only": False}
        acts = []
        for i, z in enumerate(z_seq):
            if i == 1:
                for r in rings.values():
                    r.fz = loads
            H, A = hw.control.chunk_horizon, hw.control.action_dim
            a = np.zeros((H, A)); a[:, 2] = +0.01; a[:, 6] = 0.8
            plan = SimpleNamespace(actions=a, p_evt=np.array([0.05, 0.95, 0, 0, 0]),
                                   cpk=None, latency_s=0.5)
            rec = lp._apply_veto(plan, np.array([0, 0, z, 0, 3.14, 0.0]), 0.8, state, i)
            acts.append((rec["action"], float(plan.actions[0, 6]), float(np.sum(plan.actions[:, 2]))))
        return acts, ex

    # phantom: closed at 0.08, rising to 0.14 with no pad load -> recovery
    acts, ex = run_case(0.0, [0.08, 0.10, 0.14])
    assert acts[-1][0] == "recovery_tactile", acts
    assert acts[-1][1] == 0.20 and acts[-1][2] <= 0.0     # opened + no lift
    assert ex.cleared == 1
    # carried: same motion with 9 N on both pads -> untouched
    acts, ex = run_case(9.0, [0.08, 0.10, 0.14, 0.30])
    assert all(a[0] != "recovery_tactile" for a in acts), acts
    assert ex.cleared == 0


def test_latched_grip_is_what_gets_recorded():
    """Verification 09-04 must-fix: the latch floors the command the arm
    receives — STREAM_ACTIONS / _grip_hist must carry that SENT value, not
    the policy's proposal (else the next fine-tune learns 'open mid-carry'
    and the veto's close detector sees phantom transitions)."""
    from phantom.deploy.planner import VETO_REWRITE_ACTIONS
    assert "recovery_tactile" in VETO_REWRITE_ACTIONS
    import re
    import phantom.deploy.executor as _ex
    src = open(_ex.__file__).read()
    body = src[src.index("while self._last_action_k < k:"):src.index("verdict = self.safety.check")]
    assert "g_sent" in body and "self._grip_latch" in body
    assert "_grip_hist.append((t0, g_sent))" in body
    assert "record_action(t0, a)" in body and "a[6] = g_sent" in body


def test_lift_complete_fz_zero_disables_tick_detector():
    import contextlib
    import time as _time
    import uuid as _uuid

    from phantom.deploy.safety import SafetyAction, SafetyMonitor
    from phantom.recording.ringbuffer import SharedRingBuffer

    hw = make_small_hw()
    hw = hw.model_copy(update={"safety": hw.safety.model_copy(update={
        "lift_complete_fz_n": 0.0, "lift_complete_hold_s": 0.0})})
    ws = hw.safety.workspace_m
    mid = np.array([np.mean(ws.x), np.mean(ws.y), np.mean(ws.z), 0, 3.14, 0])
    uid = _uuid.uuid4().hex[:8]
    h, w, c = hw.cameras.scene.color.hwc
    names = [s.name for s in hw.tactile.sensors]
    rings = {"arm": SharedRingBuffer(f"s_fza_{uid}", 16, {
        "ft": ((6,), "float64"), "protective_stop": ((), "uint8"),
        "tcp_pose": ((6,), "float64")}, create=True),
        "camera_scene": SharedRingBuffer(f"s_fzc_{uid}", 4, {"color": ((h, w, c), "uint8")}, create=True)}
    for n in names:
        rings[f"tactile_{n}"] = SharedRingBuffer(f"s_fzt_{uid}_{n[:3]}", 16, {"wrench": ((6,), "float32")}, create=True)
    with contextlib.ExitStack() as stack:
        for r in rings.values():
            stack.callback(r.close)
        mon = SafetyMonitor(hw, rings)
        for k in range(3):
            tt = _time.perf_counter() + 0.01 * k
            rings["arm"].push(tt, ft=np.zeros(6), protective_stop=np.uint8(0),
                              tcp_pose=np.array([0, 0, 0.40, 0, 3.14, 0.0]))
            rings["camera_scene"].push(tt, color=np.zeros((h, w, c), dtype=np.uint8))
            for n in names:
                rings[f"tactile_{n}"].push(tt, wrench=np.array([0, 0, 0.3 * k, 0, 0, 0], np.float32))
            assert mon.check(tt, mid).action == SafetyAction.OK


def test_p_none_recovery_never_opens_loaded_fingers():
    """Defensive (verification 09-04): the model-opinion recovery arm must
    yield to the pads — both loaded means the object is held."""
    import time as _time
    from types import SimpleNamespace

    from phantom.deploy.planner import PlannerLoop, TerminalVeto

    hw = make_small_hw()
    names = [s.name for s in hw.tactile.sensors]

    class _Ring:
        fz = 0.0

        def latest(self, k):
            return np.full(k, _time.perf_counter()), {"wrench": np.tile(
                np.array([0, 0, self.fz, 0, 0, 0], np.float32), (k, 1))}

    rings = {f"tactile_{n}": _Ring() for n in names}
    veto = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=3, z_floor=0.05,
                        z_margin=0.015, open_aperture=0.20)
    lp = PlannerLoop(hw, _Pol(hw, p_none=[0.05], grip_cmd=[0.8]), _Snaps(hw), _Ex(), veto=veto,
                     session=SimpleNamespace(rings=rings))
    state = {"closed_idx": 3, "retries": 0, "g_min": 0.2, "in_close": True,
             "grip_seen_t": float("-inf"), "close_permitted": True,
             "allowed_floor_only": False, "closed_floor_only": False}
    lp._pad_loads(state, 1.0)              # baseline with untouched pads
    for r in rings.values():
        r.fz = 9.0                          # now firmly loaded
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    a = np.zeros((H, A)); a[:, 6] = 0.8; a[:, 2] = 0.01
    plan = SimpleNamespace(actions=a, p_evt=np.array([0.95, 0.05, 0, 0, 0]), cpk=None, latency_s=0.5)
    rec = lp._apply_veto(plan, np.array([0, 0, 0.20, 0, 3.14, 0.0]), 0.8, state, 4)
    assert rec["action"] == "recovery_skipped_loaded", rec
    assert float(plan.actions[0, 6]) == 0.8 and state["retries"] == 0
