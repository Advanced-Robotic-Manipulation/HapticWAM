"""FIX-NOW batch F2-F9 (docs/review_20260828/VALIDATION_0830.md §3.1, 2026-08-30).

Every test here first REPRODUCES a defect the validation round executed on this
tree, then pins the fix:

F2  the terminal veto's close detector was a per-replan RATE test, so the rig's
    own aperture ramp (0.31 -> 0.52 over five replans) never armed it;
F3  `closed_at` never expired (a real grasp was reopened four replans later),
    it latched on plan ACCEPTANCE rather than on the executed close, and a veto
    rewrite left the pre-veto `plan.cpk` to condition the next replan;
F4  the "already low enough to close" band was measured from the lowered safety
    floor with a 15 mm margin — 30-90 mm below the demo close distribution;
F5  `PlannerLoop.run` had no wall-clock budget, so `--nfe 1` x 40 replans was a
    ~7 s episode against 16-31 s demos, and the replan cap stopped silently;
F6  every STOP left the gripper commanded closed on the gels;
F7  `hitbox_margin < z_floor_margin` silently restored the 08-28 floor-is-a-STOP
    bug, `zfloor:` lied under `--no-z-floor`, and an ACC-less checkpoint made
    the veto a no-op that logs `close_allowed`;
F8  the homing jitter ran on its own unseeded RNG and the realised pose was
    never tagged;
F9  the trace kept only the veto's arithmetic, dropped the per-seed head
    descents and carried no TCP pose.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.planner import PlannerLoop, TerminalVeto
from phantom.deploy.safety import SafetyAction, SafetyEvent, SafetyVerdict
from phantom.inference.policy import ObsSnapshot, Plan
from phantom_test_utils import make_small_hw


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

class _Ex:
    """Executor stub that also models the action-grid steps playback ENTERED.

    `entered_grip_after` is the only view the planner may use to decide that a
    close was actually EXECUTED (F3): a close living in the tail of a chunk the
    executor never reached must not arm the recovery."""

    def __init__(self, accepts=None, enter=True):
        self.stopped_reason = None
        self.submitted: list[np.ndarray] = []
        self.cpks: list = []
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
        self.cpks.append(plan.cpk)
        if ok and self.enter:
            for g in plan.actions[:, 6]:
                self._t += 0.1
                self._hist.append((self._t, float(g)))
        return ok

    def entered_grip_after(self, t):
        return [(ts, g) for ts, g in self._hist if ts > t]


def _snap(hw, z, grip):
    s = ObsSnapshot(t=time.perf_counter(),
                    rgb=np.zeros((4, 4, 3), np.uint8),
                    wrist_window=np.zeros((hw.wrist_ft.window_len, 6), np.float32),
                    ur_state=np.zeros(2 * hw.arm.dof + 14, np.float32))
    s.ur_state[2 * hw.arm.dof + 2] = z
    s.ur_state[-2] = grip
    return s


class _Snaps:
    """Scripted (tcp z, measured aperture) per replan."""

    def __init__(self, hw, z=0.25, grips=(0.30,)):
        self.hw, self.z, self.grips, self.n = hw, z, list(grips), 0

    def build(self):
        g = self.grips[min(self.n, len(self.grips) - 1)]
        self.n += 1
        return _snap(self.hw, self.z, g)


class _Pol:
    """Chunk with a scripted gripper command and p_evt[none] per replan."""

    def __init__(self, hw, p_none, grip_cmd=(0.8,), dz=+0.01, diag=None,
                 cpk="CPK", sleep_s=0.0):
        self.hw = hw
        self.p_none = list(p_none)
        self.grip_cmd = list(grip_cmd)
        self.dz, self.diag, self.cpk, self.sleep_s = dz, diag, cpk, sleep_s
        self.n = 0

    def replan(self, snap, prev_plan, tcp_pose):
        if self.sleep_s:
            time.sleep(self.sleep_s)
        H, A = self.hw.control.chunk_horizon, self.hw.control.action_dim
        a = np.zeros((H, A))
        a[:, 2] = self.dz
        a[:, 6] = self.grip_cmd[min(self.n, len(self.grip_cmd) - 1)]
        pn = self.p_none[min(self.n, len(self.p_none) - 1)]
        self.n += 1
        p = Plan(t_created=float(self.n), t0_pose=np.asarray(tcp_pose, float).copy(),
                 actions=a, action_times=np.arange(H) / 10.0, sigma=np.zeros(3),
                 gate=0.0, p_evt=np.array([pn, 1 - pn, 0, 0, 0]), cpk=self.cpk,
                 latency_s=0.9)
        p.diag = dict(self.diag or {})
        return p


VETO = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=3, z_floor=0.10,
                    z_margin=0.015, open_aperture=0.25)


def _loop(hw, pol, ex, veto, z=0.25, grips=(0.30,)):
    return PlannerLoop(hw, pol, _Snaps(hw, z=z, grips=grips), ex, veto=veto)


def _actions(trace):
    return [t["terminal_veto"]["action"] for t in trace]


# ---------------------------------------------------------------------------
# F2 — the close detector is the TRAINING running-minimum rule
# ---------------------------------------------------------------------------

# the rig's recorded terminal phase: the aperture ramps while still descending,
# 0.31 -> 0.52 over five replans (per-replan rise 0.04)
RAMP = (0.31, 0.35, 0.39, 0.44, 0.48)
RAMP_CMD = tuple(round(g + 0.04, 2) for g in RAMP)


def test_the_close_mask_fires_on_the_recorded_aperture_ramp():
    """Reproduction (validation_0830/deploy-recipe.md §2.1): with the per-replan
    rate test every replan of the recorded ramp logged `none`, so `closed_at`
    was never armed and the recovery could never run. Against the running
    minimum the ramp is one close, and the mask fires while p_contact is 0.01."""
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.99], grip_cmd=RAMP_CMD), ex, VETO, grips=RAMP)
    loop.run(max_replans=len(RAMP))
    acts = _actions(loop.trace)
    assert "close_masked" in acts, acts
    # the early, genuinely small openings are still not a close
    assert acts[0] == "none" and acts[1] == "none", acts
    # ... and once it fires the commanded aperture is held, not closed
    k = acts.index("close_masked")
    assert np.allclose(ex.submitted[k][:, 6], RAMP[k])


def test_the_running_minimum_is_not_reset_by_a_partial_reopen():
    """`g_min` is a running minimum over the episode (train/common.close_index),
    so a brief re-open mid-ramp cannot rearm the detector at a higher floor."""
    hw = make_small_hw()
    ex = _Ex()
    grips = (0.31, 0.40, 0.36, 0.44, 0.48)
    cmds = tuple(round(g + 0.04, 2) for g in grips)
    loop = _loop(hw, _Pol(hw, [0.99], grip_cmd=cmds), ex, VETO, grips=grips)
    loop.run(max_replans=len(grips))
    assert "close_masked" in _actions(loop.trace)


# ---------------------------------------------------------------------------
# F3 — the latch: executed close, one-replan window, cpk invalidation
# ---------------------------------------------------------------------------

def test_the_recovery_still_fires_on_the_replan_after_an_executed_close():
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.2, 0.95]), ex, VETO)
    loop.run(max_replans=2)
    assert _actions(loop.trace) == ["close_allowed", "recovery_open"]


def test_a_real_grasp_is_not_reopened_four_replans_later():
    """Reproduction (VALIDATION_0830 P0 #3): `p_none = [.2,.05,.05,.05,.95]`
    produced `recovery_open` FOUR replans after the close — the gripper forced
    to the open aperture and every z delta zeroed on a grasp that held."""
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.2, 0.05, 0.05, 0.05, 0.95]), ex, VETO)
    loop.run(max_replans=5)
    assert "recovery_open" not in _actions(loop.trace), _actions(loop.trace)
    # the grasp is left alone: nothing is ever commanded to the OPEN aperture
    # and no lift is zeroed (the close mask may still hold the aperture it
    # measures, which on a held object IS the grasp).
    assert all(not np.allclose(a[:, 6], VETO.open_aperture) for a in ex.submitted)
    assert np.allclose(ex.submitted[-1][:, 2], 0.01)


def test_the_latch_is_cleared_unconditionally_after_its_window():
    hw = make_small_hw()
    ex = _Ex()
    # the window is the replan the executed close is OBSERVED on plus one
    # (planner.VETO_RECOVERY_REPLANS); a high-p_none frame after that is a
    # transport/regrasp frame on a grasp that held, not a phantom close.
    loop = _loop(hw, _Pol(hw, [0.2, 0.05, 0.05, 0.95]), ex, VETO)
    loop.run(max_replans=4)
    assert _actions(loop.trace)[3] != "recovery_open"


def test_a_close_that_was_never_EXECUTED_does_not_arm_the_recovery():
    """Codex's state bug: a close living only in the unexecuted chunk tail
    armed the latch through plan acceptance alone."""
    hw = make_small_hw()
    ex = _Ex(enter=False)               # accepted, but no step is ever entered
    loop = _loop(hw, _Pol(hw, [0.2, 0.95]), ex, VETO)
    loop.run(max_replans=2)
    assert "recovery_open" not in _actions(loop.trace)


def test_a_veto_rewrite_drops_the_stale_contact_package():
    """`plan.cpk` is the imagined contact of the chunk the policy PROPOSED. If
    the veto rewrites the chunk, the next replan must not be conditioned on it
    (planner.py:471 -> policy.py:244)."""
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.99]), ex, VETO)         # -> close_masked
    loop.run(max_replans=1)
    assert loop.trace[0]["terminal_veto"]["action"] == "close_masked"
    assert ex.cpks[0] is None
    ex2 = _Ex()
    loop2 = _loop(hw, _Pol(hw, [0.2, 0.95]), ex2, VETO)  # -> recovery_open
    loop2.run(max_replans=2)
    assert ex2.cpks[0] == "CPK" and ex2.cpks[1] is None


# ---------------------------------------------------------------------------
# F4 — the at_floor band is the demo close band, not the safety floor
# ---------------------------------------------------------------------------

def test_the_at_floor_band_reaches_the_demo_close_height():
    """waffles: tcp_z_min 52 mm, safety floor 42 mm, demo close p95 + 15 mm =
    Z_MAX 103 mm. A close commanded at 65 mm (inside the demo band) with an
    under-confident gate used to be masked."""
    hw = make_small_hw()
    v = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=3, z_floor=0.042,
                     z_ref=0.052, z_margin=0.051, open_aperture=0.25)
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.99]), ex, v, z=0.065)
    loop.run(max_replans=1)
    rec = loop.trace[0]["terminal_veto"]
    assert rec["action"] == "close_allowed" and rec["at_floor"] is True
    # and it is still a band, not "anything goes": 150 mm is not a demo close
    ex2 = _Ex()
    loop2 = _loop(hw, _Pol(hw, [0.99]), ex2, v, z=0.150)
    loop2.run(max_replans=1)
    assert loop2.trace[0]["terminal_veto"]["action"] == "close_masked"


def test_build_veto_measures_the_band_from_tcp_z_min_up_to_the_task_z_max():
    from phantom.eval.grasp_label import Z_MAX_MM
    from phantom.scripts.run_deploy import build_parser, build_veto
    stats = SimpleNamespace(tcp_z_min=0.052, gripper_mean=0.25)
    args = build_parser().parse_args(["--system", "teacher", "--task",
                                      "waffles", "--terminal-veto"])
    v = build_veto(args, stats, z_floor=0.042)
    assert v.z_ref == pytest.approx(0.052)
    assert v.z_ref + v.z_margin == pytest.approx(Z_MAX_MM["waffles"] / 1000.0)


def test_the_veto_z_margin_is_exposed_as_a_flag():
    from phantom.scripts.run_deploy import build_parser, build_veto
    stats = SimpleNamespace(tcp_z_min=0.052, gripper_mean=0.25)
    args = build_parser().parse_args(["--system", "teacher", "--task", "waffles",
                                      "--terminal-veto", "--veto-z-margin", "0.02"])
    assert build_veto(args, stats, z_floor=0.042).z_margin == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# F5 — a wall-clock budget beside the replan cap, and a named cap reason
# ---------------------------------------------------------------------------

def test_the_episode_time_budget_ends_the_loop():
    """`--max-replans 40` at `--nfe 1` (172 ms replans) is a 7 s episode; the
    budget must end the episode on wall clock, not on a count."""
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.5], sleep_s=0.02), ex, None)
    t0 = time.perf_counter()
    loop.run(max_replans=1000, max_episode_s=0.25)
    dt = time.perf_counter() - t0
    assert 0.2 < dt < 1.5, dt
    assert len(loop.trace) < 1000
    assert loop.stop_reason == "episode_time_cap"


def test_the_replan_cap_names_its_stop_reason():
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.5]), ex, None)
    loop.run(max_replans=2)
    assert len(loop.trace) == 2 and loop.stop_reason == "replan_cap"


def test_run_deploy_passes_a_35_s_episode_budget_by_default(monkeypatch, tmp_path):
    kw = _run_main(monkeypatch, tmp_path)
    assert kw["max_episode_s"] == pytest.approx(35.0)
    kw2 = _run_main(monkeypatch, tmp_path, argv=("--max-episode-s", "12"))
    assert kw2["max_episode_s"] == pytest.approx(12.0)


def test_the_episode_budget_reaches_the_planner_loop(monkeypatch):
    """runtime.run_episode must forward the budget (and surface the planner's
    own cap reason when the executor has none)."""
    import inspect

    from phantom.deploy import runtime as RT
    assert "max_episode_s" in inspect.signature(RT.DeploymentRuntime.run_episode).parameters
    src = inspect.getsource(RT.DeploymentRuntime.run_episode)
    assert "max_episode_s=max_episode_s" in src
    assert "stop_reason" in src


# ---------------------------------------------------------------------------
# F6 — a stop that means "let go" releases the gripper
# ---------------------------------------------------------------------------

class _CountingGripper:
    def __init__(self):
        self.moves: list[float] = []

    def move(self, position, speed, force):
        self.moves.append(float(position))

    def get_state(self):
        return SimpleNamespace(t_host=time.perf_counter(), position=0.42, obj=3.0)


def _ex_with_gripper(hw, g, **kw):
    return ChunkExecutor(hw, SimpleNamespace(servo_stop=lambda: None,
                                             stop=lambda *a: None),
                         g, safety=None, open_aperture=0.25, **kw)


@pytest.mark.parametrize("reason,events,released", [
    ("safety_stop", [("tactile_fz", SafetyAction.STOP_EPISODE)], True),
    ("safety_stop", [("tactile_depth", SafetyAction.STOP_EPISODE)], True),
    ("safety_stop", [("wrench_limit", SafetyAction.STOP_EPISODE)], True),
    ("safety_stop", [("hitbox_exit", SafetyAction.STOP_EPISODE)], False),
    ("veto_retry_cap", [], True),
    ("camera_scene_stale", [], False),
    ("replan_cap", [], False),
])
def test_only_the_letgo_stops_command_the_gripper_open(reason, events, released):
    hw = make_small_hw()
    g = _CountingGripper()
    ex = _ex_with_gripper(hw, g)
    ex._halt(reason, events=[SafetyEvent(0.0, k, 1.0, a) for k, a in events])
    assert (g.moves == [0.25]) is released, (reason, g.moves)


def test_a_fingertip_stop_leaves_the_gripper_commanded_OPEN_end_to_end():
    """Reproduction (validation_0830/safety-final.md §4): with a chunk
    commanding a 0.90 close and a tactile STOP at tick ~40, the gripper's LAST
    commanded target was 0.90 — squeezing the gels through the label prompt."""
    hw = make_small_hw()
    g = _CountingGripper()

    class _Safety:
        def __init__(self):
            self.n = 0

        def check(self, t, target):
            self.n += 1
            if self.n < 20:
                return SafetyVerdict(action=SafetyAction.OK)
            return SafetyVerdict(action=SafetyAction.STOP_EPISODE, events=[
                SafetyEvent(t, "tactile_fz", 9.9, SafetyAction.STOP_EPISODE)])

        def clamp_target(self, t):
            return t

    stopped = []
    arm = SimpleNamespace(
        servo_l=lambda *a, **k: None,
        servo_stop=lambda: None,
        stop=lambda *a: stopped.append(1),
        get_state=lambda: SimpleNamespace(tcp_pose=np.zeros(6)))
    ex = ChunkExecutor(hw, arm, g, _Safety(), open_aperture=0.25)
    H, A = hw.control.chunk_horizon, hw.control.action_dim
    a = np.zeros((H, A))
    a[:, 6] = 0.90
    now = time.perf_counter()
    plan = Plan(t_created=now, t0_pose=np.zeros(6), actions=a,
                action_times=now + np.arange(H) / hw.control.action_rate_hz,
                sigma=np.zeros(3), gate=0.0, p_evt=np.zeros(5), cpk=None,
                latency_s=0.0)
    ex.start()
    try:
        assert ex.submit(plan)
        for _ in range(200):
            if ex.stopped_reason is not None:
                break
            time.sleep(0.01)
    finally:
        ex.stop()
    assert ex.stopped_reason == "safety_stop" and stopped
    assert g.moves and g.moves[-1] == 0.25, g.moves


def test_the_post_stop_prompt_names_the_gripper_release_script():
    from phantom.scripts import run_deploy as RD
    assert "GRIPPER_OPEN.sh" in RD.VERDICT_PROMPT


# ---------------------------------------------------------------------------
# F7 — envelope invariant, honest zfloor tag, ACC head assert
# ---------------------------------------------------------------------------

def test_a_hitbox_floor_above_the_z_floor_is_refused():
    """`--hitbox-margin 5 --z-floor-margin 10` silently restored the 08-28 bug:
    the clamp pins the target ON a floor the hitbox then calls an exit."""
    from phantom.deploy.safety import apply_hitbox, apply_z_floor
    from phantom.scripts.run_deploy import envelope_conflict
    hw = make_small_hw()
    lo, hi = np.array([-.4, -.4, .052]), np.array([.4, .4, .44])
    ok = apply_z_floor(apply_hitbox(hw, lo, hi, 0.030), 0.042)
    assert envelope_conflict(ok) is None
    bad = apply_z_floor(apply_hitbox(hw, lo, hi, 0.005), 0.042)
    msg = envelope_conflict(bad)
    assert msg and "hitbox" in msg


def test_the_zfloor_tag_is_honest_when_the_floor_is_off(monkeypatch, tmp_path):
    kw = _run_main(monkeypatch, tmp_path, argv=("--no-z-floor",))
    assert "zfloor:none" in kw["tags"]
    kw2 = _run_main(monkeypatch, tmp_path)
    assert any(t.startswith("zfloor:") and t != "zfloor:none" for t in kw2["tags"])


def test_run_deploy_refuses_to_start_on_an_inverted_envelope(monkeypatch, tmp_path,
                                                             caplog):
    """Through main(): the operator tightens the advertised --hitbox-margin
    below --z-floor-margin and every deep descent becomes a hitbox_exit."""
    with caplog.at_level(logging.ERROR, logger="run_deploy"):
        _run_main(monkeypatch, tmp_path, hitbox=True, expect_rc=2,
                  argv=("--hitbox-margin", "0.005", "--z-floor-margin", "0.010"))
    assert any("UNSAFE ENVELOPE" in r.getMessage() for r in caplog.records)
    # ... and the ordinary margins still start
    _run_main(monkeypatch, tmp_path, hitbox=True,
              argv=("--hitbox-margin", "0.030", "--z-floor-margin", "0.010"))


def test_build_policy_refuses_an_acc_less_checkpoint_under_the_veto(monkeypatch,
                                                                   tmp_path):
    """Through the real checkpoint-loading entry point, and BEFORE the model is
    built: `build_model` must never be reached."""
    from phantom.scripts import run_deploy as RD
    ckpt = tmp_path / "no_acc.pt"
    ckpt.write_bytes(b"stub")
    monkeypatch.setattr(RD.torch, "load", lambda *a, **k: {
        "phantom_modules": {"net.blocks.0.weight": 1}, "ema": None})

    def _boom(*a, **k):
        raise AssertionError("build_model must not be reached")

    monkeypatch.setattr(RD, "build_model", _boom)
    args = RD.build_parser().parse_args(
        ["--system", "teacher", "--task", "waffles", "--terminal-veto",
         "--ckpt", str(ckpt), "--device", "cpu"])
    with pytest.raises(RuntimeError, match="ACC"):
        RD.build_policy(args, make_small_hw(), None)


def test_a_checkpoint_without_an_acc_head_is_refused_under_the_veto():
    """Fails open otherwise: p_evt = zeros(5) -> p_contact = 1.0, so the close
    mask and the K-seed rejection become no-ops that log `close_allowed`."""
    from phantom.scripts.run_deploy import assert_acc_head, has_acc_head
    good = {"phantom_modules": {"net.phantom_acc.mlp.weight": 1}}
    bad = {"phantom_modules": {"net.blocks.0.weight": 1}, "ema": None}
    assert has_acc_head(good) and not has_acc_head(bad)
    assert_acc_head(good, "x.pt")
    with pytest.raises(RuntimeError, match="ACC"):
        assert_acc_head(bad, "x.pt")


# ---------------------------------------------------------------------------
# F8 — the homing jitter is seeded, and the realised pose is tagged
# ---------------------------------------------------------------------------

def test_the_homing_jitter_is_seeded_from_the_episode_seed(monkeypatch, tmp_path):
    """`--seed` advertises "reproducible noise draws for paired trials" while
    the start-pose jitter (waffles: +-27 mm per axis) ran on its own unseeded
    RNG — the same order as the effect the A/B measures."""
    a = _run_main(monkeypatch, tmp_path, argv=("--seed", "4242"), episodes=2)
    b = _run_main(monkeypatch, tmp_path, argv=("--seed", "4242"), episodes=2)
    c = _run_main(monkeypatch, tmp_path, argv=("--seed", "77"), episodes=2)
    assert a["draws"] == b["draws"], (a["draws"], b["draws"])
    assert a["draws"][0] != a["draws"][1], "each episode must jitter differently"
    assert a["draws"] != c["draws"]


def test_the_realised_start_pose_is_tagged(monkeypatch, tmp_path):
    """`start:` is the pose the arm actually stands at when the episode starts
    (read from the arm), `start_req:` the sampled homing target. 09-11: the
    tag came from move_to_start's return value and the auto-home path
    re-sampled without updating it, so 3 of 5 seeds carried a start 60-96 mm
    from the truth."""
    monkeypatch.setattr(_StubArm, "get_state",
                        lambda self: SimpleNamespace(tcp_pose=np.array([0.15, 0.26, 0.37, 0, 0, 0])))
    kw = _run_main(monkeypatch, tmp_path)
    req = [t for t in kw["tags"] if t.startswith("start_req:")]
    real = [t for t in kw["tags"] if t.startswith("start:")]
    assert len(req) == 1 and "111" in req[0] and "0.33" in req[0], kw["tags"]
    assert len(real) == 1 and real[0].startswith("start:150,260,370mm/g"), kw["tags"]


def test_an_untouched_replan_has_no_pre_veto_chunk():
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.2]), ex, VETO)
    loop.run(max_replans=1)
    assert loop.trace[0].get("actions_pre_veto") is None


def test_the_trace_keeps_the_per_seed_head_descents_and_the_tcp_pose():
    hw = make_small_hw()
    ex = _Ex()
    diag = {"nfe": 1, "k_seeds": 4, "head_dz_mm": [-93.1, -45.0, -26.2, -60.0]}
    loop = _loop(hw, _Pol(hw, [0.2], diag=diag), ex, None, z=0.123)
    loop.run(max_replans=1)
    row = loop.trace[0]
    assert row["diag"]["head_dz_mm"] == diag["head_dz_mm"]
    assert row["tcp_pose"][2] == pytest.approx(0.123)


def test_the_retry_cap_row_carries_the_pose_too():
    hw = make_small_hw()
    ex = _Ex()
    v = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=0, z_floor=0.10,
                     open_aperture=0.25)
    loop = _loop(hw, _Pol(hw, [0.2, 0.95]), ex, v, z=0.2)
    loop.run(max_replans=4)
    assert loop.trace[-1]["terminal_veto"]["action"] == "retry_cap"
    assert loop.trace[-1]["tcp_pose"][2] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# run_deploy.main() with every device stubbed (F5/F7/F8 CLI level)
# ---------------------------------------------------------------------------

class _StubArm:
    def is_ready_for_control(self):
        return True, ""

    def reconnect_control(self):
        pass

    def program_running(self):
        return True

    def get_state(self):
        return SimpleNamespace(tcp_pose=np.zeros(6))


def _run_main(monkeypatch, tmp_path, *, argv=(), episodes=1, hitbox=False,
              expect_rc=0, all_calls=False):
    """Drive run_deploy.main() over a stubbed real-arm session; return the
    kwargs of the LAST run_episode call plus the homing RNG draws (or, with
    all_calls, the list of every run_episode call's kwargs)."""
    from phantom.deploy import start_pose as sp
    from phantom.deploy.runtime import EpisodeResult
    from phantom.scripts import run_deploy as RD

    hw = make_small_hw(mode={"drivers": "real"})
    seen: dict = {"draws": []}
    calls: list[dict] = []

    class _StubRuntime:
        def __init__(self, hw, policy, mode, out_root, **kw):
            self.rig = SimpleNamespace(
                arm=_StubArm(),
                gripper=SimpleNamespace(get_state=lambda: SimpleNamespace(
                    position=0.0, obj=3.0)))
            self.recorder = SimpleNamespace(relabel=lambda *a, **k: None)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run_episode(self, **kw):
            seen.update(kw)
            calls.append(dict(kw))
            return EpisodeResult(episode_path=None, stopped_reason=None,
                                 n_replans=1, safety_events=0, trace_path=None)

    monkeypatch.setattr(RD, "load_hardware", lambda *a, **k: hw)
    monkeypatch.setattr(RD, "load_paths", lambda *a, **k: SimpleNamespace(
        validate=lambda **k: None, episodes_root=lambda: tmp_path))
    monkeypatch.setattr(RD, "build_policy", lambda *a, **k: SimpleNamespace(
        nfe=1, guidance=1.0, rf=SimpleNamespace()))
    monkeypatch.setattr(RD, "DeploymentRuntime", _StubRuntime)
    monkeypatch.setattr(RD.time, "sleep", lambda *_: None)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    stats = SimpleNamespace(tcp_z_min=0.052, gripper_mean=0.25, task="waffles",
                            tcp_min=(np.array([-.4, -.4, .052]) if hitbox else None),
                            tcp_max=(np.array([.4, .4, .44]) if hitbox else None))
    monkeypatch.setattr(sp, "load_start_stats", lambda: {"waffles": stats})
    monkeypatch.setattr(sp, "start_sigma_report",
                        lambda *a, **k: (np.zeros(7), "in distribution"))

    def _move(arm, gripper, hw_, st, rng=None, **k):
        seen["draws"].append(float(np.asarray(rng.standard_normal(1))[0]))
        return np.array([0.111, 0.222, 0.333, 0.0, 0.0, 0.0]), 0.33

    monkeypatch.setattr(sp, "move_to_start", _move)
    rc = RD.main(["--system", "teacher", "--task", "waffles", "--tiny",
                  "--episodes", str(episodes), *argv])
    assert rc == expect_rc, rc
    seen["rc"] = rc
    return calls if all_calls else seen
