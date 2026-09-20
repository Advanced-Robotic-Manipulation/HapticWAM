"""Descend-then-release supervisor (rig 2026-09-12).

The measured defect: 19 of 31 latched carries never released, because once the latch
held over the crate the policies kept commanding descent at 32-68 mm/s straight through
the demonstrated release band until the 45 N wrench guard ended the episode, and 5 of the
12 that did release kept descending afterwards (3 of those ended on the guard too).

These tests pin the supervisor that fixes it: a crate-region floor on the commanded z, a
forced latch release over the crate after a dwell, a vertical retract, and a re-latch
block so the latch cannot re-engage on the object that was just released. Everything is
opt-in: with no config the executor path is unchanged.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.deploy.descend_then_release import (
    DescendThenReleaseConfig,
    DescendThenReleaseSupervisor,
    make_placement_descent,
)
from phantom.deploy.executor import ChunkExecutor
from tests.test_servo_result import _StubArm

CONFIG_PATH = "configs/placement_descent_rig_0913.json"


def cfg(**over):
    base = dict(release_z_m=0.081, release_y_min_m=0.0, task="waffles")
    base.update(over)
    return DescendThenReleaseConfig(**base)


def sup(**over):
    return DescendThenReleaseSupervisor(cfg(**over))


def tick(s, t, *, z, y=0.06, latched=True, cmd_z=None, grip=0.64, open_aperture=0.25,
         pose=True):
    """One servo tick at measured (y, z) with the policy commanding `cmd_z`."""
    tcp = np.array([-0.35, y, z, 0.0, 0.0, 0.0]) if pose else None
    target = np.array([-0.35, y, z if cmd_z is None else cmd_z, 0.0, 0.0, 0.0])
    return s.step(t, measured_tcp=tcp, latched=latched, commanded_target=target,
                  policy_grip=grip, open_aperture=open_aperture)


def carry(s, t0=0.0, *, y=0.06, n=20, dt=0.008):
    """The carry apex the native latch's lift gate requires (z 0.35 m)."""
    t = t0
    for _ in range(n):
        tick(s, t, z=0.35, y=y)
        t += dt
    return t


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_the_config_refuses_a_floor_above_its_own_release_gate():
    with pytest.raises(ValueError):
        cfg(release_z_m=0.20)                      # floor above the 0.16 gate
    with pytest.raises(ValueError):
        cfg(release_z_m=-0.05)
    with pytest.raises(ValueError):
        cfg(variant="descend_then_release_v2")
    with pytest.raises(ValueError):
        cfg(release_y_min_m=-0.30)                 # outside the crate region
    with pytest.raises(ValueError):
        cfg(retract_speed_m_s=0.5)                 # not a placement retract
    with pytest.raises(ValueError):
        cfg(dwell_s=0.0)


def test_an_unmeasured_task_leaves_the_supervisor_off():
    spec = json.loads(open(CONFIG_PATH).read())
    waffles = DescendThenReleaseConfig.from_spec(spec, "waffles")
    carton = DescendThenReleaseConfig.from_spec(spec, "Carton")
    assert (waffles.release_z_m, waffles.release_y_min) == (0.081, 0.0)
    assert (carton.release_z_m, carton.release_y_min) == (0.118, 0.0)
    assert waffles.dwell_s == 0.35 and waffles.retract_m == 0.08
    assert waffles.y_crate_edge_m == -0.10
    # egg / whiteboard have no measured release band: OFF, never a guess
    assert DescendThenReleaseConfig.from_spec(spec, "egg") is None
    assert DescendThenReleaseConfig.from_spec(spec, "whiteboard") is None


def test_a_task_may_not_redefine_a_shared_field():
    with pytest.raises(ValueError):
        DescendThenReleaseConfig.from_spec(
            {"dwell_s": 0.35, "tasks": {"waffles": {"release_z_m": 0.081,
                                                    "dwell_s": 1.0}}}, "waffles")
    with pytest.raises(ValueError):
        DescendThenReleaseConfig.from_spec({"release_z_m": 0.081}, "waffles")


def test_the_factory_accepts_a_dict_and_refuses_anything_else():
    assert make_placement_descent(None) is None
    s = make_placement_descent({"release_z_m": 0.081})
    assert isinstance(s, DescendThenReleaseSupervisor)
    assert make_placement_descent(cfg()).config.task == "waffles"
    with pytest.raises(TypeError):
        make_placement_descent(0.081)


# ---------------------------------------------------------------------------
# the crate-region floor
# ---------------------------------------------------------------------------

def test_the_floor_raises_a_command_below_the_demo_band_over_the_crate():
    s = sup()
    d = tick(s, 0.0, z=0.12, y=0.06, cmd_z=0.05, latched=False)
    assert d.overrode and d.target[2] == pytest.approx(0.081)
    assert d.grip is None and not d.clear_latch and not d.block_latch


def test_the_floor_leaves_a_command_inside_the_band_untouched():
    s = sup()
    d = tick(s, 0.0, z=0.12, y=0.06, cmd_z=0.105, latched=False)
    assert not d.overrode and d.target[2] == pytest.approx(0.105)


def test_the_floor_does_not_apply_at_the_grasp_site():
    """The object sits at y ~ -0.27: the grasp must keep its full descent."""
    s = sup()
    d = tick(s, 0.0, z=0.07, y=-0.27, cmd_z=0.066, latched=False)
    assert not d.overrode and d.target[2] == pytest.approx(0.066)


def test_the_floor_survives_the_natural_release_because_it_needs_no_latch():
    """5 of the 12 carries that DID release kept descending afterwards; the
    floor is gated on the crate region only, so it still holds them."""
    s = sup()
    d = tick(s, 0.0, z=0.10, y=0.07, cmd_z=0.02, latched=False)
    assert d.overrode and d.target[2] == pytest.approx(0.081)


def test_x_y_and_orientation_are_never_touched_by_the_floor():
    s = sup()
    d = tick(s, 0.0, z=0.10, y=0.07, cmd_z=0.0, latched=False)
    assert d.target[0] == pytest.approx(-0.35) and d.target[1] == pytest.approx(0.07)
    assert np.allclose(d.target[3:], 0.0)


# ---------------------------------------------------------------------------
# the forced release: what must NOT fire
# ---------------------------------------------------------------------------

def test_nothing_fires_during_the_reach_phase():
    s = sup()
    t = 0.0
    for _ in range(400):                            # 3.2 s unlatched at the object
        d = tick(s, t, z=0.09, y=-0.27, latched=False)
        t += 0.008
        assert d.grip is None and not d.clear_latch and d.state == "idle"


def test_nothing_fires_at_the_carry_apex():
    """y is over the crate but z 0.35 m is far above the 0.16 m release gate."""
    s = sup()
    t = 0.0
    for _ in range(400):
        d = tick(s, t, z=0.35, y=0.06)
        t += 0.008
        assert d.grip is None and not d.clear_latch and d.state == "idle"


def test_nothing_fires_before_a_carry_happened():
    """Latched low and already over the crate, never lifted: the lift gate
    (release_gate_z_m + lift_m) blocks the release exactly as the native latch's
    own gate does."""
    s = sup()
    t = 0.0
    for _ in range(400):
        d = tick(s, t, z=0.10, y=0.06)
        t += 0.008
        assert not d.clear_latch and d.state == "idle"
    assert d.overrode is False or d.target[2] >= 0.081


def test_nothing_fires_short_of_the_crate_release_y():
    """The crate centre is y ~ +0.05 and the demos release at y 0.048-0.097. A
    carry stalled at y -0.09 is inside the region floor but on the crate's near
    rim: forcing a release there would drop the object outside the crate (rig
    episode teacher_waffles_1789228368_008, which bottomed out at exactly that y)."""
    s = sup()
    t = carry(s, y=-0.09)
    for _ in range(400):
        d = tick(s, t, z=0.083, y=-0.0895)
        t += 0.008
    assert not d.clear_latch and s.releases == 0
    assert d.target[2] == pytest.approx(0.083)       # measured z is above the floor


def test_the_dwell_must_be_continuous():
    s = sup()
    t = carry(s)
    for _ in range(30):                              # 0.24 s in the zone
        tick(s, t, z=0.12)
        t += 0.008
    assert s.state == "watching"
    d = tick(s, t, z=0.20)                           # lifted back out of the zone
    t += 0.008
    assert d.state == "idle" and s.dwell_since is None
    for _ in range(30):
        d = tick(s, t, z=0.12)
        t += 0.008
    assert not d.clear_latch                         # the 0.35 s dwell restarted


def test_a_missing_measured_pose_withholds_every_override():
    s = sup()
    t = carry(s)
    for _ in range(200):
        d = tick(s, t, z=0.12, pose=False)
        t += 0.008
        assert not d.overrode and d.grip is None and not d.clear_latch


# ---------------------------------------------------------------------------
# the forced release: the full sequence
# ---------------------------------------------------------------------------

def _release(s, *, z=0.12, y=0.06, grip=0.64, open_aperture=0.25, dt=0.008):
    """Drive the supervisor to the forced release; return (t, decision)."""
    t = carry(s, y=y)
    for _ in range(400):
        d = tick(s, t, z=z, y=y, grip=grip, open_aperture=open_aperture)
        t += dt
        if d.clear_latch:
            return t, d
    raise AssertionError("no forced release")


def test_the_release_fires_after_the_dwell_at_the_measured_height():
    s = sup()
    t0 = carry(s)
    t, d = _release(s)
    assert d.event == "forced_release" and d.state == "opening"
    assert t - t0 == pytest.approx(0.35, abs=0.02)   # the configured dwell
    assert s.release_z == pytest.approx(0.12) and s.release_y == pytest.approx(0.06)
    assert d.target[2] == pytest.approx(0.12)        # held, never driven lower
    assert d.clear_latch and d.block_latch and d.overrode


def test_the_release_commands_the_more_open_of_the_two_apertures():
    s = sup()
    _, d = _release(s, grip=0.64, open_aperture=0.25)
    assert d.grip == pytest.approx(0.25)             # episode open aperture wins
    s2 = sup()
    _, d2 = _release(s2, grip=0.10, open_aperture=0.25)
    assert d2.grip == pytest.approx(0.10)            # the policy asked for more
    s3 = sup()
    _, d3 = _release(s3, grip=float("nan"), open_aperture=0.25)
    assert d3.grip == pytest.approx(0.25)            # a NaN request is ignored


def test_the_release_is_held_open_then_retracts_and_hands_back():
    s = sup()
    t, d = _release(s)
    t_open, z = t, 0.12
    # open hold: the pose is pinned at the release height while the fingers open
    for _ in range(400):
        d = tick(s, t, z=z)
        t += 0.008
        if d.state != "opening":
            break
        assert d.target[2] == pytest.approx(0.12)
        assert d.grip == pytest.approx(0.25) and d.block_latch
    assert t - t_open == pytest.approx(0.30, abs=0.02)
    # retract: the commanded z rises at retract_speed_m_s, x/y/rot held
    assert d.state == "retracting"
    zs = []
    for _ in range(400):
        d = tick(s, t, z=z)
        t += 0.008
        if d.state != "retracting":
            break
        zs.append(float(d.target[2]))
        z = min(0.20, z + 0.0008)                     # the arm follows the command
        assert d.grip == pytest.approx(0.25) and d.block_latch
    # 0.12 release + 0.08 retract, reached within z_tolerance_m of the target
    assert zs == sorted(zs) and max(zs) == pytest.approx(0.20, abs=0.006)
    assert d.state == "released" and d.event == "retract_complete"
    # handed back: the policy owns motion and the gripper again
    d = tick(s, t, z=0.20, grip=0.7)
    assert d.grip is None and not d.overrode


def test_the_retract_gives_up_after_max_retract_s():
    s = sup(max_retract_s=0.5)
    t, _ = _release(s)
    for _ in range(400):
        d = tick(s, t, z=0.12)                        # the arm never rises
        t += 0.008
        if d.state == "released":
            break
    assert d.event == "retract_timeout"


def test_the_release_height_never_goes_below_the_demo_floor():
    """A carry already below the p10 band releases AT the floor, not lower."""
    s = sup()
    t, d = _release(s, z=0.070)
    assert s.release_z == pytest.approx(0.070)
    assert d.target[2] == pytest.approx(0.081)        # commanded back up to the floor


# ---------------------------------------------------------------------------
# no re-latch on the released object
# ---------------------------------------------------------------------------

def test_the_latch_stays_blocked_while_the_tool_is_still_over_the_crate():
    s = sup()
    t, _ = _release(s)
    for _ in range(600):                              # through open hold + retract
        d = tick(s, t, z=min(0.20, 0.12 + (t % 1.0) * 0.1), grip=0.8)
        t += 0.008
        assert d.block_latch, d.state
    assert d.state == "released"


def test_the_supervisor_rearms_only_after_leaving_the_crate():
    s = sup()
    t, _ = _release(s)
    for _ in range(600):
        d = tick(s, t, z=0.20, grip=0.8)
        t += 0.008
    assert d.state == "released" and d.block_latch
    d = tick(s, t, z=0.20, y=-0.27, grip=0.8)         # back over the object
    assert d.state == "idle" and not d.block_latch and d.event == "supervisor_rearmed"
    # a fresh carry is supervised again
    t += 0.008
    t2, d2 = _release(s)
    assert d2.clear_latch and s.releases == 2


# ---------------------------------------------------------------------------
# executor integration
# ---------------------------------------------------------------------------

class _Ring:
    def __init__(self):
        self.z, self.y = 0.30, 0.06

    def latest(self, n):
        return np.array([1.0]), {"tcp_pose": np.array([[-0.38, self.y, self.z,
                                                        0.0, 0.0, 0.0]])}


class _Safety:
    def __init__(self):
        self.contact_load = {"left": 9.0, "right": 9.0}
        self.ring = _Ring()
        self.rings = {"arm": self.ring}


def _ex(monkeypatch, *, descent=None):
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    ex = ChunkExecutor(hw, arm=_StubArm(set()), gripper=None, safety=_Safety(),
                       open_aperture=0.25, placement_descent=descent)
    clock = [100.0]
    monkeypatch.setattr("phantom.deploy.executor.time.perf_counter", lambda: clock[0])
    return ex, clock


def _target(z, y=0.06):
    return np.array([-0.38, y, z, 0.0, 0.0, 0.0])


def test_the_executor_release_clears_the_latch_and_opens_the_fingers(monkeypatch):
    ex, clock = _ex(monkeypatch, descent=cfg())
    ex.safety.ring.z = 0.07
    assert ex._latched_grip(0.64) == 0.64 and ex._grip_latch == 0.64
    t = 100.0
    ex.safety.ring.z = 0.35                            # carry apex
    for _ in range(20):
        ex._apply_placement_descent(t, _target(0.35), 0.64)
        t += 0.008
    ex.safety.ring.z = 0.12                            # over the crate, in the band
    forced = None
    for _ in range(200):
        target, grip, overrode = ex._apply_placement_descent(t, _target(0.05), 0.64)
        t += 0.008
        assert target[2] >= 0.081 - 1e-12 and overrode  # the floor always holds
        if ex._grip_latch is None:
            forced = (target, grip)
            break
    assert forced is not None, "the supervisor never released"
    assert forced[1] == pytest.approx(0.25)            # the episode open aperture
    assert ex._latch_rearm_blocked is True
    # the pads are STILL loaded (the object has not fallen yet): no re-latch
    for _ in range(50):
        assert ex._latched_grip(0.64) == 0.64 and ex._grip_latch is None
        ex._apply_placement_descent(t, _target(0.12), 0.64)
        t += 0.008
    assert ex._descent_blocks_latch() is True


def test_the_latch_rearms_once_the_tool_leaves_the_crate(monkeypatch):
    ex, clock = _ex(monkeypatch, descent=cfg())
    ex.safety.ring.z = 0.07
    ex._latched_grip(0.64)
    t = 100.0
    ex.safety.ring.z = 0.35
    for _ in range(20):
        ex._apply_placement_descent(t, _target(0.35), 0.64)
        t += 0.008
    ex.safety.ring.z = 0.12
    for _ in range(400):
        ex._apply_placement_descent(t, _target(0.05), 0.64)
        t += 0.008
        if ex._grip_latch is None:
            break
    # retract out of the band, then back over the object with the pads unloaded
    ex.safety.ring.z, ex.safety.ring.y = 0.30, -0.27
    ex.safety.contact_load = {"left": 0.0, "right": 0.0}
    for _ in range(400):
        ex._apply_placement_descent(t, _target(0.30, y=-0.27), 0.2)
        ex._latched_grip(0.2)
        t += 0.008
    assert ex._descent_blocks_latch() is False
    ex.safety.contact_load = {"left": 9.0, "right": 9.0}
    ex.safety.ring.z = 0.07
    ex._apply_placement_descent(t, _target(0.07, y=-0.27), 0.60)
    assert ex._latched_grip(0.60) == 0.60 and ex._grip_latch == 0.60


def test_a_stale_plan_still_gets_the_forced_release_to_the_gripper(monkeypatch):
    """`grip` is None while a stale plan holds the pose; a release must still
    reach the gripper mailbox."""
    ex, _ = _ex(monkeypatch, descent=cfg())
    ex.safety.ring.z = 0.07
    ex._latched_grip(0.64)
    t = 100.0
    ex.safety.ring.z = 0.35
    for _ in range(20):
        ex._apply_placement_descent(t, _target(0.35), None)
        t += 0.008
    ex.safety.ring.z = 0.12
    for _ in range(200):
        _, grip, _ = ex._apply_placement_descent(t, _target(0.05), None)
        t += 0.008
        if ex._grip_latch is None:
            assert grip == pytest.approx(0.25)
            return
    raise AssertionError("no forced release")


def test_off_by_default_is_byte_identical(monkeypatch):
    ex, clock = _ex(monkeypatch)
    assert ex.placement_descent is None
    assert ex._descent_blocks_latch() is False
    # the 09-08 placement-release sequence still behaves exactly as pinned in
    # tests/test_grip_latch_release.py
    object.__setattr__(ex.hw.safety, "grip_latch_release_drop", 0.06)
    object.__setattr__(ex.hw.safety, "grip_latch_release_s", 0.5)

    def t(grip, z):
        ex.safety.ring.z = z
        clock[0] += 0.008
        return ex._latched_grip(grip)

    assert t(0.64, 0.07) == 0.64 and ex._grip_latch == 0.64
    for _ in range(50):
        assert t(0.64, 0.30) == 0.64
    for _ in range(30):
        assert t(0.35, 0.10) == 0.64
    for _ in range(40):
        g = t(0.35, 0.10)
    assert g == 0.35 and ex._grip_latch is None
    # and the servo loop never enters the supervisor when it is not configured
    src = inspect.getsource(ChunkExecutor._run)
    assert "if self.placement_descent is not None:" in src
    assert src.count("_apply_placement_descent") == 1


def test_the_supervisor_can_never_undo_a_geometric_clamp():
    """The hook re-clamps the target through the SafetyMonitor, like completion."""
    src = inspect.getsource(ChunkExecutor._run)
    hook = src.split("if self.placement_descent is not None:")[1]
    assert "self.safety.clamp_target(target)" in hook.split("boundary =")[0]


def test_the_halt_state_and_the_trace_carry_the_supervisor(monkeypatch):
    from phantom.deploy import planner as PL
    from phantom.deploy import runtime as RT

    ex, _ = _ex(monkeypatch, descent=cfg())
    ex._halt("safety_stop")
    assert ex.halt_state["placement_descent"]["variant"] == "descend_then_release_v1"
    assert ex.halt_state["placement_descent"]["task"] == "waffles"
    assert "placement_descent" in inspect.getsource(PL.PlannerLoop.run)
    assert "placement_descent" in inspect.getsource(RT.DeploymentRuntime.run_episode)


# ---------------------------------------------------------------------------
# entry point: run_deploy
# ---------------------------------------------------------------------------

def _main(monkeypatch, tmp_path, argv=(), *, task="waffles", expect_rc=0):
    """run_deploy.main() over a stubbed session; return the DeploymentRuntime kwargs."""
    from phantom.deploy import start_pose as sp
    from phantom.deploy.runtime import EpisodeResult
    from phantom.scripts import run_deploy as RD
    from phantom_test_utils import make_small_hw

    hw = make_small_hw(mode={"drivers": "real"})
    seen: dict = {}

    class _StubRuntime:
        def __init__(self, hw_, policy, mode, out_root, **kw):
            seen.update(kw)
            self.rig = SimpleNamespace(
                arm=_StubArm(set()),
                gripper=SimpleNamespace(get_state=lambda: SimpleNamespace(
                    position=0.0, obj=3.0)))
            self.recorder = SimpleNamespace(relabel=lambda *a, **k: None)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run_episode(self, **kw):
            seen["episode"] = dict(kw)
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
    stats = SimpleNamespace(tcp_z_min=0.052, gripper_mean=0.25, task=task,
                            tcp_min=None, tcp_max=None)
    monkeypatch.setattr(sp, "load_start_stats", lambda: {task: stats})
    monkeypatch.setattr(sp, "start_sigma_report",
                        lambda *a, **k: (np.zeros(7), "in distribution"))
    monkeypatch.setattr(sp, "move_to_start",
                        lambda *a, **k: (np.zeros(6), 0.33))
    rc = RD.main(["--system", "teacher", "--task", task, "--tiny",
                  "--episodes", "1", *argv])
    assert rc == expect_rc, rc
    return seen


def test_run_deploy_is_off_without_the_flag(monkeypatch, tmp_path):
    seen = _main(monkeypatch, tmp_path)
    assert seen["placement_descent"] is None
    assert "placement_descent" not in seen["deploy_overrides"]
    assert "descent:off" in seen["episode"]["tags"]


def test_run_deploy_parses_the_placement_descent_flag(monkeypatch, tmp_path):
    seen = _main(monkeypatch, tmp_path,
                 argv=("--placement-descent", CONFIG_PATH))
    descent = seen["placement_descent"]
    assert descent.release_z_m == pytest.approx(0.081)
    assert descent.release_y_min == pytest.approx(0.0)
    assert descent.task == "waffles"
    record = seen["deploy_overrides"]["placement_descent"]
    assert record["enabled"] is True
    assert record["release_z_m"] == pytest.approx(0.081)
    assert record["variant"] == "descend_then_release_v1"
    assert len(record["config_sha256"]) == 64


def test_run_deploy_leaves_an_unmeasured_task_off(monkeypatch, tmp_path):
    seen = _main(monkeypatch, tmp_path, task="egg",
                 argv=("--placement-descent", CONFIG_PATH))
    assert seen["placement_descent"] is None
    record = seen["deploy_overrides"]["placement_descent"]
    assert record["enabled"] is False and record["task"] == "egg"


def test_run_deploy_overrides_the_native_release_dwell(monkeypatch, tmp_path):
    seen = _main(monkeypatch, tmp_path, argv=("--grip-latch-release-s", "0.35"))
    assert seen["deploy_overrides"]["grip_latch_release_s"] == pytest.approx(0.35)
    assert seen["base_hw"].safety.grip_latch_release_s == pytest.approx(0.5)
    seen2 = _main(monkeypatch, tmp_path)
    assert "grip_latch_release_s" not in seen2["deploy_overrides"]
    _main(monkeypatch, tmp_path, argv=("--grip-latch-release-s", "-1"), expect_rc=2)


def test_the_episode_tag_names_the_supervisor(monkeypatch, tmp_path):
    """A paired cell must never silently mix the two controllers."""
    seen = _main(monkeypatch, tmp_path, argv=("--placement-descent", CONFIG_PATH))
    assert "descent:81mm" in seen["episode"]["tags"]
    carton = _main(monkeypatch, tmp_path, task="Carton",
                   argv=("--placement-descent", CONFIG_PATH))
    assert "descent:118mm" in carton["episode"]["tags"]


def test_recorded_grip_follows_the_supervisor_override():
    """S1 (rig 09-13): STREAM_ACTIONS must carry the aperture that went out."""
    from phantom.deploy.executor import ChunkExecutor
    ex = ChunkExecutor.__new__(ChunkExecutor)
    assert ex._recorded_grip(0.62) == 0.62                 # no supervisor override -> plan value
    ex._descent_grip_sent = 0.10
    assert ex._recorded_grip(0.62) == 0.10                 # forced open recorded as sent
    ex._descent_grip_sent = None
    assert ex._recorded_grip(0.62) == 0.62
