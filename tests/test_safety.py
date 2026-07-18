"""Safety circuit: SafetyMonitor verdicts, recovery hysteresis, and the
gripper pad-force ceiling (previously untested)."""

import time
import uuid
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from phantom.deploy.safety import SafetyAction, SafetyMonitor
from phantom.recording.ringbuffer import SharedRingBuffer
from phantom_test_utils import make_hw, make_small_hw


@contextmanager
def _rings(hw):
    """Minimal arm + one tactile ring, benign data pre-pushed."""
    uid = uuid.uuid4().hex[:8]
    r = hw.recording
    fields_shape = (r.field_ds.h, r.field_ds.w, hw.tactile.field_ch)
    rings = {
        "arm": SharedRingBuffer(f"t_arm_{uid}", 16, {
            "ft": ((6,), "float64"), "protective_stop": ((), "uint8")}, create=True),
        "tactile_left": SharedRingBuffer(f"t_tac_{uid}", 16, {
            "fields_ds": (fields_shape, "float32")}, create=True),
    }
    try:
        push_arm(rings, ft=np.zeros(6))
        push_tactile(rings, depth=0.0, hw=hw)
        yield rings
    finally:
        for ring in rings.values():
            ring.close()   # owner: unlinks the shm too


def push_arm(rings, ft, pstop=False, ts=None):
    rings["arm"].push(ts if ts is not None else time.perf_counter(),
                      ft=np.asarray(ft, dtype=np.float64),
                      protective_stop=np.uint8(pstop))


def push_tactile(rings, depth, hw, ts=None):
    from phantom.data.derived import channel_slices
    r = hw.recording
    fields = np.zeros((r.field_ds.h, r.field_ds.w, hw.tactile.field_ch), dtype=np.float32)
    fields[..., channel_slices(hw.tactile)["depth"]] = depth
    rings["tactile_left"].push(ts if ts is not None else time.perf_counter(),
                               fields_ds=fields)


IN_BOX = np.array([0.0, -0.45, 0.25, 0.0, 3.14, 0.0])   # inside yaml workspace_m


def test_benign_is_ok():
    hw = make_small_hw()
    with _rings(hw) as rings:
        v = SafetyMonitor(hw, rings).check(time.perf_counter(), IN_BOX)
        assert v.action == SafetyAction.OK and not v.events


def test_wrench_limit_stops_episode():
    hw = make_small_hw()
    with _rings(hw) as rings:
        push_arm(rings, ft=[0, 0, hw.safety.wrench_limit_N + 5, 0, 0, 0])
        v = SafetyMonitor(hw, rings).check(time.perf_counter(), IN_BOX)
        assert v.action == SafetyAction.STOP_EPISODE
        assert any(e.kind == "wrench_limit" for e in v.events)


def test_protective_stop_wins_priority():
    hw = make_small_hw()
    with _rings(hw) as rings:
        push_arm(rings, ft=[0, 0, 100, 0, 0, 0], pstop=True)
        v = SafetyMonitor(hw, rings).check(time.perf_counter(), IN_BOX)
        assert v.action == SafetyAction.PROTECTIVE_STOP


def test_tactile_depth_estop():
    hw = make_small_hw()
    with _rings(hw) as rings:
        push_tactile(rings, depth=hw.safety.tactile_depth_limit + 0.1, hw=hw)
        v = SafetyMonitor(hw, rings).check(time.perf_counter(), IN_BOX)
        assert v.action == SafetyAction.STOP_EPISODE
        assert any(e.kind == "tactile_depth" for e in v.events)


def test_stale_tactile_stops():
    hw = make_small_hw()
    with _rings(hw) as rings:
        push_tactile(rings, depth=0.0, hw=hw, ts=time.perf_counter() - 5.0)
        v = SafetyMonitor(hw, rings).check(time.perf_counter(), IN_BOX)
        assert v.action == SafetyAction.STOP_EPISODE
        assert any("stale" in e.kind for e in v.events)


def test_workspace_clamp():
    hw = make_small_hw()
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        outside = IN_BOX.copy()
        outside[0] = 5.0
        v = mon.check(time.perf_counter(), outside)
        assert v.action == SafetyAction.CLAMP
        clamped = mon.clamp_target(outside)
        assert hw.safety.workspace_m.contains(clamped[:3])


def test_recovered_hysteresis():
    hw = make_small_hw()
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        assert mon.recovered()
        # above frac*limit but below limit: NOT recovered (no chatter)
        push_arm(rings, ft=[0, 0, 0.85 * hw.safety.wrench_limit_N, 0, 0, 0])
        push_tactile(rings, depth=0.0, hw=hw)
        assert not mon.recovered(frac=0.8)
        push_arm(rings, ft=np.zeros(6))
        push_tactile(rings, depth=0.0, hw=hw)
        assert mon.recovered(frac=0.8)


def test_recovered_requires_fresh_tactile():
    hw = make_small_hw()
    with _rings(hw) as rings:
        mon = SafetyMonitor(hw, rings)
        push_tactile(rings, depth=0.0, hw=hw, ts=time.perf_counter() - 5.0)
        assert not mon.recovered()


# ---------------------------------------------------------------------------
# gripper pad-force ceiling
# ---------------------------------------------------------------------------

def test_gripper_force_mapping(default_hw):
    g = default_hw.gripper
    lo, hi = g.force_range_N
    assert g.max_force_cmd == pytest.approx((g.cmd_force_limit_N - lo) / (hi - lo))
    # yaml default_force must sit under the pad ceiling
    assert g.default_force <= g.max_force_cmd
    assert lo + g.default_force * (hi - lo) <= g.cmd_force_limit_N


def test_gripper_default_force_over_ceiling_rejected():
    with pytest.raises(ValidationError, match="pad ceiling|cmd_force_limit"):
        make_hw(gripper={"default_force": 0.3})   # ≈ 85 N on the 2F-85


def test_gripper_driver_clamps_force(default_hw):
    from phantom.drivers.mock.robotiq import MockGripper
    g = MockGripper(default_hw.gripper)
    g.connect()
    g.activate()
    g.move(0.5, 0.5, 1.0)                        # ask for max force
    assert g.last_force == pytest.approx(default_hw.gripper.max_force_cmd)
    g.move(0.5, 0.5, 0.01)                       # under the ceiling: untouched
    assert g.last_force == pytest.approx(0.01)
    g.disconnect()


def test_gripper_close_position_clamped():
    """Pads on both fingers: full close must be capped at max_close_cmd."""
    from phantom.drivers.mock.robotiq import MockGripper
    hw = make_hw(gripper={"max_close_cmd": 0.61})
    g = MockGripper(hw.gripper)
    g.connect(); g.activate()
    g.move(1.0, 0.5, 0.01)                      # ask for FULL close
    assert g._target == pytest.approx(0.61)     # capped — pads never meet
    g.move(0.3, 0.5, 0.01)
    assert g._target == pytest.approx(0.3)      # under the cap: untouched
    g.disconnect()
