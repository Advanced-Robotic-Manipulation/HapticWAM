"""collect/safeguard.py — trip, latch, resume, runtime config, staleness."""

import time

import numpy as np
import pytest

from phantom_test_utils import make_small_hw

from phantom.data_collect.config import SafeguardConfig
from phantom.data_collect.safeguard import ArmGuard, TactileSafeguard


class FakeRing:
    """Minimal stand-in for SharedRingBuffer.latest(1)."""

    def __init__(self):
        self.ts = []
        self.data = {}

    def push(self, ts, **fields):
        self.ts = [ts]
        self.data = {k: np.asarray([v]) for k, v in fields.items()}

    def latest(self, n=1):
        return (np.asarray(self.ts), self.data) if self.ts else (np.asarray([]), {})


@pytest.fixture
def hw():
    return make_small_hw()


def _rings(hw, *, force=0.0, depth=0.0):
    """Rings with a FRESH sample (the safeguard trips on stale streams)."""
    rings = {}
    fields = np.zeros((*hw.recording.field_ds.hw, hw.tactile.field_ch),
                      dtype=np.float32)
    fields[..., 2] = depth              # depth channel (index 2 of the 8-stack)
    wrench = np.array([force, 0, 0, 0, 0, 0], dtype=np.float32)
    for s in hw.tactile.sensors:
        r = FakeRing()
        r.push(time.perf_counter(), fields_ds=fields, wrench=wrench,
               area=np.float32(0.0))
        rings[f"tactile_{s.name}"] = r
    return rings


def test_no_trip_below_limits(hw):
    sg = TactileSafeguard(hw, _rings(hw, force=1.0), SafeguardConfig())
    assert sg.check_once() is None
    assert sg.tripped is None


def test_force_trip_latches_and_fires_once(hw):
    calls = []
    sg = TactileSafeguard(hw, _rings(hw, force=9.0),
                          SafeguardConfig(force_limit_n=4.0),
                          on_trip=calls.append)
    info = sg.check_once()
    assert info is not None and info.kind == "force" and info.value > 4.0
    # latched: repeated checks return the same trip, callback fired once
    assert sg.check_once() is info
    assert sg.check_once() is info
    assert len(calls) == 1
    # release the load — STILL latched (only reset() clears it)
    sg.rings.update(_rings(hw, force=0.0))
    assert sg.check_once() is info
    sg.reset()
    assert sg.tripped is None
    assert sg.check_once() is None


def test_depth_fallback_trip(hw):
    sg = TactileSafeguard(hw, _rings(hw, depth=0.9),
                          SafeguardConfig(depth_limit=0.5))
    info = sg.check_once()
    assert info is not None and info.kind == "depth"


def test_runtime_threshold_edit(hw):
    sg = TactileSafeguard(hw, _rings(hw, force=3.0),
                          SafeguardConfig(force_limit_n=4.0))
    assert sg.check_once() is None          # 3 N under the 4 N default
    sg.configure(force_limit_n=2.0)         # tighten from the panel
    assert sg.check_once().kind == "force"
    with pytest.raises(ValueError):
        sg.configure(force_limit_n=-1.0)


def test_disable_clears_latch_and_stops_checking(hw):
    sg = TactileSafeguard(hw, _rings(hw, force=9.0),
                          SafeguardConfig(force_limit_n=4.0))
    assert sg.check_once() is not None
    sg.configure(enabled=False)
    assert sg.tripped is None
    assert sg.check_once() is None          # disabled: never trips
    sg.configure(enabled=True)
    assert sg.check_once() is not None      # re-enabled: trips again


def test_trip_names_the_sensor(hw):
    rings = _rings(hw, force=0.0)
    hot = np.array([9.0, 0, 0, 0, 0, 0], dtype=np.float32)
    name = hw.tactile.sensors[1].name
    fields = rings[f"tactile_{name}"].data["fields_ds"][0]
    rings[f"tactile_{name}"].push(time.perf_counter(), fields_ds=fields,
                                  wrench=hot, area=np.float32(0.0))
    sg = TactileSafeguard(hw, rings, SafeguardConfig(force_limit_n=4.0))
    assert sg.check_once().sensor == name


def test_stale_stream_trips(hw):
    """A dead/hung tactile worker must TRIP, not silently blind the guard
    behind its last benign sample (review finding)."""
    rings = _rings(hw, force=0.0)                 # benign values...
    name = hw.tactile.sensors[0].name
    r = rings[f"tactile_{name}"]
    r.ts = [time.perf_counter() - 5.0]            # ...but frozen 5 s ago
    sg = TactileSafeguard(hw, rings, SafeguardConfig())
    info = sg.check_once()
    assert info is not None and info.kind == "stale" and info.sensor == name


# --------------------------------------------------------------------------
def _push_arm(r, *, f=0.0, pstop=False):
    r.push(1.0, ft=np.array([f, 0, 0, 0, 0, 0], dtype=np.float64),
           protective_stop=np.uint8(pstop), q=np.zeros(6),
           tcp_pose=np.zeros(6), tcp_speed=np.zeros(6))


def _arm_ring(*, pstop=False, f=0.0):
    r = FakeRing()
    _push_arm(r, f=f, pstop=pstop)
    return r


def test_arm_guard_pstop(hw):
    g = ArmGuard(hw, {"arm": _arm_ring(pstop=True)})
    assert g.check() == "pstop"
    assert not g.recovered()


def test_arm_guard_wrench_is_deviation_debounced(hw):
    """Wrench trips on a SUSTAINED deviation from the rolling baseline — never
    on the first sample (that just seeds the baseline; the CB3 estimate carries
    a huge static bias) and only after the debounce count."""
    r = _arm_ring(f=hw.safety.wrench_limit_N + 30)   # big absolute value...
    g = ArmGuard(hw, {"arm": r})
    assert g.check() is None                         # ...but it's the baseline -> no trip
    _push_arm(r, f=hw.safety.wrench_limit_N + 30 + hw.safety.wrench_limit_N + 20)
    n = hw.safety.wrench_debounce_ticks
    seen = [g.check() for _ in range(n)]
    assert seen[:-1] == [None] * (n - 1)             # debounced
    assert seen[-1] == "wrench"                      # trips only when sustained


def test_arm_guard_ignores_transient_spike(hw):
    """A one-tick acceleration spike (bigger than the limit) must not trip —
    the false-trip mode the raw absolute check suffered."""
    r = _arm_ring(f=0.0)
    g = ArmGuard(hw, {"arm": r})
    g.check()
    big = hw.safety.wrench_limit_N + 100
    for _ in range(5):
        _push_arm(r, f=big); assert g.check() is None   # spike...
        _push_arm(r, f=0.0); assert g.check() is None   # ...gone: counter resets


def test_arm_guard_tracks_out_slow_bias(hw):
    """A slowly-growing pose/gravity bias (well past the absolute limit) is
    tracked by the baseline and never trips."""
    r = _arm_ring(f=0.0)
    g = ArmGuard(hw, {"arm": r})
    g.check()
    lim = hw.safety.wrench_limit_N
    f = 0.0
    for _ in range(400):
        f += lim * 0.01                              # slow ramp, << per-tick delta budget
        _push_arm(r, f=f)
        assert g.check() is None
    assert f > lim * 3                               # drifted far past the absolute limit


def test_arm_guard_recovered_uses_deviation(hw):
    r = _arm_ring(f=0.0)
    g = ArmGuard(hw, {"arm": r})
    g.check()
    _push_arm(r, f=hw.safety.wrench_limit_N * 0.5)   # within 0.8*limit deviation
    assert g.recovered()
    _push_arm(r, f=hw.safety.wrench_limit_N * 0.95)  # above 0.8*limit
    assert not g.recovered()


# ------------------------------------------- shipped config characterization
# The 2026-08 data audit found 165/180 whiteboard v4 episodes peaking ABOVE the
# 30 N DM-Tac pad ceiling (max 58.8 N) with the safeguard's limit set to 15 N
# and nothing ever tripping. These two tests pin down why: the mechanism is
# sound and would have caught every one of those peaks — it is switched OFF in
# the shipped config, so check_once() never looks at a sample. Flipping that
# default must break a test, not a pad.
def _shipped_safeguard_cfg():
    """SafeguardConfig exactly as committed — the sibling data_collect.local
    .yaml is deliberately NOT merged, so a rig-local override cannot mask a
    change to the default everyone else ships."""
    import yaml
    from phantom.data_collect.config import DEFAULT_COLLECT_YAML
    raw = yaml.safe_load(DEFAULT_COLLECT_YAML.read_text())
    return SafeguardConfig.model_validate(raw.get("safeguard", {}))


def test_shipped_config_leaves_the_pad_guard_switched_off(hw):
    cfg = _shipped_safeguard_cfg()
    assert cfg.enabled is False, (
        "the pad guard is ON now — update this test and tell the operator, the "
        "whiteboard task normally loads the pads well past force_limit_n")
    sg = TactileSafeguard(hw, _rings(hw, force=58.8), cfg)   # audit's peak
    assert sg.check_once() is None, "disabled guard must not trip"
    assert sg.tripped is None


def test_shipped_force_limit_would_catch_a_whiteboard_peak(hw):
    """Same config with the panel toggle ON: the limit and the reduction are
    right (per-pad ‖F‖ of getForce()[:3], calibrated N). 40 N trips, the 10 N
    grasp-stop working load does not — the limits nest 10 < 15 < 30 N."""
    cfg = _shipped_safeguard_cfg().model_copy(update={"enabled": True})
    assert cfg.force_limit_n == 15.0

    calls = []
    sg = TactileSafeguard(hw, _rings(hw, force=40.0), cfg, on_trip=calls.append)
    info = sg.check_once()
    assert info is not None and info.kind == "force"
    assert info.value == pytest.approx(40.0) and info.limit == 15.0
    assert len(calls) == 1                       # teleop freezes, gripper opens

    sg.reset()
    sg.rings.update(_rings(hw, force=10.0))      # a normal force-limited grasp
    assert sg.check_once() is None
