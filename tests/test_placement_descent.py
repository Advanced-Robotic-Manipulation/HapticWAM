"""Descend-then-release supervisor: permission gating and vertical descent override (CPU only)."""
import numpy as np
import pytest

from phantom.deploy.placement_descent import PlacementDescentConfig, PlacementDescentSupervisor
from phantom.deploy.release_controller import PlacementReleaseConfig, PlacementReleaseController


def cfg(**kw):
    base = dict(release_z_max_m=.21, descent_speed_m_s=.10, max_descent_s=6.0, activation_command_delta=12 / 255)
    base.update(kw)
    return PlacementDescentConfig(**base)


def test_config_validation():
    with pytest.raises(ValueError):
        cfg(descent_speed_m_s=.5)
    with pytest.raises(ValueError):
        cfg(release_z_max_m=-.1)
    with pytest.raises(ValueError):
        cfg(variant="placement_descent_v2")
    assert cfg().to_dict()["variant"] == "placement_descent_v1"


def test_idle_without_intent_and_descends_with_intent():
    s = PlacementDescentSupervisor(cfg())
    tcp = np.array([-.42, .10, .37, 0, 0, 0])
    s.update(0.0, measured_tcp=tcp, policy_grip=.62, reference_command=.62, in_volume=True, loaded=True)
    assert s.state == "idle" and s.permission_allowed
    assert np.array_equal(s.motion_target(0.0, tcp, np.ones(6)), np.ones(6))
    # policy asks to open 0.07 below the loaded reference while high over the bin
    s.update(0.008, measured_tcp=tcp, policy_grip=.55, reference_command=.62, in_volume=True, loaded=True)
    assert s.state == "descending" and not s.permission_allowed
    target = s.motion_target(0.016, tcp, np.array([-.40, .12, .40, 1, 1, 1]))
    assert target[0] == pytest.approx(-.42) and target[1] == pytest.approx(.10)   # x/y held
    assert target[2] < .37 and target[2] >= .21
    assert np.allclose(target[3:], tcp[3:])                                     # orientation held
    # descend at 0.1 m/s: after 1.6 s the commanded z is at the release height
    z = None
    for i in range(1, 201):
        t = 0.016 + 0.008 * i
        z = s.motion_target(t, tcp, np.zeros(6))[2]
    assert z == pytest.approx(.21)
    # measured arrives at the release height -> permission opens, x/y still held
    low = tcp.copy(); low[2] = .215
    s.update(2.0, measured_tcp=low, policy_grip=.55, reference_command=.62, in_volume=True, loaded=True)
    assert s.state == "at_release_height" and s.permission_allowed
    hold = s.motion_target(2.008, low, np.array([-.30, .0, .35, 0, 0, 0]))
    assert hold[0] == pytest.approx(-.42) and hold[2] == pytest.approx(.21)


def test_withdrawn_intent_cancels_and_timeout_yields():
    s = PlacementDescentSupervisor(cfg(max_descent_s=1.0))
    tcp = np.array([-.42, .10, .37, 0, 0, 0])
    s.update(0.0, measured_tcp=tcp, policy_grip=.55, reference_command=.62, in_volume=True, loaded=True)
    assert s.state == "descending"
    s.update(0.5, measured_tcp=tcp, policy_grip=.62, reference_command=.62, in_volume=True, loaded=True)
    assert s.state == "idle" and s.last_event == "descent_cancelled"
    s.update(0.6, measured_tcp=tcp, policy_grip=.55, reference_command=.62, in_volume=True, loaded=True)
    s.update(1.7, measured_tcp=tcp, policy_grip=.55, reference_command=.62, in_volume=True, loaded=True)
    assert s.state == "yielded" and s.permission_allowed and s.yield_reason == "descent_timeout"
    assert np.array_equal(s.motion_target(1.8, tcp, np.ones(6)), np.ones(6))
    s.update(1.9, measured_tcp=tcp, policy_grip=.62, reference_command=.62, in_volume=True, loaded=True)
    assert s.state == "idle"


def test_release_config_accepts_descent_and_reports_variant():
    base = dict(tcp_min_m=(-.56, -.06, -.006), tcp_max_m=(-.22, .18, .26), finish_after_release=True,
                unlatched_finish=None, relative_release=None)
    plain = PlacementReleaseConfig(**base)
    assert "descent" not in plain.to_dict()
    with_descent = PlacementReleaseConfig(**base, descent=dict(release_z_max_m=.21))
    assert with_descent.to_dict()["descent"]["release_z_max_m"] == .21
    with pytest.raises(ValueError):
        PlacementReleaseConfig(**base, descent=dict(release_z_max_m=.30))   # above the legacy volume
    c = PlacementReleaseController(with_descent)
    assert c.variant.endswith("_descent_v1") and c.descent is not None
    assert np.array_equal(c.descent_target(0.0, np.zeros(6), np.ones(6)), np.ones(6))
    # legacy path: a policy opening over the bin but high is NOT committed while the supervisor descends
    c.note_latch(.62)
    assert c.phase == "holding"
    tcp = np.array([-.42, .10, .25, 0, 0, 0])
    for i in range(60):
        c.update(0.008 * i, tcp=tcp, policy_grip=.40, measured_grip=.62, pad_loads={"l": 3.0, "r": 3.0}, eligible=True)
    assert c.phase == "holding" and c.descent.state == "descending"
    low = tcp.copy(); low[2] = .21
    for i in range(60, 120):
        c.update(0.008 * i, tcp=low, policy_grip=.40, measured_grip=.62, pad_loads={"l": 3.0, "r": 3.0}, eligible=True)
    assert c.phase == "releasing" and c.descent.state == "at_release_height"
