"""Placement release honors played policy intent without reading object state."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from phantom_test_utils import make_small_hw

from phantom.sim.policy_adapter import SimulationPolicyAdapter
from phantom.sim.release_controller import (
    PlacementReleaseConfig,
    PlacementReleaseController,
)
from tools.sim.deployment_filters import TerminalVetoFilter

INSIDE = np.array([-0.39, 0.07, 0.14, 0, np.pi, 0])
OUTSIDE = np.array([-0.39, -0.25, 0.14, 0, np.pi, 0])
CONFIG = {"tcp_min_m": [-0.55, -0.076, 0.058], "tcp_max_m": [-0.23, 0.149, 0.274]}


def update(
    controller, t, *, tcp=INSIDE, grip=0.4, measured=0.55, loads=None, eligible=True
):
    return controller.update(
        t,
        tcp=tcp,
        policy_grip=grip,
        measured_grip=measured,
        pad_loads={"left": 3.0, "right": 3.0} if loads is None else loads,
        eligible=eligible,
    )


def controller():
    return PlacementReleaseController(PlacementReleaseConfig.from_dict(CONFIG))


def test_requires_previously_loaded_latch_and_continuous_policy_opening_in_volume():
    c = controller()
    for t in (0, 1):
        assert not update(c, t)
    assert c.phase == "unarmed"
    c.note_latch(0.63)
    assert not update(c, 2, tcp=OUTSIDE)
    assert not update(c, 3, tcp=OUTSIDE)
    assert not update(c, 4)
    assert not update(c, 4.19)
    # A brief close, leaving the volume or a veto-generated opening resets dwell.
    assert not update(c, 4.192, grip=0.46)
    assert not update(c, 4.2)
    assert not update(c, 4.3, eligible=False)
    assert not update(c, 4.4)
    assert not update(c, 4.592)
    assert update(c, 4.6)
    assert c.last_event == "policy_release_committed"
    assert c.committed_at == 4.6


def test_rearm_waits_for_measured_open_unload_then_new_policy_close():
    c = controller()
    c.note_latch(0.63)
    update(c, 0)
    assert update(c, 0.2)
    # Residual contact cannot re-arm and leaving the release volume cannot revoke release.
    assert update(c, 0.4, grip=0.7, tcp=OUTSIDE)
    c.note_latch(0.7)
    assert c.phase == "releasing"
    assert update(c, 0.6, measured=0.2, loads={})
    assert update(c, 0.8, measured=0.55, loads={"left": 0, "right": 0})
    assert c.unloaded_since is None
    assert update(c, 1, measured=0.3, loads={"left": 0, "right": 0})
    assert update(c, 1.2, measured=0.3, loads={"left": 0, "right": 0})
    assert c.phase == "waiting_for_close"
    assert update(c, 1.21, grip=0.7, eligible=False)
    assert not update(c, 1.22, grip=0.7)
    c.note_latch(0.7)
    assert c.phase == "holding"
    c.reset()
    assert c.phase == "unarmed" and c.committed_at is None


@pytest.mark.parametrize(
    "changes",
    [
        {"tcp_max_m": [-0.56, 0.14, 0.27]},
        {"opening_hold_s": 0},
        {"unloaded_force_max_n": float("nan")},
        {"rearm_close_command_min": 0.4},
        {"tcp_min_m": [0, 1]},
    ],
)
def test_invalid_release_configuration_rejected(changes):
    with pytest.raises(ValueError):
        PlacementReleaseConfig.from_dict({**CONFIG, **changes})


def setup(*, enabled=True, veto=False):
    hw = make_small_hw(
        safety={
            "wrist_extension_stop_m": None,
            "reach_clamp_m": None,
            "grip_latch_fz_n": 2.5,
            "lift_complete_z_m": 0,
            "workspace_m": {"x": [-0.7, 0.15], "y": [-0.5, 0.3], "z": [0.03, 0.8]},
        }
    )
    ad = SimulationPolicyAdapter(
        hw,
        SimpleNamespace(),
        mode="student",
        release_config=CONFIG if enabled else None,
        plan_filter=TerminalVetoFilter(
            hw, {"z_ref": 0.0415, "z_margin": 0.0615}, implementation="fd4a032"
        )
        if veto
        else None,
    )
    return ad


def observe(ad, t, *, tcp=INSIDE, load=3.0, measured=0.55, qd=None):
    ad.observe(
        t,
        rgb=np.zeros((12, 16, 3), dtype=np.uint8),
        q=np.array([0, -1.4, 1.5, -1.7, 1.4, 0]),
        qd=np.zeros(6) if qd is None else qd,
        tcp_pose=tcp,
        tcp_speed=np.zeros(6),
        gripper_state=np.array([measured, 2]),
        wrist_ft=np.zeros(6),
        tactile={
            s.name: {"wrench": np.array([0, 0, -load, 0, 0, 0])}
            for s in ad.hw.tactile.sensors
        },
    )


def plan(t, grip, *, diag=None):
    actions = np.zeros((16, 7))
    actions[:, 6] = grip
    return SimpleNamespace(
        t_created=t,
        t0_pose=INSIDE.copy(),
        actions=actions,
        action_times=t + np.arange(16) / 10,
        sigma=np.zeros(3),
        gate=0.0,
        p_evt=np.array([0.99, 0.01, 0, 0, 0]),
        cpk="contact",
        _cpk_token=123,
        latency_s=0.0,
        diag=diag or {},
    )


def tick(ad, t, **observed):
    observe(ad, t, **observed)
    cmd = ad.step(t)
    ad.report_execution(t, accepted=True, gripper_command=cmd.gripper)
    return cmd


def arm(ad):
    observe(ad, 0, load=0, measured=0.2)
    assert ad.submit(plan(0, 0.63), 0)
    cmd = ad.step(0)
    ad.report_execution(0, accepted=True, gripper_command=cmd.gripper)
    assert tick(ad, 0.008).gripper == pytest.approx(0.63)
    assert ad._grip_latch == pytest.approx(0.63)


@pytest.mark.parametrize("enabled", [True, False])
def test_executor_retains_preload_then_passes_exact_policy_command_when_opted_in(
    enabled,
):
    ad = setup(enabled=enabled)
    arm(ad)
    assert ad.submit(plan(0.008, 0.4), 0.008)
    for i in range(2, 27):
        assert tick(ad, i * 0.008).gripper == pytest.approx(0.63)
    cmd = tick(ad, 0.216)
    assert cmd.gripper == pytest.approx(0.4 if enabled else 0.63)
    if enabled:
        assert (
            cmd.diagnostics["placement_release"]["event"] == "policy_release_committed"
        )
        assert ad.gripper_cmd_at(np.array([0.216]))[0] == pytest.approx(0.4)
        # Residual pad loads persist; accepted subsequent openings are not re-latched.
        assert tick(ad, 0.224).gripper == pytest.approx(0.4)


@pytest.mark.parametrize("kind", ["outside", "recovery", "safety"])
def test_no_release_outside_volume_from_recovery_or_during_safety_stop(kind):
    ad = setup()
    arm(ad)
    diag = {"terminal_veto": {"action": "recovery_open"}} if kind == "recovery" else {}
    assert ad.submit(plan(0.008, 0.4, diag=diag), 0.008)
    for i in range(2, 28):
        kwargs = {"tcp": OUTSIDE} if kind == "outside" else {}
        if kind == "safety" and i == 27:
            kwargs["qd"] = np.full(6, 100.0)
        cmd = tick(ad, i * 0.008, **kwargs)
    assert cmd.gripper == pytest.approx(0.63)
    assert ad.release_controller.committed_at is None
    assert cmd.stopped == (kind == "safety")


def test_close_mask_preserves_only_original_opening_samples_inside_release_window():
    ad = setup(veto=True)
    arm(ad)
    # A future close makes the native whole chunk mask fire; the played opening
    # prefix must survive only in the configured volume after a loaded latch.
    raw = plan(0.008, 0.4)
    raw.actions[8:, 6] = 0.65
    original = deepcopy(raw)
    ad.plan_filter.state["g_min"] = 0.2
    ad.plan_filter.state["close_permitted"] = False
    filtered = ad.plan_filter(raw, ad.snapshot(), ad)
    np.testing.assert_array_equal(filtered.actions[:8, 6], 0.4)
    np.testing.assert_allclose(filtered.actions[8:, 6], 0.55)
    np.testing.assert_array_equal(filtered.actions[:, :6], raw.actions[:, :6])
    np.testing.assert_array_equal(raw.actions, original.actions)
    assert filtered._cpk_token is None
    assert filtered.diag["terminal_veto"][
        "placement_release_passthrough_indices"
    ] == list(range(8))
    assert ad.submit(filtered, 0.008)
    for i in range(2, 28):
        cmd = tick(ad, i * 0.008)
    assert cmd.gripper == pytest.approx(0.4)
    # Current feedback, not the captured snapshot, governs mask permission.
    observe(ad, 0.224, tcp=OUTSIDE)
    filtered = ad.plan_filter(
        raw, SimpleNamespace(t=0.008, ur_state=ad.snapshot().ur_state), ad
    )
    np.testing.assert_allclose(filtered.actions[:, 6], 0.55)
