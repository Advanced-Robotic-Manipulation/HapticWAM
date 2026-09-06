"""Sensor/action parity and fail-closed simulator bridge regressions."""

import json
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from phantom_test_utils import make_small_hw

from phantom.sim.policy_adapter import SimulationPolicyAdapter

POSE = np.array([-0.30, -0.12, 0.25, 0.0, np.pi, 0.0])
Q = np.array([0.0, -1.4, 1.5, -1.7, 1.4, 0.0])


class Policy:
    def __init__(self):
        self.calls = []
        self.resets = 0

    def reset_episode(self):
        self.resets += 1

    def replan(self, obs, prev, tcp):
        self.calls.append((obs, prev, tcp.copy()))
        return plan(obs.t, tcp=tcp, latency=0.2)


def plan(t=0.0, *, tcp=POSE, latency=0.0, delta=0.001, grip=0.4):
    actions = np.zeros((16, 7), dtype=np.float64)
    actions[:, 0] = delta
    actions[:, 6] = grip
    return SimpleNamespace(
        t_created=t,
        t0_pose=tcp.copy(),
        actions=actions,
        action_times=t + latency + np.arange(16) / 10,
        sigma=np.zeros(3),
        gate=0.0,
        p_evt=np.zeros(5),
        cpk=None,
        latency_s=latency,
        diag={},
    )


def adapter(**kwargs):
    hw = make_small_hw(
        safety={
            "wrist_extension_stop_m": None,
            "reach_clamp_m": None,
            "workspace_m": {"x": [-0.7, 0.15], "y": [-0.5, 0.3], "z": [0.03, 0.8]},
        }
    )
    return SimulationPolicyAdapter(hw, Policy(), **kwargs)


def observe(
    ad,
    t=0.0,
    *,
    pose=POSE,
    q=Q,
    qd=None,
    ft=None,
    grip=0.31,
    rgb=None,
    camera_t=None,
    tactile=None,
    retain_rgb=False,
):
    frame = np.zeros((12, 16, 3), dtype=np.uint8) if rgb is None else rgb
    ad.observe(
        t,
        rgb=None if retain_rgb else frame,
        q=q,
        qd=np.zeros(6) if qd is None else qd,
        tcp_pose=pose,
        tcp_speed=np.zeros(6),
        gripper_state=np.array([grip, 3]),
        wrist_ft=np.zeros(6) if ft is None else ft,
        camera_t=camera_t,
        tactile=tactile,
    )


def tick(ad, t, *, accepted=True, grip=True):
    command = ad.step(t)
    ad.report_execution(
        t, accepted=accepted, gripper_command=command.gripper if grip else None
    )
    return command


def test_snapshot_preserves_rgb_order_and_26d_proprio():
    ad = adapter()
    rgba = np.zeros((12, 16, 4), dtype=np.uint8)
    rgba[:] = [11, 22, 33, 255]
    observe(ad, rgb=rgba)
    snap = ad.snapshot()
    np.testing.assert_array_equal(snap.rgb[0, 0], [11, 22, 33])
    assert snap.rgb.dtype == np.uint8
    assert snap.ur_state.shape == (26,)
    np.testing.assert_allclose(snap.ur_state[:6], Q)
    np.testing.assert_allclose(snap.ur_state[12:18], POSE)
    np.testing.assert_allclose(snap.ur_state[-2:], [0.31, 3])
    assert snap.prev_chunk is None  # matches deploy's one-row warm-up


def test_wrist_window_resamples_timestamps_and_masked_modes_zero_it():
    ad = adapter()
    for t in [0.0, 0.013, 0.027, 0.048, 0.075, 0.1]:
        observe(ad, t, ft=np.arange(1, 7) * t)
    snap = ad.snapshot()
    expected = np.linspace(0, 0.1, ad.hw.wrist_ft.window_len)[:, None] * np.arange(1, 7)
    np.testing.assert_allclose(snap.wrist_window, expected, atol=1e-7)
    ablated = adapter(mode="vision_only")
    observe(ablated, ft=np.ones(6) * 7)
    assert not ablated.snapshot().wrist_window.any()


def test_measured_prev_chunk_uses_rotvec_continuity_and_confirmed_gripper():
    ad = adapter()
    for i in range(21):
        t = i / 10
        pose = POSE.copy()
        pose[0] += t * 0.01
        # Equivalent axis-angle representations must not create 2pi actions.
        pose[4] = np.pi if i % 2 == 0 else -np.pi
        observe(ad, t, pose=pose)
        ad.step(t)
        ad.report_execution(t, accepted=True, gripper_command=0.42)
    prev = ad.snapshot().prev_chunk
    np.testing.assert_allclose(prev[:, 0], 0.001, atol=1e-7)
    np.testing.assert_allclose(prev[:, 3:6], 0, atol=1e-7)
    np.testing.assert_allclose(prev[:, 6], 0.42)


def test_policy_latency_activates_on_sim_time_and_preserves_accepted_feedback():
    ad = adapter()
    observe(ad)
    pending = ad.replan()
    assert pending.action_times[0] == pytest.approx(0.2)
    assert ad.policy.calls[-1][1] is None
    assert not tick(ad, 0).diagnostics["plan_activated"]
    observe(ad, 0.199)
    assert not tick(ad, 0.199).diagnostics["plan_activated"]
    observe(ad, 0.2)
    activated = tick(ad, 0.2)
    assert activated.diagnostics["plan_activated"]
    expected = POSE + pending.actions[0, :6] * 0.001 * ad.hw.control.action_rate_hz
    np.testing.assert_allclose(activated.tcp_pose, expected)
    observe(ad, 0.3)
    ad.replan(latency_s=0)
    assert ad.policy.calls[-1][1] is ad._plan


def test_first_activation_tick_uses_the_live_executor_governor():
    ad = adapter()
    observe(ad)
    pending = ad.replan()
    pending.sigma[:] = ad.hw.safety.governor.sigma_hi
    tick(ad, 0)
    observe(ad, 0.192)
    tick(ad, 0.192)
    observe(ad, 0.2)
    command = tick(ad, 0.2)
    governed_dt = 0.008 * ad.hw.safety.governor.min_scale
    assert command.diagnostics["plan_activated"]
    assert command.diagnostics["play_time_s"] == pytest.approx(governed_dt)
    np.testing.assert_allclose(
        command.tcp_pose,
        POSE + pending.actions[0, :6] * governed_dt * ad.hw.control.action_rate_hz,
    )


def test_long_inference_does_not_add_an_extra_model_period_after_activation():
    ad = adapter()
    observe(ad)
    ad.replan(latency_s=0.4)
    tick(ad, 0)
    observe(ad, 0.4)
    tick(ad, 0.4)
    assert ad.ready_for_replan(0.4)


def test_additive_delta_action_semantics_and_head_cap():
    ad = adapter(max_play_steps=3)
    p = plan()
    p.actions[:, :6] = [0.001, -0.002, 0.003, 0.01, 0.02, -0.03]
    halfway, grip = ad._pose_at(p, 0.15)
    np.testing.assert_allclose(halfway, POSE + 1.5 * p.actions[0, :6])
    np.testing.assert_allclose(
        ad._pose_at(p, 90)[0], POSE + (3 - 1e-6) * p.actions[0, :6]
    )
    assert grip == 0.4


def test_swap_rebases_to_confirmed_target_without_mutating_proposal():
    ad = adapter()
    observe(ad)
    p = plan(tcp=POSE + 1)
    before = deepcopy(p)
    assert ad.submit(p, 0.05)
    np.testing.assert_allclose(ad._pose_at(ad._plan, 0.05)[0], POSE)
    np.testing.assert_array_equal(p.t0_pose, before.t0_pose)
    np.testing.assert_array_equal(p.actions, before.actions)
    stale = plan(t=-10)
    assert not ad.submit(stale, 0.05)
    assert ad._plan is not stale


def test_rotation_branch_and_per_tick_command_rate_limits():
    ad = adapter(max_play_steps=None)
    observe(ad)
    p = plan(delta=0.5)
    p.actions[:, 4] = -2 * np.pi
    assert ad.submit(p, 0)
    command = tick(ad, 0)
    period = 1 / ad.hw.control.executor_rate_hz
    assert (
        np.linalg.norm(command.tcp_pose[:3] - POSE[:3])
        <= ad.hw.arm.limits.tcp_speed_m_s * period + 1e-10
    )
    assert (
        np.linalg.norm(command.tcp_pose[3:] - POSE[3:])
        <= ad.hw.arm.limits.joint_speed_rad_s * period + 1e-10
    )
    observe(ad, 0.1)
    delayed = tick(ad, 0.1)
    assert delayed.dt == pytest.approx(2 * period)
    assert (
        np.linalg.norm(delayed.tcp_pose[:3] - command.tcp_pose[:3])
        <= ad.hw.arm.limits.tcp_speed_m_s * 2 * period + 1e-10
    )


def test_governor_changes_playback_time_without_changing_path():
    fast, slow = adapter(), adapter()
    for ad in (fast, slow):
        observe(ad)
        p = plan()
        p.sigma[:] = 0 if ad is fast else ad.hw.safety.governor.sigma_hi
        ad.submit(p, 0)
        tick(ad, 0)
    assert slow._play_time == pytest.approx(
        fast._play_time * slow.hw.safety.governor.min_scale
    )


def test_ik_rejection_never_enters_executed_history_and_stops_at_limit():
    ad = adapter(ik_reject_limit=2)
    observe(ad)
    ad.submit(plan(grip=0.8), 0)
    tick(ad, 0, accepted=False, grip=False)
    np.testing.assert_array_equal(ad._last_cmd, POSE)
    assert ad.gripper_cmd_at([0]) is None
    observe(ad, 0.01)
    tick(ad, 0.01, accepted=False, grip=False)
    assert ad.stopped_reason == "ik_rejected"
    assert ad.ik_rejects_total == 2
    observe(ad, 0.02)
    command = tick(ad, 0.02)
    assert command.stopped
    assert command.gripper == 0.31


def test_feedback_can_report_ik_shortening_and_requires_pending_timestamp():
    ad = adapter()
    observe(ad)
    ad.step(0)
    with pytest.raises(RuntimeError, match="report_execution"):
        ad.step(0.1)
    with pytest.raises(ValueError, match="timestamp"):
        ad.report_execution(0.01, accepted=True)
    shortened = POSE + np.array([0.001, 0, 0, 0, 0, 0])
    ad.report_execution(0, accepted=True, tcp_pose=shortened)
    np.testing.assert_array_equal(ad._last_cmd, shortened)


def test_accepted_targets_do_not_fabricate_measured_motion():
    ad = adapter()
    for t in (0.0, 0.1, 0.2):
        # The controller is accepted, but contact/stall prevents any movement.
        observe(ad, t, pose=POSE)
        command = ad.step(t)
        requested = command.tcp_pose.copy()
        requested[0] += 0.001
        ad.report_execution(t, accepted=True, tcp_pose=requested, gripper_command=0.45)
    assert ad._last_cmd[0] > POSE[0]
    snap = ad.snapshot()
    np.testing.assert_allclose(snap.ur_state[12:18], POSE)
    np.testing.assert_array_equal(snap.prev_chunk[:, :6], 0)


def test_missing_camera_stops_before_any_motion_command():
    ad = adapter()
    observe(ad, retain_rgb=True)
    cmd = tick(ad, 0)
    assert cmd.stopped and cmd.reason == "camera_scene_stale"
    np.testing.assert_array_equal(cmd.tcp_pose, POSE)


def test_adapter_import_does_not_load_torch_or_hardware_drivers():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import phantom.sim.policy_adapter; import phantom.sim.remote_policy; "
                "assert 'torch' not in sys.modules; "
                "assert not any(m.startswith('phantom.drivers') for m in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_runner_gripper_status_distinguishes_motion_contact_and_settled():
    from tools.sim.run_waffles import measured_gripper_status

    assert measured_gripper_status(0.3, 0.3, 0) == 3
    assert measured_gripper_status(0.3, 0.8, 0) == 0
    assert measured_gripper_status(0.5, 0.8, 2) == 2
    assert measured_gripper_status(0.5, 0.2, 2) == 1


def test_runner_tcp_speed_is_world_angular_velocity_with_rotvec_branch_guard():
    from scipy.spatial.transform import Rotation

    from tools.sim.run_waffles import measured_tcp_twist

    first = POSE.copy()
    second = POSE.copy()
    second[4] = -np.pi
    np.testing.assert_allclose(measured_tcp_twist(first, second, 0.008), 0, atol=1e-10)
    second[3:] = (
        Rotation.from_rotvec([0.002, 0, 0]) * Rotation.from_rotvec(first[3:])
    ).as_rotvec()
    np.testing.assert_allclose(
        measured_tcp_twist(first, second, 0.008)[3:], [0.25, 0, 0], atol=1e-10
    )


def test_runner_restores_recorded_envelope_without_reapplying_hitbox_margin():
    from tools.sim.run_waffles import apply_episode_overrides

    hw = adapter().hw
    override = {
        "hitbox_m": {
            "x": [-0.4, -0.2],
            "y": [-0.2, 0],
            "z": [0.1, 0.4],
            "margin_m": 0.05,
        },
        "z_floor_m": 0.15,
        "tcp_speed_m_s": 0.1,
    }
    effective = apply_episode_overrides(hw, override)
    assert effective.safety.hitbox_m.x == (-0.4, -0.2)
    assert effective.safety.workspace_m.z[0] == 0.15
    assert effective.arm.limits.tcp_speed_m_s == 0.1
    assert hw.safety.workspace_m.z[0] == 0.03


def test_policy_audit_preserves_inference_attempts_and_links_confirmed_drive_commands(
    tmp_path,
):
    from tools.sim.run_waffles import PolicyAudit

    audit = PolicyAudit(tmp_path)
    index = audit.begin_replan(0.25)
    assert (
        json.loads((tmp_path / "planner_trace.json").read_text())[0]["status"]
        == "inference_started"
    )
    p = plan(t=0.25, latency=0.4)
    audit.finish_replan(index, p, 0.41)
    audit.executed(
        {
            "t": 0.65,
            "target_q": Q,
            "requested_tcp": POSE,
            "ik_success": True,
            "diagnostics": {"plan_activated": True},
        }
    )
    # The line is visible before close, so a later inference crash cannot erase it.
    command = json.loads((tmp_path / "execution_trace.jsonl").read_text())
    assert command["status"] == "drive_submitted"
    assert command["active_replan_id"] == 0
    np.testing.assert_array_equal(command["target_q"], Q)
    failed = audit.begin_replan(0.8)
    audit.fail_replan(failed, RuntimeError("checkpoint inference failed"))
    audit.close()
    rows = json.loads((tmp_path / "planner_trace.json").read_text())
    assert rows[0]["activated_at"] == 0.65
    assert rows[0]["latency_s"] == 0.4
    assert rows[0]["inference_wall_time_s"] == 0.41
    assert rows[1]["status"] == "inference_error"
    assert rows[1]["error"] == "checkpoint inference failed"


def test_policy_audit_distinguishes_returned_plans_that_never_activated(tmp_path):
    from tools.sim.run_waffles import PolicyAudit

    audit = PolicyAudit(tmp_path)
    index = audit.begin_replan(0)
    audit.finish_replan(index, plan(latency=10), 0.01)
    audit.close()
    row = json.loads((tmp_path / "planner_trace.json").read_text())[0]
    assert row["status"] == "not_activated_before_end"


def test_policy_latency_override_retains_native_inference_timing_in_diagnostics():
    ad = adapter()
    observe(ad)
    p = ad.replan(latency_s=0)
    assert p.latency_s == 0
    assert p.diag["sim_native_inference_latency_s"] == 0.2
    assert p.diag["sim_effective_inference_latency_s"] == 0


def test_joint_speed_and_stale_camera_reuse_real_safety_monitor():
    ad = adapter()
    observe(ad, qd=np.full(6, ad.hw.safety.joint_speed_stop_rad_s + 0.1))
    command = tick(ad, 0)
    assert command.stopped
    assert "joint_speed" in command.diagnostics["safety_events"]
    ad = adapter()
    observe(ad)
    observe(ad, 2, retain_rgb=True)
    with pytest.raises(RuntimeError, match="camera_scene_stale"):
        ad.snapshot()
    command = tick(ad, 2)
    assert command.stopped
    assert "camera_scene_stale" in command.diagnostics["safety_events"]


def test_teacher_requires_explicit_tactile_and_builds_consecutive_frames():
    ad = adapter(mode="teacher")
    observe(ad)
    with pytest.raises(RuntimeError, match="teacher requires explicit tactile"):
        ad.snapshot()
    h, w = ad.hw.recording.field_ds.hw
    kh, kw = ad.hw.recording.keyframe_ds.hw
    ih, iw, _ = ad.hw.tactile.infer_img.hwc
    tac = {
        s.name: {
            "fields_ds": np.zeros((h, w, 8)),
            "keyframe": np.zeros((kh, kw, 8)),
            "infer_img": np.zeros((ih, iw), np.uint8),
            "wrench": np.zeros(6),
            "area": 0.0,
        }
        for s in ad.hw.tactile.sensors
    }
    observe(ad, 0.01, tactile=tac)
    observe(ad, 0.02, tactile=tac)
    snap = ad.snapshot()
    assert snap.fields.shape == (2, kh, kw, 8)
    assert snap.gel.shape == (2, ih, iw)
    assert snap.contact_state.shape == (2, 11)
    assert snap.reactive == 0


def test_gripper_driver_cap_and_explicit_command_history():
    ad = adapter()
    observe(ad)
    ad.submit(plan(grip=3.0), 0)
    command = ad.step(0)
    assert command.gripper == ad.hw.gripper.max_close_cmd
    assert ad.gripper_cmd_at([0]) is None
    ad.report_execution(0, accepted=True, gripper_command=command.gripper)
    np.testing.assert_allclose(ad.gripper_cmd_at([0]), [command.gripper])


def test_episode_reset_clears_state_and_resets_policy():
    ad = adapter()
    observe(ad)
    tick(ad, 0)
    ad.request_stop("manual")
    ad.reset()
    assert ad.policy.resets == 1
    assert ad.stopped_reason is None
    assert ad.gripper_cmd_at([0]) is None
    with pytest.raises(RuntimeError, match="observe"):
        ad.snapshot()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_sensor_and_plan_data_are_refused(bad):
    ad = adapter()
    q = Q.copy()
    q[0] = bad
    with pytest.raises(ValueError, match="finite"):
        observe(ad, q=q)
    observe(ad)
    p = plan()
    p.actions[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        ad.submit(p, 0)


def test_delayed_inference_uses_causal_inputs_and_current_safety_feedback():
    ad = adapter()
    for index in range(41):
        stamp = index / 100
        pose = POSE.copy()
        pose[0] += stamp / 100
        observe(ad, stamp, pose=pose, rgb=np.full((12, 16, 3), index, np.uint8))
    proposed = ad.replan(t=0.4, observation_delay_s=0.16, inference_delay_add_s=0.3)
    snap = ad.policy.calls[-1][0]
    assert snap.t == pytest.approx(0.4)
    assert snap.rgb[0, 0, 0] == 24
    assert snap.ur_state[12] == pytest.approx(POSE[0] + 0.0024)
    assert ad.rings["arm"].latest_ts() == pytest.approx(0.4)
    assert proposed.action_times[0] == pytest.approx(0.6)
    assert ad._pending[0] == pytest.approx(0.9)
    assert proposed.diag["sim_native_inference_latency_s"] == pytest.approx(0.2)
    assert proposed.diag["sim_effective_inference_latency_s"] == pytest.approx(0.5)
    assert ad._last_replan_t == pytest.approx(0.4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation_delay_s": -0.1},
        {"observation_delay_s": float("nan")},
        {"inference_delay_add_s": -0.1},
        {"inference_delay_add_s": float("nan")},
    ],
)
def test_invalid_inference_transport_delays_are_rejected(kwargs):
    ad = adapter()
    observe(ad)
    with pytest.raises(ValueError):
        ad.replan(**kwargs)
    assert not ad.policy.calls
