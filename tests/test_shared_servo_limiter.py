"""Physical command bounds and native extraction parity; no RTDE/Isaac."""

import json
from pathlib import Path

import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.drivers.real.ur import URArm
from phantom.drivers.servo_limiter import ServoLimits, ServoStep, select_servo_step
from phantom.sim.kinematics import dh_frames, forward_pose, inverse_kinematics

LIMITS = ServoLimits(0.40, 1.0)
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/servo_limiter_native_v1.json").read_text()
)


class CapturedCtrl:
    """Reproduce captured full-UR3 IK responses, checking every queried pose."""

    def __init__(self, case):
        self.remaining = list(case["calls"])
        self.streamed = []

    def isConnected(self):
        return True

    def getInverseKinematics(self, pose, qnear):
        expected = self.remaining.pop(0)
        np.testing.assert_array_equal(pose, expected["pose"])
        np.testing.assert_array_equal(qnear, expected["qref"])
        return expected["q"]

    def servoJ(self, q, *_):
        self.streamed.append(list(q))
        return True


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_extraction_preserves_queries_streamed_joints_and_feedback(case):
    hw = load_hardware("configs/hardware.nuc.mock.yaml", quiet=True)
    if case["enabled"]:
        hw = hw.model_copy(
            update={
                "safety": hw.safety.model_copy(
                    update={"elbow_min_rad": 0.40, "servo_joint_speed_max_rad_s": 1.0}
                )
            }
        )
    arm = URArm(hw)
    arm._ctrl = CapturedCtrl(case)
    arm._last_qsol = case["qref"].copy()
    arm._last_cmd_pose = np.array(case["previous_pose"])
    result = arm.servo_l(np.array(case["target_pose"]), 0.008, 0.1, 300)
    assert not arm._ctrl.remaining
    assert arm._ctrl.streamed == case["streamed"]
    assert result.sent == case["sent"] and result.reason == case["reason"]
    np.testing.assert_array_equal(result.pose, case["reported_pose"])
    assert arm._limiter_hits == case["limiter_hits"]
    assert arm._limiter_holds == case["limiter_holds"]


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["enabled"]])
def test_sim_selector_matches_native_enabled_selection(case):
    ctrl = CapturedCtrl(case)
    result = select_servo_step(
        case["target_pose"],
        case["previous_pose"],
        case["qref"],
        0.008,
        ctrl.getInverseKinematics,
        LIMITS,
    )
    assert not ctrl.remaining
    assert result.accepted == case["sent"] and result.reason == case["reason"]
    assert result.ik_calls <= 7
    if result.accepted:
        np.testing.assert_array_equal(result.q, case["streamed"][-1])
        np.testing.assert_array_equal(result.pose, case["reported_pose"])


def nominal_solver(pose, qref):
    result = inverse_kinematics(pose, qref, max_joint_delta_rad=0.35)
    return (
        result.q.tolist() if result.success or result.reason == "branch_guard" else []
    )


def wrist_distance(q):
    frames = dh_frames(q)
    return np.linalg.norm(frames[4, :3, 3] - frames[1, :3, 3])


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("elbow_pair", [(0.42, 0.415), (0.401, 0.395), (0.30, 0.31)])
def test_full_ur3_command_is_inside_margin_or_bounded_inward_escape(sign, elbow_pair):
    qref = np.array([-0.04, -1.1, sign * elbow_pair[0], 1.11, 1.95, -3.20])
    wanted = qref.copy()
    wanted[2] = sign * elbow_pair[1]
    result = select_servo_step(
        forward_pose(wanted),
        forward_pose(qref),
        qref.tolist(),
        0.008,
        nominal_solver,
        LIMITS,
    )
    assert result.accepted, result
    assert np.max(abs(result.q - qref)) <= 0.008 + 1e-12
    assert np.max(abs(result.q - qref)) <= LIMITS.branch_tolerance_rad
    boundary = qref.copy()
    boundary[2] = sign * LIMITS.elbow_min_rad
    assert wrist_distance(result.q) <= wrist_distance(boundary) + 1e-12 or (
        wrist_distance(result.q) < wrist_distance(qref)
    )
    # Compare against independent complete DH-chain geometry, not the elbow
    # inequality used by the limiter. The final wrist still uses its raw branch.
    assert abs(result.q[-1] - qref[-1]) <= 0.008 + 1e-12
    assert result.q[-1] < -np.pi
    np.testing.assert_allclose(forward_pose(result.q)[:3], result.pose[:3], atol=1e-4)


def test_branch_jump_is_rejected_without_searching_another_branch():
    qref = [0, -1.1, 0.42, 1.11, 1.95, -3.2]
    calls = []

    def branch_jump(pose, seed):
        calls.append(pose)
        q = list(seed)
        q[4] += 2 * np.pi
        return q

    result = select_servo_step(
        forward_pose(qref), forward_pose(qref), qref, 0.008, branch_jump, LIMITS
    )
    assert not result.accepted and result.reason == "ik_branch"
    assert len(calls) == 1


@pytest.mark.parametrize("bad", [[], [0.0] * 3, [float("nan")] * 6])
def test_no_streamable_ik_means_bounded_hold(bad):
    qref = [0, -1.1, 0.42, 1.11, 1.95, -3.2]
    result = select_servo_step(
        forward_pose(qref),
        forward_pose(qref),
        qref,
        0.008,
        lambda _p, _q: bad,
        LIMITS,
    )
    assert not result.accepted and result.reason == "limiter_hold"
    assert result.ik_calls <= 7


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf")])
def test_invalid_clock_never_calls_solver(dt):
    def forbidden(*_):
        pytest.fail("IK must not receive a malformed clock")

    qref = [0, -1.1, 0.42, 1.11, 1.95, -3.2]
    result = select_servo_step(
        forward_pose(qref), forward_pose(qref), qref, dt, forbidden, LIMITS
    )
    assert not result.accepted and result.reason == "invalid_dt"


def test_existing_margin_is_geometric_and_measured_stop_is_not_relaxed():
    q = [0.2, -1.2, 0.4, 0.9, 1.4, -3.2]
    assert wrist_distance(q) == pytest.approx(0.4617110021669664, abs=1e-14)
    assert 0.468 - wrist_distance(q) == pytest.approx(0.0062889978330336, abs=1e-14)


def test_limiter_stall_is_a_controller_stop_preserving_grip_and_feedback():
    from tests.test_sim_policy_adapter import POSE, adapter, observe, plan
    from tools.sim.run_waffles import report_servo_limiter_execution

    ad = adapter()
    rejection = ServoStep(None, None, "limiter_hold", "hold", None, "elbow", 7)
    rejects = 0
    for i in range(25):
        t = i * 0.008
        observe(ad, t, grip=0.55)
        if i == 0:
            # A plan must be playing: without one the executor is still in
            # warm-up and commands nothing, so no limiter rejection can occur.
            assert ad.submit(plan(), t)
        command = ad.step(t)
        assert not command.stopped
        pose, rejects = report_servo_limiter_execution(ad, t, rejection, 0.61, rejects)
        assert pose is None and rejects == i + 1
        if i < 24:
            assert ad.stopped_reason is None
    observe(ad, 0.2, grip=0.55)
    stopped = ad.step(0.2)
    assert stopped.stopped and stopped.reason == "servo_limiter_stall"
    assert stopped.gripper == pytest.approx(0.61)
    np.testing.assert_array_equal(stopped.tcp_pose, POSE)
    assert ad.ik_rejects == 25
    ad.report_execution(
        0.2, accepted=True, tcp_pose=POSE, gripper_command=stopped.gripper
    )


def test_cli_limiter_is_disabled_by_default(monkeypatch):
    from tools.sim.run_waffles import arguments

    monkeypatch.setattr(
        "sys.argv", ["run_waffles", "--episode", "unused", "--output", "unused"]
    )
    assert not arguments().servo_reach_limiter
