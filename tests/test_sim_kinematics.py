"""Physics-independent checks of UR3 frames, local IK and real RTDE pairs."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from phantom.sim.kinematics import (
    DEFAULT_TCP_OFFSET,
    dh_frames,
    forward_kinematics,
    forward_pose,
    geometric_jacobian,
    inverse_kinematics,
    link_transforms,
    matrix_to_pose,
    pose_errors,
    pose_to_matrix,
)

Q = np.array([0.2347, -1.5198, 1.0754, 0.7233, 1.1747, -3.0047])


def test_pose_rotvec_roundtrip_at_pi_and_zero():
    for r in [np.zeros(3), [np.pi, 0, 0], [0.4, -0.2, 3.3]]:
        transform = pose_to_matrix(np.r_[[0.1, -0.2, 0.3], r])
        np.testing.assert_allclose(
            pose_to_matrix(matrix_to_pose(transform)), transform, atol=1e-12
        )


def test_nominal_dh_zero_configuration_matches_manufacturer_dimensions():
    tool0 = forward_kinematics(np.zeros(6), np.zeros(6))
    np.testing.assert_allclose(tool0[:3, 3], [-0.4569, -0.19425, 0.06655], atol=1e-12)
    np.testing.assert_allclose(dh_frames(Q)[1, :3, 3], [0, 0, 0.1519], atol=1e-12)


def test_official_urdf_matches_dh_and_measured_base_convention():
    for q in [Q, np.zeros(6), [1, -0.9, 2, -1.5, -0.3, -5.7]]:
        links = link_transforms(q)
        np.testing.assert_allclose(links["base"], np.eye(4), atol=1e-9)
        np.testing.assert_allclose(
            links["tool0"], forward_kinematics(q, np.zeros(6)), atol=1e-9
        )
        np.testing.assert_allclose(
            links["base_link"][:3, :3], np.diag([-1, -1, 1]), atol=1e-9
        )


def test_tcp_offset_is_applied_in_tool_frame():
    flange = forward_kinematics(Q, np.zeros(6))
    tcp = forward_kinematics(Q)
    np.testing.assert_allclose(
        tcp[:3, 3] - flange[:3, 3], flange[:3, :3] @ DEFAULT_TCP_OFFSET[:3], atol=1e-12
    )


def test_jacobian_matches_finite_difference_world_twist():
    eps = 1e-7
    base = forward_kinematics(Q)
    measured = np.empty((6, 6))
    for i in range(6):
        plus = Q.copy()
        plus[i] += eps
        t = forward_kinematics(plus)
        measured[:3, i] = (t[:3, 3] - base[:3, 3]) / eps
        measured[3:, i] = (
            Rotation.from_matrix(t[:3, :3] @ base[:3, :3].T).as_rotvec() / eps
        )
    np.testing.assert_allclose(geometric_jacobian(Q), measured, atol=1e-7)


def test_local_ik_converges_without_joint_wrapping():
    seed = Q.copy()
    seed[-1] -= 2 * np.pi
    target_q = seed + [0.008, -0.01, 0.02, -0.015, 0.01, -0.01]
    result = inverse_kinematics(forward_pose(target_q), seed)
    assert result.success, result
    assert result.position_error_m < 1e-4
    assert result.rotation_error_rad < 1e-3
    assert abs(result.q[-1] - seed[-1]) < 0.1


def test_local_ik_rejects_unreachable_and_branch_change():
    result = inverse_kinematics([3, 3, 3, 0, 0, 0], Q, max_iterations=25)
    assert not result.success
    target = forward_pose(Q + [0.08, 0, 0, 0, 0, 0])
    result = inverse_kinematics(target, Q, max_joint_delta_rad=0.005)
    assert not result.success
    assert result.reason == "branch_guard"


def test_recorded_ur3_feedback_validates_units_order_and_tcp():
    data = json.loads(
        (Path(__file__).parent / "fixtures/sim/ur3_recorded_pairs.json").read_text()
    )
    records = data["records"]
    assert len({r["episode"] for r in records}) == 48
    pred = np.array([forward_pose(r["q"]) for r in records])
    p, o = pose_errors(pred, np.array([r["tcp"] for r in records]))
    # Factory calibration is absent: finite millimetre-scale nominal mismatch
    # is expected. A frame/order/variant error is centimetres or much larger.
    assert np.sqrt(np.mean(p * p)) < 0.002
    assert p.max() < 0.003
    assert np.rad2deg(np.sqrt(np.mean(o * o))) < 0.3


@pytest.mark.parametrize("q", [[0] * 5, [0, 0, 0, 0, 0, float("nan")]])
def test_invalid_joint_vectors_rejected(q):
    with pytest.raises(ValueError):
        forward_kinematics(q)
