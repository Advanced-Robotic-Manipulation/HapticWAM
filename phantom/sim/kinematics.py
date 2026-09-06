"""UR3 CB3 kinematics in the recorded UR controller base frame.

All distances are metres, angles radians, joint order follows RTDE (shoulder
pan/lift, elbow, wrist 1/2/3). A pose is xyz + axis-angle rotation vector, never
Euler angles. The nominal parameters are manufacturer values, not the serial
number's factory calibration. See assets/sim/ur3/PROVENANCE.json.

The official ROS URDF's ``base_link`` is rotated pi about Z relative to the
controller's ``base``. ``link_transforms`` already accounts for that difference.
The DH end frame equals ROS ``tool0``; ROS ``flange`` has a different orientation.
No hardware driver or simulator is imported by this module.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR3_A = np.array([0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0])
UR3_D = np.array([0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819])
UR3_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
DEFAULT_TCP_OFFSET = np.array([0.0, 0.0, 0.18, 0.0, 0.0, 0.0])


def _vector(value, size: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (size,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return a


def pose_to_matrix(pose) -> np.ndarray:
    """Convert UR xyz/rotation-vector pose to homogeneous transformation."""
    p = _vector(pose, 6, "pose")
    result = np.eye(4)
    result[:3, 3] = p[:3]
    result[:3, :3] = Rotation.from_rotvec(p[3:]).as_matrix()
    return result


def matrix_to_pose(matrix) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError("matrix must be a finite (4, 4) transform")
    return np.r_[m[:3, 3], Rotation.from_matrix(m[:3, :3]).as_rotvec()]


def dh_transform(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [c, -s * ca, s * sa, a * c],
            [s, c * ca, -c * sa, a * s],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def dh_frames(q) -> np.ndarray:
    """Return base plus six *DH* frames, shape (7,4,4).

    DH frames are not official visual-mesh origins. For mesh placement use
    ``link_transforms`` with the official URDF.
    """
    q = _vector(q, 6, "q")
    frames = [np.eye(4)]
    for qi, a, d, alpha in zip(q, UR3_A, UR3_D, UR3_ALPHA):
        frames.append(frames[-1] @ dh_transform(qi, a, d, alpha))
    return np.asarray(frames)


def forward_kinematics(q, tcp_offset=DEFAULT_TCP_OFFSET) -> np.ndarray:
    """UR controller base to TCP 4x4 transform (nominal UR3 CB3)."""
    return dh_frames(q)[-1] @ pose_to_matrix(tcp_offset)


def forward_pose(q, tcp_offset=DEFAULT_TCP_OFFSET) -> np.ndarray:
    return matrix_to_pose(forward_kinematics(q, tcp_offset))


def geometric_jacobian(q, tcp_offset=DEFAULT_TCP_OFFSET) -> np.ndarray:
    """6x6 world-expressed Jacobian: translation then angular velocity."""
    frames = dh_frames(q)
    endpoint = (frames[-1] @ pose_to_matrix(tcp_offset))[:3, 3]
    axes = frames[:-1, :3, 2]
    origins = frames[:-1, :3, 3]
    return np.vstack([np.cross(axes, endpoint - origins).T, axes.T])


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    success: bool
    position_error_m: float
    rotation_error_rad: float
    iterations: int
    reason: str


def inverse_kinematics(
    target_pose,
    q_seed,
    tcp_offset=DEFAULT_TCP_OFFSET,
    *,
    position_tolerance_m=1e-4,
    rotation_tolerance_rad=1e-3,
    max_iterations=80,
    max_joint_delta_rad=0.30,
    damping=0.01,
    step_limit_rad=0.15,
) -> IKResult:
    """Damped local IK preserving the current joint branch.

    ``max_joint_delta_rad`` gates the accepted solution relative to the seed;
    this is a branch guard, not a joint-speed limit. Callers apply rate limits
    and workspace guards at their control timestep. Unreachable targets return
    success=False and must not be sent as successful commands.
    """
    target = pose_to_matrix(target_pose)
    seed = _vector(q_seed, 6, "q_seed")
    q = seed.copy()
    for i in range(max_iterations + 1):
        current = forward_kinematics(q, tcp_offset)
        dp = target[:3, 3] - current[:3, 3]
        dr = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
        pe, re = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        if pe <= position_tolerance_m and re <= rotation_tolerance_rad:
            branch_ok = float(np.max(np.abs(q - seed))) <= max_joint_delta_rad
            return IKResult(
                q, branch_ok, pe, re, i, "converged" if branch_ok else "branch_guard"
            )
        if i == max_iterations:
            break
        jac = geometric_jacobian(q, tcp_offset)
        # Scale angle residuals by a characteristic tool length for conditioning.
        weights = np.diag([1.0, 1.0, 1.0, 0.20, 0.20, 0.20])
        j = weights @ jac
        error = weights @ np.r_[dp, dr]
        delta = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(6), error)
        largest = np.max(np.abs(delta))
        if largest > step_limit_rad:
            delta *= step_limit_rad / largest
        q += delta
    return IKResult(q, False, pe, re, max_iterations, "no_convergence")


def _urdf_origin(element) -> np.ndarray:
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return t


def link_transforms(q, urdf_path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Official URDF link transforms expressed in UR controller base frame.

    Useful for camera landmark fitting and offline mesh rendering. Origins of
    visual/collision meshes need their additional URDF <origin> transform.
    """
    q = _vector(q, 6, "q")
    if urdf_path is None:
        urdf_path = Path(__file__).resolve().parents[2] / "assets/sim/ur3/ur3_cb3.urdf"
    root = ET.parse(urdf_path).getroot()
    joints = list(root.findall("joint"))
    children = {j.find("child").get("link") for j in joints}
    roots = {l.get("name") for l in root.findall("link")} - children
    # URDF root world/base_link differs from measured UR base by pi yaw.
    origin = np.eye(4)
    origin[:3, :3] = Rotation.from_euler("z", np.pi).as_matrix()
    transforms = {name: origin.copy() for name in roots}
    positions = dict(zip(JOINT_NAMES, q))
    while joints:
        progressed = False
        for joint in joints[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            t = _urdf_origin(joint.find("origin"))
            name = joint.get("name")
            if joint.get("type") in ("revolute", "continuous"):
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                rot = np.eye(4)
                rot[:3, :3] = Rotation.from_rotvec(
                    axis * positions.get(name, 0.0)
                ).as_matrix()
                t = t @ rot
            transforms[joint.find("child").get("link")] = transforms[parent] @ t
            joints.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("URDF contains an unresolved joint tree")
    return transforms


def pose_errors(predicted, recorded) -> tuple[np.ndarray, np.ndarray]:
    """Per-row Euclidean translation and SO(3) geodesic rotation errors."""
    p, r = np.asarray(predicted, dtype=float), np.asarray(recorded, dtype=float)
    if p.shape != r.shape or p.ndim != 2 or p.shape[1] != 6:
        raise ValueError("poses must have equal (N,6) shapes")
    position = np.linalg.norm(p[:, :3] - r[:, :3], axis=1)
    orientation = (
        Rotation.from_rotvec(p[:, 3:]) * Rotation.from_rotvec(r[:, 3:]).inv()
    ).magnitude()
    return position, orientation
