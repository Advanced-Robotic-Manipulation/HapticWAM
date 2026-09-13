"""Explicit synthetic PhysX pinch/lift/release diagnostic.

This is a controlled initial-contact experiment, not a recorded demonstration,
policy rollout, or validation of real waffle handling. Initialization opens the
two physical jaws, places the free dynamic packet between them once, zeros the
packet's initial velocity, and requests a pinch. After initialization this
module never writes an object pose, changes collision/mass/friction settings,
attaches the packet, or disables gravity. Retention and release must arise from
the existing articulation drives, pad contacts and PhysX solver.

    probe = initialize(packet, robot, finger_ids, qfull, cfg, tool_pos, tool_quat)
    # At each physics/control tick (quaternions above use Isaac wxyz):
    q_target, finger_gap = probe.targets(t, current_or_initial_arm_q)
    desired[arm_ids] = q_target
    desired[finger_ids] = finger_gap
    robot.apply_action(ArticulationAction(joint_positions=desired))

The runner should record actual packet/robot state and pad contact forces for
ten simulation seconds, along with ``probe.metadata`` and ``probe.diagnostics``.
The requested path is a 10 cm world-Z lift, not a claimed achieved lift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from phantom.sim.task_objects import object_config
from scipy.spatial.transform import Rotation

from phantom.sim.kinematics import JOINT_NAMES, forward_pose, inverse_kinematics


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return result.copy()


@dataclass
class ContactProbe:
    initial_q: np.ndarray
    close_gap: float
    release_gap: float
    physics_dt: float
    metadata: dict
    lift_m: float = 0.10
    lift_start_s: float = 1.0
    lift_duration_s: float = 2.0
    release_s: float = 7.0
    joint_speed_rad_s: float = 1.0
    diagnostics: dict = field(default_factory=dict)
    _last_t: float | None = field(default=None, init=False, repr=False)
    _last_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _ik_rejects: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self.initial_q = _vector(self.initial_q, 6, "initial_q")
        self._initial_tcp = forward_pose(self.initial_q)

    def targets(self, t, q0):
        """Return a branch-preserving joint target and each jaw's gap in metres.

        ``q0`` supplies the first IK seed (current or initial measured arm q).
        Subsequent solves use the previous accepted target, as the real UR
        driver's branch guard does. Supplying the original q0 on every call
        therefore does not reintroduce a global/alternate IK branch. A failed
        solve holds the preceding target and is exposed in diagnostics.
        """
        t = float(t)
        seed0 = _vector(q0, 6, "q0")
        if (
            not np.isfinite(t)
            or t < 0
            or (self._last_t is not None and t < self._last_t)
        ):
            raise ValueError("probe time must be finite, nonnegative and monotonic")
        dt = self.physics_dt if self._last_t is None else t - self._last_t
        dt = min(max(dt, 0.0), 2 * self.physics_dt)
        seed = seed0 if self._last_target is None else self._last_target
        u = float(np.clip((t - self.lift_start_s) / self.lift_duration_s, 0, 1))
        fraction = u * u * (3 - 2 * u)
        requested = self._initial_tcp.copy()
        requested[2] += self.lift_m * fraction
        result = inverse_kinematics(requested, seed, max_joint_delta_rad=0.35)
        if result.success:
            delta = np.clip(
                result.q - seed,
                -self.joint_speed_rad_s * dt,
                self.joint_speed_rad_s * dt,
            )
            target = seed + delta
        else:
            target = seed.copy()
            self._ik_rejects += 1
        gap = self.release_gap if t >= self.release_s else self.close_gap
        phase = (
            "release"
            if t >= self.release_s
            else "hold"
            if t >= self.lift_start_s + self.lift_duration_s
            else "lift"
            if t >= self.lift_start_s
            else "pinch"
        )
        self._last_t, self._last_target = t, target.copy()
        target_pose = forward_pose(target)
        self.diagnostics = {
            "t": t,
            "phase": phase,
            "requested_lift_m": self.lift_m * fraction,
            "finger_gap_m": gap,
            "ik_success": result.success,
            "ik_reason": result.reason,
            "ik_position_error_m": result.position_error_m,
            "ik_rotation_error_rad": result.rotation_error_rad,
            "ik_rejects_total": self._ik_rejects,
            "commanded_lift_m": float(target_pose[2] - self._initial_tcp[2]),
            "target_position_error_m": float(
                np.linalg.norm(target_pose[:3] - requested[:3])
            ),
        }
        return target, float(gap)


def initialize(
    packet, robot, finger_ids, qfull, cfg, tool_world_position, tool_world_quaternion
):
    """Set up one synthetic pinch, then return the time-based target generator.

    Tool pose must be the measured world pose of ``tool0`` (not the TCP with
    its 18 cm offset). Quaternion order is Isaac's scalar-first wxyz. The
    packet's local Y/width axis is aligned to the jaws' closing X axis; its
    long axis extends across the pads. Per-jaw prismatic gap is half the clear
    distance between the pad inner faces, as authored in scene.gripper_urdf.
    """
    # Lazy native import: inspecting the schedule needs no running Isaac app.
    from isaacsim.core.utils.types import ArticulationAction

    position = _vector(tool_world_position, 3, "tool_world_position")
    quaternion = _vector(tool_world_quaternion, 4, "tool_world_quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("tool_world_quaternion must represent a rotation")
    tool_rotation = Rotation.from_quat((quaternion / norm)[[1, 2, 3, 0]])
    gripper = cfg["gripper"]
    packet_size = _vector(object_config(cfg)["size"], 3, "packet size")
    if np.any(packet_size <= 0):
        raise ValueError("packet dimensions must be positive")
    width = float(packet_size[1])
    half_stroke = float(gripper["stroke"]) / 2
    initial_gap = width / 2 + 0.002
    close_gap = max(0.0, width / 2 - 0.002)
    if initial_gap > half_stroke:
        raise ValueError(
            "packet width plus initial clearance exceeds the physical jaw stroke"
        )
    names = list(robot.dof_names)
    arm_ids = np.asarray([names.index(name) for name in JOINT_NAMES], dtype=int)
    finger_ids = np.asarray(finger_ids, dtype=int)
    if finger_ids.shape != (2,) or len(set(finger_ids.tolist())) != 2:
        raise ValueError(
            "finger_ids must identify exactly two distinct physical jaw joints"
        )
    if any(index < 0 or index >= len(names) for index in finger_ids):
        raise ValueError("finger index is outside the articulation")
    if set(arm_ids.tolist()) & set(finger_ids.tolist()):
        raise ValueError("finger indices overlap arm joints")
    full = _vector(qfull, len(names), "qfull")
    grip_rotation = tool_rotation * Rotation.from_euler(
        "z", float(gripper.get("yaw", 0))
    )
    center = position + grip_rotation.apply([0, 0, float(gripper["pad_center_z"])])
    packet_rotation = grip_rotation * Rotation.from_euler("z", np.pi / 2)
    packet_xyzw = packet_rotation.as_quat()

    initial = full.copy()
    initial[finger_ids] = initial_gap
    robot.set_joint_positions(initial)
    robot.set_joint_velocities(np.zeros_like(initial))
    # The only object pose/velocity writes in this module are these explicit
    # initial conditions; all subsequent object motion belongs to PhysX.
    packet.set_world_pose(position=center, orientation=packet_xyzw[[3, 0, 1, 2]])
    packet.set_linear_velocity(np.zeros(3))
    packet.set_angular_velocity(np.zeros(3))
    pinch = initial.copy()
    pinch[finger_ids] = close_gap
    robot.apply_action(ArticulationAction(joint_positions=pinch))

    metadata = {
        "kind": "synthetic_physical_contact_probe",
        "recording_validation": False,
        "initialization": "single free-body placement between opened physical pads; no attachment",
        "object_pose_writes_after_initialization": 0,
        "tool_quaternion_order": "wxyz",
        "initial_packet_position": center.tolist(),
        "initial_packet_quaternion_wxyz": packet_xyzw[[3, 0, 1, 2]].tolist(),
        "packet_width_m": width,
        "initial_finger_gap_m": initial_gap,
        "pinch_finger_gap_m": close_gap,
        "release_finger_gap_m": half_stroke,
        "requested_lift_m": 0.10,
        "schedule_s": {
            "pinch": [0, 1],
            "lift": [1, 3],
            "hold": [3, 7],
            "release": [7, 10],
        },
        "physical_parameters": "unchanged scene mass, friction, compliance and force-limited drives",
        "success_source": "runner must evaluate actual packet retention, release and pad contact forces",
    }
    return ContactProbe(
        full[arm_ids], close_gap, half_stroke, float(cfg["physics"]["dt"]), metadata
    )
