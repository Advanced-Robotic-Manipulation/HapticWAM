"""Synthetic width/pinch/lift/release check for the coupled W2L candidate.

The free packet is placed once between the gel faces. This deliberately isolates
the mechanical gripper from reach/camera/reset errors; it is not a real replay
or evidence that a policy can acquire an object from the table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from phantom.sim.contact_probe import ContactProbe
from phantom.sim.gripper_articulation import (
    is_articulated, is_adaptive, joint_targets, drive_targets,
    link_transforms_from_joint_positions, load_sensor_geometry,
)
from phantom.sim.kinematics import JOINT_NAMES


def jaw_geometry(repo, cfg, closure):
    frames = link_transforms_from_joint_positions(repo, cfg, joint_targets(closure, cfg))
    left, right = frames['left_pad'], frames['right_pad']
    thickness = float(cfg['gripper']['gel_geometry']['pad_thickness_m'])
    lf = left[:3, 3] + left[:3, :3] @ [-thickness/2, 0, 0]
    rf = right[:3, 3] + right[:3, :3] @ [thickness/2, 0, 0]
    if is_adaptive(cfg):
        geometry = load_sensor_geometry(repo, cfg)
        gel = next(p for p in geometry['parts'] if p['role'] == 'gel')
        return native_probe_aperture(left, right, lf, rf,
                                     [gel['bounding_box_m'][key][2] for key in ('min', 'max')])
    axis = left[:3, 0]
    if np.dot(axis, right[:3, 0]) < .999:
        raise ValueError('Synthetic width probe requires parallel mounted gel faces')
    return float(np.dot(lf-rf, axis)), (lf+rf)/2, left[:3, :3]


def native_probe_aperture(left, right, lf, rf, longitudinal_bounds):
    """Conservative plane aperture for a centred synthetic native-jaw probe.

    Use the minimum closing-axis gap over the common full-gel longitudinal
    interval. Curved gel edges are bounded by the extended front plane; this
    is a synthetic initialization envelope, not finite-body clearance. The
    legacy parallel-jaw probe and its rejection guard remain unchanged.
    """
    if np.dot(left[:3, 1], right[:3, 1]) < 1-1e-8:
        raise ValueError('Native synthetic probe requires aligned sensor width axes')
    x = left[:3, 0]+right[:3, 0]
    z = left[:3, 2]+right[:3, 2]
    if np.linalg.norm(x) < 1e-8 or np.linalg.norm(z) < 1e-8:
        raise ValueError('Degenerate native jaw bisector')
    x /= np.linalg.norm(x)
    z -= x*np.dot(x, z)
    z /= np.linalg.norm(z)
    rotation = np.column_stack((x, np.cross(z, x), z))
    center = (lf+rf)/2
    bounds = np.asarray(longitudinal_bounds, float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError('Increasing finite gel longitudinal bounds are required')
    intervals = [sorted(float(np.dot(face+frame[:3, 2]*depth-center, z)) for depth in bounds)
                 for face, frame in ((lf, left), (rf, right))]
    lo, hi = max(v[0] for v in intervals), min(v[1] for v in intervals)
    if lo >= hi:
        raise ValueError('No common native gel longitudinal interval')
    gaps = []
    for depth in (lo, hi):
        origin = center+depth*z
        intersections = []
        for face, frame in ((lf, left), (rf, right)):
            denominator = np.dot(frame[:3, 0], x)
            if denominator <= 1e-8:
                raise ValueError('Native front plane does not face the common closing axis')
            intersections.append(np.dot(frame[:3, 0], face-origin)/denominator)
        gaps.append(intersections[0]-intersections[1])
    return float(min(gaps)), center, rotation


def closure_for_gap(repo, cfg, width):
    width = float(width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError('Width must be finite and positive')
    lo, hi = 0., float(cfg['gripper']['pad_touch_command'])
    open_gap = jaw_geometry(repo, cfg, lo)[0]
    close_gap = jaw_geometry(repo, cfg, hi)[0]
    if not close_gap <= width <= open_gap:
        raise ValueError(f'Width {width:g} m outside modeled gap [{close_gap:g}, {open_gap:g}]')
    for _ in range(40):
        middle = (lo+hi)/2
        if jaw_geometry(repo, cfg, middle)[0] > width:
            lo = middle
        else:
            hi = middle
    return (lo+hi)/2


@dataclass
class ArticulatedContactProbe:
    arm_path: ContactProbe
    cfg: dict
    close_command: float
    metadata: dict
    diagnostics: dict = field(default_factory=dict)

    def targets(self, t, q0):
        q, _ = self.arm_path.targets(t, q0)
        closure = 0. if t >= self.arm_path.release_s else self.close_command
        self.diagnostics = dict(self.arm_path.diagnostics)
        self.diagnostics.pop('finger_gap_m', None)
        self.diagnostics.update(normalized_motor_target=closure, finger_joint_units='radians')
        return q, drive_targets(closure, self.cfg)


def initialize(packet, robot, finger_ids, qfull, cfg, tool_world_position, tool_world_quaternion):
    from isaacsim.core.utils.types import ArticulationAction

    if not is_articulated(cfg):
        raise ValueError('This probe requires the articulated W2L configuration')
    repo = Path(__file__).resolve().parents[2]
    width = float(cfg['waffle']['size'][1])
    # A small initial clearance limits free fall before the force-limited
    # closure. The target overtravel is a declared synthetic loading condition.
    initial_command = closure_for_gap(repo, cfg, width + .0004)
    close_command = closure_for_gap(repo, cfg, width - .002)
    actual_gap, center_tool, pad_rotation = jaw_geometry(repo, cfg, initial_command)
    quat = np.asarray(tool_world_quaternion, dtype=float)
    tool_rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
    center = np.asarray(tool_world_position) + tool_rotation.apply(center_tool)
    rotation = tool_rotation * Rotation.from_matrix(pad_rotation) * Rotation.from_euler('z', np.pi/2)
    packet_quat = rotation.as_quat()[[3, 0, 1, 2]]
    full = np.asarray(qfull, dtype=float).copy()
    full[finger_ids] = joint_targets(initial_command, cfg)
    robot.set_joint_positions(full)
    robot.set_joint_velocities(np.zeros_like(full))
    packet.set_world_pose(position=center, orientation=packet_quat)
    packet.set_linear_velocity(np.zeros(3))
    packet.set_angular_velocity(np.zeros(3))
    target = full.copy()
    target[finger_ids] = drive_targets(close_command, cfg)
    robot.apply_action(ArticulationAction(joint_positions=target))
    arm_ids = [list(robot.dof_names).index(n) for n in JOINT_NAMES]
    metadata = {
        'kind': 'synthetic_articulated_width_pinch_lift_release',
        'recording_validation': False,
        'initialization': 'One placement of a free dynamic packet between modeled gel faces; no attachment or gravity change',
        'object_pose_writes_after_initialization': 0,
        'packet_width_m': width,
        'initial_pad_gap_m': actual_gap,
        'gap_definition': ('Minimum opposing front-plane aperture over common full-gel longitudinal overlap; conservative curved-edge envelope'
                           if is_adaptive(cfg) else 'Parallel opposing gel front-plane separation'),
        'target_unloaded_pad_gap_m': width-.002,
        'initial_motor_command': initial_command,
        'close_motor_command': close_command,
        'initial_packet_position': center.tolist(),
        'initial_packet_quaternion_wxyz': packet_quat.tolist(),
        'requested_lift_m': .1,
        'schedule_s': {'pinch':[0,1], 'lift':[1,3], 'hold':[3,7], 'release':[7,10]},
        'success_source': 'Actual free object lift/retention/release and measured gel contact; never the requested trajectory',
        'physical_parameters': 'Unchanged candidate mass, force limits, friction and compliance',
    }
    arm = ContactProbe(full[arm_ids], 0., 0., float(cfg['physics']['dt']), metadata)
    return ArticulatedContactProbe(arm, cfg, close_command, metadata)
