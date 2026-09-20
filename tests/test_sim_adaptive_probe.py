"""Analytic tilted-plane aperture and one-time native probe initialization."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from phantom.sim import gripper_adaptive as adaptive
from phantom.sim import gripper_probe as probe
from phantom.sim.kinematics import JOINT_NAMES as ARM_JOINTS

REPO = Path(__file__).resolve().parents[1]


def symmetric_faces(gap, angle):
    left, right = np.eye(4), np.eye(4)
    left[:3, :3] = Rotation.from_euler('y', angle).as_matrix()
    right[:3, :3] = Rotation.from_euler('y', -angle).as_matrix()
    left[0, 3], right[0, 3] = gap / 2, -gap / 2
    return left, right, left[:3, 3].copy(), right[:3, 3].copy()


@pytest.mark.parametrize('degrees', [0., 2., -7., 25.])
def test_symmetric_splay_matches_exact_plane_envelope_and_rigid_transform(degrees):
    gap, half_length = .085, .024
    angle = math.radians(degrees)
    left, right, lf, rf = symmetric_faces(gap, angle)
    expected = gap - 2 * half_length * abs(math.sin(angle))
    width, center, rotation = probe.native_probe_aperture(left, right, lf, rf, [-half_length, half_length])
    assert width == pytest.approx(expected, abs=1e-14)
    np.testing.assert_allclose(center, 0., atol=1e-14)
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-14)
    # Independent plane equation x_L=g/2+tan(a)z, x_R=-g/2-tan(a)z;
    # common projected Z support is +/-half_length*cos(a).
    z = np.array([-half_length, half_length]) * math.cos(angle)
    assert np.min(gap + 2 * math.tan(angle) * z) == pytest.approx(expected, abs=1e-14)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler('xyz', [.6, -.4, 1.2]).as_matrix()
    world[:3, 3] = [.18, -.27, .44]
    wl, wr = world @ left, world @ right
    transformed = probe.native_probe_aperture(wl, wr, wl[:3, 3], wr[:3, 3], [-half_length, half_length])
    assert transformed[0] == pytest.approx(expected, abs=1e-14)
    np.testing.assert_allclose(transformed[1], world[:3, 3], atol=1e-14)
    np.testing.assert_allclose(transformed[2], world[:3, :3], atol=1e-14)


def test_splayed_aperture_bisection_matches_analytic_inverse(monkeypatch):
    half_length, angle, open_center_gap, travel, touch = .024, .12, .100, .08, .9

    def tilted_geometry(_repo, _cfg, closure):
        frames = symmetric_faces(open_center_gap - travel * closure / touch, angle)
        return probe.native_probe_aperture(*frames, [-half_length, half_length])

    monkeypatch.setattr(probe, 'jaw_geometry', tilted_geometry)
    cfg = {'gripper': {'pad_touch_command': touch}}
    tilt_reduction = 2 * half_length * math.sin(angle)
    for width in [.0272, .035, .042, .05, .085]:
        expected = (open_center_gap - tilt_reduction - width) * touch / travel
        actual = probe.closure_for_gap(REPO, cfg, width)
        assert actual == pytest.approx(expected, abs=1e-12)
        assert tilted_geometry(REPO, cfg, actual)[0] == pytest.approx(width, abs=1e-12)
    for bad in [.100, .005, 0., float('nan')]:
        with pytest.raises(ValueError):
            probe.closure_for_gap(REPO, cfg, bad)


def test_twist_degenerate_axes_bad_bounds_and_disjoint_support_are_rejected():
    left, right, lf, rf = symmetric_faces(.085, .1)
    twisted = right.copy()
    twisted[:3, :3] = right[:3, :3] @ Rotation.from_euler('z', .01).as_matrix()
    with pytest.raises(ValueError, match='aligned sensor width axes'):
        probe.native_probe_aperture(left, twisted, lf, rf, [-.024, .024])
    opposite = left.copy()
    opposite[:3, :3] = left[:3, :3] @ Rotation.from_euler('y', math.pi).as_matrix()
    with pytest.raises(ValueError, match='Degenerate'):
        probe.native_probe_aperture(left, opposite, lf, rf, [-.024, .024])
    for bounds in [[0., 0.], [.02, -.02], [float('nan'), .02], [0.]]:
        with pytest.raises(ValueError, match='Increasing finite'):
            probe.native_probe_aperture(left, right, lf, rf, bounds)
    with pytest.raises(ValueError, match='No common'):
        probe.native_probe_aperture(left, right, lf + [0, 0, -.1], rf + [0, 0, .1], [-.024, .024])


@pytest.mark.parametrize('configuration', ['waffles_w2l_adaptive_parallel_20260909.json', 'waffles_w2l_adaptive_splayed_20260909.json'])
def test_real_native_assets_have_monotonic_widths_and_valid_inverse(configuration):
    cfg = json.loads((REPO / 'configs/sim' / configuration).read_text())
    assert cfg['gripper']['model'] == adaptive.MODEL
    commands = np.linspace(0, cfg['gripper']['pad_touch_command'], 31)
    gaps = [probe.jaw_geometry(REPO, cfg, float(c))[0] for c in commands]
    assert np.all(np.diff(gaps) < 0)
    for width in [.0272, .035, .042, .05]:
        command = probe.closure_for_gap(REPO, cfg, width)
        actual, _, rotation = probe.jaw_geometry(REPO, cfg, command)
        assert actual == pytest.approx(width, abs=1e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        assert np.linalg.det(rotation) == pytest.approx(1.)


def test_initialize_places_once_and_never_uses_spring_rest_as_measured_pose(monkeypatch):
    # Stub only the Isaac action container; use the actual CAD/FK, inversion,
    # probe schedule and native drive/initialization functions.
    for name in ['isaacsim', 'isaacsim.core', 'isaacsim.core.utils', 'isaacsim.core.utils.types']:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['isaacsim.core.utils.types'].ArticulationAction = SimpleNamespace

    class Robot:
        dof_names = (*ARM_JOINTS, *adaptive.JOINT_NAMES)

        def __init__(self):
            self.positions, self.velocities, self.actions = [], [], []

        def set_joint_positions(self, q):
            self.positions.append(np.asarray(q).copy())

        def set_joint_velocities(self, q):
            self.velocities.append(np.asarray(q).copy())

        def apply_action(self, action):
            self.actions.append(action)

    class Packet:
        def __init__(self):
            self.poses, self.linear, self.angular = [], [], []

        def set_world_pose(self, **pose):
            self.poses.append(pose)

        def set_linear_velocity(self, v):
            self.linear.append(v)

        def set_angular_velocity(self, v):
            self.angular.append(v)

    cfg = json.loads((REPO / 'configs/sim/waffles_w2l_adaptive_parallel_20260909.json').read_text())
    inputs = json.loads((REPO / 'tests/fixtures/reference/teacher_scene_v10_20260908/pickup_registration_evidence/inputs.json').read_text())
    arm_q = np.asarray(inputs['frames'][0]['arm_q'])
    qfull = np.r_[arm_q, np.zeros(8)]
    unchanged = qfull.copy()
    finger_ids = np.arange(6, 14)
    robot, packet = Robot(), Packet()
    diagnostic = probe.initialize(packet, robot, finger_ids, qfull, cfg, [-.35, -.25, .3], [1., 0., 0., 0.])
    meta = diagnostic.metadata
    seed = adaptive.joint_targets(meta['initial_motor_command'], cfg)
    np.testing.assert_allclose(robot.positions[0][finger_ids], seed, atol=1e-14)
    np.testing.assert_allclose(robot.actions[0].joint_positions[finger_ids], adaptive.drive_targets(meta['close_motor_command'], cfg), atol=1e-14)
    assert robot.actions[0].joint_positions[finger_ids][1] == pytest.approx(2.62)
    assert robot.positions[0][finger_ids][1] < .8
    assert meta['initial_pad_gap_m'] == pytest.approx(cfg['waffle']['size'][1] + .0004, abs=1e-12)
    assert not meta['recording_validation']
    assert meta['object_pose_writes_after_initialization'] == 0
    for t in [0., .004, 7.]:
        _, targets = diagnostic.targets(t, arm_q)
        closure = 0. if t >= 7 else meta['close_motor_command']
        np.testing.assert_allclose(targets, adaptive.drive_targets(closure, cfg), atol=1e-14)
    assert len(robot.positions) == len(robot.velocities) == len(robot.actions) == 1
    assert len(packet.poses) == len(packet.linear) == len(packet.angular) == 1
    np.testing.assert_array_equal(qfull, unchanged)
