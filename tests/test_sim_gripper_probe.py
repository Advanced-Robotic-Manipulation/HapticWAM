"""CPU geometry and trajectory checks; no Isaac process or simulated success."""

import copy
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_sim_gripper_articulation import candidate  # noqa: F401 -- shared CAD fixture
from phantom.sim import contact_probe, gripper_probe
from phantom.sim.gripper_articulation import (
    finger_joint_names, joint_targets, link_transforms_from_joint_positions,
)
from phantom.sim.kinematics import JOINT_NAMES, forward_pose


# Fixed measured start of the developmental 35 mm probe. The higher dataset
# mean start does not have 10 cm of fixed-orientation vertical reach clearance.
START_Q = np.array([.16796258, -1.5094893, 1.4485282, .5152859, 1.318228, -3.151522])


@pytest.mark.parametrize('width', [.027214, .035, .042, .05])
def test_width_inversion_is_symmetric_about_both_actual_inner_faces(candidate, width):
    repo, cfg = candidate
    closure = gripper_probe.closure_for_gap(repo, cfg, width)
    gap, center, rotation = gripper_probe.jaw_geometry(repo, cfg, closure)
    frames = link_transforms_from_joint_positions(repo, cfg, joint_targets(closure, cfg))
    left, right = frames['left_pad'], frames['right_pad']
    thickness = cfg['gripper']['gel_geometry']['pad_thickness_m']
    left_face = left[:3, 3] - left[:3, 0] * thickness / 2
    right_face = right[:3, 3] + right[:3, 0] * thickness / 2
    # Both inner faces are on either side of the same center, with no hidden
    # off-axis/shear term that a scalar aperture alone would miss.
    np.testing.assert_allclose(rotation.T @ (left_face-center), [width/2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(rotation.T @ (right_face-center), [-width/2, 0, 0], atol=1e-12)
    assert gap == pytest.approx(width, abs=1e-12)
    assert gripper_probe.jaw_geometry(repo, cfg, closure-.01)[0] > width
    assert gripper_probe.jaw_geometry(repo, cfg, closure+.01)[0] < width


def test_closed_limit_keeps_parallel_symmetric_frames_and_reports_overlap(candidate):
    repo, cfg = candidate
    zero, end = 0., cfg['gripper']['pad_touch_command']
    initial = gripper_probe.jaw_geometry(repo, cfg, zero)
    final = gripper_probe.jaw_geometry(repo, cfg, end)
    assert initial[0] == pytest.approx(.078)
    # This uncalibrated fixture overlaps at the nominal motor limit; do not
    # describe this command as a measured zero-aperture physical calibration.
    assert final[0] < 0
    np.testing.assert_allclose(initial[2], final[2], atol=1e-12)
    np.testing.assert_allclose(initial[1][:2], final[1][:2], atol=1e-12)
    assert final[1][2] > initial[1][2]


@pytest.mark.parametrize('width', [0, -.001, float('nan'), float('inf'), .079])
def test_impossible_widths_fail_instead_of_saturating(candidate, width):
    repo, cfg = candidate
    with pytest.raises(ValueError):
        gripper_probe.closure_for_gap(repo, cfg, width)


def test_nonparallel_jaws_cannot_be_hidden_in_projected_gap(candidate):
    repo, cfg = candidate
    cfg['gripper']['articulation']['mounts']['right']['rpy'][1] = .1
    with pytest.raises(ValueError, match='parallel mounted gel faces'):
        gripper_probe.closure_for_gap(repo, cfg, .035)


class Robot:
    def __init__(self, cfg):
        # Interleave fingers and arm joints to catch any positional assumptions.
        self.dof_names = [name for pair in zip(finger_joint_names(cfg), JOINT_NAMES) for name in pair]
        self.positions = []; self.velocities = []; self.actions = []

    def set_joint_positions(self, value): self.positions.append(value.copy())
    def set_joint_velocities(self, value): self.velocities.append(value.copy())
    def apply_action(self, action): self.actions.append(action.joint_positions.copy())


class Packet:
    def __init__(self): self.poses = []; self.linear = []; self.angular = []
    def set_world_pose(self, **value): self.poses.append(copy.deepcopy(value))
    def set_linear_velocity(self, value): self.linear.append(value.copy())
    def set_angular_velocity(self, value): self.angular.append(value.copy())


def initialize_cpu(candidate, monkeypatch):
    repo, cfg = candidate
    cfg['waffle'] = {'size': [.170, .035, .090]}
    cfg['physics'] = {'dt': .02}
    # Replace only the lazy native action value object. Actual CAD FK, aperture
    # inversion, initialization transforms and UR3 path integration remain real.
    action_module = types.ModuleType('isaacsim.core.utils.types')
    action_module.ArticulationAction = lambda **values: types.SimpleNamespace(**values)
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.types', action_module)
    monkeypatch.setattr(gripper_probe, '__file__', str(repo/'phantom/sim/gripper_probe.py'))
    robot = Robot(cfg); packet = Packet()
    fingers = np.array([robot.dof_names.index(n) for n in finger_joint_names(cfg)])
    arm_ids = np.array([robot.dof_names.index(n) for n in JOINT_NAMES])
    full = np.zeros(12); full[arm_ids] = START_Q
    tool_position = np.array([.2, -.3, .4])
    tool_rotation = Rotation.from_rotvec([.4, -.5, .2])
    tool_quat = tool_rotation.as_quat()[[3, 0, 1, 2]]
    probe = gripper_probe.initialize(packet, robot, fingers, full, cfg, tool_position, tool_quat)
    return probe, packet, robot, fingers, arm_ids, full, tool_position, tool_rotation


def test_initialization_places_free_packet_once_and_preserves_arm_dof_order(candidate, monkeypatch):
    probe, packet, robot, fingers, arm_ids, full, position, tool_rotation = initialize_cpu(candidate, monkeypatch)
    repo, cfg = candidate
    assert len(packet.poses) == len(packet.linear) == len(packet.angular) == 1
    np.testing.assert_array_equal(full[arm_ids], START_Q)
    np.testing.assert_array_equal(robot.positions[0][arm_ids], START_Q)
    np.testing.assert_array_equal(robot.actions[0][arm_ids], START_Q)
    np.testing.assert_array_equal(probe.arm_path.initial_q, START_Q)
    np.testing.assert_array_equal(robot.velocities[0], np.zeros(12))
    np.testing.assert_array_equal(packet.linear[0], np.zeros(3))
    np.testing.assert_array_equal(packet.angular[0], np.zeros(3))
    first = probe.metadata['initial_motor_command']
    gap, center, pad_rotation = gripper_probe.jaw_geometry(repo, cfg, first)
    assert gap == pytest.approx(.0354, abs=1e-12)
    assert gripper_probe.jaw_geometry(repo, cfg, probe.close_command)[0] == pytest.approx(.033, abs=1e-12)
    np.testing.assert_allclose(robot.positions[0][fingers], joint_targets(first, cfg))
    np.testing.assert_allclose(robot.actions[0][fingers], joint_targets(probe.close_command, cfg))
    np.testing.assert_allclose(packet.poses[0]['position'], position+tool_rotation.apply(center))
    orientation = Rotation.from_quat(packet.poses[0]['orientation'][[1, 2, 3, 0]])
    # Packet local width Y must align (up to sign) with the jaw closing X.
    assert abs(np.dot(orientation.apply([0, 1, 0]), tool_rotation.apply(pad_rotation[:, 0]))) == pytest.approx(1)
    assert probe.metadata['recording_validation'] is False
    assert probe.metadata['object_pose_writes_after_initialization'] == 0


def test_actual_cpu_ur3_path_lifts_holds_and_opens_without_further_object_writes(candidate, monkeypatch):
    probe, packet, robot, *_ = initialize_cpu(candidate, monkeypatch)
    cfg = candidate[1]; records = []
    for t in np.linspace(0, 10, 501):
        arm, fingers = probe.targets(t, START_Q)
        records.append((float(t), arm, fingers, dict(probe.diagnostics)))
        expected_closure = probe.close_command if t < 7 else 0.
        np.testing.assert_allclose(fingers, joint_targets(expected_closure, cfg))
        assert 'finger_gap_m' not in probe.diagnostics
        assert probe.diagnostics['finger_joint_units'] == 'radians'
    poses = np.array([forward_pose(record[1]) for record in records])
    initial = forward_pose(START_Q)
    assert np.linalg.norm(poses[-1, :3]-(initial[:3]+[0, 0, .1])) < .0001
    assert (Rotation.from_rotvec(initial[3:]).inv()*Rotation.from_rotvec(poses[-1, 3:])).magnitude() < .001
    assert max(record[3]['ik_rejects_total'] for record in records) == 0
    assert np.max(abs(np.diff(np.array([record[1] for record in records]), axis=0))) <= .02+1e-12
    assert [records[i][3]['phase'] for i in [0, 50, 150, 350]] == ['pinch', 'lift', 'hold', 'release']
    # Only one initialization pose/velocity write and one initial drive request;
    # subsequent targets never obtain access to or manipulate the free packet.
    assert len(packet.poses) == len(packet.linear) == len(packet.angular) == 1
    assert len(robot.positions) == len(robot.actions) == 1


def test_failed_arm_ik_holds_last_accepted_target_but_gripper_still_releases(candidate, monkeypatch):
    probe, *_ = initialize_cpu(candidate, monkeypatch)
    first, _ = probe.targets(0, START_Q)
    attempted = []
    def failed_ik(requested, seed, **kwargs):
        attempted.append(seed.copy())
        return types.SimpleNamespace(success=False, q=seed+100, reason='fixture_reach_failure', position_error_m=.1, rotation_error_rad=.2)
    monkeypatch.setattr(contact_probe, 'inverse_kinematics', failed_ik)
    target, fingers = probe.targets(7, START_Q+1.)
    np.testing.assert_array_equal(attempted[0], first)
    np.testing.assert_array_equal(target, first)
    np.testing.assert_array_equal(fingers, joint_targets(0, candidate[1]))
    assert probe.diagnostics['ik_rejects_total'] == 1
    assert probe.diagnostics['ik_reason'] == 'fixture_reach_failure'
    assert probe.diagnostics['phase'] == 'release'
    with pytest.raises(ValueError, match='monotonic'):
        probe.targets(6, START_Q)


def test_scene_urdf_candidate_dispatch_retains_sensor_parts_and_excludes_legacy_sliders(candidate, tmp_path, monkeypatch):
    from phantom.sim import scene
    repo, cfg = candidate
    monkeypatch.setattr(scene, '__file__', str(repo/'phantom/sim/scene.py'))
    cfg['gripper']['articulation']['geometry_manifest'] = str(repo/'assets/sim/dmtac_w2l/geometry.json')
    source = Path(__file__).resolve().parents[1]/'assets/sim/ur3/ur3_cb3.urdf'
    original = source.read_bytes(); destination = tmp_path/'candidate.urdf'
    scene.gripper_urdf(source, destination, cfg)
    assert source.read_bytes() == original
    root = ET.parse(destination).getroot()
    assert not root.findall("joint[@type='prismatic']")
    for name in finger_joint_names(cfg):
        assert root.find(f"joint[@name='{name}']").get('type') == 'revolute'
    for side in ['left', 'right']:
        assert root.find(f"link[@name='{side}_sensor_housing']") is not None
        gel = root.find(f"link[@name='{side}_pad']")
        assert len(gel.findall('collision')) == 1
        assert 'gel.stl' in gel.find('collision/geometry/mesh').get('filename')
