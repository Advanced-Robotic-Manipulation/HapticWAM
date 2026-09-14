"""Native closed-loop geometry and CPU USD force authoring; no Isaac launch."""
import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from phantom.sim import gripper_adaptive as native

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    value = json.loads((REPO/'configs/sim/waffles_w2l_articulated_provisional_20260908.json').read_text())
    value['gripper']['model'] = native.MODEL
    return value


def build(cfg):
    root = ET.Element('robot',name='native_test')
    ET.SubElement(root,'link',name='tool0')
    metadata = native.append_gripper_urdf(root,REPO,cfg)
    return root,metadata


def origin(e):
    matrix = np.eye(4)
    if e is not None:
        matrix[:3,3] = np.fromstring(e.get('xyz','0 0 0'),sep=' ')
        matrix[:3,:3] = Rotation.from_euler('xyz',np.fromstring(e.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return matrix


def fk(root,q):
    angles = dict(zip(native.JOINT_NAMES,q))
    poses = {'tool0':np.eye(4)}
    pending = list(root.findall('joint'))
    while pending:
        old_count = len(pending)
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in poses:
                continue
            relative = origin(joint.find('origin'))
            if joint.get('type') == 'revolute':
                rotation = np.eye(4)
                axis = np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                rotation[:3,:3] = Rotation.from_rotvec(axis*angles[joint.get('name')]).as_matrix()
                relative = relative@rotation
            poses[joint.find('child').get('link')] = poses[parent]@relative
            pending.remove(joint)
        assert len(pending)<old_count
    return poses


def closure_errors(poses):
    return [np.linalg.norm((poses[s['body0']]@np.r_[s['local_pos0'],1])[:3]-(poses[s['body1']]@np.r_[s['local_pos1'],1])[:3]) for s in native.loop_specs()]


def test_exact_native_tree_and_pivot_frame_transfer(cfg):
    root,meta = build(cfg)
    assert len(root.findall("joint[@type='revolute']")) == 8
    assert [j.get('name') for j in root.findall('joint') if j.find('mimic') is not None] == ['right_outer_knuckle_joint']
    for side in ['left','right']:
        follower = root.find(f"joint[@name='{side}_inner_finger_joint']")
        assert follower.find('parent').get('link') == side+'_inner_knuckle'
        np.testing.assert_allclose(origin(follower.find('origin'))[:3,3],[0,.037,.044],atol=1e-15)
        coupler = root.find(f"joint[@name='{side}_outer_finger_joint']")
        assert coupler.get('type') == 'revolute'
        assert float(coupler.find('limit').get('upper')) == 0
        link=root.find(f"link[@name='{side}_inner_finger']")
        for kind in ['visual','collision']:
            np.testing.assert_allclose(origin(link.find(kind+'/origin'))[:3,3],native.FOLLOWER_MESH_OFFSET,atol=1e-15)
    assert meta['loop_constraints'] == native.loop_specs()
    np.testing.assert_allclose(origin(root.find("joint[@name='tool_gripper']/origin"))[:3,3],0,atol=1e-15)


def test_initialization_closes_both_loops_across_full_encoder_range(cfg):
    root,_ = build(cfg)
    for pos in range(256):
        q = native.joint_targets(pos/255,cfg)
        assert max(closure_errors(fk(root,q))) < 1e-9
        for value,name in zip(q,native.JOINT_NAMES):
            lo,hi = native.LIMITS[name]
            assert lo-1e-12<=value<=hi+1e-12
    end = native.joint_targets(1.,cfg)
    assert end[6]<0 and end[7]<0  # spring limit is respected with a moving coupler.
    assert end[1]==pytest.approx(.8)


def test_registered_root_moves_all_physical_components_without_changing_linkage(cfg):
    original, _ = build(cfg)
    correction = np.array([.0114, .0075, .0058, .0076, -.0038, .0602])
    cfg['gripper']['articulation']['root_correction_xyz_rotvec'] = correction.tolist()
    registered, metadata = build(cfg)
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(correction[3:]).as_matrix()
    delta[:3, 3] = correction[:3]
    installed = origin(original.find("joint[@name='tool_gripper']/origin"))
    expected = installed @ delta @ np.linalg.inv(installed)
    for closure in [0., .2, .5, .9]:
        state = native.joint_targets(closure, cfg)
        before, after = fk(original, state), fk(registered, state)
        for name in before.keys() - {'tool0'}:
            np.testing.assert_allclose(after[name], expected @ before[name], atol=1e-12)
        assert max(closure_errors(after)) < 1e-9
    # Mesh/inertial origins and every joint except the common mount are identical.
    for element in original:
        if element.get('name') != 'tool_gripper':
            other = registered.find(f"{element.tag}[@name='{element.get('name')}']")
            assert ET.tostring(element) == ET.tostring(other)
    assert metadata['root_transform']['correction_xyz_rotvec'] == correction.tolist()


@pytest.mark.parametrize('correction', [[0.] * 5, [0.] * 7, [0., 0., float('nan'), 0., 0., 0.]])
def test_invalid_root_registration_is_rejected(cfg, correction):
    cfg['gripper']['articulation']['root_correction_xyz_rotvec'] = correction
    with pytest.raises(ValueError, match='root_correction_xyz_rotvec'):
        build(cfg)


def test_passive_motion_is_observable_and_cannot_be_faked_by_mimics(cfg):
    root,_ = build(cfg)
    q=native.joint_targets(.5,cfg)
    before=fk(root,q)
    disturbed=q.copy();disturbed[2]+=.1
    after=fk(root,disturbed)
    assert max(closure_errors(after))>.001
    assert not np.allclose(before['right_pad'],after['right_pad'])
    np.testing.assert_allclose(before['left_pad'],after['left_pad'])
    assert native.coupling_residuals(disturbed,cfg)=={'right_outer_knuckle_joint':0.}
    # Passive angles are diagnosed with the physical loop, not old rigid mimics.


def test_force_targets_and_feedback_do_not_replace_passive_state(cfg):
    q=native.joint_targets(.5,cfg);targets=native.drive_targets(.5,cfg)
    assert targets[0]==pytest.approx(q[0])
    assert targets[1]==targets[4]==pytest.approx(2.62)
    assert q[1] != targets[1]
    assert targets[2]==targets[5]==targets[6]==targets[7]==0
    q[1]+=.1;q[2]-=.15;q[3]+=.02
    assert native.closure_from_joint_positions(q,cfg)==pytest.approx(.5)
    assert native.coupling_residuals(q,cfg)=={'right_outer_knuckle_joint':pytest.approx(.02)}
    for bad in [[0]*6,[float('nan')]*8]:
        with pytest.raises(ValueError):native.closure_from_joint_positions(bad,cfg)
    with pytest.raises(ValueError):native.joint_targets(float('inf'),cfg)


@pytest.mark.parametrize('step', [.004, .002, 0., -.001, float('nan'), float('inf')])
def test_runtime_rejects_unstable_or_invalid_native_timestep(cfg, step):
    cfg['physics']['dt'] = step
    with pytest.raises(ValueError):
        native.validate_physics_timestep(cfg)


def test_stable_probe_timestep_is_a_prerequisite_not_a_success_claim(cfg):
    cfg['physics']['dt'] = .001
    assert native.validate_physics_timestep(cfg) is None


def test_instantaneous_mechanics_matches_independent_fk_for_asymmetric_states(cfg):
    root, _ = build(cfg)
    rng = np.random.default_rng(908117)
    limits = np.array([native.LIMITS[name] for name in native.JOINT_NAMES])
    # Independent left/right states exercise index mapping and loop branches;
    # arbitrary bounded positions need not satisfy closure or driver coupling.
    states = rng.uniform(limits[:, 0], limits[:, 1], size=(40, 8))
    for q in states:
        saved = q.copy()
        result = native.mechanical_diagnostics(q)
        np.testing.assert_array_equal(q, saved)
        expected = closure_errors(fk(root, q))
        np.testing.assert_allclose(
            list(result['loop_closure_per_side_m'].values()), expected, atol=1e-15)
        assert result['loop_closure_max_m'] == pytest.approx(max(expected), abs=1e-15)
        assert result['coupling_max_abs_rad'] == pytest.approx(abs(q[3] - q[0]))
        assert result['gates']['joint_limits']


def test_instantaneous_mechanics_accepts_valid_seeds_and_rejects_drive_rest_states(cfg):
    for pos in range(256):
        result = native.mechanical_diagnostics(native.joint_targets(pos / 255, cfg))
        assert result['passed'] and all(result['gates'].values())
        assert result['loop_closure_max_m'] < 1e-10
        assert result['joint_limit_violation_max_rad'] == 0
    result = native.mechanical_diagnostics(native.drive_targets(.5, cfg))
    assert not result['passed'] and not result['gates']['joint_limits']
    assert result['joint_limit_violation_max_rad'] == pytest.approx(2.62 - .8)
    assert result['joint_positions_rad'][1] == result['joint_positions_rad'][4] == 2.62


def test_instantaneous_mechanics_separates_loop_driver_and_limit_failures(cfg):
    seed = native.joint_targets(.5, cfg)
    detached = seed.copy()
    detached[6] = -.05
    result = native.mechanical_diagnostics(detached)
    assert result['gates'] == {
        'joint_coupling': True, 'joint_limits': True, 'adaptive_loop_closure': False}
    assert result['loop_closure_per_side_m']['left_distal_loop'] > .001
    assert result['loop_closure_per_side_m']['right_distal_loop'] < 1e-10

    # Each chain closes perfectly, yet its independent driver angle disagrees.
    asymmetric = seed.copy()
    asymmetric[[3, 4, 5, 7]] = native.joint_targets(.51, cfg)[[3, 4, 5, 7]]
    result = native.mechanical_diagnostics(asymmetric)
    assert result['gates'] == {
        'joint_coupling': False, 'joint_limits': True, 'adaptive_loop_closure': True}

    # Tiny coupler overtravel leaves the pin inside the 1 mm tolerance while
    # distinguishing the separate 0.002 rad actual joint-limit tolerance.
    for overtravel, accepted in ((.0019, True), (.0021, False)):
        q = seed.copy()
        q[6] = overtravel
        result = native.mechanical_diagnostics(q)
        assert result['gates']['joint_limits'] is accepted
        assert result['gates']['adaptive_loop_closure']
        assert result['joint_limit_violation_max_rad'] == pytest.approx(overtravel)


@pytest.mark.parametrize('bad', ([0.] * 6, [[0.] * 8], [float('nan')] * 8,
                                 [float('inf')] * 8, [float('-inf')] * 8))
def test_instantaneous_mechanics_rejects_malformed_or_nonfinite_input(bad):
    with pytest.raises(ValueError, match='eight finite'):
        native.mechanical_diagnostics(bad)


def test_instantaneous_guard_thresholds_match_frozen_qualification_protocol():
    protocol = json.loads((REPO / 'docs/results/w2l_native_linkage_20260909/adaptive_protocol.json').read_text())
    result = native.mechanical_diagnostics(np.zeros(8))
    assert result['thresholds'] == {
        'coupling_max_abs_rad': protocol['kinematics']['joint_coupling_max_abs_rad'],
        'joint_limit_violation_max_rad': protocol['kinematics']['adaptive_joint_limit_tolerance_rad'],
        'loop_closure_max_m': protocol['kinematics']['adaptive_loop_max_translation_m'],
    }


def test_preserves_existing_mass_and_sensor_mesh_frame(cfg):
    from phantom.sim.gripper_articulation import MODEL as OLD
    from phantom.sim.gripper_articulation import append_gripper_urdf as old_append
    root,meta=build(cfg)
    oldcfg=copy.deepcopy(cfg);oldcfg['gripper']['model']=OLD
    previous=ET.Element('robot',name='old');ET.SubElement(previous,'link',name='tool0')
    old_append(previous,REPO,oldcfg)
    for link in root.findall('link'):
        name=link.get('name')
        if name=='tool0':continue
        old=previous.find(f"link[@name='{name}']")
        assert float(link.find('inertial/mass').get('value'))==pytest.approx(float(old.find('inertial/mass').get('value')))
        if name.endswith('_inner_finger'):
            np.testing.assert_allclose(origin(link.find('inertial/origin'))[:3,3]-origin(old.find('inertial/origin'))[:3,3],native.FOLLOWER_MESH_OFFSET,atol=1e-15)
        assert link.find('inertial/inertia').attrib==old.find('inertial/inertia').attrib
    for side in ['left','right']:
        mount=origin(root.find(f"joint[@name='{side}_pad_mount']/origin"))
        oldmount=origin(previous.find(f"joint[@name='{side}_pad_mount']/origin"))
        np.testing.assert_allclose(mount[:3,3]-oldmount[:3,3],native.FOLLOWER_MESH_OFFSET,atol=1e-15)
        np.testing.assert_allclose(mount[:3,:3],oldmount[:3,:3],atol=1e-15)
    assert meta['bare_gripper_mass_kg']==pytest.approx(.925)


def usd_stage(cfg):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    root,_=build(cfg)
    stage=Usd.Stage.CreateInMemory();paths={}
    for name in native.JOINT_NAMES:
        joint=UsdPhysics.RevoluteJoint.Define(stage,'/Robot/Joints/'+name)
        joint.CreateAxisAttr('X');paths[name]=str(joint.GetPath())
        prim=joint.GetPrim();prim.AddAppliedSchema('NewtonMimicAPI')
        prim.CreateAttribute('newton:mimicCoef1',Sdf.ValueTypeNames.Float).Set(1.)
        prim.AddAppliedSchema('PhysxMimicJointAPI:rotX')
        prim.CreateAttribute('physxMimicJoint:rotX:gearing',Sdf.ValueTypeNames.Float).Set(-1.)
    for link in root.findall('link'):
        if link.get('name')=='tool0':continue
        prim=UsdGeom.Xform.Define(stage,'/Robot/'+link.get('name')).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
    return stage, paths


def test_usd_native_loops_passive_springs_and_only_driver_coupling(cfg):
    from pxr import UsdPhysics
    stage, paths = usd_stage(cfg)
    meta=native.configure_gripper_physics(stage,paths,cfg)
    for name in native.JOINT_NAMES:
        prim=stage.GetPrimAtPath(paths[name]);drive=UsdPhysics.DriveAPI(prim,'angular')
        schemas=list(prim.GetMetadata('apiSchemas').GetAppliedItems())
        assert 'NewtonMimicAPI' not in schemas
        assert not prim.GetAttribute('newton:mimicCoef1')
        assert ('PhysxMimicJointAPI:rotX' in schemas)==(name=='right_outer_knuckle_joint')
        if name==native.MASTER:
            assert drive.GetStiffnessAttr().Get()==pytest.approx(12*math.pi/180)
            assert drive.GetDampingAttr().Get()==pytest.approx(.16*math.pi/180)
            assert drive.GetMaxForceAttr().Get()==pytest.approx(1.7)
        elif name in native.SPRINGS:
            assert drive.GetStiffnessAttr().Get()==pytest.approx(.05*math.pi/180)
            assert drive.GetDampingAttr().Get()==pytest.approx(.00125*math.pi/180)
            assert drive.GetTargetPositionAttr().Get()==pytest.approx(math.degrees(2.62))
            assert math.isinf(drive.GetMaxForceAttr().Get())
        else:
            assert drive.GetStiffnessAttr().Get()==0
            assert drive.GetDampingAttr().Get()==0
            assert drive.GetMaxForceAttr().Get()==0
    assert meta['mimic_joints']==['right_outer_knuckle_joint']
    assert meta['driver_force_distribution'] == 'single_master'
    assert meta['master_stiffness_nm_rad'] == 12
    assert meta['master_damping_nm_s_rad'] == .16
    assert meta['master_max_torque_nm'] == 1.7
    for row in meta['loop_constraints']:
        joint=UsdPhysics.SphericalJoint(stage.GetPrimAtPath(row['path']))
        assert joint.GetExcludeFromArticulationAttr().Get()
        assert not joint.GetCollisionEnabledAttr().Get()
        np.testing.assert_allclose(joint.GetLocalPos0Attr().Get(),row['local_pos0'],atol=1e-8)
        np.testing.assert_allclose(joint.GetLocalPos1Attr().Get(),row['local_pos1'],atol=1e-8)
        assert str(joint.GetBody0Rel().GetTargets()[0])=='/Robot/'+row['body0']
    assert not meta['opposing_sensor_contacts_disabled']
    second=native.configure_gripper_physics(stage,paths,cfg)
    assert len(second['loop_constraints'])==2


def test_equal_split_changes_only_two_motor_drives_and_preserves_default(cfg):
    from pxr import UsdPhysics
    before = copy.deepcopy(cfg)
    default_stage, paths = usd_stage(cfg)
    default_meta = native.configure_gripper_physics(default_stage, paths, cfg)
    explicit = copy.deepcopy(cfg)
    explicit['gripper']['articulation']['driver_force_distribution'] = 'single_master'
    explicit_stage, explicit_paths = usd_stage(explicit)
    assert native.configure_gripper_physics(explicit_stage, explicit_paths, explicit) == default_meta
    assert explicit_stage.GetRootLayer().ExportToString() == default_stage.GetRootLayer().ExportToString()

    split = copy.deepcopy(cfg)
    split['gripper']['articulation']['driver_force_distribution'] = 'equal_split'
    stage, split_paths = usd_stage(split)
    meta = native.configure_gripper_physics(stage, split_paths, split)
    assert cfg == before
    assert meta['driver_force_distribution'] == 'equal_split'
    assert meta['common_mode_drive'] == default_meta['common_mode_drive'] == {
        'stiffness_nm_rad': 12., 'damping_nm_s_rad': .16, 'max_torque_nm': 1.7}
    for name in native.DRIVERS:
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(split_paths[name]), 'angular')
        assert drive.GetTypeAttr().Get() == 'force'
        assert drive.GetStiffnessAttr().Get() == pytest.approx(6. * math.pi / 180)
        assert drive.GetDampingAttr().Get() == pytest.approx(.08 * math.pi / 180)
        assert drive.GetMaxForceAttr().Get() == pytest.approx(.85)
        assert drive.GetTargetPositionAttr().Get() == 0
        assert drive.GetTargetVelocityAttr().Get() == 0
        assert meta['driver_drives'][name] == {
            'stiffness_nm_rad': 6., 'damping_nm_s_rad': .08, 'max_torque_nm': .85}
        original = UsdPhysics.DriveAPI(default_stage.GetPrimAtPath(paths[name]), 'angular')
        for getter in ['GetStiffnessAttr', 'GetDampingAttr', 'GetMaxForceAttr']:
            getattr(drive, getter)().Set(getattr(original, getter)().Get())
    # Restore the six changed drive attributes and compare the complete authored
    # USD: mimic settings, springs, loops, limits and collision filters are intact.
    assert stage.GetRootLayer().ExportToString() == default_stage.GetRootLayer().ExportToString()
    assert meta['master_stiffness_nm_rad'] == 6.
    assert meta['master_damping_nm_s_rad'] == .08
    assert meta['master_max_torque_nm'] == .85
    assert 'differential spring/damping' in meta['driver_actuator_transfer']['equal_split_difference']
    assert 'not an actual implicit-solver torque readback' in meta['driver_actuator_transfer']['force_law_scope']
    assert ET.tostring(build(split)[0]) == ET.tostring(build(cfg)[0])
    for closure in np.linspace(0, 1, 17):
        np.testing.assert_array_equal(native.drive_targets(closure, split), native.drive_targets(closure, cfg))
        np.testing.assert_array_equal(native.joint_targets(closure, split), native.joint_targets(closure, cfg))


@pytest.mark.parametrize('bad', ['both_full', 'single', '', None, 0])
def test_invalid_driver_distribution_fails_explicitly(cfg, bad):
    cfg['gripper']['articulation']['driver_force_distribution'] = bad
    with pytest.raises(ValueError, match='driver_force_distribution'):
        native.driver_settings(cfg)


def test_split_common_mode_budget_and_off_equality_torque_bounds(cfg):
    cfg['gripper']['articulation']['driver_force_distribution'] = 'equal_split'
    settings = native.driver_settings(cfg)
    common = settings['common_mode_drive']
    for key, value in common.items():
        assert sum(row[key] for row in settings['driver_drives'].values()) == value
    k, d, cap = common['stiffness_nm_rad'], common['damping_nm_s_rad'], common['max_torque_nm']

    def force(name, target, q, qd):
        drive = settings['driver_drives'][name]
        return np.clip(drive['stiffness_nm_rad'] * (target - q)
            - drive['damping_nm_s_rad'] * qd, -drive['max_torque_nm'], drive['max_torque_nm'])

    # Exercise both signs of saturation as well as the unsaturated PD regime.
    for error in [-1., -.05, 0., .05, 1.]:
        for qd in [-2., 0., 2.]:
            torques = [force(name, .4 + error, .4, qd) for name in native.DRIVERS]
            expected = np.clip(k * error - d * qd, -cap, cap)
            assert sum(torques) == pytest.approx(expected, abs=1e-14)
            assert torques[0] == torques[1]
    # Away from equality the split introduces a differential spring/damper.
    # Scalar clipping is 1-Lipschitz, so these bounds still hold at saturation.
    rng = np.random.default_rng(909128)
    unequal_torque_seen = clipped_torque_seen = False
    for target, ql, qr, vl, vr in rng.uniform(-2, 2, size=(500, 5)):
        left, right = [force(name, target, q, v) for name, q, v in zip(native.DRIVERS, [ql, qr], [vl, vr])]
        average_torque = np.clip(k * (target - (ql + qr) / 2) - d * (vl + vr) / 2, -cap, cap)
        single_master = np.clip(k * (target - ql) - d * vl, -cap, cap)
        bound = (k * abs(ql - qr) + d * abs(vl - vr)) / 4
        assert abs(left - average_torque / 2) <= bound + 1e-14
        assert abs(right - average_torque / 2) <= bound + 1e-14
        assert abs(left + right - average_torque) <= 2 * bound + 1e-14
        assert abs(left + right - single_master) <= 2 * bound + 1e-14
        assert abs(left) + abs(right) <= cap
        unequal_torque_seen |= abs(left - right) > 1e-3
        clipped_torque_seen |= abs(left) == cap / 2
    assert unequal_torque_seen and clipped_torque_seen
    ql, qr, vl, vr, target = .41, .39, .03, -.01, .4
    left = force(native.DRIVERS[0], target, ql, vl)
    right = force(native.DRIVERS[1], target, qr, vr)
    assert left - right == pytest.approx(-k * (ql - qr) / 2 - d * (vl - vr) / 2)
    assert left + right == pytest.approx(k * (target - (ql + qr) / 2) - d * (vl + vr) / 2)


def test_bad_settings_fail_explicitly(cfg):
    cfg['gripper']['articulation']['angle_at_touch_rad']=.9
    with pytest.raises(ValueError,match='driver limit'):native.joint_targets(.5,cfg)
    cfg['gripper']['articulation']['angle_at_touch_rad']=.8
    cfg['gripper']['articulation']['native_passive']={'spring_damping_nm_s_rad':-.1}
    with pytest.raises(ValueError):native.drive_targets(.5,cfg)


def test_passive_speed_opt_in_changes_only_six_speed_limits(cfg):
    original, default_meta = build(cfg)
    explicit = copy.deepcopy(cfg)
    explicit['gripper']['articulation']['passive_max_velocity_rad_s'] = 2.
    assert ET.tostring(build(explicit)[0]) == ET.tostring(original)
    assert set(default_meta['joint_max_velocity_rad_s'].values()) == {2.}

    candidate = copy.deepcopy(cfg)
    candidate['gripper']['articulation']['passive_max_velocity_rad_s'] = 20.
    changed, metadata = build(candidate)
    for joint in changed.findall("joint[@type='revolute']"):
        name = joint.get('name')
        expected = 2. if name in native.DRIVERS else 20.
        assert float(joint.find('limit').get('velocity')) == expected
        assert metadata['joint_max_velocity_rad_s'][name] == expected
        joint.find('limit').set('velocity', '2.0')
    assert ET.tostring(changed) == ET.tostring(original)
    for closure in [0., .2, .5, 1.]:
        np.testing.assert_array_equal(native.drive_targets(closure, candidate), native.drive_targets(closure, cfg))
        np.testing.assert_array_equal(native.joint_targets(closure, candidate), native.joint_targets(closure, cfg))


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf'), float('-inf')])
def test_invalid_passive_speed_rejected_before_scene_creation(cfg, bad):
    cfg['gripper']['articulation']['passive_max_velocity_rad_s'] = bad
    with pytest.raises(ValueError, match='passive_max_velocity_rad_s'):
        build(cfg)


def test_usd_passive_speed_reconfiguration_keeps_all_other_physics(cfg):
    stage, paths = usd_stage(cfg)
    original = native.configure_gripper_physics(stage, paths, cfg)
    before = stage.GetRootLayer().ExportToString()
    cfg['gripper']['articulation']['passive_max_velocity_rad_s'] = 20.
    candidate = native.configure_gripper_physics(stage, paths, cfg)
    for name, path in paths.items():
        speed = stage.GetPrimAtPath(path).GetAttribute('physxJoint:maxJointVelocity')
        expected = 2. if name in native.DRIVERS else 20.
        assert speed.Get() == pytest.approx(math.degrees(expected))
        speed.Set(math.degrees(2.))
    assert stage.GetRootLayer().ExportToString() == before
    candidate.pop('joint_max_velocity_rad_s')
    original.pop('joint_max_velocity_rad_s')
    assert candidate == original


def test_pinned_armature_matches_reference_without_changing_rigid_body_model(cfg):
    original_cfg = copy.deepcopy(cfg)
    original, original_meta = build(cfg)
    candidate = copy.deepcopy(cfg)
    candidate['gripper']['articulation']['armature_model'] = 'pinned_mjcf_v1'
    changed, metadata = build(candidate)
    reference = ET.parse(native._verify_reference(REPO)).getroot()
    classes = {
        **dict.fromkeys(native.DRIVERS, 'driver'),
        **dict.fromkeys(native.SPRINGS, 'spring_link'),
        **dict.fromkeys(native.FOLLOWERS, 'follower'),
        **dict.fromkeys(native.COUPLERS, 'coupler'),
    }
    expected = {
        name: float(reference.find(f".//default[@class='{kind}']/joint").get('armature'))
        for name, kind in classes.items()
    }
    assert metadata['armature_model'] == 'pinned_mjcf_v1'
    assert metadata['joint_armature_kg_m2'] == expected
    assert metadata['armature_transfer']['source_reference_sha256'] == native.REFERENCE_SHA256
    assert set(original_meta['joint_armature_kg_m2'].values()) == {0.}
    # Armature is authored after URDF import. All link/sensor masses, COMs,
    # inertias, joint limits, speeds, poses and topology stay byte-identical.
    assert ET.tostring(changed) == ET.tostring(original)
    for closure in [0., .2, .5, 1.]:
        np.testing.assert_array_equal(native.drive_targets(closure, candidate), native.drive_targets(closure, cfg))
        np.testing.assert_array_equal(native.joint_targets(closure, candidate), native.joint_targets(closure, cfg))
    assert native.driver_settings(candidate) == native.driver_settings(cfg)
    assert native.passive_settings(candidate) == native.passive_settings(cfg)
    assert native.joint_velocity_limits(candidate) == dict.fromkeys(native.JOINT_NAMES, 2.)
    assert cfg == original_cfg


def test_armature_default_is_zero_and_explicit_none_is_identical(cfg):
    original, original_meta = build(cfg)
    stage, paths = usd_stage(cfg)
    metadata = native.configure_gripper_physics(stage, paths, cfg)
    before = stage.GetRootLayer().ExportToString()
    for path in paths.values():
        prim = stage.GetPrimAtPath(path)
        assert 'PhysxJointAPI' in prim.GetMetadata('apiSchemas').GetAppliedItems()
        value = prim.GetAttribute('physxJoint:armature')
        assert value.HasAuthoredValueOpinion()
        assert value.Get() == 0.
    cfg['gripper']['articulation']['armature_model'] = 'none'
    explicit, explicit_meta = build(cfg)
    assert ET.tostring(explicit) == ET.tostring(original)
    assert explicit_meta == original_meta
    assert native.configure_gripper_physics(stage, paths, cfg) == metadata
    assert stage.GetRootLayer().ExportToString() == before


def test_usd_armature_changes_only_eight_values_and_resets_reused_stage(cfg):
    stage, paths = usd_stage(cfg)
    original_meta = native.configure_gripper_physics(stage, paths, cfg)
    before = stage.GetRootLayer().ExportToString()
    cfg['gripper']['articulation']['armature_model'] = 'pinned_mjcf_v1'
    candidate_meta = native.configure_gripper_physics(stage, paths, cfg)
    for name, path in paths.items():
        prim = stage.GetPrimAtPath(path)
        assert 'PhysxJointAPI' in prim.GetMetadata('apiSchemas').GetAppliedItems()
        value = prim.GetAttribute('physxJoint:armature')
        # kg m^2 is direct in USD. A radians/degrees conversion must fail.
        assert value.Get() == pytest.approx(.005 if name in native.DRIVERS else .001)
        value.Set(0.)
    assert stage.GetRootLayer().ExportToString() == before
    for key in ('armature_model', 'joint_armature_kg_m2', 'armature_transfer'):
        candidate_meta.pop(key)
        original_meta.pop(key)
    assert candidate_meta == original_meta
    # Apply the candidate again, then remove the config key: a previously
    # imported/configured asset must not leak nonzero armature into defaults.
    native.configure_gripper_physics(stage, paths, cfg)
    cfg['gripper']['articulation'].pop('armature_model')
    native.configure_gripper_physics(stage, paths, cfg)
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize('bad', ['native', 'pinned_mjcf', '', None, 0, False, [], {}])
def test_unknown_armature_model_fails_before_authoring(cfg, bad):
    stage, paths = usd_stage(cfg)
    before = stage.GetRootLayer().ExportToString()
    cfg['gripper']['articulation']['armature_model'] = bad
    with pytest.raises(ValueError, match='armature_model'):
        build(cfg)
    with pytest.raises(ValueError, match='armature_model'):
        native.configure_gripper_physics(stage, paths, cfg)
    assert stage.GetRootLayer().ExportToString() == before
