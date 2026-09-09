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


def test_usd_native_loops_passive_springs_and_only_driver_coupling(cfg):
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


def test_bad_settings_fail_explicitly(cfg):
    cfg['gripper']['articulation']['angle_at_touch_rad']=.9
    with pytest.raises(ValueError,match='driver limit'):native.joint_targets(.5,cfg)
    cfg['gripper']['articulation']['angle_at_touch_rad']=.8
    cfg['gripper']['articulation']['native_passive']={'spring_damping_nm_s_rad':-.1}
    with pytest.raises(ValueError):native.drive_targets(.5,cfg)
