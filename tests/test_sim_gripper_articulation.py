"""Physical linkage invariants and CPU USD authoring; no Isaac or hardware."""

import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from phantom.sim.gripper_articulation import (
    JOINT_MULTIPLIERS, MASTER, MODEL, append_gripper_urdf,
    closure_from_joint_positions, configure_gripper_physics, coupling_residuals,
    finger_drive_type, finger_joint_names, joint_targets,
    is_articulated, link_transforms_from_joint_positions,
)


@pytest.fixture
def candidate(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo/'assets/sim/robotiq',tmp_path/'assets/sim/robotiq')
    directory = tmp_path/'assets/sim/dmtac_w2l'
    directory.mkdir(parents=True)
    parts = []
    for name,role,mass,lo,hi in [
        ('gel','gel',.020,[-.002,-.018,-.0135],[.002,.018,.0135]),
        ('housing','body',.111,[.002,-.023,-.04],[.033,.023,.03614]),
        ('adapter','adapter',.008,[.003,-.0195,-.04],[.0095,.0195,-.0148]),
    ]:
        path = directory/(name+'.stl')
        path.write_text('solid fixture\nendsolid fixture\n')
        parts.append({'id':name,'role':role,'mass_kg':mass,'visual_mesh':str(path.relative_to(tmp_path)), 'collision_meshes':[str(path.relative_to(tmp_path))], 'origin':{'xyz':[0,0,0],'rpy':[0,0,0]}, 'bounding_box_m':{'min':lo,'max':hi}})
    (directory/'geometry.json').write_text(json.dumps({'parts':parts}))
    cfg = {'gripper':{'model':MODEL,'stroke':.07,'pad_touch_command':.9,'yaw':-.377,
                     'gel_geometry':{'pad_thickness_m':.004},
                     'articulation':{'angle_at_touch_rad':.8,'bare_gripper_mass_kg':.925,
                       'drive_stiffness_nm_rad':12.,'drive_damping_nm_s_rad':.16,'drive_max_torque_nm':1.7,
                       'mounts':{'left':{'xyz':[0,-.0272011,.067656],'rpy':[0,0,math.pi/2]},
                                 'right':{'xyz':[0,-.0272011,.067656],'rpy':[0,0,-math.pi/2]}}}}}
    return tmp_path,cfg


def build(candidate):
    repo,cfg = candidate
    root=ET.Element('robot',name='test');ET.SubElement(root,'link',name='tool0')
    metadata=append_gripper_urdf(root,repo,cfg)
    return root,metadata


@pytest.mark.parametrize('closure',[0.,.2,.45,.7,.9,1.])
def test_nominal_tree_is_one_correlated_dof_and_master_feedback(candidate,closure):
    _,cfg=candidate
    q=joint_targets(closure,cfg)
    np.testing.assert_allclose(q,q[0]*np.array([1,1,-1,1,1,-1]))
    assert closure_from_joint_positions(q,cfg)==pytest.approx(min(closure,.9))
    assert max(map(abs,coupling_residuals(q,cfg).values()))==0
    q[3]+=.05
    assert closure_from_joint_positions(q,cfg)==pytest.approx(min(closure,.9))
    assert coupling_residuals(q,cfg)['right_outer_knuckle_joint']==pytest.approx(.05)


def test_real_sensor_links_share_linkage_motion_and_no_separate_visual_skin(candidate):
    repo,cfg=candidate
    root,metadata=build(candidate)
    assert not root.findall("joint[@type='prismatic']")
    assert set(j.get('name') for j in root.findall("joint[@type='revolute']"))==set(JOINT_MULTIPLIERS)
    q0,q1=joint_targets(0,cfg),joint_targets(.45,cfg)
    start=link_transforms_from_joint_positions(repo,cfg,q0)
    end=link_transforms_from_joint_positions(repo,cfg,q1)
    for side in ['left','right']:
        # Axial advance is the independent analytic URDF chain result, not zero.
        assert end[side+'_pad'][2,3]-start[side+'_pad'][2,3]==pytest.approx(.01124775241292933)
        for name in metadata['sensor_links'][side]:
            np.testing.assert_allclose(end[name],end[side+'_pad'],atol=1e-14)
        # Body and gel have separate rigid links; housing geometry cannot be
        # misclassified as gel solely because it shares a parent rigid body.
        gel=root.find(f"link[@name='{side}_pad']")
        assert len(gel.findall('collision'))==1
        assert 'gel.stl' in gel.find('collision/geometry/mesh').get('filename')
    # Asymmetric physical follower error changes only the corresponding branch.
    q2=q1.copy();q2[3]+=.02
    disturbed=link_transforms_from_joint_positions(repo,cfg,q2)
    assert not np.allclose(disturbed['left_pad'],end['left_pad'])
    np.testing.assert_allclose(disturbed['right_pad'],end['right_pad'])


def test_negative_mimics_have_valid_physical_limits_and_mass_is_accounted(candidate):
    root,metadata=build(candidate)
    mass=0.
    for link in root.findall('link'):
        if link.get('name')=='tool0':continue
        inertial=link.find('inertial')
        assert inertial is not None
        m=float(inertial.find('mass').get('value'));assert m>0;mass+=m
        i=inertial.find('inertia')
        matrix=np.array([[float(i.get('ixx')),float(i.get('ixy')),float(i.get('ixz'))],
                         [float(i.get('ixy')),float(i.get('iyy')),float(i.get('iyz'))],
                         [float(i.get('ixz')),float(i.get('iyz')),float(i.get('izz'))]])
        assert np.linalg.eigvalsh(matrix).min()>0
    assert mass==pytest.approx(.925+2*(.131+.008))
    assert metadata['bare_gripper_mass_kg']==pytest.approx(.925)
    for name,mult in JOINT_MULTIPLIERS.items():
        limit=root.find(f"joint[@name='{name}']/limit")
        assert float(limit.get('lower'))<=.8*mult<=float(limit.get('upper'))


def test_right_geometry_rotates_inside_aligned_link_frame(candidate):
    root,_=build(candidate)
    left=root.find("link[@name='left_pad']/collision/origin")
    right=root.find("link[@name='right_pad']/collision/origin")
    np.testing.assert_allclose(np.fromstring(left.get('rpy'),sep=' '),[0,0,0])
    np.testing.assert_allclose(np.fromstring(right.get('rpy'),sep=' '),[0,0,math.pi])


@pytest.mark.parametrize('model', [MODEL, 'robotiq_2f85_w2l_adaptive_v2'])
def test_task_visual_colors_preserve_cad_physics_and_default_scene(candidate, model):
    repo, cfg = candidate
    cfg['gripper']['model'] = model
    manifest = repo/'assets/sim/dmtac_w2l/geometry.json'
    original_bytes = manifest.read_bytes()
    baseline, _ = build(candidate)
    cfg['gripper']['visual_materials'] = {'housing': [.025, .030, .028, 1.]}
    changed, _ = build(candidate)
    for side in ('left', 'right'):
        color = changed.find(f"link[@name='{side}_sensor_housing']/visual/material/color")
        np.testing.assert_allclose(np.fromstring(color.get('rgba'), sep=' '), [.025, .030, .028, 1.])
    # Removing visual color leaves byte-identical authored geometry, transforms,
    # collision, inertia, linkage and all other visual properties in both trees.
    for tree in (baseline, changed):
        for material in tree.findall('.//visual/material'):
            color = material.find('color')
            if color is not None:
                material.remove(color)
    assert ET.tostring(changed) == ET.tostring(baseline)
    assert manifest.read_bytes() == original_bytes
    del cfg['gripper']['visual_materials']
    restored, _ = build(candidate)
    for side in ('left', 'right'):
        color = restored.find(f"link[@name='{side}_sensor_housing']/visual/material/color")
        np.testing.assert_allclose(np.fromstring(color.get('rgba'), sep=' '), [.82, .88, .90, 1.])


@pytest.mark.parametrize('overrides', [
    None, [], {'missing_component': [0., 0., 0., 1.]},
    {'housing': [0., 0., 0.]}, {'housing': [0., float('nan'), 0., 1.]},
    {'housing': [1.01, 0., 0., 1.]}, {'housing': 'black'},
])
def test_invalid_visual_material_overrides_fail_explicitly(candidate, overrides):
    _, cfg = candidate
    cfg['gripper']['visual_materials'] = overrides
    with pytest.raises(ValueError, match='visual_materials'):
        build(candidate)


def test_physx_mimics_have_correct_sign_single_drive_units_and_local_filters(candidate):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd,UsdGeom,UsdPhysics,Sdf
    _,cfg=candidate
    stage=Usd.Stage.CreateInMemory();paths={}
    for name in JOINT_MULTIPLIERS:
        joint=UsdPhysics.RevoluteJoint.Define(stage,'/Robot/Joints/'+name)
        joint.CreateAxisAttr('X');paths[name]=str(joint.GetPath())
        joint.GetPrim().AddAppliedSchema('NewtonMimicAPI')
        joint.GetPrim().CreateAttribute('newton:mimicCoef1',Sdf.ValueTypeNames.Float).Set(1.)
        joint.GetPrim().CreateRelationship('newton:mimicJoint').SetTargets(['/Old/Reference'])
    for side in ['left','right']:
        for name in ['inner_knuckle','inner_finger','pad','sensor_housing','sensor_adapter']:
            prim=UsdGeom.Xform.Define(stage,f'/Robot/{side}_{name}').GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim)
    metadata=configure_gripper_physics(stage,paths,cfg)
    for name,mult in JOINT_MULTIPLIERS.items():
        prim=stage.GetPrimAtPath(paths[name]);drive=UsdPhysics.DriveAPI(prim,'angular')
        authored=list(prim.GetMetadata('apiSchemas').GetAppliedItems())
        assert 'NewtonMimicAPI' not in authored
        assert not prim.GetAttribute('newton:mimicCoef1')
        if name==MASTER:
            assert drive.GetStiffnessAttr().Get()==pytest.approx(12*math.pi/180)
            assert drive.GetDampingAttr().Get()==pytest.approx(.16*math.pi/180)
            assert drive.GetMaxForceAttr().Get()==pytest.approx(1.7)
        else:
            assert drive.GetStiffnessAttr().Get()==0
            assert drive.GetDampingAttr().Get()==0
            assert drive.GetMaxForceAttr().Get()==0
            assert 'PhysxMimicJointAPI:rotX' in authored
            assert prim.GetAttribute('physxMimicJoint:rotX:gearing').Get()==-mult
            assert str(prim.GetRelationship('physxMimicJoint:rotX:referenceJoint').GetTargets()[0])==paths[MASTER]
    assert len(metadata['structural_collision_filters'])==8
    for a,b in metadata['structural_collision_filters']:
        assert a.split('/')[-1].split('_')[0]==b.split('/')[-1].split('_')[0]
    assert not metadata['opposing_sensor_contacts_disabled']


def test_legacy_mapping_is_unchanged_and_bad_measurements_rejected(candidate):
    _,cfg=candidate
    legacy={'gripper':{'stroke':.07,'pad_touch_command':.9}}
    assert finger_joint_names(legacy)==('left_finger_joint','right_finger_joint')
    assert finger_drive_type(legacy)=='linear'
    assert finger_drive_type(cfg)=='angular'
    np.testing.assert_allclose(joint_targets(.45,legacy),[.0175,.0175])
    assert closure_from_joint_positions([.0125,.0225],legacy)==pytest.approx(.45)
    with pytest.raises(ValueError):joint_targets(float('nan'),cfg)
    with pytest.raises(ValueError):closure_from_joint_positions([.4,.4],cfg)
    with pytest.raises(ValueError):closure_from_joint_positions([float('nan')]*6,cfg)


def test_missing_mount_and_impossible_bare_mass_fail_explicitly(candidate):
    _,cfg=candidate
    cfg['gripper']['articulation']['bare_gripper_mass_kg']=.01
    with pytest.raises(ValueError,match='base mass'):build(candidate)
    cfg['gripper']['articulation']['bare_gripper_mass_kg']=.925
    del cfg['gripper']['articulation']['mounts']['right']
    with pytest.raises(ValueError,match='mounts'):build(candidate)


def test_cad_mass_tensor_and_com_are_transformed_with_right_geometry(candidate):
    repo,_=candidate
    path=repo/'assets/sim/dmtac_w2l/geometry.json'
    data=json.loads(path.read_text())
    part=data['parts'][1]
    part['center_of_mass_m']=[.012,.003,.004]
    tensor=np.array([[3e-5,1e-6,2e-6],[1e-6,4e-5,3e-6],[2e-6,3e-6,5e-5]])
    part['inertia_kg_m2']=tensor.tolist();path.write_text(json.dumps(data))
    root,_=build(candidate)
    inertial=root.find("link[@name='right_sensor_housing']/inertial")
    np.testing.assert_allclose(np.fromstring(inertial.find('origin').get('xyz'),sep=' '),[-.012,-.003,.004],atol=1e-14)
    i=inertial.find('inertia')
    assert float(i.get('ixy'))==pytest.approx(1e-6)
    assert float(i.get('ixz'))==pytest.approx(-2e-6)
    assert float(i.get('iyz'))==pytest.approx(-3e-6)


def test_articulated_width_probe_inverts_actual_linkage_gap(candidate):
    from phantom.sim.gripper_probe import closure_for_gap,jaw_geometry
    repo,cfg=candidate
    sweep=np.linspace(0,.9,31)
    gaps=[jaw_geometry(repo,cfg,c)[0] for c in sweep]
    assert np.all(np.diff(gaps)<0)
    assert gaps[0]==pytest.approx(.078)
    # Width inversion uses linkage FK, not the old linear slider equation.
    closures=[]
    for width in [.0272,.035,.042,.05]:
        closure=closure_for_gap(repo,cfg,width)
        actual_gap,_,rotation=jaw_geometry(repo,cfg,closure)
        assert actual_gap==pytest.approx(width,abs=1e-12)
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-14)
        values=joint_targets(closure,cfg)
        assert values[0]==pytest.approx(closure/.9*.8)
        assert closure_from_joint_positions(values,cfg)==pytest.approx(closure)
        assert max(map(abs,coupling_residuals(values,cfg).values()))==0
        closures.append(closure)
    assert np.all(np.diff(closures)<0)
    with pytest.raises(ValueError,match='outside modeled gap'):
        closure_for_gap(repo,cfg,gaps[0]+.001)
    with pytest.raises(ValueError,match='finite and positive'):
        closure_for_gap(repo,cfg,0)
    # This fixture crosses zero gap before the commanded touch limit. The
    # algorithm must not mistake that nominal overlap for usable object width.
    assert gaps[-1]<0


def test_width_probe_rejects_nonparallel_candidate_mount(candidate):
    from phantom.sim.gripper_probe import jaw_geometry
    repo,cfg=candidate
    cfg['gripper']['articulation']['mounts']['right']['rpy'][1]=.2
    with pytest.raises(ValueError,match='parallel'):
        jaw_geometry(repo,cfg,.45)


def test_unknown_named_model_cannot_silently_use_old_slider(candidate):
    _,cfg=candidate
    assert is_articulated(cfg)
    assert not is_articulated({'gripper':{}})
    assert not is_articulated({'gripper':{'model':'legacy_sliding_pad_proxy'}})
    cfg['gripper']['model']=MODEL+'typo'
    for helper in [is_articulated,finger_joint_names,finger_drive_type]:
        with pytest.raises(ValueError,match='refusing silent legacy fallback'):
            helper(cfg)
    with pytest.raises(ValueError,match='refusing silent legacy fallback'):
        joint_targets(.45,cfg)
