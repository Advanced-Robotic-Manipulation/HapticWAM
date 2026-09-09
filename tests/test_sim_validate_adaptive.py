"""Independent mechanical gates for native eight-joint traces; no object claims."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from phantom.sim.gripper_articulation import (
    ADAPTIVE_MODEL,
    MODEL,
    closure_from_joint_positions,
    drive_targets,
    finger_joint_names,
    joint_targets,
    link_transforms_from_joint_positions,
)
from tools.sim import validate_w2l_replay as audit

REPO=Path(__file__).resolve().parents[1]


def update_pad_readback(trace,cfg):
    """Produce physical-state fixtures through production FK, not audit._fk_function."""
    positions=[];orientations=[]
    for values,tcp in zip(trace['finger_q'],trace['tcp']):
        frames=link_transforms_from_joint_positions(REPO,cfg,values)
        rotation=Rotation.from_rotvec(tcp[3:]);tool=tcp[:3]-rotation.apply([0,0,.18])
        pads=[];quats=[]
        for side in ['left','right']:
            local=frames[side+'_pad']
            pads.append(tool+rotation.apply(local[:3,3]))
            quaternion=(rotation*Rotation.from_matrix(local[:3,:3])).as_quat()
            quats.append(quaternion[[3,0,1,2]])
        positions.append(pads);orientations.append(quats)
    trace['pad_position']=np.asarray(positions)
    trace['pad_orientation_wxyz']=np.asarray(orientations)


@pytest.fixture
def adaptive():
    cfg=json.loads((REPO/'configs/sim/waffles_w2l_articulated_provisional_20260908.json').read_text())
    cfg['gripper']['model']=ADAPTIVE_MODEL
    cfg['gripper']['articulation']['native_passive']={
        'spring_reference_rad':2.62,'spring_stiffness_nm_rad':.05,'spring_damping_nm_s_rad':.00125}
    protocol=json.loads(audit.DEFAULT_PROTOCOL.read_text())
    protocol['kinematics'].update(adaptive_loop_max_translation_m=.001,adaptive_joint_limit_tolerance_rad=.002)
    closures=[0.,.2,.45,.9]
    n=len(closures);tcp=np.zeros((n,6))
    tcp[:,:3]=np.array([.2,.3,.5])+np.arange(n)[:,None]*[.004,-.005,.006]
    tcp[:,3:]=np.array([.31,-.23,.17])+np.arange(n)[:,None]*[.01,.02,-.015]
    trace={'t':np.arange(n)*.1,'tcp':tcp,
           'finger_q':np.array([joint_targets(c,cfg)for c in closures]),
           'target_finger_q':np.array([drive_targets(c,cfg)for c in closures])}
    run={'mode':'dynamics','gripper_model':ADAPTIVE_MODEL,'finger_joint_names':list(finger_joint_names(cfg)),
         'finger_joint_units':'radians','gripper_feedback_source':'measured master joint angle'}
    update_pad_readback(trace,cfg)
    return trace,run,cfg,protocol


def evaluate(data):
    trace,run,cfg,protocol=data
    return audit.mechanics_metrics(trace,run,cfg,protocol,REPO)


def test_consistent_native_seed_passes_loop_driver_limit_and_physical_pad_gates(adaptive):
    row=evaluate(adaptive)
    assert set(row['gates'])=={'joint_coupling','pad_world_fk','adaptive_loop_closure','joint_limits'}
    assert all(row['gates'].values())
    assert row['pad_fk_translation_max_m']<1e-12
    assert row['pad_fk_rotation_max_rad']<1e-12
    assert max(row['loop_closure_per_side_max_m'].values())<1e-10
    assert row['joint_limit_violation_max_rad']<1e-10
    assert set(row['coupling_per_joint_max_abs_rad'])=={'right_outer_knuckle_joint'}
    # A consistent native seed need not satisfy the old six-joint parallel mimics.
    q=adaptive[0]['finger_q']
    assert np.max(abs(q[:,2]+q[:,0]))>.005
    assert 'actual 8 joint positions' in row['fk_definition']


def test_passive_spring_drive_reference_is_not_an_actual_joint_limit_violation(adaptive):
    trace,_,cfg,_=adaptive
    assert np.all(trace['target_finger_q'][:,[1,4]]==2.62)
    assert np.all(trace['finger_q'][:,[1,4]]<=.8+1e-10)
    np.testing.assert_allclose([closure_from_joint_positions(q,cfg)for q in trace['target_finger_q']],[0,.2,.45,.9],atol=1e-12)
    assert evaluate(adaptive)['gates']['joint_limits']


@pytest.mark.parametrize('corruption',['run_model_v1','config_model_v1','six_names','reordered_names','degree_units'])
def test_legacy_or_misdeclared_joint_schema_cannot_be_scored_as_native(adaptive,corruption):
    _,run,cfg,_=adaptive
    if corruption=='run_model_v1':run['gripper_model']=MODEL
    elif corruption=='config_model_v1':cfg['gripper']['model']=MODEL
    elif corruption=='six_names':run['finger_joint_names']=run['finger_joint_names'][:6]
    elif corruption=='reordered_names':run['finger_joint_names'][6:]=run['finger_joint_names'][6:][::-1]
    elif corruption=='degree_units':run['finger_joint_units']='degrees'
    with pytest.raises(ValueError,match='joint names/radian units/model'):
        evaluate(adaptive)


@pytest.mark.parametrize('field',['finger_q','target_finger_q'])
def test_six_column_trace_is_rejected_even_with_correct_native_run_metadata(adaptive,field):
    adaptive[0][field]=adaptive[0][field][:,:6]
    with pytest.raises(ValueError,match=field+' must be finite with shape'):
        evaluate(adaptive)


@pytest.mark.parametrize('key',['adaptive_loop_max_translation_m','adaptive_joint_limit_tolerance_rad'])
@pytest.mark.parametrize('invalid',['missing',0.,-.01,float('nan'),float('inf')])
def test_native_thresholds_must_be_explicit_finite_and_positive(adaptive,key,invalid):
    k=adaptive[3]['kinematics']
    if invalid=='missing':del k[key]
    else:k[key]=invalid
    with pytest.raises(ValueError,match='explicit positive loop and joint-limit thresholds'):
        evaluate(adaptive)


def test_detached_loop_fails_even_with_perfect_physical_pad_readback(adaptive):
    trace,_,cfg,_=adaptive
    trace['finger_q'][:]=joint_targets(.35,cfg)
    update_pad_readback(trace,cfg)
    old_pad_positions=trace['pad_position'].copy()
    # The outer-finger coupler is on the other loop branch from the sensor;
    # detaching it does not change the sensor's URDF-tree pose.
    trace['finger_q'][:,6]-=.04
    update_pad_readback(trace,cfg)
    np.testing.assert_allclose(trace['pad_position'],old_pad_positions,atol=1e-12)
    row=evaluate(adaptive)
    assert row['gates']['pad_world_fk'] and row['gates']['joint_coupling'] and row['gates']['joint_limits']
    assert not row['gates']['adaptive_loop_closure']
    assert row['loop_closure_per_side_max_m']['left_distal_loop']>.001
    assert row['loop_closure_per_side_max_m']['right_distal_loop']<1e-10


def test_individually_closed_left_right_loops_do_not_hide_motor_driver_mismatch(adaptive):
    trace,_,cfg,_=adaptive
    left=joint_targets(.35,cfg);right=joint_targets(.36,cfg)
    trace['finger_q'][:]=left
    trace['finger_q'][:,[3,4,5,7]]=right[[3,4,5,7]]
    update_pad_readback(trace,cfg)
    row=evaluate(adaptive)
    assert row['gates']['pad_world_fk'] and row['gates']['adaptive_loop_closure'] and row['gates']['joint_limits']
    assert row['coupling_max_abs_rad']==pytest.approx(.01*.8/.9,abs=1e-12)
    assert not row['gates']['joint_coupling']


@pytest.mark.parametrize('coupler',[.0019,.003])
def test_actual_coupler_limit_is_checked_separately_from_loop_distance(adaptive,coupler):
    trace,_,cfg,_=adaptive
    trace['finger_q'][:]=joint_targets(.35,cfg)
    trace['finger_q'][:,6]=coupler
    update_pad_readback(trace,cfg)
    row=evaluate(adaptive)
    assert row['gates']['pad_world_fk'] and row['gates']['joint_coupling'] and row['gates']['adaptive_loop_closure']
    assert row['joint_limit_violation_max_rad']==pytest.approx(coupler,abs=1e-12)
    assert row['gates']['joint_limits']==(coupler<=.002)


def test_pad_body_offset_fails_without_breaking_joint_coupling_or_loop_gates(adaptive):
    adaptive[0]['pad_position'][2,1,2]+=.0011
    row=evaluate(adaptive)
    assert row['gates']['joint_coupling'] and row['gates']['adaptive_loop_closure'] and row['gates']['joint_limits']
    assert row['pad_fk_translation_max_m']==pytest.approx(.0011,abs=1e-12)
    assert not row['gates']['pad_world_fk']
