"""Outcome/integrity regressions for the independent CPU-only run validator."""

import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_sim_gripper_articulation import candidate  # noqa: F401 -- shared CAD fixture
from phantom.sim.gripper_articulation import MODEL, finger_joint_names, joint_targets, link_transforms_from_joint_positions
from tools.sim import validate_w2l_replay as audit


@pytest.fixture
def protocol():
    return json.loads(audit.DEFAULT_PROTOCOL.read_text())


def make_trace(candidate,mode='contact_probe'):
    repo,cfg=copy.deepcopy(candidate)
    cfg['camera']={'resolution':[640,480],'fx':600.,'fy':601.,'cx':320.,'cy':240.}
    duration=10. if mode=='contact_probe' else 20.
    t=np.linspace(0,duration,int(duration*10)+1);n=len(t)
    rise=np.clip((t-1)/2,0,1)*.12
    tcp=np.zeros((n,6));tcp[:,:3]=[.2,.3,.38];tcp[:,2]+=rise
    tcp[:,3:]=[.2,-.3,.1]  # A tilted tool exposes unrotated TCP-offset bugs.
    obj=np.tile([.2,.3,.2],(n,1));obj[:,2]+=rise
    active=t>=.5
    if mode=='contact_probe':
        active&=t<7.3
        obj[:,2]-=np.clip((t-7.3)/.7,0,1)*.14
    values=joint_targets(.45,cfg)
    local=link_transforms_from_joint_positions(repo,cfg,values)
    tool_rotation=Rotation.from_rotvec(tcp[:,3:]);tool_position=tcp[:,:3]-tool_rotation.apply([0,0,.18])
    pads=[];quats=[]
    for side in ['left','right']:
        transform=local[side+'_pad']
        pads.append(tool_position+tool_rotation.apply(transform[:3,3]))
        orientation=(tool_rotation*Rotation.from_matrix(transform[:3,:3])).as_quat()
        quats.append(orientation[:,[3,0,1,2]])
    trace={'t':t,'physics_t':t.copy(),'q':np.zeros((n,6)),'tcp':tcp,
           'finger_q':np.tile(values,(n,1)),'target_finger_q':np.tile(values,(n,1)),
           'pad_position':np.stack(pads,axis=1),'pad_orientation_wxyz':np.stack(quats,axis=1),
           'waffle_position':obj,'pad_packet_normal_force':np.tile(active[:,None],(1,2)).astype(float)}
    run={'mode':mode,'episode':'fixture','gripper_model':MODEL,'finger_joint_names':list(finger_joint_names(cfg)),
         'finger_joint_units':'radians','gripper_feedback_source':'measured master joint angle',
         'object_dynamics':{'rigid_body_dynamic':True,'kinematic':False,'attachments':[],'pose_writes_after_initialization':0},
         'camera_projection':{'readback_matches_config':True,'resolution':[640,480],
                              'intrinsics_px':[[600,0,320],[0,601,240],[0,0,1]]}}
    gel=[{'t':float(stamp),'filter_paths':[['/World/Waffle'],['/World/Waffle']],
          'per_pad':[{'filter_labels_match_columns':True,'contact_filter_column_count':1,
                      'per_filter_contacts':[{'filter_path':'/World/Waffle','gel_compression_n':1.}] if yes else []}
                     for _ in range(2)]} for stamp,yes in zip(t,active)]
    reference={name:trace[name].copy() for name in ['t','q','tcp']}
    return repo,cfg,trace,gel,run,reference


def result(data,protocol,case_id='fixture'):
    repo,cfg,trace,gel,run,reference=data
    return audit.evaluate(trace,gel,run,cfg,protocol,reference=reference,case_id=case_id,repo=repo)


def test_complete_free_object_probe_passes_without_claiming_real_validation(candidate,protocol):
    row=result(make_trace(candidate),protocol)
    assert row['passed']
    assert all(row['object_gates'].values())
    assert row['object']['hold_median_lift_m']==pytest.approx(.12)
    assert row['object']['release_relative_vertical_fall_m']==pytest.approx(.14)
    assert row['mechanics']['pad_fk_translation_max_m']<1e-12
    assert not row['real_motion_physics_reproduced']
    assert not row['registration_or_hardware_qualified']
    assert not row['placement_claim']


def test_body_contact_without_active_gel_does_not_count_as_pick(candidate,protocol):
    data=make_trace(candidate,'dynamics')
    for row in data[3]:
        for pad in row['per_pad']:pad['per_filter_contacts']=[]
    row=result(data,protocol)
    assert row['object']['pad_body_acquisition_onset_s'] is not None
    assert row['object']['active_gel_acquisition_onset_s'] is None
    assert not row['passed']
    assert not row['object_gates']['active_gel_pick']


def test_opposite_signed_follower_errors_are_not_averaged_away(candidate,protocol):
    data=make_trace(candidate);trace=data[2]
    trace['finger_q'][:,1]+=.006;trace['finger_q'][:,4]-=.006
    row=result(data,protocol)
    assert row['mechanics']['coupling_max_abs_rad']==pytest.approx(.006)
    assert not row['common_gates']['joint_coupling']
    assert not row['passed']


def test_actual_pad_world_pose_detects_independent_visual_or_physical_offset(candidate,protocol):
    data=make_trace(candidate);data[2]['pad_position'][:,0,2]+=.0011
    row=result(data,protocol)
    assert row['mechanics']['pad_fk_translation_max_m']==pytest.approx(.0011)
    assert not row['common_gates']['pad_world_fk']


@pytest.mark.parametrize('key',['finger_q','target_finger_q','pad_position','pad_orientation_wxyz'])
def test_missing_measured_mechanics_cannot_pass(candidate,protocol,key):
    data=make_trace(candidate);del data[2][key]
    with pytest.raises(KeyError):result(data,protocol)


def test_unknown_packet_filter_coverage_is_not_assumed_zero(candidate,protocol):
    data=make_trace(candidate);data[3][0]['filter_paths'][0]=['/World/Bench/Slab']
    with pytest.raises(ValueError,match='packet-filter coverage'):result(data,protocol)


def test_unaligned_or_sparse_contact_times_cannot_bridge_grasp(candidate,protocol):
    data=make_trace(candidate);data[3][10]['t']+=.01
    with pytest.raises(ValueError,match='same sampled timestamps'):result(data,protocol)
    data=make_trace(candidate);del data[3][11:14]
    with pytest.raises(ValueError,match='sample gap'):result(data,protocol)


def test_lift_before_contact_cannot_be_reused_as_post_grasp_lift(candidate,protocol):
    data=make_trace(candidate,'dynamics');t=data[2]['t']
    data[2]['waffle_position'][:,2]=.2+.12*((t>1)&(t<3))
    for row in data[3]:
        if row['t']<4:
            for pad in row['per_pad']:pad['per_filter_contacts']=[]
    row=result(data,protocol)
    assert row['object']['maximum_lift_m']==pytest.approx(.12)
    assert row['object']['maximum_lift_after_acquisition_m']==0
    assert not row['object_gates']['physical_lift']


def test_probe_requires_actual_release_and_object_fall(candidate,protocol):
    data=make_trace(candidate);t=data[2]['t']
    data[2]['waffle_position'][t>=7.3,2]=.32
    row=result(data,protocol)
    assert row['object_gates']['synthetic_release_unloaded']
    assert not row['object_gates']['synthetic_release_fall']
    assert not row['passed']


@pytest.mark.parametrize('corruption',['camera','clock','attachment','pose_write'])
def test_reported_success_cannot_override_integrity_failure(candidate,protocol,corruption):
    data=make_trace(candidate)
    if corruption=='camera':data[4]['camera_projection']['intrinsics_px'][0][0]+=20
    elif corruption=='clock':data[2]['physics_t']+=.001
    elif corruption=='attachment':data[4]['object_dynamics']['attachments']=['hidden_fixed_joint']
    else:data[4]['object_dynamics']['pose_writes_after_initialization']=1
    row=result(data,protocol)
    assert all(row['object_gates'].values())
    assert not row['passed']


def test_direct_success_never_replaces_failed_dynamic_fixture(candidate,protocol):
    direct=result(make_trace(candidate,'replay'),protocol,case_id='direct')
    assert direct['passed'] and not direct['real_motion_physics_reproduced']
    data=make_trace(candidate,'dynamics');data[2]['q']+=.06
    dynamic=result(data,protocol,case_id='dynamic')
    assert not dynamic['common_gates']['joint_rmse']
    combined=audit.aggregate([direct,dynamic],['direct','dynamic'])
    assert not combined['passed']
    assert combined['failed_cases']==['dynamic']
    assert not combined['real_motion_physics_reproduced']


def test_7073_requires_retention_in_the_frozen_interval(candidate,protocol):
    data=make_trace(candidate,'dynamics')
    assert result(data,protocol,case_id='7073_dynamic')['passed']
    for row in data[3]:
        if 16.1<=row['t']<=16.6:
            for pad in row['per_pad']:pad['per_filter_contacts']=[]
    row=result(data,protocol,case_id='7073_dynamic')
    assert row['object_gates']['active_gel_pick']
    assert not row['object_gates']['active_gel_carry']
    assert row['object']['carry_active_gel']['longest_false_gap_s']>.3


def test_complete_coverage_is_separate_from_perfect_common_state_error(candidate,protocol):
    data=make_trace(candidate,'dynamics')
    for key in data[2]:data[2][key]=data[2][key][:-3]
    del data[3][-3:]
    row=result(data,protocol)
    assert row['common_gates']['joint_rmse']
    assert not row['common_gates']['complete_coverage']


def test_extra_time_cannot_extend_the_measured_task_deadline(candidate,protocol):
    data=make_trace(candidate,'dynamics')
    for key in data[5]:data[5][key]=data[5][key][:-3]
    row=result(data,protocol)
    assert row['common_gates']['joint_rmse']
    assert not row['common_gates']['complete_coverage']


def test_aggregate_keeps_missing_and_duplicate_denominators():
    passed={'case_id':'dynamic','mode':'dynamics','passed':True,'real_motion_physics_reproduced':True}
    assert audit.aggregate([passed],['dynamic','probe'])['missing_cases']==['probe']
    assert not audit.aggregate([passed,passed],['dynamic'])['passed']
    assert not audit.aggregate([],[])['passed']


def test_file_validation_records_invalid_run_and_config_hash(candidate,protocol,tmp_path):
    repo,cfg,trace,gel,run,_=make_trace(candidate)
    directory=tmp_path/'run';directory.mkdir()
    (directory/'run.json').write_text(json.dumps(run))
    config=directory/'effective_config.json';config.write_text(json.dumps(cfg))
    (directory/'gel_contact_trace.json').write_text(json.dumps(gel))
    np.savez(directory/'sim_trace.npz',**trace)
    good=audit.validate_run(directory,expected_config_sha256=audit.sha(config),case_id='probe',repo=repo)
    assert good['passed']
    bad=audit.validate_run(directory,expected_config_sha256='incorrect',case_id='probe',repo=repo)
    assert bad['status']=='invalid' and not bad['passed']
    assert bad['input_sha256']['effective_config.json']==audit.sha(config)
    missing=audit.validate_run(directory,directory/'absent_protocol.json',case_id='probe',repo=repo)
    assert missing['status']=='invalid' and missing['protocol_sha256'] is None
