#!/usr/bin/env python3
"""Independently validate completed articulated-gripper traces on CPU.

No physics, hardware or source recording is modified. Direct joint replay and
synthetic probes never qualify real-motion physics reproduction by themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from phantom.sim.gripper_articulation import (
    JOINT_MULTIPLIERS, MODEL, append_gripper_urdf, finger_joint_names,
)
from phantom.sim.gripper_visual import _origin, _rotation
from tools.sim.compare_replay import physics_clock_metrics, state_metrics, validate_trace
from tools.sim.evaluate_pick_place import sustained_onset

DEFAULT_PROTOCOL = REPO/'docs/results/w2l_gripper_build_20260908/validation_protocol.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(trace,key,shape):
    value=np.asarray(trace[key],float)
    if value.shape!=shape or not np.isfinite(value).all():
        raise ValueError(f'{key} must be finite with shape {shape}')
    return value


def checked_times(values,label,max_gap):
    t=np.asarray(values,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError(f'{label} times must be finite and strictly increasing')
    if np.diff(t).max()>max_gap+1e-9:
        raise ValueError(f'{label} has a sample gap greater than {max_gap}s')
    return t


def active_packet_contacts(rows,max_gap):
    t=checked_times([r['t'] for r in rows],'gel',max_gap)
    values=[]
    for row in rows:
        if len(row['per_pad'])!=2:
            raise ValueError('Gel trace must contain two independently sampled pads')
        pair=[]
        for side,pad in enumerate(row['per_pad']):
            labels=row['filter_paths'][side]
            if labels.count('/World/Waffle')!=1 or pad['filter_labels_match_columns'] is not True or pad['contact_filter_column_count']!=len(labels):
                raise ValueError('Gel trace needs explicit valid packet-filter coverage per pad')
            matches=[r for r in pad['per_filter_contacts'] if r.get('filter_path')=='/World/Waffle']
            if len(matches)>1:
                raise ValueError('Duplicate packet contact filter in gel trace')
            # The contact reader emits populated filters only. A missing entry
            # is a measured zero only because valid filter coverage is above.
            force=float(matches[0]['gel_compression_n']) if matches else 0.
            if not np.isfinite(force) or force<0:
                raise ValueError('Active packet compression must be finite and nonnegative')
            pair.append(force)
        values.append(pair)
    return t,np.asarray(values)


def interval_stats(t,condition,start,end,max_gap):
    """Sample fraction plus bounded ZOH false gaps over full requested coverage."""
    condition=np.asarray(condition,bool)
    if t[0]>start or t[-1]<end-max_gap:
        raise ValueError(f'Trace does not cover requested interval [{start}, {end}]')
    selected=(t>=start)&(t<=end)
    if selected.sum()<3:
        raise ValueError('Insufficient samples in diagnostic interval')
    points=np.r_[start,t[(t>start)&(t<end)],end]
    index=np.searchsorted(t,points[:-1],side='right')-1
    index=np.maximum(index,0)
    duration=np.diff(points)
    run=longest=0.
    for yes,dt in zip(condition[index],duration):
        run=0. if yes else run+float(dt)
        longest=max(longest,run)
    return {'sample_fraction':float(condition[selected].mean()),'samples':int(selected.sum()),'longest_false_gap_s':longest,'interval_s':[start,end]}


def _fk_function(repo,cfg):
    root=ET.Element('robot',name='audit');ET.SubElement(root,'link',name='tool0')
    append_gripper_urdf(root,repo,cfg)
    pending=list(root.findall('joint'));ready={'tool0'};ordered=[]
    while pending:
        progress=False
        for joint in pending[:]:
            parent,child=joint.find('parent').get('link'),joint.find('child').get('link')
            if parent not in ready:continue
            axis=None if joint.get('type')=='fixed' else np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            ordered.append((parent,child,_origin(joint.find('origin')),joint.get('name'),axis))
            ready.add(child);pending.remove(joint);progress=True
        if not progress:raise ValueError('Unresolved audit FK tree')
    names=finger_joint_names(cfg)
    def fk(values):
        positions=dict(zip(names,values));frames={'tool0':np.eye(4)}
        for parent,child,origin,name,axis in ordered:
            frames[child]=frames[parent] @ origin @ (_rotation(axis,positions[name]) if axis is not None else np.eye(4))
        return [frames['left_pad'],frames['right_pad']]
    return fk


def mechanics_metrics(trace,run,cfg,protocol,repo):
    n=len(trace['t']);names=list(finger_joint_names(cfg))
    if run.get('finger_joint_names')!=names or run.get('finger_joint_units')!='radians' or run.get('gripper_model')!=MODEL or run.get('gripper_feedback_source')!='measured master joint angle':
        raise ValueError('Run does not declare the expected articulated joint names/radian units/model')
    q=array(trace,'finger_q',(n,len(names)))
    array(trace,'target_finger_q',q.shape)
    tcp=array(trace,'tcp',(n,6))
    pads=array(trace,'pad_position',(n,2,3));quats=array(trace,'pad_orientation_wxyz',(n,2,4))
    if np.min(np.linalg.norm(quats,axis=-1))<1e-6:
        raise ValueError('Pad orientation contains zero quaternion')
    multipliers=np.asarray(list(JOINT_MULTIPLIERS.values()))
    residual=q-q[:,0,None]*multipliers
    tool_rotation=Rotation.from_rotvec(tcp[:,3:]).as_matrix()
    offset=np.asarray(protocol['kinematics']['tool_tcp_offset_m'])
    tool_position=tcp[:,:3]-np.einsum('nij,j->ni',tool_rotation,offset)
    fk=_fk_function(repo,cfg)
    position_error=[];angle_error=[]
    for i in range(n):
        local=fk(q[i])
        for side in range(2):
            expected_position=tool_position[i]+tool_rotation[i]@local[side][:3,3]
            expected_rotation=tool_rotation[i]@local[side][:3,:3]
            measured_rotation=Rotation.from_quat(quats[i,side,[1,2,3,0]]).as_matrix()
            position_error.append(np.linalg.norm(pads[i,side]-expected_position))
            angle_error.append(Rotation.from_matrix(expected_rotation.T@measured_rotation).magnitude())
    maximum=float(np.max(np.abs(residual[:,1:])))
    pmax=float(np.max(position_error))
    return {'coupling_max_abs_rad':maximum,'coupling_per_joint_max_abs_rad':dict(zip(names[1:],np.max(abs(residual[:,1:]),axis=0).tolist())),
            'pad_fk_translation_max_m':pmax,'pad_fk_translation_rmse_m':float(np.sqrt(np.mean(np.square(position_error)))),
            'pad_fk_rotation_max_rad':float(np.max(angle_error)),'pad_fk_rotation_is_diagnostic_only':True,
            'gates':{'joint_coupling':maximum<=protocol['kinematics']['joint_coupling_max_abs_rad'],
                     'pad_world_fk':pmax<=protocol['kinematics']['pad_fk_max_translation_m']},
            'fk_definition':'actual six joint positions; actual tool0 derived from measured sim TCP minus rotated 0.18 m offset; both pad rigid bodies compared separately'}


def camera_readback_matches(run,cfg):
    """Independently compare saved lens readback, not just a reported flag."""
    camera=cfg['camera'];readback=run.get('camera_projection',{})
    expected=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1.]])
    actual=np.asarray(readback.get('intrinsics_px',[]),float)
    return (readback.get('readback_matches_config') is True
            and readback.get('resolution')==camera['resolution']
            and actual.shape==(3,3) and np.isfinite(actual).all()
            and np.allclose(actual,expected,rtol=0,atol=1e-5))


def evaluate(trace,gel_rows,run,cfg,protocol,*,reference=None,case_id='',repo=REPO):
    mode=run['mode']
    if mode not in ('replay','dynamics','contact_probe'):
        raise ValueError('Only completed measured replay/dynamics or synthetic contact_probe is supported')
    k=protocol['kinematics'];p=protocol['pick']
    validate_trace(trace,'simulation')
    t=checked_times(trace['t'],'physics',k['max_sample_gap_s']);n=len(t)
    obj=array(trace,'waffle_position',(n,3));tcp=array(trace,'tcp',(n,6))
    body=array(trace,'pad_packet_normal_force',(n,2))
    if np.any(body<0):raise ValueError('Pad-body packet normal force cannot be negative')
    gt,gel=active_packet_contacts(gel_rows,k['max_sample_gap_s'])
    if gt.shape!=t.shape or np.max(abs(gt-t))>k['physics_clock_max_abs_error_s']:
        raise ValueError('Gel and physics states must use the same sampled timestamps')
    active=np.all(gel>p['bilateral_active_gel_force_n'],axis=1)
    body_bilateral=np.all(body>p['bilateral_active_gel_force_n'],axis=1)
    mechanics=mechanics_metrics(trace,run,cfg,protocol,repo)
    clock=physics_clock_metrics(trace)
    dynamics=run.get('object_dynamics',{})
    common={'camera_readback':camera_readback_matches(run,cfg),
            'physics_clock':clock.get('passed') is True and clock.get('max_abs_error_s',float('inf'))<=k['physics_clock_max_abs_error_s'],
            'free_dynamic_object':dynamics.get('rigid_body_dynamic') is True and dynamics.get('kinematic') is False and dynamics.get('attachments')==[] and dynamics.get('pose_writes_after_initialization')==0,
            **mechanics['gates']}
    rise=obj[:,2]-obj[0,2]
    acquisition=sustained_onset(gt,active,p['bilateral_acquisition_hold_s'])
    lift=sustained_onset(t,rise>p['lift_onset_m'],p['lift_hold_s'],after=acquisition if acquisition is not None else float('inf'))
    post_acquisition_peak=float(rise[t>=acquisition].max()) if acquisition is not None else None
    object_metrics={'initial_position_m':obj[0].tolist(),'maximum_lift_m':float(rise.max()),'lift_onset_s':lift,
                    'active_gel_acquisition_onset_s':acquisition,'pad_body_acquisition_onset_s':sustained_onset(t,body_bilateral,p['bilateral_acquisition_hold_s']),
                    'active_gel_peak_n':gel.max(0).tolist(),'pad_body_peak_n':body.max(0).tolist(),
                    'maximum_lift_after_acquisition_m':post_acquisition_peak,
                    'active_gel_is_separate_from_pad_body_contact':True,
                    'temporal_scope':'sampled current-step forces; sub-frame contact interruptions are unobserved'}
    object_gates={};state=None
    if mode=='contact_probe':
        s=protocol['synthetic_probe'];duration=s['duration_s']
        hold=(t>=s['hold_interval_s'][0])&(t<=s['hold_interval_s'][1])
        if hold.sum()<3:raise ValueError('Probe does not cover hold interval')
        retention=interval_stats(gt,active,*s['hold_interval_s'],k['max_sample_gap_s'])
        release=interval_stats(gt,active,*s['release_interval_s'],k['max_sample_gap_s'])
        hold_lift=float(np.median(rise[hold]))
        baseline=float(np.median(obj[hold,2]-tcp[hold,2]))
        relative_fall=baseline-float(obj[-1,2]-tcp[-1,2])
        object_metrics.update(hold_median_lift_m=hold_lift,hold_active_gel=retention,release_active_gel=release,release_relative_vertical_fall_m=relative_fall)
        object_gates={'synthetic_hold_lift':hold_lift>=s['minimum_hold_median_lift_m'],
                      'synthetic_retention':retention['sample_fraction']>=s['bilateral_active_hold_fraction_min'],
                      'synthetic_release_unloaded':release['sample_fraction']==0.,
                      'synthetic_release_fall':relative_fall>=s['minimum_world_vertical_object_fall_relative_to_tcp_m']}
    else:
        if reference is None:raise ValueError('Measured replay validation requires reference/replay.npz')
        state=state_metrics(reference,trace);duration=float(reference['t'][-1])
        prefix='direct' if mode=='replay' else 'dynamics'
        common.update(joint_rmse=state['joint_error_rad']['rmse_all']<=k[prefix+'_joint_rmse_rad'],
                      tcp_rmse=state['tcp_translation']['rmse']/1000<=k[prefix+'_tcp_rmse_m'])
        if mode=='replay':common['rotation_rmse']=state['tcp_rotation_geodesic']['rmse']<=k['direct_rotation_rmse_deg']
        object_gates={'active_gel_pick':acquisition is not None,'physical_lift':lift is not None and post_acquisition_peak>=p['minimum_peak_lift_m']}
        if '7073' in case_id or '1788867073' in run.get('episode',''):
            c=protocol['carry_7073'];start,end=c['interval_export_s']
            carry=(t>=start)&(t<=end)
            retention=interval_stats(gt,active,start,end,k['max_sample_gap_s'])
            if carry.sum()<3:raise ValueError('Insufficient physics samples in7073 carry interval')
            relative=Rotation.from_rotvec(tcp[carry,3:]).inv().apply(obj[carry]-tcp[carry,:3])
            anchor=np.median(relative[:min(5,len(relative))],axis=0)
            slip=np.linalg.norm(relative-anchor,axis=1)
            slip95,slipmax=float(np.quantile(slip,.95)),float(slip.max())
            object_metrics.update(carry_active_gel=retention,carry_tcp_relative_slip_p95_m=slip95,carry_tcp_relative_slip_max_m=slipmax)
            object_gates.update(active_gel_carry=retention['sample_fraction']>=c['bilateral_active_fraction_min'] and retention['longest_false_gap_s']<=c['maximum_bilateral_gap_s'],
                                carry_retention=slip95<=c['tcp_relative_slip_p95_m'] and slipmax<=c['tcp_relative_slip_max_m'])
    common['complete_coverage']=bool(abs(t[0])<=k['coverage_start_tolerance_s'] and abs(t[-1]-duration)<=k['coverage_end_tolerance_s'] and abs(gt[0])<=k['coverage_start_tolerance_s'] and abs(gt[-1]-duration)<=k['coverage_end_tolerance_s'])
    common={key:bool(value) for key,value in common.items()};object_gates={key:bool(value) for key,value in object_gates.items()}
    passed=all(common.values()) and all(object_gates.values())
    return {'schema_version':1,'case_id':case_id,'mode':mode,'status':'passed' if passed else 'failed','passed':passed,
            'common_gates':common,'object_gates':object_gates,'mechanics':mechanics,'state':state,'physics_clock':clock,'object':object_metrics,
            'real_motion_physics_reproduced':mode=='dynamics' and passed,'synthetic_probe_is_real_validation':False,
            'direct_replay_is_physics_validation':False,'registration_or_hardware_qualified':False,'placement_claim':False,
            'camera_scope':'Saved lens schema readback only; not rendered landmark or real-camera registration validation'}


def aggregate(results,required_cases):
    by_id={row['case_id']:row for row in results}
    duplicates=len(by_id)!=len(results)
    missing=[name for name in required_cases if name not in by_id]
    extra=[name for name in by_id if name not in required_cases]
    dynamics=[r for r in results if r.get('mode')=='dynamics']
    passed=bool(not duplicates and not missing and not extra and dynamics and all(r.get('passed') is True for r in results))
    return {'passed':passed,'required_cases':required_cases,'missing_cases':missing,'unexpected_cases':extra,'duplicate_case_ids':duplicates,
            'dynamics_cases':len(dynamics),'failed_cases':[r['case_id'] for r in results if r.get('passed') is not True],
            'real_motion_physics_reproduced':passed and all(r.get('real_motion_physics_reproduced') is True for r in dynamics),
            'registration_or_hardware_qualified':False,'direct_or_probe_can_replace_failed_dynamics':False}


def validate_run(directory,protocol_path=DEFAULT_PROTOCOL,*,reference_path=None,case_id='',expected_config_sha256=None,repo=REPO):
    directory=Path(directory);protocol_path=Path(protocol_path)
    paths={name:directory/name for name in ['run.json','effective_config.json','sim_trace.npz','gel_contact_trace.json']}
    hashes={name:sha(path) for name,path in paths.items() if path.is_file()}
    if (directory/'launch.log').is_file():hashes['launch.log']=sha(directory/'launch.log')
    dependencies={str(path.relative_to(REPO)):sha(path) for path in [
        REPO/'phantom/sim/gripper_articulation.py',REPO/'phantom/sim/gripper_visual.py',
        REPO/'tools/sim/compare_replay.py',REPO/'tools/sim/evaluate_pick_place.py']}
    try:
        if expected_config_sha256 is not None and hashes.get('effective_config.json')!=expected_config_sha256:
            raise ValueError('Effective config differs from the predeclared candidate hash')
        protocol=json.loads(protocol_path.read_text());run=json.loads(paths['run.json'].read_text());cfg=json.loads(paths['effective_config.json'].read_text())
        for path in [Path(repo)/'assets/sim/robotiq/robotiq_2f85.urdf',
                     Path(repo)/cfg['gripper']['articulation'].get('geometry_manifest','assets/sim/dmtac_w2l/geometry.json')]:
            dependencies[str(path)]=sha(path)
        with np.load(paths['sim_trace.npz'],allow_pickle=False) as data:trace={key:data[key] for key in data.files}
        reference=None
        if reference_path is not None:
            with np.load(reference_path,allow_pickle=False) as data:reference={key:data[key] for key in data.files}
            hashes['reference.npz']=sha(reference_path)
        result=evaluate(trace,json.loads(paths['gel_contact_trace.json'].read_text()),run,cfg,protocol,reference=reference,case_id=case_id,repo=repo)
    except (ValueError,KeyError,TypeError,OSError,IndexError) as error:
        result={'schema_version':1,'case_id':case_id,'status':'invalid','passed':False,'error':str(error),'real_motion_physics_reproduced':False,'registration_or_hardware_qualified':False}
    result.update(input_sha256=hashes,protocol_sha256=sha(protocol_path) if protocol_path.is_file() else None,validator_sha256=sha(__file__),validator_dependency_sha256=dependencies,directory=str(directory),expected_config_sha256=expected_config_sha256)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--case',required=True);p.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    p.add_argument('--expected-config-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=validate_run(a.run,a.protocol,reference_path=a.reference,case_id=a.case,expected_config_sha256=a.expected_config_sha256)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'case':a.case,'status':result['status'],'passed':result['passed'],'output':str(a.output)}))
    return 0 if result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
