#!/usr/bin/env python3
"""CPU aperture profile of the actual mounted gel planes, including splayed jaws.

This diagnostic does not relax the parallel-jaw guard in gripper_probe, move a
body, or claim that a plane extrapolation is a finite contact clearance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from phantom.sim.gripper_articulation import (
    coupling_residuals,
    finger_joint_names,
    joint_targets,
    link_transforms_from_joint_positions,
    load_sensor_geometry,
)
from phantom.sim.gripper_visual import _origin


def _unit(vector, label):
    norm=float(np.linalg.norm(vector))
    if norm<1e-10:
        raise ValueError(f'Undefined {label}; opposing/degenerate pad axes')
    return vector/norm


def _transform(value):
    value=np.asarray(value,float)
    if (value.shape!=(4,4) or not np.isfinite(value).all()
            or not np.allclose(value[3],[0,0,0,1],atol=1e-10)
            or not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-8)
            or not np.isclose(np.linalg.det(value[:3,:3]),1,atol=1e-8)):
        raise ValueError('Each pad transform must be a finite proper rigid transform')
    return value


def _inside_triangles(point, triangles):
    """Closed union of finite planar STL triangles; no convex-hull filling."""
    triangles=np.asarray(triangles,float)
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    cross=lambda u,v:u[:,0]*v[:,1]-u[:,1]*v[:,0]
    det=cross(b-a,c-a);valid=abs(det)>1e-18
    if not valid.any():return False
    u=cross(np.broadcast_to(point,a.shape)-a,c-a)[valid]/det[valid]
    v=cross(b-a,np.broadcast_to(point,a.shape)-a)[valid]/det[valid]
    return bool(np.any((u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)))


def inspect_surfaces(left, right, *, front_x_m, bounds_yz_m,
                     active_size_yz_m, active_center_yz_m=(0,0),
                     depths_m=None, front_triangles_yz_m=None):
    """Measure along a rigid-transform-invariant common closing/distal frame.

    Both surface frames use the aligned pad-link convention: left inward is-X,
    right inward is+X, and +Z is distal. front_x_m and bounds are side mappings.
    Bounds are [minimumYZ,maximumYZ]. Optional planar triangles make the finite
    footprint test exact to the supplied tessellation, rather than a bbox test.
    """
    frames={'left':_transform(left),'right':_transform(right)}
    left,right=frames['left'],frames['right']
    active=np.asarray(active_size_yz_m,float);center=np.asarray(active_center_yz_m,float)
    if active.shape!=(2,) or center.shape!=(2,) or not np.isfinite(np.r_[active,center]).all() or np.any(active<=0):
        raise ValueError('Active size/center require finite YZ values and positive dimensions')
    faces={};bounds={}
    for side,frame in frames.items():
        offset=float(front_x_m[side]);bound=np.asarray(bounds_yz_m[side],float)
        if not np.isfinite(offset) or bound.shape!=(2,2) or not np.isfinite(bound).all() or np.any(bound[1]<=bound[0]):
            raise ValueError('Finite front plane and increasing YZ bounds are required')
        faces[side]=frame[:3,3]+frame[:3,0]*offset;bounds[side]=bound
    x=_unit(left[:3,0]+right[:3,0],'closing-axis bisector')
    distal=left[:3,2]+right[:3,2]
    z=_unit(distal-x*np.dot(x,distal),'distal-axis bisector')
    y=np.cross(z,x);rotation=np.column_stack((x,y,z))
    origin=(faces['left']+faces['right'])/2
    if depths_m is None:
        # Common coordinates, not the separate tilted pads' material arclength.
        lo=max(bound[0,1]for bound in bounds.values())
        hi=min(bound[1,1]for bound in bounds.values())
        depths_m={'full_gel_proximal':lo,'active_proximal':center[1]-active[1]/2,
                  'center':center[1],'active_distal':center[1]+active[1]/2,'full_gel_distal':hi}
    result=[]
    for name,depth in depths_m.items():
        depth=float(depth)
        if not np.isfinite(depth):raise ValueError('Depths must be finite')
        line_origin=origin+z*depth+y*center[0]
        samples={};intersections={};parameters={}
        for side,frame in frames.items():
            normal=frame[:3,0]
            denominator=float(np.dot(normal,x))
            if abs(denominator)<1e-8:
                raise ValueError('Closing-axis line is parallel to a gel plane')
            parameter=float(np.dot(normal,faces[side]-line_origin)/denominator)
            point=line_origin+parameter*x
            local=frame[:3,:3].T@(point-frame[:3,3]);local_yz=local[1:]
            in_bounds=bool(np.all(local_yz>=bounds[side][0]-1e-10) and np.all(local_yz<=bounds[side][1]+1e-10))
            in_front=None
            if front_triangles_yz_m is not None:
                in_front=_inside_triangles(local_yz,front_triangles_yz_m[side])
            in_active=bool(np.sum(((local_yz-center)/(active/2))**2)<=1+1e-10)
            samples[side]={'intersection_m':point.tolist(),'local_yz_m':local_yz.tolist(),
                           'within_full_gel_bbox':in_bounds,'within_finite_planar_front':in_front,
                           'within_assumed_active_ellipse':in_active}
            intersections[side]=point;parameters[side]=parameter
        delta=intersections['left']-intersections['right']
        signed_gap=parameters['left']-parameters['right']
        finite=(None if front_triangles_yz_m is None else
                all(v['within_finite_planar_front']for v in samples.values()))
        result.append({'name':str(name),'common_distal_depth_m':depth,
            'closing_axis_gap_m':signed_gap,
            'left_normal_separation_m':float(np.dot(delta,left[:3,0])),
            'right_normal_separation_m':float(np.dot(delta,right[:3,0])),
            'both_within_finite_planar_front':finite,
            'both_within_full_gel_bbox':all(v['within_full_gel_bbox']for v in samples.values()),
            'both_within_assumed_active_ellipse':all(v['within_assumed_active_ellipse']for v in samples.values()),
            'positive_opening_between_finite_fronts':bool(signed_gap>0 and finite) if finite is not None else None,
            'surfaces':samples})
    angle=lambda a,b:float(np.arccos(np.clip(np.dot(a,b),-1,1)))
    return {'common_frame':{'origin_m':origin.tolist(),'rotation_columns_closing_width_distal':rotation.tolist()},
        'relative_face_angle_rad':angle(left[:3,0],right[:3,0]),
        'relative_distal_axis_angle_rad':angle(left[:3,2],right[:3,2]),
        'samples':result,
        'metric_definitions':{
            'closing_axis':'Unit bisector of the two aligned surface +X axes; positive from right to left for nominal open jaws.',
            'distal_axis':'Bisector of the +Z axes, projected perpendicular to closing axis. Common origin is midpoint of gel front origins.',
            'depths':'Nominal gel/active YZ coordinates in the common frame, not each tilted surface arclength. Inspect finite-footprint flags.',
            'closing_axis_gap_m':'Signed distance between intersections of one closing-axis line and both front planes; negative means planes cross in reversed order.',
            'normal_separation_m':'Signed perpendicular distance from the opposite intersection to that face plane. Not a minimum finite-body clearance.',
            'finite_planar_front':'Point is within union of most-protruding planar visual-STL triangles. Curved/beveled gel edges and housing are not included.',
            'active_ellipse':'Configured optical-region hypothesis; it is not a calibrated active-area outline.',
            'limits':'Geometric measurements only: no force, object fit, mechanical hinge calibration or collision-clearance certification.'}}


def _front_triangles(path, front_x):
    raw=Path(path).read_bytes()
    if len(raw)<84:raise ValueError('A binary STL with front-plane triangles is required')
    count=struct.unpack_from('<I',raw,80)[0]
    if len(raw)!=84+50*count:raise ValueError('Expected the binary STL exported by the supplier CAD build')
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    triangles=np.frombuffer(raw,dtype=dtype,count=count,offset=84)['vertices'].astype(float)
    front=triangles[np.max(abs(triangles[:,:,0]-front_x),axis=1)<=1e-7,:,1:]
    if not len(front):raise ValueError('No triangles on the declared CAD front plane within0.1micrometre')
    return front


def inspect_config(repo, cfg, values):
    repo=Path(repo);geometry=load_sensor_geometry(repo,cfg)
    gel=next(p for p in geometry['parts']if p['role']=='gel')
    front_x=float(geometry['gel']['front_plane_x_m'])
    original=gel.get('origin',{'xyz':[0,0,0],'rpy':[0,0,0]})
    transform=_origin(ET.Element('origin',xyz=' '.join(map(str,original['xyz'])),rpy=' '.join(map(str,original['rpy']))))
    mirror=np.diag([-1.,-1.,1.,1.])
    links=link_transforms_from_joint_positions(repo,cfg,values)
    # Right mesh is mirrored inside an aligned pad link; undo that final mirror
    # when constructing its aligned surface axes, preserving arbitrary origin.
    surfaces={'left':links['left_pad']@transform,'right':links['right_pad']@mirror@transform@mirror}
    bounds=np.asarray([gel['bounding_box_m']['min'][1:],gel['bounding_box_m']['max'][1:]],float)
    right_bounds=bounds.copy();right_bounds[:,0]=-bounds[::-1,0]
    triangles=_front_triangles(repo/gel['visual_mesh'],front_x)
    right_triangles=triangles.copy();right_triangles[:,:,0]*=-1
    active=cfg['gripper']['gel_geometry']
    report=inspect_surfaces(surfaces['left'],surfaces['right'],
        front_x_m={'left':front_x,'right':-front_x},bounds_yz_m={'left':bounds,'right':right_bounds},
        active_size_yz_m=active['active_size_yz_m'],active_center_yz_m=active.get('active_center_yz_m',[0,0]),
        front_triangles_yz_m={'left':triangles,'right':right_triangles})
    report['actual_joint_positions_rad']=np.asarray(values,float).tolist()
    report['joint_names']=list(finger_joint_names(cfg));report['coupling_residuals_rad']=coupling_residuals(values,cfg)
    report['pad_surface_transforms']={k:v.tolist()for k,v in surfaces.items()}
    report['gel_geometry']={'front_plane_x_m':front_x,'full_bounds_yz_m':bounds.tolist(),
        'finite_planar_front_triangle_count':len(triangles),
        'finite_planar_front_bounds_yz_m':[triangles.min((0,1)).tolist(),triangles.max((0,1)).tolist()]}
    source_paths=[repo/'assets/sim/robotiq/robotiq_2f85.urdf',repo/gel['visual_mesh'],
        repo/cfg['gripper']['articulation'].get('geometry_manifest','assets/sim/dmtac_w2l/geometry.json'),
        repo/'phantom/sim/gripper_articulation.py',Path(__file__)]
    report['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in source_paths}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--pos',type=float,help='Raw encoder0..255, mapped through current configuration; default0')
    selection.add_argument('--finger-q',nargs=6,type=float,help='Six measured radians in reported finger_joint_names order; no mimic averaging')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if args.pos is not None and (not np.isfinite(args.pos) or not 0<=args.pos<=255):
        parser.error('--pos must be finite and in0..255')
    values=np.asarray(args.finger_q,float) if args.finger_q is not None else joint_targets((args.pos or 0)/255,cfg)
    report=inspect_config(ROOT,cfg,values)
    report['joint_state_source']='provided --finger-q; caller must establish recorded provenance' if args.finger_q is not None else 'mapped --pos through current uncalibrated motor model'
    report['input_sha256'][str(args.config.resolve())]=hashlib.sha256(args.config.read_bytes()).hexdigest()
    report['effective_config']=cfg
    report['configuration_status']=cfg.get('calibration',{}).get('status')
    output=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(output)
    else:print(output,end='')


if __name__=='__main__':main()
