#!/usr/bin/env python3
"""Compare frozen camera with a fixed RGB-mode prior and a robot-landmark pose fit.

This is an image-registration diagnostic, not a per-device optical calibration.
Only camera pose is fitted; packet, bin and arm geometry remain fixed. Factory
intrinsics must replace the provisional prior before claiming calibrated FOV.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phantom.sim.kinematics import link_transforms  # noqa: E402
from phantom.sim.geometry import bin_geometry  # noqa: E402


def fov(camera):
    width, height = camera['resolution']
    return [math.degrees(math.atan(c/f)+math.atan((extent-c)/f))
            for c,f,extent in [(camera['cx'],camera['fx'],width),
                               (camera['cy'],camera['fy'],height)]]


def project(camera, xyz):
    inverse = np.linalg.inv(camera['world_from_cv'])
    points = np.array(xyz) @ inverse[:3,:3].T + inverse[:3,3]
    if np.any(points[:,2] <= 0):
        raise ValueError('Landmark behind camera')
    return points[:,:2]/points[:,2,None]*[camera['fx'],camera['fy']]+[camera['cx'],camera['cy']]


def camera_pose(camera, parameters):
    c = copy.deepcopy(camera)
    transform = np.eye(4)
    transform[:3,:3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    transform[:3,3] = parameters[3:6]
    world = np.linalg.inv(transform)
    c['world_from_cv'] = world.tolist()
    c['position'] = world[:3,3].tolist()
    c['target'] = (world[:3,3]+world[:3,2]).tolist()
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/sim/waffles_measured_geometry_photo_frame_20260908_r2.json')
    parser.add_argument('--out',type=Path,default=ROOT/'docs/results/d435_camera_audit_20260908')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(args.config.read_text())
    annotation=json.loads((args.out/'annotations.json').read_text())
    frames=json.loads((args.out/'landmark_frames.json').read_text())
    points,pixels,weights,fit_flags,names=[],[],[],[],[]
    for frame in frames:
        index=int(frame['frame'][6:10])
        if str(index) not in annotation['observations']:continue
        obs=annotation['observations'][str(index)]
        transforms=link_transforms(frame['q'])
        for name,link,local,sigma in [
            ('wrist1_cap_px','wrist_1_link',[0,.0464,-.00177,1],3.),
            ('forearm_cap_px','forearm_link',[-.21317,0,-.00528,1],8.),
        ]:
            points.append((transforms[link]@np.array(local))[:3])
            pixels.append(obs[name]);weights.append(sigma)
            fit_flags.append(index in annotation['fit_frames'])
            names.append(f'{index}:{name}')
    points,pixels,weights=np.array(points),np.array(pixels),np.array(weights)
    fit_flags=np.array(fit_flags)
    # Official RGB full-HD vertical FOV42deg; provisional4:3 profile preserves
    # vertical coverage and square pixels (SDK's narrower-aspect horizontal crop).
    # Neither this rounded specification nor the crop calculation is factory K.
    focal=240/math.tan(math.radians(42)/2)
    prior=copy.deepcopy(cfg['camera'])
    prior.update(fx=focal,fy=focal,cx=320.,cy=240.,camera_model='opencv_pinhole',
                 distortion_model='none',distortion_coefficients=[])
    prior['source']='D435 RGB42deg vertical product-spec prior with4:3 horizontal crop and square pixels; not queried factory intrinsics. Position/orientation retain R2 estimate.'
    inverse=np.linalg.inv(prior['world_from_cv'])
    initial=np.r_[Rotation.from_matrix(inverse[:3,:3]).as_rotvec(),inverse[:3,3]]
    mat_z=cfg['mat']['center'][2]+cfg['mat']['size'][2]/2
    def residual(parameters, height_sigma):
        camera=camera_pose(prior,parameters)
        try:
            result=((project(camera,points[fit_flags])-pixels[fit_flags])/weights[fit_flags,None]).ravel()
        except ValueError:
            return np.full(2*int(fit_flags.sum())+int(height_sigma is not None),1e6)
        if height_sigma is not None:
            result=np.r_[result,(camera['position'][2]-mat_z-.963)/height_sigma]
        return result
    result=least_squares(residual,initial,args=(.020,),loss='soft_l1',f_scale=2,max_nfev=1000)
    fitted=camera_pose(prior,result.x)
    fitted['source']='Fixed provisional RGB K; camera pose fitted to four timepoints of known UR CAD landmarks with963mm above-mat height prior (20mm judgment scale). Not hardware calibration.'
    free=least_squares(residual,result.x,args=(None,),loss='soft_l1',f_scale=2,max_nfev=1000)
    free_camera=camera_pose(prior,free.x)
    variants={'r2_fitted_optics':cfg['camera'],'rgb_fov_only':prior,'rgb_robot_pose_fit':fitted}
    b=bin_geometry(cfg['bin'])
    corners=np.array([b.center+[x*b.outer_size[0]/2,y*b.outer_size[1]/2,b.outer_size[2]] for x,y in [(-1,1),(1,1),(1,-1),(-1,-1)]])
    corners=b.from_interior_frame(corners)
    box_pixels=np.array(annotation['box_outer_rim_corners_px'])
    report={
        'status':'provisional_RGB_mode_prior_and_image_registration_not_factory_calibration',
        'exact_D435_intrinsics_available':False,
        'sources':{'rgb_spec':'https://www.realsenseai.com/products/stereo-depth-camera-d435/',
                   'mode_intrinsics_implementation':'https://github.com/realsenseai/librealsense/blob/master/src/ds/d400/d400-private.cpp'},
        'prior':{'vfov_deg':42.,'fx_fy_px':focal,'principal_point_assumed':[320,240],
                 'nominal_4by3_fov_deg':fov(prior),'interpretation':'Mode-specific estimate, not69x42 full-HD transplanted into4:3.',
                 'distortion':'zero assumed; actual coefficients unavailable'},
        'fit_frames':annotation['fit_frames'],'validation_frames':annotation['validation_frames'],
        'validation_scope':annotation['validation_scope'],
        'height_prior_sigma_m':.020,'pose_fit_converged':bool(result.success),
        'unconstrained_pose_height_above_mat_m':free_camera['position'][2]-mat_z,
        'variants':{},
    }
    for name,camera in variants.items():
        prediction=project(camera,points)
        error=np.linalg.norm(prediction-pixels,axis=1)
        bp=project(camera,corners)
        report['variants'][name]={
            'fov_deg':fov(camera),'camera_height_above_mat_m':camera['position'][2]-mat_z,
            'fit_landmark_rmse_px':float(np.sqrt(np.mean(error[fit_flags]**2))),
            'withheld_timepoint_landmark_rmse_px':float(np.sqrt(np.mean(error[~fit_flags]**2))),
            'wrist1_all_rmse_px':float(np.sqrt(np.mean(error[::2]**2))),
            'forearm_all_rmse_px':float(np.sqrt(np.mean(error[1::2]**2))),
            'fixed_bin_corner_rmse_px':float(np.sqrt(np.mean(np.sum((bp-box_pixels)**2,axis=1)))),
            'fixed_bin_projected_px':bp.tolist(),
            'landmarks':[{'name':n,'observed_px':o.tolist(),'predicted_px':v.tolist(),'error_px':float(e),'used_in_fit':bool(t)} for n,o,v,e,t in zip(names,pixels,prediction,error,fit_flags)],
        }
        if name!='r2_fitted_optics':
            candidate=copy.deepcopy(cfg);candidate['camera']=camera
            candidate['calibration']['revision']=f'd435_{name}_20260908_r3'
            candidate['calibration']['status']='camera_diagnostic_not_factory_calibrated_or_policy_qualified'
            candidate['calibration']['validation']='See docs/results/d435_camera_audit_20260908/camera_comparison.json; no policy success claim.'
            candidate['calibration']['camera_warning']='FactoryRGBintrinsics absent; principal point/distortion and nominalFOV remain assumptions. Pose fit is episode-specific; verify setup continuity.'
            (args.out/f'{name}.json').write_text(json.dumps(candidate,indent=2)+'\n')
    (args.out/'camera_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{i:v for i,v in data.items() if i not in ['landmarks','fixed_bin_projected_px']} for k,data in report['variants'].items()},indent=2))
    print('Unconstrained RGB-prior camera height above mat:',report['unconstrained_pose_height_above_mat_m'])

if __name__=='__main__':main()
