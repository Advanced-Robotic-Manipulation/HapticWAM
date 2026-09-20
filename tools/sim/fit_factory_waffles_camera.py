#!/usr/bin/env python3
"""Register the measured table plane to the UR base and reconstruct visible layout.

Factory K and reported height are fixed. Only in-plane translation/yaw use robot
and mount/mat landmarks. Forearm volume centers are excluded from fitting.
This tool never imports a real driver or alters a recording.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phantom.sim.kinematics import link_transforms
from phantom.sim.camera import camera_projection
from fit_d435_camera_audit import project


def on_plane(camera, pixels, height):
    world = np.asarray(camera['world_from_cv'])
    inv_k = np.linalg.inv(camera_projection(camera).matrix)
    rays = np.c_[pixels, np.ones(len(pixels))] @ inv_k.T @ world[:3, :3].T
    return world[:3, 3] + rays * ((height-world[2, 3])/rays[:, 2])[:, None]


def rms(value):
    return float(np.sqrt(np.mean(np.sum(np.asarray(value)**2, axis=-1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'tests/fixtures/reference/d435_factory_calibration_20260908')
    parser.add_argument('--config-out', type=Path, default=ROOT/'configs/sim/waffles_d435_factory_20260908_r4.json')
    args = parser.parse_args()
    out = args.out
    audit = ROOT/'tests/fixtures/reference/d435_camera_audit_20260908'
    cfg = json.loads((ROOT/'configs/sim/waffles_measured_geometry_photo_frame_20260908_r2.json').read_text())
    profile_path = out/'d435_rgb_profile.json'
    profile = json.loads(profile_path.read_text())
    k = profile['intrinsics']
    if profile['device']['serial_number'] != '944622074411' or profile['profile'] != {
        'stream':'color','format':'RGB8','width':640,'height':480,'fps':15,'stream_index':0}:
        raise ValueError('Factory profile does not match the recorded rig RGB mode')
    if np.any(np.asarray(k['distortion_coefficients']) != 0):
        raise ValueError('Nonzero RealSense distortion requires convention-aware transfer')
    camera = copy.deepcopy(cfg['camera'])
    camera.update(fx=k['fx'], fy=k['fy'], cx=k['ppx'], cy=k['ppy'],
                  camera_model='opencv_pinhole', distortion_model='none', distortion_coefficients=[])
    camera_projection(camera)
    grid = json.loads((out/'grid_fit.json').read_text())
    if grid['inputs']['profile_sha256'] != hashlib.sha256(profile_path.read_bytes()).hexdigest():
        raise ValueError('Grid fit used another factory profile; refit the table first')
    np.testing.assert_allclose(grid['inputs']['K'],camera_projection(camera).matrix,rtol=0,atol=1e-9)
    expected_height = .963 + cfg['mat']['size'][2]
    if not np.isclose(grid['fits']['measured_height_all_points']['height_above_table_m'],expected_height,rtol=0,atol=1e-9):
        raise ValueError('Grid fit height differs from measured camera height plus mat thickness')
    # Fix measured height, preserving the table-fit camera normal; choose the
    # grid fit before UR registration, never by policy performance.
    g = np.asarray(grid['fits']['measured_height_all_points']['camera_from_grid'])
    mat_z = cfg['mat']['center'][2] + cfg['mat']['size'][2]/2
    annotations = json.loads((audit/'annotations.json').read_text())
    frames = json.loads((audit/'landmark_frames.json').read_text())
    points, pixels, fit, indices = [], [], [], []
    for frame in frames:
        index = int(frame['frame'][6:10])
        if str(index) not in annotations['observations']:
            continue
        points.append((link_transforms(frame['q'])['wrist_1_link'] @ [0,.0464,-.00177,1])[:3])
        pixels.append(annotations['observations'][str(index)]['wrist1_cap_px'])
        fit.append(index in annotations['fit_frames']); indices.append(index)
    points, pixels, fit = np.array(points), np.array(pixels), np.array(fit)
    # Refined real boundary and photo-aligned grid-line observations. Their
    # subpixel edge-fit residual is not measurement uncertainty: use 5 mm for
    # photo datum/base-centering and 3 px for hand-labelled wrist faces.
    right_edge = np.array([[462.58694,390],[465.33436,420],[468.53970,455],[470.37132,475]])
    aligned_row = np.array([[230,291.26205],[245,291.22268],[255,291.19644]])
    def state(parameters):
        yaw, x, y = parameters
        r = Rotation.from_euler('z', yaw).as_matrix()
        w = np.eye(4); w[:3,:3] = r @ g[:3,:3].T
        w[:3,3] = [x,y,mat_z+.963]
        c = copy.deepcopy(camera)
        c.update(world_from_cv=w.tolist(), position=w[:3,3].tolist(), target=(w[:3,3]+w[:3,2]).tolist())
        return c, r, w@g
    def residual(parameters):
        c, r, _ = state(parameters)
        edge = on_plane(c,right_edge,mat_z) @ r
        row = on_plane(c,aligned_row,mat_z) @ r
        return np.r_[((project(c,points[fit])-pixels[fit])/3.).ravel(),
                     (edge[:,0]+.150)/.005, (row[:,1]+.075)/.005]
    result = least_squares(residual, [.015,-.344,-.452], loss='soft_l1', f_scale=2, max_nfev=1000)
    if not result.success:
        raise RuntimeError(result.message)
    c, r, world_grid = state(result.x)
    c['source'] = 'Connected D435 factory RGB K; camera normal from 50mm table lattice at measured963mm above mat; UR registration from wrist caps and photo mount/mat constraints.'
    c['factory_profile'] = {'path':str(profile_path.relative_to(ROOT)), 'sha256':hashlib.sha256(profile_path.read_bytes()).hexdigest(), 'serial':profile['device']['serial_number'], 'sdk_distortion_model':k['distortion_model'], 'zero_coefficients_make_pinhole_equivalent':True}
    cfg['camera'] = c
    cfg['table'].update(grid_origin_xy=world_grid[:2,3].tolist(), grid_yaw=float(result.x[0]))
    # Visible grid extends past inherited table bounds. This support envelope
    # encloses visible evidence, and is not a tape measurement of table extent.
    cfg['table']['center'][1] = .10
    cfg['table']['size'][1] = 1.20
    cfg['robot_mount'].update(size=[.200,.150,.0125], yaw=float(result.x[0]))
    # The photo identifies plateX−100mm, mat edge another50mm away; its Y−75mm
    # edge follows one mat line. Fit rendering uses the85mm measured grid.
    mat_local_center = np.array([-.350,-.245,0.])
    mat_world_center = r @ mat_local_center
    cfg['mat'].update(center=[*mat_world_center[:2],cfg['mat']['center'][2]], size=[.400,.380,.003],
                      frame_model='local_planar',yaw=float(result.x[0]),
                      grid_origin_xy=[.100,.170])
    layout = json.loads((out/'layout_observations.json').read_text())
    intersections = layout['mat']['grid_intersections']
    grid_pixels = np.asarray(intersections['pixels'])
    grid_indices = np.asarray(intersections['grid_indices_uv'])
    grid_local = (on_plane(c,grid_pixels,mat_z) - cfg['mat']['center']) @ r
    x_phase = np.mean(grid_local[:,0]-grid_indices[:,0]*.085)
    # Count each horizontal row once, including the two clearer upper segments.
    upper_row = np.asarray(layout['mat']['photo_aligned_horizontal_grid_line']['pixels'])
    middle_row = np.array([[230,336.99879],[245,337.03687],[255,337.06226]])
    row_phase = [np.mean(grid_local[j:j+4,1]-grid_indices[j:j+4,1]*.085) for j in [0,4]]
    for row_pixels,index in [(upper_row,2),(middle_row,1)]:
        local = (on_plane(c,row_pixels,mat_z)-cfg['mat']['center']) @ r
        row_phase.append(np.mean(local[:,1])-index*.085)
    phase = (np.array([x_phase,np.mean(row_phase)])+.0425)%.085-.0425
    cfg['mat']['grid_origin_xy'] = phase.tolist()
    nearest_grid = np.round((grid_local[:,:2]-phase)/.085)*.085+phase
    grid_world = np.c_[nearest_grid,np.full(len(nearest_grid),.0015)] @ r.T + cfg['mat']['center']
    # Packet top-face labels were defined before this fit. Keep measured size;
    # fit only XY/yaw at the image-supported upright top height.
    packet_pixels = np.array([[246,433],[367,400],[359,377],[242,409]])
    packet_local = np.array([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*np.array(cfg['waffle']['size'])/2
    def packet_corners(p):
        return packet_local @ Rotation.from_euler('z',p[2]).as_matrix().T + [p[0],p[1],mat_z+.045]
    packet_fit = least_squares(lambda p:(project(c,packet_corners(p))-packet_pixels).ravel(),[*cfg['waffle']['center'][:2],cfg['waffle']['yaw']])
    if not packet_fit.success:
        raise RuntimeError(f'Packet layout fit failed: {packet_fit.message}')
    cfg['waffle'].update(center=[*packet_fit.x[:2],mat_z+.045],yaw=float(packet_fit.x[2]))
    # Bin opening labels come from a separate visible-layout audit. Fit position
    # and yaw only; all measured outside/opening/height dimensions remain fixed.
    box_pixels = np.asarray(layout['box']['opening_corners_px'])
    box_size = np.r_[cfg['bin']['opening_size'],cfg['bin']['outer_size'][2]]
    box_local = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]])*box_size/2
    box_local[:,2] = cfg['bin']['center'][2]+box_size[2]
    def bin_corners(p):
        return box_local @ Rotation.from_euler('z',p[2]).as_matrix().T + [p[0],p[1],0]
    box_fit = least_squares(lambda p:(project(c,bin_corners(p))-box_pixels).ravel(),[*cfg['bin']['center'][:2],result.x[0]])
    if not box_fit.success:
        raise RuntimeError(f'Box layout fit failed: {box_fit.message}')
    cfg['bin'].update(center=[*box_fit.x[:2],cfg['bin']['center'][2]],yaw=float(box_fit.x[2]))
    cfg['uncertainty']['geometry'] = 'Measured packet/bin sizes and pitches; photo-estimated mount thickness/base centering. Mat400mm width estimated from image; mat length380mm and table outer extent remain support-envelope estimates. Bin untapered proxy and support height remain approximate.'
    cfg['uncertainty']['camera'] = 'Factory RGB K read from matching D435 serial; zero SDK coefficients. Extrinsics estimated using metric grid and measuredheight; UR registration uses nominal CAD wrist faces and photo plate/mat alignment. Historical radiometry and rolling-shutter motion are not calibrated.'
    cfg['calibration'].update(revision='d435_factory_and_metric_grid_20260908_r4',status='factory_intrinsics_and_metric_image_registration_not_policy_qualified',
        frame_warning='MountX200/Y150mm, yaw aligned with table/mat by photo hypothesis; base centered on plate. TableZ−12.5mm and mat3mm retained estimates.',
        camera_warning='Factory K fixed. Camera tilt from metric table grid; noisy forearm volume-center correspondence excluded. No policy success claim.',
        validation='tests/fixtures/reference/d435_factory_calibration_20260908/registration.json')
    report = {
        'input_sha256': {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [profile_path,out/'grid_fit.json',out/'layout_observations.json',
                         audit/'annotations.json',audit/'landmark_frames.json',
                         out/'heldout_annotations.json',Path(__file__)]},
        'status':cfg['calibration']['status'], 'fit_parameters_yaw_x_y':result.x.tolist(),
        'camera_position_ur_m':c['position'], 'world_from_grid':world_grid.tolist(),
        'height_above_mat_m':.963,'height_above_table_m':.966,
        'tilt_from_vertical_deg':grid['fits']['measured_height_all_points']['inclination_from_table_normal_deg'],
        'registration_assumptions':['UR origin centered on150x200mmplate; plate and mat/table axes aligned in photos', '50mmgap at plateX−100mm; matline follows plateY−75mm', '3px wrist and5mm datum scales are judgment weights, not statistical confidence'],
        'forearm_exclusion':'Old local point is whole blue-volume bounding-box center; image labels are view-dependent silhouette/band. See forearm_landmark_audit.json.',
        'fit_wrist_rmse_px':rms(project(c,points[fit])-pixels[fit]),
        'withheld_timepoint_wrist_rmse_px':rms(project(c,points[~fit])-pixels[~fit]),
        'wrist_landmarks':[{'frame':int(i),'used_in_fit':bool(f),'observed_px':p.tolist(),'predicted_px':v.tolist(),'error_px':float(np.linalg.norm(p-v))} for i,f,p,v in zip(indices,fit,pixels,project(c,points))],
        'mat_right_edge_observations_px':right_edge.tolist(),'aligned_mat_row_observations_px':aligned_row.tolist(),
        'mat_gap_implied_mm':((-on_plane(c,right_edge,mat_z)@r)[:,0]-.100).tolist(),
        'aligned_mat_line_plate_y_residual_mm':((on_plane(c,aligned_row,mat_z)@r)[:,1]+.075).tolist(),
        'mat_grid_phase_local_m':phase.tolist(),'mat_grid_intersection_rmse_px':rms(project(c,grid_world)-grid_pixels),
        'mat_grid_caveat':'Measured85mm pitch preserved; image/table pose implies82–83mm, so phase fitting leaves2–3percent scale residual.',
        'packet_top_corner_rmse_px':rms(project(c,packet_corners(packet_fit.x))-packet_pixels),
        'packet_top_observations_px':packet_pixels.tolist(),'packet_top_predictions_px':project(c,packet_corners(packet_fit.x)).tolist(),
        'bin_opening_corner_rmse_px':rms(project(c,bin_corners(box_fit.x))-box_pixels),
        'bin_opening_observations_px':box_pixels.tolist(),'bin_opening_predictions_px':project(c,bin_corners(box_fit.x)).tolist(),
        'bin_fit_caveat':'Approximate rounded opening corners with3px judgment uncertainty; measured360x260mmopening fixed, onlyXY/yaw fitted. Cavity/rim/taper and episode-specific box movement remain.'
    }
    report['mat_gap_implied_mm'] = (1000*np.array(report['mat_gap_implied_mm'])).tolist()
    report['aligned_mat_line_plate_y_residual_mm'] = (1000*np.array(report['aligned_mat_line_plate_y_residual_mm'])).tolist()
    heldout = json.loads((out/'heldout_annotations.json').read_text())
    report['heldout_episodes'] = []
    for episode in heldout['episodes']:
        rows=[]
        for frame in episode['frames']:
            name='wrist1_cap_px'; predicted=project(c,[frame['cad_world_points_ur_base_m'][name]])[0]; observed=np.array(frame[name])
            rows.append({'frame':frame['frame_file'],'camera_t_s':frame['camera_t_s'],'observed_px':observed.tolist(),'predicted_px':predicted.tolist(),'error_px':float(np.linalg.norm(predicted-observed))})
        report['heldout_episodes'].append({'episode':episode['episode'],'wrist_rmse_px':float(np.sqrt(np.mean([row['error_px']**2 for row in rows]))),'landmarks':rows})
    args.config_out.write_text(json.dumps(cfg,indent=2)+'\n')
    report['config_sha256'] = hashlib.sha256(args.config_out.read_bytes()).hexdigest()
    (out/'registration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key in ['fit_parameters_yaw_x_y','camera_position_ur_m','fit_wrist_rmse_px','withheld_timepoint_wrist_rmse_px','mat_gap_implied_mm','aligned_mat_line_plate_y_residual_mm','packet_top_corner_rmse_px','bin_opening_corner_rmse_px']},indent=2))
    print('Heldout wrist RMSE:',[(e['episode'],e['wrist_rmse_px']) for e in report['heldout_episodes']])

if __name__ == '__main__':
    main()
