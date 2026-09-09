#!/usr/bin/env python3
"""Measure gripper-root orientation from visible metal screw faces, on CPU.

The fit uses precontact RGB and measured robot joints, never pickup outcomes or
sensor silhouettes. Its translation is conditional camera/UR registration, not
a measured lateral offset of the physical flange. No scene is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phantom.sim.kinematics import dh_frames
from tools.sim.fit_w2l_registration import nominal_frames

# Visible round cap centres, manually associated with CAD features and refined
# to image-only low-saturation bright components. Coordinates use original RGB.
INITIAL_PIXELS = np.array([
    [426.90909, 308.09091], [389.5, 336.21429],
    [393.05263, 274.05263], [374.9, 325.95],
    [364.09091, 221.63636], [327.5, 243.5],
    [380.05263157894734, 256.0], [330.625, 258.75],
])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose(value):
    transform = np.eye(4)
    transform[:3, 3] = value[:3]
    transform[:3, :3] = Rotation.from_rotvec(value[3:6]).as_matrix()
    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'docs/results/w2l_opening_width_20260909')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evidence = ROOT/'docs/results/teacher_scene_v10_20260908/pickup_registration_evidence'
    config_path = ROOT/'configs/sim/waffles_w2l_articulated_provisional_20260908.json'
    config = json.loads(config_path.read_text())
    frames = json.loads((evidence/'inputs.json').read_text())['frames'][:5]
    camera = config['camera']
    world_from_camera = np.array(camera['world_from_cv'])
    intrinsics = np.array([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1]])
    yaw = np.eye(4)
    yaw[:3, :3] = Rotation.from_euler('z', -np.pi/2+config['gripper']['yaw']).as_matrix()
    gain = .8/.9
    base_frames = [dh_frames(frame['arm_q'])[-1] @ yaw for frame in frames]
    links, points = [], []
    for side, sign in [('left', -1), ('right', 1)]:
        for suffix, coordinates in [
            ('outer_finger', [[13.5, 0, 0], [13.5, 6.11476047, 47.12495049]]),
            ('inner_knuckle', [[19.5, 0, 0], [13.5, 37.11594806, 43.45719785]]),
        ]:
            for point in coordinates:
                point = np.array(point)/1000
                point[0] *= sign
                links.append(side+'_'+suffix)
                points.append(point)

    def project(parameters, index, theta=None):
        frame = frames[index]
        chain = nominal_frames(gain*frame['gripper'][0] if theta is None else theta)
        result = []
        for link, point in zip(links, points):
            transform = base_frames[index] @ pose(parameters) @ chain[link]
            world = transform[:3, :3] @ point + transform[:3, 3]
            camera_point = world_from_camera[:3, :3].T @ (world-world_from_camera[:3, 3])
            uv = intrinsics @ camera_point
            result.append(uv[:2]/uv[2])
        return np.array(result)

    initial = np.r_[np.zeros(6), gain*frames[0]['gripper'][0]]
    first_fit = least_squares(
        lambda value: (project(value, 0, value[6])-INITIAL_PIXELS).ravel(), initial,
        bounds=(np.r_[np.full(3, -.05), np.full(3, -.5), -.1],
                np.r_[np.full(3, .05), np.full(3, .5), .6]),
        x_scale='jac', max_nfev=500,
    )
    predictions = project(first_fit.x, 0, first_fit.x[6])
    first = {
        'initial_parameters': initial.tolist(), 'fit_parameters': first_fit.x.tolist(),
        'observed': INITIAL_PIXELS.tolist(), 'initial_projection': project(initial, 0, initial[6]).tolist(),
        'fitted_projection': predictions.tolist(),
        'per_point_error_px': np.linalg.norm(predictions-INITIAL_PIXELS, axis=1).tolist(),
        'rms_px': float(np.sqrt(np.mean((predictions-INITIAL_PIXELS)**2))),
        'rms_definition': 'per coordinate; Euclidean point RMS is sqrt(2) times this value',
        'note': 'Precontact metal screw-face fit; no sensor mount or loaded outcome in objective',
    }
    (args.out/'screw_fit_diagnostic.json').write_text(json.dumps(first, indent=2)+'\n')

    observations, tracking, previous = [], [], None
    last = INITIAL_PIXELS.astype(np.float32).reshape(-1, 1, 2)
    for i, frame in enumerate(frames):
        image_path = evidence/frame['image']
        assert sha(image_path) == frame['image_sha256']
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fb = np.zeros(8)
        if previous is not None:
            options = {'winSize': (25, 25), 'maxLevel': 4,
                       'criteria': (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 50, .001)}
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, last, None, **options)
            back, status2, _ = cv2.calcOpticalFlowPyrLK(gray, previous, tracked, None, **options)
            if not status.all() or not status2.all():
                raise ValueError('A screw landmark could not be tracked')
            fb = np.linalg.norm(back-last, axis=2).ravel()
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = ((hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 160)).astype(np.uint8)
            _, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
            for j, point in enumerate(tracked[:, 0]):
                candidates = [(np.linalg.norm(center-point), center)
                              for center, area in zip(centers[1:], stats[1:, 4]) if 4 <= area <= 80]
                distance, center = min(candidates, key=lambda value: value[0])
                if distance < 2.5:
                    tracked[j, 0] = center
            last = tracked
        observed = last[:, 0].copy()
        predicted = project(first_fit.x, i)
        observations.append(observed)
        tracking.append({
            'frame_index': frame['frame_index'], 'POS': 255*frame['gripper'][0],
            'observed': observed.tolist(), 'prediction_frozen_frame0': predicted.tolist(),
            'forward_backward_px': fb.tolist(),
            'per_point_error_px': np.linalg.norm(predicted-observed, axis=1).tolist(),
            'rms_px': float(np.sqrt(np.mean((predicted-observed)**2))),
        })
        overlay = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for j, (actual, expected) in enumerate(zip(observed, predicted)):
            a, b = tuple(np.rint(2*actual).astype(int)), tuple(np.rint(2*expected).astype(int))
            cv2.circle(overlay, a, 5, (255, 0, 255), 1)
            cv2.drawMarker(overlay, b, (0, 255, 255), cv2.MARKER_CROSS, 11, 1)
            cv2.putText(overlay, str(j), (a[0]+6, a[1]), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.imwrite(str(args.out/f'screw_track_{frame["frame_index"]:03d}.png'), overlay)
        previous = gray
    (args.out/'screw_tracking_validation.json').write_text(json.dumps(tracking, indent=2)+'\n')

    train, withheld = [0, 2, 4], [1, 3]
    fit = least_squares(
        lambda value: np.concatenate([(project(value, i)-observations[i]).ravel() for i in train]),
        first_fit.x[:6], bounds=(np.r_[np.full(3, -.05), np.full(3, -.5)],
                                np.r_[np.full(3, .05), np.full(3, .5)]), x_scale='jac',
    )
    result = {
        'scope': 'Precontact screw-only constant root pose; no sensor or object outcome fit',
        'fit_frames': [frames[i]['frame_index'] for i in train],
        'withheld_frames': [frames[i]['frame_index'] for i in withheld],
        'root_delta_xyz_rotvec': fit.x.tolist(), 'fixed_theta_gain': gain,
        'per_frame_coordinate_rms_px': [float(np.sqrt(np.mean((project(fit.x, i)-observations[i])**2))) for i in range(5)],
        'warning': 'Translation is conditional camera/UR registration, not measured lateral flange offset',
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            config_path, evidence/'inputs.json', Path(__file__),
            ROOT/'assets/sim/robotiq/robotiq_2f85.urdf',
            ROOT/'docs/results/w2l_opening_width_20260909/nominal_face_features.json',
            *[evidence/frame['image'] for frame in frames],
        ]},
        'feature_assumptions': 'Outer/distal cap centres from CAD face loops at abs(X)=13.5 mm; proximal cap X=19.5 mm uses link envelope and is checked separately by exclusion/depth sensitivity',
    }
    (args.out/'screw_global_diagnostic.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['root_delta_xyz_rotvec', 'per_frame_coordinate_rms_px']}))


if __name__ == '__main__':
    main()
