#!/usr/bin/env python3
"""Reproduce camera-only registration proposals with successful physics frozen.

Inputs are Sept4 fit-episode RGB landmarks, pinned UR3 CAD and fixed physical
bin/packet coordinates. The old approximate TCP-pixel assumption is excluded.
Run from repository root; output contains uncertainty and competing metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import json
    from pathlib import Path

    import cv2
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from phantom.sim.kinematics import link_transforms
    from phantom.sim.geometry import bin_geometry

    E = Path("artifacts/isaac_waffles/evidence")
    frames = json.loads((E / "landmark_frames.json").read_text())
    old = json.loads((E / "camera_fit.json").read_text())
    cfg = json.loads(
        Path(
            "artifacts/isaac_waffles/tuning/ellipse75_stroke70/effective_config.json"
        ).read_text()
    )
    C = np.array(old["T_camera_cv_from_ur_base"])
    K = np.array(old["K"])
    ann = {
        0: [534, 157],
        55: [468, 219],
        83: [440, 265],
        111: [434, 273],
        139: [432, 276],
        167: [438, 274],
        195: [491, 225],
        223: [559, 135],
    }
    ann2 = {
        0: [494, 59],
        55: [441, 126],
        83: [418, 175],
        111: [411, 188],
        139: [412, 190],
        167: [416, 188],
        195: [454, 125],
        223: [505, 43],
    }
    pts = []
    pix = []
    sig = []
    labels = []
    for f in frames:
        i = int(f["frame"][6:10])
        trans = link_transforms(f["q"])
        if i in ann:
            pts.append((trans["wrist_1_link"] @ np.array([0, 0.0464, -0.00177, 1]))[:3])
            pix.append(ann[i])
            sig.append(3.0)
            labels.append(f"{i}:wrist1_face")
        if i in ann2:
            pts.append(
                (trans["forearm_link"] @ np.array([-0.21317, 0, -0.00528, 1]))[:3]
            )
            pix.append(ann2[i])
            sig.append(8.0)
            labels.append(f"{i}:forearm_cap_band")
    b = cfg["bin"]
    geometry = bin_geometry(b)
    bc, bs = geometry.center, geometry.outer_size
    binpts = np.array(
        [
            bc + np.array([x * bs[0] / 2, y * bs[1] / 2, bs[2]])
            for x, y in [(-1, 1), (1, 1), (1, -1), (-1, -1)]
        ]
    )
    binpix = np.array([[194, 92], [430, 92], [439, 249], [178, 249]])
    binpts = geometry.from_interior_frame(binpts)
    robotpts = np.array(pts)
    robotpix = np.array(pix)
    robotsig = np.array(sig)
    pts = np.r_[robotpts, binpts]
    pix = np.r_[robotpix, binpix]
    sig = np.r_[robotsig, [4] * 4]
    labels += ["bin_rear_left", "bin_rear_right", "bin_front_right", "bin_front_left"]

    def proj(p, points):
        pc = points @ Rotation.from_rotvec(p[:3]).as_matrix().T + p[3:6]
        uv = pc[:, :2] / pc[:, 2, None] * p[6] + [320, 240]
        return uv

    x0 = np.r_[Rotation.from_matrix(C[:3, :3]).as_rotvec(), C[:3, 3], K[0, 0]]

    def err(x):
        return np.r_[((proj(x, pts) - pix) / sig[:, None]).ravel(), (x[6] - 615) / 120]

    fit = least_squares(
        err,
        x0,
        bounds=(np.r_[[-np.inf] * 6, 400], np.r_[[np.inf] * 6, 1100]),
        loss="soft_l1",
        f_scale=2,
        max_nfev=2000,
    )

    def report(x):
        pred = proj(x, pts)
        errs = np.linalg.norm(pred - pix, axis=1)
        bp = pred[-4:].astype(np.float32)
        inter, _ = cv2.intersectConvexConvex(bp, binpix.astype(np.float32))
        a = cv2.contourArea(bp)
        b = cv2.contourArea(binpix.astype(np.float32))
        iou = float(inter / (a + b - inter))
        return {
            "params": x.tolist(),
            "wrist_rmse": float(np.sqrt(np.mean(errs[:16:2] ** 2))),
            "forearm_rmse": float(np.sqrt(np.mean(errs[1:16:2] ** 2))),
            "bin_corner_rmse": float(np.sqrt(np.mean(errs[-4:] ** 2))),
            "bin_iou": iou,
            "bin_projected": bp.tolist(),
            "errors": errs.tolist(),
        }

    o = cfg["waffle"]
    yaw = o["yaw"]
    ro = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    oh = np.array(o["size"]) / 2
    op = np.array(
        [
            np.array(o["center"]) + ro @ np.array([x * oh[0], y * oh[1], oh[2]])
            for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        ]
    )
    objpix = np.array(
        [[242, 409], [359, 377], [367, 400], [246, 433]], dtype=np.float32
    )
    white = json.loads((E / "gripper_centroid_estimate.json").read_text())[
        "observations"
    ]
    shellpts = []
    shellpix = []
    g = cfg["gripper"]
    Rz = lambda t: np.array(
        [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
    )
    for r in white:
        from phantom.sim.kinematics import forward_kinematics

        T = forward_kinematics(r["q"], np.zeros(6))
        A = T[:3, :3] @ Rz(g["yaw"])
        h = 0.006 + g["stroke"] / 2 * (1 - r["closure"] / 0.9) + 0.011
        S = np.array(
            [
                T[:3, 3] + A @ np.array([sign * h, 0, g["pad_center_z"] - 0.004])
                for sign in [-1, 1]
            ]
        )
        uv = proj(x0, S)
        S = S[np.argsort(uv[:, 1])]
        shellpts.extend(S)
        shellpix.extend(r["centroids"])
    shellpts = np.array(shellpts)
    shellpix = np.array(shellpix)
    fitw = least_squares(
        lambda x: np.r_[err(x), ((proj(x, shellpts) - shellpix) / 10).ravel()],
        fit.x,
        bounds=(np.r_[[-np.inf] * 6, 400], np.r_[[np.inf] * 6, 1100]),
        loss="soft_l1",
        f_scale=2,
        max_nfev=2000,
    )
    objordered = objpix[[3, 2, 1, 0]]
    fitbalanced = least_squares(
        lambda x: np.r_[
            err(x),
            ((proj(x, shellpts) - shellpix) / 10).ravel(),
            ((proj(x, op) - objordered) / 5).ravel(),
        ],
        fitw.x,
        bounds=(np.r_[[-np.inf] * 6, 400], np.r_[[np.inf] * 6, 1100]),
        loss="soft_l1",
        f_scale=2,
        max_nfev=2000,
    )
    sigbalanced = sig.copy()
    sigbalanced[-4:] = 5.0
    fitbalanced = least_squares(
        lambda x: np.r_[
            ((proj(x, pts) - pix) / sigbalanced[:, None]).ravel(),
            (x[6] - 615) / 120,
            ((proj(x, shellpts) - shellpix) / 10).ravel(),
            ((proj(x, op) - objordered) / 4).ravel(),
        ],
        fitbalanced.x,
        bounds=(np.r_[[-np.inf] * 6, 400], np.r_[[np.inf] * 6, 1100]),
        loss="soft_l1",
        f_scale=2,
        max_nfev=2000,
    )
    import copy
    import hashlib

    def extended_report(parameters):
        result = report(parameters)
        packet_projected = proj(parameters, op).astype(np.float32)
        overlap, _ = cv2.intersectConvexConvex(packet_projected, objpix)
        result["packet_top_quad_iou"] = float(
            overlap
            / (cv2.contourArea(packet_projected) + cv2.contourArea(objpix) - overlap)
        )
        result["packet_top_quad_corner_rmse_px"] = float(
            np.sqrt(np.mean(np.sum((packet_projected - objordered) ** 2, axis=1)))
        )
        result["white_shell_centroid_rmse_px"] = float(
            np.sqrt(
                np.mean(np.sum((proj(parameters, shellpts) - shellpix) ** 2, axis=1))
            )
        )
        weighted = np.r_[
            ((proj(parameters, pts) - pix) / sigbalanced[:, None]).ravel(),
            ((proj(parameters, shellpts) - shellpix) / 10).ravel(),
            ((proj(parameters, op) - objordered) / 4).ravel(),
        ]
        result["weighted_registration_rmse_sigma"] = float(
            np.sqrt(np.mean(weighted**2))
        )
        return result

    def camera_override(camera, parameters, source):
        result = copy.deepcopy(camera)
        cv_from_world = np.eye(4)
        cv_from_world[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
        cv_from_world[:3, 3] = parameters[3:6]
        world_from_cv = np.linalg.inv(cv_from_world)
        result.update(
            fx=float(parameters[6]),
            fy=float(parameters[6]),
            cx=320.0,
            cy=240.0,
            world_from_cv=world_from_cv.tolist(),
            position=world_from_cv[:3, 3].tolist(),
            target=(world_from_cv[:3, 3] + world_from_cv[:3, 2]).tolist(),
            source=source,
        )
        return result

    out = Path("artifacts/isaac_waffles/tuning/configs")
    frozen = {}
    written = []
    for task, source in [
        ("fit", "ellipse75_stroke70.json"),
        ("teleop", "teleop_ellipse75_stroke70.json"),
    ]:
        base = json.loads((out / source).read_text())
        physics = {k: v for k, v in base.items() if k != "camera"}
        frozen[task] = hashlib.sha256(
            json.dumps(physics, sort_keys=True).encode()
        ).hexdigest()
        for label, params in [("refined", fit.x), ("balanced", fitbalanced.x)]:
            candidate = copy.deepcopy(base)
            candidate["camera"] = camera_override(
                base["camera"],
                params,
                "camera_refinement_frozen_physics.json; fit Sept4 only, uncalibrated optics; "
                + label,
            )
            assert {k: v for k, v in candidate.items() if k != "camera"} == physics
            filename = out / f"camera_{label}_{task}.json"
            filename.write_text(json.dumps(candidate, indent=2) + "\n")
            written.append(str(filename))
    summary = {
        "status": "camera-only proposals with all non-camera settings frozen; not hardware calibration",
        "fit_episode": "ep_teacher_waffles_1788535016_005 (Sept4 fit subset only)",
        "excluded": "No held-out episodes and no former approximate TCP-pixel labels were used.",
        "fixed_physics_sha256": frozen,
        "candidates_written": written,
        "baseline": extended_report(x0),
        "cad_and_bin_only": extended_report(fit.x),
        "balanced": extended_report(fitbalanced.x),
        "observation_uncertainty_px": {
            "wrist1_CAD_face": 3,
            "forearm_cap_band": 8,
            "bin_rim": 5,
            "white_shell_centroid": 10,
            "packet_top_quad": 4,
        },
        "balanced_objective": "soft_l1 residuals normalized by the listed uncertainties; weak fx=fy615+/-120px prior, fixed(cx,cy)=(320,240), no distortion; optimize camera pose and one shared focal length only.",
        "robot_and_bin_observations": [
            {
                "name": n,
                "world_m": p.tolist(),
                "pixel": u.tolist(),
                "uncertainty_px": float(s),
            }
            for n, p, u, s in zip(labels, pts, pix, sigbalanced)
        ],
        "white_shell_observations": [
            {"world_m": p.tolist(), "pixel": u.tolist()}
            for p, u in zip(shellpts, shellpix)
        ],
        "packet_top_observations": [
            {"world_m": p.tolist(), "pixel": u.tolist()} for p, u in zip(op, objordered)
        ],
        "limitations": [
            "Forearm blue cap band visible center is view-dependent and not a precise point marker.",
            "White shell centroids depend on illumination, occlusion and visible surface; 10px uncertainty reflects this.",
            "Bin and packet quad IoUs refer to manually annotated projected quadrilaterals, not full segmentation masks.",
            "All errors are in-sample; test rendered held-out episodes separately.",
            "Tabletop+53mm in UR base is still an effective unmeasured parameter. Physics success and this image fit do not resolve actual base mounting/table height, tool contact geometry or camera calibration uniquely.",
            "The CAD+bin-only proposal worsens foreground registration; prefer balanced proposal for review.",
        ],
    }
    # Additional visible mat-side check: same fit image, excluded from fitting.
    mat = cfg["mat"]
    mat_center, mat_size = np.array(mat["center"]), np.array(mat["size"])
    mat_points = np.array(
        [
            mat_center + [x * mat_size[0] / 2, y * mat_size[1] / 2, mat_size[2] / 2]
            for x, y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ]
    )
    mat_observed = np.array(
        [[300.0, 220.0, 448.0], [400.0, 215.0, 458.0], [470.0, 211.0, 465.0]]
    )
    mat_report = {
        "status": "manual fit-frame side-edge check, not used in fitting; about5px uncertainty",
        "observations_row_left_right_px": mat_observed.tolist(),
    }
    for label, parameters in [("baseline", x0), ("balanced", fitbalanced.x)]:
        uv = proj(parameters, mat_points)
        predicted = np.array(
            [
                [
                    np.interp(row, uv[:2, 1][::-1], uv[:2, 0][::-1]),
                    np.interp(row, uv[2:, 1][::-1], uv[2:, 0][::-1]),
                ]
                for row in mat_observed[:, 0]
            ]
        )
        mat_report[label] = {
            "predicted_left_right_px": predicted.tolist(),
            "horizontal_edge_rmse_px": float(
                np.sqrt(np.mean((predicted - mat_observed[:, 1:]) ** 2))
            ),
        }
    summary["mat_side_edge_check"] = mat_report
    (E / "camera_refinement_frozen_physics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(E / "camera_refinement_frozen_physics.json")


if __name__ == "__main__":
    main()
