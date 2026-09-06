#!/usr/bin/env python3
"""Reproduce the fit-episode white-shell centroid registration diagnostic.

Run from the repository root after prepare_waffles and fit_camera_landmarks.
Requires numpy/scipy/OpenCV. This estimates appearance parameters from RGB;
its systematic residuals and unknown shell geometry preclude interpreting it
as a calibration of physical contact geometry. No held-out data are loaded.
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

    from phantom.sim.kinematics import forward_kinematics

    root = Path("artifacts/isaac_waffles/evidence")
    frames = json.loads((root / "landmark_frames.json").read_text())
    a = np.load(root / "fit/ep_teacher_waffles_1788535016_005/replay.npz")
    cam = json.loads((root / "camera_fit.json").read_text())
    K = np.array(cam["K"])
    C = np.array(cam["T_camera_cv_from_ur_base"])
    # Hand ROIs separate genuine pad whites from package typography and bolt highlights.
    rois = {
        0: ((270, 295, 331, 362), (316, 379, 397, 438)),
        27: ((270, 300, 330, 365), (312, 382, 390, 439)),
        55: ((275, 326, 324, 382), (309, 403, 370, 445)),
        83: ((277, 345, 319, 393), (302, 419, 355, 454)),
        111: ((276, 350, 320, 399), (302, 418, 355, 458)),
        139: ((274, 363, 316, 412), (294, 417, 345, 456)),
        167: ((275, 371, 318, 415), (291, 414, 342, 450)),
        195: ((260, 345, 318, 397), (282, 395, 351, 440)),
        223: ((218, 268, 300, 317), (240, 323, 331, 376)),
    }
    records = []
    for f in frames:
        i = int(f["frame"][6:10])
        bgr = cv2.imread(str(root / f["frame"])).astype(int)
        hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
        mask = (
            (hsv[:, :, 1] < 75)
            & (hsv[:, :, 2] > 190)
            & (bgr.min(axis=2) > 160)
            & (bgr[:, :, 0] >= bgr[:, :, 2] - 8)
        )
        cent = []
        n = []
        for xmin, ymin, xmax, ymax in rois[i]:
            yy, xx = np.where(mask[ymin:ymax, xmin:xmax])
            cent.append([xx.mean() + xmin, yy.mean() + ymin])
            n.append(len(xx))
        g = float(
            a["native_gripper"][
                max(
                    0,
                    np.searchsorted(a["native_gripper_t"], f["time"], side="right") - 1,
                ),
                0,
            ]
        )
        records.append(
            {
                "frame": f["frame"],
                "q": f["q"],
                "closure": g,
                "centroids": cent,
                "mask_pixels": n,
            }
        )

    def predicted(p, r):
        yaw, z = p[:2]
        h = 0.006 + 0.085 / 2 * (1 - np.clip(r["closure"] / 0.9, 0, 1))
        local = np.array(
            [
                [h * np.cos(yaw), h * np.sin(yaw), z, 1],
                [-h * np.cos(yaw), -h * np.sin(yaw), z, 1],
            ]
        )
        world = local @ forward_kinematics(r["q"], np.zeros(6)).T
        pc = world @ C.T
        uv = pc[:, :3] @ K.T
        uv = uv[:, :2] / uv[:, 2, None]
        # Match label order top-left vs bottom-right consistently; current yaw around0.
        return uv[np.argsort(uv[:, 1])]

    def residual(p):
        return np.concatenate(
            [(predicted(p, r) - r["centroids"]).ravel() for r in records]
        )

    fit = least_squares(
        residual,
        [0, 0.1725],
        bounds=([-0.6, 0.12], [0.6, 0.23]),
        loss="soft_l1",
        f_scale=4,
    )

    def metrics(par, subset=records):
        errors = np.array(
            [np.linalg.norm(predicted(par, r) - r["centroids"], axis=1) for r in subset]
        )
        return {
            "point_rmse_px": float(np.sqrt(np.mean(errors**2))),
            "max_point_error_px": float(errors.max()),
            "per_frame_point_errors_px": errors.tolist(),
        }

    fits = {}
    for name, subset in [
        ("all", records),
        ("open_only", records[:5]),
        ("later_closure", records[5:]),
    ]:
        rfun = lambda p, subset=subset: np.concatenate(
            [(predicted(p, r) - r["centroids"]).ravel() for r in subset]
        )
        f = least_squares(
            rfun,
            [0, 0.1725],
            bounds=([-0.6, 0.12], [0.6, 0.23]),
            loss="soft_l1",
            f_scale=4,
        )
        fits[name] = {
            "yaw_rad": float(f.x[0]),
            "pad_center_z_m": float(f.x[1]),
            "metrics": metrics(f.x, subset),
        }
    rng = np.random.default_rng(20260906)
    boots = []
    for _ in range(100):
        subset = [records[i] for i in rng.integers(0, len(records), len(records))]
        f = least_squares(
            lambda p, subset=subset: np.concatenate(
                [(predicted(p, r) - r["centroids"]).ravel() for r in subset]
            ),
            fit.x,
            bounds=([-0.6, 0.12], [0.6, 0.23]),
            loss="soft_l1",
            f_scale=4,
        )
        boots.append(f.x)

    # Sides of outer white shells lie beyond the physical contact centers. Estimate
    # the separation bias as a diagnostic, without calling it a contact calibration.
    def pred_visible(p, r):
        yaw, z, side_bias = p
        h = 0.006 + 0.085 / 2 * (1 - np.clip(r["closure"] / 0.9, 0, 1)) + side_bias
        local = np.array(
            [
                [h * np.cos(yaw), h * np.sin(yaw), z, 1],
                [-h * np.cos(yaw), -h * np.sin(yaw), z, 1],
            ]
        )
        world = local @ forward_kinematics(r["q"], np.zeros(6)).T
        pc = world @ C.T
        uv = pc[:, :3] @ K.T
        uv = uv[:, :2] / uv[:, 2, None]
        return uv[np.argsort(uv[:, 1])]

    fvis = least_squares(
        lambda p, subset=subset: np.concatenate(
            [(pred_visible(p, r) - r["centroids"]).ravel() for r in records]
        ),
        [-0.3, 0.165, 0.005],
        bounds=([-0.6, 0.12, 0], [0.6, 0.23, 0.02]),
        loss="soft_l1",
        f_scale=4,
    )
    pct = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    output = {
        "status": "low_confidence_visual_centroid_fit_not_contact_calibration",
        "source_episode": "fit/ep_teacher_waffles_1788535016_005",
        "source_frames": "landmark_frames.json and scene_NNNN.png only (all one fit episode)",
        "camera_fit": "camera_fit.json held fixed; itself estimated from same fit episode",
        "requested_model": {
            "pad_size_m": [0.012, 0.026, 0.055],
            "half_separation_formula_m": ".006+.085/2*(1-clip(recorded_POS/.9,0,1))",
            "baseline_yaw_rad": 0,
            "baseline_pad_center_z_m": 0.1725,
        },
        "fit": fits["all"],
        "baseline_metrics": metrics([0, 0.1725]),
        "robustness_subsets": fits,
        "conditional_bootstrap_95pct": {
            "yaw_rad": pct[:, 0].tolist(),
            "pad_center_z_m": pct[:, 1].tolist(),
            "method": "100 episode-frame bootstrap resamples, camera and hand ROIs fixed. Does not capture camera/model systematic bias.",
        },
        "diagnostic_extra_visible_shell_offset_fit": {
            "yaw_rad": float(fvis.x[0]),
            "pad_center_z_m": float(fvis.x[1]),
            "outward_shell_centroid_offset_m": float(fvis.x[2]),
            "point_rmse_px": float(
                np.sqrt(
                    np.mean(
                        np.array(
                            [
                                np.linalg.norm(
                                    pred_visible(fvis.x, r) - r["centroids"], axis=1
                                )
                                for r in records
                            ]
                        )
                        ** 2
                    )
                )
            ),
        },
        "segmentation": {
            "method": "BGR/Hue-Saturation-Value thresholds within manually bounded pad ROIs, centroid of all passing pixels",
            "thresholds": "HSV S<75, V>190; min(B,G,R)>160; B>=R-8 (suppresses beige wafer graphics)",
            "frame_rois_xyxy": rois,
        },
        "observations": records,
        "recommendation": "Do not replace contact geometry from this fit alone. Optional visual yaw around -0.38 rad and z around .166 m may reduce image error, but it leaves large systematic residuals. Review rendered alignment; retain friction/compliance/contact-validation status as unvalidated.",
        "limitations": [
            "Visible white-shell centroid is not the contact-pad center, and depends on viewing angle, shell shape, occlusion and illumination.",
            "Persistent residuals of roughly15-20px on the left shell indicate model bias not resolved by yaw/z.",
            "Camera was fit on same episode and has unknown metric calibration; optical and tool parameters are coupled.",
            "Closed-pad centroids do not obey the nominal85mm stroke linearly; unknown custom pad mounts and Robotiq linkage geometry matter.",
            "Bootstrap is conditional numerical stability only, not a physical calibration confidence interval.",
            "No held-out frames or physics outcomes were used to fit these parameters.",
        ],
    }
    (root / "gripper_centroid_estimate.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps({"fit": output["fit"], "confidence": output["status"]}, indent=2))


if __name__ == "__main__":
    main()
