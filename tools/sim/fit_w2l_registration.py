#!/usr/bin/env python3
"""Fit supplier W2L exterior side faces to precontact recorded RGB on CPU.

No episode outcome enters this objective. Camera, robot q, object and CAD scale
are fixed. The result is a conditional mount/closure candidate, not a gel-gap
calibration. Requires numpy, scipy, OpenCV and Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phantom.sim.gripper_visual import _origin, _rotation
from phantom.sim.kinematics import dh_frames


def nominal_frames(theta):
    root = ET.parse(ROOT / "assets/sim/robotiq/robotiq_2f85.urdf").getroot()
    result = {"robotiq_arg2f_base_link": np.eye(4)}
    pending = list(root.findall("joint"))
    while pending:
        for j in pending[:]:
            parent = j.find("parent").get("link")
            if parent not in result:
                continue
            axis = (
                np.array([1.0, 0, 0])
                if j.find("axis") is None
                else np.fromstring(j.find("axis").get("xyz"), sep=" ")
            )
            multiplier = 0.0 if j.get("type") == "fixed" else 1.0
            if j.find("mimic") is not None:
                multiplier = float(j.find("mimic").get("multiplier", "1"))
            result[j.find("child").get("link")] = (
                result[parent]
                @ _origin(j.find("origin"))
                @ _rotation(axis, multiplier * theta)
            )
            pending.remove(j)
    return result


def stl_vertices(path):
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], "little")
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    values = np.frombuffer(data, offset=84, count=count, dtype=dtype)
    return np.unique(values["vertices"].reshape(-1, 3), axis=0).astype(float)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def white_panel(image, near=False):
    # Same image-only segmentation as the previously published nine-frame audit.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > (190 if near else 140))).astype(
        np.uint8
    )
    x0, y0, x1, y1 = (275, 340, 360, 405) if near else (235, 245, 320, 375)
    roi = np.zeros_like(mask)
    roi[y0:y1, x0:x1] = 1
    _, labels, stats, centres = cv2.connectedComponentsWithStats(mask * roi, 8)
    areas = stats[:, 4].copy()
    areas[0] = 0
    if near:
        areas[centres[:, 1] > 400] = 0
    j = int(areas.argmax())
    if areas[j] < 150:
        raise ValueError("No independently resolved white panel")
    ys, xs = np.where(labels == j)
    return cv2.convexHull(np.c_[xs, ys].astype(np.float32))[:, 0].astype(float)


def resample(poly, count=40):
    nxt = np.roll(poly, -1, axis=0)
    length = np.linalg.norm(nxt - poly, axis=1)
    cum = np.r_[0, np.cumsum(length)]
    at = np.arange(count) / count * cum[-1]
    idx = np.searchsorted(cum, at, side="right") - 1
    return poly[idx] + ((at - cum[idx]) / length[idx])[:, None] * (nxt - poly)[idx]


def boundary_distance(points, poly):
    nxt = np.roll(poly, -1, axis=0)
    edge = nxt - poly
    t = np.einsum("nki,ki->nk", points[:, None] - poly, edge) / np.maximum(
        (edge * edge).sum(1), 1e-12
    )
    projected = poly + t.clip(0, 1)[..., None] * edge
    return np.sqrt(np.min(((points[:, None] - projected) ** 2).sum(2), axis=1) + 1e-12)


def run(out, surface):
    out.mkdir(parents=True, exist_ok=True)
    evidence = (
        ROOT / "tests/fixtures/reference/teacher_scene_v10_20260908/pickup_registration_evidence"
    )
    input_path = evidence / "inputs.json"
    inputs = json.loads(input_path.read_text())
    cfg_path = ROOT / "configs/sim/waffles_d435_factory_20260908_r4.json"
    cfg = json.loads(cfg_path.read_text())
    c = cfg["camera"]
    W = np.asarray(c["world_from_cv"])
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
    yaw = Rotation.from_euler("z", cfg["gripper"]["yaw"]).as_matrix()
    data_path = (
        ROOT / "tests/fixtures/reference/w2l_gripper_build_20260908/registration_cad_faces.npz"
    )
    cad = np.load(data_path)
    native = [
        (
            cad["housing_vertices_mm"]
            if surface == "housing"
            else cad[f"{'rounded' if surface == 'rounded' else 'face'}_{i}_vertices_mm"]
        )
        / 1000
        for i in [258, 285]
    ]
    # Native STEP XYZ = width, longitudinal, depth. This working frame has
    # X=outward depth from named front-layer surface, Y=width, Z=distal from base.
    native_origin = np.array([0, -45.880029670997054, 1.74572194037]) / 1000
    axis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    local = [(p - native_origin) @ axis.T for p in native]
    normals = [
        axis @ np.array([s * 0.9866201466098214, 0, 0.16303584361610354])
        for s in [-1, 1]
    ]
    images = []
    observations = []
    frames = inputs["frames"]
    transforms = [dh_frames(f["arm_q"])[-1] for f in frames]
    for i, f in enumerate(frames):
        p = evidence / f["image"]
        assert sha(p) == f["image_sha256"]
        im = cv2.imread(str(p))
        images.append(im)
        for side in [1, -1] if i < 5 else [1]:
            poly = white_panel(im, side == -1)
            observations.append(
                {
                    "frame": i,
                    "side": side,
                    "polygon": poly,
                    "sample": resample(poly),
                    "partition": "train"
                    if i in [0, 2, 4, 6]
                    else "withheld"
                    if i in [1, 3, 5]
                    else "post_window_excluded",
                }
            )

    def project(world):
        points = (world - W[:3, 3]) @ W[:3, :3]
        p = points @ K.T
        return p[:, :2] / p[:, 2:]

    def predict(parameters, obs, want_transform=False):
        translation = parameters[:3]
        correction = Rotation.from_rotvec(parameters[3:6]).as_matrix()
        gain, offset = parameters[6:8]
        f, side = frames[obs["frame"]], obs["side"]
        theta = gain * f["gripper"][0] + offset
        # Independent analytic nominal 2F85 inner-pad attachment, already checked
        # against the pinned URDF at 901 angles in the code/motion audit.
        attachment = np.array(
            [
                side
                * (
                    0.0306011
                    + 0.0376 * np.cos(theta)
                    - 0.043 * np.sin(theta)
                    - 0.0220203446692936
                ),
                0,
                0.054904 + 0.0376 * np.sin(theta) + 0.043 * np.cos(theta) + 0.03242,
            ]
        )
        mirror = np.diag([side, side, 1])
        body_R = yaw @ mirror @ correction
        body_t = yaw @ (attachment + mirror @ translation)
        tool = transforms[obs["frame"]]
        world_R = tool[:3, :3] @ body_R
        world_t = tool[:3, :3] @ body_t + tool[:3, 3]
        visibility = [
            np.dot(W[:3, 3] - (world_R @ p.mean(0) + world_t), world_R @ n)
            for p, n in zip(local, normals)
        ]
        chosen = int(np.argmax(visibility))
        projected = project(local[chosen] @ world_R.T + world_t)
        hull = projected[ConvexHull(projected).vertices]
        if want_transform:
            return hull, chosen, body_R, body_t, theta
        return hull

    train = [o for o in observations if o["partition"] == "train"]

    def residual(parameters):
        rows = []
        for obs in train:
            p = predict(parameters, obs)
            rows.extend(boundary_distance(obs["sample"], p))
            rows.extend(boundary_distance(resample(p), obs["polygon"]))
        return np.asarray(rows)

    # Explicit loose bounds preserve unknown installation rather than silently
    # permitting arbitrary scale or per-frame motion. No contact widths enter.
    lower = [-0.040, -0.040, -0.020, -0.65, -0.65, -1.4, 0.30, 0]
    upper = [0.040, 0.040, 0.050, 0.65, 0.65, 1.4, 1.40, 0.25]
    initial = np.array([-0.005, 0, 0.014, 0, 0, 0, 0.8, 0])
    candidates = []
    for yaw_seed in [-0.3, 0, 0.3]:
        start = initial.copy()
        start[5] = yaw_seed
        fit = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.5,
            max_nfev=350,
            diff_step=1e-4,
        )
        candidates.append(fit)
    fit = min(candidates, key=lambda x: x.cost)
    rows = []
    canvas = Image.new("RGB", (640 * 3, 515 * 3), (22, 25, 30))
    for i, original in enumerate(images):
        im = original.copy()
        for obs in [o for o in observations if o["frame"] == i]:
            predicted, face, body_R, body_t, theta = predict(fit.x, obs, True)
            before = predict(initial, obs)
            err = np.r_[
                boundary_distance(obs["sample"], predicted),
                boundary_distance(resample(predicted), obs["polygon"]),
            ]
            initial_err = np.r_[
                boundary_distance(obs["sample"], before),
                boundary_distance(resample(before), obs["polygon"]),
            ]
            cv2.polylines(im, [obs["polygon"].astype(np.int32)], True, (20, 220, 40), 1)
            cv2.polylines(
                im, [np.rint(predicted).astype(np.int32)], True, (0, 150, 255), 1
            )
            # Raw STEP metres to tool0, including the working-frame origin.
            step_R = body_R @ axis
            step_t = body_t - step_R @ native_origin
            matrix = np.eye(4)
            matrix[:3, :3] = step_R
            matrix[:3, 3] = step_t
            rows.append(
                {
                    "frame_index": frames[i]["frame_index"],
                    "image": frames[i]["image"],
                    "timestamp_s": frames[i]["timestamp_s"],
                    "elapsed_s": frames[i]["elapsed_s"],
                    "side": "positive_far" if obs["side"] == 1 else "negative_near",
                    "partition": obs["partition"],
                    "observed_white_hull_px": obs["polygon"].tolist(),
                    "projected_cad_face_index": [258, 285][face],
                    "projected_hull_px": predicted.tolist(),
                    "boundary_rms_px": float(np.sqrt(np.mean(err**2))),
                    "boundary_p95_px": float(np.quantile(err, 0.95)),
                    "initial_boundary_rms_px": float(np.sqrt(np.mean(initial_err**2))),
                    "theta_rad": float(theta),
                    "tool0_from_native_step_m": matrix.tolist(),
                }
            )
        px, py = i % 3 * 640, i // 3 * 515
        canvas.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (px, py))
        phase = next(o["partition"] for o in observations if o["frame"] == i)
        ImageDraw.Draw(canvas).text(
            (px + 8, py + 482),
            f"frame {frames[i]['frame_index']} {phase} | green measured white; orange supplier side face",
            fill=(245, 245, 245),
        )
    canvas.save(out / "registration_overlay.png")
    metrics = {}
    for partition in ["train", "withheld", "post_window_excluded"]:
        rr = [r for r in rows if r["partition"] == partition]
        metrics[partition] = {
            "observations": len(rr),
            "mean_boundary_rms_px": float(np.mean([r["boundary_rms_px"] for r in rr])),
            "initial_mean_boundary_rms_px": float(
                np.mean([r["initial_boundary_rms_px"] for r in rr])
            ),
        }
    record = {
        "schema_version": 1,
        "status": "rejected_free_mount_diagnostic",
        "build_ready": False,
        "surface": surface,
        "scope": "Conditional shared rigid W2L mount and affine POS-to-angle fit to precontact exterior side-face RGB; fixed R4 camera and CAD scale",
        "script_sha256": sha(Path(__file__)),
        "input_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                input_path,
                cfg_path,
                data_path,
                ROOT / "assets/sim/robotiq/robotiq_2f85.urdf",
            ]
        },
        "parameters": {
            "translation_in_positive_housing_axes_m": fit.x[:3].tolist(),
            "rotation_correction_rotvec_rad": fit.x[3:6].tolist(),
            "theta_gain_rad_per_POS_normalized": float(fit.x[6]),
            "theta_offset_rad": float(fit.x[7]),
            "reference_attachment": "nominal right_inner_finger_pad origin, expressed in closing X / width Y / axial Z gripper frame before common configured yaw",
            "native_step_origin_m": native_origin.tolist(),
            "native_step_to_working_axes": axis.tolist(),
        },
        "parameter_bounds": {"lower": lower, "upper": upper},
        "optimizer": {
            "success": bool(fit.success),
            "message": fit.message,
            "cost": float(fit.cost),
            "nfev": int(fit.nfev),
            "jacobian_condition_number": float(np.linalg.cond(fit.jac)),
            "starts": [
                {
                    "parameters": f.x.tolist(),
                    "cost": float(f.cost),
                    "success": bool(f.success),
                }
                for f in candidates
            ],
        },
        "metrics": metrics,
        "rows": rows,
        "notes": [
            "White-plane silhouette is the semantic feature, not active gel. The same convex-hull operation is applied to observed and CAD faces.",
            "Training indices0,30,60,89; withheld15,45,74. Near plane excluded after60 because it merges with packet.104/119 excluded from all fitting and precontact validation.",
            "Exposure/rolling-shutter lag, nominal UR calibration, fixed camera-to-robot residual and actual adapter identity remain unvalidated.",
            "The observed precontact POS span is narrow; affine extrapolation to loaded grasp cannot be considered a measured gap calibration.",
            "No packet pose, packet dimension, contact force, lift success or policy outcome enters the objective.",
        ],
    }
    (out / "registration.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "parameters": record["parameters"],
                "metrics": metrics,
                "optimizer": record["optimizer"],
            },
            indent=2,
        )
    )


def check_candidate(out, config_path, pose_correction=False):
    """Compare explicit mounted candidate; fit only bounded translations.

    Nominal jaw-angle mapping and mount orientations remain fixed. Independent
    jaw translations are diagnostic estimates conditional on candidate identity.
    """
    out.mkdir(parents=True, exist_ok=True)
    config_path = config_path.resolve()
    cfg = json.loads(config_path.read_text())
    g = cfg["gripper"]
    c = cfg["camera"]
    W = np.array(c["world_from_cv"])
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
    geometry_path = ROOT / "assets/sim/dmtac_w2l/geometry.json"
    mesh_path = ROOT / "assets/sim/dmtac_w2l/meshes/housing_visual.stl"
    points = stl_vertices(mesh_path)
    evidence = (
        ROOT / "tests/fixtures/reference/teacher_scene_v10_20260908/pickup_registration_evidence"
    )
    ip = evidence / "inputs.json"
    source = json.loads(ip.read_text())
    frames = source["frames"]
    yaw = np.eye(4)
    yaw[:3, :3] = Rotation.from_euler("z", -np.pi / 2 + g["yaw"]).as_matrix()
    observations = []
    images = []
    for i, f in enumerate(frames):
        im = cv2.imread(str(evidence / f["image"]))
        images.append(im)
        assert sha(evidence / f["image"]) == f["image_sha256"]
        tool = dh_frames(f["arm_q"])[-1]
        theta = g["articulation"]["angle_at_touch_rad"] * np.clip(
            f["gripper"][0] / g["pad_touch_command"], 0, 1
        )
        chain = nominal_frames(theta)
        for side, parent in [
            ("left", "right_inner_finger"),
            ("right", "left_inner_finger"),
        ]:
            if side == "right" and i >= 5:
                continue
            mm = g["articulation"]["mounts"][side]
            mount = np.eye(4)
            mount[:3, :3] = Rotation.from_euler("xyz", mm["rpy"]).as_matrix()
            mount[:3, 3] = mm["xyz"]
            mirror = _rotation([0, 0, 1], np.pi if side == "right" else 0)
            world_from_parent = tool @ yaw @ chain[parent]
            poly = white_panel(im, side == "right")
            observations.append(
                {
                    "frame": i,
                    "side": side,
                    "theta": theta,
                    "parent": world_from_parent,
                    "mount": mount,
                    "mirror": mirror,
                    "polygon": poly,
                    "sample": resample(poly),
                    "partition": "train"
                    if i in [0, 2, 4, 6]
                    else "withheld"
                    if i in [1, 3, 5]
                    else "post_window_excluded",
                }
            )
    stride = 6 if pose_correction else 3
    zero = np.zeros(stride * 2)

    def predict(delta, obs):
        mount = obs["mount"].copy()
        offset = 0 if obs["side"] == "left" else stride
        mount[:3, 3] += delta[offset : offset + 3]
        if pose_correction:
            mount[:3, :3] = (
                mount[:3, :3]
                @ Rotation.from_rotvec(delta[offset + 3 : offset + 6]).as_matrix()
            )
        transform = obs["parent"] @ mount @ obs["mirror"]
        world = points @ transform[:3, :3].T + transform[:3, 3]
        pc = (world - W[:3, 3]) @ W[:3, :3]
        p = pc @ K.T
        p = p[:, :2] / p[:, 2:]
        return p[ConvexHull(p).vertices]

    def errors(delta, obs):
        p = predict(delta, obs)
        return np.r_[
            boundary_distance(obs["sample"], p),
            boundary_distance(resample(p), obs["polygon"]),
        ]

    training = [o for o in observations if o["partition"] == "train"]
    bound = np.tile(
        [0.030, 0.030, 0.030, 0.25, 0.25, 0.25]
        if pose_correction
        else [0.020, 0.020, 0.020],
        2,
    )
    fit = least_squares(
        lambda d: np.concatenate([errors(d, o) for o in training]),
        zero,
        bounds=(-bound, bound),
        loss="soft_l1",
        f_scale=1.5,
        max_nfev=250,
        diff_step=1e-4,
        x_scale="jac",
    )
    rows = []
    canvas = Image.new("RGB", (1920, 1545), (22, 25, 30))
    for i, im in enumerate(images):
        im = im.copy()
        for o in [o for o in observations if o["frame"] == i]:
            before = predict(zero, o)
            after = predict(fit.x, o)
            cv2.polylines(im, [o["polygon"].astype(np.int32)], True, (20, 220, 40), 1)
            cv2.polylines(
                im, [np.rint(before).astype(np.int32)], True, (255, 160, 0), 1
            )
            cv2.polylines(im, [np.rint(after).astype(np.int32)], True, (0, 150, 255), 1)
            e0 = errors(zero, o)
            e1 = errors(fit.x, o)
            rows.append(
                {
                    "frame_index": frames[i]["frame_index"],
                    "side": o["side"],
                    "partition": o["partition"],
                    "nominal_theta_rad": float(o["theta"]),
                    "baseline_boundary_rms_px": float(np.sqrt(np.mean(e0**2))),
                    "corrected_boundary_rms_px": float(np.sqrt(np.mean(e1**2))),
                    "observed_hull_px": o["polygon"].tolist(),
                    "baseline_projected_hull_px": before.tolist(),
                    "corrected_projected_hull_px": after.tolist(),
                }
            )
        x = i % 3 * 640
        y = i // 3 * 515
        canvas.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (x, y))
        ImageDraw.Draw(canvas).text(
            (x + 8, y + 484),
            f"frame{frames[i]['frame_index']} green:white RGB | blue:fixed CAD mount | orange:translation diagnostic",
            fill="white",
        )
    canvas.save(out / "registration_candidate_overlay.png")
    mounts = {}
    for side, j in [("left", 0), ("right", stride)]:
        rotation = Rotation.from_euler("xyz", g["articulation"]["mounts"][side]["rpy"])
        if pose_correction:
            rotation = rotation * Rotation.from_rotvec(fit.x[j + 3 : j + 6])
        mounts[side] = {
            "xyz": (
                np.array(g["articulation"]["mounts"][side]["xyz"]) + fit.x[j : j + 3]
            ).tolist(),
            "rpy": rotation.as_euler("xyz").tolist(),
        }
    metrics = {}
    for p in ["train", "withheld", "post_window_excluded"]:
        rr = [r for r in rows if r["partition"] == p]
        metrics[p] = {
            "observations": len(rr),
            "baseline_mean_boundary_rms_px": float(
                np.mean([r["baseline_boundary_rms_px"] for r in rr])
            ),
            "corrected_mean_boundary_rms_px": float(
                np.mean([r["corrected_boundary_rms_px"] for r in rr])
            ),
        }
    result = {
        "schema_version": 1,
        "status": "rejected_rigid_registration_diagnostic",
        "build_ready": False,
        "method": (
            "Fixed nominal angle mapping. Per-parent translation within±30mm and rotation-vector components within±0.25rad."
            if pose_correction
            else "Fixed nominal angle mapping and mount rotation. Per-parent translation within±20mm."
        )
        + " Fit uses precontact white-exterior convex hull. Whole CAD housing is projected; real segmentation may exclude shaded/front portions.",
        "config_path": str(config_path),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [config_path, geometry_path, mesh_path, ip]
        },
        "pose_correction": pose_correction,
        "parameter_bounds_abs": bound.tolist(),
        "fitted_delta_parameters": fit.x.tolist(),
        "translation_delta_parent_m": {
            "left": fit.x[:3].tolist(),
            "right": fit.x[stride : stride + 3].tolist(),
        },
        "candidate_mounts": mounts,
        "bound_hit": bool(np.any(abs(fit.x) > 0.999 * bound)),
        "theta_mapping": {
            "angle_at_touch_rad": g["articulation"]["angle_at_touch_rad"],
            "pad_touch_command": g["pad_touch_command"],
        },
        "metrics": metrics,
        "rows": rows,
        "optimization": {
            "success": bool(fit.success),
            "cost": float(fit.cost),
            "condition": float(np.linalg.cond(fit.jac)),
        },
        "scope_limits": [
            "No actuator angle or mounted gap calibration is inferred.",
            "Fixed R4 camera-to-UR residual can enter mount translations; these are conditional registration estimates.",
            "Whole-housing silhouette versus white-painted-region segmentation is a declared imperfect correspondence.",
            "No loaded closure, tactile/contact forces, object motion or policy success enters fitting.",
        ],
        "rejection_reason": "Bound-hitting pose changes and imperfect whole-housing versus visible-white-region correspondence; not a calibrated installed mount.",
        "command_argv": sys.argv,
        "script_sha256": sha(Path(__file__)),
    }
    (out / "registration_candidate.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["rows", "source_sha256"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "tests/fixtures/reference/w2l_gripper_build_20260908"
    )
    parser.add_argument(
        "--surface", choices=["flat", "rounded", "housing"], default="rounded"
    )
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--pose-correction", action="store_true")
    args = parser.parse_args()
    if args.candidate_config:
        check_candidate(args.out, args.candidate_config, args.pose_correction)
    else:
        run(args.out, args.surface)
