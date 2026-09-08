#!/usr/bin/env python3
"""Check Isaac rendered marker centroids against independent pinhole equations.

Simulation only: seven static emissive spheres, no robot, policy or hardware.
Use the Isaac Python environment. The output directory must not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONFIG = {
    "resolution": [640, 480],
    "fx": 617.2,
    "fy": 611.8,
    "cx": 318.6,
    "cy": 242.1,
    "camera_model": "opencv_pinhole",
    "distortion_model": "none",
    "distortion_coefficients": [],
}
# Fixed physical positions in OpenCV coordinates (+X right, +Y down, +Z forward).
# Different depths prevent a single fitted image-plane transform from defining
# this fixture. These are independent of the projection configuration helper.
POINTS_CV = np.array([
    [-0.475, -0.337, 1.0],
    [0.49, -0.355, 1.1],
    [0.0, 0.0, 1.2],
    [-0.625, 0.55, 1.6],
    [0.75, 0.53, 1.6],
    [-0.2, 0.13, 0.85],
    [0.3, 0.23, 1.0],
])


def run_fixture(app, output):
    import cv2
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    from phantom.sim.camera import configure_camera_intrinsics, validate_camera_intrinsics

    world = World(stage_units_in_meters=1, physics_dt=1 / 60, rendering_dt=1 / 30)
    stage = omni.usd.get_context().get_stage()
    material = UsdShade.Material.Define(stage, "/World/MarkerMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/MarkerMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for index, point in enumerate(POINTS_CV):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/Marker{index}")
        # Approximately 4-pixel radius at all depths. These are geometry only.
        sphere.CreateRadiusAttr(float(0.0065 * point[2]))
        sphere.AddTranslateOp().Set(Gf.Vec3d(*map(float, point * [1, -1, -1])))
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
    camera = Camera("/World/FixtureCamera", frequency=30, resolution=(640, 480))
    # Identity USD camera looks along -Z with +Y up. Thus world=[X,-Y,-Z].
    camera.set_world_pose(
        position=np.zeros(3), orientation=np.array([1.0, 0.0, 0.0, 0.0]), camera_axes="usd"
    )
    configure_camera_intrinsics(camera, CONFIG)
    camera.set_clipping_range(0.01, 10.0)
    camera.set_lens_aperture(0.0)  # Pinhole fixture; disable depth of field.
    world.reset()
    camera.initialize()
    projection = validate_camera_intrinsics(camera, CONFIG)
    for _ in range(40):
        world.step(render=True)
    rgba = np.asarray(camera.get_rgba())
    if rgba.shape != (480, 640, 4):
        raise RuntimeError(f"Unexpected rendered RGBA shape: {rgba.shape}")
    rgb = rgba[:, :, :3].astype(np.uint8)
    cv2.imwrite(str(output / "render.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Independent projection equations: intentionally do not use either the
    # configuration helper or Isaac's projection/intrinsics getters here.
    expected = np.column_stack([
        617.2 * POINTS_CV[:, 0] / POINTS_CV[:, 2] + 318.6,
        611.8 * POINTS_CV[:, 1] / POINTS_CV[:, 2] + 242.1,
    ])
    # Image coordinates describe pixel edges; array-index centroid+0.5 gives
    # pixel centers in that convention. Record raw centers too for auditing.
    mask = np.all(rgb > 100, axis=2).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    detected = np.array([
        centers[i] for i in range(1, count) if 10 <= stats[i, cv2.CC_STAT_AREA] <= 300
    ]).reshape(-1, 2)
    if len(detected) != len(POINTS_CV):
        raise RuntimeError(f"Expected seven isolated markers, detected {len(detected)}")
    assignment = np.argmin(np.linalg.norm(expected[:, None] - (detected + 0.5), axis=2), axis=1)
    if len(set(assignment.tolist())) != len(POINTS_CV):
        raise RuntimeError("Marker assignment is not one-to-one")
    raw_centers = detected[assignment]
    observed = raw_centers + 0.5
    errors = np.linalg.norm(observed - expected, axis=1)
    success = bool(errors.mean() <= 1.0 and errors.max() <= 1.5)
    annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for i, (pixel, err) in enumerate(zip(expected, errors)):
        location = tuple(np.rint(pixel).astype(int))
        cv2.drawMarker(annotated, location, (0, 200, 0), cv2.MARKER_CROSS, 13, 1)
        cv2.putText(annotated, f"{i}: {err:.2f}px", (max(2, location[0] - 40), max(14, location[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / "annotated.png"), annotated)
    report = {
        "status": "pass" if success else "fail",
        "fixture": "seven static emissive sphere centroids, asymmetric K, zero distortion",
        "scope": "Native renderer projection check; not physical D435 calibration or scene alignment",
        "config": CONFIG,
        "camera_projection": projection,
        "thresholds_fixed_before_render": {"mean_error_px_max": 1.0, "maximum_error_px_max": 1.5},
        "mean_centroid_error_px": float(errors.mean()),
        "maximum_centroid_error_px": float(errors.max()),
        "pixel_coordinate_convention": "Image boundary coordinates; pixel center is array index + 0.5",
        "centroid_method": "Connected components of all RGB channels >100; unweighted pixel centroids",
        "markers": [{
            "index": i, "point_cv_m": p.tolist(), "expected_pixel": e.tolist(),
            "rendered_centroid_array_index": r.tolist(), "rendered_pixel_center": o.tolist(),
            "error_px": float(error),
        } for i, (p, e, r, o, error) in enumerate(zip(POINTS_CV, expected, raw_centers, observed, errors))],
        "source_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), REPO / "phantom/sim/camera.py"]
        },
        "no_policy_or_hardware": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    stage.GetRootLayer().Export(str(output / "fixture.usda"))
    print(json.dumps({k: report[k] for k in ["status", "mean_centroid_error_px", "maximum_centroid_error_px"]}), flush=True)
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": True, "width": 640, "height": 480,
        "renderer": "RaytracedLighting", "anti_aliasing": 0,
        "multi_gpu": False, "sync_loads": True,
    })
    exit_code = 1
    try:
        exit_code = run_fixture(app, args.output)
    except BaseException:
        import traceback

        failure = traceback.format_exc()
        (args.output / "FAILED.txt").write_text(failure)
        print(failure, flush=True)
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
