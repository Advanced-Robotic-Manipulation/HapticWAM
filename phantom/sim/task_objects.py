"""Task object geometry in metres, independent of Isaac imports.

The historical ``waffle`` configuration and trace names remain readable. New
scenes use ``object`` and carry their identity explicitly. Egg shells and paper
cartons are rigid-body approximations; this module does not model fracture.
"""
from __future__ import annotations

import numpy as np


def object_config(config: dict) -> dict:
    if "object" in config and "waffle" in config:
        raise ValueError("Specify one task object: object or historical waffle, not both")
    obj = config["object"] if "object" in config else config["waffle"]
    size = np.asarray(obj["size"], float)
    if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Object size must contain three finite positive dimensions")
    if "mass" in obj and (not np.isfinite(obj["mass"]) or obj["mass"] <= 0):
        raise ValueError("Object mass must be finite and positive")
    if "nominal_capacity_ml" in obj:
        capacity = float(obj["nominal_capacity_ml"])
        if obj.get("kind") != "carton" or not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("nominal_capacity_ml requires a carton and a finite positive capacity")
        # The liquid must fit even inside the exterior bounding box. Passing
        # this necessary bound does not identify XYZ, wall thickness, folds,
        # headspace, or the mass/density of the contents.
        if float(np.prod(size)) * 1e6 + 1e-9 < capacity:
            raise ValueError("Carton exterior bounding volume is smaller than nominal_capacity_ml")
    return obj


def object_identity(config: dict) -> tuple[str, str]:
    obj = object_config(config)
    kind = obj.get("kind", "waffle")
    if kind not in ("waffle", "carton", "egg"):
        raise ValueError(f"Unsupported task object kind: {kind}")
    return kind, "/World/" + {"waffle": "Waffle", "carton": "Carton", "egg": "Egg"}[kind]


def egg_mesh(size, rings=40, segments=64, taper=0.17):
    """Closed asymmetric ovoid, long axis local Z and pointed end at +Z.

    Exact XYZ bounds; outward triangles with shared poles. The broad end is
    rounded, unlike a scaled sphere. Taper is an image estimate. Convex-hull
    collision keeps rolling/contact geometry close to the rendered surface.
    """
    size = np.asarray(size, float)
    if size.shape != (3,) or (size <= 0).any() or not np.isfinite(size).all():
        raise ValueError("Egg size must be finite positive XYZ")
    if rings < 8 or segments < 12 or segments % 4 or not 0 <= taper <= 0.25:
        raise ValueError("Invalid egg mesh resolution or taper")
    theta = np.arange(1, rings) * np.pi / rings
    phi = np.arange(segments) * 2 * np.pi / segments
    z = np.cos(theta)
    radius = np.sin(theta) * (1 - taper * z)
    radius /= radius.max()
    points = [[0., 0., size[2]/2]]
    for h, r in zip(z, radius):
        points.extend(np.c_[size[0]/2*r*np.cos(phi), size[1]/2*r*np.sin(phi),
                            np.full(segments, size[2]/2*h)].tolist())
    bottom = len(points)
    points.append([0., 0., -size[2]/2])
    faces = []
    for i in range(segments):
        j = (i+1) % segments
        faces.append([0, 1+i, 1+j])
        for row in range(rings-2):
            a, b = 1+row*segments+i, 1+row*segments+j
            faces.extend([[a, a+segments, b+segments], [a, b+segments, b]])
        faces.append([bottom, 1+(rings-2)*segments+j, 1+(rings-2)*segments+i])
    return np.asarray(points), np.asarray(faces, dtype=np.int32)


def orientation_wxyz(obj: dict) -> np.ndarray:
    """Use explicit XYZ Euler pose when present, historical yaw otherwise."""
    from scipy.spatial.transform import Rotation
    rpy = obj.get("rpy", [0, 0, obj.get("yaw", 0)])
    if np.asarray(rpy).shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError("Object rpy must contain three finite radians")
    return Rotation.from_euler("xyz", rpy).as_quat()[[3, 0, 1, 2]]
