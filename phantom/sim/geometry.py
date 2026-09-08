"""Bin geometry shared by scene construction and object-state scoring.

All lengths are metres; optional bin yaw rotates its cavity about the UR Z axis.
``bin.center`` is the XY centre at the *underside* of the bottom, not the volume
centre. Cavity bounds and primitive boxes are expressed before that rotation.

Legacy configurations use ``size`` and a shared side/bottom ``wall`` dimension.
The opt-in ``rectangular_envelope`` model instead requires ``outer_size``
(XYZ), ``opening_size`` (XY), and ``floor_thickness``. Its constant rectangular
cavity is an explicit geometric approximation: the difference between outer
envelope and opening is NOT a measurement of uniform plastic wall thickness.
Taper, ribs, handles, feet and rim profiles need additional measurements/CAD.
In particular, approximate internal depth does not imply a zero-thickness floor;
the caller must supply and document a positive floor estimate independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def table_hole_centers(config: dict) -> np.ndarray:
    """Visual lattice on the table, optionally registered to an observed hole.

    The default retains historical half-cell placement. A registered origin is
    one hole center in UR XY; yaw rotates the lattice, not the table collider.
    """
    center = np.asarray(config["center"], float)[:2]
    size = _dimensions(config["size"], 3, "table.size")[:2]
    pitch = float(config["hole_pitch"])
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("table.hole_pitch must be finite and positive")
    if "grid_origin_xy" not in config and "grid_yaw" not in config:
        return np.array([
            [center[0] - size[0] / 2 + (i + .5) * pitch,
             center[1] - size[1] / 2 + (j + .5) * pitch]
            for i in range(int(size[0] / pitch))
            for j in range(int(size[1] / pitch))
        ])
    origin = np.asarray(config["grid_origin_xy"], float)
    yaw = float(config.get("grid_yaw", 0))
    if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(yaw):
        raise ValueError("Registered table grid needs finite XY origin and yaw")
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    corners = center + np.array([[-1,-1], [-1,1], [1,-1], [1,1]]) * size / 2
    bounds = (corners - origin) @ rotation / pitch
    lo, hi = np.floor(bounds.min(0)).astype(int), np.ceil(bounds.max(0)).astype(int)
    ij = np.array([[i,j] for i in range(lo[0],hi[0]+1) for j in range(lo[1],hi[1]+1)])
    xy = (ij * pitch) @ rotation.T + origin
    radius = float(config["hole_radius"])
    keep = (np.abs(xy-center) <= size/2-radius).all(axis=1)
    return xy[keep]


@dataclass(frozen=True)
class BinBox:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class MountPlateGeometry:
    size: np.ndarray
    yaw: float

    @property
    def center(self) -> np.ndarray:
        # The robot base/mounting plane defines Z=0; plate extends below it.
        return np.array([0.0, 0.0, -self.size[2] / 2])


def mount_plate_geometry(config: dict) -> MountPlateGeometry:
    """Optional robot_mount size/yaw; old scenes retain 160x160x22 mm at yaw 0.

    Yaw is radians about the UR-frame +Z axis. Neither this plate option nor a
    photo estimate moves the robot origin or establishes the mat/table datum.
    """
    size = _dimensions(config.get("size", [0.16, 0.16, 0.022]), 3, "robot_mount.size")
    yaw = float(config.get("yaw", 0.0))
    if not np.isfinite(yaw):
        raise ValueError("robot_mount.yaw must be finite radians")
    return MountPlateGeometry(size, yaw)


@dataclass(frozen=True)
class BinGeometry:
    model: str
    center: np.ndarray
    outer_size: np.ndarray
    opening_size: np.ndarray
    floor_thickness: float
    legacy_wall: float | None = None
    yaw: float = 0.0

    def to_interior_frame(self, world_points) -> np.ndarray:
        """Undo box yaw about its center; bounds retain the historical datum."""
        points = np.asarray(world_points, dtype=float)
        if not self.yaw:
            return points
        result = points.copy()
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s], [s, c]])
        result[..., :2] = (points[..., :2] - self.center[:2]) @ rotation + self.center[:2]
        return result

    def from_interior_frame(self, aligned_points) -> np.ndarray:
        """Rotate canonical box points into the UR world frame."""
        points = np.asarray(aligned_points, dtype=float)
        if not self.yaw:
            return points
        result = points.copy()
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s], [s, c]])
        result[..., :2] = (points[..., :2] - self.center[:2]) @ rotation.T + self.center[:2]
        return result

    @property
    def interior_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds before yaw; transform world points with to_interior_frame."""
        if self.model == "legacy":
            # Preserve historical floating-point operation order for scoring.
            lower_xy = self.center[:2] - self.outer_size[:2] / 2 + self.legacy_wall
            upper_xy = self.center[:2] + self.outer_size[:2] / 2 - self.legacy_wall
        else:
            lower_xy = self.center[:2] - self.opening_size / 2
            upper_xy = self.center[:2] + self.opening_size / 2
        return (
            np.r_[lower_xy, self.center[2] + self.floor_thickness],
            np.r_[upper_xy, self.center[2] + self.outer_size[2]],
        )

    def collision_boxes(self) -> tuple[BinBox, ...]:
        """Five solid boxes; measured proxy stays inside the outer envelope.

        Legacy parts reproduce the original intersecting boxes exactly. The
        measured proxy partitions the material envelope without corner/floor
        overlaps, leaving the specified rectangular cavity unobstructed.
        """
        bx, by, bz = self.center
        sx, sy, height = self.outer_size
        floor = self.floor_thickness
        measured = self.model == "rectangular_envelope"
        if measured:
            wx, wy = (self.outer_size[:2] - self.opening_size) / 2
        else:
            wx = wy = self.legacy_wall
        side_height = height - floor if measured else height
        side_z = bz + floor + side_height / 2 if measured else bz + height / 2
        boxes = [BinBox("Bottom", (bx, by, bz + floor / 2), (sx, sy, floor))]
        for name, sign in (("Left", -1), ("Right", 1)):
            boxes.append(
                BinBox(
                    name,
                    (bx + sign * (sx - wx) / 2, by, side_z),
                    (wx, sy, side_height),
                )
            )
        for name, sign in (("Front", -1), ("Back", 1)):
            boxes.append(
                BinBox(
                    name,
                    (bx, by + sign * (sy - wy) / 2, side_z),
                    (self.opening_size[0] if measured else sx, wy, side_height),
                )
            )
        return tuple(boxes)


def _dimensions(value, count, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError(f"{name} must contain {count} finite positive dimensions")
    return array


def bin_geometry(config: dict) -> BinGeometry:
    """Resolve one unambiguous bin schema; never silently ignore measured keys."""
    center = np.asarray(config["center"], dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("bin.center must contain three finite coordinates")
    yaw = float(config.get("yaw", 0.0))
    if not np.isfinite(yaw):
        raise ValueError("bin.yaw must be finite radians")
    model = config.get("geometry_model", "legacy")
    if model == "legacy":
        if any(
            key in config for key in ("outer_size", "opening_size", "floor_thickness")
        ):
            raise ValueError(
                "measured bin dimensions require geometry_model='rectangular_envelope'"
            )
        size = _dimensions(config["size"], 3, "bin.size")
        wall = float(config["wall"])
        if not np.isfinite(wall) or wall <= 0 or (size <= 2 * wall).any():
            raise ValueError(
                "bin.wall must be positive and less than half every bin dimension"
            )
        return BinGeometry(model, center, size, size[:2] - 2 * wall, wall, wall, yaw)
    if model != "rectangular_envelope":
        raise ValueError(f"Unsupported bin.geometry_model: {model!r}")
    if "size" in config or "wall" in config:
        raise ValueError(
            "rectangular_envelope must not mix legacy bin.size/bin.wall with measured dimensions"
        )
    size = _dimensions(config["outer_size"], 3, "bin.outer_size")
    opening = _dimensions(config["opening_size"], 2, "bin.opening_size")
    floor = float(config["floor_thickness"])
    if (opening >= size[:2]).any():
        raise ValueError("bin.opening_size must be smaller than the outer XY envelope")
    if not np.isfinite(floor) or not 0 < floor < size[2]:
        raise ValueError(
            "bin.floor_thickness must be positive and less than outside height"
        )
    return BinGeometry(model, center, size, opening, floor, yaw=yaw)
