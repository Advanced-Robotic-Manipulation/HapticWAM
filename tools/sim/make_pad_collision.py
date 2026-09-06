#!/usr/bin/env python3
"""Generate an estimated rounded tactile collision proxy, not measured CAD.

The contact face is an ellipse in YZ extruded along X. This preserves the
inner-face span while avoiding artificial cuboid corner contacts with the mat.
It should be evaluated as a shape hypothesis against both fit recordings.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def elliptical_pad_triangles(size=(0.012, 0.026, 0.055), segments=48):
    half = np.asarray(size, dtype=float) / 2
    if half.shape != (3,) or np.any(half <= 0) or segments < 8:
        raise ValueError("Need positive XYZ sizes and at least eight segments")
    angle = np.arange(segments) * (2 * np.pi / segments)
    ring = np.c_[np.zeros(segments), half[1] * np.cos(angle), half[2] * np.sin(angle)]
    left, right = ring.copy(), ring.copy()
    left[:, 0], right[:, 0] = -half[0], half[0]
    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        triangles.extend(
            [
                [[-half[0], 0, 0], left[j], left[i]],
                [[half[0], 0, 0], right[i], right[j]],
                [left[i], left[j], right[j]],
                [left[i], right[j], right[i]],
            ]
        )
    return np.asarray(triangles, dtype=np.float64)


def write_stl(path, triangles):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"PHANTOM estimated elliptical pad; metres".ljust(80, b" "))
        stream.write(struct.pack("<I", len(triangles)))
        for points in triangles:
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            normal /= np.linalg.norm(normal)
            stream.write(struct.pack("<12fH", *normal, *points.reshape(-1), 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets/sim/pads/pad_elliptical_12x26x55mm.stl"),
    )
    parser.add_argument("--size", type=float, nargs=3, default=[0.012, 0.026, 0.055])
    args = parser.parse_args()
    triangles = elliptical_pad_triangles(args.size)
    write_stl(args.out, triangles)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": "estimated_collision_proxy_not_measured_geometry",
                "units": "metres",
                "shape": "ellipse in YZ extruded along X",
                "size_xyz_m": args.size,
                "segments": 48,
                "triangles": len(triangles),
                "source": "procedural tools/sim/make_pad_collision.py",
                "rationale": "Rounded real tactile shell observed in RGB. Full box corners caused mat contact that blocked simulated closure in raised75/raised90 diagnostics.",
                "validation": "No claim of calibrated contact patch, compliance or real-world transfer.",
            },
            indent=2,
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
