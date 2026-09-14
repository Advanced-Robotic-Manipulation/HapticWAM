"""Validated detailed contacts for one pad/object rigid-body pair.

Only populated PhysX entries are read. This observer does not change physics,
forces, success thresholds or the material assigned to any collider.
"""
from __future__ import annotations

import numpy as np

PAD_OBJECT_CONTACT_CAPACITY = 256


def convex_support_paths(object_path, obj):
    """Permit patch interpolation only for the existing single-hull objects.

Compound and SDF eggs can contain separate native contact patches, even when
their authored exterior approximates a convex egg. Keep their gel estimate
on the conservative point fallback until patch contiguity is validated.
"""
    if obj.get("kind") == "egg" and obj.get("collision_approximation", "convexHull") != "convexHull":
        return ()
    return (str(object_path),)


def summarize_pad_object_contacts(contact_data, *, max_contact_count):
    """Reduce one sensor/filter pair without accepting truncated force data.

The six-array API tuple has force, point, normal, separation, count and start
buffers. The caller supplies its actual capacity. Reaching capacity is treated
as possible saturation, as in the support and gel contact reducers.
"""
    if isinstance(max_contact_count, (bool, np.bool_)) or not isinstance(max_contact_count, (int, np.integer)) or max_contact_count <= 0:
        raise ValueError("max_contact_count must be a positive integer")
    force, points, normals, separations, counts, starts = contact_data
    force = np.asarray(force, dtype=float).reshape(-1)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    separations = np.asarray(separations, dtype=float).reshape(-1)
    counts, starts = np.asarray(counts), np.asarray(starts)
    if counts.shape != (1, 1) or starts.shape != (1, 1):
        raise RuntimeError("pad/object contacts require one sensor and one rigid-body filter")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != counts.astype(np.int64)).any():
        raise RuntimeError("invalid pad/object contact count")
    count = int(counts[0, 0])
    if count >= max_contact_count:
        raise RuntimeError("pad/object contact buffer may be saturated; increase max_contact_count")
    if not count:
        return {"contact_count": 0, "normal_force_n": 0., "centroid_world_m": None}
    start = starts[0, 0]
    if not np.isfinite(start) or start < 0 or start != int(start):
        raise RuntimeError("invalid pad/object contact start")
    first = int(start)
    if first + count > min(len(force), len(points), len(normals), len(separations)):
        raise RuntimeError("pad/object contact range exceeds one or more buffers")
    selection = slice(first, first + count)
    raw, xyz, normal, separation = force[selection], points[selection], normals[selection], separations[selection]
    if not np.isfinite(np.r_[raw, xyz.ravel(), normal.ravel(), separation]).all():
        raise RuntimeError("nonfinite populated pad/object contact data")
    if ((abs(raw) > 0) & (np.linalg.norm(normal, axis=1) < 1e-9)).any():
        raise RuntimeError("loaded pad/object contact has zero normal")
    weight = abs(raw)
    total = float(weight.sum())
    return {
        "contact_count": count,
        "normal_force_n": total,
        "centroid_world_m": np.average(xyz, axis=0, weights=weight) if total > 1e-8 else None,
    }
