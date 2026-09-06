"""Select actual inner gel-surface contacts from independent PhysX views.

This helper does not change colliders, filters used for task scoring, or safety.
The new views observe explicit environment bodies plus the opposite pad. Net
forces on backing/linkage remain physical but are not painted into gel images.
Contact force data contains normal forces; friction/shear is not reconstructed
from these samples. SDK image-axis orientation is an explicit estimate.

Optional ``manifold_patch`` coverage treats each body's aligned inner-face
manifold as a convex support patch with uniform pressure. PhysX manifold points
are sparse solver constraints, not samples of a pressure image. The overlap of
that support polygon with the active ellipse determines the gel force fraction
and centroid. This spatial-pressure assumption is not a calibration; disconnected
patches on one body can be bridged by their convex hull. Sparse/collinear manifolds
explicitly retain point selection. An inscribed 512-sided ellipse makes overlap
conservative (ellipse area deficit below 0.0026%). Neither mode changes physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

_ELLIPSE_SEGMENTS = 512


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _convex_hull(points):
    """Counterclockwise monotone-chain hull; no artificial patch thickness."""
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) < 3:
        return points
    chains = []
    for sequence in (points, points[::-1]):
        chain = []
        for point in sequence:
            while (
                len(chain) >= 2
                and _cross2(chain[-1] - chain[-2], point - chain[-1]) <= 0
            ):
                chain.pop()
            chain.append(point)
        chains.append(chain[:-1])
    return np.asarray(chains[0] + chains[1])


def _polygon_area_centroid(polygon):
    if len(polygon) < 3:
        return 0.0, None
    following = np.roll(polygon, -1, axis=0)
    cross = _cross2(polygon, following)
    twice_area = float(cross.sum())
    if twice_area <= 2e-12:
        return 0.0, None
    centroid = ((polygon + following) * cross[:, None]).sum(axis=0) / (3 * twice_area)
    return twice_area / 2, centroid


def _ellipse_patch_overlap(hull, geometry):
    """Clip an inscribed ellipse polygon against the convex support hull."""
    theta = np.arange(_ELLIPSE_SEGMENTS) * (2 * np.pi / _ELLIPSE_SEGMENTS)
    polygon = np.column_stack((np.cos(theta), np.sin(theta)))
    polygon *= np.asarray(geometry.active_size_yz_m) / 2
    polygon += np.asarray(geometry.active_center_yz_m)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        if len(polygon) < 3:
            return 0.0, None
        following = np.roll(polygon, -1, axis=0)
        distances = _cross2(b - a, polygon - a)
        next_distances = np.roll(distances, -1)
        output = []
        for p, q, dp, dq in zip(polygon, following, distances, next_distances):
            inside_p, inside_q = dp >= 0, dq >= 0
            if inside_p:
                output.append(p)
            if inside_p != inside_q:
                output.append(p + dp / (dp - dq) * (q - p))
        polygon = np.asarray(output, dtype=float).reshape(-1, 2)
    return _polygon_area_centroid(polygon)


@dataclass(frozen=True)
class GelSurfaceGeometry:
    pad_thickness_m: float = 0.012
    active_size_yz_m: tuple[float, float] = (0.027, 0.036)
    active_center_yz_m: tuple[float, float] = (0.0, 0.0)
    inner_face_tolerance_m: float = 0.002
    minimum_normal_alignment: float = 0.75
    sdk_uv_axes: tuple[int, int] = (2, 1)  # image columns=padZ, rows=padY
    sdk_uv_signs: tuple[int, int] = (1, 1)

    def __post_init__(self):
        x = np.r_[
            self.pad_thickness_m, self.active_size_yz_m, self.inner_face_tolerance_m
        ]
        if not np.isfinite(x).all() or np.any(x <= 0):
            raise ValueError("gel geometry dimensions must be finite and positive")
        if not 0 < self.minimum_normal_alignment <= 1:
            raise ValueError("normal alignment must lie in(0,1]")
        if set(self.sdk_uv_axes) != {1, 2} or not set(self.sdk_uv_signs) <= {-1, 1}:
            raise ValueError("SDK UV axes must be a signed permutation of padY/Z")
        if not np.isfinite(self.active_center_yz_m).all():
            raise ValueError("gel center must be finite")


def select_gel_contacts(
    contact_data,
    pad_position,
    pad_orientation_wxyz,
    *,
    side,
    geometry=None,
    max_contact_count=None,
    filter_paths=None,
    coverage="point",
):
    """Filter the six-array result of RigidPrim.get_contact_force_data(dt).

    Normals can be oriented toward either actor by the PhysX API. Surface
    position selects the inner face; absolute normal alignment then avoids a
    left/right sign assumption. The result's compression is positive for both.
    Only populated contact indices are inspected; unused buffers may containNaN.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if coverage not in ("point", "manifold_patch"):
        raise ValueError("coverage must be point or manifold_patch")
    g = geometry or GelSurfaceGeometry()
    force, points, normals, _, counts, starts = contact_data
    force = np.asarray(force).reshape(-1)
    points, normals = (
        np.asarray(points).reshape(-1, 3),
        np.asarray(normals).reshape(-1, 3),
    )
    count_array = np.asarray(counts)
    filter_count = count_array.shape[-1] if count_array.ndim else 1
    counts, starts = count_array.ravel(), np.asarray(starts).ravel()
    if (
        counts.shape != starts.shape
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
    ):
        raise RuntimeError("invalid PhysX contact counts")
    if np.any(counts != counts.astype(np.int64)):
        raise RuntimeError("noninteger PhysX contact count")
    if max_contact_count is not None and counts.sum() >= max_contact_count:
        raise RuntimeError(
            "gel contact buffer may be saturated; increase max_contact_count"
        )
    populated = []
    per_filter_ids = [[] for _ in range(filter_count)]
    for pair_index, (count, start) in enumerate(zip(counts.astype(np.int64), starts)):
        if count == 0:
            continue
        if not np.isfinite(start) or start < 0 or start != int(start):
            raise RuntimeError("invalid PhysX contact start index")
        first = int(start)
        if first + count > min(len(force), len(points), len(normals)):
            raise RuntimeError("PhysX contact range exceeds buffer")
        contact_ids = list(range(first, first + int(count)))
        populated.extend(contact_ids)
        per_filter_ids[pair_index % filter_count].extend(contact_ids)
    ids = np.unique(populated).astype(np.int64)
    position = np.asarray(pad_position, dtype=float).reshape(3)
    quat = np.asarray(pad_orientation_wxyz, dtype=float).reshape(4)
    if not np.isfinite(np.r_[position, quat]).all() or np.linalg.norm(quat) < 1e-9:
        raise RuntimeError("invalid measured pad pose")
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    weights = abs(force[ids]).astype(float)
    world_points, world_normals = points[ids], normals[ids]
    if not np.isfinite(
        np.r_[weights, world_points.ravel(), world_normals.ravel()]
    ).all():
        raise RuntimeError("nonfinite populated PhysX contact data")
    local = (world_points - position) @ rotation
    local_normals = world_normals @ rotation
    lengths = np.linalg.norm(local_normals, axis=1)
    if np.any((weights > 0) & (lengths < 1e-9)):
        raise RuntimeError("loaded PhysX contact has zero normal")
    alignment = abs(local_normals[:, 0]) / np.maximum(lengths, 1e-9)
    # Left pad is positiveX: inner face at−thickness/2; right is the reverse.
    face_x = (-1 if side == "left" else 1) * g.pad_thickness_m / 2
    face = abs(local[:, 0] - face_x) <= g.inner_face_tolerance_m
    centered_yz = local[:, 1:3] - np.asarray(g.active_center_yz_m)
    ellipse = (
        np.sum((centered_yz / (np.asarray(g.active_size_yz_m) / 2)) ** 2, axis=1) <= 1
    )
    aligned = alignment >= g.minimum_normal_alignment
    accepted = face & ellipse & aligned & (weights > 0)
    compression = weights * alignment
    point_accepted = accepted.copy()
    fractions = accepted.astype(float)
    centroid_locations = local.copy()
    patch_diagnostics = {}
    if coverage == "manifold_patch":
        # Shared buffer indices cannot identify one body unambiguously. Keep
        # those contacts in legacy point mode, without double counting force.
        memberships = np.zeros(len(ids), dtype=int)
        for indices in per_filter_ids:
            memberships[np.searchsorted(ids, np.unique(indices))] += 1
        for column, indices in enumerate(per_filter_ids):
            rows = np.searchsorted(ids, np.unique(indices)).astype(np.int64)
            candidates = rows[
                face[rows]
                & aligned[rows]
                & (weights[rows] > 0)
                & (memberships[rows] == 1)
            ]
            hull = _convex_hull(local[candidates, 1:3])
            support_area, _ = _polygon_area_centroid(hull)
            diagnostic = {
                "method": "point_fallback",
                "fallback_reason": "fewer_than_three_noncollinear_unique_inner_face_points",
                "eligible_contact_count": len(candidates),
                "shared_filter_contact_count": int((memberships[rows] > 1).sum()),
                "support_polygon_yz_m": hull.tolist(),
                "support_area_m2": support_area,
                "overlap_area_m2": None,
                "overlap_fraction": None,
                "overlap_centroid_yz_m": None,
            }
            if support_area > 0:
                overlap_area, patch_centroid = _ellipse_patch_overlap(hull, g)
                overlap_fraction = float(np.clip(overlap_area / support_area, 0, 1))
                fractions[candidates] = overlap_fraction
                if patch_centroid is not None:
                    centroid_locations[candidates, 0] = np.average(
                        local[candidates, 0], weights=compression[candidates]
                    )
                    centroid_locations[candidates, 1:3] = patch_centroid
                diagnostic.update(
                    {
                        "method": "manifold_patch",
                        "fallback_reason": None,
                        "overlap_area_m2": overlap_area,
                        "overlap_fraction": overlap_fraction,
                        "overlap_centroid_yz_m": None
                        if patch_centroid is None
                        else patch_centroid.tolist(),
                    }
                )
            patch_diagnostics[column] = diagnostic
        accepted = fractions > 0
    gel_compression = compression * fractions
    total = float(gel_compression.sum())
    uv = np.zeros(2)
    centroid = None
    if total > 0:
        centroid = np.average(
            centroid_locations[accepted], axis=0, weights=gel_compression[accepted]
        )
        values = {
            axis: (centroid[axis] - g.active_center_yz_m[axis - 1])
            / (g.active_size_yz_m[axis - 1] / 2)
            for axis in (1, 2)
        }
        uv = np.asarray(
            [values[axis] * sign for axis, sign in zip(g.sdk_uv_axes, g.sdk_uv_signs)]
        )
    labels_match = filter_paths is not None and len(filter_paths) == filter_count
    by_filter = []
    for column, indices in enumerate(per_filter_ids):
        indices = np.unique(indices).astype(np.int64)
        if not len(indices):
            continue
        rows = np.searchsorted(ids, indices)
        kept = accepted[rows]
        by_filter.append(
            {
                "filter_column": column,
                "filter_path": filter_paths[column] if labels_match else None,
                "contact_count": len(indices),
                "accepted_contact_count": int(kept.sum()),
                "ignored_contact_count": int((~kept).sum()),
                "normal_force_magnitude_n": float(weights[rows].sum()),
                "gel_compression_n": float(gel_compression[rows].sum()),
                "ignored_normal_force_n": float(
                    (weights[rows] * (1 - fractions[rows])).sum()
                ),
                "contact_points_pad_m": local[rows].tolist(),
                "contact_normals_pad": local_normals[rows].tolist(),
                "normal_force_by_contact_n": weights[rows].tolist(),
                "accepted_by_contact": kept.tolist(),
                "point_inside_active_gel_by_contact": point_accepted[rows].tolist(),
                "gel_force_fraction_by_contact": fractions[rows].tolist(),
                "coverage": patch_diagnostics.get(column, {"method": "point"}),
            }
        )
    return {
        "normal_force_n": total,
        "contact_uv": uv,
        "has_gel_contact": total > 0,
        "accepted_contact_count": int(accepted.sum()),
        "populated_unique_contact_count": len(ids),
        "duplicate_filter_contact_indices": len(populated) - len(ids),
        "observed_filtered_normal_force_n": float(weights.sum()),
        "accepted_contact_normal_magnitude_n": float((weights * fractions).sum()),
        "ignored_contact_normal_force_n": float((weights * (1 - fractions)).sum()),
        "unprojected_accepted_normal_force_n": float(
            (weights * fractions * (1 - alignment)).sum()
        ),
        "ignored_by_reason_n": {
            "not_inner_face": float(weights[~face].sum()),
            "outside_active_ellipse": float(weights[face & ~ellipse].sum()),
            "normal_not_aligned": float(weights[face & ellipse & ~aligned].sum()),
        }
        if coverage == "point"
        else {
            "not_inner_face": float(weights[~face].sum()),
            "outside_active_ellipse": float(
                (weights * (1 - fractions))[face & aligned].sum()
            ),
            "normal_not_aligned": float(weights[face & ~aligned].sum()),
        },
        "contact_centroid_pad_m": None if centroid is None else centroid.tolist(),
        "force_semantics": "positive compression projected onto gel normal; no friction-force estimate",
        "sdk_uv_orientation_calibrated": False,
        "contact_filter_column_count": filter_count,
        "filter_labels_match_columns": labels_match,
        "per_filter_contacts": by_filter,
        "coverage_mode": coverage,
        "coverage_assumption": "point selection"
        if coverage == "point"
        else "uniform pressure on each body's convex inner-face support polygon; sparse manifolds use points; Gaussian proxy remains a spatial approximation",
        "ellipse_polygon_segments": _ELLIPSE_SEGMENTS
        if coverage == "manifold_patch"
        else None,
    }


class GelContactViews:
    """Construct beside existing pad views; initialize after world.reset()."""

    def __init__(
        self,
        pad_paths,
        environment_paths,
        *,
        rigid_prim_cls=None,
        max_contact_count=256,
        geometry=None,
        coverage="point",
    ):
        if len(pad_paths) != 2 or pad_paths[0] == pad_paths[1]:
            raise ValueError("two distinct pad paths required in left/right order")
        if rigid_prim_cls is None:
            from isaacsim.core.prims import RigidPrim

            rigid_prim_cls = RigidPrim
        self.geometry = geometry or GelSurfaceGeometry()
        if coverage not in ("point", "manifold_patch"):
            raise ValueError("coverage must be point or manifold_patch")
        self.coverage = coverage
        self.max_contact_count = int(max_contact_count)
        if self.max_contact_count <= 0:
            raise ValueError("max_contact_count must be positive")
        self.views, self.filter_paths = [], []
        for i, path in enumerate(pad_paths):
            filters = list(
                dict.fromkeys(
                    [p for p in environment_paths if p != path] + [pad_paths[1 - i]]
                )
            )
            self.filter_paths.append(filters)
            self.views.append(
                rigid_prim_cls(
                    prim_paths_expr=path,
                    name=f"gel_contact_{i}",
                    track_contact_forces=True,
                    contact_filter_prim_paths_expr=filters,
                    max_contact_count=self.max_contact_count,
                    reset_xform_properties=False,
                )
            )

    def initialize(self):
        for view in self.views:
            view.initialize()

    def get_all(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("physicsdt must be positive seconds")
        results = []
        for i, view in enumerate(self.views):
            position, quat = view.get_world_poses()
            results.append(
                select_gel_contacts(
                    view.get_contact_force_data(dt=dt),
                    np.asarray(position).reshape(-1, 3)[0],
                    np.asarray(quat).reshape(-1, 4)[0],
                    side=("left", "right")[i],
                    geometry=self.geometry,
                    max_contact_count=self.max_contact_count,
                    filter_paths=self.filter_paths[i],
                    coverage=self.coverage,
                )
            )
        return {
            "normal_force_n": np.asarray([r["normal_force_n"] for r in results]),
            "contact_uv": np.stack([r["contact_uv"] for r in results]),
            "has_gel_contact": np.asarray([r["has_gel_contact"] for r in results]),
            "ignored_normal_force_n": np.asarray(
                [r["ignored_contact_normal_force_n"] for r in results]
            ),
            "per_pad": results,
            "filter_paths": self.filter_paths,
            "coverage_mode": self.coverage,
            "tangential_force_status": "not observed by normal-contact API; do not infer from net force",
        }
