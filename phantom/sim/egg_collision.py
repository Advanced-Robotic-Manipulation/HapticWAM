"""Deterministic convex pieces for the existing egg surface, without inflation.

PhysX can reduce one dense ovoid to a much coarser 64-polygon hull. These
pieces partition an inscribed surface into latitude slabs and azimuth wedges;
each stays below both native hull vertex and polygon limits. Faces are planar
polygons, not a claim that the *triangulated* mesh has at most 64 triangles.

Attach all pieces beneath the existing single rigid body. This module creates
geometry only: no mass changes, extra bodies, joints, offsets or attachments.
The dense rendered surface can remain unchanged. The native CPU cook probe
preserves the required accuracy when mesh coordinates are multiplied by 100
and compensated by a local 0.01 scale; plain metre coordinates still simplify
too aggressively. This reciprocal authoring transform does not enlarge the
physical object. Contact behavior still requires native replay validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phantom.sim.task_objects import egg_mesh


PHYSX_MESH_COORDINATE_SCALE = 100.0


@dataclass(frozen=True)
class EggCollisionPiece:
    """One closed convex mesh with outward, planar polygon faces."""

    name: str
    points: np.ndarray
    faces: tuple[tuple[int, ...], ...]

    @property
    def triangles(self) -> np.ndarray:
        return np.asarray(
            [[face[0], face[i], face[i + 1]] for face in self.faces for i in range(1, len(face) - 1)],
            dtype=np.int32,
        )

    @property
    def volume_m3(self) -> float:
        triangles = self.points[self.triangles]
        return float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def egg_compound_collision(
    size,
    taper=0.17,
    *,
    latitude_stride=2,
    azimuth_stride=2,
    latitude_slabs=2,
    azimuth_sectors=8,
) -> tuple[EggCollisionPiece, ...]:
    """Partition an inscribed egg into 16 pieces by default.

    Exterior vertices are copied verbatim from ``egg_mesh(size, 40, 64)``.
    Interior vertices lie on the egg axis at shared slab boundaries. The
    default shell retains 20 latitude intervals and 32 azimuth intervals.
    Adjacent pieces share their complete boundaries and have disjoint
    interiors. Their union is the convex hull of the retained surface.

    Strides must divide the dense source resolution. Partitions must divide
    the retained intervals, and sectors must span at most a half-plane so
    every wedge is convex. Over-budget candidates are rejected explicitly.
    """
    latitude_stride = _positive_integer(latitude_stride, "latitude_stride")
    azimuth_stride = _positive_integer(azimuth_stride, "azimuth_stride")
    latitude_slabs = _positive_integer(latitude_slabs, "latitude_slabs")
    azimuth_sectors = _positive_integer(azimuth_sectors, "azimuth_sectors")
    if 40 % latitude_stride or 64 % azimuth_stride:
        raise ValueError("Strides must divide the dense 40-latitude / 64-azimuth source")
    rings, segments = 40 // latitude_stride, 64 // azimuth_stride
    if rings < 8 or segments < 12 or rings % latitude_slabs or segments % azimuth_sectors or azimuth_sectors < 2:
        raise ValueError("Partitions must divide retained intervals; need at least 8 latitudes, 12 azimuths and 2 sectors")
    source, _ = egg_mesh(size, rings=40, segments=64, taper=taper)
    grid = np.empty((rings + 1, segments, 3), dtype=float)
    grid[0], grid[-1] = source[0], source[-1]
    for ring in range(1, rings):
        indices = 1 + (ring * latitude_stride - 1) * 64 + np.arange(segments) * azimuth_stride
        grid[ring] = source[indices]
    rings_per_slab, segments_per_sector = rings // latitude_slabs, segments // azimuth_sectors
    pieces = []
    for slab in range(latitude_slabs):
        first_ring, last_ring = slab * rings_per_slab, (slab + 1) * rings_per_slab
        for sector in range(azimuth_sectors):
            points, faces, point_indices = [], [], {}

            def add_point(point):
                key = tuple(float(x) for x in point)
                if key not in point_indices:
                    point_indices[key] = len(points)
                    points.append(key)
                return point_indices[key]

            rows = []
            for ring in range(first_ring, last_ring + 1):
                rows.append([
                    add_point(grid[ring, (sector * segments_per_sector + column) % segments])
                    for column in range(segments_per_sector + 1)
                ])
            top_axis = add_point([0., 0., grid[first_ring, 0, 2]])
            bottom_axis = add_point([0., 0., grid[last_ring, 0, 2]])
            for row in range(rings_per_slab):
                for column in range(segments_per_sector):
                    faces.append([rows[row][column], rows[row + 1][column], rows[row + 1][column + 1], rows[row][column + 1]])
            faces.extend([
                [top_axis, *rows[0]],
                [bottom_axis, *reversed(rows[-1])],
                [top_axis, *(row[0] for row in rows), bottom_axis],
                [top_axis, *(row[-1] for row in rows), bottom_axis],
            ])
            points = np.asarray(points, dtype=float)
            center = points.mean(axis=0)
            outward_faces = []
            for face in faces:
                # Collapse repeated pole/axis indices while retaining order.
                face = list(dict.fromkeys(face))
                if len(face) < 3:
                    continue
                vertices = points[face]
                normal = np.cross(vertices - vertices[0], np.roll(vertices, -1, axis=0) - vertices[0]).sum(axis=0)
                if np.linalg.norm(normal) <= 1e-18:
                    raise ValueError("Degenerate compound egg face")
                if np.dot(normal, vertices.mean(axis=0) - center) < 0:
                    face.reverse()
                outward_faces.append(tuple(face))
            piece = EggCollisionPiece(f"slab_{slab:02d}_sector_{sector:02d}", points, tuple(outward_faces))
            if len(points) > 64 or len(piece.faces) > 64:
                raise ValueError(f"{piece.name} exceeds native hull budget: {len(points)} vertices, {len(piece.faces)} polygons; use more pieces or larger strides")
            if max(map(len, piece.faces)) > 32:
                raise ValueError(f"{piece.name} exceeds the 32-vertices-per-polygon budget")
            pieces.append(piece)
    return tuple(pieces)


def _distance_to_triangles(points, triangles):
    """Exact unsigned distance to a triangle surface, in bounded-size batches."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ab2 = np.einsum("ij,ij->i", ab, ab)
    ac2 = np.einsum("ij,ij->i", ac, ac)
    abac = np.einsum("ij,ij->i", ab, ac)
    denominator = ab2 * ac2 - abac ** 2
    normal = np.cross(ab, ac)
    normal2 = np.einsum("ij,ij->i", normal, normal)
    if np.any(denominator <= 0):
        raise ValueError("Cannot measure a degenerate triangle surface")
    result = []
    for start in range(0, len(points), 32):
        block = points[start:start + 32]
        ap = block[:, None, :] - a
        abp = np.einsum("pti,ti->pt", ap, ab)
        acp = np.einsum("pti,ti->pt", ap, ac)
        u = (ac2 * abp - abac * acp) / denominator
        v = (ab2 * acp - abac * abp) / denominator
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        plane_distance2 = np.einsum("pti,ti->pt", ap, normal) ** 2 / normal2
        distance2 = np.where(inside, plane_distance2, np.inf)
        for left, right in [(a, b), (b, c), (c, a)]:
            edge = right - left
            delta = block[:, None, :] - left
            amount = np.clip(np.einsum("pti,ti->pt", delta, edge) / np.einsum("ti,ti->t", edge, edge), 0, 1)
            residual = delta - amount[:, :, None] * edge
            distance2 = np.minimum(distance2, np.einsum("pti,pti->pt", residual, residual))
        result.extend(np.sqrt(distance2.min(axis=1)))
    return np.asarray(result)


def egg_collision_report(size, pieces, taper=0.17) -> dict:
    """Quantify uncooked shape, volume and budgets against the dense egg.

    For the convex inscribed union the maximum dense-vertex distance to its
    hull is the continuous Hausdorff distance: distance to a convex set is
    convex, so a maximum over the dense polytope occurs at a vertex. This is
    also a bound on support error in *every* direction, unlike a finite ray
    sample. For arbitrary replacement/cooked pieces these hull metrics alone
    cannot establish closed seams or absence of gaps between the pieces. The
    report is not evidence of what PhysX actually cooked.
    """
    from scipy.spatial import ConvexHull

    if not pieces:
        raise ValueError("Need at least one egg collision piece")
    source, _ = egg_mesh(size, rings=40, segments=64, taper=taper)
    points = np.unique(np.vstack([piece.points for piece in pieces]), axis=0)
    dense_hull, compound_hull = ConvexHull(source), ConvexHull(points)
    source_halfspaces = source @ compound_hull.equations[:, :3].T + compound_hull.equations[:, 3]
    source_outside = source_halfspaces.max(axis=1) > 1e-12
    distances = np.zeros(len(source))
    distances[source_outside] = _distance_to_triangles(source[source_outside], points[compound_hull.simplices])
    # Exterior source vertices and interior axis points must all remain inside
    # the source hull. This explicit check catches accidental renormalization.
    piece_halfspaces = points @ dense_hull.equations[:, :3].T + dense_hull.equations[:, 3]
    outside = float(np.max(piece_halfspaces))
    piece_outside = piece_halfspaces.max(axis=1) > 1e-12
    outward_distances = np.zeros(len(points))
    outward_distances[piece_outside] = _distance_to_triangles(points[piece_outside], source[dense_hull.simplices])
    volume = sum(piece.volume_m3 for piece in pieces)
    return {
        "scope": "Uncooked geometry only; native cooking and contact dynamics require separate validation",
        "shape_metric_geometry": "Convex hull of all piece vertices versus dense source hull. Equality with the piece union additionally requires closed seams and a valid partition; source builder and tests establish this for authored pieces only.",
        "piece_count": len(pieces),
        "piece_vertex_counts": [len(piece.points) for piece in pieces],
        "piece_polygon_counts": [len(piece.faces) for piece in pieces],
        "piece_triangle_counts": [len(piece.triangles) for piece in pieces],
        "max_vertices_per_polygon": max(len(face) for piece in pieces for face in piece.faces),
        "piece_volumes_m3": [piece.volume_m3 for piece in pieces],
        "compound_volume_m3": volume,
        "compound_union_hull_volume_m3": float(compound_hull.volume),
        "dense_reference_volume_m3": float(dense_hull.volume),
        "volume_loss_fraction": float(1 - volume / dense_hull.volume),
        "maximum_outward_halfspace_violation_m": outside,
        "dense_to_compound_hausdorff_m": float(distances.max()),
        "compound_hull_to_dense_hausdorff_m": float(outward_distances.max()),
        "all_direction_support_error_bound_m": float(max(distances.max(), outward_distances.max())),
        "worst_dense_vertex_m": source[int(np.argmax(distances))].tolist(),
        "native_vertex_polygon_budgets_met": all(len(piece.points) <= 64 and len(piece.faces) <= 64 for piece in pieces),
        "summed_volume_matches_union_hull": bool(np.isclose(volume, compound_hull.volume, rtol=1e-10, atol=1e-16)),
        "no_exterior_enlargement": bool(outside <= 1e-12),
    }
