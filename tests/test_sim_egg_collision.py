"""Compound collision must preserve shape, partition volume and cooker budgets."""
from collections import Counter, defaultdict

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from phantom.sim.egg_collision import (
    _distance_to_triangles,
    egg_collision_report,
    egg_compound_collision,
)
from phantom.sim.task_objects import egg_mesh


def plane(points):
    normal = np.cross(points - points[0], np.roll(points, -1, axis=0) - points[0]).sum(axis=0)
    normal /= np.linalg.norm(normal)
    return np.r_[normal, -normal @ points[0]]


@pytest.fixture(scope="module")
def pieces():
    return egg_compound_collision([.043, .043, .057])


def test_every_piece_is_closed_outward_convex_and_within_native_budgets(pieces):
    assert len(pieces) <= 16
    for piece in pieces:
        assert len(piece.points) <= 64
        assert len(piece.faces) <= 64
        assert max(map(len, piece.faces)) <= 32
        edges = Counter((a, b) for face in piece.faces for a, b in zip(face, (*face[1:], face[0])))
        assert all(count == 1 and edges[(b, a)] == 1 for (a, b), count in edges.items())
        for face in piece.faces:
            polygon = piece.points[list(face)]
            equation = plane(polygon)
            np.testing.assert_allclose(polygon @ equation[:3] + equation[3], 0, atol=1e-12)
            assert np.max(piece.points @ equation[:3] + equation[3]) < 1e-12
        assert piece.volume_m3 > 0
        assert piece.volume_m3 == pytest.approx(ConvexHull(piece.points).volume, rel=1e-12)


def test_partition_has_matching_internal_faces_and_no_positive_volume_overlap(pieces):
    face_matches = defaultdict(list)
    for piece in pieces:
        for face in piece.faces:
            polygon = piece.points[list(face)]
            key = tuple(sorted(map(tuple, polygon)))
            face_matches[key].append(plane(polygon))
    assert any(len(matches) == 2 for matches in face_matches.values())
    for matches in face_matches.values():
        assert len(matches) in (1, 2)
        if len(matches) == 2:
            np.testing.assert_allclose(matches[0], -matches[1], atol=1e-12)
    # A positive-radius ball inside two convex pieces would prove overlap.
    # This independent halfspace LP checks every pair, including seams/axis.
    equations = [ConvexHull(piece.points).equations for piece in pieces]
    for i, left in enumerate(equations):
        for right in equations[i + 1:]:
            constraints = np.vstack([left, right])
            result = linprog([0, 0, 0, -1], A_ub=np.c_[constraints[:, :3], np.ones(len(constraints))],
                             b_ub=-constraints[:, 3], bounds=[(None, None)] * 3 + [(0, None)], method="highs")
            assert result.status in (0, 2)
            if result.success:
                assert result.x[3] <= 1e-10
    union = ConvexHull(np.vstack([piece.points for piece in pieces]))
    assert sum(piece.volume_m3 for piece in pieces) == pytest.approx(union.volume, rel=1e-12)


@pytest.mark.parametrize("size,taper", [([.043, .043, .057], .17), ([.0455, .0455, .057], .17),
                                      ([.048, .048, .057], .17), ([.044, .043, .060], .25)])
def test_all_direction_shape_error_is_submillimeter_without_hidden_enlargement(size, taper):
    pieces = egg_compound_collision(size, taper)
    dense, _ = egg_mesh(size, taper=taper)
    dense_points = set(map(tuple, dense))
    for piece in pieces:
        # Only axis-interior vertices may be introduced; exterior is copied,
        # not recomputed at a differently normalized coarse resolution.
        assert all(tuple(point) in dense_points or (point[0] == 0 and point[1] == 0) for point in piece.points)
    report = egg_collision_report(size, pieces, taper)
    assert report["no_exterior_enlargement"]
    assert report["summed_volume_matches_union_hull"]
    assert report["all_direction_support_error_bound_m"] < .00025
    assert 0 < report["volume_loss_fraction"] < .015
    assert report["native_vertex_polygon_budgets_met"]


def test_distance_audit_handles_face_edge_and_corner_nearest_points():
    cube = np.asarray([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    triangles = cube[ConvexHull(cube).simplices]
    actual = _distance_to_triangles(np.array([[0, 0, 2], [0, 2, 2], [2, 2, 2], [1, 0, 0]]), triangles)
    np.testing.assert_allclose(actual, [1, np.sqrt(2), np.sqrt(3), 0], atol=1e-12)


def test_compound_is_deterministic_and_rejects_over_budget_dense_pieces(pieces):
    repeat = egg_compound_collision([.043, .043, .057])
    for left, right in zip(pieces, repeat):
        assert left.name == right.name and left.faces == right.faces
        np.testing.assert_array_equal(left.points, right.points)
    with pytest.raises(ValueError, match="hull budget"):
        egg_compound_collision([.043, .043, .057], latitude_stride=1, azimuth_stride=1)


@pytest.mark.parametrize("kwargs", [
    {"latitude_stride": 3}, {"azimuth_stride": 3}, {"latitude_slabs": 3},
    {"azimuth_sectors": 1}, {"azimuth_sectors": 3}, {"latitude_slabs": True},
    {"azimuth_stride": 0}, {"latitude_stride": 2.0}, {"taper": float("nan")},
])
def test_invalid_resolution_and_taper_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        egg_compound_collision([.043, .043, .057], **kwargs)


@pytest.mark.parametrize("size", [[.043, .043], [.043, 0, .057], [.043, np.inf, .057]])
def test_invalid_dimensions_fail_explicitly(size):
    with pytest.raises(ValueError):
        egg_compound_collision(size)
