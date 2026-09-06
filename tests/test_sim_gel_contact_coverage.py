"""Geometric/force invariants for sparse rigid contact manifold coverage."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.sim.gel_contact import GelContactViews, select_gel_contacts


def contact_data(yz, forces=None, counts=None, face=-0.006, normals=None):
    yz = np.asarray(yz, dtype=float).reshape(-1, 2)
    count = len(yz)
    counts = [count] if counts is None else counts
    return (
        np.ones(count) if forces is None else np.asarray(forces, dtype=float),
        np.column_stack((np.full(count, face), yz)),
        np.tile([1.0, 0, 0], (count, 1)) if normals is None else np.asarray(normals),
        np.zeros(count),
        np.array([counts]),
        np.array([np.r_[0, np.cumsum(counts)[:-1]]]),
    )


def select(data, **kwargs):
    return select_gel_contacts(
        data, [0, 0, 0], [1, 0, 0, 0], side="left", coverage="manifold_patch", **kwargs
    )


def assert_force_budget(result):
    assert result["normal_force_n"] >= 0
    assert result["ignored_contact_normal_force_n"] >= 0
    assert result["accepted_contact_normal_magnitude_n"] + result[
        "ignored_contact_normal_force_n"
    ] == pytest.approx(result["observed_filtered_normal_force_n"])
    assert result["normal_force_n"] + result[
        "unprojected_accepted_normal_force_n"
    ] == pytest.approx(result["accepted_contact_normal_magnitude_n"])
    assert sum(result["ignored_by_reason_n"].values()) == pytest.approx(
        result["ignored_contact_normal_force_n"]
    )


def test_corner_manifold_covers_ellipse_despite_no_point_inside():
    # Circumscribed rectangle, uniform pressure: exact area fraction pi/4.
    data = contact_data(
        [[-0.0135, -0.018], [0.0135, -0.018], [0.0135, 0.018], [-0.0135, 0.018]]
    )
    point = select_gel_contacts(data, [0, 0, 0], [1, 0, 0, 0], side="left")
    result = select(data)
    assert point["normal_force_n"] == 0
    assert result["normal_force_n"] == pytest.approx(np.pi, rel=3e-5)
    assert result["normal_force_n"] <= np.pi  # inscribed approximation
    np.testing.assert_allclose(result["contact_uv"], [0, 0], atol=1e-12)
    assert result["per_filter_contacts"][0]["coverage"]["method"] == "manifold_patch"
    assert_force_budget(result)


def test_half_ellipse_has_correct_force_and_pressure_centroid():
    result = select(
        contact_data([[0, -0.018], [0.0135, -0.018], [0.0135, 0.018], [0, 0.018]])
    )
    assert result["normal_force_n"] == pytest.approx(np.pi, rel=3e-5)
    # Half ellipse y>=0 has centroid y=4*radius_y/(3*pi).
    np.testing.assert_allclose(result["contact_uv"], [0, 4 / (3 * np.pi)], atol=2e-5)
    assert_force_budget(result)


@pytest.mark.parametrize("offset, expected", [(0, 4), (0.1, 0)])
def test_support_entirely_inside_or_disjoint(offset, expected):
    yz = np.array([[-0.002, -0.003], [0.002, -0.003], [0.002, 0.003], [-0.002, 0.003]])
    result = select(contact_data(yz + offset))
    assert result["normal_force_n"] == pytest.approx(expected)
    assert_force_budget(result)


def test_different_bodies_cannot_bridge_an_empty_gel_interior():
    low = np.array([[-0.025, -0.005], [-0.020, -0.005], [-0.022, 0.005]])
    high = low * [-1, 1]
    result = select(
        contact_data(np.r_[low, high], counts=[3, 3]), filter_paths=["/a", "/b"]
    )
    assert result["normal_force_n"] == 0
    assert len(result["per_filter_contacts"]) == 2
    assert_force_budget(result)


@pytest.mark.parametrize(
    "yz", [[[0, 0]], [[0, 0], [0, 0.03]], [[0, 0], [0, 0.01], [0, 0.03]]]
)
def test_sparse_and_collinear_manifolds_explicitly_preserve_point_fallback(yz):
    data = contact_data(yz)
    result = select(data)
    point = select_gel_contacts(data, [0, 0, 0], [1, 0, 0, 0], side="left")
    assert result["normal_force_n"] == point["normal_force_n"]
    np.testing.assert_array_equal(result["contact_uv"], point["contact_uv"])
    assert result["per_filter_contacts"][0]["coverage"]["method"] == "point_fallback"
    assert_force_budget(result)


@pytest.mark.parametrize("side,face", [("left", -0.006), ("right", 0.006)])
def test_force_projection_and_world_pose_preserve_patch_geometry(side, face):
    yz = [[-0.0135, -0.018], [0.0135, -0.018], [0.0135, 0.018], [-0.0135, 0.018]]
    data = list(
        contact_data(yz, forces=[1, 2, 3, 4], face=face, normals=[[0.8, 0.6, 0]] * 4)
    )
    rotation = Rotation.from_euler("xyz", [0.7, -0.6, 1.2])
    position = np.array([0.4, -0.2, 0.6])
    data[1] = rotation.apply(data[1]) + position
    data[2] = -rotation.apply(data[2])  # API normal polarity is immaterial.
    result = select_gel_contacts(
        data,
        position,
        rotation.as_quat()[[3, 0, 1, 2]],
        side=side,
        coverage="manifold_patch",
    )
    assert result["normal_force_n"] == pytest.approx(10 * 0.8 * np.pi / 4, rel=3e-5)
    np.testing.assert_allclose(result["contact_uv"], [0, 0], atol=1e-12)
    assert_force_budget(result)


def test_backing_and_unaligned_contacts_do_not_inflate_patch_force():
    data = list(
        contact_data(
            [[-0.002, -0.003], [0.002, -0.003], [0, 0.003], [0, 0], [0, 0]],
            forces=[1, 2, 3, 50, 100],
        )
    )
    data[1][3, 0] = 0.008  # Backing face is outside inner-face tolerance.
    data[2][4] = [0, 1, 0]  # Tangential edge normal is not a gel normal.
    result = select(data)
    assert result["normal_force_n"] == pytest.approx(6)
    assert result["ignored_contact_normal_force_n"] == pytest.approx(150)
    assert_force_budget(result)


def test_duplicate_filter_indices_do_not_duplicate_force_or_invent_patch():
    data = list(contact_data([[0, 0], [0.001, 0.001], [-0.001, 0.001]]))
    data[4] = np.array([[3, 3]])
    data[5] = np.array([[0, 0]])
    result = select(data)
    assert result["normal_force_n"] == 3
    assert result["duplicate_filter_contact_indices"] == 3
    assert all(
        x["coverage"]["method"] == "point_fallback"
        for x in result["per_filter_contacts"]
    )
    assert_force_budget(result)


def test_empty_contact_buffers_are_zero_and_finite():
    result = select(contact_data([]))
    assert result["normal_force_n"] == 0
    np.testing.assert_array_equal(result["contact_uv"], [0, 0])
    assert_force_budget(result)


def test_view_routes_coverage_to_selector_without_unsupported_native_keyword():
    class FakeView:
        def __init__(
            self,
            *,
            prim_paths_expr,
            name,
            track_contact_forces,
            contact_filter_prim_paths_expr,
            max_contact_count,
            reset_xform_properties,
        ):
            self.face = -0.006 if prim_paths_expr == "/left" else 0.006

        def initialize(self):
            pass

        def get_world_poses(self):
            return np.array([[0, 0, 0]]), np.array([[1, 0, 0, 0]])

        def get_contact_force_data(self, dt):
            assert dt == 0.004
            return contact_data(
                [
                    [-0.0135, -0.018],
                    [0.0135, -0.018],
                    [0.0135, 0.018],
                    [-0.0135, 0.018],
                ],
                face=self.face,
            )

    view = GelContactViews(
        ["/left", "/right"], [], rigid_prim_cls=FakeView, coverage="manifold_patch"
    )
    view.initialize()
    result = view.get_all(0.004)
    assert result["coverage_mode"] == "manifold_patch"
    np.testing.assert_allclose(result["normal_force_n"], [np.pi, np.pi], rtol=3e-5)
    with pytest.raises(ValueError, match="coverage"):
        GelContactViews(
            ["/left", "/right"], [], rigid_prim_cls=FakeView, coverage="invalid"
        )
