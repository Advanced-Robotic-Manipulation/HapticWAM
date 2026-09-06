"""Geometric/force invariants for sparse rigid contact manifold coverage."""

import json
from pathlib import Path

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


@pytest.mark.parametrize("coverage", ["manifold_patch", "manifold_patch_v2"])
def test_view_routes_coverage_to_selector_without_unsupported_native_keyword(coverage):
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
            self.filter_count = len(contact_filter_prim_paths_expr)

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
                counts=[4] + [0] * (self.filter_count - 1),
            )

    view = GelContactViews(
        ["/left", "/right"],
        ["/World/Waffle"],
        rigid_prim_cls=FakeView,
        coverage=coverage,
    )
    view.initialize()
    result = view.get_all(0.004)
    assert result["coverage_mode"] == coverage
    np.testing.assert_allclose(result["normal_force_n"], [np.pi, np.pi], rtol=3e-5)
    with pytest.raises(ValueError, match="coverage"):
        GelContactViews(
            ["/left", "/right"], [], rigid_prim_cls=FakeView, coverage="invalid"
        )


def select_v2(data, **kwargs):
    return select_gel_contacts(
        data,
        [0, 0, 0],
        [1, 0, 0, 0],
        side="left",
        coverage="manifold_patch_v2",
        filter_paths=["/World/Waffle"],
        **kwargs,
    )


RECTANGLE = np.array(
    [[-0.0135, -0.018], [0.0135, -0.018], [0.0135, 0.018], [-0.0135, 0.018]]
)


@pytest.mark.parametrize(
    "forces", [[1, 1, 1, 1], [2, 0, 2, 0], [4, 0, 0, 0], [0, 0, 0, 4]]
)
def test_v2_support_is_independent_of_constraint_load_distribution(forces):
    result = select_v2(contact_data(RECTANGLE, forces=forces))
    assert result["normal_force_n"] == pytest.approx(np.pi, rel=3e-5)
    np.testing.assert_allclose(result["contact_uv"], [0, 0], atol=1e-12)
    assert result["accepted_contact_count"] == np.count_nonzero(forces)
    assert result["per_filter_contacts"][0]["coverage"]["eligible_contact_count"] == 4
    assert_force_budget(result)


def test_v2_removes_zero_impulse_branch_discontinuity_without_adding_load():
    force = []
    for epsilon in [0, 1e-12, 1e-8, 1e-4]:
        data = contact_data(
            RECTANGLE, forces=[2 - epsilon, epsilon, 2 - epsilon, epsilon]
        )
        force.append(select_v2(data)["normal_force_n"])
    np.testing.assert_allclose(force, force[0], rtol=1e-14, atol=1e-14)
    assert select(contact_data(RECTANGLE, forces=[2, 0, 2, 0]))["normal_force_n"] == 0
    assert force[0] > 3
    no_load = select_v2(contact_data(RECTANGLE, forces=[0, 0, 0, 0]))
    assert no_load["normal_force_n"] == 0
    assert (
        no_load["per_filter_contacts"][0]["coverage"]["fallback_reason"]
        == "no_positive_eligible_load"
    )
    assert_force_budget(no_load)


@pytest.mark.parametrize("separation", [-0.001, 0, 0.001, 0.002])
def test_v2_fixed_contact_offset_envelope_is_not_force_dependent(separation):
    data = list(contact_data(RECTANGLE, forces=[2, 0, 2, 0]))
    data[3][:] = separation
    result = select_v2(data)
    assert result["normal_force_n"] == pytest.approx(np.pi, rel=3e-5)
    assert (
        result["per_filter_contacts"][0]["separation_by_contact_m"] == [separation] * 4
    )
    assert_force_budget(result)


def test_v2_excludes_speculative_vertices_beyond_authored_contact_envelope():
    data = list(contact_data(RECTANGLE, forces=[2, 0, 2, 0]))
    data[3][1::2] = 0.002001
    result = select_v2(data)
    assert result["normal_force_n"] == 0
    patch = result["per_filter_contacts"][0]["coverage"]
    assert patch["eligible_contact_count"] == 2
    assert patch["method"] == "point_fallback"
    assert_force_budget(result)


def test_v2_excluded_loaded_separation_stays_in_accounted_ignored_force():
    data = list(contact_data([[0, 0]], forces=[10]))
    data[3][:] = 0.004
    result = select_v2(data)
    assert result["normal_force_n"] == 0
    assert result["ignored_by_reason_n"]["outside_support_separation"] == 10
    assert_force_budget(result)


def test_v2_missing_separations_are_not_replaced_with_fabricated_zeros():
    data = list(contact_data(RECTANGLE, forces=[2, 0, 2, 0]))
    data[3] = None
    result = select_v2(data)
    assert result["normal_force_n"] == 0
    assert result["separation_status"] == "missing"
    assert (
        result["per_filter_contacts"][0]["coverage"]["fallback_reason"]
        == "separation_evidence_missing"
    )
    assert result["per_filter_contacts"][0]["separation_by_contact_m"] is None
    assert_force_budget(result)


def test_v2_unknown_multishape_body_does_not_bridge_disconnected_contacts():
    low = np.array([[-0.025, -0.005], [-0.020, -0.005], [-0.022, 0.005]])
    data = contact_data(np.r_[low, low * [-1, 1]])
    result = select_gel_contacts(
        data,
        [0, 0, 0],
        [1, 0, 0, 0],
        side="left",
        coverage="manifold_patch_v2",
        filter_paths=["/World/Robot/MultiColliderBody"],
    )
    assert result["normal_force_n"] == 0
    patch = result["per_filter_contacts"][0]["coverage"]
    assert patch["fallback_reason"] == "single_convex_contact_body_not_declared"
    assert patch["single_convex_body_declared"] is False
    assert_force_budget(result)


def test_v2_incoherent_surface_normals_do_not_invent_one_pressure_patch():
    data = contact_data(RECTANGLE, normals=[[0.8, 0.6, 0]] * 2 + [[0.8, -0.6, 0]] * 2)
    result = select_v2(data)
    assert result["normal_force_n"] == 0
    assert (
        result["per_filter_contacts"][0]["coverage"]["fallback_reason"]
        == "incoherent_contact_normals"
    )
    assert_force_budget(result)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_v2_populated_invalid_separation_fails_loudly(bad):
    data = list(contact_data(RECTANGLE))
    data[3][0] = bad
    with pytest.raises(RuntimeError, match="nonfinite populated.*separation"):
        select_v2(data)


def test_v2_zero_normal_zero_impulse_vertex_cannot_create_support():
    data = list(contact_data(RECTANGLE, forces=[2, 0, 2, 0]))
    data[2][1::2] = 0
    result = select_v2(data)
    assert result["normal_force_n"] == 0
    assert result["per_filter_contacts"][0]["coverage"]["eligible_contact_count"] == 2
    data[2][0] = 0
    with pytest.raises(RuntimeError, match="loaded.*zero normal"):
        select_v2(data)


def test_v2_reads_only_populated_contact_buffer_ranges_and_keeps_provenance():
    original = contact_data(RECTANGLE, forces=[2, 0, 2, 0])
    data = list(original)
    for i in range(4):
        shape = (2,) if i in (0, 3) else (2, 3)
        data[i] = np.concatenate(
            (np.full(shape, np.nan), original[i], np.full(shape, np.nan))
        )
    data[5] = np.array([[2]])
    result = select_v2(data)
    assert result["normal_force_n"] == pytest.approx(np.pi, rel=3e-5)
    assert result["populated_buffer_indices"] == [2, 3, 4, 5]
    with pytest.raises(RuntimeError, match="saturated"):
        select_v2(data, max_contact_count=4)


def test_v2_permutation_and_repeated_filter_indices_preserve_total_force():
    original = contact_data(RECTANGLE, forces=[2, 0, 2, 0])
    order = [2, 0, 3, 1]
    permuted = [np.asarray(value)[order] for value in original[:4]] + list(original[4:])
    expected = select_v2(original)
    actual = select_v2(permuted)
    assert actual["normal_force_n"] == expected["normal_force_n"]
    np.testing.assert_array_equal(actual["contact_uv"], expected["contact_uv"])
    duplicate = list(contact_data([[0, 0], [0.001, 0.001], [-0.001, 0.001]]))
    duplicate[4], duplicate[5] = np.array([[3, 3]]), np.array([[0, 0]])
    result = select_gel_contacts(
        duplicate,
        [0, 0, 0],
        [1, 0, 0, 0],
        side="left",
        coverage="manifold_patch_v2",
        filter_paths=["/World/Waffle", "/World/Alias"],
    )
    assert result["normal_force_n"] == 3
    assert result["populated_unique_contact_count"] == 3
    assert (
        result["per_filter_contacts"][0]["coverage"]["fallback_reason"]
        == "ambiguous_shared_filter_indices"
    )
    assert_force_budget(result)


def test_v2_empty_buffer_and_large_real_load_remain_finite_and_unclipped():
    empty = select_v2(contact_data([]))
    assert empty["normal_force_n"] == 0
    assert_force_budget(empty)
    data = list(contact_data([[0, 0], [0.001, 0], [0, 0.001]], forces=[150, 0, 0]))
    result = select_v2(data)
    assert result["normal_force_n"] == 150
    assert_force_budget(result)
    data[1][0, 0] = 0.008
    result = select_v2(data)
    assert result["normal_force_n"] == 0
    assert result["ignored_by_reason_n"]["not_inner_face"] == 150
    assert_force_budget(result)


@pytest.mark.parametrize("side", ["left", "right"])
def test_v2_rigid_transform_and_mirrored_side_preserve_projected_force(side):
    data = list(
        contact_data(RECTANGLE, forces=[2, 0, 2, 0], normals=[[0.999, 0.02, 0]] * 4)
    )
    if side == "right":
        data[1][:, 0] *= -1
    rotation = Rotation.from_euler("xyz", [0.7, -0.6, 1.2])
    position = np.array([0.4, -0.2, 0.6])
    data[1] = rotation.apply(data[1]) + position
    data[2] = -rotation.apply(data[2])
    result = select_gel_contacts(
        data,
        position,
        rotation.as_quat()[[3, 0, 1, 2]],
        side=side,
        coverage="manifold_patch_v2",
        filter_paths=["/World/Waffle"],
    )
    projection = 0.999 / np.linalg.norm([0.999, 0.02])
    assert result["normal_force_n"] == pytest.approx(np.pi * projection, rel=3e-5)
    np.testing.assert_allclose(result["contact_uv"], [0, 0], atol=1e-12)
    assert_force_budget(result)


def test_v1_archived_carry_readouts_remain_exact_and_v2_reports_missing_evidence():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/gel_contact_v1_carry.json").read_text()
    )
    for row in fixture["cases"]:
        records = row["per_filter_contacts"]
        counts = [len(r["normal_force_by_contact_n"]) for r in records]
        data = (
            np.concatenate([r["normal_force_by_contact_n"] for r in records]),
            np.concatenate([r["contact_points_pad_m"] for r in records]),
            np.concatenate([r["contact_normals_pad"] for r in records]),
            None,
            np.array([counts]),
            np.array([np.r_[0, np.cumsum(counts)[:-1]]]),
        )
        kwargs = {
            "side": row["side"],
            "filter_paths": [r["filter_path"] for r in records],
        }
        v1 = select_gel_contacts(
            data, [0, 0, 0], [1, 0, 0, 0], coverage="manifold_patch", **kwargs
        )
        assert v1["normal_force_n"] == row["expected_v1_force_n"]
        np.testing.assert_array_equal(v1["contact_uv"], row["expected_v1_uv"])
        v2 = select_gel_contacts(
            data, [0, 0, 0], [1, 0, 0, 0], coverage="manifold_patch_v2", **kwargs
        )
        assert v2["separation_status"] == "missing"
        assert all(
            r["coverage"]["fallback_reason"] == "separation_evidence_missing"
            for r in v2["per_filter_contacts"]
        )
        assert_force_budget(v2)
