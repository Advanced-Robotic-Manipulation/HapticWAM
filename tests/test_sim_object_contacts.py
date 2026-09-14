import numpy as np
import pytest

from tools.sim.object_contacts import convex_support_paths, summarize_pad_object_contacts


def data():
    # Populated entries start at1; unused NaNs must never enter the result.
    force = np.array([np.nan, 2., -3., 0., np.nan])[:, None]
    points = np.array([[np.nan]*3, [1., 0., 0.], [3., 0., 0.], [5., 0., 0.], [np.nan]*3])
    normals = np.array([[np.nan]*3, [1., 0., 0.], [-1., 0., 0.], [0., 0., 0.], [np.nan]*3])
    separations = np.array([np.nan, -.001, .0002, .0003, np.nan])[:, None]
    return [force, points, normals, separations, np.array([[3]]), np.array([[1]])]


def test_opposing_contacts_sum_magnitudes_and_ignore_unused_buffer():
    result = summarize_pad_object_contacts(data(), max_contact_count=256)
    assert result["contact_count"] == 3
    assert result["normal_force_n"] == 5.
    np.testing.assert_allclose(result["centroid_world_m"], [2.2, 0., 0.])


def test_empty_pair_does_not_read_invalid_unused_start_or_buffers():
    value = data()
    value[4] = np.array([[0]])
    value[5] = np.array([[np.nan]])
    assert summarize_pad_object_contacts(value, max_contact_count=256) == {
        "contact_count": 0, "normal_force_n": 0., "centroid_world_m": None
    }


@pytest.mark.parametrize("count", [64, 65])
def test_legacy_capacity_saturation_cannot_be_scored(count):
    value = data()
    value[4] = np.array([[count]])
    with pytest.raises(RuntimeError, match="saturated"):
        summarize_pad_object_contacts(value, max_contact_count=64)


@pytest.mark.parametrize("count", [-1., .5, np.nan, np.inf])
def test_invalid_count_rejected(count):
    value = data()
    value[4] = np.array([[count]])
    with pytest.raises(RuntimeError, match="count"):
        summarize_pad_object_contacts(value, max_contact_count=256)


@pytest.mark.parametrize("start", [-1., .5, np.nan, np.inf, 4.])
def test_invalid_start_or_out_of_range_rejected(start):
    value = data()
    value[5] = np.array([[start]])
    with pytest.raises(RuntimeError, match="start|range"):
        summarize_pad_object_contacts(value, max_contact_count=256)


@pytest.mark.parametrize("buffer", [0, 1, 2, 3])
def test_nonfinite_populated_data_rejected(buffer):
    value = data()
    value[buffer][1] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        summarize_pad_object_contacts(value, max_contact_count=256)


def test_nonzero_load_with_zero_normal_rejected():
    value = data()
    value[2][1] = 0
    with pytest.raises(RuntimeError, match="zero normal"):
        summarize_pad_object_contacts(value, max_contact_count=256)


def test_unexpected_body_filter_count_rejected():
    value = data()
    value[4] = np.array([[1, 2]])
    value[5] = np.array([[1, 2]])
    with pytest.raises(RuntimeError, match="one sensor"):
        summarize_pad_object_contacts(value, max_contact_count=256)


@pytest.mark.parametrize("approximation", ["sdf", "compound_convex_v1"])
def test_multishape_or_sdf_egg_does_not_claim_single_convex_patch(approximation):
    assert convex_support_paths("/World/Egg", {"kind": "egg", "collision_approximation": approximation}) == ()


@pytest.mark.parametrize("path,obj", [
    ("/World/Egg", {"kind": "egg"}),
    ("/World/Egg", {"kind": "egg", "collision_approximation": "convexHull"}),
    ("/World/Carton", {"kind": "carton"}),
    ("/World/Waffle", {"kind": "waffle"}),
])
def test_existing_convex_declarations_preserved(path, obj):
    assert convex_support_paths(path, obj) == (path,)
