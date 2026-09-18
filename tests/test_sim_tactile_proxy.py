"""Physical sensor-contract invariants; no Isaac or recorded data required."""

import numpy as np
import pytest

from phantom.sim.tactile_proxy import (
    MeasuredBaselineTactileProxy,
    TactileProxyParameters,
)


@pytest.fixture
def proxy(tmp_path):
    y, x = np.indices((288, 384))
    data = {}
    for side in ("left", "right"):
        data[f"{side}_infer_img"] = ((x * 13 + y * 7) % 180 + 30).astype(np.uint8)
        data[f"{side}_fields_ds"] = np.zeros((72, 96, 8), np.float32)
        data[f"{side}_keyframes"] = np.zeros((144, 192, 8), np.float32)
        data[f"{side}_wrench"] = np.array([0.2, -0.1, -1.0, 0.01, 0, 0], np.float32)
        data[f"{side}_area"] = np.float32(0.25)
    path = tmp_path / "baseline.npz"
    np.savez(path, **data)
    return MeasuredBaselineTactileProxy(path), data


@pytest.mark.requires_cv2
def test_no_contact_preserves_observed_baseline_and_release(proxy):
    model, baseline = proxy
    model.synthesize([3, 4], tangential_displacement_sdk_mm=[[1, 0], [0, 1]])
    result = model.synthesize([0, 0], tangential_displacement_sdk_mm=[[1, 0], [0, 1]])
    for i, side in enumerate(("left", "right")):
        np.testing.assert_array_equal(result.gel[i], baseline[f"{side}_infer_img"])
        np.testing.assert_array_equal(
            result.fields_ds[i], baseline[f"{side}_fields_ds"]
        )
        np.testing.assert_array_equal(
            result.keyframes[i], baseline[f"{side}_keyframes"]
        )
        np.testing.assert_array_equal(result.wrench[i], baseline[f"{side}_wrench"])
    np.testing.assert_array_equal(result.area, [0.25, 0.25])


@pytest.mark.requires_cv2
@pytest.mark.parametrize("uv", [[[0, 0], [0.1, -0.2]], [[1, 1], [-1, -1]]])
def test_normal_and_shear_force_conservation_at_both_resolutions(proxy, uv):
    model, _ = proxy
    result = model.synthesize([3, 5], uv, tangential_force_sdk_n=[[1, -2], [-1.5, 0.5]])
    expected = np.array([[1, -2, -3], [-1.5, 0.5, -5]])
    for fields in (result.fields_ds, result.keyframes):
        force = fields[..., 5:8].astype(np.float64).mean(axis=(1, 2)) * (288 * 384)
        np.testing.assert_allclose(force, expected, atol=1e-6)
    np.testing.assert_allclose(
        result.wrench[:, :3] - [0.2, -0.1, -1], expected, atol=1e-6
    )


@pytest.mark.requires_cv2
def test_contact_position_moves_force_centroid_in_sdk_image_axes(proxy):
    model, _ = proxy
    result = model.synthesize([2, 2], [[-0.4, 0.3], [0.4, -0.3]])
    yy, xx = np.meshgrid(
        (np.arange(72) + 0.5) / 72 * 2 - 1,
        (np.arange(96) + 0.5) / 96 * 2 - 1,
        indexing="ij",
    )
    for i, uv in enumerate(([-0.4, 0.3], [0.4, -0.3])):
        weights = -result.fields_ds[i, ..., 7]
        centroid = (
            np.array([(xx * weights).sum(), (yy * weights).sum()]) / weights.sum()
        )
        np.testing.assert_allclose(centroid, uv, atol=0.003)


@pytest.mark.requires_cv2
def test_depth_in_mm_scales_with_force_without_safety_hiding_cap(proxy):
    model, _ = proxy
    one = model.synthesize([1, 1])
    many = model.synthesize([20, 20])
    np.testing.assert_allclose(
        one.depth_peaks_mm, model.parameters.depth_mm_per_n, rtol=1e-6
    )
    np.testing.assert_allclose(many.depth_peaks_mm, one.depth_peaks_mm * 20, rtol=1e-6)
    assert np.all(many.depth_peaks_mm > 0.6)


@pytest.mark.requires_cv2
def test_torque_uses_contact_lever_arm_and_raw_sdk_torque_units(proxy):
    model, _ = proxy
    center = model.synthesize([2, 2])
    shifted = model.synthesize([2, 2], [[0.5, 0], [0, 0.5]])
    # r=(9mm,0,0),F=(0,0,-2N) => My=0.018Nm=1.8 raw SDK units.
    np.testing.assert_allclose(
        shifted.wrench[0, 3:] - center.wrench[0, 3:], [0, 1.8, 0], atol=1e-6
    )
    np.testing.assert_allclose(
        shifted.wrench[1, 3:] - center.wrench[1, 3:], [-1.35, 0, 0], atol=1e-6
    )


@pytest.mark.requires_cv2
def test_sensor_warp_preserves_image_range_and_has_no_added_brightness_blob(proxy):
    model, data = proxy
    frame = model.synthesize(
        [3, 0], tangential_displacement_sdk_mm=[[0.2, -0.1], [0, 0]]
    )
    assert frame.gel.dtype == np.uint8
    assert frame.gel.min() >= 30 and frame.gel.max() < 210
    assert np.mean(abs(frame.gel[0].astype(float) - data["left_infer_img"])) > 0
    np.testing.assert_array_equal(frame.gel[1], data["right_infer_img"])
    sensor = frame.as_sensor_dict(1.25)
    assert sensor["left"]["keyframe"].shape == (144, 192, 8)
    assert sensor["right"]["t"] == 1.25


@pytest.mark.requires_cv2
def test_optional_measured_contact_area_is_in_square_millimetres(proxy):
    model, _ = proxy
    result = model.synthesize([2, 3], contact_area_mm2=[4, 7])
    np.testing.assert_array_equal(result.area, [4.25, 7.25])


@pytest.mark.parametrize("force", [[-1, 2], [np.nan, 1], [1], [np.inf, 0]])
def test_rejects_invalid_force_instead_of_silent_zero(proxy, force):
    model, _ = proxy
    with pytest.raises(ValueError):
        model.synthesize(force)


def test_rejects_shear_without_compression_and_invalid_coordinates(proxy):
    model, _ = proxy
    with pytest.raises(ValueError, match="normal load"):
        model.synthesize([0, 1], tangential_force_sdk_n=[[1, 0], [0, 0]])
    with pytest.raises(ValueError, match="active sensor"):
        model.synthesize([1, 1], [[1.1, 0], [0, 0]])
    with pytest.raises(ValueError):
        TactileProxyParameters(patch_sigma_mm=(0, 1))


def test_gel_selector_rejects_backing_linkage_edges_and_side_contacts():
    from tools.sim.gel_contact import select_gel_contacts

    data = (
        np.array([[2], [10], [20], [5], [4]], dtype=float),
        np.array(
            [
                [-0.006, 0, 0],
                [0.008, 0, 0],
                [0.012, 0, -0.051],
                [-0.006, 0, 0.025],
                [-0.006, 0, 0],
            ]
        ),
        np.array([[1, 0, 0]] * 4 + [[0, 1, 0]], dtype=float),
        np.zeros(5),
        np.array([[5]]),
        np.array([[0]]),
    )
    result = select_gel_contacts(data, [0, 0, 0], [1, 0, 0, 0], side="left")
    assert result["normal_force_n"] == 2
    assert result["ignored_contact_normal_force_n"] == 39
    assert result["ignored_by_reason_n"] == {
        "not_inner_face": 30,
        "outside_active_ellipse": 5,
        "normal_not_aligned": 4,
    }


@pytest.mark.parametrize(
    "side,face,normal", [("left", -0.006, 1), ("right", 0.006, -1)]
)
def test_gel_normal_polarity_and_world_transform_are_consistent(side, face, normal):
    from scipy.spatial.transform import Rotation

    from tools.sim.gel_contact import select_gel_contacts

    rotation = Rotation.from_euler("xyz", [0.3, -0.7, 0.8])
    position = np.array([0.4, -0.2, 0.6])
    q = rotation.as_quat()[[3, 0, 1, 2]]
    points = rotation.apply([[face, 0.003, 0.006]]) + position
    normals = rotation.apply([[normal, 0, 0]])
    data = (
        np.array([3.0]),
        points,
        normals,
        np.zeros(1),
        np.array([[1]]),
        np.array([[0]]),
    )
    result = select_gel_contacts(data, position, q, side=side)
    assert result["normal_force_n"] == pytest.approx(3)
    np.testing.assert_allclose(result["contact_uv"], [1 / 3, 2 / 9], atol=1e-12)


def test_gel_contact_buffers_ignore_unused_nan_but_detect_overflow():
    from tools.sim.gel_contact import select_gel_contacts

    data = (
        np.array([2.0, np.nan]),
        np.array([[-0.006, 0, 0], [np.nan] * 3]),
        np.array([[1, 0, 0], [np.nan] * 3]),
        np.zeros(2),
        np.array([[1, 0]]),
        np.array([[0, -12345]]),
    )
    result = select_gel_contacts(
        data, [0, 0, 0], [1, 0, 0, 0], side="left", max_contact_count=2
    )
    assert result["normal_force_n"] == 2
    with pytest.raises(RuntimeError, match="saturated"):
        select_gel_contacts(
            data, [0, 0, 0], [1, 0, 0, 0], side="left", max_contact_count=1
        )


def test_gel_contact_view_setup_keeps_environment_and_opposite_pad():
    from tools.sim.gel_contact import GelContactViews

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
            # Deliberately mirror the used native argument names: **kwargs
            # would hide unsupported Isaac constructor keywords.
            self.kwargs = {"reset_xform_properties": reset_xform_properties}

        def initialize(self):
            self.initialized = True

    views = GelContactViews(
        ["/left", "/right"], ["/table", "/bin", "/table"], rigid_prim_cls=FakeView
    )
    views.initialize()
    assert views.filter_paths == [
        ["/table", "/bin", "/right"],
        ["/table", "/bin", "/left"],
    ]
    assert all(v.initialized for v in views.views)
    assert all(v.kwargs["reset_xform_properties"] is False for v in views.views)


def test_gel_contact_diagnostics_preserve_body_filter_columns_and_points():
    from tools.sim.gel_contact import select_gel_contacts

    data = (
        np.array([2.0, 8.0]),
        np.array([[-0.006, 0, 0], [0.008, 0, 0]]),
        np.array([[1, 0, 0], [0, 0, 1]]),
        np.zeros(2),
        np.array([[1, 1]]),
        np.array([[0, 1]]),
    )
    result = select_gel_contacts(
        data, [0, 0, 0], [1, 0, 0, 0], side="left", filter_paths=["/packet", "/table"]
    )
    packet, table = result["per_filter_contacts"]
    assert result["filter_labels_match_columns"]
    assert packet["filter_path"] == "/packet" and packet["gel_compression_n"] == 2
    assert packet["accepted_by_contact"] == [True]
    assert table["filter_path"] == "/table" and table["ignored_normal_force_n"] == 8
    assert table["accepted_by_contact"] == [False]
    np.testing.assert_array_equal(table["contact_points_pad_m"], [[0.008, 0, 0]])
    mismatch = select_gel_contacts(
        data, [0, 0, 0], [1, 0, 0, 0], side="left", filter_paths=["/wrong"]
    )
    assert not mismatch["filter_labels_match_columns"]
    assert all(x["filter_path"] is None for x in mismatch["per_filter_contacts"])
