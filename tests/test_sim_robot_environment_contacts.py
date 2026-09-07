"""Physical units, actor attribution and populated-buffer integrity tests."""

import numpy as np
import pytest

from tools.sim.robot_environment_contacts import (
    ENVIRONMENT_PATHS,
    RobotEnvironmentContactViews,
    summarize_actor_contacts,
)

ACTOR = "/World/Robot/Geometry/world/base_link/arm"
FILTERS = ["/World/Bin/Front", "/World/Waffle"]


def data():
    return (
        np.array([[np.nan], [0.02], [-0.03], [0.0], [np.nan]]),
        np.array(
            [
                [np.nan] * 3,
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 4.0],
                [1.0, 3.0, 4.0],
                [np.nan] * 3,
            ]
        ),
        np.array(
            [
                [np.nan] * 3,
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [np.nan] * 3,
            ]
        ),
        np.array([np.nan, -0.001, 0.0005, 0.0015, np.nan]),
        np.array([[2, 1]]),
        np.array([[1, 3]]),
    )


def test_impulse_units_no_force_cancellation_and_world_geometry_preserved():
    result = summarize_actor_contacts(data(), ACTOR, FILTERS, 0.004)
    assert result["normal_force_magnitude_n"] == pytest.approx(12.5)
    assert result["normal_impulse_magnitude_ns"] == pytest.approx(0.05)
    assert result["populated_unique_contact_count"] == 3
    front, packet = result["contacts"]
    assert front["filter_path"] == "/World/Bin/Front"
    assert front["normal_force_signed_n"] == [5.0, -7.5]
    assert front["points_world_m"] == [[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]]
    assert front["normals_world"] == [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    assert front["signed_separation_m"] == [-0.001, 0.0005]
    assert packet["normal_force_magnitude_n"] == 0
    assert packet["populated_buffer_indices"] == [3]
    assert packet["signed_separation_m"] == [0.0015]


def test_physics_step_conversion_is_not_applied_twice():
    a = summarize_actor_contacts(data(), ACTOR, FILTERS, 0.004)
    b = summarize_actor_contacts(data(), ACTOR, FILTERS, 0.002)
    assert b["normal_force_magnitude_n"] == 2 * a["normal_force_magnitude_n"]
    assert b["normal_impulse_magnitude_ns"] == a["normal_impulse_magnitude_ns"]


@pytest.mark.parametrize("bad_dt", [0, -1, np.nan, np.inf])
def test_invalid_dt_rejected(bad_dt):
    with pytest.raises(ValueError):
        summarize_actor_contacts(data(), ACTOR, FILTERS, bad_dt)


@pytest.mark.parametrize("index", range(4))
def test_nonfinite_populated_contact_rejected(index):
    arrays = list(data())
    arrays[index][1] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        summarize_actor_contacts(arrays, ACTOR, FILTERS, 0.004)


def test_unused_nan_padding_does_not_fake_or_invalidate_contact():
    arrays = list(data())
    arrays[4][:] = 0
    arrays[5][:] = -900
    result = summarize_actor_contacts(arrays, ACTOR, FILTERS, 0.004)
    assert result["normal_force_magnitude_n"] == 0
    assert result["contacts"] == []
    assert result["pair_contact_start_indices"] == [None, None]


def test_ambiguous_shared_filter_indices_fail_instead_of_assigning_body():
    arrays = list(data())
    arrays[5][0, 1] = 2
    with pytest.raises(RuntimeError, match="ambiguously"):
        summarize_actor_contacts(arrays, ACTOR, FILTERS, 0.004)


@pytest.mark.parametrize(
    "counts,starts",
    [
        ([[2.5, 1]], [[1, 3]]),
        ([[-1, 1]], [[1, 3]]),
        ([[2, 1]], [[-1, 3]]),
        ([[2, 4]], [[1, 3]]),
        ([[2, 1], [0, 0]], [[1, 3], [0, 0]]),
    ],
)
def test_invalid_counts_ranges_and_extra_sensor_rows_fail(counts, starts):
    arrays = list(data())
    arrays[4], arrays[5] = np.array(counts), np.array(starts)
    with pytest.raises(RuntimeError):
        summarize_actor_contacts(arrays, ACTOR, FILTERS, 0.004)


def test_saturation_is_not_silently_truncated():
    with pytest.raises(RuntimeError, match="saturated"):
        summarize_actor_contacts(data(), ACTOR, FILTERS, 0.004, max_contact_count=3)


@pytest.mark.parametrize(
    "robots,filters",
    [
        ([ACTOR, ACTOR], FILTERS),
        (["/World/Robot/.*"], FILTERS),
        (["/World/Bin/Front"], FILTERS),
        ([ACTOR], [ACTOR]),
        ([ACTOR], ["/World/Bin/.*"]),
        ([ACTOR], [FILTERS[0], FILTERS[0]]),
    ],
)
def test_paths_cannot_leak_robot_pairs_into_environment_or_use_regex(robots, filters):
    with pytest.raises(ValueError):
        RobotEnvironmentContactViews(robots, filters, rigid_prim_cls=lambda **_: None)


class StubView:
    def __init__(
        self,
        *,
        prim_paths_expr,
        name,
        track_contact_forces,
        contact_filter_prim_paths_expr,
        max_contact_count,
        reset_xform_properties,
        disable_stablization,
    ):
        assert track_contact_forces
        assert not reset_xform_properties
        assert not disable_stablization
        self.prim_paths = [prim_paths_expr]
        self.filter_paths = contact_filter_prim_paths_expr
        self.dt_arguments = []

    def initialize(self):
        pass

    def get_contact_force_data(self, *, dt):
        self.dt_arguments.append(dt)
        return data()


def test_views_use_exact_actor_mapping_and_explicit_raw_impulses():
    observer = RobotEnvironmentContactViews(
        [ACTOR, ACTOR + "/wrist"], FILTERS, rigid_prim_cls=StubView
    )
    with pytest.raises(RuntimeError, match="initialize"):
        observer.get_all(0.004)
    observer.initialize()
    result = observer.get_all(0.004)
    assert result["normal_force_magnitude_n"] == pytest.approx(25.0)
    assert result["normal_force_by_environment_n"] == {
        FILTERS[0]: 25.0,
        FILTERS[1]: 0.0,
    }
    assert all(v.dt_arguments == [1.0] for v in observer.views)
    assert [r["actor_path"] for r in result["per_actor"]] == [ACTOR, ACTOR + "/wrist"]
    assert not result["self_collision_observed"]
    assert len(ENVIRONMENT_PATHS) == 8


def test_unexpected_resolved_actor_fails_before_readout():
    observer = RobotEnvironmentContactViews([ACTOR], FILTERS, rigid_prim_cls=StubView)
    observer.views[0].prim_paths = [ACTOR, ACTOR + "/child"]
    with pytest.raises(RuntimeError, match="unexpected actor"):
        observer.initialize()


def test_actor_force_equals_sum_of_reported_body_pair_loads():
    result = summarize_actor_contacts(data(), ACTOR, FILTERS, 0.004)
    assert result["normal_force_magnitude_n"] == pytest.approx(
        sum(p["normal_force_magnitude_n"] for p in result["contacts"])
    )
