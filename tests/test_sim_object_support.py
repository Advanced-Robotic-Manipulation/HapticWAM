"""Support telemetry must include all loads, without cancellation or duplication."""

import numpy as np
import pytest

from tools.sim.object_support import PacketSupportViews, summarize_support_contacts

BIN = "/World/Bin/Bottom"
ROBOT = "/World/Robot/Housing"


def data(forces, counts, starts=None):
    n = len(forces)
    return (
        np.asarray(forces),
        np.zeros((n, 3)),
        np.zeros((n, 3)),
        np.zeros(n),
        np.array([counts]),
        np.array([np.r_[0, np.cumsum(counts)[:-1]] if starts is None else starts]),
    )


def test_support_sums_normal_magnitudes_not_cancellable_net_vectors():
    result = summarize_support_contacts(
        data([0.35, 2, -2], [1, 2]), [BIN, ROBOT], [BIN], [ROBOT]
    )
    assert result["packet_bin_normal_force"] == 0.35
    assert result["packet_robot_normal_force"] == 4
    assert result["populated_unique_contact_count"] == 3


def test_duplicate_robot_filter_indices_do_not_inflate_support():
    other = "/World/Robot/Other"
    result = summarize_support_contacts(
        data([0.35, 2], [1, 1, 1], [0, 1, 1]),
        [BIN, ROBOT, other],
        [BIN],
        [ROBOT, other],
    )
    assert result["packet_robot_normal_force"] == 2
    assert result["duplicate_filter_contact_indices"] == 1


def test_ambiguous_body_category_is_rejected_instead_of_guessing():
    with pytest.raises(RuntimeError, match="ambiguously"):
        summarize_support_contacts(
            data([1], [1, 1], [0, 0]), [BIN, ROBOT], [BIN], [ROBOT]
        )


def test_unused_buffers_ignored_but_loaded_nan_and_capacity_are_invalid():
    result = summarize_support_contacts(
        data([0.35, np.nan], [1, 0], [0, -999]),
        [BIN, ROBOT],
        [BIN],
        [ROBOT],
        max_contact_count=2,
    )
    assert result["packet_bin_normal_force"] == 0.35
    with pytest.raises(RuntimeError, match="nonfinite"):
        summarize_support_contacts(data([np.nan], [1, 0]), [BIN, ROBOT], [BIN], [ROBOT])
    with pytest.raises(RuntimeError, match="saturated"):
        summarize_support_contacts(
            data([0.35], [1, 0]), [BIN, ROBOT], [BIN], [ROBOT], max_contact_count=1
        )


@pytest.mark.parametrize(
    "counts, starts",
    [([1.5, 0], [0, 0]), ([1, 0], [-1, 0]), ([2, 0], [0, 0]), ([1], [0])],
)
def test_bad_contact_ranges_never_produce_silent_zero(counts, starts):
    with pytest.raises(RuntimeError):
        summarize_support_contacts(
            data([1], counts, starts), [BIN, ROBOT], [BIN], [ROBOT]
        )


def test_native_view_receives_only_supported_constructor_arguments_and_all_robot_paths():
    class NativeSignatureStub:
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
            assert prim_paths_expr == "/World/Waffle"
            assert track_contact_forces and not reset_xform_properties
            self.filters = contact_filter_prim_paths_expr

        def initialize(self):
            self.ready = True

        def get_contact_force_data(self, dt):
            assert self.ready and dt == 0.004
            return data([0.35, 2, 3], [1, 1, 1, 0])

    paths = [ROBOT, "/World/Robot/left_pad", "/World/Robot/right_pad"]
    view = PacketSupportViews(
        "/World/Waffle", [BIN], paths, rigid_prim_cls=NativeSignatureStub
    )
    view.initialize()
    result = view.get_all(0.004)
    assert result["packet_robot_normal_force"] == 5
    assert result["packet_bin_normal_force"] == 0.35
    assert result["robot_paths"] == paths
    for invalid in [BIN, "/World/Waffle", "/World/Bench/Slab"]:
        with pytest.raises(ValueError, match="exclude bin"):
            PacketSupportViews(
                "/World/Waffle",
                [BIN],
                paths + [invalid],
                rigid_prim_cls=NativeSignatureStub,
            )
