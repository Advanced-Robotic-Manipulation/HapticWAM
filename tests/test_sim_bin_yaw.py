"""Both physical scorers must contain packet corners in the rotated opening."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from phantom.sim.policy_metrics import evaluate_policy_trace
from tools.sim.evaluate_pick_place import evaluate_arrays
from test_sim_pick_place_evaluation import complete_trial
from test_sim_policy_metrics import successful_trial  # noqa: F401 (pytest fixture)


def measured_bin(center):
    return {
        "geometry_model": "rectangular_envelope",
        "center": list(center),
        "outer_size": [.4, .3, .19],
        "opening_size": [.36, .26],
        "floor_thickness": .005,
    }


def rotate_world_trace(trace, config, yaw):
    """Rotate scene state about the bin origin while preserving its Z datum."""
    changed, cfg = deepcopy(trace), deepcopy(config)
    center = np.array(cfg["bin"]["center"])
    rotation = Rotation.from_euler("z", yaw)
    matrix = rotation.as_matrix()
    for name in ("waffle_position", "pad_position"):
        if name in changed:
            changed[name] = (changed[name] - center) @ matrix.T + center
    changed["tcp"][:, :3] = (changed["tcp"][:, :3] - center) @ matrix.T + center
    changed["tcp"][:, 3:] = (
        rotation * Rotation.from_rotvec(changed["tcp"][:, 3:])
    ).as_rotvec()
    packet_rotation = Rotation.from_quat(
        changed["waffle_orientation_wxyz"][:, [1, 2, 3, 0]]
    )
    changed["waffle_orientation_wxyz"] = (
        rotation * packet_rotation
    ).as_quat()[:, [3, 0, 1, 2]]
    cfg["bin"]["yaw"] = yaw
    return changed, cfg


@pytest.mark.parametrize("yaw", [np.pi / 2, -.7])
@pytest.mark.parametrize("offset,inside", [([.065, 0], True), ([0, .14], False)])
def test_policy_containment_is_invariant_under_common_bin_packet_yaw(
    successful_trial, yaw, offset, inside
):
    trace, cfg, run = successful_trial
    cfg["waffle"]["size"] = [.2, .02, .02]
    cfg["bin"] = measured_bin(cfg["bin"]["center"])
    for name in ("waffle_position", "pad_position"):
        trace[name][..., :2] += offset
    trace["tcp"][:, :2] += offset
    baseline = evaluate_policy_trace(trace, cfg, run=run)
    changed, rotated = rotate_world_trace(trace, cfg, yaw)
    result = evaluate_policy_trace(changed, rotated, run=run)
    assert result["valid_for_scoring"]
    assert baseline["object"]["final_inside_bin"] is inside
    assert result["object"]["final_inside_bin"] is inside
    assert result["outcomes"] == baseline["outcomes"]
    assert result["event_times_s"] == baseline["event_times_s"]


@pytest.mark.parametrize("yaw", [np.pi / 2, -.7])
@pytest.mark.parametrize("offset,inside", [([.065, 0], True), ([0, .14], False)])
def test_recorded_replay_containment_uses_the_same_rotated_opening(yaw, offset, inside):
    trace, run, cfg, real, timeline = complete_trial()
    cfg["bin"] = measured_bin([-.21, .32, .007])
    translation = np.array(cfg["bin"]["center"]) + [*offset, 0]
    trace["waffle_position"] += translation
    trace["tcp"][:, :3] += translation
    baseline = evaluate_arrays(trace, run, cfg, real, timeline)
    changed, rotated = rotate_world_trace(trace, cfg, yaw)
    result = evaluate_arrays(changed, run, rotated, real, timeline)
    assert baseline["gates"]["final_oriented_bin_containment"]["pass"] is inside
    assert result["gates"]["final_oriented_bin_containment"]["pass"] is inside
    assert result["metrics"]["final_bin"]["contained_fraction"] == float(inside)
    assert result["physical_verdict"] == baseline["physical_verdict"]
    # The rotate/un-rotate round-trip reassociates the same products, so the
    # two datums can differ by ~1 ULP and do so differently per CPU (arm64 vs
    # x86-64). The invariant under test is yaw-invariance of the datum, not
    # bit-identity: 1e-12 m is a million times below the metric's meaning.
    assert result["metrics"]["final_bin"]["minimum_oriented_corner_z_m"] == pytest.approx(
        baseline["metrics"]["final_bin"]["minimum_oriented_corner_z_m"], abs=1e-12
    )


def test_explicit_zero_yaw_preserves_policy_results_exactly(successful_trial):
    trace, cfg, run = successful_trial
    baseline = evaluate_policy_trace(trace, cfg, run=run)
    cfg["bin"]["yaw"] = 0.0
    assert evaluate_policy_trace(trace, cfg, run=run) == baseline


def test_explicit_zero_yaw_preserves_recorded_replay_results_exactly():
    trial = complete_trial()
    baseline = evaluate_arrays(*trial)
    trial[2]["bin"]["yaw"] = 0.0
    assert evaluate_arrays(*trial) == baseline
