"""Signed external contact wrench, body attribution and exact lever-arm tests."""

from copy import deepcopy

import numpy as np
import pytest

from tools.sim.gripper_wrist import (
    GripperContactWrist,
    aggregate_gripper_wrench,
    gripper_actor_paths,
)

BASE = "/World/Robot/tool0/gripper_housing"
ACTORS = [BASE, BASE + "/left_pad", BASE + "/right_pad"]


def actor(path, scalar, point, normal):
    return {
        "actor_path": path,
        "contacts": [
            {
                "actor_path": path,
                "filter_path": "/World/Bin/Front",
                "populated_buffer_indices": [0],
                "normal_force_signed_n": [scalar],
                "points_world_m": [point],
                "normals_world": [normal],
            }
        ],
    }


def report():
    return {
        "per_actor": [
            actor(ACTORS[0], 2, [1, 3, 3], [1, 0, 0]),
            actor(ACTORS[1], -3, [1, 2, 4], [0, 1, 0]),
            actor(ACTORS[2], 2, [1, 2, 3], [-1, 0, 0]),
            # Proximal contact must never leak through even if supplied by a wider reader.
            actor("/World/Robot/upper_arm", 1000, [1, 9, 3], [1, 0, 0]),
        ]
    }


def test_signed_vectors_exact_contact_moments_bias_and_proximal_exclusion():
    raw = report()
    original = deepcopy(raw)
    bias = np.arange(6, dtype=float)
    value, rec = aggregate_gripper_wrench(raw, ACTORS, [1, 2, 3, 0, np.pi, 0], bias)
    # Opposed x forces cancel; signed left load remains -y. Housing lever arm
    # gives -2 z torque; left contact gives +3 x torque. No TCP-axis rotation.
    np.testing.assert_array_equal(rec["normal_wrench_world"], [0, -3, 0, 3, 0, -2])
    np.testing.assert_array_equal(value, bias + [0, -3, 0, 3, 0, -2])
    assert [r["actor_path"] for r in rec["per_actor"]] == ACTORS
    assert raw == original


@pytest.mark.parametrize("kind", ["actor", "contact", "self", "missing"])
def test_ambiguous_or_self_contact_attribution_fails(kind):
    raw = report()
    if kind == "actor":
        raw["per_actor"].append(deepcopy(raw["per_actor"][0]))
    elif kind == "contact":
        raw["per_actor"][0]["contacts"] *= 2
    elif kind == "self":
        raw["per_actor"][0]["contacts"][0]["filter_path"] = ACTORS[1]
    else:
        raw["per_actor"].pop(0)
    with pytest.raises(RuntimeError):
        aggregate_gripper_wrench(raw, ACTORS, np.zeros(6), np.zeros(6))


def test_actor_whitelist_cannot_replace_a_pad_with_a_proximal_body():
    with pytest.raises(ValueError, match="exactly housing"):
        gripper_actor_paths(BASE, [ACTORS[1], "/World/Robot/upper_arm"])
    with pytest.raises(ValueError, match="unique"):
        gripper_actor_paths(BASE, [ACTORS[1], ACTORS[1]])


def test_articulated_sensor_housing_load_is_included_once_with_its_lever_arm():
    housing = BASE + "/right_outer_knuckle/right_sensor_body"
    raw = report()
    raw["per_actor"].append(actor(housing, 4, [1, 2, 5], [1, 0, 0]))
    value, rec = aggregate_gripper_wrench(
        raw, ACTORS, [1, 2, 3, 0, 0, 0], np.zeros(6),
        additional_actor_paths=[housing],
    )
    np.testing.assert_array_equal(value, [4, -3, 0, 3, 8, -2])
    assert len(rec["per_actor"]) == 4
    with pytest.raises(ValueError, match="mounted gripper"):
        gripper_actor_paths(BASE, ACTORS[1:], additional_actor_paths=["/World/Robot/upper_arm"])
    with pytest.raises(ValueError, match="unique"):
        gripper_actor_paths(BASE, ACTORS[1:], additional_actor_paths=[ACTORS[1]])


class StubView:
    def __init__(self, **kwargs):
        self.prim_paths = [kwargs["prim_paths_expr"]]
        self.filters = kwargs["contact_filter_prim_paths_expr"]
        self.calls = []
        assert kwargs["track_contact_forces"]
        assert not kwargs["reset_xform_properties"]
        assert not kwargs["disable_stablization"]

    def initialize(self):
        pass

    def get_contact_force_data(self, *, dt):
        self.calls.append(dt)
        counts = np.zeros((1, len(self.filters)), dtype=int)
        counts[0, self.filters.index("/World/Bin/Front")] = 1
        return (
            np.array([0.02]),
            np.array([[0.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([0.0]),
            counts,
            np.zeros_like(counts),
        )


def test_wrapper_converts_impulses_once_and_reads_each_whitelisted_actor_once():
    sensor = GripperContactWrist(BASE, ACTORS[1:], rigid_prim_cls=StubView)
    with pytest.raises(RuntimeError, match="initialize"):
        sensor.sample(np.zeros(6), np.zeros(6), 0.004)
    sensor.initialize()
    value, rec = sensor.sample(np.zeros(6), np.zeros(6), 0.004)
    np.testing.assert_array_equal(value, [15, 0, 0, 0, 0, -15])
    assert all(v.calls == [1.0] for v in sensor.reader.views)
    assert rec["physics_dt_s"] == 0.004 and rec["api_dt_argument"] == 1
    assert sensor.metadata()["policy_and_safety_input_in_policy_mode"]
    assert "proximal arm contacts" in sensor.metadata()["omitted"]
