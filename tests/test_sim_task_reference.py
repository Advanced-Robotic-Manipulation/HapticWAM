"""Prevent deployment footage from silently becoming the teleop baseline."""
import pytest

from tools.sim.prepare_task_scene import check_reference_policy


def test_explicit_teleop_is_accepted():
    check_reference_policy({"policy": "teleop", "driver_modes": {"drivers": "real"}})


@pytest.mark.parametrize("meta", [{}, {"policy": ""}, {"policy": "teacher"}, {"policy": "student"}])
def test_unknown_or_policy_provenance_requires_explicit_override(meta):
    with pytest.raises(ValueError, match="explicitly labelled teleop"):
        check_reference_policy(meta)


def test_deliberate_deployment_diagnostic_remains_available():
    check_reference_policy({"policy": "teacher"}, allow_policy_rollout=True)
