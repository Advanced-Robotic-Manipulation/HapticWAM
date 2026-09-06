"""Deployment watchdog must inspect accepted motion without false hold stops."""

import numpy as np
import pytest

from tools.sim.deployment_filters import PlannerStallWatchdog


def pose(x):
    return np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_two_stalled_replan_intervals_stop_and_inputs_are_copied():
    guard = PlannerStallWatchdog()
    target = pose(0.0)
    guard.check(0.0, pose(0.0), target)
    target[0] = 0.01
    assert guard.check(1.0, pose(0.0), target)["stop_reason"] is None
    target[0] = 0.02
    result = guard.check(2.0, pose(0.0), target)
    assert result["stop_reason"] == "motion_stall"
    assert result["strikes"] == 2


def test_successful_tracking_resets_consecutive_strikes():
    guard = PlannerStallWatchdog()
    guard.check(0.0, pose(0.0), pose(0.0))
    assert guard.check(1.0, pose(0.0), pose(0.01))["strikes"] == 1
    assert guard.check(2.0, pose(0.01), pose(0.02))["strikes"] == 0
    assert guard.check(3.0, pose(0.01), pose(0.03))["stop_reason"] is None


def test_held_accepted_target_does_not_count_rejected_proposals_as_motion():
    guard = PlannerStallWatchdog()
    for t in range(5):
        result = guard.check(float(t), pose(0.0), pose(0.0))
        assert result["strikes"] == 0
        assert result["stop_reason"] is None


def test_watchdog_rejects_nonfinite_feedback():
    with pytest.raises(ValueError, match="finite six-vector"):
        PlannerStallWatchdog().check(0.0, pose(np.nan), pose(0.0))
