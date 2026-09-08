"""Main's fault diagnostics and the opt-in verified hold coexist with fake RTDE."""

import sys

import numpy as np
import pytest

from phantom.drivers.base import ControlLost
from phantom.drivers.servo_hold import ConstraintHoldBudget
from test_servo_constraint_hold_integration import POSE, native_arm, target
from test_ur_servo_limiter import FakeRecv, LIMITS, make_arm, pose_at_reach, prime


@pytest.mark.parametrize("verified_hold", [False, True])
def test_control_loss_preserves_both_command_anchors_and_safety_diagnostics(
    monkeypatch, verified_hold
):
    limits = dict(LIMITS)
    if verified_hold:
        limits["servo_constraint_hold_s"] = 2.5
    arm = make_arm(limits)
    pose = pose_at_reach(0.40)
    prime(arm, pose)
    q_before = np.asarray(arm._last_qsol).copy()
    pose_before = arm._last_cmd_pose.copy()

    class StoppedRecv(FakeRecv):
        def isProtectiveStopped(self):
            return True

        def getRobotMode(self):
            return 7

        def getSafetyMode(self):
            return 3

        def getSafetyStatusBits(self):
            return (1 << 2) | (1 << 8)

        def getActualQd(self):
            return [0.0] * 6

        def getActualTCPForce(self):
            return [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]

    arm._recv = StoppedRecv(pose)
    arm._ctrl.servoJ = lambda *_: False
    monkeypatch.setitem(sys.modules, "dashboard_client", None)
    with pytest.raises(ControlLost, match="protective_stopped.*violation") as error:
        arm.servo_l(pose_at_reach(0.4005), 0.008, 0.1, 300)
    assert error.value.stop_reason == "control_lost"
    np.testing.assert_array_equal(arm._last_qsol, q_before)
    np.testing.assert_array_equal(arm._last_cmd_pose, pose_before)
    assert arm.control_loss_last["safety_status_names"] == [
        "protective_stopped", "violation"
    ]
    assert "|F|=5.0N" in arm.control_loss_last["summary"]


def test_opt_in_recovery_clears_fault_streak_without_forgiving_verified_holds(
    monkeypatch,
):
    arm, clock = native_arm(monkeypatch, timeout=5)
    arm._ctrl.mode = "branch"
    assert not arm.servo_l(target(), 0.008, 0.1, 300).sent
    assert arm._branch_rejects == arm._ik_rejects == 1
    since = arm._hold_since

    arm._ctrl.mode = "hold"
    clock[0] += 0.008
    assert arm.servo_l(target(), 0.008, 0.1, 300).reason == "constraint_hold"
    assert arm._branch_rejects == arm._ik_rejects == 1
    assert arm._hold_since == since

    arm._ctrl.mode = "stationary"
    clock[0] += 0.008
    assert arm.servo_l(POSE, 0.008, 0.1, 300).sent
    assert arm._branch_rejects == arm._ik_rejects == 0
    assert arm._hold_since is None
    # A valid identical joint command is IK recovery, but no physical progress:
    # the separate constraint deadline must survive until progress or timeout.
    assert arm.constraint_hold_last["active"]


def test_teardown_without_control_object_clears_both_hold_budgets(monkeypatch):
    arm, _ = native_arm(monkeypatch)
    arm._ctrl = None
    arm._constraint_hold_budget = ConstraintHoldBudget(0.5)
    arm._constraint_hold_budget.check(0.0, arm._last_qsol, held=True)
    arm._ik_rejects = arm._branch_rejects = 3
    arm._hold_since = 0.0
    arm._teardown_ctrl()
    assert arm._ik_rejects == arm._branch_rejects == 0
    assert arm._hold_since is None
    state = arm._constraint_hold_budget.check(20.0, arm._last_qsol, held=True)
    assert state["started_at_s"] == 20.0 and not state["timed_out"]
