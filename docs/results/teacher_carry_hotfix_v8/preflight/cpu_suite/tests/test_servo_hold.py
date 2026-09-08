"""CPU contracts for bounded constraint waiting; no robot or simulator imports."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.drivers.servo_hold import (
    ConstraintHoldBudget,
    ServoHoldTimeout,
    verified_constraint_hold,
)
from phantom.drivers.servo_limiter import ServoLimits

Q = np.array([0.1, -1.5, 0.6, 0.5, 1.2, -3.0])
LIMITS = ServoLimits(elbow_min_rad=0.4, joint_speed_max_rad_s=1.0)


def selection(**changes):
    values = dict(
        q=None,
        pose=None,
        reason="limiter_hold",
        mode="hold",
        fraction=None,
        violation="elbow",
        ik_calls=7,
        all_ik_valid=True,
        all_ik_on_branch=True,
    )
    return SimpleNamespace(**(values | changes))


def test_stationary_hold_uses_elapsed_time_not_twenty_five_tick_reject_limit():
    budget = ConstraintHoldBudget(1.0)
    for tick in range(125):
        report = budget.check(tick / 125, Q, held=True)
        assert report["active"] and not report["timed_out"]
        assert report["started_at_s"] == 0.0
        assert report["elapsed_s"] == tick / 125
    assert budget.check(1.0, Q, held=True)["timed_out"]


def test_accepted_noops_and_small_oscillation_do_not_restart_deadline():
    budget = ConstraintHoldBudget(0.5)
    budget.check(8.0, Q, held=True)
    for tick in range(1, 50):
        q = Q.copy()
        q[0] += (-1) ** tick * 0.0009
        report = budget.check(8 + tick / 100, q, held=tick % 2 == 0)
        assert report["active"] and report["started_at_s"] == 8.0
        assert not report["timed_out"]
    assert budget.check(8.5, Q, held=False)["timed_out"]


def test_cumulative_accepted_progress_from_original_anchor_clears_before_deadline():
    budget = ConstraintHoldBudget(1.0)
    anchor = Q.copy()
    budget.check(0.0, anchor, held=True)
    anchor[1] = 99.0  # Caller mutation cannot move the stored anchor.
    for tick in range(1, 4):
        q = Q.copy()
        q[1] += tick * 0.0004
        report = budget.check(tick / 10, q, held=False)
        assert report["active"] == (tick < 3)
    assert report["progress_rad"] == pytest.approx(0.0012)
    assert report["started_at_s"] is None and report["elapsed_s"] == 0
    # A later genuine hold owns a new deadline at its accepted anchor.
    renewed = budget.check(0.9, q, held=True)
    assert renewed["started_at_s"] == 0.9 and renewed["progress_rad"] == 0


def test_changed_joints_without_an_accepted_setpoint_do_not_clear():
    budget = ConstraintHoldBudget(1.0)
    budget.check(0.0, Q, held=True)
    q = Q + 0.02
    assert budget.check(0.1, q, held=True)["active"]
    assert not budget.check(0.2, q, held=False)["active"]


def test_deadline_wins_over_same_tick_progress_and_timeout_is_latched():
    budget = ConstraintHoldBudget(0.5)
    budget.check(10.0, Q, held=True)
    at_deadline = budget.check(10.5, Q + 0.1, held=False)
    assert at_deadline["active"] and at_deadline["timed_out"]
    late = budget.check(12.0, Q + 0.2, held=False)
    assert late["started_at_s"] == 10.0 and late["timed_out"]
    budget.reset()
    assert budget.check(0.0, Q, held=False) == {
        "active": False,
        "elapsed_s": 0.0,
        "timed_out": False,
        "started_at_s": None,
        "progress_rad": 0.0,
    }


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), True])
@pytest.mark.parametrize("name", ["timeout_s", "progress_rad"])
def test_bad_configuration_rejected(name, value):
    settings = {"timeout_s": 0.5, "progress_rad": 0.001, name: value}
    with pytest.raises(ValueError, match=name):
        ConstraintHoldBudget(**settings)


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), -1, None, True])
def test_bad_clock_fails_without_clearing_or_advancing_reference(bad_time):
    budget = ConstraintHoldBudget(1.0)
    budget.check(0, Q, held=True)
    with pytest.raises(ValueError, match="nondecreasing"):
        budget.check(bad_time, Q + 0.1, held=False)
    assert budget.check(0, Q, held=True)["started_at_s"] == 0
    assert budget.check(1, Q, held=True)["timed_out"]


@pytest.mark.parametrize(
    "bad_q",
    [[], np.zeros((1, 6)), [0] * 5, [0, 0, 0, 0, 0, np.nan], [np.inf] * 6, None],
)
def test_bad_joint_feedback_does_not_mutate_deadline(bad_q):
    budget = ConstraintHoldBudget(1.0)
    budget.check(0, Q, held=True)
    with pytest.raises(ValueError, match="six finite"):
        budget.check(0.9, bad_q, held=False)
    assert not budget.check(0.5, Q, held=True)["timed_out"]
    assert budget.check(1, Q, held=True)["timed_out"]


def test_held_flag_is_explicit_and_custom_progress_threshold_is_used():
    budget = ConstraintHoldBudget(1.0, progress_rad=0.01)
    with pytest.raises(ValueError, match="boolean"):
        budget.check(0, Q, held=1)
    budget.check(0, Q, held=True)
    assert budget.check(0.1, Q + 0.002, held=False)["active"]
    assert not budget.check(0.2, Q + 0.011, held=False)["active"]


def test_timeout_exception_has_stable_ordinary_stop_reason():
    error = ServoHoldTimeout()
    assert isinstance(error, RuntimeError)
    assert str(error) == error.reason == "servo_constraint_hold_timeout"


@pytest.mark.parametrize("violation", ["elbow", "joint_speed"])
def test_verified_hold_requires_feasible_principal_anchor_and_real_constraint(
    violation,
):
    assert verified_constraint_hold(selection(violation=violation), Q, 0.008, LIMITS)
    for elbow in [-np.pi, -0.4, 0.4, np.pi]:
        q = Q.copy()
        q[2] = elbow
        assert verified_constraint_hold(
            selection(violation=violation), q, 0.008, LIMITS
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"all_ik_valid": False},
        {"all_ik_on_branch": False},
        {"reason": "ik_branch"},
        {"reason": "no_solution"},
        {"violation": "no_solution"},
        {"violation": None},
        {"ik_calls": 0},
        {"ik_calls": True},
        {"ik_calls": 2.5},
        {"q": Q},
        {"pose": np.zeros(6)},
        {"fraction": 0.1},
        {"mode": "slide"},
    ],
)
def test_faults_and_real_motion_are_never_reclassified_as_constraint_holds(changes):
    assert not verified_constraint_hold(selection(**changes), Q, 0.008, LIMITS)


def test_missing_audit_flags_fail_closed_and_are_not_invented():
    for name in ("all_ik_valid", "all_ik_on_branch"):
        candidate = selection()
        delattr(candidate, name)
        assert not verified_constraint_hold(candidate, Q, 0.008, LIMITS)


@pytest.mark.parametrize("elbow", [0, 0.399, -0.399, np.pi + 0.001, 2 * np.pi + 0.6])
def test_outside_envelope_or_wrapped_elbow_cannot_be_a_verified_rest(elbow):
    q = Q.copy()
    q[2] = elbow
    assert not verified_constraint_hold(selection(), q, 0.008, LIMITS)


@pytest.mark.parametrize(
    "settings",
    [
        {"elbow_min_rad": None, "joint_speed_max_rad_s": None},
        {"elbow_min_rad": -0.1},
        {"elbow_min_rad": np.pi},
        {"elbow_min_rad": np.nan},
        {"joint_speed_max_rad_s": 0},
        {"joint_speed_max_rad_s": np.inf},
        {"branch_tolerance_rad": 0},
        {"branch_tolerance_rad": np.nan},
    ],
)
def test_disabled_or_bad_limits_never_verify(settings):
    assert not verified_constraint_hold(
        selection(), Q, 0.008, replace(LIMITS, **settings)
    )


def test_constraint_must_be_enabled_and_bad_inputs_fail_closed():
    assert not verified_constraint_hold(
        selection(violation="elbow"), Q, 0.008, replace(LIMITS, elbow_min_rad=None)
    )
    assert not verified_constraint_hold(
        selection(violation="joint_speed"),
        Q,
        0.008,
        replace(LIMITS, joint_speed_max_rad_s=None),
    )
    for dt in [None, 0, -1, np.nan, np.inf]:
        assert not verified_constraint_hold(selection(), Q, dt, LIMITS)
    assert not verified_constraint_hold(selection(), [np.nan] * 6, 0.008, LIMITS)
    assert not verified_constraint_hold(None, Q, 0.008, LIMITS)
