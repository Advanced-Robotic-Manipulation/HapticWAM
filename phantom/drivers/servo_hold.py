"""Bounded, opt-in waiting for a verified servo constraint; no device access.

The caller must distinguish an accepted setpoint from an IK/branch fault.
``held=False`` means an accepted setpoint, not merely receipt of a new plan.
This module neither clears fault counters nor sends robot commands.
"""

from __future__ import annotations

import math

import numpy as np

from phantom.drivers.servo_limiter import feasible


class ServoHoldTimeout(RuntimeError):
    """An ordinary bounded-hold stop, distinct from a driver/executor crash."""

    reason = "servo_constraint_hold_timeout"

    def __init__(self, message: str | None = None):
        super().__init__(self.reason if message is None else message)


def _finite_positive(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _joints(value):
    try:
        q = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("q must contain six finite joint angles") from error
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("q must contain six finite joint angles")
    return q


class ConstraintHoldBudget:
    """One fixed deadline per stationary hold, measured by caller clock.

    Only an accepted joint command with cumulative displacement from the
    original hold anchor at least ``progress_rad`` clears a live budget. A
    timeout wins over progress on the same tick and stays latched until reset.
    Repeated equal timestamps are allowed; a backwards clock is rejected.
    """

    def __init__(self, timeout_s, progress_rad=0.001):
        self.timeout_s = _finite_positive(timeout_s, "timeout_s")
        self.progress_threshold_rad = _finite_positive(progress_rad, "progress_rad")
        self.reset()

    def reset(self):
        self._last_t = None
        self._started_at = None
        self._anchor = None
        self._timed_out = False

    def check(self, t, q, *, held: bool) -> dict:
        try:
            now = float(t)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("t must be finite and nondecreasing") from error
        if (
            isinstance(t, (bool, np.bool_))
            or not math.isfinite(now)
            or (self._last_t is not None and now < self._last_t)
        ):
            raise ValueError("t must be finite and nondecreasing")
        joints = _joints(q)
        if not isinstance(held, (bool, np.bool_)):
            raise ValueError("held must be a boolean")
        # Validate every input before mutating the clock/reference, including
        # after timeout. Bad feedback must never establish a new deadline.
        self._last_t = now
        if self._started_at is None and held:
            self._started_at = now
            self._anchor = joints.copy()
        progress = (
            float(np.max(np.abs(joints - self._anchor)))
            if self._anchor is not None
            else 0.0
        )
        if self._started_at is not None:
            if now - self._started_at >= self.timeout_s:
                self._timed_out = True
            if (
                not self._timed_out
                and not held
                and progress >= self.progress_threshold_rad
            ):
                self._started_at = self._anchor = None
        return {
            "active": self._started_at is not None,
            "elapsed_s": 0.0 if self._started_at is None else now - self._started_at,
            "timed_out": self._timed_out,
            "started_at_s": self._started_at,
            # On the clearing tick retain the observed displacement so the
            # caller can audit why this deadline ended. Later idle ticks are 0.
            "progress_rad": progress,
        }


def verified_constraint_hold(selection, qref, dt, limits) -> bool:
    """True only for a fault-free constraint wait at a feasible joint anchor.

    Every IK attempt must be finite and on the anchor branch. Missing audit
    flags fail closed, as do invalid inputs and disabled/malformed limits.
    This establishes commanded rest feasibility, not measured tracking,
    collision freedom, load retention or actual robot safety certification.
    """
    try:
        if (
            selection.reason != "limiter_hold"
            or selection.mode != "hold"
            or selection.q is not None
            or selection.pose is not None
            or selection.fraction is not None
            or selection.violation not in ("elbow", "joint_speed")
            or getattr(selection, "all_ik_valid", False) is not True
            or getattr(selection, "all_ik_on_branch", False) is not True
            or isinstance(selection.ik_calls, bool)
            or not isinstance(selection.ik_calls, (int, np.integer))
            or selection.ik_calls <= 0
        ):
            return False
        _finite_positive(dt, "dt")
        q = _joints(qref)
        emin, vmax = limits.elbow_min_rad, limits.joint_speed_max_rad_s
        if emin is None and vmax is None:
            return False
        if emin is not None and (
            isinstance(emin, (bool, np.bool_))
            or not math.isfinite(float(emin))
            or not 0 <= float(emin) < math.pi
        ):
            return False
        if vmax is not None:
            _finite_positive(vmax, "joint_speed_max_rad_s")
        _finite_positive(limits.branch_tolerance_rad, "branch_tolerance_rad")
        if abs(q[2]) > math.pi:
            return False
        if selection.violation == "elbow" and emin is None:
            return False
        if selection.violation == "joint_speed" and vmax is None:
            return False
        # Identical q/qref is at rest, but cannot use the limiter's special
        # recovery exception to legitimize an anchor below the elbow envelope.
        return bool(feasible(q, q, float(dt), limits))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
