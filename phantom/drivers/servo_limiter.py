"""Shared, opt-in servo-step selection; no simulator or hardware connection.

This extracts the existing UR driver's reach/joint-speed algorithm. The slide
is a candidate search direction, not a projection of the actual wrist center:
every candidate must pass IK, branch, elbow and speed checks. It constrains
commanded joints and cannot guarantee measured tracking or stopping distance.
Raw joint branches are retained; the historical absolute elbow predicate is
intended for principal elbow branches, not arbitrary added full revolutions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from phantom.data.derived import rotvec_nearest


@dataclass(frozen=True)
class ServoLimits:
    elbow_min_rad: float | None = None
    joint_speed_max_rad_s: float | None = None
    branch_tolerance_rad: float = 0.35
    bisection_iterations: int = 3
    shoulder_height_m: float = 0.1519

    @property
    def enabled(self) -> bool:
        return self.elbow_min_rad is not None or self.joint_speed_max_rad_s is not None


def ik_valid(q) -> bool:
    try:
        return len(q) == 6 and all(math.isfinite(float(v)) for v in q)
    except Exception:  # noqa: BLE001 - malformed IK values must fail closed
        return False


def speed_violation(q, qref, dt: float, limits: ServoLimits) -> bool:
    vmax = limits.joint_speed_max_rad_s
    if vmax is None or qref is None or dt <= 0 or q is None or not len(q):
        return False
    return max(abs(a - b) for a, b in zip(q, qref)) / dt > float(vmax)


def limit_violation(q, qref, dt: float, limits: ServoLimits) -> str | None:
    if not limits.enabled:
        return None
    if q is None or not len(q):
        return "no_solution"
    if limits.elbow_min_rad is not None and abs(float(q[2])) < float(
        limits.elbow_min_rad
    ):
        return "elbow"
    if speed_violation(q, qref, dt, limits):
        return "joint_speed"
    return None


def feasible(q, qref, dt: float, limits: ServoLimits) -> bool:
    if not ik_valid(q):
        return False
    if (
        qref is not None
        and max(abs(a - b) for a, b in zip(q, qref)) > limits.branch_tolerance_rad
    ):
        return False
    violation = limit_violation(q, qref, dt, limits)
    if violation == "elbow" and qref is not None:
        emin = float(limits.elbow_min_rad or 0.0)
        if (
            abs(float(qref[2])) < emin
            and abs(float(q[2])) > abs(float(qref[2])) + 1e-6
            and not speed_violation(q, qref, dt, limits)
        ):
            return True
    return violation is None


def limited_step(
    solve_ik: Callable,
    prev: np.ndarray,
    target: np.ndarray,
    qref,
    dt: float,
    limits: ServoLimits,
):
    """Existing bounded bisection/slide, returning pose, joints, fraction, mode.

    ``solve_ik(pose, qref)`` returns six finite joints or an invalid/empty
    sequence. This function never transmits a command. None means hold.
    """
    best = None
    best_disp = -1.0
    candidates = [("step", target)]
    shoulder = np.array([0.0, 0.0, limits.shoulder_height_m])
    radius = prev[:3] - shoulder
    norm = float(np.linalg.norm(radius))
    if norm > 1e-6:
        delta = target[:3] - prev[:3]
        tangent = target.copy()
        tangent[:3] = (
            prev[:3] + delta - radius * (float(np.dot(delta, radius)) / (norm * norm))
        )
        candidates.append(("slide", tangent))
    for mode, candidate in candidates:
        low, high, found = 0.0, 1.0, None
        for _ in range(limits.bisection_iterations):
            mid = 0.5 * (low + high)
            pose = prev + mid * (candidate - prev)
            q = solve_ik(pose, qref)
            if feasible(q, qref, dt, limits):
                found = (pose, list(q), mid, mode)
                low = mid
            else:
                high = mid
        if found is not None:
            displacement = float(np.linalg.norm(found[0][:3] - prev[:3]))
            if displacement > best_disp:
                best, best_disp = found, displacement
        if best is not None and best[3] == "step" and best[2] >= 0.5:
            break
    return best


@dataclass(frozen=True)
class ServoStep:
    q: np.ndarray | None
    pose: np.ndarray | None
    reason: str
    mode: str
    fraction: float | None
    violation: str | None
    ik_calls: int
    all_ik_valid: bool = False
    all_ik_on_branch: bool = False

    @property
    def accepted(self) -> bool:
        return self.q is not None


def select_servo_step(
    target_pose, previous_pose, qref, dt, solve_ik, limits: ServoLimits
) -> ServoStep:
    """CPU selection matching enabled URArm.servo_l, without sending a command.

    The caller owns the last accepted anchor and consecutive-reject stop. This
    entry point additionally rejects malformed inputs before calling IK. Native
    driver wrappers use the lower-level extracted functions to preserve their
    historical disabled path exactly.
    """
    if not limits.enabled:
        raise ValueError("select_servo_step requires explicit enabled limits")
    for value, name in (
        (target_pose, "target"),
        (previous_pose, "anchor"),
        (qref, "qref"),
    ):
        if not ik_valid(value):
            return ServoStep(None, None, "invalid_" + name, "hold", None, None, 0)
    if not np.isfinite(dt) or dt <= 0:
        return ServoStep(None, None, "invalid_dt", "hold", None, None, 0)
    if limits.elbow_min_rad is not None and not (0 <= limits.elbow_min_rad < np.pi):
        raise ValueError("elbow_min_rad must be within [0, pi)")
    if limits.joint_speed_max_rad_s is not None and not (
        np.isfinite(limits.joint_speed_max_rad_s) and limits.joint_speed_max_rad_s > 0
    ):
        raise ValueError("joint_speed_max_rad_s must be finite and positive")
    target = np.asarray(target_pose, dtype=float).copy()
    prev = np.asarray(previous_pose, dtype=float)
    calls = 0
    all_valid = True
    all_on_branch = True

    def counted(pose, seed):
        nonlocal calls, all_valid, all_on_branch
        calls += 1
        result = solve_ik(pose, seed)
        valid = ik_valid(result)
        all_valid = all_valid and valid
        on_branch = valid and max(
            abs(float(a) - float(b)) for a, b in zip(result, seed)
        ) <= limits.branch_tolerance_rad
        all_on_branch = all_on_branch and on_branch
        return result

    q = counted(target, qref)
    if not ik_valid(q):
        q = []
    if (
        len(q)
        and max(abs(a - b) for a, b in zip(q, qref)) > limits.branch_tolerance_rad
    ):
        return ServoStep(None, None, "ik_branch", "hold", None, None, calls,
                         all_valid, all_on_branch)
    violation = limit_violation(q, qref, dt, limits)
    if violation is None:
        return ServoStep(np.asarray(q), target, "sent", "unchanged", 1.0, None, calls,
                         all_valid, all_on_branch)
    target[3:6] = rotvec_nearest(prev[3:6], target[3:6])
    result = limited_step(counted, prev, target, qref, dt, limits)
    if result is None:
        return ServoStep(None, None, "limiter_hold", "hold", None, violation, calls,
                         all_valid, all_on_branch)
    pose, q, fraction, mode = result
    return ServoStep(np.asarray(q), pose, "sent", mode, fraction, violation, calls,
                     all_valid, all_on_branch)
