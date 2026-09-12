"""Optional, bounded FK-aware rate refinement; never submits or approves a drive.

The original selection is retained exactly when its FK already meets the
existing per-tick rate contract. Numerical overshoot gets at most three
re-solves inside that contract. The caller still runs its original final
FK/rate/envelope and submission checks. No measured state or terminal hold is
changed here, and no safety limit or final comparison tolerance is enlarged.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from phantom.data.derived import rotvec_nearest
from phantom.drivers.servo_limiter import select_servo_step


def rate_displacements(previous_pose, final_pose):
    previous, final = (np.asarray(v, dtype=float) for v in (previous_pose, final_pose))
    if any(v.shape != (6,) or not np.isfinite(v).all() for v in (previous, final)):
        raise ValueError("finite six-dimensional FK and prior pose required")
    return (float(np.linalg.norm(final[:3] - previous[:3])),
            float(np.linalg.norm(rotvec_nearest(previous[3:], final[3:]) - previous[3:])))


def refine_rate_selection(selection, previous_pose, qref, dt, solve_ik, limits,
                          forward_pose, *, linear_speed, angular_speed, solver_inset_m):
    """Return a selection and evidence; final acceptance remains with the caller."""
    values = (dt, linear_speed, angular_speed, solver_inset_m)
    if any(isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("finite positive rate-refinement limits required")
    evidence = {"variant": "fk_rate_interior_v1", "selector_retry_limit": 3,
                "maximum_extra_ik_calls": 3 * (1 + 2 * limits.bisection_iterations),
                "initial_selection_preserved": True, "attempts": [],
                "final_rate_limits_changed": False, "drive_submitted": False}
    if not selection.accepted or not selection.all_ik_valid or not selection.all_ik_on_branch:
        evidence["reason"] = "original_selection_not_verified"
        return selection, evidence
    previous = np.asarray(previous_pose, dtype=float)
    initial_fk = np.asarray(forward_pose(selection.q), dtype=float)
    linear, angular = rate_displacements(previous, initial_fk)
    linear_cap, angular_cap = linear_speed * dt, angular_speed * dt
    evidence.update(initial_fk_tcp=initial_fk.tolist(), initial_translation_m=linear,
                    initial_rotation_vector_rad=angular, translation_cap_m=linear_cap,
                    rotation_vector_cap_rad=angular_cap)

    def fits(displacement):
        # Exactly the original final-rate comparison, without extra tolerance.
        return displacement[0] <= linear_cap + 1e-9 and displacement[1] <= angular_cap + 1e-9

    if fits((linear, angular)):
        evidence["reason"] = "original_fk_within_rate_budget"
        return selection, evidence
    target = np.asarray(selection.pose, dtype=float).copy()
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("finite six-dimensional solver target required")
    target[3:] = rotvec_nearest(previous[3:], target[3:])
    delta = target - previous
    linear_norm, angular_norm = np.linalg.norm(delta[:3]), np.linalg.norm(delta[3:])
    # Reuse the existing geometric solver inset, capped at half a tick's
    # translation budget. Apply the same fractional reserve to orientation;
    # final FK checks, not an assumed IK orientation error, decide acceptance.
    reserve = min(solver_inset_m, linear_cap / 2.)
    reserve_fraction = reserve / linear_cap
    scales = (min(1., (linear_cap - reserve) / linear_norm) if linear_norm else 1.,
              min(1., angular_cap * (1. - reserve_fraction) / angular_norm) if angular_norm else 1.)
    common_scale = min(scales)
    evidence.update(solver_inset_m=solver_inset_m, translation_reserve_m=reserve,
                    orientation_reserve_fraction=reserve_fraction,
                    original_solver_target=target.tolist(), common_initial_pose_fraction=common_scale)
    total_calls = selection.ik_calls
    for attempt in range(3):
        factor = .5 ** attempt
        candidate = previous.copy()
        # One common fraction preserves this local Cartesian/rotvec direction.
        # Independent axis shrinking can demand faster compensating joints.
        candidate += delta * common_scale * factor
        trial = select_servo_step(candidate, previous, qref, dt, solve_ik, limits)
        total_calls += trial.ik_calls
        item = {"attempt": attempt + 1, "target_tcp": candidate.tolist(),
                "translation_fraction": common_scale * factor,
                "rotation_vector_fraction": common_scale * factor,
                "ik_calls": trial.ik_calls, "mode": trial.mode,
                "all_ik_valid": trial.all_ik_valid, "all_ik_on_branch": trial.all_ik_on_branch,
                "selection_accepted": trial.accepted}
        evidence["attempts"].append(item)
        if not trial.all_ik_valid or not trial.all_ik_on_branch:
            # Never hide a malformed/off-branch intermediate solve behind a
            # later successful one. The unchanged final verifier rejects this.
            evidence["reason"] = "unverified_refinement_solve"
            return replace(selection, ik_calls=total_calls,
                           all_ik_valid=trial.all_ik_valid,
                           all_ik_on_branch=trial.all_ik_on_branch), evidence
        if not trial.accepted:
            continue
        fk = np.asarray(forward_pose(trial.q), dtype=float)
        displacement = rate_displacements(previous, fk)
        item.update(final_fk_tcp=fk.tolist(), translation_m=displacement[0],
                    rotation_vector_rad=displacement[1], rate_budget_pass=fits(displacement))
        if fits(displacement):
            evidence.update(reason="interior_fk_within_rate_budget", initial_selection_preserved=False,
                            selected_attempt=attempt + 1,
                            target_joint_change_max_rad=float(np.max(np.abs(trial.q - selection.q))))
            return replace(trial, mode="rate_interior", fraction=None, ik_calls=total_calls), evidence
    evidence["reason"] = "bounded_refinement_exhausted"
    # Preserve the original rejected proposal for the unchanged final guard;
    # failed refinement never creates a hold, ACK, grace interval or new budget.
    return replace(selection, ik_calls=total_calls), evidence
