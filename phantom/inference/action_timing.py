"""Default-off observation-epoch candidate; numpy-only timing primitives.

An action is an interval delta: action j ends at observation+(j+1)/rate.
The playback grid therefore contains interval START times. Absolute gripper
closure is still zero-order held at the played index (0=open, 1=closed).
These helpers do not change action values, speed/safety limits or motor targets.
"""
from __future__ import annotations

import numpy as np

ACTION_TIME_ORIGINS = ("inference_ready", "observation")


def validate_action_time_origin(value):
    if value not in ACTION_TIME_ORIGINS:
        raise ValueError(f"action_time_origin must be one of {ACTION_TIME_ORIGINS}")
    return value


def plan_action_time_origin(plan):
    return validate_action_time_origin(
        (getattr(plan, "diag", None) or {}).get("action_time_origin", "inference_ready")
    )


def observation_cpk_history(obs_t, prev_plan, latent_dt):
    """Age the previous prediction by observation chronology, never latency.

    Expired *predicted* contact is removed, while current teacher tactile and
    previous action comparison remain available. No measured modality is
    zeroed or replaced. Within support, retain the existing round-and-bound
    summary-index convention; beyond support there is no contemporaneous CPK.
    """
    if not np.isfinite(obs_t) or not np.isfinite(latent_dt) or latent_dt <= 0:
        raise ValueError("invalid observation time or contact latent period")
    diag = {"prev_cpk_time_reference": "observation_age",
            "prev_cpk_disposition": "no_previous_plan"}
    if prev_plan is None:
        return None, 0, diag
    age = float(obs_t - prev_plan.t_created)
    if not np.isfinite(age) or age < 0:
        raise ValueError("previous plan observation chronology is invalid")
    diag["prev_cpk_observation_age_s"] = age
    cpk = prev_plan.cpk
    if cpk is None:
        diag["prev_cpk_disposition"] = "previous_package_absent"
        return None, 0, diag
    horizon = cpk.horizon
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("previous contact package horizon must be positive")
    support = horizon * latent_dt
    raw_step = int(round(age/latent_dt))
    diag.update(prev_cpk_supported_horizon_s=float(support),
                prev_cpk_unclipped_summary_step=raw_step)
    if age > support:
        diag.update(prev_cpk_disposition="expired_removed", prev_cpk_summary_step=None)
        return None, 0, diag
    step = min(raw_step, int(horizon)-1)
    diag.update(prev_cpk_disposition="within_supported_horizon",
                prev_cpk_summary_step=step,
                prev_cpk_index_bounded=step != raw_step)
    return cpk, step, diag


def observation_submission_gate(plan, now, rate, max_play_steps,
                                grip_play_steps, min_lead, has_last_command):
    """Lead from actual cap interpolation, intersected with original plan lead.

    Existing pose_at consumes the last allowed delta through cap-1e-6, not
    just that interval's start. This prospective endpoint_v2 admission rule
    uses that exact endpoint while preserving the original full-plan final
    interval-START bound. Grip cap remains separate and absolute. No action
    index, physical limit or minimum-lead threshold is extended.
    """
    times = np.asarray(plan.action_times, dtype=float)
    actions = np.asarray(plan.actions)
    if (actions.ndim != 2 or actions.shape[1] != 7 or not len(actions)
            or times.shape != (len(actions),) or not np.isfinite(times).all()
            or not np.isfinite(actions).all() or not np.isfinite(now)
            or not np.isfinite(rate) or rate <= 0
            or not np.isfinite(min_lead) or min_lead < 0
            or (len(times) > 1 and not np.allclose(np.diff(times), 1/rate))):
        raise ValueError("invalid observation-epoch action timing or actions")
    if not np.isfinite(plan.t_created) or not np.isclose(
            times[0], plan.t_created, rtol=0, atol=1e-8):
        raise ValueError("observation-epoch action grid must start at t_created")
    for cap in (max_play_steps, grip_play_steps):
        if cap is not None and (isinstance(cap, bool) or int(cap) != cap or cap < 1):
            raise ValueError("play caps must be positive integer indices or None")
    H = len(actions)
    pose_cap = min(H, max_play_steps) if max_play_steps else H
    grip_cap = min(pose_cap, grip_play_steps) if grip_play_steps else pose_cap
    age = float(now - times[0])
    phase = max(0.0, age * rate)
    pose_endpoint = float(times[0] + (pose_cap - 1e-6)/rate)
    full_plan_last_start = float(times[-1])
    covered_until = min(full_plan_last_start, pose_endpoint)
    reason = None
    if not has_last_command:
        reason = "no_last_accepted_command"
    elif age < -1e-9:
        reason = "future_action_epoch"
    elif phase >= pose_cap:
        reason = "pose_cap_expired"
    elif covered_until <= now + min_lead + 1e-9:
        reason = "insufficient_capped_lead"
    return {
        "accepted": reason is None,
        "reason": reason,
        "age_s": age,
        "phase_steps": phase,
        "pose_cap_absolute_steps": int(pose_cap),
        "grip_cap_absolute_steps": int(grip_cap),
        "remaining_pose_interval_steps": max(0.0, pose_cap-phase),
        "remaining_conservative_lead_s": covered_until-now,
        "minimum_lead_s": float(min_lead),
        "lead_convention": "cap_interpolation_endpoint_intersect_original_last_start",
        "timing_criteria_revision": "endpoint_v2",
        "pose_interpolation_endpoint_t": pose_endpoint,
        "original_full_plan_last_start_t": full_plan_last_start,
        "covered_until_t": covered_until,
    }



def fresh_startup_anchor(pose, captured_at, now, stale_limit):
    anchor = np.asarray(pose, dtype=float)
    age = float(now-captured_at)
    if (anchor.shape != (6,) or not np.isfinite(anchor).all()
            or not np.isfinite([captured_at, now, stale_limit, age]).all()
            or age < 0 or age > stale_limit):
        raise ValueError("startup arm feedback is missing, invalid, future or stale")
    return anchor.copy()


def submission_diagnostics(timing, anchor_kind, feedback_t=None, anchor=None):
    # Flat scalars/lists survive both native planner trace and sim callbacks.
    result = {"action_time_submission_"+key: value for key, value in timing.items()}
    result["action_time_anchor_kind"] = anchor_kind
    if feedback_t is not None:
        result["action_time_anchor_feedback_t"] = float(feedback_t)
    if anchor is not None:
        result["action_time_anchor_pose"] = np.asarray(anchor).tolist()
    return result
