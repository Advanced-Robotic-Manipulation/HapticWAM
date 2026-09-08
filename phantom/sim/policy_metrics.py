"""Policy task scoring from free-body state, independent of recorded trajectories.

Times are physical simulation seconds; positions are metres in the UR base
frame. Quaternions use WXYZ. The bin config's Z coordinate is its bottom, as
in :mod:`phantom.sim.scene`, not its geometric centre. Contact is the PhysX
pad-to-packet normal force, never net pad force, Robotiq OBJ or a planner flag.

These fixed thresholds define a simulator benchmark, not calibrated hardware
success. Object provenance is an audited metadata claim: arrays cannot prove
the absence of hidden attachments or pose writes. Collision reporting is
explicitly incomplete unless independent whole-robot monitoring is supplied.

``require_support_verified_release`` opts into packet-to-bin and packet-to-all-
robot contact telemetry. It requires bin support and robot unloading throughout
the same settling interval; the default preserves the historical scoring gates.
"""

from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np

from phantom.sim.geometry import bin_geometry

DEFAULT_THRESHOLDS = {
    "contact_force_n": 0.1,
    "acquisition_hold_s": 0.15,
    "lift_height_m": 0.03,
    "lift_hold_s": 0.5,
    "carry_distance_m": 0.05,
    "carry_hold_s": 0.25,
    "contact_gap_s": 0.15,
    "settle_hold_s": 0.5,
    "settle_speed_m_s": 0.03,
    "settle_angular_speed_rad_s": 0.5,
    "bin_tolerance_m": 0.002,
    "reach_distance_m": 0.02,
    "reach_max_closure": 0.25,
    "drop_height_m": 0.02,
    "drop_hold_s": 0.1,
    "clock_tolerance_s": 0.001,
    "max_sample_gap_s": 0.2,
    "require_support_verified_release": False,
    "support_robot_force_max_n": 0.1,
    "support_bin_force_min_n": 0.1,
}


def _runs(mask):
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(bounds[::2], bounds[1::2] - 1))


def _sustained(t, mask, hold_s):
    """Return onset and confirmation indices, without extrapolating sample time."""
    for first, last in _runs(mask):
        if t[last] - t[first] + 1e-9 >= hold_s:
            confirmed = min(last, int(np.searchsorted(t, t[first] + hold_s - 1e-9)))
            return int(first), int(confirmed)
    return None


def _rotations(q):
    w, x, y, z = q.T
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    ).transpose(2, 0, 1)


def _summary(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": len(a),
        "mean": float(a.mean()),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _control_metrics(execution, planners, run):
    reasons = Counter(
        str(row.get("ik_reason", "unknown"))
        for row in execution
        if row.get("ik_success") is False
    )
    times = np.asarray([row.get("t", np.nan) for row in execution], float)
    valid_times = bool(
        np.isfinite(times).all() and (len(times) < 2 or (np.diff(times) > 0).all())
    )
    hold = np.array(
        [
            bool(row.get("stopped"))
            or row.get("ik_success") is False
            or bool(row.get("diagnostics", {}).get("stale_plan_hold"))
            for row in execution
        ]
    )
    hold_duration = (
        float(np.dot(np.diff(times), hold[:-1]))
        if valid_times and len(times) > 1
        else None
    )
    stale_count = sum(
        bool(row.get("diagnostics", {}).get("stale_plan_hold")) for row in execution
    )
    stop_reason = run.get("policy_stop_reason")
    if stop_reason is None:
        stop_reason = next(
            (
                row.get("stop_reason")
                for row in reversed(execution)
                if row.get("stopped")
            ),
            None,
        )
    return {
        "stop_reason": stop_reason,
        "execution_rows": len(execution),
        "execution_clock_valid": valid_times if execution else None,
        "ik_rejects": sum(reasons.values()),
        "ik_reject_reasons": dict(reasons),
        "hold_duration_s": hold_duration,
        "stale_plan_hold_rows": stale_count,
        "replans": len(planners),
        "inference_errors": sum(
            row.get("status") == "inference_error" for row in planners
        ),
        "plans_activated": sum(row.get("activated_at") is not None for row in planners),
        "effective_latency_s": _summary(
            [row.get("latency_s", np.nan) for row in planners]
        ),
        "inference_wall_time_s": _summary(
            [row.get("inference_wall_time_s", np.nan) for row in planners]
        ),
        "activation_delay_s": _summary(
            [
                row["activated_at"]
                - row.get("t_created", row.get("t", row["activated_at"]))
                for row in planners
                if row.get("activated_at") is not None
            ]
        ),
        "lift_complete_is_task_success": False,
    }


def evaluate_policy_trace(
    trace,
    config,
    thresholds=None,
    *,
    run=None,
    execution_trace=None,
    planner_trace=None,
):
    """Return JSON-native policy metrics; malformed traces cannot earn success.

    ``run`` is run.json; optional execution/planner arguments are decoded rows.
    Missing provenance makes ``valid_for_scoring`` false. Observed task flags
    are still returned separately, so an invalid run is never silently dropped
    or conflated with a valid failed policy. No real episode or action timing
    is accepted by this API. Reach is diagnostic, excluded from full-task gates.
    """
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        unknown = set(thresholds) - set(limits)
        if unknown:
            raise ValueError(f"Unknown policy thresholds: {sorted(unknown)}")
        limits.update(thresholds)
    if not isinstance(limits["require_support_verified_release"], (bool, np.bool_)):
        raise TypeError("require_support_verified_release must be boolean")
    if any(
        not np.isfinite(v) or v <= 0
        for key, v in limits.items()
        if key != "require_support_verified_release"
    ):
        raise ValueError("Policy thresholds must be finite and positive")
    if limits["reach_max_closure"] > 1:
        raise ValueError("reach_max_closure must use normalized closure")
    run = {} if run is None else run
    execution = [] if execution_trace is None else list(execution_trace)
    planners = [] if planner_trace is None else list(planner_trace)
    names = [
        "reach_before_closure",
        "acquired",
        "lifted",
        "carried",
        "released_in_bin",
        "full_task",
        "dropped",
    ]
    result = {
        "schema_version": 1,
        "valid_for_scoring": False,
        "invalid_reasons": [],
        "thresholds": limits,
        "outcomes": dict.fromkeys(names, False),
        "event_times_s": dict.fromkeys(
            [
                "reach",
                "closure",
                "acquisition",
                "lift",
                "carry",
                "release_in_bin",
                "full_task",
                "first_drop",
            ]
        ),
        "object": {},
        "placement_support": {
            "required": bool(limits["require_support_verified_release"]),
            "telemetry_status": "missing",
            "verified_placement": None,
            "confirmation_time_s": None,
            "robot_contact_peak_n": None,
            "bin_contact_peak_n": None,
            "semantics": "Positive normal contact magnitudes with ALL robot bodies, including both pads, and the five physical bin colliders; never cancellable net force",
        },
        "control": _control_metrics(execution, planners, run),
        "collisions": {
            "observability": "partial: pad contacts and reported events; no whole-robot collision certification",
            "reported_count": 0,
            "pad_environment_force_peak_n": None,
        },
    }
    invalid = result["invalid_reasons"]
    if run.get("mode") != "policy":
        invalid.append("run_mode_is_not_policy")
    provenance = run.get("object_dynamics", {})
    if not (
        provenance.get("rigid_body_dynamic") is True
        and provenance.get("kinematic") is False
        and provenance.get("attachments") == []
        and provenance.get("pose_writes_after_initialization") == 0
    ):
        invalid.append("free_body_provenance_missing_or_failed")
    t = np.asarray(trace.get("t", []), float)
    if (
        t.ndim != 1
        or len(t) < 3
        or not np.isfinite(t).all()
        or not (np.diff(t) > 0).all()
    ):
        invalid.append(
            "timestamps_must_be_finite_strictly_increasing_with_at_least_three_rows"
        )
        return result
    n = len(t)
    shapes = {
        "physics_t": (n,),
        "q": (n, 6),
        "tcp": (n, 6),
        "gripper": (n, 2),
        "waffle_position": (n, 3),
        "waffle_orientation_wxyz": (n, 4),
        "pad_position": (n, 2, 3),
        "pad_packet_normal_force": (n, 2),
    }
    arrays = {}
    bad_array = False
    for key, shape in shapes.items():
        a = np.asarray(trace.get(key, []), float)
        if a.shape != shape or not np.isfinite(a).all():
            invalid.append(f"missing_misaligned_or_nonfinite:{key}")
            bad_array = True
        arrays[key] = a
    support_keys = ("packet_robot_normal_force", "packet_bin_normal_force")
    support_arrays = {}
    support_errors = []
    for key in support_keys:
        a = np.asarray(trace.get(key, []), float)
        if a.shape != (n,) or not np.isfinite(a).all():
            support_errors.append(f"missing_misaligned_or_nonfinite:{key}")
        elif (a < -1e-8).any():
            support_errors.append(f"negative_contact_normal_magnitude:{key}")
        else:
            support_arrays[key] = np.maximum(a, 0)
    support_available = not support_errors
    support_result = result["placement_support"]
    if support_available:
        support_result.update(
            telemetry_status="available",
            verified_placement=False,
            robot_contact_peak_n=float(support_arrays[support_keys[0]].max()),
            bin_contact_peak_n=float(support_arrays[support_keys[1]].max()),
        )
    else:
        support_result["telemetry_status"] = (
            "invalid" if any(key in trace for key in support_keys) else "missing"
        )
        support_result["telemetry_errors"] = support_errors
        if limits["require_support_verified_release"]:
            invalid.extend(support_errors)
            bad_array = True
    if bad_array:
        return result
    intervals = np.diff(t)
    skew = float(np.max(np.abs(arrays["physics_t"] - t)))
    result["trace_sampling"] = {
        "rows": n,
        "duration_s": float(t[-1] - t[0]),
        "maximum_interval_s": float(intervals.max()),
        "physics_clock_max_error_s": skew,
        "event_time_uncertainty_s": float(intervals.max()),
    }
    if (
        skew > limits["clock_tolerance_s"]
        or not (np.diff(arrays["physics_t"]) > 0).all()
    ):
        invalid.append("physics_clock_mismatch")
    if intervals.max() > limits["max_sample_gap_s"] + 1e-9:
        invalid.append("trace_sample_gap_exceeds_event_resolution")
    q = arrays["waffle_orientation_wxyz"]
    norms = np.linalg.norm(q, axis=1)
    if not np.allclose(norms, 1, atol=1e-3, rtol=0):
        invalid.append("object_quaternion_not_unit")
        return result
    if (arrays["pad_packet_normal_force"] < -1e-8).any():
        invalid.append("negative_contact_normal_magnitude")
    if (arrays["gripper"][:, 0] < 0).any() or (arrays["gripper"][:, 0] > 1).any():
        invalid.append("gripper_closure_outside_normalized_range")
    # Invalid state/clock data must not generate a partial or full success flag.
    trace_invalid = [
        reason
        for reason in invalid
        if reason
        not in ("run_mode_is_not_policy", "free_body_provenance_missing_or_failed")
    ]
    if trace_invalid:
        return result
    q = q / norms[:, None]
    rotation = _rotations(q)
    position = arrays["waffle_position"]
    size = np.asarray(config["waffle"]["size"], float)
    geometry = bin_geometry(config["bin"])
    if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Object must have finite positive physical dimensions")
    corners = np.asarray(list(product([-1, 1], repeat=3))) * size / 2
    world_corners = np.einsum("nij,kj->nki", rotation, corners) + position[:, None]
    tol = limits["bin_tolerance_m"]
    lower, upper = geometry.interior_bounds
    over_bin = (
        (world_corners[:, :, :2] >= lower[:2] - tol)
        & (world_corners[:, :, :2] <= upper[:2] + tol)
    ).all(axis=(1, 2))
    inside_bin = (
        over_bin
        & (world_corners[:, :, 2].min(axis=1) >= lower[2] - tol)
        & (world_corners[:, :, 2].max(axis=1) <= upper[2] + tol)
    )
    speed = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1) / intervals]
    angular = np.r_[
        0.0,
        2
        * np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", q[1:], q[:-1])), 0, 1))
        / intervals,
    ]
    contact = arrays["pad_packet_normal_force"] > limits["contact_force_n"]
    bilateral, unloaded = contact.all(axis=1), ~contact.any(axis=1)
    retained = bilateral.copy()
    for first, last in _runs(~bilateral):
        if (
            first > 0
            and last < n - 1
            and t[last + 1] - t[first - 1] <= limits["contact_gap_s"] + 1e-9
        ):
            retained[first : last + 1] = True
    lift = position[:, 2] - position[0, 2]
    midpoint = arrays["pad_position"].mean(axis=1)
    local_midpoint = np.einsum("nji,nj->ni", rotation, midpoint - position)
    reach_distance = np.linalg.norm(
        np.maximum(np.abs(local_midpoint) - size / 2, 0), axis=1
    )
    closure = arrays["gripper"][:, 0]
    closed_indices = np.flatnonzero(closure >= limits["reach_max_closure"])
    close_i = int(closed_indices[0]) if len(closed_indices) else n
    # Some recordings start partially closed. Preserve the frozen absolute
    # reach gate, but independently measure approach error before the first
    # additional closing command at the hardware's two-count deadband.
    command_closures = [
        (float(row["t"]), float(row["gripper_command"]))
        for row in execution
        if row.get("gripper_command") is not None
        and np.isfinite(row.get("t", np.nan))
        and np.isfinite(row["gripper_command"])
    ]
    if command_closures:
        onset = next(
            (when for when, value in command_closures if value > closure[0] + 2 / 255),
            None,
        )
        onset_source = "execution gripper_command > initial measured closure + 2/255"
    else:
        increasing = np.flatnonzero(closure > closure[0] + 0.02)
        onset = float(t[increasing[0]]) if len(increasing) else None
        onset_source = "fallback sampled measured closure > initial closure + 0.02"
    before_closing = (
        int(np.searchsorted(t, onset, side="left") - 1) if onset is not None else -1
    )
    reach_error = float(reach_distance[before_closing]) if before_closing >= 0 else None
    result["reach_diagnostic"] = {
        "absolute_threshold_already_exceeded_at_start": bool(
            closure[0] >= limits["reach_max_closure"]
        ),
        "initial_measured_closure": float(closure[0]),
        "first_closing_motion_s": onset,
        "closing_motion_source": onset_source,
        "reach_error_before_first_closing_motion_m": reach_error,
        "sample_time_s": float(t[before_closing]) if before_closing >= 0 else None,
        "definition": "Last sampled pad-midpoint distance to the object OBB before the first additional closing command; diagnostic only, no change to frozen full-task gates.",
    }
    reached = np.flatnonzero(
        (reach_distance <= limits["reach_distance_m"])
        & (closure < limits["reach_max_closure"])
        & (np.arange(n) < close_i)
    )
    events, outcomes = result["event_times_s"], result["outcomes"]
    if close_i < n:
        events["closure"] = float(t[close_i])
    if len(reached):
        outcomes["reach_before_closure"] = True
        events["reach"] = float(t[reached[0]])
    max_carry, first_lift = 0.0, None
    settled_without_support = (
        inside_bin
        & unloaded
        & (speed <= limits["settle_speed_m_s"])
        & (angular <= limits["settle_angular_speed_rad_s"])
    )
    supported = np.zeros(n, dtype=bool)
    if support_available:
        supported = (
            support_arrays["packet_robot_normal_force"]
            <= limits["support_robot_force_max_n"]
        ) & (
            support_arrays["packet_bin_normal_force"]
            > limits["support_bin_force_min_n"]
        )
    settled = settled_without_support & (
        supported if limits["require_support_verified_release"] else True
    )
    # Each retained-contact episode must independently progress through grasp,
    # lift and carry. Separate failed attempts cannot be spliced into success.
    episodes = _runs(retained)
    for episode_index, (start, end) in enumerate(episodes):
        in_episode = (np.arange(n) >= start) & (np.arange(n) <= end)
        acquisition = _sustained(
            t, bilateral & in_episode, limits["acquisition_hold_s"]
        )
        if acquisition is None:
            continue
        outcomes["acquired"] = True
        if events["acquisition"] is None:
            events["acquisition"] = float(t[acquisition[0]])
        lift_event = _sustained(
            t,
            in_episode
            & (np.arange(n) >= acquisition[1])
            & (lift >= limits["lift_height_m"]),
            limits["lift_hold_s"],
        )
        if lift_event is None:
            continue
        outcomes["lifted"] = True
        if first_lift is None:
            first_lift = lift_event[0]
            events["lift"] = float(t[first_lift])
        distance = np.linalg.norm(position[:, :2] - position[lift_event[0], :2], axis=1)
        max_carry = max(
            max_carry,
            float(distance[in_episode & (np.arange(n) >= lift_event[0])].max()),
        )
        carry = _sustained(
            t,
            in_episode
            & (np.arange(n) >= lift_event[1])
            & (lift >= limits["lift_height_m"])
            & (distance >= limits["carry_distance_m"]),
            limits["carry_hold_s"],
        )
        if carry is None:
            continue
        outcomes["carried"] = True
        if events["carry"] is None:
            events["carry"] = float(t[carry[0]])
        next_start = (
            episodes[episode_index + 1][0] if episode_index + 1 < len(episodes) else n
        )
        released = np.flatnonzero(
            unloaded & (np.arange(n) > end) & (np.arange(n) < next_start)
        )
        # Require release above the bin footprint, then actual stable placement.
        # Landing in the bin after an outside throw is not a retained placement.
        if not len(released) or not over_bin[released[0]]:
            continue
        release_interval = (np.arange(n) >= released[0]) & (np.arange(n) < next_start)
        if support_available:
            support_confirmation = _sustained(
                t,
                settled_without_support & supported & release_interval,
                limits["settle_hold_s"],
            )
            if support_confirmation is not None:
                support_result["verified_placement"] = True
                if support_result["confirmation_time_s"] is None:
                    support_result["confirmation_time_s"] = float(
                        t[support_confirmation[1]]
                    )
        placement = _sustained(
            t,
            settled & release_interval,
            limits["settle_hold_s"],
        )
        if placement is not None:
            outcomes["released_in_bin"] = outcomes["full_task"] = True
            events["release_in_bin"] = float(t[released[0]])
            events["full_task"] = float(t[placement[1]])
            break
    if first_lift is not None:
        last_held_z = np.empty(n)
        previous = position[first_lift, 2]
        for i in range(n):
            if bilateral[i]:
                previous = position[i, 2]
            last_held_z[i] = previous
        drop = _sustained(
            t,
            (np.arange(n) >= first_lift)
            & unloaded
            & ~over_bin
            & (last_held_z - position[:, 2] >= limits["drop_height_m"]),
            limits["drop_hold_s"],
        )
        if drop is not None:
            outcomes["dropped"] = True
            events["first_drop"] = float(t[drop[0]])
    result["object"] = {
        "initial_center_m": position[0].tolist(),
        "final_center_m": position[-1].tolist(),
        "max_lift_m": float(lift.max()),
        "carry_distance_m": max_carry,
        "final_inside_bin": bool(inside_bin[-1]),
        "final_contacts_unloaded": bool(unloaded[-1]),
        "final_speed_m_s": float(speed[-1]),
        "final_angular_speed_rad_s": float(angular[-1]),
        "minimum_pad_midpoint_to_object_m": float(reach_distance.min()),
        "reach_error_before_first_closing_motion_m": reach_error,
        "bilateral_contact_duration_s": float(np.dot(intervals, bilateral[:-1])),
        "pad_packet_normal_force_peak_n": float(
            arrays["pad_packet_normal_force"].max()
        ),
    }
    collisions = [
        event
        for event in run.get("events", [])
        if "collision" in str(event.get("event", "")).lower()
    ]
    result["collisions"]["reported_count"] = len(collisions)
    result["collisions"]["reported_events"] = collisions
    if "pad_force" in trace and "pad_packet_force" in trace:
        net, packet = (
            np.asarray(trace["pad_force"]),
            np.asarray(trace["pad_packet_force"]),
        )
        if (
            net.shape == packet.shape == (n, 2, 3)
            and np.isfinite(net).all()
            and np.isfinite(packet).all()
        ):
            result["collisions"]["pad_environment_force_peak_n"] = float(
                np.linalg.norm(net - packet, axis=-1).max()
            )
    result["valid_for_scoring"] = not invalid
    result["limitations"] = [
        "Reach uses the pad midpoint and estimated object OBB; it is diagnostic, not a full-task gate.",
        "Events require observed elapsed time; sub-sample contact, collisions and drops are not certified.",
        "A lift_complete stop or Robotiq OBJ flag cannot substitute for object-state success.",
        "Contact-free force cancellation and robot-body collisions are not excluded by the pad-force diagnostic.",
        "Without required support telemetry, zero pad load cannot exclude support by another robot part inside the bin.",
    ]
    return result
