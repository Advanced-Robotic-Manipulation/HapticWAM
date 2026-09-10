"""Robot-feedback observer of an already executed, unlatched policy opening.

This never permits an opening, arms a latch, or infers object-task success.
Accepted-command provenance and independently advancing capture timestamps are
required; a requested action or repeatedly held sensor sample is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnlatchedFinishConfig:
    tcp_min_m: tuple[float, float, float]
    tcp_max_m: tuple[float, float, float]
    command_delta_min: float = 12 / 255
    measured_delta_min: float = 6 / 255
    reclose_tolerance: float = 2 / 255
    activity_hold_s: float = 0.2
    history_timeout_s: float = 10.0
    feedback_max_age_s: float = 0.25

    def __post_init__(self):
        for name in ("tcp_min_m", "tcp_max_m"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite base-frame metres")
            object.__setattr__(self, name, tuple(float(v) for v in value))
        if not np.all(np.asarray(self.tcp_min_m) < np.asarray(self.tcp_max_m)):
            raise ValueError("finish-observation bounds must have positive volume")
        for name in ("command_delta_min", "measured_delta_min", "reclose_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be a finite closure fraction in (0, 1)")
        if self.reclose_tolerance >= min(self.command_delta_min, self.measured_delta_min):
            raise ValueError("reclose tolerance must be below movement thresholds")
        for name in ("activity_hold_s", "history_timeout_s", "feedback_max_age_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


def feedback_capture_times(arm_t, gripper_t, tactile_times, sensors):
    """Capture times must come from the SAME reads that supplied the values."""
    required = {"arm": arm_t, "gripper": gripper_t}
    required.update({f"tactile_{s.name}": tactile_times.get(s.name) for s in sensors})
    return required


def acknowledge_gripper(previous, t, command, policy_eligible):
    """Keep first successful ACK of an unchanged value/provenance run.

    Repeatedly streaming an identical target must not move its causal origin
    ahead of slower measured sensors. A provenance change creates a new run,
    even when the numerical target is identical.
    """
    if previous is not None and previous[2:] == (float(command), bool(policy_eligible)):
        return previous
    return (1 if previous is None else previous[0] + 1,
            float(t), float(command), bool(policy_eligible))


class UnlatchedFinishObserver:
    def __init__(self, config: UnlatchedFinishConfig):
        self.config = config
        self.reset()

    def reset(self, reason="reset"):
        self.phase = "cold"
        self.reason = reason
        self.origin = None
        self.reference = None
        self.opening = None
        self.activity_since = None
        self.unloaded_since = None
        self.last_samples = None
        self.last_ack = None
        self.finished_at = None

    def in_volume(self, tcp):
        xyz = np.asarray(tcp, dtype=float)[:3]
        return bool(xyz.shape == (3,) and np.isfinite(xyz).all()
                    and np.all(xyz >= self.config.tcp_min_m)
                    and np.all(xyz <= self.config.tcp_max_m))

    def update(self, t, *, tcp, measured, loads, eligible, ack, samples,
               unloaded_force_max_n, unloaded_hold_s, finish_permitted):
        """ack is (generation, actual acceptance time, closure, policy_eligible).

        Callers acknowledge after I/O succeeds, not at mailbox submission. An
        unchanged accepted command may stay valid; it need not be resent. Each
        dwell advances only when every required feedback stream advances.
        """
        c = self.config
        if self.finished_at is not None:
            return True
        if not eligible or (ack is not None and not ack[3]):
            self.reset("no_accepted_original_policy")
            return False
        serial, accepted_t, command, _ = ack if ack is not None else (0, t, measured, False)
        if (not np.isfinite(t) or not np.isfinite(accepted_t) or accepted_t > t + 1e-9
                or not np.isfinite(command) or not 0 <= command <= 1
                or not np.isfinite(measured) or not 0 <= measured <= 1
                or not np.isfinite(np.asarray(tcp)).all()
                or len(loads) < 2
                or not all(np.isfinite(f) and f >= 0 for f in loads.values())):
            self.reset("invalid_feedback_or_ack")
            return False
        required = {"arm", "gripper"} | {f"tactile_{name}" for name in loads}
        if (samples is None or not required.issubset(samples)
                or len(required) < 4
                or any(samples[k] is None or not np.isfinite(samples[k])
                       or not -1e-9 <= t - samples[k] <= c.feedback_max_age_s
                       for k in required)):
            self.reset("stale_or_missing_feedback")
            return False
        current = {k: float(samples[k]) for k in required}
        if ack is None:
            # Snapshot a real open starting state before the first policy
            # command is accepted. A subsequent ACK and measured close are
            # still required. Repeated missing ACKs cannot create activity.
            if self.phase == "cold" and self.origin is None:
                self.origin = {"command": measured, "measured": measured, "t": min(current.values())}
            self.reason = "awaiting_accepted_command"
            return False
        if self.last_ack is not None and (serial < self.last_ack[0] or accepted_t < self.last_ack[1]):
            self.reset("ack_regressed")
            return False
        if self.reference is not None and t - self.reference["t"] > c.history_timeout_s:
            self.reset("grasp_history_expired")
        if self.opening is not None:
            if not self.in_volume(tcp):
                self.reset("left_finish_region")
                return False
            if command > self.reference["command"] - c.command_delta_min + c.reclose_tolerance:
                self.reset("accepted_command_left_open_band")
                return False
            if measured > self.reference["measured"] + c.reclose_tolerance:
                self.reset("measured_closed_past_grasp_reference")
                return False
        # A duplicate tactile frame must not turn a short pulse into a dwell.
        if self.last_samples is not None:
            if any(current[k] < self.last_samples.get(k, -np.inf) for k in current):
                self.reset("capture_time_regressed")
                return False
            if not all(current[k] > self.last_samples.get(k, -np.inf) for k in current):
                self.reason = "awaiting_distinct_feedback"
                return False
        self.last_samples = current
        self.last_ack = (serial, accepted_t)
        sample_t = min(current.values())
        # A sensor sample captured before the actual acknowledgement cannot
        # prove a response to that command. It also cannot start a new dwell.
        relevant_ack_t = self.opening["accepted_t"] if self.opening is not None else accepted_t
        if min(current.values()) < relevant_ack_t - 1e-9:
            self.reason = "awaiting_post_ack_feedback"
            return False
        if self.origin is None:
            self.origin = {"command": command, "measured": measured, "t": sample_t}
        loaded = all(f >= unloaded_force_max_n for f in loads.values())
        unloaded = all(f < unloaded_force_max_n for f in loads.values())
        if self.phase == "cold":
            closing = (command - self.origin["command"] >= c.command_delta_min
                       and measured - self.origin["measured"] >= c.measured_delta_min)
            if not (closing and loaded):
                self.activity_since = None
                # A fresh unloaded open command becomes the start of a new
                # close cycle. This never records a grasp by itself.
                if command < self.origin["command"] - c.reclose_tolerance:
                    self.origin = {"command": command, "measured": measured, "t": sample_t}
                self.reason = "awaiting_loaded_measured_close"
                return False
            if self.activity_since is None:
                self.activity_since = sample_t
            if sample_t - self.activity_since < c.activity_hold_s - 1e-9:
                self.reason = "grasp_activity_dwell"
                return False
            self.phase = "grasp_activity"
            self.reference = {"command": command, "measured": measured, "t": sample_t,
                              "ack_generation": serial, "accepted_t": accepted_t}
            self.reason = "grasp_activity_observed"
            return False
        if self.phase == "grasp_activity":
            ref = self.reference
            opening = (serial > ref["ack_generation"] and accepted_t > ref["accepted_t"]
                       and ref["command"] - command >= c.command_delta_min
                       and ref["measured"] - measured >= c.measured_delta_min)
            if opening and self.in_volume(tcp):
                self.phase = "opening_observed"
                self.opening = {"command": command, "measured": measured, "t": sample_t,
                                "accepted_t": accepted_t, "ack_generation": serial}
                self.reason = "accepted_measured_policy_opening"
            elif loaded and command >= ref["command"] and measured >= ref["measured"]:
                # Reference the latest loaded close plateau. Never move the
                # reference down with a relaxation, or up after contact ends.
                self.reference = {"command": command, "measured": measured, "t": sample_t,
                                  "ack_generation": serial, "accepted_t": accepted_t}
                self.reason = "loaded_close_reference"
            elif opening and not self.in_volume(tcp):
                self.reset("opening_outside_finish_region")
                return False
        if self.phase == "opening_observed":
            # The measured opening EVENT was already acknowledged above.
            # Loaded and unloaded finite-drive equilibria differ: retaining
            # that event does not require the same position-error offset after
            # load disappears. A new accepted close or movement beyond the old
            # grasp reference cancels it; fresh bilateral unload must persist.
            if not unloaded:
                if self.unloaded_since is not None:
                    self.reset("load_returned_after_unload")
                    return False
                self.unloaded_since = None
                self.reason = "awaiting_bilateral_unload"
                return False
            if self.unloaded_since is None:
                self.unloaded_since = sample_t
            if sample_t - self.unloaded_since >= unloaded_hold_s - 1e-9:
                if finish_permitted:
                    self.phase = "finished"
                    self.finished_at = float(t)
                    self.reason = "unlatched_release_finished"
                    return True
                self.reason = "finish_hold_not_permitted"
            else:
                self.reason = "unloaded_dwell"
        return False

    def diagnostics(self, tcp):
        return {"revision": "acknowledged_opening_event_v1", "phase": self.phase, "reason": self.reason,
                "in_finish_region": self.in_volume(tcp), "origin": self.origin,
                "grasp_reference": self.reference, "opening": self.opening,
                "activity_since_s": self.activity_since, "unloaded_since_s": self.unloaded_since,
                "last_capture_times": self.last_samples, "last_ack": self.last_ack,
                "finished_at_s": self.finished_at}
