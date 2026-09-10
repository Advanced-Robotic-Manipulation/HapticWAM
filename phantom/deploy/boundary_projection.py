"""Explicit, default-off upper-Y command projection shared by native and sim.

This is a bounded controller workaround, not a larger safety envelope. Physical
safety runs once in SafetyMonitor before selection. The final IK/FK target is
checked again immediately before submission, without repeating stateful safety.
No gripper observation or command is changed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math

import numpy as np


class BoundaryProjectionStop(RuntimeError):
    """Ordinary fail-closed controller stop; retain the accepted grip."""

    def __init__(self, reason):
        self.reason = "boundary_projection_" + reason
        super().__init__(self.reason)


@dataclass(frozen=True)
class BoundaryProjectionConfig:
    # Every enabled value must be explicitly declared in the treatment JSON.
    variant: str
    inward_margin_m: float
    maximum_excursion_m: float
    rearm_inward_m: float
    feedback_max_age_s: float
    solver_inset_m: float
    max_tick_s: float
    rearm_dwell_s: float
    budget_s: float

    def __post_init__(self):
        if self.variant != "upper_y_projection_v1":
            raise ValueError("unknown boundary projection variant")
        for f in fields(self):
            if f.name == "variant":
                continue
            v = getattr(self, f.name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float)):
                raise ValueError(f"{f.name} must be an explicit finite positive number")
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{f.name} must be finite and positive")
        if self.budget_s != 2.5:
            raise ValueError("upper_y_projection_v1 has a fixed 2.5 s budget")
        if self.rearm_inward_m <= self.inward_margin_m + self.solver_inset_m:
            raise ValueError("rearm plane must be strictly inward of projection plane")
        if self.feedback_max_age_s > 0.016:
            raise ValueError("boundary projection requires feedback no older than 16 ms")
        if self.max_tick_s > 0.016:
            raise ValueError("boundary projection supports at most 16 ms command steps")
        if self.maximum_excursion_m > 0.002:
            raise ValueError("upper-Y projection cannot allow excursions larger than 2 mm")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("boundary projection config must be an object")
        names = {f.name for f in fields(cls)}
        if set(value) != names:
            raise ValueError(f"boundary projection requires exactly {sorted(names)}")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def boundary_config(value):
    if value is None:
        return None
    return value if isinstance(value, BoundaryProjectionConfig) else BoundaryProjectionConfig.from_dict(value)


def _pose(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (6,) or not np.isfinite(a).all():
        raise BoundaryProjectionStop("invalid_pose")
    return a.copy()


class UpperYBoundaryProjection:
    def __init__(self, config, hw, rings):
        self.config = boundary_config(config)
        self.hw, self.rings = hw, rings
        self.hb = hw.safety.hitbox_m
        if self.hb is None:
            raise ValueError("boundary projection requires an explicit task hitbox")
        if hw.safety.servo_constraint_hold_s is None:
            raise ValueError("boundary projection requires the shared verified servo limiter")
        self.upper = float(self.hb.y[1])
        self.ceiling = self.upper - self.config.inward_margin_m
        self.plane = self.ceiling - self.config.solver_inset_m
        self.rearm = self.upper - self.config.rearm_inward_m
        if self.rearm <= self.hb.y[0]:
            raise ValueError("boundary projection rearm plane is outside task hitbox")
        self.started_at = None
        self.last_t = None
        self.anchor = None
        self.anchor_t = None
        self.anchor_q = None
        self.pending = None
        self.submitted = None
        self.submission_sequence = 0
        self.selected_at = None
        self.raw_interior = False
        self.measured_interior = False
        self.failure = None
        self.decision_clock = None  # Native executor supplies the causal wall clock.
        self.rearm_since = None
        self.rearm_last_ack = None
        self.feedback_t = None
        self.finish_request = None
        self.seal = None
        self.last = {}

    def _stop(self, reason):
        self.failure = reason
        self.last.update(reason=reason, stopped=True)
        raise BoundaryProjectionStop(reason)

    def _clock(self, t):
        if not math.isfinite(t) or (self.last_t is not None and t < self.last_t):
            self._stop("invalid_clock")
        self.last_t = float(t)
        if self.failure is not None:
            self._stop(self.failure)
        if self.seal is None and self.started_at is not None and t >= self.started_at + self.config.budget_s:
            self._stop("timeout")
        if (self.seal is None and self.started_at is not None and self.anchor_t is not None
                and t - self.anchor_t > self.config.max_tick_s + 1e-9):
            self._stop("tick_overrun")

    def _envelope(self, p, *, ceiling=None):
        sf = self.hw.safety
        return bool(
            sf.workspace_m.contains(p[:3]) and self.hb.contains(p[:3])
            and (sf.reach_clamp_m is None or np.linalg.norm(p[:3]) <= sf.reach_clamp_m)
            and (ceiling is None or p[1] <= ceiling)
        )

    def _feedback(self, t, sample=None):
        try:
            ts, arm = self.rings["arm"].latest(1) if sample is None else sample
            capture = float(ts[0])
            p = _pose(arm["tcp_pose"][0])
            q = _pose(arm["q"][0])
            qd = _pose(arm["qd"][0])
        except (KeyError, IndexError, TypeError, ValueError, BoundaryProjectionStop):
            self._stop("invalid_feedback")
        age = t - capture
        self.last.update(measured_tcp_pose=p.tolist(), measured_q=q.tolist(),
                         measured_qd=qd.tolist(), feedback_capture_s=capture, feedback_age_s=age)
        if not math.isfinite(capture) or age < -1e-9 or age > self.config.feedback_max_age_s:
            self._stop("stale_feedback")
        if self.feedback_t is not None and capture < self.feedback_t:
            self._stop("backwards_feedback")
        self.feedback_t = capture
        if not self._envelope(p):
            self._stop("measured_envelope")
        return p

    def select(self, t, raw, clamped, *, physical_preempted=False):
        """Called once per safety tick; only Y of the ordinary clamped pose changes."""
        self.last = {"variant": self.config.variant, "stopped": False,
                     "started_at_s": self.started_at,
                     "deadline_s": None if self.started_at is None else self.started_at + self.config.budget_s,
                     "projection_plane_y_m": self.plane, "final_ceiling_y_m": self.ceiling,
                     "raw_tcp_pose": np.asarray(raw).tolist(),
                     "clamped_tcp_pose": np.asarray(clamped).tolist(),
                     "selected_tcp_pose": None, "final_verified_tcp_pose": None,
                     "accepted_tcp_pose": None, "accepted_at_s": None,
                     "selected_target_kind": "continuation", "drive_submission": None}
        self.pending = None
        self.submitted = None
        self.selected_at = None
        if physical_preempted:
            self.last.update(reason="physical_safety_preempted")
            return clamped
        # Freeze feedback before reading the native decision clock. A sample
        # legitimately published during this tick must not look "future" just
        # because its capture happened after the executor's tick-start t0.
        sample = self.rings["arm"].latest(1)
        if self.decision_clock is not None:
            t = float(self.decision_clock())
        self._clock(t)
        try:
            raw, target = _pose(raw), _pose(clamped)
        except (ValueError, TypeError, BoundaryProjectionStop):
            self._stop("invalid_target")
        measured = self._feedback(t, sample)
        if self.seal is not None:
            # Continuation authority is permanently retired after verified
            # FINISH hold ACK. No replan or interior progress can unseal it.
            if not np.array_equal(raw, self.finish_request["tcp_pose"]):
                self._stop("sealed_target_change")
            self.selected_at = float(t)
            self.last.update(reason="sealed_terminal_hold", selected_tcp_pose=self.seal["tcp_pose"],
                             sealed_requested_tcp_pose=raw.tolist(),
                             selected_target_kind="sealed_terminal_hold",
                             projected=False, terminal_hold=dict(self.seal))
            return raw
        # Match the existing hitbox's ordinary reach/workspace-clamped domain,
        # including the intended floor clamp. No other remaining hitbox face
        # may be violated. Raw Y still bounds the permitted excursion so a
        # large Y command cannot be laundered through ordinary clamping.
        if not (self.hb.x[0] <= target[0] <= self.hb.x[1]
                and self.hb.y[0] <= target[1] <= self.upper + self.config.maximum_excursion_m
                and raw[1] <= self.upper + self.config.maximum_excursion_m
                and self.hb.z[0] <= target[2] <= self.hb.z[1]):
            self._stop("unsupported_excursion")
        projected = target[1] > self.plane
        if projected and raw[1] > self.upper and not (
            self.hb.x[0] <= raw[0] <= self.hb.x[1]
            and self.hb.z[0] <= raw[2] <= self.hb.z[1]
        ):
            self._stop("multiple_raw_faces")
        if projected:
            if (self.anchor is None or self.anchor_t is None
                    or not 0 <= t - self.anchor_t <= self.config.feedback_max_age_s
                    or not self._envelope(self.anchor, ceiling=self.ceiling)):
                self._stop("unverified_anchor")
            if self.started_at is None:
                self.started_at = float(t)
            target[1] = self.plane
        self.raw_interior = bool(np.array_equal(raw, clamped) and raw[1] <= self.rearm)
        self.measured_interior = bool(measured[1] <= self.rearm)
        if not self.raw_interior or not self.measured_interior:
            self.rearm_since = self.rearm_last_ack = None
        self.selected_at = float(t)
        self.last.update(selected_tcp_pose=target.tolist(), projected=projected,
                         reason="projected" if projected else "unmodified",
                         started_at_s=self.started_at,
                         deadline_s=None if self.started_at is None else self.started_at + self.config.budget_s,
                         rearm_raw_interior=self.raw_interior,
                         rearm_measured_interior=self.measured_interior)
        return target

    def request_finish(self, t, grip, grip_ack):
        """Called only on the existing controller's first authentic FINISH.

        Capture achieved joints for a stop, not another IK approximation.
        Sealing occurs only after final FK/rate verification and successful ACK
        before the original deadline. No future command or rearm is licensed.
        """
        if self.finish_request is not None or self.seal is not None:
            self._stop("repeated_finish_transition")
        sample = self.rings["arm"].latest(1)
        now = t if self.decision_clock is None else float(self.decision_clock())
        self._clock(now)
        measured = self._feedback(now, sample)
        try:
            seq, ack_t, ack_grip, _eligible = grip_ack
            valid_ack = (seq >= 1 and math.isfinite(ack_t) and ack_t <= t
                         and math.isfinite(grip) and grip == ack_grip)
        except (TypeError, ValueError):
            valid_ack = False
        if not valid_ack:
            self._stop("finish_without_grip_ack")
        self.finish_request = {"finish_at_s": float(t), "requested_at_s": float(now),
                               "tcp_pose": measured.tolist(),
                               "q": _pose(sample[1]["q"][0]).tolist(),
                               "gripper_command": float(grip), "gripper_ack": list(grip_ack)}
        self.last["terminal_hold_request"] = dict(self.finish_request)
        self.last.update(selected_target_kind="finish_transition", selected_tcp_pose=measured.tolist())
        return measured

    def terminal_joint_target(self):
        target = self.seal if self.seal is not None else self.finish_request
        return None if target is None else np.asarray(target["q"], dtype=float).copy()

    def check_grip(self, command):
        target = self.seal if self.seal is not None else self.finish_request
        if target is not None and command != target["gripper_command"]:
            self._stop("sealed_grip_change")

    def terminal_selection(self, previous_pose, qref, dt, limits):
        """Verified measured/stored joint hold; never re-solve IK after sealing."""
        from phantom.drivers.servo_limiter import ServoStep, feasible
        q = self.terminal_joint_target()
        if q is None:
            return None
        if not feasible(q, qref, dt, limits):
            self._stop("terminal_joint_constraint")
        return ServoStep(q, np.asarray(previous_pose).copy(), "terminal_hold", "terminal_hold", 0.0,
                         None, 0, True, True)

    def verify_final(self, t, pose, q, *, verified, dt, previous_pose):
        """Before driver submission; caller supplies actual final FK and verified IK."""
        sample = self.rings["arm"].latest(1)
        if self.decision_clock is not None:
            t = float(self.decision_clock())
        self._clock(t)
        if self.selected_at is None or not 0 <= t - self.selected_at <= self.config.feedback_max_age_s:
            self._stop("missing_current_selection")
        measured = self._feedback(t, sample)
        try:
            pose, q = _pose(pose), _pose(q)
        except (TypeError, ValueError, BoundaryProjectionStop):
            self._stop("invalid_final_target")
        self.last.update(final_verified_tcp_pose=pose.tolist(), final_verified_q=q.tolist(),
                         final_verification_at_s=float(t))
        if not verified:
            self._stop("unverified_final_ik")
        if self.seal is not None:
            if not (np.array_equal(q, self.seal["q"]) and np.array_equal(pose, self.seal["tcp_pose"])):
                self._stop("sealed_final_change")
        elif self.finish_request is not None:
            requested = np.asarray(self.finish_request["tcp_pose"])
            if (not np.array_equal(q, self.finish_request["q"])
                    or np.linalg.norm(pose[:3] - requested[:3]) > self.config.solver_inset_m):
                self._stop("finish_not_achieved_hold")
        if not math.isfinite(dt) or not 0 < dt <= self.config.max_tick_s:
            self._stop("invalid_command_dt")
        # Solver shortening/sliding and numerical FK error may change the
        # final step. Check the real submitted displacement, too.
        from phantom.data.derived import rotvec_nearest
        previous = _pose(previous_pose)
        self.last.update(final_verification_previous_tcp_pose=previous.tolist(), final_verification_dt_s=float(dt))
        if self.anchor is not None and not np.array_equal(previous, self.anchor):
            self._stop("anchor_mismatch")
        if not self._envelope(previous):
            self._stop("unsafe_final_anchor")
        linear = float(np.linalg.norm(pose[:3] - previous[:3]))
        angular = float(np.linalg.norm(rotvec_nearest(previous[3:], pose[3:]) - previous[3:]))
        if (linear > self.hw.arm.limits.tcp_speed_m_s * dt + 1e-9
                or angular > self.hw.arm.limits.joint_speed_rad_s * dt + 1e-9):
            self._stop("final_rate")
        if not self._envelope(pose, ceiling=self.ceiling):
            self._stop("final_envelope")
        self.measured_interior = bool(measured[1] <= self.rearm)
        self.pending = (float(t), pose.copy(), q.copy())
        return pose

    def note_submission(self, t, pose, q, *, mechanism):
        """A receipt recorded only after the real submission API returns success."""
        pose, q = _pose(pose), _pose(q)
        if (self.pending is None or not np.array_equal(pose, self.pending[1])
                or not np.array_equal(q, self.pending[2]) or t < self.pending[0]):
            self._stop("unverified_submission")
        self.submission_sequence += 1
        self.submitted = {"sequence": self.submission_sequence, "submitted_at_s": float(t),
                          "tcp_pose": pose.tolist(), "q": q.tolist(), "mechanism": mechanism}
        self.last["drive_submission"] = dict(self.submitted)

    def acknowledge(self, t, pose):
        """Only after actual servo acceptance; rejected commands cannot seed recovery."""
        pose = _pose(pose)
        if self.pending is None or not np.array_equal(pose, self.pending[1]):
            self._stop("unverified_ack")
        vt, _, q = self.pending
        if self.submitted is None or not self.submitted["submitted_at_s"] <= t:
            self._stop("ack_without_submission")
        self.last.update(accepted_tcp_pose=pose.tolist(), accepted_q=q.tolist(), accepted_at_s=float(t))
        if not 0 <= t - vt <= self.config.feedback_max_age_s:
            self._stop("stale_ack")
        # Compare against the PRIOR accepted command. Replacing anchor_t first
        # would hide a late native ACK behind a zero-length new interval.
        self._clock(t)
        self.anchor, self.anchor_q, self.anchor_t = pose.copy(), q.copy(), float(t)
        self.pending = None
        if self.finish_request is not None and self.seal is None:
            self.seal = {**self.finish_request, "tcp_pose": pose.tolist(), "q": q.tolist(),
                         "verified_at_s": float(vt), "accepted_at_s": float(t), "sealed_at_s": float(t),
                         "submission_sequence": self.submitted["sequence"],
                         "continuation_deadline_s": None if self.started_at is None else self.started_at + self.config.budget_s,
                         "authority": "irrevocable_terminal_hold_only"}
            self.last.update(reason="terminal_hold_sealed", terminal_hold=dict(self.seal))
            self.rearm_since = self.rearm_last_ack = None
            return
        if self.seal is not None:
            self.last["terminal_hold"] = dict(self.seal)
            return
        # A replan, commanded tangent motion or a shorter IK step never rearms.
        # Timeout has already won in verify_final. Both raw and measured state
        # must be inward, and final FK must actually reach the same interior.
        if self.started_at is not None and self.raw_interior and self.measured_interior and pose[1] <= self.rearm:
            if (self.rearm_since is None or self.rearm_last_ack is None
                    or t - self.rearm_last_ack > self.config.max_tick_s + 1e-9):
                self.rearm_since = float(t)
            self.rearm_last_ack = float(t)
            if t - self.rearm_since >= self.config.rearm_dwell_s:
                self.started_at = None
                self.rearm_since = None
                self.last.update(rearmed=True, reason="verified_interior_rearm")
        else:
            self.rearm_since = self.rearm_last_ack = None
        self.last["rearm_since_s"] = self.rearm_since


def make_boundary_projection(config, hw, rings):
    return None if config is None else UpperYBoundaryProjection(config, hw, rings)
