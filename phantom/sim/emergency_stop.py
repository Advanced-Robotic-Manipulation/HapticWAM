"""Measured-joint emergency submission evidence, independent of task FINISH.

Successful motor targets are the runner's existing measured stop and recovery
gripper targets. This is an API submission receipt, never proof of braking or
physical achievement. Hard-envelope violations are recorded, never corrected
by commanding another pose after overload. Invalid feedback/submission aborts
the simulator before further stepping; no real hardware driver is involved.
"""

from __future__ import annotations

from copy import deepcopy
import json
import time

import numpy as np

from phantom.sim.kinematics import forward_pose


class EmergencySubmissionFault(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(
            "emergency_stop_"
            + receipt.get("fault_reason", receipt["submission_result"])
        )


def json_finite(value):
    """Retain malformed feedback in fault evidence without JSON NaN tokens."""
    if isinstance(value, dict):
        return {str(k): json_finite(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_finite(value.tolist())
    if isinstance(value, (tuple, list)):
        return [json_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return (
        value if value is None or isinstance(value, (str, int, bool)) else repr(value)
    )


def hard_violations(
    hw, measured_tcp, measured_q, measured_qd, submitted_fk, submitted_q
):
    """Same hard bounds as the original monitor, plus submitted FK envelope.

    The original inward continuation reserve is intentionally absent here and
    remains evaluated by the unchanged original BoundaryPhysicsAudit.
    """
    sf, found = hw.safety, []
    for scope, pose, joints in (
        ("measured", measured_tcp, measured_q),
        ("submitted", submitted_fk, submitted_q),
    ):
        for name, box in (("hitbox", sf.hitbox_m), ("workspace", sf.workspace_m)):
            if box is None:
                continue
            for i, axis in enumerate("xyz"):
                low, high = getattr(box, axis)
                if pose[i] < low:
                    found.append(f"{scope}_{name}_{axis}_low")
                if pose[i] > high:
                    found.append(f"{scope}_{name}_{axis}_high")
        if sf.reach_clamp_m is not None and np.linalg.norm(pose[:3]) > sf.reach_clamp_m:
            found.append(f"{scope}_reach")
        a2, a3, d4 = sf.ur_dh_a2_a3_d4_m
        wrist = np.sqrt(a2 * a2 + a3 * a3 + 2 * a2 * a3 * np.cos(joints[2]) + d4 * d4)
        if sf.wrist_extension_stop_m is not None and wrist > sf.wrist_extension_stop_m:
            found.append(f"{scope}_wrist_extension")
    if np.max(np.abs(measured_qd)) > sf.joint_speed_stop_rad_s:
        found.append("measured_joint_speed")
    return found


class MeasuredStopAuthority:
    """One fixed achieved-joint emergency target with checked tail submissions."""

    def __init__(
        self,
        *,
        hw,
        stop_t,
        stop_reason,
        safety_events,
        physics_dt,
        feedback_max_age_s,
        emit,
        clock=time.monotonic,
    ):
        if not (
            np.isfinite(stop_t)
            and np.isfinite(physics_dt)
            and physics_dt > 0
            and np.isfinite(feedback_max_age_s)
            and feedback_max_age_s > 0
        ):
            raise ValueError("finite positive emergency timing limits required")
        self.hw, self.emit, self.clock = hw, emit, clock
        self.stop_t, self.stop_reason = float(stop_t), str(stop_reason)
        self.safety_events = deepcopy(safety_events)
        self.dt, self.max_age = float(physics_dt), float(feedback_max_age_s)
        self.sequence, self.target, self.previous_capture = 0, None, None
        self.initial_gripper_command = self.initial_arm_indices = None
        self.failed = False

    def submit(
        self,
        *,
        step,
        t,
        capture_t,
        q,
        qd,
        tcp,
        candidate,
        arm_ids,
        gripper_command,
        apply,
        wrist_capture_t=None,
        tactile_capture_t=None,
    ):
        if self.failed:
            raise RuntimeError("emergency authority cannot resume after a fault")
        self.sequence += 1
        initial = self.target is None
        receipt = {
            "schema": "measured_emergency_stop_receipt_v1",
            "authority": "measured_emergency_stop",
            "sequence": self.sequence,
            "step": step,
            "t_s": t,
            "phase": "stop_transition" if initial else "stop_hold",
            "stop_t_s": self.stop_t,
            "original_stop_reason": self.stop_reason,
            "original_safety_events": self.safety_events,
            "measurement_source": "synchronous_simulator_articulation_readback",
            "capture_t_s": capture_t,
            "feedback_max_age_s": self.max_age,
            "feedback_age_s": None,
            "physics_dt_s": self.dt,
            "measured_q": q,
            "measured_qd": qd,
            "measured_tcp": tcp,
            "candidate_joint_positions": candidate,
            "arm_joint_indices": arm_ids,
            "gripper_command": gripper_command,
            "wrist_capture_t_s": wrist_capture_t,
            "tactile_capture_t_s": tactile_capture_t,
            "tactile_tail_scope": "last actual sample only; no inferred unload",
            "apply_called": False,
            "submission_accepted": False,
            "submission_result": "not_attempted",
            "hard_violations": [],
            "model_finish_authority": False,
            "boundary_ack_created": False,
            "physical_achievement_claimed": False,
        }

        def fault(reason, error=None):
            self.failed = True
            receipt["submission_result"] = reason
            if error is not None:
                receipt.update(error_type=type(error).__name__, error=str(error))
            try:
                self.emit(json_finite(receipt))
            except BaseException as persistence_error:
                receipt.update(
                    receipt_persistence_error_type=type(persistence_error).__name__,
                    receipt_persistence_error=str(persistence_error),
                )
                error = persistence_error
            raise EmergencySubmissionFault(json_finite(receipt)) from error

        try:
            q, qd, tcp, target = [
                np.asarray(v, dtype=float).copy() for v in (q, qd, tcp, candidate)
            ]
            t, capture_t, gripper_command = (
                float(t),
                float(capture_t),
                float(gripper_command),
            )
            indices = np.asarray(arm_ids)
            if (
                type(step) is bool
                or not isinstance(step, (int, np.integer))
                or step < 0
                or indices.shape != (6,)
                or indices.dtype.kind not in "iu"
                or len(set(indices.tolist())) != 6
                or target.ndim != 1
                or np.any(indices < 0)
                or np.any(indices >= len(target))
            ):
                raise ValueError(
                    "six unique in-range integer arm indices and integral step required"
                )
            arm_ids = indices.astype(int)
        except (ValueError, TypeError, IndexError, OverflowError) as error:
            fault("invalid_feedback_or_target", error)
        receipt.update(
            step=int(step),
            t_s=t,
            capture_t_s=capture_t,
            feedback_age_s=t - capture_t,
            measured_q=q,
            measured_qd=qd,
            measured_tcp=tcp,
            candidate_joint_positions=target,
            arm_joint_indices=arm_ids,
            gripper_command=gripper_command,
        )
        if (
            any(v.shape != (6,) or not np.isfinite(v).all() for v in (q, qd, tcp))
            or target.ndim != 1
            or not np.isfinite(target).all()
            or not np.isfinite(gripper_command)
            or not 0 <= gripper_command <= self.hw.gripper.max_close_cmd
        ):
            fault("invalid_feedback_or_target")
        if (
            not np.isfinite(t)
            or not np.isfinite(capture_t)
            or not -1e-9 <= t - capture_t <= self.max_age
            or (self.previous_capture is not None and capture_t < self.previous_capture)
        ):
            fault("stale_or_backwards_feedback")
        if initial and (
            abs(t - self.stop_t) > 1e-9 or not np.array_equal(target[arm_ids], q)
        ):
            fault("stop_target_is_not_achieved_q")
        if not initial and not np.array_equal(target, self.target):
            fault("stop_hold_target_changed")
        if not initial and (
            gripper_command != self.initial_gripper_command
            or not np.array_equal(arm_ids, self.initial_arm_indices)
        ):
            fault("stop_hold_command_identity_changed")
        try:
            fk = forward_pose(target[arm_ids])
            if np.asarray(fk).shape != (6,) or not np.isfinite(fk).all():
                raise ValueError("invalid independently computed submitted FK")
            receipt["submitted_fk_tcp"] = fk
            receipt["hard_violations"] = hard_violations(
                self.hw, tcp, q, qd, fk, target[arm_ids]
            )
        except (ValueError, TypeError, IndexError, OverflowError) as error:
            fault("invalid_kinematic_or_hard_envelope", error)
        # A hard violation never prevents issuing the measured halt. It remains
        # a failed physical safety result in the independent prospective audit.
        receipt.update(apply_called=True, apply_started_monotonic_s=self.clock())
        try:
            result = apply(target.copy())
        except BaseException as error:  # pause/abort includes interrupted API calls
            receipt["apply_finished_monotonic_s"] = self.clock()
            fault("apply_exception", error)
        receipt["apply_finished_monotonic_s"] = self.clock()
        receipt["api_return_type"] = type(result).__name__
        receipt["api_return_value"] = json_finite(result)
        if result is False:
            fault("apply_explicit_false")
        # Match Isaac's None-returning setter and the normal verified callback:
        # absence of an explicit rejection confirms API submission, not motion.
        receipt.update(
            submission_accepted=True, submission_result="api_returned_without_rejection"
        )
        clean = json_finite(receipt)
        try:
            self.emit(clean)
        except BaseException as error:
            # The actual API call succeeded, but no reliable evidence/adapter
            # commit exists. Preserve that truth and stop before another step.
            self.failed = True
            clean.update(
                fault_reason="receipt_persistence_failed",
                receipt_persistence_error_type=type(error).__name__,
                receipt_persistence_error=str(error),
            )
            raise EmergencySubmissionFault(clean) from error
        self.target = target.copy()
        self.initial_gripper_command = gripper_command
        self.initial_arm_indices = arm_ids.copy()
        self.previous_capture = float(capture_t)
        return clean


def write_receipt(stream, receipt):
    stream.write(json.dumps(receipt, allow_nan=False) + "\n")
    stream.flush()


def persist_submission_fault(*, error, audit, row, pause):
    """Persist unaccepted status and always pause before unwinding the runner."""
    status = (
        "emergency_stop_evidence_failed"
        if error.receipt.get("submission_accepted")
        else "emergency_stop_submission_failed"
    )
    failed = dict(
        row,
        status=status,
        accepted_tcp=None,
        ik_success=None,
        ik_reason=status,
        emergency_stop=error.receipt,
    )
    try:
        audit.executed(failed)
        audit.execution.flush()
    finally:
        pause()
