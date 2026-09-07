#!/usr/bin/env python3
"""Read-only, single-step checks of the existing native limiter on saved commands.

No simulator, learned model, RTDE connection or hypothetical closed-loop outcome.
Each check reanchors to the original preceding command; it is not a replay of
the altered path. The native algorithm is called unchanged with nominal UR3 IK.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v3"
A2, A3, D4 = 0.24365, 0.21325, 0.11235
ELBOW_MIN = 0.40
V_MAX = 1.0
STOP_RADIUS = 0.468


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def radius(q):
    return float(np.sqrt(A2 * A2 + A3 * A3 + 2 * A2 * A3 * np.cos(q[2]) + D4 * D4))


def radial_velocity(q, qd):
    return float(-A2 * A3 * np.sin(q[2]) * qd[2] / radius(q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.nice(19)
    sys.path.insert(0, str(args.source))
    import phantom

    phantom.__path__.append(str(BASE / "source_teacher_v2_delivery/phantom"))
    from phantom.drivers.real.ur import URArm
    from phantom.sim.kinematics import forward_pose, inverse_kinematics

    class NominalCtrl:
        """Only a CPU IK callback; no real driver construction or connect."""

        def __init__(self):
            self.calls = 0

        def getInverseKinematics(self, pose, qnear=None):
            self.calls += 1
            result = inverse_kinematics(pose, qnear, max_joint_delta_rad=0.35)
            return result.q.tolist() if result.success else []

    arm = object.__new__(URArm)
    arm.hw = SimpleNamespace(
        safety=SimpleNamespace(
            elbow_min_rad=ELBOW_MIN,
            servo_joint_speed_max_rad_s=V_MAX,
            ur_dh_d1_m=0.1519,
        )
    )
    nominal = NominalCtrl()
    definitions = [
        (
            "baseline_failed_carry",
            ROOT / "components/baseline/rollouts/teacher__fixed_anchor__seed904301",
        ),
        (
            "bridge_failed_carry",
            ROOT
            / "corrected_profile_bridge/rollouts/teacher__fixed_anchor__seed904301",
        ),
        (
            "minimal_failed_carry",
            ROOT
            / "minimal_profile_diagnostic/rollouts/teacher__fixed_anchor__seed904301",
        ),
        (
            "gel_only_placement",
            ROOT / "components/gel_v2_only/rollouts/teacher__fixed_anchor__seed904301",
        ),
        (
            "original_placement",
            BASE
            / "runs/teacher_pick_place_v1/campaign/rollouts/teacher__placement_xm10_ym10mm__seed4242",
        ),
    ]
    output = {
        "scope": "Existing native limiter, independent single-step counterfactuals on immutable commands; no altered trajectory or task-success prediction.",
        "method": "Call unchanged URArm._limited_step with nominal full-six-joint UR3 IK. Anchor each sample at the prior actual target_q and its nominal FK. Driver hardware IK calibration and real-time cost are not reproduced.",
        "candidate_existing_parameters": {
            "elbow_min_rad": ELBOW_MIN,
            "servo_joint_speed_max_rad_s": V_MAX,
            "measured_wrist_extension_stop_m_unchanged": STOP_RADIUS,
            "command_radius_at_min_elbow_m": radius([0, 0, ELBOW_MIN]),
            "margin_to_measured_stop_m": STOP_RADIUS - radius([0, 0, ELBOW_MIN]),
            "native_bisection_iterations": arm.LIMITER_BISECT,
            "provenance": "Existing historical optional hardware configuration; not optimized using these outcomes.",
        },
        "source_sha256": {
            p: sha(
                (args.source / p)
                if (args.source / p).exists()
                else (BASE / "source_teacher_v2_delivery" / p)
            )
            for p in (
                "phantom/drivers/real/ur.py",
                "phantom/sim/kinematics.py",
                "phantom/config/hardware.py",
            )
        },
        "cases": [],
    }
    for label, folder in definitions:
        execution = [
            json.loads(s)
            for s in (folder / "execution_trace.jsonl").read_text().splitlines()
            if s
        ]
        stop = next((r for r in execution if r["stopped"]), None)
        live = [r for r in execution if stop is None or r["t"] <= stop["t"]]
        commands = [
            r for r in live if not r["stopped"] and r.get("target_q") is not None
        ]
        first_inside_margin = next(
            (r for r in commands if abs(r["target_q"][2]) < ELBOW_MIN), None
        )
        first_outside_stop = next(
            (r for r in commands if radius(r["target_q"]) > STOP_RADIUS), None
        )
        samples = []
        previous = None
        # At most 41 rows: a decimated two-second neighborhood after first
        # margin entry, plus exact first margin/stop crossings and last command.
        first_t = None if first_inside_margin is None else first_inside_margin["t"]
        selected_times = {
            r["t"]
            for r in (first_inside_margin, first_outside_stop, commands[-1])
            if r is not None
        }
        if first_t is not None:
            window = [r for r in commands if first_t <= r["t"] <= first_t + 2.0]
            selected_times.update(r["t"] for r in window[::8])
        for row in commands:
            if previous is None:
                previous = row
                continue
            if row["t"] not in selected_times:
                previous = row
                continue
            qref = np.asarray(previous["target_q"], dtype=float)
            q = np.asarray(row["target_q"], dtype=float)
            dt = float(row["t"] - previous["t"])
            prev = forward_pose(qref)
            target = forward_pose(q)
            violation = arm._limit_violation(q.tolist(), qref.tolist(), dt)
            calls_before = nominal.calls
            if arm._feasible(q.tolist(), qref.tolist(), dt):
                result = (target, q.tolist(), 1.0, "unchanged")
            else:
                result = arm._limited_step(nominal, prev, target, qref.tolist(), dt)
            record = {
                "t_s": row["t"],
                "dt_s": dt,
                "actual_target_q": q.tolist(),
                "actual_measured_q": row["measured_q"],
                "actual_measured_qd": row["measured_qd"],
                "actual_target_radius_m": radius(q),
                "actual_measured_radius_m": radius(row["measured_q"]),
                "actual_measured_outward_radial_velocity_m_s": radial_velocity(
                    row["measured_q"], row["measured_qd"]
                ),
                "actual_target_joint_speed_max_rad_s": float(
                    np.max(np.abs(q - qref)) / dt
                ),
                "actual_target_max_raw_branch_delta_rad": float(
                    np.max(np.abs(q - qref))
                ),
                "violation": violation,
                "hypothetical_mode": "hold" if result is None else result[3],
                "extra_nominal_ik_solves": nominal.calls - calls_before,
            }
            if result is not None:
                candidate_pose, candidate_q, fraction, _ = result
                cq = np.asarray(candidate_q)
                feasible = arm._feasible(candidate_q, qref.tolist(), dt)
                velocity = float(np.max(np.abs(cq - qref)) / dt)
                radius_ok = radius(cq) <= radius([0, 0, ELBOW_MIN]) + 1e-12
                escape_ok = radius(cq) < radius(qref) - 1e-12
                fk = forward_pose(cq)
                record.update(
                    hypothetical_fraction=float(fraction),
                    hypothetical_q=cq.tolist(),
                    hypothetical_radius_m=radius(cq),
                    hypothetical_joint_speed_max_rad_s=velocity,
                    hypothetical_feasible=feasible,
                    hypothetical_inside_margin_or_inward_escape=radius_ok or escape_ok,
                    hypothetical_pose_vs_fk_translation_error_m=float(
                        np.linalg.norm(np.asarray(candidate_pose)[:3] - fk[:3])
                    ),
                )
                assert (
                    feasible and velocity <= V_MAX + 1e-10 and (radius_ok or escape_ok)
                ), record
            samples.append(record)
            previous = row
        tracking = np.asarray(
            [radius(r["measured_q"]) - radius(r["target_q"]) for r in commands]
        )
        q = np.asarray([r["target_q"] for r in commands])
        actual_max_command_speed = float(
            np.max(
                np.abs(np.diff(q, axis=0))
                / np.diff([r["t"] for r in commands])[:, None]
            )
        )
        c = {
            "label": label,
            "case_directory": str(folder),
            "execution_sha256": sha(folder / "execution_trace.jsonl"),
            "command_rows_before_stop": len(commands),
            "first_command_entering_existing_margin_s": first_t,
            "first_command_exceeding_unchanged_stop_radius_s": None
            if first_outside_stop is None
            else first_outside_stop["t"],
            "first_safety_stop_s": None if stop is None else stop["t"],
            "first_safety_events": []
            if stop is None
            else stop["diagnostics"]["safety_events"],
            "lead_from_margin_to_actual_stop_s": None
            if first_t is None or stop is None
            else stop["t"] - first_t,
            "measured_minus_target_radius_m_min_max": [
                float(tracking.min()),
                float(tracking.max()),
            ],
            "maximum_actual_command_joint_speed_rad_s": actual_max_command_speed,
            "actual_raw_elbow_rad_min_max": [
                float(q[:, 2].min()),
                float(q[:, 2].max()),
            ],
            "single_step_checks": samples,
        }
        output["cases"].append(c)
        print(
            json.dumps(
                {
                    "case": label,
                    "margin_entry_s": first_t,
                    "stop_s": c["first_safety_stop_s"],
                    "single_step_checks": len(samples),
                }
            ),
            flush=True,
        )
    output["nominal_ik_calls"] = nominal.calls
    output["limits"] = [
        "No hypothetical force, pose tracking, policy observation, later action or task success was generated.",
        "A command envelope does not bound physical tracking overshoot or stopping distance. Measured safety remains active.",
        "Native abs(elbow) is equivalent to the wrist-radius test only on the intended principal elbow branch; raw multi-turn offsets need an explicit shared periodic-geometry predicate before generalization.",
        "The shoulder-to-TCP slide is a search direction, not a wrist-sphere projection. Every returned IK candidate must still pass elbow, branch and speed checks.",
        "One-step samples reanchor to the old commanded trajectory even after the first hypothetical intervention. This is a feasibility check, not an alternate rollout.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
