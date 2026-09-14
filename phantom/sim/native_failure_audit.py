"""Read-only, bounded evidence for native articulation validity failures.

This module imports no Isaac runtime and never changes simulation properties.
Unavailable diagnostic getters are recorded without masking the original gate.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return json_value(value.item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def read_diagnostic(getter):
    try:
        return {"available": True, "value": json_value(getter())}
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


class NativeFailureHistory:
    """Retain at most 0.1 s / 4096 completed-physics-step samples."""

    def __init__(self, window_s=0.1, max_samples=4096):
        self.window_s = float(window_s)
        self.rows = deque(maxlen=max_samples)

    def observe(self, phase, t, names, q, qd, drive_references, *, velocity_read_error=None):
        t = float(t)
        if self.rows and (self.rows[-1]["phase"] != phase or t < self.rows[-1]["t_s"]):
            self.rows.clear()
        row = {"phase": phase, "t_s": t,
               "finger_q_rad": json_value(q), "finger_qd_rad_s": json_value(qd),
               "desired_drive_references_rad": dict(zip(names, json_value(drive_references)))}
        if velocity_read_error is not None:
            row["finger_qd_read_error"] = str(velocity_read_error)
        self.rows.append(row)
        while self.rows and self.rows[0]["t_s"] < t - self.window_s - 1e-12:
            self.rows.popleft()

    def report(self):
        return {"requested_window_s": self.window_s, "max_samples": self.rows.maxlen,
                "sampling": "Each mechanics check, before submitting the next command; targets are the retained references that produced the current state",
                "samples": list(self.rows)}


def _prim_attributes(prim, prefixes):
    return {attr.GetName(): {"value": json_value(attr.Get()),
                            "authored": bool(attr.HasAuthoredValueOpinion())}
            for attr in prim.GetAttributes() if attr.GetName().startswith(prefixes)}


def runtime_readback(stage, articulation_root, joint_paths, robot, names, indices):
    """Distinguish USD authored/resolved attributes from live tensor getters."""
    joints = {}
    for name in names:
        prim = stage.GetPrimAtPath(str(joint_paths[name]))
        joints[name] = {"path": str(prim.GetPath()),
                        "applied_schemas": list(prim.GetAppliedSchemas()),
                        "attributes": _prim_attributes(prim, ("physics:", "drive:", "physxJoint:", "physxMimicJoint:", "newton:mimic")),
                        "relationships": {rel.GetName(): [str(p) for p in rel.GetTargets()]
                                          for rel in prim.GetRelationships()
                                          if rel.GetName().startswith(("physics:", "physxMimicJoint:", "newton:mimic"))}}

    def properties():
        values = robot.dof_properties
        return {name: {field: values[index][field] for field in values.dtype.names}
                for name, index in zip(names, indices)}

    def applied_action():
        action = robot.get_applied_action()
        return {key: getattr(action, key, None) for key in
                ("joint_indices", "joint_positions", "joint_velocities", "joint_efforts")}

    def armatures():
        # Read the solver's additive joint inertia, not the authored USD value.
        # A missing/invalid tensor getter stays unavailable; never substitute
        # zero or a configured armature and label it a native measurement.
        values = robot._articulation_view._physics_view.get_dof_armatures()
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[0] != 1:
            raise RuntimeError("Expected one articulation in native armature readback")
        if len(names) != len(indices):
            raise RuntimeError("Armature joint names and indices must match")
        selected = values[0, indices]
        if not np.isfinite(selected).all() or (selected < 0).any():
            raise RuntimeError("Native armature must be finite and nonnegative")
        return dict(zip(names, selected))

    return {
        "read_only": True,
        "units": "USD angular positions/limits/velocities use degrees; USD angular gains are per degree. Live DOF positions/velocities/gains use radians. Force drive limits are N m. USD and live revolute armature use kg m^2 directly.",
        "usd_articulation": _prim_attributes(stage.GetPrimAtPath(articulation_root), ("physxArticulation:",)),
        "usd_physics_scenes": {str(p.GetPath()): _prim_attributes(p, ("physxScene:", "physics:"))
                               for p in stage.Traverse() if p.GetTypeName() == "PhysicsScene"},
        "joints": joints,
        "live_readbacks": {
            "position_iterations": read_diagnostic(lambda: robot.get_solver_position_iteration_count()),
            "velocity_iterations": read_diagnostic(lambda: robot.get_solver_velocity_iteration_count()),
            "dof_properties": read_diagnostic(properties),
            "armatures_kg_m2": read_diagnostic(armatures),
            "controller_gains": read_diagnostic(lambda: robot.get_articulation_controller().get_gains()),
            "applied_action_all_dofs": read_diagnostic(applied_action),
            "measured_joint_efforts_nm": read_diagnostic(lambda: robot.get_measured_joint_efforts(joint_indices=indices)),
            "applied_joint_efforts_nm": read_diagnostic(lambda: robot.get_applied_joint_efforts(joint_indices=indices)),
        },
        "effort_caveat": "Measured projected joint efforts include joint reactions and are not assumed to equal the actuator drive torque; applied efforts report explicit efforts, not necessarily implicit PD forces.",
    }
