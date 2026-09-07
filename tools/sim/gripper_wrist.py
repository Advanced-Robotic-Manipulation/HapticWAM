"""Explicit idealized wrist input from three gripper bodies' external contacts.

The reused contact reader exposes signed PhysX normal impulses and exact world
contact points. This wrapper, unlike its diagnostic-only reader, supplies a
declared policy/safety input. It does not estimate a calibrated UR3 wrench.
"""

from __future__ import annotations

import numpy as np

from tools.sim.robot_environment_contacts import (
    ENVIRONMENT_PATHS,
    RobotEnvironmentContactViews,
    _validate_paths,
)


def gripper_actor_paths(housing_path, pad_paths):
    paths, _ = _validate_paths([housing_path, *pad_paths], ENVIRONMENT_PATHS)
    if len(paths) != 3 or {p.rsplit("/", 1)[-1] for p in paths} != {
        "gripper_housing",
        "left_pad",
        "right_pad",
    }:
        raise ValueError("wrist proxy requires exactly housing, left_pad and right_pad")
    if not paths[0].endswith("/gripper_housing"):
        raise ValueError("first wrist actor must be gripper_housing")
    return paths


def aggregate_gripper_wrench(report, actor_paths, tcp_pose, recorded_bias):
    """Sum signed F and (world contact point - measured TCP) cross F once.

    Proximal actors in a larger diagnostic report are deliberately excluded.
    Duplicate selected actors/contact indices raise instead of double counting.
    Forces and moments stay in world/base axes, matching the old proxy's axis
    convention. The recorded six-channel baseline is added unchanged.
    """
    actors = gripper_actor_paths(actor_paths[0], actor_paths[1:])
    tcp = np.asarray(tcp_pose, dtype=float)
    bias = np.asarray(recorded_bias, dtype=float)
    if (
        tcp.shape != (6,)
        or bias.shape != (6,)
        or not np.isfinite(np.r_[tcp, bias]).all()
    ):
        raise ValueError("finite TCP pose and recorded wrist bias six-vectors required")
    seen_actors, rows = set(), []
    total = np.zeros(6)
    for actor in report["per_actor"]:
        path = actor["actor_path"]
        if path not in actors:
            continue
        if path in seen_actors:
            raise RuntimeError("duplicate gripper actor would double-count contact")
        seen_actors.add(path)
        wrench, seen_indices = np.zeros(6), set()
        for contact in actor["contacts"]:
            if (
                contact["actor_path"] != path
                or contact["filter_path"] not in ENVIRONMENT_PATHS
            ):
                raise RuntimeError(
                    "wrist contact must identify an external environment body"
                )
            ids = list(contact["populated_buffer_indices"])
            if len(ids) != len(set(ids)) or seen_indices.intersection(ids):
                raise RuntimeError(
                    "duplicate gripper contact indices would double-count load"
                )
            seen_indices.update(ids)
            scalar = np.asarray(contact["normal_force_signed_n"], dtype=float)
            point = np.asarray(contact["points_world_m"], dtype=float)
            normal = np.asarray(contact["normals_world"], dtype=float)
            if (
                scalar.shape != (len(ids),)
                or point.shape != (len(ids), 3)
                or normal.shape != (len(ids), 3)
                or not np.isfinite(np.r_[scalar, point.ravel(), normal.ravel()]).all()
            ):
                raise RuntimeError("invalid populated gripper wrist contact data")
            force = scalar[:, None] * normal
            wrench[:3] += force.sum(axis=0)
            wrench[3:] += np.cross(point - tcp[:3], force).sum(axis=0)
        total += wrench
        rows.append(
            {
                "actor_path": path,
                "normal_wrench_world": wrench.tolist(),
                "contacts": actor["contacts"],
            }
        )
    if seen_actors != set(actors):
        raise RuntimeError("missing gripper actor contact readout")
    value = bias + total
    if not np.isfinite(value).all():
        raise RuntimeError("nonfinite gripper wrist wrench")
    return value, {
        "tcp_pose": tcp.tolist(),
        "normal_wrench_world": total.tolist(),
        "recorded_bias": bias.tolist(),
        "wrist_ft": value.tolist(),
        "per_actor": rows,
    }


class GripperContactWrist:
    """Construct before World.reset; initialize after reset; sample control-rate."""

    def __init__(self, housing_path, pad_paths, *, rigid_prim_cls=None):
        self.actor_paths = gripper_actor_paths(housing_path, pad_paths)
        self.reader = RobotEnvironmentContactViews(
            self.actor_paths, rigid_prim_cls=rigid_prim_cls
        )

    def initialize(self):
        self.reader.initialize()

    def sample(self, tcp_pose, recorded_bias, physics_dt):
        report = self.reader.get_all(physics_dt)
        value, record = aggregate_gripper_wrench(
            report, self.actor_paths, tcp_pose, recorded_bias
        )
        record.update(physics_dt_s=float(physics_dt), api_dt_argument=1.0)
        return value, record

    def metadata(self):
        return {
            "model": "gripper_contact_proxy",
            "policy_and_safety_input_in_policy_mode": True,
            "actor_paths": self.actor_paths,
            "environment_paths": list(ENVIRONMENT_PATHS),
            "force_units": "N: signed normal impulse from API dt=1 divided once by physics dt",
            "moment_units": "N m: sum (contact point - measured TCP position) cross signed force",
            "axes": "world/base XYZ for force and moment; no TCP-axis rotation",
            "bias": "unchanged recorded initial wrist_ft six-vector added to contact wrench",
            "sampling": "current physics step at executor control rate, cached for observation and logging; no averaging or clipping",
            "omitted": [
                "tangential/friction forces",
                "gravity",
                "inertia",
                "self contacts",
                "proximal arm contacts",
                "current-based UR3 transfer model",
            ],
            "validation": "idealized contact proxy; sensor frame/bias compatibility and physical UR3 transfer uncalibrated",
            "trace": "wrist_contact_trace.jsonl",
        }
