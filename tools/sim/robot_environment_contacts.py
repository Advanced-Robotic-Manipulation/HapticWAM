"""Optional diagnostic observation of robot/environment contact constraints.

This is never a policy input, safety input, task-success signal or actuator.
Each exact robot-body path gets its own view so contact rows cannot silently
acquire the wrong actor label. The eight environment filters are exact paths.
The Isaac API is queried with dt=1, returning impulses; this module explicitly
converts populated normal impulses to newtons using the actual physics step.

Reporting adds ContactReportAPI with zero reporting threshold. It does not
change collision pairs, drive/material settings, transforms or stabilization.
A separate diagnostic replay must still verify observation invariance.
"""

from __future__ import annotations

import re

import numpy as np

ENVIRONMENT_PATHS = (
    "/World/Bench/Slab",
    "/World/Mat/Base",
    "/World/Bin/Bottom",
    "/World/Bin/Left",
    "/World/Bin/Right",
    "/World/Bin/Front",
    "/World/Bin/Back",
    "/World/Waffle",
)


def _validate_paths(robot_paths, environment_paths):
    robots, environment = list(map(str, robot_paths)), list(map(str, environment_paths))
    if not robots or len(robots) != len(set(robots)):
        raise ValueError("nonempty unique explicit robot body paths required")
    if any(re.fullmatch(r"/World/Robot(?:/[A-Za-z0-9_]+)+", p) is None for p in robots):
        raise ValueError(
            "robot actors must be exact /World/Robot/ body paths, no patterns"
        )
    if not environment or len(environment) != len(set(environment)):
        raise ValueError("nonempty unique explicit environment paths required")
    if any(p not in ENVIRONMENT_PATHS for p in environment):
        raise ValueError(
            "environment filters must be explicit table/mat/bin/packet paths"
        )
    return robots, environment


def summarize_actor_contacts(
    contact_data,
    actor_path,
    environment_paths,
    physics_dt,
    *,
    max_contact_count=None,
):
    """Reduce one actor's six buffers obtained with get_contact_force_data(dt=1).

    Counts/starts must have one sensor row and one column per explicit filter.
    Only populated entries are examined. Scalar sign, world normals and signed
    geometric separation are preserved, while safety-relevant load summaries
    sum magnitudes so opposed constraints cannot cancel. Shared indices across
    filter labels raise: this observer will not invent a contacting body name.
    """
    _, paths = _validate_paths([actor_path], environment_paths)
    if not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("physics_dt must be finite positive seconds")
    if max_contact_count is not None and (
        not isinstance(max_contact_count, (int, np.integer)) or max_contact_count <= 0
    ):
        raise ValueError("max_contact_count must be a positive integer")
    impulse, points, normals, separations, counts, starts = contact_data
    impulse = np.asarray(impulse, dtype=float).reshape(-1)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    separations = np.asarray(separations, dtype=float).reshape(-1)
    counts, starts = np.asarray(counts), np.asarray(starts)
    shape = (1, len(paths))
    if counts.shape != shape or starts.shape != shape:
        raise RuntimeError("contact sensor/filter dimensions do not identify one actor")
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts != counts.astype(np.int64))
    ):
        raise RuntimeError("invalid populated contact counts")
    if max_contact_count is not None and counts.sum() >= max_contact_count:
        raise RuntimeError("robot/environment contact buffer may be saturated")
    contacts, seen, total_impulse = [], set(), 0.0
    for column, (path, count, start) in enumerate(zip(paths, counts[0], starts[0])):
        if count == 0:
            continue
        if not np.isfinite(start) or start < 0 or start != int(start):
            raise RuntimeError("invalid populated contact start")
        first, count = int(start), int(count)
        if first + count > min(
            len(impulse), len(points), len(normals), len(separations)
        ):
            raise RuntimeError("populated contact range exceeds one or more buffers")
        ids = np.arange(first, first + count)
        if seen.intersection(ids):
            raise RuntimeError(
                "shared contact indices ambiguously identify environment bodies"
            )
        seen.update(ids)
        raw, xyz, normal, sep = (
            impulse[ids],
            points[ids],
            normals[ids],
            separations[ids],
        )
        if not np.isfinite(np.r_[raw, xyz.ravel(), normal.ravel(), sep]).all():
            raise RuntimeError("nonfinite populated robot/environment contact data")
        if np.any((abs(raw) > 0) & (np.linalg.norm(normal, axis=1) < 1e-9)):
            raise RuntimeError("loaded contact has invalid zero world normal")
        magnitude_impulse = float(abs(raw).sum())
        total_impulse += magnitude_impulse
        contacts.append(
            {
                "actor_path": str(actor_path),
                "filter_path": path,
                "filter_column": column,
                "contact_count": count,
                "populated_buffer_indices": ids.tolist(),
                "normal_impulse_signed_ns": raw.tolist(),
                "normal_force_signed_n": (raw / physics_dt).tolist(),
                "normal_impulse_magnitude_ns": magnitude_impulse,
                "normal_force_magnitude_n": magnitude_impulse / physics_dt,
                "points_world_m": xyz.tolist(),
                "normals_world": normal.tolist(),
                "signed_separation_m": sep.tolist(),
            }
        )
    return {
        "actor_path": str(actor_path),
        "environment_paths": paths,
        "sensor_rows": 1,
        "filter_columns": len(paths),
        "pair_contact_counts": counts[0].astype(int).tolist(),
        "pair_contact_start_indices": [
            int(start) if count else None for count, start in zip(counts[0], starts[0])
        ],
        "populated_unique_contact_count": len(seen),
        "normal_impulse_magnitude_ns": total_impulse,
        "normal_force_magnitude_n": total_impulse / physics_dt,
        "contacts": contacts,
    }


class RobotEnvironmentContactViews:
    """Create before reset; initialize after reset; observe at chosen trace times."""

    def __init__(
        self,
        robot_paths,
        environment_paths=ENVIRONMENT_PATHS,
        *,
        rigid_prim_cls=None,
        max_contact_count=512,
    ):
        self.robot_paths, self.environment_paths = _validate_paths(
            robot_paths, environment_paths
        )
        if not isinstance(max_contact_count, int) or max_contact_count <= 0:
            raise ValueError("max_contact_count must be a positive integer")
        self.max_contact_count = max_contact_count
        if rigid_prim_cls is None:
            from isaacsim.core.prims import RigidPrim

            rigid_prim_cls = RigidPrim
        self.views = [
            rigid_prim_cls(
                prim_paths_expr=path,
                name=f"robot_environment_diagnostic_{i}",
                track_contact_forces=True,
                contact_filter_prim_paths_expr=self.environment_paths,
                max_contact_count=max_contact_count,
                reset_xform_properties=False,
                # Installed 6.0 deprecated RigidPrim retains this exact spelling.
                disable_stablization=False,
            )
            for i, path in enumerate(self.robot_paths)
        ]
        self.initialized = False

    def initialize(self):
        for path, view in zip(self.robot_paths, self.views):
            view.initialize()
            if list(view.prim_paths) != [path]:
                raise RuntimeError("robot contact view resolved an unexpected actor")
        self.initialized = True

    def get_all(self, dt):
        if not self.initialized:
            raise RuntimeError("initialize robot/environment contact views first")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("physics step must be finite positive seconds")
        actors = [
            summarize_actor_contacts(
                view.get_contact_force_data(dt=1.0),
                path,
                self.environment_paths,
                dt,
                max_contact_count=self.max_contact_count,
            )
            for path, view in zip(self.robot_paths, self.views)
        ]
        by_filter = {p: 0.0 for p in self.environment_paths}
        for actor in actors:
            for contact in actor["contacts"]:
                by_filter[contact["filter_path"]] += contact["normal_force_magnitude_n"]
        return {
            "physics_dt_s": float(dt),
            "api_dt_argument": 1.0,
            "robot_paths": self.robot_paths,
            "environment_paths": self.environment_paths,
            "max_contact_count_per_actor": self.max_contact_count,
            "normal_force_magnitude_n": sum(
                a["normal_force_magnitude_n"] for a in actors
            ),
            "normal_force_by_environment_n": by_filter,
            "per_actor": actors,
            "semantics": "Scene-rate diagnostic only. Raw PhysX signed normal impulses / actual dt; sum magnitudes before cancellation. World points/normals and signed separation preserved. No friction-force estimate, no collision/safety/policy feedback.",
            "self_collision_observed": False,
        }
