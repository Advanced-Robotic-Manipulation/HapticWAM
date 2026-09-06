"""Observe packet support from positive PhysX normal contact magnitudes.

One independent packet view filters explicit bin and robot bodies. This does
not change collisions or motion. Robot support includes both pads, backing,
housing and arm bodies; bin support includes the five physical bin colliders.
Normal magnitudes are summed before any vector cancellation. Filter coverage
is an audited setup claim: this helper cannot discover omitted physical bodies.
"""

from __future__ import annotations

import numpy as np


def summarize_support_contacts(
    contact_data, filter_paths, bin_paths, robot_paths, *, max_contact_count=None
):
    """Reduce populated force buffers without double-counting shared indices."""
    force, _, _, _, counts, starts = contact_data
    force = np.asarray(force, dtype=float).reshape(-1)
    counts, starts = np.asarray(counts).ravel(), np.asarray(starts).ravel()
    if (
        counts.shape != starts.shape
        or len(counts) != len(filter_paths)
        or not np.isfinite(counts).all()
        or (counts < 0).any()
        or (counts != counts.astype(np.int64)).any()
    ):
        raise RuntimeError("invalid packet support contact counts/filter columns")
    if max_contact_count is not None and counts.sum() >= max_contact_count:
        raise RuntimeError("packet support contact buffer may be saturated")
    bins, robots = set(bin_paths), set(robot_paths)
    if bins & robots or set(filter_paths) != bins | robots:
        raise ValueError("support filters must partition into bin and robot bodies")
    groups = {"bin": set(), "robot": set()}
    records = []
    for path, count, start in zip(filter_paths, counts.astype(np.int64), starts):
        category = "bin" if path in bins else "robot"
        if count == 0:
            continue
        if not np.isfinite(start) or start < 0 or start != int(start):
            raise RuntimeError("invalid packet support contact start")
        first = int(start)
        if first + count > len(force):
            raise RuntimeError("packet support contact range exceeds buffer")
        ids = set(range(first, first + int(count)))
        values = force[sorted(ids)]
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite populated packet support force")
        groups[category].update(ids)
        records.append(
            {
                "filter_path": path,
                "category": category,
                "contact_count": int(count),
                "normal_force_n": float(np.abs(values).sum()),
            }
        )
    if groups["bin"] & groups["robot"]:
        raise RuntimeError("packet contact indices ambiguously label bin and robot")
    return {
        "packet_robot_normal_force": float(
            np.abs(force[sorted(groups["robot"])]).sum()
        ),
        "packet_bin_normal_force": float(np.abs(force[sorted(groups["bin"])]).sum()),
        "populated_unique_contact_count": len(groups["bin"] | groups["robot"]),
        "duplicate_filter_contact_indices": int(counts.sum())
        - len(groups["bin"] | groups["robot"]),
        "per_filter_contacts": records,
        "force_semantics": "sum of positive normal magnitudes; no net-vector cancellation",
    }


class PacketSupportViews:
    """Create before reset, initialize afterward; call get_all(physics_dt)."""

    def __init__(
        self,
        packet_path,
        bin_paths,
        robot_paths,
        *,
        rigid_prim_cls=None,
        max_contact_count=1024,
    ):
        self.packet_path = str(packet_path)
        self.bin_paths = list(dict.fromkeys(map(str, bin_paths)))
        self.robot_paths = list(dict.fromkeys(map(str, robot_paths)))
        if not self.bin_paths or not self.robot_paths:
            raise ValueError("explicit nonempty bin and robot body filters required")
        if not all(path.startswith("/World/Bin/") for path in self.bin_paths):
            raise ValueError("bin body filters must be physical /World/Bin/ paths")
        if not all(
            path == "/World/Robot" or path.startswith("/World/Robot/")
            for path in self.robot_paths
        ):
            raise ValueError(
                "robot body filters must exclude bin, packet and environment paths"
            )
        if (
            set(self.bin_paths) & set(self.robot_paths)
            or self.packet_path in self.bin_paths + self.robot_paths
        ):
            raise ValueError("packet, bin and robot filters must be distinct")
        self.filter_paths = self.bin_paths + self.robot_paths
        self.max_contact_count = int(max_contact_count)
        if self.max_contact_count <= 0:
            raise ValueError("max_contact_count must be positive")
        if rigid_prim_cls is None:
            from isaacsim.core.prims import RigidPrim

            rigid_prim_cls = RigidPrim
        self.view = rigid_prim_cls(
            prim_paths_expr=self.packet_path,
            name="packet_support_contacts",
            track_contact_forces=True,
            contact_filter_prim_paths_expr=self.filter_paths,
            max_contact_count=self.max_contact_count,
            reset_xform_properties=False,
        )

    def initialize(self):
        self.view.initialize()

    def get_all(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("physics dt must be finite and positive seconds")
        result = summarize_support_contacts(
            self.view.get_contact_force_data(dt=dt),
            self.filter_paths,
            self.bin_paths,
            self.robot_paths,
            max_contact_count=self.max_contact_count,
        )
        result.update(
            packet_path=self.packet_path,
            bin_paths=self.bin_paths,
            robot_paths=self.robot_paths,
        )
        return result
