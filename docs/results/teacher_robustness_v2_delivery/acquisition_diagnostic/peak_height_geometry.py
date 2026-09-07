#!/usr/bin/env python3
"""Independent rigid-packet tilt/lowest-corner diagnostic; no score changes."""

import json

import audit_first_start as audit
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    cases = [
        "fta3000_nfe1_k4__start_1787395963__seed903101",
        "v5_6_nfe1_k4__start_1787396028__seed903101",
        "v5_6_nfe1_k4__start_1787396273__seed903101",
        "fta1500_nfe5_k1__start_1787396273__seed903102",
    ]
    rows = []
    for case in cases:
        root = audit.ROOT / "rollouts" / case
        config = json.loads((root / "effective_config.json").read_text())
        z = np.load(root / "sim_trace.npz")
        p, q = z["waffle_position"], z["waffle_orientation_wxyz"]
        rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        half = np.array(config["waffle"]["size"]) / 2
        bottom_z = p[:, 2] - np.sum(np.abs(rotation[:, 2]) * half, axis=1)
        angle = (
            Rotation.from_quat(q[:, [1, 2, 3, 0]])
            * Rotation.from_quat(q[0, [1, 2, 3, 0]]).inv()
        ).magnitude()
        i = int(np.argmax(p[:, 2]))
        rows.append(
            {
                "case_id": case,
                "initial_lowest_corner_z_m": float(bottom_z[0]),
                "peak_center_t_s": float(z["t"][i]),
                "peak_center_rise_m": float(p[i, 2] - p[0, 2]),
                "lowest_corner_rise_at_peak_center_m": float(bottom_z[i] - bottom_z[0]),
                "orientation_change_at_peak_center_deg": float(np.rad2deg(angle[i])),
                "maximum_lowest_corner_rise_m": float((bottom_z - bottom_z[0]).max()),
                "peak_center_position_m": p[i].tolist(),
                "peak_center_orientation_wxyz": q[i].tolist(),
                "packet_size_m": (2 * half).tolist(),
                "trace_sha256": audit.sha(root / "sim_trace.npz"),
            }
        )
    print(
        json.dumps(
            {
                "trials": rows,
                "semantics": "Rigid OBB lowest corner uses exact stored quaternion and configured packet size. Indicates tilt versus complete geometric clearance; does not measure support forces, deformable packet contact, or alter frozen center-height success thresholds.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
