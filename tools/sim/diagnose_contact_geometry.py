#!/usr/bin/env python3
"""Summarize recorded grasp geometry and physical trial contact attribution.

Read-only diagnostics: does not tune scenes, run Isaac or modify recordings.
Net pad force and packet-filtered force must be separated because a finger
blocked on the mat can report simulated OBJ=2 without holding the packet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def recorded_summary(path):
    arrays = np.load(path / "replay.npz")
    index = int(np.argmin(arrays["tcp"][:, 2]))
    native_grip = arrays["native_gripper"]
    detected = native_grip[:, 1] == 2
    result = {
        "episode": path.name,
        "duration_s": float(arrays["t"][-1]),
        "minimum_tcp_z_m": float(arrays["tcp"][index, 2]),
        "minimum_tcp_time_s": float(arrays["t"][index]),
        "minimum_tcp_pose": arrays["tcp"][index].tolist(),
        "gripper_feedback_at_minimum": arrays["gripper"][index].tolist(),
        "OBJ2_sample_count": int(detected.sum()),
        "first_OBJ2_s": float(arrays["native_gripper_t"][np.flatnonzero(detected)[0]])
        if detected.any()
        else None,
    }
    for stream in ["actions", "actions_abs", "actions_plan"]:
        key = "native_" + stream
        if key in arrays:
            result[stream + "_closure_range"] = [
                float(arrays[key][:, -1].min()),
                float(arrays[key][:, -1].max()),
            ]
    return result


def trial_summary(path):
    arrays = np.load(path / "sim_trace.npz")
    if "pad_packet_force" not in arrays:
        return {"trial": path.name, "status": "no packet-filtered contact telemetry"}
    net = np.linalg.norm(arrays["pad_force"], axis=-1)
    packet = np.linalg.norm(arrays["pad_packet_force"], axis=-1)
    z = arrays["waffle_position"][:, 2]
    samples = []
    for time in [7.0, 8.0, 9.0, 10.0, 11.0]:
        i = int(np.argmin(abs(arrays["t"] - time)))
        samples.append(
            {
                "time_s": float(arrays["t"][i]),
                "packet_position_m": arrays["waffle_position"][i].tolist(),
                "gripper_feedback": arrays["gripper"][i].tolist(),
                "pad_net_force_world_N": arrays["pad_force"][i].tolist(),
                "pad_packet_force_world_N": arrays["pad_packet_force"][i].tolist(),
            }
        )
    return {
        "trial": path.name,
        "initial_packet_z_m": float(z[0]),
        "peak_packet_rise_m": float(z.max() - z[0]),
        "final_packet_position_m": arrays["waffle_position"][-1].tolist(),
        "peak_net_force_per_pad_N": net.max(axis=0).tolist(),
        "peak_packet_force_per_pad_N": packet.max(axis=0).tolist(),
        "other_contact_blockage_sample_count_per_pad": ((net > 5) & (packet < 0.1))
        .sum(axis=0)
        .tolist(),
        "criterion": "net force >5N and packet force <0.1N indicates another contact, often mat/table; does not classify pair automatically.",
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/isaac_waffles"))
    args = parser.parse_args()
    evidence = args.root / "evidence"
    recorded = [
        recorded_summary(path)
        for path in sorted((evidence / "fit").iterdir())
        if (path / "replay.npz").exists()
    ]
    trials = [
        trial_summary(path)
        for path in sorted((args.root / "tuning").iterdir())
        if (path / "sim_trace.npz").exists()
    ]
    result = {
        "status": "analytical diagnosis and unvalidated contact-shape hypotheses",
        "recorded_fit_episodes": recorded,
        "physical_trials": trials,
        "findings": [
            "Baseline 12x26x55mm pad boxes never touched the original -5mm packet center; recorded TCP minima are around72-75mm.",
            "Raised55 packet contact was mainly upward world force (packet top contact), concentrated near a pad corner.",
            "Raised75/90 showed large net pad force with almost zero packet-filtered force: mat/table blockage prevented closure.",
            "A simulated OBJ2 stall can be caused by floor contact; it is not a grasp-success detector.",
            "Measured motor POS does not measure physical aperture; custom pads touch nearPOS=.9 while85mm is bare nominal stroke.",
        ],
        "candidate_physical_parameters": {
            "contact_shape": "ellipse: YZ face radii13mm/27.5mm extruded12mm alongX; backing and linkage also rounded",
            "effective_stroke_m": [0.070, 0.0765, 0.085],
            "stroke_rationale": ".0765=.085*.9 assumes bare85mm stroke and custom touch atPOS.9; atPOS.6 gap25.5mm vs28.33mm baseline.",
            "table_plane_shift_m": [0.055, 0.075],
            "contact_center_z_m": [0.16556, 0.185],
            "status": "sensitivity candidates, not calibrated values or validated transfers",
        },
        "appearance_fit_caveat": "Six-parameter white-shell fit reached4.30px RMS with midpoint[.00602,-.01231,.16133] in tool0, yaw-.14566, outward-shell offset11.875mm and stroke70.096mm. White shell centers are not contact centers; negative toolY raises the collider at grasp and does not solve contact reach.",
    }
    target = evidence / "contact_geometry_diagnosis.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
