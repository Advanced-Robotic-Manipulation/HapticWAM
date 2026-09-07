#!/usr/bin/env python3
"""Extract the first complete measured per-pad capture in the initial0.5s.

Read-only source streams. Writes only new baseline/provenance files beneath the
explicit teacher_v2_preflights output directory. No model or device imports.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

SOURCE = Path(
    "/home/physicalai/phantom-icra-2027/data/full/tasks/waffles/ep_waffles_1787395928_000"
)
OUTPUT = Path(
    "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_v2_preflights"
)
SHAPES = {
    "infer_img": (288, 384),
    "fields_ds": (72, 96, 8),
    "keyframes": (144, 192, 8),
    "wrench": (6,),
    "area": (),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload_hash(a):
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}, sort_keys=True)
    return digest(header.encode() + b"\n" + a.tobytes(order="C"))


def read_stream(name):
    group = zarr.open_group(str(SOURCE / f"{name}.zarr"), mode="r")
    ts = np.asarray(group["ts"][:])
    assert len(ts) == len(group["data"]) and np.isfinite(ts).all()
    assert np.all(np.diff(ts) > 0), name
    return group["data"], ts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-sha256", required=True)
    args = parser.parse_args()
    target = OUTPUT / "aug22_5928_no_contact_sensor_baseline.npz"
    provenance = target.with_suffix(".json")
    assert OUTPUT.is_dir() and not target.exists() and not provenance.exists()
    arrays = {}
    arm, arm_t = read_stream("arm_tcp_pose")
    first_arm_t = float(arm_t[0])
    selections, inventories = {}, {}
    for side in ("left", "right"):
        streams = {name: read_stream(f"tactile_{side}_{name}") for name in SHAPES}
        key_t = streams["keyframes"][1]
        candidates = key_t[(key_t >= first_arm_t) & (key_t <= first_arm_t + 0.5)]
        capture_t = next(
            float(t)
            for t in candidates
            if all(np.any(ts == t) for _, ts in streams.values())
        )
        selected = {}
        inventory = {}
        for name, (data, ts) in streams.items():
            i = int(np.searchsorted(ts, capture_t))
            assert ts[i] == capture_t
            value = np.asarray(data[i])
            assert value.shape == SHAPES[name] and np.isfinite(value).all()
            assert name != "infer_img" or value.dtype == np.uint8
            arrays[f"{side}_{name}"] = value
            selected[name] = {
                "source_stream": str(SOURCE / f"tactile_{side}_{name}.zarr"),
                "index": i,
                "t_master_s": capture_t,
                "relative_to_first_arm_s": capture_t - first_arm_t,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "selected_payload_sha256": payload_hash(value),
                "zarr_data_metadata_sha256": digest(
                    (SOURCE / f"tactile_{side}_{name}.zarr/data/.zarray").read_bytes()
                ),
            }
            inventory[name] = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "first_t_master_s": float(ts[0]),
                "last_t_master_s": float(ts[-1]),
                "timestamps_sha256": digest(ts.tobytes()),
            }
        assert float(arrays[f"{side}_area"]) == 0.0, "selected SDK area is nonzero"
        f = arrays[f"{side}_fields_ds"].astype(float)
        w = arrays[f"{side}_wrench"].astype(float)
        selected["summary"] = {
            "area_sdk": float(arrays[f"{side}_area"]),
            "wrench_sdk": w.tolist(),
            "depth_min_max_mm_recorded_convention": [
                float(f[..., 2].min()),
                float(f[..., 2].max()),
            ],
            "mean_distributed_force_times_110592": (
                f[..., 5:8].mean((0, 1)) * 110592
            ).tolist(),
            "max_field_force_reconstruction_error_sdk": float(
                np.max(abs(f[..., 5:8].mean((0, 1)) * 110592 - w[:3]))
            ),
        }
        area, ts_area = streams["area"]
        measured_area = np.asarray(area[:])
        nonzero = np.flatnonzero(measured_area > 0)
        selected["first_recorded_nonzero_area_relative_s"] = (
            None if not len(nonzero) else float(ts_area[nonzero[0]] - first_arm_t)
        )
        no_contact = ts_area <= first_arm_t + 2.0
        selected["first_two_seconds_area_max_sdk"] = float(
            measured_area[no_contact].max()
        )
        selections[side], inventories[side] = selected, inventory
    cutoff = max(selections[side]["infer_img"]["t_master_s"] for side in selections)
    ai = int(np.searchsorted(arm_t, cutoff, side="right") - 1)
    rgb, rgb_t = read_stream("camera_scene_color")
    ci = int(np.searchsorted(rgb_t, cutoff, side="right") - 1)
    camera_frame = np.asarray(rgb[ci])
    if camera_frame.ndim == 1:
        import cv2

        camera_frame = cv2.cvtColor(
            cv2.imdecode(camera_frame, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
    meta_bytes = (SOURCE / "meta.json").read_bytes()
    previous = (
        OUTPUT.parent / "teacher_pick_place_v1/sept4_no_contact_sensor_baseline.npz"
    )
    previous_comparison = {}
    with np.load(previous, allow_pickle=False) as old:
        for side in selections:
            previous_comparison[side] = {
                "gel_absolute_pixel_difference_mean": float(
                    np.mean(
                        abs(
                            arrays[f"{side}_infer_img"].astype(float)
                            - old[f"{side}_infer_img"].astype(float)
                        )
                    )
                ),
                "old_wrench_sdk": old[f"{side}_wrench"].tolist(),
                "new_wrench_sdk": arrays[f"{side}_wrench"].tolist(),
            }
    with target.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with np.load(target, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for key, value in arrays.items():
            assert np.array_equal(saved[key], value), key
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "measured_initial_no_contact_baseline_available_with_residuals",
        "source_episode": str(SOURCE),
        "source_meta_sha256": digest(meta_bytes),
        "extraction_script_sha256": args.script_sha256,
        "baseline_path": str(target),
        "baseline_sha256": digest(target.read_bytes()),
        "baseline_bytes": target.stat().st_size,
        "first_arm_t_master_s": first_arm_t,
        "frozen_capture_cutoff_t_master_s": cutoff,
        "cutoff_relative_to_first_arm_s": cutoff - first_arm_t,
        "selection_rule": "first post-start keyframe per pad with an exact same-timestamp infer_img/fields_ds/wrench/area capture, within first0.5s; all are at or before frozen cutoff",
        "causality_limit": "Static sensor calibration from the initial no-contact interval. These frames were not all available at the earlier q0 timestamp. No future contact, release, or policy outcome is used to select frames; the baseline never advances during simulation.",
        "invented_or_zeroed_values": False,
        "source_arrays_unchanged": True,
        "inventories": inventories,
        "selections": selections,
        "arm_at_causal_cutoff": {
            "index": ai,
            "t_master_s": float(arm_t[ai]),
            "tcp_pose": np.asarray(arm[ai]).tolist(),
        },
        "camera_at_causal_cutoff": {
            "index": ci,
            "t_master_s": float(rgb_t[ci]),
            "shape": list(camera_frame.shape),
            "decoded_rgb_sha256": digest(camera_frame.tobytes()),
        },
        "no_contact_evidence": "Both selected SDK areas and each pad's first2s recorded areas are0. Initial RGB frames0/.5/1s show the packet on the mat and no human hand. Native initial TCPz is about0.356m. This supports an unloaded initial interval but does not calibrate residual sensor offsets or actual forces.",
        "units_limit": "Wrench/area are unchanged SDK outputs. Distributed-force mean×110592 numerically reconstructs recorded wrench force channels. SI accuracy, torque units, optical deformation, and the pressure law remain unvalidated; left unloaded fz/depth residuals are preserved.",
        "previous_baseline": {
            "path": str(previous),
            "sha256": digest(previous.read_bytes()),
            "comparison": previous_comparison,
        },
        "archive_readback_exact": True,
    }
    with provenance.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "baseline_path",
                    "baseline_sha256",
                    "baseline_bytes",
                    "cutoff_relative_to_first_arm_s",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
