#!/usr/bin/env python3
"""CPU-only descriptive wrist-subguard replay on ten native Aug22 demos.

Run on compute3; JSON stdout. No hardware/model/Isaac imports or source writes.
Camera previews are optional; numeric provenance is always included.
"""

import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import zarr

ROOT = Path("/home/physicalai/phantom-icra-2027/data/full/tasks/waffles")
EPISODES = [
    "1787395928_000",
    "1787395963_001",
    "1787396028_003",
    "1787396060_004",
    "1787396094_005",
    "1787396128_006",
    "1787396273_010",
    "1787396314_011",
    "1787396346_012",
    "1787396461_000",
]


def payload_sha(a):
    a = np.asarray(a)
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}, sort_keys=True)
    return hashlib.sha256(header.encode() + b"\n" + a.tobytes(order="C")).hexdigest()


def stream(root, name, full=True):
    group = zarr.open_group(str(root / f"{name}.zarr"), mode="r")
    t = np.asarray(group["ts"][:], dtype=float)
    assert np.isfinite(t).all() and np.all(np.diff(t) > 0), name
    assert len(t) == len(group["data"]), name
    a = np.asarray(group["data"][:]) if full else group["data"]
    provenance = {
        "path": str(root / f"{name}.zarr"),
        "count": len(t),
        "timestamps_payload_sha256": payload_sha(t),
        "data_payload_sha256": payload_sha(a) if full else None,
    }
    return t, a, provenance


def evaluate(t, ft, reference):
    baseline, over_since, last = ft[0].copy(), None, None
    first_over, first_stop, peak_force, peak_torque = None, None, None, None
    prefix_force, prefix_torque, prefix_stop = 0.0, 0.0, None
    max_baseline_shift = 0.0
    for index, (time, value) in enumerate(zip(t, ft)):
        delta = value - baseline
        fn, tn = float(np.linalg.norm(delta[:3])), float(np.linalg.norm(delta[3:]))
        over = fn > 60.0 or tn > 15.0
        if over and over_since is None:
            over_since = float(time)
        if not over:
            over_since = None
        event = {
            "sample_index": index,
            "t_master_s": float(time),
            "relative_s": float(time - t[0]),
            "force_deviation_n": fn,
            "torque_deviation_nm": tn,
            "baseline_n_nm": baseline.tolist(),
            "actual_wrist_n_nm": value.tolist(),
            "over_since_relative_s": None
            if over_since is None
            else float(over_since - t[0]),
        }
        if over and first_over is None:
            first_over = event
        tripped = over and time - over_since >= 0.3
        if tripped and first_stop is None:
            first_stop = event
        if time <= t[0] + 1:
            prefix_force, prefix_torque = max(prefix_force, fn), max(prefix_torque, tn)
            if tripped and prefix_stop is None:
                prefix_stop = event
        if peak_force is None or fn > peak_force["force_deviation_n"]:
            peak_force = event
        if peak_torque is None or tn > peak_torque["torque_deviation_nm"]:
            peak_torque = event
        max_baseline_shift = max(
            max_baseline_shift, float(np.linalg.norm(baseline[:3] - ft[0, :3]))
        )
        dt = 0.0 if last is None else time - last
        last = time
        if reference == "rolling" and not over and dt > 0:
            baseline += min(1.0, dt / 2.0) * delta
    return {
        "reference": reference,
        "first_crossing": first_over,
        "first_debounced_stop": first_stop,
        "largest_force_deviation": peak_force,
        "largest_torque_deviation": peak_torque,
        "maximum_baseline_force_shift_n": max_baseline_shift,
        "initial_0_to_1s": {
            "maximum_force_deviation_n": prefix_force,
            "maximum_torque_deviation_nm": prefix_torque,
            "first_debounced_stop": prefix_stop,
        },
        "continuation_semantics": "Entire observed stream is inspected descriptively after hypothetical first stop; no alternate closed-loop trajectory is inferred.",
    }


def describe_episode(name, preview):
    root = ROOT / ("ep_waffles_" + name)
    t, ft, ft_prov = stream(root, "arm_ft")
    assert ft.shape == (len(t), 6) and np.isfinite(ft).all()
    initial = {
        "time_reference": "first native arm_ft row; no preselected simulator anchor or timestamp-offset reapplication"
    }
    provenance = {"arm_ft": ft_prov}
    for key in ["arm_q", "arm_qd", "arm_tcp_pose", "gripper"]:
        ts, a, prov = stream(root, key)
        select = (ts >= t[0]) & (ts <= t[0] + 1)
        values = a[select]
        assert len(values) and np.isfinite(values).all()
        record = {
            "samples": len(values),
            "first_relative_s": float(ts[select][0] - t[0]),
            "last_relative_s": float(ts[select][-1] - t[0]),
            "first_value": values[0].tolist(),
            "last_value": values[-1].tolist(),
            "maximum_absolute_component_change": float(
                np.abs(values - values[0]).max()
            ),
        }
        if key == "arm_tcp_pose":
            record["maximum_translation_from_first_m"] = float(
                np.linalg.norm(values[:, :3] - values[0, :3], axis=1).max()
            )
            record["z_min_max_m"] = [
                float(values[:, 2].min()),
                float(values[:, 2].max()),
            ]
        if key == "arm_qd":
            record["maximum_absolute_joint_speed_rad_s"] = float(np.abs(values).max())
        if key == "gripper":
            record["closure_min_max"] = [
                float(values[:, 0].min()),
                float(values[:, 0].max()),
            ]
        initial[key], provenance[key] = record, prov
    for side in ["left", "right"]:
        pad = {}
        for suffix in ["area", "wrench", "fields_ds"]:
            key = f"tactile_{side}_{suffix}"
            ts, a, prov = stream(root, key, full=suffix != "fields_ds")
            select = np.flatnonzero((ts >= t[0]) & (ts <= t[0] + 1))
            values = np.asarray([a[int(i)] for i in select])
            assert len(values) and np.isfinite(values).all()
            record = {
                "samples": len(select),
                "first_relative_s": float(ts[select[0]] - t[0]),
                "last_relative_s": float(ts[select[-1]] - t[0]),
                "selected_payload_sha256": payload_sha(values),
            }
            if suffix == "area":
                record["min_max_sdk"] = [float(values.min()), float(values.max())]
            elif suffix == "wrench":
                record["first_sdk_value"] = values[0].tolist()
                record["max_force_change_sdk"] = float(
                    np.linalg.norm(values[:, :3] - values[0, :3], axis=1).max()
                )
            else:
                record["recorded_depth_channel2_min_max"] = [
                    float(values[..., 2].min()),
                    float(values[..., 2].max()),
                ]
            pad[suffix], provenance[key] = record, prov
        initial[side + "_tactile"] = pad
    tc, camera, camera_prov = stream(root, "camera_scene_color", full=False)
    frames = []
    for requested in [0, 0.5, 1]:
        i = min(int(np.searchsorted(tc, t[0] + requested)), len(tc) - 1)
        stored = np.asarray(camera[i])
        if stored.dtype.kind == "S":
            frame = cv2.imdecode(
                np.frombuffer(stored.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
        elif stored.ndim == 1:
            frame = cv2.imdecode(stored, cv2.IMREAD_COLOR)
        else:
            frame = cv2.cvtColor(stored, cv2.COLOR_RGB2BGR)
        assert frame is not None
        row = {
            "index": i,
            "requested_relative_s": requested,
            "actual_relative_s": float(tc[i] - t[0]),
            "stored_payload_sha256": payload_sha(stored),
            "decoded_bgr_sha256": payload_sha(frame),
        }
        if preview:
            frame = cv2.resize(frame, (320, 240))
            cv2.rectangle(frame, (0, 0), (320, 22), (15, 15, 15), -1)
            cv2.putText(
                frame,
                f"{name} t={tc[i] - t[0]:.3f}s",
                (3, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
            )
            row["preview_jpeg_base64"] = base64.b64encode(
                cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
            ).decode()
        frames.append(row)
    initial["rgb_frames"] = frames
    initial["unloaded_adjudication"] = (
        "pending manual RGB review; zero tactile area alone does not exclude non-gel contacts"
    )
    provenance["camera_scene_color"] = camera_prov
    return {
        "episode": root.name,
        "path": str(root),
        "duration_s": float(t[-1] - t[0]),
        "arm_ft_samples": len(t),
        "first_arm_ft_n_nm": ft[0].tolist(),
        "first_arm_ft_master_s": float(t[0]),
        "timestamp_step_min_median_max_s": [
            float(np.diff(t).min()),
            float(np.median(np.diff(t))),
            float(np.diff(t).max()),
        ],
        "meta_sha256": hashlib.sha256((root / "meta.json").read_bytes()).hexdigest(),
        "stream_provenance": provenance,
        "rolling": evaluate(t, ft, "rolling"),
        "episode_fixed": evaluate(t, ft, "episode_fixed"),
        "initial_0_to_1s_evidence": initial,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-preview", action="store_true")
    args = parser.parse_args()
    rows = [describe_episode(name, args.with_preview) for name in EPISODES]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Ten native Aug22 wrist-stream descriptive replays; separate qualification for future opt-in guard. No simulator/model/hardware, source writes, or primary-study score changes.",
                "rules": {
                    "force_limit_n": 60,
                    "torque_limit_nm": 15,
                    "debounce_s": 0.3,
                    "rolling_tau_s": 2,
                    "initialization": "exact first native arm_ft row in each stream",
                    "time": "native stored master-clock timestamps; not resampled",
                    "baseline_update": "rolling only while deviation is below both limits; episode_fixed never updates",
                },
                "counts": {
                    mode: {
                        "full_stream_debounced_triggers": sum(
                            r[mode]["first_debounced_stop"] is not None for r in rows
                        ),
                        "initial_0_to_1s_debounced_triggers": sum(
                            r[mode]["initial_0_to_1s"]["first_debounced_stop"]
                            is not None
                            for r in rows
                        ),
                    }
                    for mode in ["rolling", "episode_fixed"]
                },
                "episodes": rows,
                "limits": [
                    "Recorded later contacts are not adjudicated, so trigger counts are not false-positive rates.",
                    "No current-based UR3 wrench calibration or real-hardware safety qualification is inferred.",
                    "Three RGB samples support only sampled visual evidence; tactile area/wrench and arm motion characterize the full initial second.",
                    "Continuing the fixed recorded stream after a hypothetical stop is descriptive, not a predicted altered demonstration.",
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
