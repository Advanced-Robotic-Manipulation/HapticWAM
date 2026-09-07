#!/usr/bin/env python3
"""Read native timestamps only and audit current loader coverage; no recording writes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from phantom.config.hardware import load_hardware
from phantom.train.common import close_index

ROOT = Path("/home/physicalai/phantom-icra-2027")
REPO = ROOT / "phantom"
hw = load_hardware(REPO / "configs/hardware.nuc.yaml", quiet=True)
# These backbone fields are fixed in the verified ftA1500 payload.
frames_pix, fps = 13, 4.0
past = max(
    hw.wrist_ft.window_s,
    hw.control.chunk_horizon / hw.control.action_rate_hz,
    2 / hw.recording.field_ds_rate_hz,
)
future = (
    max(
        (frames_pix - 1) / fps,
        hw.control.chunk_horizon / hw.control.action_rate_hz,
        hw.derived.event_lookahead_s,
    )
    + 1 / hw.recording.field_ds_rate_hz
)
result = {
    "method": "current loader formula with saved ftA1500 backbone13frames/4fps; not a reconstructed historical optimizer manifest",
    "hardware_config_hash": hw.config_hash(),
    "past_s": past,
    "future_s": future,
    "action_rate_hz": hw.control.action_rate_hz,
    "horizon": hw.control.chunk_horizon,
    "field_ds_rate_hz": hw.recording.field_ds_rate_hz,
    "files": {
        p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
        for p in [
            "phantom/train/common.py",
            "phantom/data/windows.py",
            "configs/hardware.nuc.yaml",
        ]
    },
    "manifest_files": {},
    "episodes": [],
}
for p in [
    ROOT / "data/full/manifests/all.jsonl",
    ROOT / "data/val_eval/manifests/all.jsonl",
    ROOT / "data/val124/manifests/all.jsonl",
]:
    result["manifest_files"][str(p)] = {
        "exists": p.exists(),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,
    }
for path in sorted((ROOT / "data/full/tasks/waffles").iterdir()):
    if not (path / "actions.zarr").exists():
        continue
    try:
        groups = {
            k: zarr.open_group(str(path / f"{k}.zarr"), mode="r")
            for k in [
                "arm_q",
                "camera_scene_color",
                "arm_ft",
                "actions",
                "gripper",
                "tactile_left_fields_ds",
                "tactile_right_fields_ds",
            ]
        }
        ts = {k: np.asarray(g["ts"][:]) for k, g in groups.items()}
    except KeyError:
        continue
    t0 = float(ts["arm_q"][0])
    required = [
        "camera_scene_color",
        "arm_ft",
        "actions",
        "tactile_left_fields_ds",
        "tactile_right_fields_ds",
    ]
    lo = max(float(ts[k][0]) for k in required) + past
    hi = min(float(ts[k][-1]) for k in required) - future
    g = np.asarray(groups["gripper"]["data"][:, 0])
    ic = close_index(g)
    close = float(ts["gripper"][ic]) if ic is not None else None
    band = [max(lo, close - 1.5), min(hi, close - 0.2)] if close else None
    result["episodes"].append(
        {
            "episode": path.name,
            "t0_master": t0,
            "loader_anchor_valid_range_s": [lo - t0, hi - t0],
            "close_rule_t_s": close - t0 if close else None,
            "grasp_band_s": [x - t0 for x in band] if band else None,
            "stream_bounds_s": {
                k: [float(v[0] - t0), float(v[-1] - t0)] for k, v in ts.items()
            },
        }
    )
print(json.dumps(result, allow_nan=False))
