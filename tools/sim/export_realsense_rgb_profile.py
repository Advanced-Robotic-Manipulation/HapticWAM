#!/usr/bin/env python3
"""Read a connected RealSense's available RGB profile without starting streams.

Run with the rig's existing pyrealsense2 Python environment. This does not
open/close sensors, start/stop pipelines, change options, or import rig drivers.
The result describes an AVAILABLE profile, not an actively negotiated stream.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import sys


OPTION_NAMES = (
    "enable_auto_exposure", "enable_auto_white_balance", "exposure", "gain",
    "white_balance", "brightness", "contrast", "gamma", "hue", "saturation",
    "sharpness", "backlight_compensation", "power_line_frequency",
    "auto_exposure_priority",
)


def read_info(owner, rs, name):
    key = getattr(rs.camera_info, name, None)
    try:
        return owner.get_info(key) if key is not None and owner.supports(key) else None
    except RuntimeError:
        return None


def read_options(sensor, rs):
    result = {}
    for name in OPTION_NAMES:
        option = getattr(rs.option, name, None)
        item = result[name] = {"value": None, "supported": False}
        try:
            if option is None or not sensor.supports(option):
                continue
            item["supported"] = True
            item["value"] = float(sensor.get_option(option))
        except RuntimeError as exc:
            item["read_error"] = str(exc)
    return result


def profile_fov(intrinsics, rs):
    """SDK pinhole FOV; distortion is reported separately, not removed here."""
    if hasattr(rs, "rs2_fov"):
        return [float(x) for x in rs.rs2_fov(intrinsics)], "pyrealsense2.rs2_fov"
    # Same half-pixel boundary convention as librealsense src/rs.cpp rs2_fov.
    fov = [
        math.degrees(math.atan2(center + 0.5, focal)
                     + math.atan2(size - (center + 0.5), focal))
        for center, size, focal in (
            (intrinsics.ppx, intrinsics.width, intrinsics.fx),
            (intrinsics.ppy, intrinsics.height, intrinsics.fy),
        )
    ]
    return fov, "librealsense_rs2_fov_equivalent_pinhole_formula"


def export_profile(rs, serial="944622074411", width=640, height=480, fps=15):
    if min(width, height, fps) <= 0:
        raise ValueError("Width, height, and fps must be positive")
    context = rs.context()
    devices = list(context.query_devices())
    serials = [read_info(device, rs, "serial_number") for device in devices]
    selected = [device for device, found in zip(devices, serials) if found == serial]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one RealSense with serial {serial}; found {len(selected)}. "
            f"Visible serials: {serials}. No sensor or stream was opened."
        )
    device = selected[0]
    matches, available = [], set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            video = profile.as_video_stream_profile()
            available.add(f"{video.width()}x{video.height()}@{profile.fps()} {profile.format()}")
            if (profile.format() == rs.format.rgb8 and video.width() == width
                    and video.height() == height and profile.fps() == fps):
                matches.append((sensor, profile, video))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one exact color RGB8 {width}x{height}@{fps} profile; "
            f"found {len(matches)} for serial {serial}. "
            f"Available color profiles: {sorted(available)}. No fallback selected."
        )
    sensor, profile, video = matches[0]
    intrinsic = video.get_intrinsics()
    fov, fov_method = profile_fov(intrinsic, rs)
    return {
        "schema_version": 1,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "sdk_version": getattr(rs, "__version__", None),
        "provenance": {
            "status": "available_profile_query_not_proof_of_actively_negotiated_running_stream",
            "intrinsics_source": "connected_device_video_stream_profile.get_intrinsics",
            "options_source": "sensor.get_option_at_query_time_not_recording_time_or_factory_defaults",
            "streams_started": False,
            "sensor_options_changed": False,
        },
        "device": {name: read_info(device, rs, name) for name in (
            "name", "serial_number", "firmware_version", "product_id", "usb_type_descriptor",
        )},
        "sensor_name": read_info(sensor, rs, "name"),
        "profile": {
            "stream": "color", "format": "RGB8", "width": video.width(),
            "height": video.height(), "fps": profile.fps(),
            "stream_index": profile.stream_index(),
        },
        "intrinsics": {
            "width": intrinsic.width, "height": intrinsic.height,
            "fx": intrinsic.fx, "fy": intrinsic.fy,
            "ppx": intrinsic.ppx, "ppy": intrinsic.ppy,
            "K": [[intrinsic.fx, 0, intrinsic.ppx],
                  [0, intrinsic.fy, intrinsic.ppy], [0, 0, 1]],
            "distortion_model": str(intrinsic.model),
            "distortion_coefficients": list(intrinsic.coeffs),
        },
        "fov_degrees": {"horizontal": fov[0], "vertical": fov[1],
                        "method": fov_method, "distortion_applied": False},
        "options": read_options(sensor, rs),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="944622074411")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", type=Path, help="New JSON file; otherwise print to stdout")
    args = parser.parse_args(argv)
    try:
        import pyrealsense2 as rs
    except ImportError:
        parser.exit(2, "pyrealsense2 is unavailable. Use the rig's existing camera Python environment; no dependencies were installed.\n")
    try:
        result = export_profile(rs, args.serial, args.width, args.height, args.fps)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(content)
        else:
            sys.stdout.write(content)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"RGB profile export failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
