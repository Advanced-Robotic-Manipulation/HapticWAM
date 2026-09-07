"""Read-only native recording-start audit; emits compact JSON to stdout.

Run from compute3's live repository using its Python, by stdin. No source data
are changed. Stored stream timestamps already use MasterClock seconds.
"""

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import zarr

from phantom.data.episode_store import EpisodeReader

DATA = Path("/home/physicalai/phantom-icra-2027/data")
NUMERIC = ("arm_q", "arm_tcp_pose", "gripper", "arm_ft", "arm_qd")
manifests = {}
for label, relative in [
    ("original", "val_eval/manifests/all.jsonl"),
    ("expanded_validation", "val124/manifests/all.jsonl"),
]:
    path = DATA / relative
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    manifests[label] = {row["episode"]: row for row in rows}

result = {
    "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
    "data_root": str(DATA),
    "method": __doc__,
    "episodes": [],
}
for path in sorted((DATA / "full/tasks/waffles").iterdir()):
    if not path.is_dir():
        continue
    reader = EpisodeReader(path)
    meta_bytes = (path / "meta.json").read_bytes()
    meta = json.loads(meta_bytes)
    row = {
        "episode": path.name,
        "path": str(path),
        "meta_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "meta": meta,
        "streams": reader.streams(),
        "manifest_memberships": {
            label: rows.get(path.name) for label, rows in manifests.items()
        },
        "numeric": {},
        "errors": [],
    }
    arrays, stamps = {}, {}
    for name in NUMERIC:
        if not reader.has(name):
            continue
        try:
            group = zarr.open_group(str(path / f"{name}.zarr"), mode="r")
            data = np.asarray(group["data"][:])
            info = {"shape": list(data.shape), "first_data": data[0].tolist()}
            if "ts" not in group:
                info["timestamp_status"] = "missing"
            else:
                ts = np.asarray(group["ts"][:], dtype=float)
                info.update(
                    timestamp_status="native",
                    first_t=float(ts[0]),
                    last_t=float(ts[-1]),
                    samples=len(ts),
                    strict_monotonic=bool(np.all(np.diff(ts) > 0)),
                    median_dt_s=float(np.median(np.diff(ts))),
                    sample_payload_sha256=hashlib.sha256(
                        ts.tobytes() + data.tobytes()
                    ).hexdigest(),
                )
                arrays[name], stamps[name] = data, ts
            row["numeric"][name] = info
        except Exception as exc:
            row["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    required = ("arm_q", "arm_tcp_pose", "gripper", "arm_ft")
    if all(name in stamps for name in required):
        common_start = max(stamps[name][0] for name in required)
        iq = int(np.searchsorted(stamps["arm_q"], common_start))
        t = float(stamps["arm_q"][iq])
        samples = {}
        for name in required + ("arm_qd",):
            if name not in stamps:
                continue
            i = int(np.searchsorted(stamps[name], t, side="right") - 1)
            samples[name] = {
                "index": i,
                "t_master": float(stamps[name][i]),
                "age_s": float(t - stamps[name][i]),
                "value": arrays[name][i].tolist(),
            }
        first_q_t = float(stamps["arm_q"][0])
        initial = {
            "q": samples["arm_q"]["value"],
            "gripper": samples["gripper"]["value"][0],
            "wrist_ft": samples["arm_ft"]["value"],
            "measured_tcp_pose": samples["arm_tcp_pose"]["value"],
            "t_master": t,
            "source_samples": samples,
            "offset_from_first_arm_sample_s": t - first_q_t,
            "tcp_same_timestamp_as_q": samples["arm_tcp_pose"]["t_master"] == t,
        }
        diagnostics = {}
        for duration in (0.25, 0.5, 1.0):
            mask = (stamps["arm_q"] >= t) & (stamps["arm_q"] <= t + duration)
            q = arrays["arm_q"][mask]
            tcp_mask = (stamps["arm_tcp_pose"] >= t) & (
                stamps["arm_tcp_pose"] <= t + duration
            )
            tcp = arrays["arm_tcp_pose"][tcp_mask]
            ft_mask = (stamps["arm_ft"] >= t) & (stamps["arm_ft"] <= t + duration)
            ft = arrays["arm_ft"][ft_mask]
            d = {
                "arm_samples": len(q),
                "max_joint_change_rad": float(np.max(np.abs(q - q[0]))),
                "max_tcp_displacement_m": float(
                    np.max(np.linalg.norm(tcp[:, :3] - tcp[0, :3], axis=1))
                ),
                "wrist_force_norm_min_max_n": [
                    float(x)
                    for x in [
                        np.linalg.norm(ft[:, :3], axis=1).min(),
                        np.linalg.norm(ft[:, :3], axis=1).max(),
                    ]
                ],
                "wrist_ft_median": np.median(ft, axis=0).tolist(),
            }
            if "arm_qd" in arrays:
                m = (stamps["arm_qd"] >= t) & (stamps["arm_qd"] <= t + duration)
                d["max_measured_joint_speed_rad_s"] = float(
                    np.max(np.abs(arrays["arm_qd"][m]))
                )
            diagnostics[str(duration)] = d
        initial["start_diagnostics"] = diagnostics
        row["initial_state"] = initial
        if reader.has("camera_scene_color"):
            ts = reader.ts("camera_scene_color")
            frames = []
            for offset in (0.0, 0.5, 1.0):
                target = t + offset
                index = int(np.argmin(np.abs(ts - target)))
                rgb = np.asarray(reader.data("camera_scene_color")[index])
                ok, encoded = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 75],
                )
                assert ok
                frames.append(
                    {
                        "index": index,
                        "t_master": float(ts[index]),
                        "offset_from_anchor_s": float(ts[index] - t),
                        "shape": list(rgb.shape),
                        "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "jpeg_base64": base64.b64encode(encoded.tobytes()).decode(),
                    }
                )
            row["first_frames"] = frames
    result["episodes"].append(row)
print(json.dumps(result, allow_nan=False))
