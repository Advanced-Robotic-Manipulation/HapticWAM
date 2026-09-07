#!/usr/bin/env python3
"""Extract one existing rendered frame per completed case; JSON to stdout."""

import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(
    "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_robustness_v2_delivery/screen"
)


def main():
    progress = json.loads((ROOT / "progress.json").read_text())
    frames = []
    for policy in [
        "fta1500_nfe1_k4",
        "fta3000_nfe1_k4",
        "v5_6_nfe1_k4",
        "fta1500_nfe5_k1",
    ]:
        for seed in [903101, 903102]:
            case = f"{policy}__start_1787395928__seed{seed}"
            assert progress["trials"][case]["status"] == "completed"
            folder = ROOT / "rollouts" / case
            executions = [
                json.loads(line)
                for line in (folder / "execution_trace.jsonl").read_text().splitlines()
                if line
            ]
            stop = next((row for row in executions if row["stopped"]), None)
            z = np.load(folder / "sim_trace.npz")
            if stop:
                event_t, event = stop["t"], "first safety stop"
            else:
                initial = float(z["gripper"][0, 0])
                event_t = next(
                    row["t"]
                    for row in executions
                    if row["gripper_command"] > initial + 2 / 255
                )
                event = "first additional closure"
            index = int(np.searchsorted(z["t"], event_t, side="right") - 1)
            cap = cv2.VideoCapture(str(folder / "sim.mp4"))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frame_count == len(z["t"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            cap.release()
            assert ok
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            assert ok
            frames.append(
                {
                    "case_id": case,
                    "event": event,
                    "event_time_s": event_t,
                    "frame_index": index,
                    "frame_time_s": float(z["t"][index]),
                    "source_video": str(folder / "sim.mp4"),
                    "source_frame_count": frame_count,
                    "source_video_sha256": hashlib.sha256(
                        (folder / "sim.mp4").read_bytes()
                    ).hexdigest(),
                    "jpeg_sha256": hashlib.sha256(encoded).hexdigest(),
                    "jpeg_base64": base64.b64encode(encoded).decode(),
                }
            )
    print(json.dumps({"frames": frames}))


if __name__ == "__main__":
    main()
