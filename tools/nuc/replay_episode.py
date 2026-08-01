#!/usr/bin/env python3
"""Replay a recorded PHANTOM episode in a local rerun viewer.

Standalone -- does NOT touch the live collection pipeline or its viewer
(own app id, own port 9899, own capped viewer window).

Usage:
  replay_episode.py EP_DIR                 # open viewer, scrub the episode
  replay_episode.py SESSION_DIR            # list episodes in the session
  replay_episode.py SESSION_DIR --ep N     # replay episode with index N
  replay_episode.py EP_DIR --save out.rrd  # no window, write an .rrd file

Streams shown: scene camera, both tactile infer images, tactile wrench +
contact area, arm F/T, TCP speed, gripper, actions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr
import rerun as rr

SCALARS = getattr(rr, "Scalars", None) or getattr(rr, "Scalar")
PORT = 9899          # far from the live pipeline's 9878 proxy
MEM = "2GB"


def set_t(ts: float) -> None:
    try:
        rr.set_time("t", timestamp=ts)
    except TypeError:  # older sdk
        rr.set_time_seconds("t", ts)


def load(ep: Path, name: str):
    p = ep / f"{name}.zarr"
    if not p.is_dir():
        return None, None
    g = zarr.open(str(p), mode="r")
    return g["data"], np.asarray(g["ts"])


def log_scalars(ep: Path, stream: str, entity: str, labels, stride: int = 1) -> None:
    data, ts = load(ep, stream)
    if data is None:
        return
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, None]
    for i in range(0, len(ts), stride):
        set_t(float(ts[i]))
        for j, lab in enumerate(labels[: arr.shape[1]]):
            rr.log(f"{entity}/{lab}", SCALARS(float(arr[i, j])))


def log_images(ep: Path, stream: str, entity: str, stride: int = 1) -> None:
    data, ts = load(ep, stream)
    if data is None:
        return
    for i in range(0, len(ts), stride):
        set_t(float(ts[i]))
        frame = data[i]
        if isinstance(frame, (bytes, bytearray)):
            rr.log(entity, rr.EncodedImage(contents=bytes(frame), media_type="image/jpeg"))
        else:
            a = np.asarray(frame)
            if a.ndim == 1 and a.dtype == np.uint8:   # encoded bytes as array
                rr.log(entity, rr.EncodedImage(contents=a.tobytes(), media_type="image/jpeg"))
            else:
                rr.log(entity, rr.Image(a))


def replay(ep: Path, save: str | None) -> None:
    meta = json.loads((ep / "meta.json").read_text())
    print(f"episode: {ep.name}")
    print(f"  task={meta.get('task')} success={meta.get('success')} "
          f"status={meta.get('status')} operator={meta.get('operator')}")

    rr.init(f"phantom_replay/{ep.name}", spawn=False)
    if save:
        rr.save(save)
    else:
        rr.spawn(port=PORT, memory_limit=MEM)

    rr.log("meta", rr.TextDocument(json.dumps(meta, indent=2)), static=True)

    log_images(ep, "camera_scene_color", "camera/scene")
    for side in ("left", "right"):
        log_images(ep, f"tactile_{side}_infer_img", f"tactile/{side}/img")
        log_scalars(ep, f"tactile_{side}_wrench", f"tactile/{side}/wrench",
                    ["fx", "fy", "fz", "tx", "ty", "tz"])
        log_scalars(ep, f"tactile_{side}_area", f"tactile/{side}", ["area_mm2"])
    log_scalars(ep, "arm_ft", "arm/ft", ["fx", "fy", "fz", "tx", "ty", "tz"], stride=5)
    log_scalars(ep, "arm_tcp_speed", "arm/tcp_speed",
                ["vx", "vy", "vz", "wx", "wy", "wz"], stride=5)
    log_scalars(ep, "gripper", "gripper", ["pos", "force"], stride=3)
    log_scalars(ep, "actions", "actions", ["dx", "dy", "dz", "rx", "ry", "rz", "grip"])
    print("done." + ("" if save else " Viewer window is up -- scrub the 't' timeline."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--ep", type=int, default=None, help="episode index within a session dir")
    ap.add_argument("--save", default=None, help="write .rrd instead of opening a viewer")
    args = ap.parse_args()

    p = args.path
    if not p.is_dir():
        sys.exit(f"not a directory: {p}")
    if (p / "meta.json").exists():
        replay(p, args.save)
        return 0
    eps = sorted(p.glob("ep_*"))
    if not eps:
        sys.exit(f"no ep_* inside {p}")
    if args.ep is None:
        print(f"{len(eps)} episodes in {p.name}: (pick with --ep N)")
        for e in eps:
            try:
                m = json.loads((e / "meta.json").read_text())
                print(f"  {e.name}  success={m.get('success')} status={m.get('status')}")
            except Exception:  # noqa: BLE001
                print(f"  {e.name}  (meta unreadable)")
        return 0
    matches = [e for e in eps if e.name.endswith(f"_{args.ep:03d}")]
    if not matches:
        sys.exit(f"no episode with index {args.ep:03d}")
    replay(matches[0], args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
