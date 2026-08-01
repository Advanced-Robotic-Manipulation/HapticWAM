"""Offline tactile-field recompute from archived raw frames (bench item (e)).

Rebuilds fields_ds / keyframes / wrench / area streams for every episode that
has a tactile_<sensor>_raw_img stream, using the vendor SDK's documented
offline path (docs/sensor_sdk.md):

    sensor.setBaseFrame(first_raw)                      # re-reference
    img2d, deform, depth, shear = sensor.process(raw, getdepth=True,
                                                 getshear=True, getforce=True,
                                                 getdistforce=True)

RIG-ONLY: needs the `dmrobotics` SDK (py3.8-3.11, linux/windows). Written
against SDK_Publish_1.2.10 as documented; the exact `process()` return arity
with force outputs enabled is a BENCH (e) item — verify on the rig, then also
check bit-parity against a live-recorded fields_ds stretch before flipping
`recording.archive_raw_img: true` for real collection.

    python -m phantom.scripts.recompute_fields --data <episodes_root>
        [--hardware configs/hardware.yaml] [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from phantom.config.hardware import load_hardware
from phantom.data.episode_store import EpisodeReader, EpisodeWriter, list_episodes
from phantom.data.schema import tactile_stream

log = logging.getLogger("recompute_fields")


def _pool_ds(stack: np.ndarray, hw_out: tuple[int, int]) -> np.ndarray:
    H, W, C = stack.shape
    h, w = hw_out
    return stack.reshape(h, H // h, w, W // w, C).mean(axis=(1, 3))


def recompute_episode(ep: Path, hw, *, overwrite: bool = False) -> bool:
    from dmrobotics import Sensor, SensorOptions  # rig-only
    from dmrobotics.src.dmSDK import DMTacImage

    r = EpisodeReader(ep)
    did = False
    for s_idx, s in enumerate(hw.tactile.sensors):
        raw_stream = tactile_stream(s.name, "raw_img")
        out_stream = tactile_stream(s.name, "fields_ds")
        if not r.has(raw_stream):
            continue
        if r.has(out_stream) and not overwrite:
            log.info("%s/%s: fields exist; skipping (use --overwrite)", ep.name, s.name)
            continue
        serial = s.dev_id if isinstance(s.dev_id, str) else None
        sdk = Sensor(SensorOptions(dev_id=s.dev_id, backend=hw.tactile.sdk_backend))
        try:
            g = r._g(raw_stream)
            ts = np.asarray(g["ts"][:])
            raws = g["data"]
            first = DMTacImage(img=np.asarray(raws[0]), serial=serial or sdk.getDevID())
            sdk.setBaseFrame(first)

            writer = EpisodeWriter(ep, hw, r.meta)  # appends into the same dir
            f_dtype = np.float16 if hw.recording.field_dtype == "float16" else np.float32
            kf_every = max(1, round(hw.recording.field_ds_rate_hz
                                    / hw.recording.keyframe_rate_hz))
            ds_rows, wrench_rows, area_rows, kf_ts, kf_rows = [], [], [], [], []
            for k in range(raws.shape[0]):
                raw = DMTacImage(img=np.asarray(raws[k]), serial=first.serial)
                out = sdk.process(raw, getdepth=True, getshear=True,
                                  getforce=True, getdistforce=True)
                # documented base return: img2d, deform, depth, shear; force
                # outputs extend the tuple — BENCH (e) pins the exact arity
                img2d, deform, depth, shear = out[:4]
                extras = list(out[4:])
                dist = next((e for e in extras
                             if getattr(e, "ndim", 0) == 3), None)
                force = next((e for e in extras
                              if getattr(e, "size", 0) == 6), None)
                if dist is None:
                    raise RuntimeError(
                        "process() did not return a distributed-force field — "
                        "record the actual return signature (bench item e) and "
                        "update this script")
                stack = np.concatenate(
                    [deform, depth[..., None], shear, dist], axis=-1,
                    dtype=np.float32)
                ds_rows.append(_pool_ds(stack, hw.recording.field_ds.hw)
                               .astype(f_dtype))
                wrench_rows.append(np.zeros(6, np.float32) if force is None
                                   else np.asarray(force, np.float32).reshape(-1))
                area_rows.append(np.float32((np.abs(depth)
                                             > hw.derived.tau_contact_depth).mean()))
                if k % kf_every == 0:
                    kf_ts.append(ts[k])
                    kf_rows.append(_pool_ds(stack, hw.recording.keyframe_ds.hw)
                                   .astype(f_dtype))
            writer.append(out_stream, ts, np.stack(ds_rows))
            writer.append(tactile_stream(s.name, "wrench"), ts,
                          np.stack(wrench_rows))
            writer.append(tactile_stream(s.name, "area"), ts,
                          np.asarray(area_rows, np.float32))
            writer.append(tactile_stream(s.name, "keyframes"),
                          np.asarray(kf_ts), np.stack(kf_rows))
            did = True
            log.info("%s/%s: %d frames recomputed", ep.name, s.name, len(ts))
        finally:
            sdk.disconnect()
    return did


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    hw = load_hardware(args.hardware)
    n = 0
    for ep in list_episodes(Path(args.data)):
        n += bool(recompute_episode(ep, hw, overwrite=args.overwrite))
    log.info("recomputed %d episodes", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
