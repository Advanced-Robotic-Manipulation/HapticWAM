"""Offline derived-channel pass: computes mask/CoP/slip/events at field rate
from a recorded episode and writes them as derived_* streams.

Runs offline (not in the recording hot loop) so it is RECOMPUTABLE when the
derived thresholds change after the hardware bench — just rerun this pass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from phantom.config.hardware import HardwareConfig
from phantom.data import derived as dv
from phantom.data.episode_store import EpisodeReader, list_episodes
from phantom.data.schema import tactile_stream

log = logging.getLogger(__name__)
_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def postprocess_episode(ep: Path, hw: HardwareConfig, *, overwrite: bool = False) -> None:
    r = EpisodeReader(ep)
    for s in hw.tactile.sensors:
        out_stream = tactile_stream(s.name, "mask_frac")
        if r.has(out_stream) and not overwrite:
            continue
        src = tactile_stream(s.name, "fields_ds")
        if not r.has(src):
            log.warning("%s: no %s stream; skipping", ep.name, src)
            continue
        g = r._g(src)
        ts = g["ts"][:]
        fields = g["data"]      # lazy zarr array (T, h, w, 8)
        T = len(ts)
        mask_frac = np.zeros(T, dtype=np.float32)
        cop = np.zeros((T, 2), dtype=np.float32)
        slip = np.zeros(T, dtype=np.float32)
        chunk = hw.recording.zarr_chunk_frames
        prev_frame = None
        for lo in range(0, T, chunk):
            hi = min(lo + chunk, T)
            block = np.asarray(fields[lo:hi], dtype=np.float32)
            for i in range(block.shape[0]):
                k = lo + i
                dt = float(ts[k] - ts[k - 1]) if k > 0 else 1.0 / hw.recording.field_ds_rate_hz
                d = dv.derive_timestep(block[i], prev_frame, dt, hw)
                mask_frac[k] = d["mask_frac"]
                cop[k] = np.nan_to_num(d["cop"], nan=0.0)
                slip[k] = d["slip"]
                prev_frame = block[i]
        events = dv.event_labels(mask_frac, slip, hw.derived)

        for name, arr in (("mask_frac", mask_frac), ("cop", cop),
                          ("slip", slip), ("events", events)):
            path = ep / f"{tactile_stream(s.name, name)}.zarr"
            grp = zarr.open_group(str(path), mode="w")
            grp.create_dataset("data", data=arr, chunks=(chunk, *arr.shape[1:]),
                               compressor=_COMPRESSOR)
            grp.create_dataset("ts", data=ts, chunks=(chunk,))
    log.info("postprocessed %s", ep.name)


def postprocess_root(root: Path, hw: HardwareConfig, *, overwrite: bool = False) -> int:
    eps = list_episodes(root)
    for ep in eps:
        postprocess_episode(ep, hw, overwrite=overwrite)
    return len(eps)
