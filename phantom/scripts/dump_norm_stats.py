"""Compute per-channel normalization statistics over an episode root and write
norm_stats.json beside the data (consumed by WindowSampler and the training
programs).

    python -m phantom.scripts.dump_norm_stats --data <episodes_root> [--samples 200]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from phantom.config.hardware import load_hardware
from phantom.data.episode_store import EpisodeReader, list_episodes
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ARM_FT, STREAM_ARM_Q,
                                 STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                                 STREAM_ARM_TCP_SPEED, STREAM_GRIPPER, NormStats,
                                 tactile_stream)

log = logging.getLogger("norm_stats")


def _acc(sums: dict, key: str, arr: np.ndarray, axis_keep: int = -1) -> None:
    """Accumulate channel-wise moments (channels = last axis, or scalar)."""
    a = np.asarray(arr, dtype=np.float64)
    flat = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(-1, 1)
    s = sums.setdefault(key, [0.0, 0.0, 0])
    s[0] = s[0] + flat.sum(0)
    s[1] = s[1] + (flat ** 2).sum(0)
    s[2] = s[2] + flat.shape[0]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--samples", type=int, default=200, help="frames sampled per stream")
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    root = Path(args.data)
    sums: dict = {}
    rng = np.random.default_rng(0)

    for ep in list_episodes(root):
        r = EpisodeReader(ep)
        for s in hw.tactile.sensors:
            for kind, key in (("fields_ds", "fields"), ("keyframes", "fields"),
                              ("wrench", "wrench")):
                stream = tactile_stream(s.name, kind)
                if not r.has(stream):
                    continue
                n = r.n(stream)
                idx = sorted(rng.choice(n, size=min(args.samples, n), replace=False).tolist())
                _acc(sums, key, np.asarray(r._g(stream)["data"][idx], dtype=np.float32))
            stream = tactile_stream(s.name, "area")
            if r.has(stream):
                _acc(sums, "area", np.asarray(r._g(stream)["data"][:], dtype=np.float32)[..., None])
        _acc(sums, "wrist_ft", r._g(STREAM_ARM_FT)["data"][:])
        # ur_state is a channel-wise concat of streams recorded at DIFFERENT
        # rates (gripper ~100 Hz vs arm ~140 Hz), so truncating to the shortest
        # would drop the tail of every episode (~25%) from the statistics and
        # bias them toward early-episode poses. Per-channel moments do not need
        # the streams aligned — accumulate each stream over its full length and
        # concatenate the resulting per-channel stats in the same order.
        for key in (STREAM_ARM_Q, STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                    STREAM_ARM_TCP_SPEED, STREAM_GRIPPER):
            _acc(sums, f"ur_state::{key}", np.asarray(r._g(key)["data"][:]))
        if r.has(STREAM_ACTIONS):
            _acc(sums, "action", r._g(STREAM_ACTIONS)["data"][:])

    stats = NormStats()
    for key, (s1, s2, n) in sums.items():
        mean = s1 / n
        var = np.maximum(s2 / n - mean ** 2, 1e-12)
        stats.mean[key] = mean.astype(np.float32)
        stats.std[key] = np.sqrt(var).astype(np.float32)
    # recombine the per-stream ur_state moments into one channel vector, in the
    # exact concat order WindowSampler packs them
    ur_keys = [f"ur_state::{k}" for k in
               (STREAM_ARM_Q, STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                STREAM_ARM_TCP_SPEED, STREAM_GRIPPER)]
    if all(k in stats.mean for k in ur_keys):
        stats.mean["ur_state"] = np.concatenate(
            [stats.mean.pop(k) for k in ur_keys]).astype(np.float32)
        stats.std["ur_state"] = np.concatenate(
            [stats.std.pop(k) for k in ur_keys]).astype(np.float32)
    # delta-field stats approximated from the field stats (deltas are small):
    if "fields" in stats.mean:
        std = stats.std["fields"]
        # channel order: [disp_x, disp_y, depth, shear_x, shear_y, fx, fy, fz]
        stats.mean["cpk_d_disp"] = np.zeros(3, dtype=np.float32)
        stats.std["cpk_d_disp"] = np.array([std[0], std[1], std[2]]) * 0.5
        stats.mean["cpk_d_fz"] = np.zeros(1, dtype=np.float32)
        stats.std["cpk_d_fz"] = np.array([std[7]]) * 0.5

    out = root / "norm_stats.json"
    stats.save(out)
    log.info("wrote %s (keys: %s)", out, sorted(stats.mean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
