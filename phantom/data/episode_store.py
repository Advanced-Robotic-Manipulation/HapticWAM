"""Zarr-backed episode store.

One directory per episode:
    <episode>/meta.json
    <episode>/<stream>.zarr/{data (T, ...), ts (T,)}

Append-only writer (the recorder drains ring buffers on an interval), lazy
reader (zarr arrays slice from disk). No torch here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from phantom.data.schema import EpisodeMeta

log = logging.getLogger(__name__)

_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


class EpisodeWriter:
    def __init__(self, path: Path, hw, meta: EpisodeMeta):
        self.path = Path(path)
        self.hw = hw
        self.meta = meta
        self.path.mkdir(parents=True, exist_ok=True)
        # provenance: WindowSampler warns when data was recorded under
        # different shape-relevant config values
        try:
            meta.config_hash = hw.config_hash()
            meta.hardware_shapes = dict(hw.shape_relevant_fields())
        except Exception:  # tolerate partial configs in tests
            pass
        meta.status = "recording"
        meta.save(self.path / "meta.json")
        self._groups: dict[str, zarr.Group] = {}
        self._chunk = int(getattr(hw.recording, "zarr_chunk_frames", 120))

    # ------------------------------------------------------------------
    def append(self, stream: str, ts: np.ndarray, data: np.ndarray) -> None:
        """Append T rows to a stream (creates it on first append)."""
        ts = np.asarray(ts, dtype=np.float64).reshape(-1)
        data = np.asarray(data)
        if len(ts) == 0:
            return
        assert data.shape[0] == len(ts), f"{stream}: {data.shape[0]} rows vs {len(ts)} ts"
        g = self._groups.get(stream)
        if g is None:
            g = zarr.open_group(str(self.path / f"{stream}.zarr"), mode="a")
            if "data" not in g:
                g.create_dataset(
                    "data", shape=(0, *data.shape[1:]), dtype=data.dtype,
                    chunks=(self._chunk, *data.shape[1:]), compressor=_COMPRESSOR)
                g.create_dataset("ts", shape=(0,), dtype=np.float64,
                                 chunks=(self._chunk,), compressor=_COMPRESSOR)
            self._groups[stream] = g
        g["data"].append(data)
        g["ts"].append(ts)

    # ------------------------------------------------------------------
    def finalize(self, *, success: bool | None = None, notes: str = "") -> None:
        if success is not None:
            self.meta.success = success
        if notes:
            self.meta.notes = notes
        self.meta.status = "finalized"
        self.meta.save(self.path / "meta.json")

    def abort(self) -> None:
        """Mark aborted. Data is kept on disk (never silently deleted) —
        list_episodes() skips non-finalized episodes by default."""
        self.meta.status = "aborted"
        self.meta.save(self.path / "meta.json")


class EpisodeReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._groups: dict[str, zarr.Group] = {}
        self._meta: EpisodeMeta | None = None

    @property
    def meta(self) -> EpisodeMeta:
        if self._meta is None:
            self._meta = EpisodeMeta.load(self.path / "meta.json")
        return self._meta

    # ------------------------------------------------------------------
    def streams(self) -> list[str]:
        return sorted(p.name[:-5] for p in self.path.iterdir()
                      if p.is_dir() and p.name.endswith(".zarr"))

    def has(self, stream: str) -> bool:
        return (self.path / f"{stream}.zarr").is_dir()

    def _g(self, stream: str) -> zarr.Group:
        g = self._groups.get(stream)
        if g is None:
            g = zarr.open_group(str(self.path / f"{stream}.zarr"), mode="r")
            self._groups[stream] = g
        return g

    def ts(self, stream: str) -> np.ndarray:
        return np.asarray(self._g(stream)["ts"][:], dtype=np.float64)

    def n(self, stream: str) -> int:
        return int(self._g(stream)["data"].shape[0])


def list_episodes(root: Path, *, include_unfinalized: bool = False) -> list[Path]:
    """Sorted ep_* directories (recursive) containing meta.json. Aborted /
    still-recording episodes are skipped unless include_unfinalized."""
    root = Path(root)
    out = []
    for meta_path in sorted(root.rglob("meta.json")):
        ep = meta_path.parent
        if not ep.name.startswith("ep_"):
            continue
        if not include_unfinalized:
            try:
                if EpisodeMeta.load(meta_path).status == "aborted":
                    continue
            except Exception:
                log.warning("unreadable meta.json in %s — skipping", ep)
                continue
        out.append(ep)
    return out
