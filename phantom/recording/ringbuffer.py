"""SharedRingBuffer: single-writer / multi-reader ring over
multiprocessing.shared_memory — the transport between driver workers and the
recorder/planner.

Layout per buffer: one shm block holding
    [write_index int64][ts float64 x cap][field arrays ...]
Readers snapshot by index (lock-free: the writer bumps write_index AFTER the
row is fully written; a torn read of the newest row is prevented by reading
index first and never touching rows >= index).

Windows note: shared memory is freed when the LAST handle closes — the owner
process must keep its handle alive for the buffer's lifetime (Rig/Session
owns it). unlink() is a no-op on Windows but kept for Linux hygiene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

_HEADER_DTYPE = np.int64
_TS_DTYPE = np.float64


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def itemsize(self) -> int:
        return int(np.dtype(self.dtype).itemsize * math.prod(self.shape or (1,)))


class SharedRingBuffer:
    def __init__(self, name: str, capacity: int, fields: dict[str, tuple[tuple[int, ...], str]],
                 *, create: bool):
        self.name = name
        self.capacity = int(capacity)
        self.specs = [FieldSpec(k, tuple(s), d) for k, (s, d) in fields.items()]
        size = 8 + self.capacity * 8
        offsets = {}
        for spec in self.specs:
            offsets[spec.name] = size
            size += self.capacity * spec.itemsize
        self._shm = shared_memory.SharedMemory(name=name, create=create, size=size)
        self._owner = create
        buf = self._shm.buf
        self._idx = np.ndarray((1,), dtype=_HEADER_DTYPE, buffer=buf, offset=0)
        if create:
            self._idx[0] = 0
        self._ts = np.ndarray((self.capacity,), dtype=_TS_DTYPE, buffer=buf, offset=8)
        self._arrays: dict[str, np.ndarray] = {}
        for spec in self.specs:
            self._arrays[spec.name] = np.ndarray(
                (self.capacity, *spec.shape), dtype=spec.dtype,
                buffer=buf, offset=offsets[spec.name])

    # ------------------------------------------------------------------
    @property
    def write_index(self) -> int:
        return int(self._idx[0])

    def push(self, ts: float, **values: np.ndarray) -> None:
        """Single-writer only."""
        i = self.write_index
        row = i % self.capacity
        for name, arr in self._arrays.items():
            arr[row] = values[name]
        self._ts[row] = ts
        self._idx[0] = i + 1   # publish AFTER the row is complete

    # ------------------------------------------------------------------
    def _rows(self, lo: int, hi: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        idx = np.arange(lo, hi) % self.capacity
        return self._ts[idx].copy(), {k: a[idx].copy() for k, a in self._arrays.items()}

    def latest(self, n: int = 1) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        hi = self.write_index
        lo = max(0, hi - min(n, self.capacity))
        return self._rows(lo, hi)

    def latest_ts(self) -> float | None:
        """Timestamp of the newest row WITHOUT copying its payload (None if the
        ring is still empty). latest(1) on a camera ring copies ~1 MB of pixels;
        the safety loop runs at executor rate and only needs the age."""
        hi = self.write_index
        if hi == 0:
            return None
        return float(self._ts[(hi - 1) % self.capacity])

    def drain(self, since_seq: int) -> tuple[int, np.ndarray, dict[str, np.ndarray]]:
        """All rows with seq >= since_seq still in the ring -> (next_seq, ts, data).
        If the reader fell behind (overwritten rows), it silently resumes from
        the oldest available row — the recorder watchdog reports the gap."""
        hi = self.write_index
        lo = max(since_seq, hi - self.capacity)
        ts, data = self._rows(lo, hi)
        return hi, ts, data

    def window(self, t0: float, t1: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        ts, data = self.latest(self.capacity)
        sel = (ts >= t0) & (ts <= t1)
        return ts[sel], {k: v[sel] for k, v in data.items()}

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._arrays.clear()
        self._ts = self._idx = None  # release buffer views before closing shm
        self._shm.close()
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def spec_dict(self) -> dict:
        """Everything a child process needs to attach (pickle-safe)."""
        return {"name": self.name, "capacity": self.capacity,
                "fields": {s.name: (s.shape, s.dtype) for s in self.specs}}

    @classmethod
    def attach(cls, spec: dict) -> "SharedRingBuffer":
        return cls(spec["name"], spec["capacity"], spec["fields"], create=False)
