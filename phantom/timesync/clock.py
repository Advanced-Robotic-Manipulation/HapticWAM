"""MasterClock: one timebase for every stream (pipeline.md §1: 'everything is
timestamped on one clock (RTDE master)').

Master time = the UR controller's RTDE timestamp. Calibration takes N paired
(t_host around a receive call, t_rtde) samples; offset = median of
(t_rtde - t_host) over the lowest-latency decile. Drift is re-estimated with a
slow EWMA; jumps > jump_warn_s are logged.

In mock mode (MockArm reports t_rtde = perf_counter - t0), the calibration
degenerates gracefully to a constant offset.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from phantom.drivers.base import Arm

log = logging.getLogger(__name__)


class MasterClock:
    def __init__(self, offset: float = 0.0):
        self.offset = offset            # t_master = t_host + offset
        self._ewma_alpha = 0.05
        self._jump_warn_s = 0.005
        self.calibration: dict = {"offset": offset, "n_samples": 0, "spread_s": 0.0}

    # ------------------------------------------------------------------
    @classmethod
    def calibrate(cls, arm: Arm, n_samples: int = 200) -> "MasterClock":
        pairs = []
        for _ in range(n_samples):
            t_before = time.perf_counter()
            st = arm.get_state()
            t_after = time.perf_counter()
            latency = t_after - t_before
            t_host_mid = 0.5 * (t_before + t_after)
            pairs.append((latency, st.t_rtde - t_host_mid))
        pairs.sort(key=lambda p: p[0])
        best = pairs[:max(1, n_samples // 10)]          # lowest-latency decile
        offsets = np.array([p[1] for p in best])
        clock = cls(offset=float(np.median(offsets)))
        clock.calibration = {"offset": clock.offset, "n_samples": n_samples,
                             "spread_s": float(offsets.std())}
        log.info("MasterClock calibrated: offset=%.6f s, spread=%.2e s",
                 clock.offset, offsets.std())
        return clock

    # ------------------------------------------------------------------
    def host_to_master(self, t_host: float) -> float:
        return t_host + self.offset

    def now_master(self) -> float:
        return time.perf_counter() + self.offset

    def update(self, t_host: float, t_rtde: float) -> None:
        """Slow drift tracking from an ongoing (t_host, t_rtde) pair stream."""
        new_offset = t_rtde - t_host
        delta = new_offset - self.offset
        if abs(delta) > self._jump_warn_s:
            log.warning("MasterClock jump: %.3f ms", delta * 1e3)
        self.offset += self._ewma_alpha * delta


class IdentityClock(MasterClock):
    """t_master == t_host (mock rigs, single-machine tests)."""

    def __init__(self):
        super().__init__(offset=0.0)

    def update(self, t_host: float, t_rtde: float) -> None:
        pass
