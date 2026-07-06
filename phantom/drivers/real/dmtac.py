"""Real DM-Tac W2L driver over the `dmrobotics` Python SDK.

Boundary rules (so downstream code never thinks about SDK quirks):
 - hw_order transpose happens HERE: downstream always sees (field.h, field.w);
 - wrench is scaled by force_unit_to_N HERE (left in SDK units while the
   0.0 UNCALIBRATED sentinel is set — consumers check tactile.force_calibrated);
 - one read() = one coherent grab of all field getters.

`dmrobotics` is imported lazily inside connect() so this module imports fine
on machines without the SDK.

BENCH day-1 (docs/hardware_bench_day1.md) verifies: getForce/getDistributeForce
units (a), image channel counts + true concurrent rates (b), numpy (H, W)
ordering (c), sustained throughput (d), offline recompute (e).
"""

from __future__ import annotations

import time

import numpy as np

from phantom.config.hardware import TactileConfig, TactileSensorEntry
from phantom.drivers.base import TactileFrame, TactileSensor


class DmTacSensor(TactileSensor):
    def __init__(self, cfg: TactileConfig, sensor: TactileSensorEntry):
        super().__init__(cfg, sensor)
        self._sdk = None
        self._seq = 0
        self._last_read_t = 0.0

    def connect(self) -> None:
        try:
            from dmrobotics import Sensor, SensorOptions  # lazy: rig machines only
        except ImportError as e:
            raise RuntimeError(
                "dmrobotics SDK not installed — install it on the rig machine, or set "
                "mode.overrides.tactile: mock in configs/hardware.yaml") from e
        self._sdk = Sensor(SensorOptions(self.sensor.dev_id))
        self.reset_reference()
        self._seq = 0

    def disconnect(self) -> None:
        sdk, self._sdk = self._sdk, None
        if sdk is not None and hasattr(sdk, "release"):
            sdk.release()

    def reset_reference(self) -> None:
        if self._sdk is None:
            raise RuntimeError("reset_reference() before connect()")
        self._sdk.getBaseFrame()

    # ------------------------------------------------------------------
    def _canon(self, arr: np.ndarray) -> np.ndarray:
        """Transpose SDK field arrays into canonical (field.h, field.w) order."""
        arr = np.asarray(arr)
        if self.cfg.hw_order == "wh":
            # SDK axis0 is the field.w side -> swap the two leading spatial axes
            arr = np.swapaxes(arr, 0, 1)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def read(self) -> TactileFrame:
        if self._sdk is None:
            raise RuntimeError("read() before connect()")
        # pace to the configured rate (SDK getters return the latest processed frame)
        period = 1.0 / self.cfg.rate_hz
        now = time.perf_counter()
        wait = self._last_read_t + period - now
        if wait > 0:
            time.sleep(wait)
        t_host = time.perf_counter()
        self._last_read_t = t_host

        deformation = self._canon(self._sdk.getDeformation2D())   # (H, W, 2)
        depth = self._canon(self._sdk.getDepth())                 # (H, W)
        shear = self._canon(self._sdk.getShear())                 # (H, W, 2)
        dist_force = self._sdk.getDistributeForce()
        # SDK may return a (fx, fy, fz) tuple of (H, W) arrays or one (H, W, 3)
        if isinstance(dist_force, (tuple, list)):
            dist_force = np.stack([self._canon(a) for a in dist_force], axis=-1)
        else:
            dist_force = self._canon(dist_force)
        wrench = np.asarray(self._sdk.getForce(), dtype=np.float32).reshape(-1)
        if self.cfg.force_calibrated:
            wrench = wrench * self.cfg.force_unit_to_N
        area = float(self._sdk.getContactArea())
        infer_img = np.asarray(self._sdk.getInferImg())

        frame = TactileFrame(
            t_host=t_host, seq=self._seq,
            deformation=deformation, depth=depth, shear=shear, dist_force=dist_force,
            wrench=wrench, contact_area_mm2=area,
            infer_img=infer_img, raw_img=None,
        )
        self._seq += 1
        self._check_first_frame(frame)
        return frame

    def read_raw_img(self) -> np.ndarray:
        """Full raw camera frame (archive path, bench item (e))."""
        if self._sdk is None:
            raise RuntimeError("read_raw_img() before connect()")
        return np.asarray(self._sdk.getRawImg())
