"""Real DM-Tac W2L driver over the `dmrobotics` Python SDK (verified v1.2.10).

Boundary rules (so downstream code never thinks about SDK quirks):
 - every SDK field getter returns ``(fid, data)`` — the fid is unpacked HERE;
 - ``getRawImg``/``getInferImg`` return ``(fid, DMTacImage)`` — pixels live at
   ``DMTacImage.img`` (uint8 grayscale) and are extracted HERE;
 - hw_order transpose happens HERE: downstream always sees (field.h, field.w);
 - the wrench is scaled to SI HERE (Fx,Fy,Fz * force_unit_to_N; Mx,My,Mz *
   torque_unit_to_Nm — the SDK reports torques in 1e-2 N*m) while the 0.0
   UNCALIBRATED sentinel is unset (consumers check tactile.wrench_calibrated);
 - one read() = one coherent grab of all enabled field getters, paced by
   ``wait_for_new`` (frame-id sync) with a sleep fallback.

SDK facts this file is written against (from SDK_Publish_1.2.10 source + the
official dev manual v2.0):
 - ``SensorOptions`` channel enables ALL default to **False** — they must be
   set explicitly or every getter returns nothing;
 - ``SensorOptions(dev_id=...)`` accepts an int device index or a str serial
   ("识别码", e.g. "X26040565"); str is recommended for multi-sensor rigs;
 - ``getDistributeForce()`` is documented as ``(fid, ndarray(H, W, 3))`` in the
   manual but shown as ``(fid, fx, fy, fz)`` in vendor snippets — both layouts
   are handled;
 - ``getDevStatus()``: 0 OK / 1 RESETTING / 2 DISCONNECTED — reads are guarded;
 - backends: "cpu" | "cuda" | "flux" (lowercase in the real SDK).

`dmrobotics` is imported lazily inside connect() so this module imports fine
on machines without the SDK (macOS dev boxes: the SDK ships linux/windows
x86_64 binaries for Python 3.8-3.11 only).

BENCH day-1 (docs/hardware_bench_day1.md) still verifies: getDistributeForce
units (a), image channel counts + true concurrent rates (b), numpy (H, W)
ordering (c), sustained throughput (d), offline recompute (e).
"""

from __future__ import annotations

import time

import numpy as np

from phantom.config.hardware import TactileConfig, TactileSensorEntry
from phantom.drivers.base import TactileFrame, TactileSensor

_STATUS_OK = 0
_STATUS_RESETTING = 1
_STATUS_DISCONNECTED = 2


class DmTacSensor(TactileSensor):
    def __init__(self, cfg: TactileConfig, sensor: TactileSensorEntry):
        super().__init__(cfg, sensor)
        self._sdk = None
        self._seq = 0
        self._last_fid = -1
        self._last_read_t = 0.0

    # ------------------------------------------------------------------
    def connect(self) -> None:
        try:
            from dmrobotics import Mode, Sensor, SensorOptions  # lazy: rig machines only
        except ImportError as e:
            raise RuntimeError(
                "dmrobotics SDK not installed — install it on the rig machine (Python "
                "3.8-3.11, linux/windows x86_64 only), or set "
                "mode.overrides.tactile: mock in configs/hardware.yaml") from e
        kwargs = dict(
            dev_id=self.sensor.dev_id,
            backend=self.cfg.sdk_backend,
            mode=Mode.HIGH if self.cfg.sdk_mode == "high" else Mode.STANDARD,
            max_fps=int(round(self.cfg.rate_hz)),
            # enables default to False in the real SDK — set every channel we consume
            enable_raw=True,           # raw grayscale archive -> offline recompute path
            enable_deformation=True,
            enable_depth=True,
            enable_shear=True,
            enable_force=True,
        )
        # Network streaming options (verified on the Denmark rig): pass only
        # when configured so unset fields keep the SDK defaults.
        if self.sensor.remote_addr is not None:
            kwargs["remote_addr"] = self.sensor.remote_addr
        if self.sensor.pc_port is not None:
            kwargs["pc_port"] = self.sensor.pc_port
        if self.cfg.pc_host is not None:
            kwargs["pc_host"] = self.cfg.pc_host
        opt = SensorOptions(**kwargs)
        self._sdk = Sensor(opt)
        self._verify_identity()
        self.reset_reference()
        self._last_fid = -1
        self._warmup()
        self._seq = 0

    def _warmup(self, n: int = 5) -> None:
        """Discard the first frames after reset. The SDK emits a transient for
        a frame or two while the reconstruction nets prime — observed on real
        L26xxx units as an all-zeros frame then a huge depth spike (peak ~2.0
        vs settled ~0.09). Without this the first read() — and the safety
        monitor that consumes it — sees a spurious high-indentation e-stop."""
        for _ in range(n):
            try:
                self.read()
            except Exception:
                time.sleep(0.05)

    def _verify_identity(self) -> None:
        """When configured by serial, assert we actually connected to that unit."""
        if not isinstance(self.sensor.dev_id, str):
            return
        got = str(self._sdk.getDevID())
        want = self.sensor.dev_id
        if not (got == want or got.endswith(want) or want.endswith(got)):
            raise RuntimeError(
                f"tactile[{self.sensor.name}] connected to serial {got!r}, "
                f"configured {want!r} — check yellow cable labels / hardware.yaml")

    def disconnect(self) -> None:
        sdk, self._sdk = self._sdk, None
        if sdk is not None:
            sdk.disconnect()

    def reset_reference(self) -> None:
        if self._sdk is None:
            raise RuntimeError("reset_reference() before connect()")
        # getBaseFrame() returns the current reference frame; calling reset()
        # re-zeros it. Keep the pad untouched during reset (vendor requirement).
        self._sdk.reset()
        deadline = time.perf_counter() + 5.0
        while self._sdk.getDevStatus() == _STATUS_RESETTING:
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"tactile[{self.sensor.name}] stuck RESETTING >5s — pad touched during reset?")
            time.sleep(0.02)

    # ------------------------------------------------------------------
    def _canon(self, arr: np.ndarray) -> np.ndarray:
        """Transpose SDK field arrays into canonical (field.h, field.w) order."""
        arr = np.asarray(arr)
        if self.cfg.hw_order == "wh":
            # SDK axis0 is the field.w side -> swap the two leading spatial axes
            arr = np.swapaxes(arr, 0, 1)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _unpack_field(self, ret) -> np.ndarray:
        """SDK field getters return (fid, ndarray)."""
        _fid, arr = ret
        return self._canon(arr)

    def _unpack_dist_force(self, ret) -> np.ndarray:
        """(fid, ndarray(H,W,3)) per the manual, or (fid, fx, fy, fz) per snippets."""
        if len(ret) == 2:
            return self._canon(ret[1])
        _fid, fx, fy, fz = ret
        return np.stack([self._canon(fx), self._canon(fy), self._canon(fz)], axis=-1)

    @staticmethod
    def _unpack_image(ret) -> np.ndarray:
        """(fid, DMTacImage) -> uint8 pixel array (grayscale)."""
        _fid, im = ret
        return np.asarray(im.img)

    def _wait_ready(self) -> None:
        """Block until the device is OK and a new frame id is available.

        DISCONNECTED is tolerated within the deadline window: field-debugged
        2026-08-13 (waffles deploy) — a RealSense pipeline starting on the
        same USB controller makes the sensor's stream drop transiently
        (img FPS:0.0 -> USB re-enumerate), and the vendor SDK RECONNECTS BY
        ITSELF within ~2-4 s. Treating the first DISCONNECTED as fatal killed
        the tactile worker on every deploy start; only a disconnect that
        persists past the deadline is real."""
        deadline = time.perf_counter() + 6.0
        warned = False
        while True:
            st = self._sdk.getDevStatus()
            if st == _STATUS_OK:
                break
            if st == _STATUS_DISCONNECTED and not warned:
                warned = True
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"tactile[{self.sensor.name}] not ready (status={st}) >6s"
                    + (" — DISCONNECTED did not self-recover" if warned else ""))
            time.sleep(0.02)
        try:
            self._sdk.wait_for_new(self._last_fid, timeout_ms=int(2000.0 / self.cfg.rate_hz) + 50)
        except Exception:
            # fallback: sleep-pace to the configured rate
            period = 1.0 / self.cfg.rate_hz
            wait = self._last_read_t + period - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

    def read(self) -> TactileFrame:
        if self._sdk is None:
            raise RuntimeError("read() before connect()")
        self._wait_ready()
        t_host = time.perf_counter()
        self._last_read_t = t_host

        fid, deform_arr = self._sdk.getDeformation2D()
        self._last_fid = fid
        deformation = self._canon(deform_arr)                                   # (H, W, 2)
        depth = self._unpack_field(self._sdk.getDepth())                        # (H, W)
        shear = self._unpack_field(self._sdk.getShear())                        # (H, W, 2)
        dist_force = self._unpack_dist_force(self._sdk.getDistributeForce())    # (H, W, 3)

        _fid, force_arr = self._sdk.getForce()                                  # (1, 6)
        wrench = np.asarray(force_arr, dtype=np.float32).reshape(-1)
        if self.cfg.wrench_calibrated:
            wrench = wrench * np.array(
                [self.cfg.force_unit_to_N] * 3 + [self.cfg.torque_unit_to_Nm] * 3,
                dtype=np.float32)
        area = float(self._sdk.getContactArea())
        infer_img = self._unpack_image(self._sdk.getInferImg())

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
        """Full raw grayscale camera frame (archive path, bench item (e))."""
        if self._sdk is None:
            raise RuntimeError("read_raw_img() before connect()")
        return self._unpack_image(self._sdk.getRawImg())
