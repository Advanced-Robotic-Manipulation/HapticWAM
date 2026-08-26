"""Real Intel RealSense driver over pyrealsense2 (lazy import)."""

from __future__ import annotations

import logging
import time

import numpy as np

from phantom.config.hardware import CameraEntry
from phantom.drivers.base import Camera, CameraFrame

log = logging.getLogger(__name__)

# Consecutive wait_for_frames timeouts before the pipeline is rebuilt. A
# librealsense pipeline can wedge permanently after a USB stall (observed on
# the rig 2026-08-18: hub contention with the DmTac gels); wait_for_frames
# then times out forever and the poller's retry never heals it. Rebuilding
# the pipeline does.
_REBUILD_AFTER_TIMEOUTS = 3
# ...but the rebuild itself fails while the device is still busy — exactly the
# state a USB stall leaves it in. One failed rebuild used to kill the camera
# for the whole session (blocker 2026-08-26): it left an unstarted pipeline,
# the next read() blew up inside disconnect(), _pipe stayed None and every
# later read() raised before ever reaching the rebuild path. So the rebuild is
# RETRIED, backed off, and bounded; past the bound the camera is declared dead
# and every read() raises loudly instead of the ring silently freezing.
_REBUILD_MAX_ATTEMPTS = 6
_REBUILD_BACKOFF_S = 0.5       # doubles per consecutive failure
_REBUILD_BACKOFF_MAX_S = 5.0


class RealSenseCamera(Camera):
    def __init__(self, cfg: CameraEntry, name: str):
        super().__init__(cfg, name)
        self._pipe = None
        self._rs = None
        self._seq = 0
        self._timeouts = 0
        self._rebuilds = 0              # CONSECUTIVE failed rebuild attempts
        self._next_rebuild_t = 0.0      # backoff gate (perf_counter)
        self._last_frame_t = 0.0        # perf_counter of the last frame read
        self._dead_reason: str | None = None

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError("pyrealsense2 not installed — pip install pyrealsense2, "
                               "or set mode.overrides.cameras: mock") from e
        self._rs = rs
        config = rs.config()
        if self.cfg.serial:
            config.enable_device(self.cfg.serial)
        config.enable_stream(rs.stream.color, self.cfg.color.w, self.cfg.color.h,
                             rs.format.rgb8, int(self.cfg.fps))
        if self.cfg.depth_enabled:
            config.enable_stream(rs.stream.depth, self.cfg.color.w, self.cfg.color.h,
                                 rs.format.z16, int(self.cfg.fps))
        # build into a LOCAL and publish only once started: a failed start()
        # must never leave self._pipe pointing at an unstarted pipeline (every
        # later wait_for_frames then raises "cannot be called before start()"
        # and disconnect() raises out of stop()).
        pipe = rs.pipeline()
        pipe.start(config)
        self._pipe = pipe
        self._seq = 0
        self._timeouts = 0
        self._dead_reason = None
        # NOTE: _rebuilds is deliberately NOT reset here — connect() is itself
        # the rebuild step, and only a real frame proves the camera recovered.

    def disconnect(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is None:
            return
        try:
            pipe.stop()
        except Exception as e:
            # an unstarted/wedged pipeline raises from stop(); dropping the
            # reference is the part that matters and it already happened
            log.debug("camera %s: pipeline.stop() failed: %s", self.name, e)

    # ------------------------------------------------------------------
    @property
    def healthy(self) -> bool:
        """False once the pipeline is gone and the bounded rebuild gave up."""
        return self._dead_reason is None and self._pipe is not None

    def last_frame_age(self) -> float:
        """Seconds since the last frame actually read (inf before the first).

        The planner sees the same quantity through the camera ring's newest
        timestamp; this is the driver-side view for a direct owner."""
        if self._last_frame_t == 0.0:
            return float("inf")
        return time.perf_counter() - self._last_frame_t

    def _try_rebuild(self, why: str) -> bool:
        """Bounded, backed-off pipeline rebuild. Never raises.

        Returns True iff a started pipeline is live afterwards. On the last
        allowed attempt the camera is marked dead so read() fails loudly
        instead of letting the ring freeze on the pre-stall frame."""
        now = time.perf_counter()
        if now < self._next_rebuild_t:
            return False                       # backing off
        if self._rebuilds >= _REBUILD_MAX_ATTEMPTS:
            if self._dead_reason is None:
                self._dead_reason = (f"{self._rebuilds} consecutive pipeline "
                                     f"rebuilds failed ({why})")
                log.error("camera %s is DEAD: %s", self.name, self._dead_reason)
            return False
        self._rebuilds += 1
        log.warning("camera %s: %s — rebuilding the pipeline (attempt %d/%d)",
                    self.name, why, self._rebuilds, _REBUILD_MAX_ATTEMPTS)
        self.disconnect()
        try:
            self.connect()
        except Exception as e:
            backoff = min(_REBUILD_BACKOFF_S * 2 ** (self._rebuilds - 1),
                          _REBUILD_BACKOFF_MAX_S)
            self._next_rebuild_t = time.perf_counter() + backoff
            log.warning("camera %s: pipeline rebuild %d/%d failed (%s) — retrying "
                        "in %.1fs", self.name, self._rebuilds,
                        _REBUILD_MAX_ATTEMPTS, e, backoff)
            return False
        log.info("camera %s: pipeline rebuilt", self.name)
        return True

    def read(self) -> CameraFrame:
        if self._dead_reason is not None:
            raise RuntimeError(f"camera {self.name} is dead: {self._dead_reason} "
                               "— power-cycle the USB link and restart the session")
        if self._pipe is None:
            # Re-arm. Covers both "read() before connect()" and a rebuild that
            # could not restart the device yet; either way the retry is bounded
            # and ends in _dead_reason rather than in silence.
            if not self._try_rebuild("no live pipeline"):
                raise RuntimeError(
                    f"camera {self.name}: no live pipeline "
                    f"({self._dead_reason or 'rebuild backing off'})")
        try:
            frames = self._pipe.wait_for_frames()
        except RuntimeError:
            self._timeouts += 1
            if self._timeouts >= _REBUILD_AFTER_TIMEOUTS:
                self._try_rebuild(f"{self._timeouts} consecutive frame timeouts")
                self._timeouts = 0
            raise
        self._timeouts = 0
        self._rebuilds = 0          # a real frame is the only proof of recovery
        self._next_rebuild_t = 0.0
        t_host = time.perf_counter()
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = None
        if self.cfg.depth_enabled:
            df = frames.get_depth_frame()
            if df:
                depth = np.asanyarray(df.get_data())
        frame = CameraFrame(t_host=t_host, seq=self._seq, color=color, depth=depth,
                            t_device=frames.get_timestamp() / 1000.0)
        self._seq += 1
        self._last_frame_t = t_host
        return frame
