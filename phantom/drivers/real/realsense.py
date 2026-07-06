"""Real Intel RealSense driver over pyrealsense2 (lazy import)."""

from __future__ import annotations

import time

import numpy as np

from phantom.config.hardware import CameraEntry
from phantom.drivers.base import Camera, CameraFrame


class RealSenseCamera(Camera):
    def __init__(self, cfg: CameraEntry, name: str):
        super().__init__(cfg, name)
        self._pipe = None
        self._rs = None
        self._seq = 0

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
        self._pipe = rs.pipeline()
        self._pipe.start(config)
        self._seq = 0

    def disconnect(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            pipe.stop()

    def read(self) -> CameraFrame:
        if self._pipe is None:
            raise RuntimeError("read() before connect()")
        frames = self._pipe.wait_for_frames()
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
        return frame
