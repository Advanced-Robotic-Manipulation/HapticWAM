"""Mock RealSense camera: noise canvas + a moving square tracking the scenario
blob position, paced at configured fps. Shapes from config only."""

from __future__ import annotations

import time

import numpy as np

from phantom.config.hardware import CameraEntry
from phantom.drivers.base import Camera, CameraFrame
from phantom.drivers.mock.scenario import ContactScenario


class MockCamera(Camera):
    def __init__(self, cfg: CameraEntry, name: str, scenario: ContactScenario | None = None):
        super().__init__(cfg, name)
        self.scenario = scenario
        self._rng = np.random.default_rng(hash(name) % 2**31)
        self._t0 = 0.0
        self._seq = 0
        self._connected = False
        self._last_frame_t = 0.0     # parity: Camera.last_frame_age() reads it
        # parity with RealSenseCamera: a mock camera never wedges on its own,
        # but a dry run / test can set this to exercise the dead-worker path
        self._dead_reason: str | None = None

    def connect(self) -> None:
        self._t0 = time.perf_counter()
        self._seq = 0
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def healthy(self) -> bool:
        return self._connected and self._dead_reason is None

    @property
    def dead_reason(self) -> str | None:
        return self._dead_reason

    def render(self, t_scenario: float, t_host: float) -> CameraFrame:
        h, w, c = self.cfg.color.hwc
        img = (self._rng.random((h, w, c)) * 40).astype(np.uint8)
        if self.scenario is not None:
            st = self.scenario.state(t_scenario)
            cx = int((st.blob_center_uv[0] + 1) / 2 * (w - 1))
            cy = int((st.blob_center_uv[1] + 1) / 2 * (h - 1))
            size = max(4, int(20 + 30 * st.press_depth))
            y0, y1 = max(0, cy - size), min(h, cy + size)
            x0, x1 = max(0, cx - size), min(w, cx + size)
            color = np.array([40, 200, 60] if st.in_contact else [180, 60, 60],
                             dtype=np.uint8)[:c]
            img[y0:y1, x0:x1] = color
        frame = CameraFrame(t_host=t_host, seq=self._seq, color=img, depth=None,
                            t_device=t_scenario)
        self._seq += 1
        self._last_frame_t = t_host
        return frame

    def read(self) -> CameraFrame:
        if self._dead_reason is not None:      # parity with RealSenseCamera
            raise RuntimeError(f"camera {self.name} is dead: {self._dead_reason}")
        if not self._connected:
            raise RuntimeError("read() before connect()")
        period = 1.0 / self.cfg.fps
        now = time.perf_counter()
        target = self._t0 + self._seq * period
        if target < now - period:
            self._seq = int((now - self._t0) / period) + 1
            target = self._t0 + self._seq * period
        if target > now:
            time.sleep(target - now)
            now = target
        return self.render(now - self._t0, now)
