"""Mock Robotiq gripper: first-order position tracking; object_detected while
the scenario is in contact and the gripper is sufficiently closed."""

from __future__ import annotations

import threading
import time

import numpy as np

from phantom.config.hardware import GripperConfig
from phantom.drivers.base import Gripper, GripperState
from phantom.drivers.mock.scenario import ContactScenario


class MockGripper(Gripper):
    def __init__(self, cfg: GripperConfig, scenario: ContactScenario | None = None):
        super().__init__(cfg)
        self.scenario = scenario
        self._lock = threading.Lock()
        self._pos = 0.0
        self._target = 0.0
        self._speed = cfg.default_speed
        self._t0 = 0.0
        self._last_t = 0.0
        self._seq = 0
        self._connected = False
        self._activated = False

    def connect(self) -> None:
        self._t0 = self._last_t = time.perf_counter()
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def activate(self) -> None:
        if not self._connected:
            raise RuntimeError("activate() before connect()")
        self._activated = True

    def move(self, position: float, speed: float, force: float) -> None:
        if not self._activated:
            raise RuntimeError("move() before activate()")
        with self._lock:
            self._target = float(np.clip(position, 0.0, 1.0))
            self._speed = float(np.clip(speed, 0.05, 1.0))

    def _step(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now
        max_step = self._speed * 2.0 * dt  # full stroke in ~0.5 s at speed 1
        d = np.clip(self._target - self._pos, -max_step, max_step)
        self._pos = float(np.clip(self._pos + d, 0.0, 1.0))

    def get_state(self) -> GripperState:
        if not self._connected:
            raise RuntimeError("get_state() before connect()")
        with self._lock:
            self._step()
            moving = abs(self._target - self._pos) > 1e-3
            in_contact = (self.scenario is not None
                          and self.scenario.state(time.perf_counter() - self._t0).in_contact)
            detected = in_contact and self._pos > 0.3
            st = GripperState(
                t_host=time.perf_counter(), seq=self._seq,
                position=self._pos,
                current=0.1 + (0.5 if detected else 0.0),
                moving=moving, object_detected=detected,
            )
            self._seq += 1
        return st
