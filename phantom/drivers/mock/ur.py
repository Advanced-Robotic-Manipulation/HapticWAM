"""Mock UR arm: kinematic integrator + scenario-correlated wrist F/T.

The wrist wrench envelope comes from ContactScenario.wrist_ft_scale, which
rises BEFORE tactile onset — giving ACC's lead-time machinery real structure
to find even in fully mocked runs.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.drivers.base import Arm, ArmState
from phantom.drivers.mock.scenario import ContactScenario


class MockArm(Arm):
    def __init__(self, hw: HardwareConfig, scenario: ContactScenario | None = None):
        super().__init__(hw)
        self.scenario = scenario
        self._rng = np.random.default_rng(42)
        self._lock = threading.Lock()
        dof = hw.arm.dof
        self._q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0][:dof], dtype=np.float64)
        self._qd = np.zeros(dof)
        self._tcp = np.array([0.0, -0.45, 0.25, 0.0, 3.14, 0.0], dtype=np.float64)
        self._tcp_speed = np.zeros(6)
        self._t0 = 0.0
        self._last_cmd_t = 0.0
        self._seq = 0
        self._connected = False
        self._protective_stop = False
        self._servo_active = False

    # ------------------------------------------------------------------
    def connect(self, *, control: bool = False) -> None:
        self._t0 = time.perf_counter()
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _scenario_t(self) -> float:
        return time.perf_counter() - self._t0

    def get_state(self) -> ArmState:
        if not self._connected:
            raise RuntimeError("get_state() before connect()")
        now = time.perf_counter()
        ft = self._rng.normal(0, 0.05, 6)
        if self.scenario is not None:
            scale = self.scenario.wrist_ft_scale(self._scenario_t())
            ft += np.array([0.0, 0.0, -1.0, 0.05, 0.05, 0.0]) * scale * 10.0
        with self._lock:
            st = ArmState(
                t_host=now, seq=self._seq, t_rtde=now - self._t0,
                q=self._q.copy(), qd=self._qd.copy(),
                tcp_pose=self._tcp.copy(), tcp_speed=self._tcp_speed.copy(),
                ft=ft.astype(np.float64),
                protective_stop=self._protective_stop, robot_mode=7,
            )
            self._seq += 1
        return st

    # ------------------------------------------------------------------
    def servo_j(self, q: np.ndarray, dt: float, lookahead: float, gain: int) -> None:
        with self._lock:
            self._servo_active = True
            q = np.asarray(q, dtype=np.float64)
            dq = q - self._q
            max_step = self.hw.arm.limits.joint_speed_rad_s * dt
            dq = np.clip(dq, -max_step, max_step)
            self._q = self._q + dq
            self._qd = dq / max(dt, 1e-9)

    def servo_l(self, tcp_pose: np.ndarray, dt: float, lookahead: float, gain: int) -> None:
        with self._lock:
            self._servo_active = True
            target = np.asarray(tcp_pose, dtype=np.float64)
            d = target - self._tcp
            max_step = self.hw.arm.limits.tcp_speed_m_s * dt
            d[:3] = np.clip(d[:3], -max_step, max_step)
            d[3:] = np.clip(d[3:], -4 * max_step, 4 * max_step)
            self._tcp = self._tcp + d
            self._tcp_speed = d / max(dt, 1e-9)

    def speed_l(self, xd: np.ndarray, accel: float, dt: float) -> None:
        with self._lock:
            xd = np.asarray(xd, dtype=np.float64)
            self._tcp = self._tcp + xd * dt
            self._tcp_speed = xd.copy()

    def move_j(self, q: np.ndarray, speed: float, accel: float, blocking: bool = True) -> None:
        if self._servo_active:
            raise RuntimeError("move_j while servo mode active; call servo_stop() first")
        with self._lock:
            self._q = np.asarray(q, dtype=np.float64).copy()
            self._qd = np.zeros_like(self._q)

    def stop(self, decel: float) -> None:
        with self._lock:
            self._qd[:] = 0.0
            self._tcp_speed[:] = 0.0

    def zero_ft(self) -> None:
        pass

    def is_protective_stopped(self) -> bool:
        return self._protective_stop

    def servo_stop(self) -> None:
        self._servo_active = False
        self.stop(2.0)

    # test hook
    def trigger_protective_stop(self, on: bool = True) -> None:
        self._protective_stop = on
