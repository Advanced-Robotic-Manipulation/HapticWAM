"""Real UR arm driver over `ur_rtde`.

- RTDEReceiveInterface is ALWAYS opened (recording, safety, wrist F/T,
  master-clock timestamps).
- RTDEControlInterface is opened only when connect(control=True): it claims
  exclusive control-script ownership and must not exist in record-only
  sessions (it would fight teleop-through-pendant or other controllers).
- Protective stops kill the servo session; reconnect_control() restores it
  after manual unlock on the pendant.

`ur_rtde` is imported lazily. CB3 vs e-Series rate constraints are enforced by
the hardware config validators before this driver is ever constructed.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.drivers.base import Arm, ArmState

log = logging.getLogger(__name__)


class URArm(Arm):
    def __init__(self, hw: HardwareConfig):
        super().__init__(hw)
        self._recv = None
        self._ctrl = None
        self._seq = 0
        self._want_control = False

    # ------------------------------------------------------------------
    def connect(self, *, control: bool = False) -> None:
        try:
            import rtde_receive
        except ImportError as e:
            raise RuntimeError("ur_rtde not installed — pip install ur-rtde, or set "
                               "mode.overrides.arm: mock") from e
        self._recv = rtde_receive.RTDEReceiveInterface(
            self.hw.arm.ip, frequency=self.hw.arm.rtde_receive_hz)
        self._want_control = control
        if control:
            self._connect_control()

    def _connect_control(self) -> None:
        import rtde_control
        self._ctrl = rtde_control.RTDEControlInterface(
            self.hw.arm.ip, frequency=self.hw.arm.rtde_control_hz)
        self._ctrl.setTcp(list(self.hw.arm.tcp_offset_m))
        self._ctrl.setPayload(self.hw.arm.payload_kg, [0.0, 0.0, 0.05])

    def reconnect_control(self) -> None:
        """After a protective stop has been manually cleared on the pendant."""
        if self._ctrl is not None:
            try:
                self._ctrl.disconnect()
            except Exception:
                pass
            self._ctrl = None
        self._connect_control()

    def disconnect(self) -> None:
        if self._ctrl is not None:
            try:
                self._ctrl.servoStop()
                self._ctrl.stopScript()
            except Exception:
                pass
            self._ctrl.disconnect()
            self._ctrl = None
        if self._recv is not None:
            self._recv.disconnect()
            self._recv = None

    def _require_ctrl(self):
        if self._ctrl is None:
            raise RuntimeError("control interface not connected (connect(control=True))")
        if not self._ctrl.isConnected():
            raise RuntimeError("RTDE control stream lost (robot rebooted / protective "
                               "stop / network drop) — reconnect_control() after fixing")
        return self._ctrl

    # ------------------------------------------------------------------
    def get_state(self) -> ArmState:
        r = self._recv
        if r is None:
            raise RuntimeError("get_state() before connect()")
        if not r.isConnected():
            # fail LOUD instead of blocking the whole teleop loop forever —
            # observed on the CB3 after a mid-motion RTDE stream drop
            raise RuntimeError("RTDE receive stream lost — robot rebooted or "
                               "network dropped; restart the session")
        st = ArmState(
            t_host=time.perf_counter(), seq=self._seq,
            t_rtde=float(r.getTimestamp()),
            q=np.asarray(r.getActualQ(), dtype=np.float64),
            qd=np.asarray(r.getActualQd(), dtype=np.float64),
            tcp_pose=np.asarray(r.getActualTCPPose(), dtype=np.float64),
            tcp_speed=np.asarray(r.getActualTCPSpeed(), dtype=np.float64),
            ft=np.asarray(r.getActualTCPForce(), dtype=np.float64),
            protective_stop=bool(r.isProtectiveStopped()),
            robot_mode=int(r.getRobotMode()),
        )
        self._seq += 1
        return st

    def servo_j(self, q: np.ndarray, dt: float, lookahead: float, gain: int) -> None:
        ok = self._require_ctrl().servoJ(list(np.asarray(q, dtype=float)), 0.0, 0.0,
                                         dt, lookahead, gain)
        if ok is False:
            # ur_rtde returns False SILENTLY when the control script is no
            # longer running on the robot (a protective stop kills it) — the
            # arm just stops following while everything else keeps working
            raise RuntimeError("servoJ rejected — the RTDE control script is not "
                               "running (clear the pendant popup / protective stop "
                               "and restart the session)")

    def servo_l(self, tcp_pose: np.ndarray, dt: float, lookahead: float, gain: int) -> None:
        ctrl = self._require_ctrl()
        q = ctrl.getInverseKinematics(list(np.asarray(tcp_pose, dtype=float)))
        ok = ctrl.servoJ(q, 0.0, 0.0, dt, lookahead, gain)
        if ok is False:
            raise RuntimeError("servoJ rejected — the RTDE control script is not "
                               "running (clear the pendant popup / protective stop "
                               "and restart the session)")

    def speed_l(self, xd: np.ndarray, accel: float, dt: float) -> None:
        self._require_ctrl().speedL(list(np.asarray(xd, dtype=float)), accel, dt)

    def move_j(self, q: np.ndarray, speed: float, accel: float, blocking: bool = True) -> None:
        self._require_ctrl().moveJ(list(np.asarray(q, dtype=float)), speed, accel,
                                   not blocking)

    def stop(self, decel: float) -> None:
        try:
            self._require_ctrl().stopL(decel)
        except Exception:
            log.exception("stopL failed")

    def zero_ft(self) -> None:
        self._require_ctrl().zeroFtSensor()

    def is_protective_stopped(self) -> bool:
        if self._recv is None:
            raise RuntimeError("is_protective_stopped() before connect()")
        return bool(self._recv.isProtectiveStopped())

    def servo_stop(self) -> None:
        self._require_ctrl().servoStop()
