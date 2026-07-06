"""Real Robotiq 2F-85/140 driver over the URCap socket server (port 63352 on
the UR controller). ASCII protocol: `SET <VAR> <VAL>` / `GET <VAR>`.

Normalization to 0..1 happens here (via stroke_mm for reporting only; the
wire protocol is already 0..255)."""

from __future__ import annotations

import socket
import threading
import time

from phantom.config.hardware import GripperConfig
from phantom.drivers.base import Gripper, GripperState


class RobotiqGripper(Gripper):
    def __init__(self, cfg: GripperConfig, arm_ip: str):
        super().__init__(cfg)
        self.arm_ip = arm_ip
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._seq = 0

    # ------------------------------------------------------------------
    def connect(self) -> None:
        self._sock = socket.create_connection((self.arm_ip, self.cfg.port), timeout=2.0)

    def disconnect(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()

    def _cmd(self, cmd: str) -> str:
        if self._sock is None:
            raise RuntimeError("gripper command before connect()")
        with self._lock:
            self._sock.sendall((cmd + "\n").encode("ascii"))
            return self._sock.recv(1024).decode("ascii").strip()

    def _get(self, var: str) -> int:
        resp = self._cmd(f"GET {var}")  # e.g. "POS 87"
        parts = resp.split()
        if len(parts) != 2 or parts[0] != var:
            raise RuntimeError(f"unexpected gripper response to GET {var}: {resp!r}")
        return int(parts[1])

    def _set(self, **vals: int) -> None:
        fields = " ".join(f"{k.upper()} {v}" for k, v in vals.items())
        resp = self._cmd(f"SET {fields}")
        if not resp.startswith("ack"):
            raise RuntimeError(f"gripper SET not acked: {resp!r}")

    # ------------------------------------------------------------------
    def activate(self) -> None:
        self._set(act=1)
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            if self._get("STA") == 3:
                return
            time.sleep(0.1)
        raise RuntimeError("gripper activation timed out (GET STA != 3)")

    def move(self, position: float, speed: float, force: float) -> None:
        def to255(x: float) -> int:
            return max(0, min(255, round(x * 255)))
        # 'FOR' is a Python keyword so it can't go through _set kwargs
        resp = self._cmd(f"SET POS {to255(position)} SPE {to255(speed)} "
                         f"FOR {to255(force)} GTO 1")
        if not resp.startswith("ack"):
            raise RuntimeError(f"gripper SET not acked: {resp!r}")

    def get_state(self) -> GripperState:
        pos = self._get("POS") / 255.0
        obj = self._get("OBJ")           # 0 moving, 1/2 object detected, 3 at position
        cur = self._get("CUR") / 255.0
        st = GripperState(
            t_host=time.perf_counter(), seq=self._seq,
            position=pos, current=cur,
            moving=(obj == 0), object_detected=(obj in (1, 2)),
        )
        self._seq += 1
        return st
