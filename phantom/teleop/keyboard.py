"""Keyboard teleop — works on every machine, the dev default.

Motion keys (held): w/s = ±x, a/d = ±y, q/e = ±z, arrow keys = rx/ry,
z/x = ±rz, o/p = gripper open/close.
Episode keys: SPACE start/stop, g success, b fail, f failure-tag, ESC abort,
0 zero F/T, F12 quit.

Uses msvcrt on Windows and termios/select on POSIX (no external deps).
Non-blocking: keys read since last poll set the command for one tick.
"""

from __future__ import annotations

import sys
import threading

import numpy as np

from phantom.teleop.base import TeleopCommand, TeleopDevice

_MOTION = {
    "w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1),
    "q": (2, +1), "e": (2, -1), "z": (5, +1), "x": (5, -1),
    "i": (3, +1), "k": (3, -1), "j": (4, +1), "l": (4, -1),
}
_BUTTONS = {" ": "start_stop", "g": "success", "b": "fail", "f": "failure_tag",
            "\x1b": "abort", "0": "zero_ft", "~": "quit"}


class KeyboardTeleop(TeleopDevice):
    def __init__(self, step_m: float = 0.01, step_rad: float = 0.05):
        self.step_m = step_m
        self.step_rad = step_rad
        self._keys: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gripper = 0.0

    # ------------------------------------------------------------------
    def _reader_windows(self) -> None:
        import msvcrt
        while not self._stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                with self._lock:
                    self._keys.append(ch)
            else:
                self._stop.wait(0.005)

    def _reader_posix(self) -> None:
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self._keys.append(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def start(self) -> None:
        target = self._reader_windows if sys.platform == "win32" else self._reader_posix
        self._thread = threading.Thread(target=target, daemon=True, name="kbd-teleop")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)

    # ------------------------------------------------------------------
    def poll(self) -> TeleopCommand:
        with self._lock:
            keys, self._keys = self._keys, []
        dpose = np.zeros(6)
        buttons: dict[str, bool] = {}
        for ch in keys:
            low = ch.lower()
            if low in _MOTION:
                axis, sign = _MOTION[low]
                dpose[axis] += sign * (self.step_m if axis < 3 else self.step_rad)
            elif low in _BUTTONS:
                buttons[_BUTTONS[low]] = True
            elif low == "o":
                self._gripper = max(0.0, self._gripper - 0.1)
            elif low == "p":
                self._gripper = min(1.0, self._gripper + 0.1)
        return TeleopCommand(dpose=dpose, gripper=self._gripper, buttons=buttons)
