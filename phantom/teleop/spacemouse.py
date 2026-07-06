"""3Dconnexion SpaceMouse teleop via pyspacemouse (lazy import; rig machines).

Buttons: left = start/stop episode, right = gripper toggle. Episode
success/fail/tag keys still come from the keyboard in record_episodes.py.
"""

from __future__ import annotations

import numpy as np

from phantom.teleop.base import TeleopCommand, TeleopDevice


class SpaceMouseTeleop(TeleopDevice):
    def __init__(self, gain_m: float = 0.02, gain_rad: float = 0.1,
                 deadzone: float = 0.05):
        self.gain_m = gain_m
        self.gain_rad = gain_rad
        self.deadzone = deadzone
        self._sm = None
        self._gripper = 0.0
        self._prev_right = False

    def start(self) -> None:
        try:
            import pyspacemouse
        except ImportError as e:
            raise RuntimeError("pyspacemouse not installed — pip install pyspacemouse "
                               "or use --teleop keyboard") from e
        self._sm = pyspacemouse
        if not pyspacemouse.open():
            raise RuntimeError("no SpaceMouse found")

    def stop(self) -> None:
        if self._sm is not None:
            self._sm.close()
            self._sm = None

    def poll(self) -> TeleopCommand:
        assert self._sm is not None, "poll() before start()"
        st = self._sm.read()
        axes = np.array([st.y, -st.x, st.z, st.pitch, st.roll, -st.yaw], dtype=np.float64)
        axes[np.abs(axes) < self.deadzone] = 0.0
        dpose = np.concatenate([axes[:3] * self.gain_m, axes[3:] * self.gain_rad])
        buttons: dict[str, bool] = {}
        if st.buttons and st.buttons[0]:
            buttons["start_stop"] = True
        right = bool(st.buttons and len(st.buttons) > 1 and st.buttons[1])
        if right and not self._prev_right:
            self._gripper = 1.0 - self._gripper
        self._prev_right = right
        return TeleopCommand(dpose=dpose, gripper=self._gripper, buttons=buttons)
