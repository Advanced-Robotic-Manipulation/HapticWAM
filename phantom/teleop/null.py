"""NullTeleop: no motion device — episode control comes from the web panel
(or hotkeys). Useful for monitoring sessions, panel-driven smoke tests, and
contact-play recording where a human moves objects against the sensors by
hand and only start/stop control is needed."""

from __future__ import annotations

import numpy as np

from phantom.teleop.base import TeleopCommand, TeleopDevice


class NullTeleop(TeleopDevice):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def poll(self) -> TeleopCommand:
        return TeleopCommand(dpose=np.zeros(6), gripper=0.0, buttons={})
