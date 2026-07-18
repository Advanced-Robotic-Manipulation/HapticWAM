"""Teleop device interface. poll() is called at control.action_rate_hz."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeleopCommand:
    dpose: np.ndarray                     # (6,) Δ-EE pose per action tick
    gripper: float                        # target 0..1
    buttons: dict[str, bool] = field(default_factory=dict)
    # episode-control button semantics (any device maps its keys onto these):
    #   start_stop, success, fail, failure_tag, abort, zero_ft, quit
    # Joint-space leaders (Echo exoskeleton) set q_target instead of dpose:
    # the record loop then drives servo_j and derives the recorded Δ-EE action
    # from consecutive measured TCP poses (data/derived.py pose_delta).
    q_target: np.ndarray | None = None    # (dof,) absolute joint target, rad


class TeleopDevice(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def poll(self) -> TeleopCommand: ...
