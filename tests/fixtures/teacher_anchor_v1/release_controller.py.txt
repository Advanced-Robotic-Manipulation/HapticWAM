"""Explicit placement release variant; observes robot feedback, never object state.

The native rig aperture latch is a running maximum and cannot honor a policy
release. This opt-in gate permits a sustained policy opening inside a declared
TCP release volume after a loaded grasp. It does not generate motion or choose
an aperture, and it does not identify task success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PlacementReleaseConfig:
    tcp_min_m: tuple[float, float, float]
    tcp_max_m: tuple[float, float, float]
    open_command_max: float = 0.45
    opening_hold_s: float = 0.2
    unloaded_force_max_n: float = 0.5
    unloaded_hold_s: float = 0.2
    rearm_close_command_min: float = 0.5

    def __post_init__(self):
        for name in ("tcp_min_m", "tcp_max_m"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite base-frame metres")
            object.__setattr__(self, name, tuple(float(v) for v in value))
        if not np.all(np.asarray(self.tcp_min_m) < np.asarray(self.tcp_max_m)):
            raise ValueError("TCP release bounds must have positive volume")
        if not 0 <= self.open_command_max < self.rearm_close_command_min <= 1:
            raise ValueError(
                "release/rearm closure thresholds must satisfy 0 <= open < close <= 1"
            )
        for name in ("opening_hold_s", "unloaded_hold_s", "unloaded_force_max_n"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        return asdict(self)


class PlacementReleaseController:
    """Latch permission only; every released aperture still comes from policy.

    The gate starts cold and becomes eligible only after the native load latch was
    actually armed. Once committed, loaded pads cannot immediately re-arm it. Both
    pads must first become unloaded and the measured gripper open for a dwell, then
    a subsequent policy closing command permits a new load latch.
    """

    def __init__(self, config: PlacementReleaseConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.phase = "unarmed"
        self.opening_since = None
        self.unloaded_since = None
        self.committed_at = None
        self.last_event = None

    def note_latch(self, latch):
        if latch is not None and self.phase == "unarmed":
            self.phase = "holding"

    def in_volume(self, tcp):
        xyz = np.asarray(tcp, dtype=float)[:3]
        return bool(
            xyz.shape == (3,)
            and np.isfinite(xyz).all()
            and np.all(xyz >= self.config.tcp_min_m)
            and np.all(xyz <= self.config.tcp_max_m)
        )

    def window_active(self, tcp):
        return self.phase in (
            "holding",
            "releasing",
            "waiting_for_close",
        ) and self.in_volume(tcp)

    @property
    def suppress_latch(self):
        return self.phase in ("releasing", "waiting_for_close")

    def update(self, t, *, tcp, policy_grip, measured_grip, pad_loads, eligible):
        self.last_event = None
        c = self.config
        if self.phase == "holding":
            opening = (
                eligible
                and self.window_active(tcp)
                and np.isfinite(policy_grip)
                and policy_grip <= c.open_command_max
            )
            if not opening:
                self.opening_since = None
            elif self.opening_since is None:
                self.opening_since = float(t)
            elif t - self.opening_since >= c.opening_hold_s - 1e-9:
                self.phase = "releasing"
                self.committed_at = float(t)
                self.last_event = "policy_release_committed"
        elif self.phase == "releasing":
            unloaded_open = (
                len(pad_loads) >= 2
                and all(
                    np.isfinite(f) and 0 <= f < c.unloaded_force_max_n
                    for f in pad_loads.values()
                )
                and np.isfinite(measured_grip)
                and measured_grip <= c.open_command_max
            )
            if not unloaded_open:
                self.unloaded_since = None
            elif self.unloaded_since is None:
                self.unloaded_since = float(t)
            elif t - self.unloaded_since >= c.unloaded_hold_s - 1e-9:
                self.phase = "waiting_for_close"
                self.last_event = "measured_unloaded_open"
        elif self.phase == "waiting_for_close":
            if eligible and policy_grip >= c.rearm_close_command_min:
                self.phase = "unarmed"
                self.opening_since = self.unloaded_since = None
                self.last_event = "policy_reclose_permitted"
        return self.suppress_latch

    def stop(self):
        self.opening_since = None
        self.last_event = "safety_preempted"

    def diagnostics(self, tcp):
        return {
            "variant": "placement_policy_release_v1",
            "phase": self.phase,
            "window_active": self.window_active(tcp),
            "opening_since_s": self.opening_since,
            "committed_at_s": self.committed_at,
            "unloaded_since_s": self.unloaded_since,
            "suppress_latch": self.suppress_latch,
            "event": self.last_event,
        }
