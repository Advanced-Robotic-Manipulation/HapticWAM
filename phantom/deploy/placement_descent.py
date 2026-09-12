"""Opt-in descend-then-release supervisor for the placement phase (placement_descent_v1).

Motivation (sim zoo 2026-09-12, docs/results/sim_zoo_20260912/README.md): once the
safety-layer stops are removed (boundary projection v3 + apex cap), the v6 policy
still opens the gripper 0.35-0.39 m above the box floor, where the demonstrations
release at 0.18-0.22 m. This supervisor does not create release intent and does
not pick an aperture: it only (1) withholds release *permission* while the
measured TCP is above ``release_z_max_m`` and (2) replaces the commanded TCP
target with a vertical descent (x/y held at the measured pose, orientation held)
until the release height is reached, for as long as the policy keeps asking to
open inside the release volume. If the policy re-closes, leaves the volume or the
descent exceeds ``max_descent_s`` the supervisor yields and the ordinary release
path continues unchanged. Every transition is recorded for the trace.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PlacementDescentConfig:
    release_z_max_m: float
    descent_speed_m_s: float = 0.10
    max_descent_s: float = 6.0
    activation_command_delta: float = 12 / 255
    height_tolerance_m: float = 0.01
    variant: str = "placement_descent_v1"

    def __post_init__(self):
        if self.variant != "placement_descent_v1":
            raise ValueError("unknown placement descent variant")
        for name in ("release_z_max_m", "descent_speed_m_s", "max_descent_s", "activation_command_delta", "height_tolerance_m"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.descent_speed_m_s > 0.25:
            raise ValueError("descent faster than 0.25 m/s is not a placement descent")
        if self.activation_command_delta > 1:
            raise ValueError("activation_command_delta is a gripper command fraction")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("placement descent config must be an object")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


class PlacementDescentSupervisor:
    """States: idle -> descending -> at_release_height; yielded (timeout) until the policy re-closes."""

    def __init__(self, config: PlacementDescentConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.state = "idle"
        self.anchor = None          # held x/y/orientation (6-vector) while descending
        self.commanded_z = None
        self.started_at = None
        self.last_t = None
        self.reached_at = None
        self.last_event = None
        self.events = []
        self.yield_reason = None

    def _event(self, t, name, **values):
        self.last_event = name
        self.events.append(dict(t=float(t), event=name, **values))
        if len(self.events) > 200:
            del self.events[:100]

    @property
    def active(self):
        return self.state in ("descending", "at_release_height")

    @property
    def permission_allowed(self):
        """False only while actively holding the release for a descent."""
        return self.state != "descending"

    def update(self, t, *, measured_tcp, policy_grip, reference_command, in_volume, loaded):
        """Called once per servo tick before the release gate decides permission."""
        c = self.config
        tcp = np.asarray(measured_tcp, dtype=float)
        valid = tcp.shape == (6,) and np.isfinite(tcp).all()
        intent = (loaded and in_volume and valid and reference_command is not None
                  and np.isfinite(policy_grip) and np.isfinite(reference_command)
                  and reference_command - policy_grip >= c.activation_command_delta - 1e-12)
        if self.state == "yielded":
            if not intent:
                self.reset()
                self._event(t, "descent_rearmed_after_yield")
            return
        if not intent:
            if self.active:
                self._event(t, "descent_cancelled", reason="release_intent_withdrawn_or_outside_volume", z=float(tcp[2]) if valid else None)
                self.reset()
            return
        above = tcp[2] > c.release_z_max_m + c.height_tolerance_m
        if self.state == "idle":
            if above:
                self.state = "descending"
                self.anchor = tcp.copy()
                self.commanded_z = float(tcp[2])
                self.started_at = float(t)
                self.last_t = float(t)
                self._event(t, "descent_started", from_z=float(tcp[2]), release_z_max_m=c.release_z_max_m)
            return
        if self.state == "descending":
            if t - self.started_at > c.max_descent_s:
                self.state = "yielded"
                self.yield_reason = "descent_timeout"
                self._event(t, "descent_yielded", reason="descent_timeout", z=float(tcp[2]))
                return
            if not above:
                self.state = "at_release_height"
                self.reached_at = float(t)
                self._event(t, "release_height_reached", z=float(tcp[2]))
            return
        if self.state == "at_release_height" and tcp[2] > c.release_z_max_m + 3 * c.height_tolerance_m:
            # lifted back up (policy or disturbance): descend again
            self.state = "descending"
            self.anchor = tcp.copy()
            self.commanded_z = float(tcp[2])
            self.started_at = float(t)
            self.last_t = float(t)
            self._event(t, "descent_restarted", from_z=float(tcp[2]))

    def motion_target(self, t, measured_tcp, target):
        """Return the target to command this tick (the policy's own target when not active)."""
        if not self.active or self.anchor is None:
            return target
        out = np.asarray(self.anchor, dtype=float).copy()
        if self.state == "descending":
            dt = 0.0 if self.last_t is None else max(0.0, float(t) - self.last_t)
            self.last_t = float(t)
            self.commanded_z = max(self.config.release_z_max_m, self.commanded_z - self.config.descent_speed_m_s * dt)
        out[2] = float(self.commanded_z if self.commanded_z is not None else self.config.release_z_max_m)
        return out

    def diagnostics(self):
        return dict(variant=self.config.variant, state=self.state, permission_allowed=self.permission_allowed,
                    commanded_z_m=self.commanded_z, started_at_s=self.started_at, reached_at_s=self.reached_at,
                    yield_reason=self.yield_reason, event=self.last_event)
