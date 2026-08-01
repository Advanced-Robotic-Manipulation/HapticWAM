"""Signal conditioning for teleop leaders (pure numpy, unit-tested).

OneEuroFilter — the de-facto standard for human-motion teleop input
(Casiez et al., CHI 2012): an adaptive low-pass whose cutoff rises with
signal speed. Slow motion -> heavy smoothing (kills operator tremor and
encoder quantization), fast motion -> light smoothing (minimal lag).

AccelLimitedTracker — a per-joint double integrator with bounded velocity
and acceleration, critically-damped approach (sqrt deceleration profile,
no overshoot). Turns any step in the target — including the engage jump
when the operator dons the exoskeleton — into a smooth S-shaped move.
The felt "jerk" of raw clamping comes from unbounded acceleration; this
bounds it explicitly. Used for BOTH the ENGAGE glide (arm pose -> leader) and
the live TRACKING path: the sqrt braking law provably never overshoots, so it
cannot ring — a plain vel+accel clamp (deadbeat v_des = err/dt) overshoots ~2x
and oscillates on any step, which showed up on the rig as a ~1-2 Hz "wiggle"
when the leader-dropout extrapolation snapped the target back. The only cost is
a small steady-state tracking lag v^2/(2*a_max); a_max is picked to keep that
lag low while staying under the CB3 C153A3 acceleration ceiling.
"""

from __future__ import annotations

import numpy as np


def _alpha(cutoff_hz: np.ndarray | float, dt: float) -> np.ndarray | float:
    """First-order low-pass smoothing factor for a given cutoff and step."""
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vector one-euro filter. cutoff = min_cutoff + beta * |dx_filtered|."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3,
                 d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("min_cutoff/d_cutoff must be > 0, beta >= 0")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or t <= self._t:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return x.copy()
        dt = t - self._t
        self._t = t
        dx = (x - self._x) / dt
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        self._x = self._x + _alpha(cutoff, dt) * (x - self._x)
        return self._x.copy()


class AccelLimitedTracker:
    """Bounded-velocity / bounded-acceleration target tracker (no overshoot).

    Each step: v_des = clip(sign(e) * sqrt(2 * a_max * |e|), +-v_max) — the
    velocity that can still stop exactly at the target under a_max — then the
    actual velocity slews toward v_des at a_max. Integrates to position.
    """

    def __init__(self, v_max: float, a_max: float):
        if v_max <= 0 or a_max <= 0:
            raise ValueError("v_max and a_max must be > 0")
        self.v_max = v_max
        self.a_max = a_max
        self._q: np.ndarray | None = None
        self._v: np.ndarray | None = None

    @property
    def q(self) -> np.ndarray:
        assert self._q is not None, "step() before reset_to()"
        return self._q.copy()

    def reset_to(self, q: np.ndarray) -> None:
        """(Re)initialize at rest at q — e.g. the measured arm pose on engage."""
        self._q = np.asarray(q, dtype=np.float64).copy()
        self._v = np.zeros_like(self._q)

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        if self._q is None:
            self.reset_to(target)
            return self.q
        target = np.asarray(target, dtype=np.float64)
        err = target - self._q
        # discrete-time braking law: the continuous sqrt(2*a*|e|) profile
        # arrives with residual velocity under a finite step dt (integration
        # lag), so use its dt-aware root — guarantees v -> 0 exactly at the
        # target with steps of size dt
        half = 0.5 * self.a_max * dt
        v_mag = -half + np.sqrt(half * half + 2.0 * self.a_max * np.abs(err))
        v_des = np.clip(np.sign(err) * v_mag, -self.v_max, self.v_max)
        dv = np.clip(v_des - self._v, -self.a_max * dt, self.a_max * dt)
        self._v = self._v + dv
        self._q = self._q + self._v * dt
        # land exactly: if this step reached or crossed the target, settle on
        # it and stop (v there is tiny on the braking curve — a one-tick
        # <=2*a_max settle transient, imperceptible; kills micro-oscillation)
        crossed = (target - self._q) * err <= 0.0
        self._q[crossed] = target[crossed]
        self._v[crossed] = 0.0
        return self.q
