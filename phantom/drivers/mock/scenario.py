"""ContactScenario: the deterministic phase machine shared by ALL mock drivers
in a rig so their signals co-vary physically:

    approach -> onset -> hold -> slip -> release -> idle -> (repeat)

The scenario is the ground truth the mocks render from and the derived-channel
code must recover (the mock<->derived acceptance test), and the timeline the
SyntheticEpisodeGenerator scripts episodes with.

Key property for ACC: the wrist F/T ramp STARTS DURING APPROACH (ft_lead_s
before tactile onset) — the physical precursor structure the anticipatory gate
is designed to exploit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from phantom.config.model import EVENT_IDX

PHASES = ("approach", "onset", "hold", "slip", "release", "idle")

# scenario phase -> expected derived-event label while inside the phase
PHASE_EVENT = {
    "approach": EVENT_IDX["none"],
    "onset": EVENT_IDX["onset"],
    "hold": EVENT_IDX["hold"],
    "slip": EVENT_IDX["slip"],
    "release": EVENT_IDX["release"],
    "idle": EVENT_IDX["none"],
}


@dataclass(frozen=True)
class ScenarioState:
    t: float
    phase: str
    phase_progress: float            # 0..1 inside current phase
    blob_center_uv: tuple[float, float]   # normalized [-1,1]^2 (x=cols, y=rows)
    blob_radius: float               # normalized (fraction of half-extent)
    press_depth: float               # 0..1 indentation amplitude
    tangential_drift_uv: tuple[float, float]  # in-plane drift velocity (slip precursor)
    fz_total: float                  # total normal force, arbitrary mock units
    in_contact: bool


@dataclass(frozen=True)
class ScenarioTimings:
    approach_s: float = 1.5
    onset_s: float = 0.3
    hold_s: float = 1.5
    slip_s: float = 0.8
    release_s: float = 0.4
    idle_s: float = 1.5
    ft_lead_s: float = 0.4           # wrist F/T ramp starts this long before tactile onset

    @property
    def cycle_s(self) -> float:
        return (self.approach_s + self.onset_s + self.hold_s
                + self.slip_s + self.release_s + self.idle_s)


class ContactScenario:
    """Deterministic, seeded; state(t) is a pure function of t."""

    def __init__(self, seed: int = 0, timings: ScenarioTimings | None = None,
                 max_depth: float = 0.5, max_fz: float = 3.0):
        self.timings = timings or ScenarioTimings()
        self.max_depth = max_depth
        self.max_fz = max_fz
        rng = np.random.default_rng(seed)
        # per-cycle blob start positions / drift directions, precomputed deterministically
        self._centers = rng.uniform(-0.4, 0.4, size=(1024, 2))
        self._drift_angles = rng.uniform(0, 2 * math.pi, size=1024)

    # ------------------------------------------------------------------
    def phase_at(self, t: float) -> tuple[str, float, int]:
        """(phase, progress 0..1, cycle index) at absolute time t."""
        tm = self.timings
        cycle = int(t // tm.cycle_s)
        u = t % tm.cycle_s
        for name in PHASES:
            dur = getattr(tm, f"{name}_s")
            if u < dur:
                return name, u / dur, cycle
            u -= dur
        return "idle", 1.0, cycle

    def state(self, t: float) -> ScenarioState:
        tm = self.timings
        phase, prog, cycle = self.phase_at(t)
        c0 = self._centers[cycle % len(self._centers)]
        ang = self._drift_angles[cycle % len(self._drift_angles)]
        drift_dir = (math.cos(ang), math.sin(ang))

        depth = 0.0
        drift: tuple[float, float] = (0.0, 0.0)
        center = (float(c0[0]), float(c0[1]))
        in_contact = False

        if phase == "onset":
            depth = self.max_depth * prog
            in_contact = prog > 0.0
        elif phase == "hold":
            depth = self.max_depth
            in_contact = True
            # tangential loading builds up through hold (slip precursor)
            drift = (drift_dir[0] * 0.2 * prog, drift_dir[1] * 0.2 * prog)
        elif phase == "slip":
            depth = self.max_depth * (1.0 - 0.2 * prog)
            in_contact = True
            drift = (drift_dir[0] * 1.0, drift_dir[1] * 1.0)   # fast drift = slip
            center = (float(c0[0] + drift_dir[0] * 0.3 * prog),
                      float(c0[1] + drift_dir[1] * 0.3 * prog))
        elif phase == "release":
            depth = self.max_depth * 0.8 * (1.0 - prog)
            in_contact = prog < 0.95
            center = (float(c0[0] + drift_dir[0] * 0.3),
                      float(c0[1] + drift_dir[1] * 0.3))

        fz = self.max_fz * (depth / self.max_depth) if self.max_depth > 0 else 0.0
        radius = 0.25 * (0.5 + 0.5 * depth / max(self.max_depth, 1e-9)) if depth > 0 else 0.0
        return ScenarioState(
            t=t, phase=phase, phase_progress=prog,
            blob_center_uv=center, blob_radius=radius, press_depth=depth,
            tangential_drift_uv=drift, fz_total=fz, in_contact=in_contact,
        )

    # ------------------------------------------------------------------
    def wrist_ft_scale(self, t: float) -> float:
        """0..1 wrist-wrench envelope. Rises ft_lead_s BEFORE tactile onset."""
        tm = self.timings
        phase, prog, _ = self.phase_at(t)
        if phase == "approach":
            u = prog * tm.approach_s
            ramp_start = tm.approach_s - tm.ft_lead_s
            if u >= ramp_start:
                return 0.3 * (u - ramp_start) / max(tm.ft_lead_s, 1e-9)
            return 0.0
        if phase in ("onset", "hold", "slip"):
            return 0.3 + 0.7 * self.state(t).press_depth / max(self.max_depth, 1e-9)
        if phase == "release":
            return 0.3 * (1.0 - prog)
        return 0.0

    def expected_event(self, t: float) -> int:
        return PHASE_EVENT[self.phase_at(t)[0]]
