"""Descend-then-release supervisor for the placement phase (rig lineage, 09-12).

Measured problem (rig 2026-09-12, 31 latched carries, `docs/results/sim_zoo_20260912/README.md`
plus the forensics follow-up): once the gripper latch holds an object over the crate the
policies keep commanding descent at 32-68 mm/s straight through the demonstrated release
band (waffles release z median 105 mm, p10 81; Carton median 128, p10 118) down to
83-133 mm, where the 45 N wrench guard ends the episode. 19 of 31 carries never released.
Of the 12 that did, 5 still ended below 0.16 m and 3 of those on the wrench guard, because
the descent continued after the release. The demos do NOT press: their lowest TCP over the
crate IS the release height, so this is a controller problem, not a missing floor.

This supervisor is the smallest measured-feedback controller that fixes the descent. It
never looks at object state and never creates motion toward the crate:

1. **Crate-region z floor.** While the measured TCP is over the crate (y > y_crate_edge_m)
   the commanded z is floored at the task's p10 demonstrated release height
   (`release_z_m`). A floor only ever RAISES a command, so it cannot defeat a geometric
   safety clamp, and it is re-clamped by the SafetyMonitor afterwards. It applies whether
   or not the latch holds, which is what protects the post-release descents.
2. **Forced release.** Once the latch holds, the carry apex happened
   (`release_gate_z_m + lift_m`, the native latch's own lift gate) and the measured TCP is
   inside the release zone (y >= release_y_min_m, z <= release_gate_z_m) continuously for
   `dwell_s`, the latch is cleared and the gripper is commanded to the more open of the
   episode's open aperture and the policy's own request.
3. **Retract.** The release pose is held for `open_hold_s` (so the fingers actually open),
   then the TCP is raised `retract_m` at `retract_speed_m_s` with x/y/orientation held.
   The policy then resumes. Re-latching stays blocked until the tool leaves the crate
   region, so the latch cannot re-engage on the object that was just released.

Everything is opt-in: no config object, no supervisor, and the executor path is unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

VARIANT = "descend_then_release_v1"
STATES = ("idle", "watching", "opening", "retracting", "released")


@dataclass(frozen=True)
class DescendThenReleaseConfig:
    """Per-task placement geometry. Every distance is base-frame metres.

    `release_z_m` and `release_y_min_m` come from the task's successful teleop demos
    (`data/val124/tasks`): the p10 release height and the p10 release y. The p10 height is
    a floor on the COMMAND, i.e. the deepest the controller will ever ask for over the
    crate; the p10 y is where the demonstrations are already inside the crate, so a forced
    release there cannot drop the object on the near rim.
    """

    release_z_m: float
    task: str = ""
    release_y_min_m: float | None = None      # None -> y_crate_edge_m (region edge)
    y_crate_edge_m: float = -0.10             # measured TCP y above this = over the crate
    release_gate_z_m: float = 0.16            # no forced release above this height
    lift_m: float = 0.08                      # carry apex required above the gate
    dwell_s: float = 0.35
    open_hold_s: float = 0.3
    retract_m: float = 0.08
    retract_speed_m_s: float = 0.10
    z_tolerance_m: float = 0.005
    max_retract_s: float = 2.0
    variant: str = VARIANT

    def __post_init__(self):
        if self.variant != VARIANT:
            raise ValueError(f"unknown descend-then-release variant: {self.variant!r}")
        positive = ("release_z_m", "dwell_s", "open_hold_s", "retract_m",
                    "retract_speed_m_s", "z_tolerance_m", "max_retract_s")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("y_crate_edge_m", "release_gate_z_m", "lift_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.lift_m < 0:
            raise ValueError("lift_m must not be negative")
        if self.release_gate_z_m <= self.release_z_m:
            raise ValueError("release_gate_z_m must sit above the commanded z floor")
        if self.release_y_min_m is not None:
            if isinstance(self.release_y_min_m, bool) \
                    or not isinstance(self.release_y_min_m, (int, float)) \
                    or not np.isfinite(self.release_y_min_m):
                raise ValueError("release_y_min_m must be a finite number or null")
            if self.release_y_min_m < self.y_crate_edge_m:
                raise ValueError("release_y_min_m must lie inside the crate region")
        if self.retract_speed_m_s > 0.25:
            raise ValueError("a retract faster than 0.25 m/s is not a placement retract")
        if not isinstance(self.task, str):
            raise TypeError("task must be a string")

    @property
    def release_y_min(self) -> float:
        """Resolved forced-release y bound (defaults to the crate region edge)."""
        return float(self.y_crate_edge_m if self.release_y_min_m is None
                     else self.release_y_min_m)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("descend-then-release config must be an object")
        return cls(**values)

    @classmethod
    def from_spec(cls, spec, task):
        """Parse the on-disk config: shared fields plus a per-task `tasks` table.

        Returns None when the task has no entry — an unmeasured task (egg, whiteboard)
        leaves the supervisor OFF rather than guessing a release height.
        """
        if not isinstance(spec, dict):
            raise ValueError("descend-then-release config must be an object")
        shared = {k: v for k, v in spec.items() if k != "tasks"}
        tasks = spec.get("tasks")
        if not isinstance(tasks, dict):
            raise ValueError("descend-then-release config needs a per-task `tasks` table")
        entry = tasks.get(str(task))
        if entry is None:
            return None
        if not isinstance(entry, dict):
            raise ValueError(f"task entry for {task!r} must be an object")
        overlap = set(shared) & set(entry)
        if overlap:
            raise ValueError(f"task {task!r} redefines shared fields: {sorted(overlap)}")
        return cls(task=str(task), **shared, **entry)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DescentDecision:
    """What the executor must command this tick."""

    target: np.ndarray          # pose to command (the policy's own when `overrode` is False)
    grip: float | None          # forced gripper command, None = leave the policy's
    overrode: bool              # the pose was changed (floor, hold or retract)
    clear_latch: bool           # drop the aperture latch now
    block_latch: bool           # do not let the latch (re-)arm this tick
    state: str
    event: str | None = None


def make_placement_descent(config):
    """Shared validation; returns None when the supervisor is not configured."""
    if config is None:
        return None
    if isinstance(config, dict):
        config = DescendThenReleaseConfig.from_dict(config)
    if not isinstance(config, DescendThenReleaseConfig):
        raise TypeError("placement_descent must be a DescendThenReleaseConfig or dict")
    return DescendThenReleaseSupervisor(config)


class DescendThenReleaseSupervisor:
    """One carry's placement: watch -> open -> retract -> released -> (re-arm off crate).

    Pure state machine over measured feedback: it is driven by `step()` once per servo
    tick and owns no clock, no device and no thread.
    """

    def __init__(self, config: DescendThenReleaseConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = "idle"
        self.dwell_since: float | None = None
        self.z_max_since_latch = -np.inf
        self.anchor: np.ndarray | None = None
        self.anchor_z: float | None = None
        self.opened_at: float | None = None
        self.retract_started: float | None = None
        self.commanded_z: float | None = None
        self.open_command: float | None = None
        self.release_z: float | None = None
        self.release_y: float | None = None
        self.releases = 0
        self.last_t: float | None = None
        self.last_event: str | None = None
        self.events: list[dict] = []

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """True while the supervisor owns the commanded pose and the gripper."""
        return self.state in ("opening", "retracting")

    def _event(self, t, name, **values) -> str:
        self.last_event = name
        self.events.append(dict(t=float(t), event=name, state=self.state, **values))
        if len(self.events) > 200:
            del self.events[:100]
        return name

    def _open_value(self, policy_grip, open_aperture) -> float:
        """The MORE OPEN of the episode's open aperture and the policy's request.

        The gripper channel is 0 = open, 1 = closed, so more open is the smaller value.
        """
        values = [v for v in (open_aperture, policy_grip)
                  if v is not None and np.isfinite(v)]
        return float(np.clip(min(values), 0.0, 1.0)) if values else 0.0

    def _floor(self, target, over_crate):
        """Crate-region floor on the COMMANDED z. Returns (target, changed)."""
        if not over_crate or not np.isfinite(target[2]) \
                or target[2] >= self.config.release_z_m:
            return target, False
        target[2] = float(self.config.release_z_m)
        return target, True

    def _held(self, z):
        out = np.asarray(self.anchor, dtype=float).copy()
        out[2] = float(z)
        return out

    # ------------------------------------------------------------------
    def step(self, t, *, measured_tcp, latched, commanded_target, policy_grip,
             open_aperture) -> DescentDecision:
        """One servo tick. `commanded_target` is the pose the executor would send."""
        c = self.config
        t = float(t)
        target = np.array(commanded_target, dtype=float).reshape(-1)
        tcp = None
        if measured_tcp is not None:
            pose = np.asarray(measured_tcp, dtype=float).reshape(-1)
            if pose.shape == (6,) and np.isfinite(pose).all():
                tcp = pose
        latched = bool(latched)
        event = None

        # carry apex since the latch armed (the native latch's own lift gate)
        if not latched:
            self.z_max_since_latch = -np.inf
        elif tcp is not None:
            self.z_max_since_latch = max(self.z_max_since_latch, float(tcp[2]))

        over_crate = tcp is not None and float(tcp[1]) > c.y_crate_edge_m
        in_release_zone = (tcp is not None and float(tcp[1]) >= c.release_y_min
                           and float(tcp[2]) <= c.release_gate_z_m)
        lifted = self.z_max_since_latch >= c.release_gate_z_m + c.lift_m

        if self.state in ("idle", "watching"):
            eligible = latched and lifted and in_release_zone
            if not eligible:
                if self.state == "watching":
                    event = self._event(t, "release_dwell_reset",
                                        latched=latched, lifted=bool(lifted),
                                        in_release_zone=bool(in_release_zone))
                    self.state = "idle"
                self.dwell_since = None
            else:
                if self.dwell_since is None:
                    self.dwell_since = t
                    self.state = "watching"
                    event = self._event(t, "release_dwell_started",
                                        z=float(tcp[2]), y=float(tcp[1]))
                if t - self.dwell_since >= c.dwell_s - 1e-9:
                    self.anchor = tcp.copy()
                    self.anchor_z = max(float(tcp[2]), c.release_z_m)
                    self.commanded_z = self.anchor_z
                    self.open_command = self._open_value(policy_grip, open_aperture)
                    self.opened_at = t
                    self.release_z, self.release_y = float(tcp[2]), float(tcp[1])
                    self.releases += 1
                    self.state = "opening"
                    self.dwell_since = None
                    event = self._event(t, "forced_release", z=self.release_z,
                                        y=self.release_y, grip=self.open_command,
                                        dwell_s=c.dwell_s)
                    self.last_t = t
                    return DescentDecision(self._held(self.anchor_z), self.open_command,
                                           True, True, True, self.state, event)
            self.last_t = t
            target, changed = self._floor(target, over_crate)
            return DescentDecision(target, None, changed, False, False, self.state, event)

        if self.state == "opening":
            self.open_command = self._open_value(policy_grip, self.open_command)
            if t - self.opened_at >= c.open_hold_s - 1e-9:
                self.state = "retracting"
                self.retract_started = t
                self.commanded_z = self.anchor_z
                event = self._event(t, "retract_started", from_z=self.anchor_z,
                                    to_z=self.anchor_z + c.retract_m)
            else:
                self.last_t = t
                return DescentDecision(self._held(self.anchor_z), self.open_command,
                                       True, True, True, self.state, event)

        if self.state == "retracting":
            reached = (tcp is not None
                       and float(tcp[2]) >= self.anchor_z + c.retract_m - c.z_tolerance_m)
            timed_out = t - self.retract_started > c.max_retract_s
            if reached or timed_out:
                self.state = "released"
                event = self._event(t, "retract_complete" if reached else "retract_timeout",
                                    z=None if tcp is None else float(tcp[2]))
            else:
                dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
                self.commanded_z = min(self.anchor_z + c.retract_m,
                                       (self.commanded_z if self.commanded_z is not None
                                        else self.anchor_z) + c.retract_speed_m_s * dt)
                self.open_command = self._open_value(policy_grip, self.open_command)
                self.last_t = t
                return DescentDecision(self._held(self.commanded_z), self.open_command,
                                       True, True, True, self.state, event)

        # released: the policy owns motion again; the latch stays blocked until the
        # tool leaves the crate region, so it cannot re-engage on the placed object.
        block = bool(over_crate)
        if not over_crate:
            self.state = "idle"
            self.dwell_since = None
            self.anchor = self.anchor_z = self.commanded_z = None
            self.opened_at = self.retract_started = None
            self.open_command = None
            event = self._event(t, "supervisor_rearmed")
        self.last_t = t
        target, changed = self._floor(target, over_crate)
        return DescentDecision(target, None, changed, False, block, self.state, event)

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        c = self.config
        return {
            "variant": c.variant,
            "task": c.task,
            "state": self.state,
            "releases": self.releases,
            "release_z_m": c.release_z_m,
            "release_y_min_m": c.release_y_min,
            "y_crate_edge_m": c.y_crate_edge_m,
            "release_gate_z_m": c.release_gate_z_m,
            "dwell_s": c.dwell_s,
            "dwell_since_s": self.dwell_since,
            "z_max_since_latch_m": (None if not np.isfinite(self.z_max_since_latch)
                                    else float(self.z_max_since_latch)),
            "released_at_z_m": self.release_z,
            "released_at_y_m": self.release_y,
            "open_command": self.open_command,
            "commanded_z_m": self.commanded_z,
            "event": self.last_event,
            "events": self.events[-20:],
        }
