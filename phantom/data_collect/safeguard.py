"""DM-Tac pad safeguard + basic arm guard for teleop collection.

TactileSafeguard (the headline feature, requirement 3):
 - watches both sensors' shared-memory rings on its own thread (~50 Hz —
   trip latency is bounded by the tactile sample rate, not this loop);
 - trips when any sensor's resultant force ‖F‖ = |getForce()[:3]| (calibrated
   Newtons per the SDK manual) exceeds `force_limit_n`, OR its peak
   indentation |depth| exceeds `depth_limit` (uncalibrated SDK units — the
   always-on second layer, and the only layer if a wrench sample is missing);
 - on trip it LATCHES and fires `on_trip` exactly once: the app freezes the
   arm streamer, opens the gripper fully, stops the episode as a failure and
   shows the panel warning. NOTHING auto-resumes: the operator clears the
   latch from the panel's Resume button (`reset()`), then collection
   continues — no process restart, no reload.
 - `enabled` and `force_limit_n` are runtime-mutable (panel POST
   /api/safeguard) — disabling clears any active latch.

ArmGuard (kept from the reference behaviour): protective stop and wrist-
wrench overload freeze teleop and auto-resume with 0.8x hysteresis once the
load is released / the pendant is cleared. These are arm-level events, not
pad-level — the explicit-resume requirement applies to the tactile safeguard.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from phantom.config.hardware import HardwareConfig

from phantom.data_collect.config import SafeguardConfig

log = logging.getLogger(__name__)


@dataclass
class TripInfo:
    t: float
    sensor: str
    kind: str          # "force" | "depth" | "stale"
    value: float
    limit: float

    def describe(self) -> str:
        if self.kind == "stale":
            return (f"DM-Tac safeguard: {self.sensor} stream STALE for "
                    f"{self.value:.2f}s — sensor/worker dead, pads unprotected")
        unit = "N" if self.kind == "force" else " (depth units)"
        return (f"DM-Tac safeguard: {self.sensor} {self.kind} "
                f"{self.value:.2f}{unit} > limit {self.limit:.2f}")


class TactileSafeguard:
    def __init__(self, hw: HardwareConfig, rings: dict, cfg: SafeguardConfig,
                 *, on_trip=None, check_rate_hz: float = 50.0):
        self.hw = hw
        self.rings = rings
        self.on_trip = on_trip           # callback(TripInfo) — called ONCE per latch
        self.check_rate_hz = check_rate_hz
        self._enabled = cfg.enabled
        self._force_limit_n = cfg.force_limit_n
        self._depth_limit = cfg.depth_limit
        self._tripped: TripInfo | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ch = None                  # channel_slices, lazy
        # a frozen ring hides overloads behind its last (benign) sample — a
        # dead worker must trip, not blind the guard (same policy as
        # phantom.deploy.safety); grace period so bring-up isn't a trip storm
        self._stale_s = 3.0 / hw.recording.field_ds_rate_hz
        self._armed_at = time.perf_counter()

    # -- runtime controls (panel) ---------------------------------------
    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def force_limit_n(self) -> float:
        with self._lock:
            return self._force_limit_n

    @property
    def depth_limit(self) -> float:
        with self._lock:
            return self._depth_limit

    @property
    def tripped(self) -> TripInfo | None:
        with self._lock:
            return self._tripped

    def configure(self, *, enabled: bool | None = None,
                  force_limit_n: float | None = None,
                  depth_limit: float | None = None) -> None:
        """Panel POST /api/safeguard — applies immediately."""
        with self._lock:
            if enabled is not None:
                self._enabled = bool(enabled)
                if not self._enabled:
                    self._tripped = None       # disabling clears the latch
            if force_limit_n is not None:
                if force_limit_n <= 0:
                    raise ValueError("force_limit_n must be > 0")
                self._force_limit_n = float(force_limit_n)
            if depth_limit is not None:
                if depth_limit <= 0:
                    raise ValueError("depth_limit must be > 0")
                self._depth_limit = float(depth_limit)

    def reset(self) -> None:
        """Panel Resume button: clear the latch; collection continues."""
        with self._lock:
            self._tripped = None
        log.info("safeguard latch cleared by operator")

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="tactile-safeguard")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None

    # -- checking --------------------------------------------------------
    def check_once(self, t_now: float | None = None) -> TripInfo | None:
        """One pass over both sensors; latches + fires on_trip on the first
        violation. Also callable directly from tests / the app loop."""
        t_now = time.perf_counter() if t_now is None else t_now
        with self._lock:
            if not self._enabled or self._tripped is not None:
                return self._tripped
            f_lim = self._force_limit_n
            d_lim = self._depth_limit
        info = self._scan(t_now, f_lim, d_lim)
        if info is None:
            return None
        with self._lock:
            if self._tripped is not None:      # lost the race to another caller
                return self._tripped
            self._tripped = info
        log.error(info.describe())
        if self.on_trip is not None:
            try:
                self.on_trip(info)
            except Exception:
                log.exception("safeguard on_trip callback failed")
        return info

    def _scan(self, t_now: float, f_lim: float, d_lim: float) -> TripInfo | None:
        if self._ch is None:
            from phantom.data.derived import channel_slices
            self._ch = channel_slices(self.hw.tactile)
        for s in self.hw.tactile.sensors:
            ring = self.rings.get(f"tactile_{s.name}")
            if ring is None:
                continue
            ts, d = ring.latest(1)
            if not len(ts):
                # never produced a sample: trip after the startup grace window
                age = t_now - self._armed_at
                if age > 10 * self._stale_s:
                    return TripInfo(t_now, s.name, "stale", age, self._stale_s)
                continue
            age = t_now - float(ts[0])
            if age > self._stale_s:
                return TripInfo(t_now, s.name, "stale", age, self._stale_s)
            wrench = np.asarray(d["wrench"][0], dtype=np.float32).reshape(-1)
            if wrench.size >= 3:
                f_mag = float(np.linalg.norm(wrench[:3]))
                if f_mag > f_lim:
                    return TripInfo(t_now, s.name, "force", f_mag, f_lim)
            fields = np.asarray(d["fields_ds"][0], dtype=np.float32)
            peak_depth = float(np.abs(fields[..., self._ch["depth"]]).max())
            if peak_depth > d_lim:
                return TripInfo(t_now, s.name, "depth", peak_depth, d_lim)
        return None

    def _run(self) -> None:
        period = 1.0 / self.check_rate_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.check_once(t0)
            except Exception:
                log.exception("safeguard check failed")
            wait = period - (time.perf_counter() - t0)
            if wait > 0:
                self._stop.wait(wait)


class ArmGuard:
    """Protective stop + wrist-wrench COLLISION detector, with auto-resume
    hysteresis (arm-level, distinct from the latched pad guard).

    The CB3 has no F/T sensor: getActualTCPForce is a current-based estimate
    with a large, pose-dependent static bias (~18 N / ~5 Nm at rest, growing in
    extended poses) plus acceleration-driven spikes. Checking that raw estimate
    against absolute limits false-tripped constantly (the bias alone eats most
    of the margin, and normal teleop accelerations spike it) — the field fix had
    been to raise the limit to 500 N / 100 Nm, which defeats the guard entirely.

    Instead this tracks a slow ROLLING BASELINE of the wrench (EMA, tau seconds)
    that follows the bias/pose drift out, and trips only on a DEVIATION from
    that baseline that exceeds the limit for `wrench_debounce_ticks` consecutive
    checks. The baseline is frozen while over-limit so a real collision is never
    absorbed into it. Result: transient acceleration spikes and slow pose bias
    are ignored; a sudden sustained wrench (a crash) still trips, at a limit low
    enough to be protective again."""

    def __init__(self, hw: HardwareConfig, rings: dict):
        self.hw = hw
        self.rings = rings
        self._base: np.ndarray | None = None    # rolling baseline wrench (6,)
        self._over = 0                           # consecutive over-limit checks

    def _arm(self):
        ring = self.rings.get("arm")
        if ring is None:
            return None
        ts, d = ring.latest(1)
        return d if len(ts) else None

    def _alpha(self) -> float:
        """Per-check EMA factor (check() runs at the record-loop rate)."""
        dt = 1.0 / self.hw.control.action_rate_hz
        return min(1.0, dt / self.hw.safety.wrench_baseline_tau_s)

    def _deviation(self, ft: np.ndarray) -> tuple[float, float]:
        base = self._base if self._base is not None else ft
        dev = ft - base
        return float(np.linalg.norm(dev[:3])), float(np.linalg.norm(dev[3:]))

    def check(self) -> str | None:
        """"pstop" | "wrench" | None."""
        d = self._arm()
        if d is None:
            return None
        if d["protective_stop"][0]:
            return "pstop"
        ft = np.asarray(d["ft"][0], dtype=np.float64).reshape(-1)
        if self._base is None:
            self._base = ft.copy()
        dF, dM = self._deviation(ft)
        s = self.hw.safety
        if dF > s.wrench_limit_N or dM > s.wrench_limit_Nm:
            self._over += 1
        else:
            self._over = 0
            # adapt the baseline ONLY when calm — never track an event into it
            self._base += self._alpha() * (ft - self._base)
        return "wrench" if self._over >= s.wrench_debounce_ticks else None

    def recovered(self, frac: float = 0.8) -> bool:
        d = self._arm()
        if d is None:
            return False
        if d["protective_stop"][0]:
            return False
        ft = np.asarray(d["ft"][0], dtype=np.float64).reshape(-1)
        dF, dM = self._deviation(ft)
        s = self.hw.safety
        return dF <= frac * s.wrench_limit_N and dM <= frac * s.wrench_limit_Nm
