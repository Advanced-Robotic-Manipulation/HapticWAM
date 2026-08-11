"""Safety layer (pipeline.md §6e): checked EVERY executor tick before a
command is sent, strict priority over the governor. All events are
t_master-logged for the eval harness."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.recording.ringbuffer import SharedRingBuffer

log = logging.getLogger(__name__)


class SafetyAction(str, Enum):
    OK = "ok"
    CLAMP = "clamp"              # workspace clamp, continue
    STOP_EPISODE = "stop"        # controlled stop + abort trial
    PROTECTIVE_STOP = "pstop"    # robot did it; needs manual unlock


@dataclass
class SafetyEvent:
    t: float
    kind: str
    value: float
    action: SafetyAction


@dataclass
class SafetyVerdict:
    action: SafetyAction
    events: list[SafetyEvent] = field(default_factory=list)


class SafetyMonitor:
    def __init__(self, hw: HardwareConfig, rings: dict[str, SharedRingBuffer]):
        self.hw = hw
        self.rings = rings
        self.log_events: list[SafetyEvent] = []
        self._ring_stale_s = 3.0 / min(hw.recording.field_ds_rate_hz,
                                       hw.cameras.scene.fps)
        # Wrench guard state — same scheme as data_collect.safeguard.ArmGuard:
        # the CB3 "wrench" is a current-based estimate with a large pose-
        # dependent static bias plus acceleration spikes, so the raw value vs
        # an absolute limit false-trips on ordinary motion. Track a slow
        # rolling baseline (EMA, wrench_baseline_tau_s) and trip only on a
        # DEVIATION from it sustained for the debounce window
        # (wrench_debounce_ticks / control.action_rate_hz seconds — the ticks
        # are defined at the record-loop rate, while check() runs at the much
        # faster executor tick, so the debounce is time-based here). The
        # baseline is frozen while over-limit so a real collision is never
        # absorbed into it.
        self._wrench_base: np.ndarray | None = None
        self._wrench_over_since: float | None = None
        self._wrench_last_t: float | None = None

    # ------------------------------------------------------------------
    def check(self, t_now: float, tcp_target: np.ndarray) -> SafetyVerdict:
        hw = self.hw
        events: list[SafetyEvent] = []
        action = SafetyAction.OK

        # protective stop + wrist wrench (latest arm sample)
        ts, arm = self.rings["arm"].latest(1)
        if len(ts):
            if arm["protective_stop"][0]:
                events.append(SafetyEvent(t_now, "protective_stop", 1.0,
                                          SafetyAction.PROTECTIVE_STOP))
                action = SafetyAction.PROTECTIVE_STOP
            ft = np.asarray(arm["ft"][0], dtype=np.float64).reshape(-1)
            if self._wrench_base is None:
                self._wrench_base = ft.copy()   # episode starts at rest: bias
            dev = ft - self._wrench_base
            f_mag = float(np.linalg.norm(dev[:3]))
            t_mag = float(np.linalg.norm(dev[3:]))
            dt = (t_now - self._wrench_last_t) if self._wrench_last_t is not None else 0.0
            self._wrench_last_t = t_now
            debounce_s = (hw.safety.wrench_debounce_ticks
                          / hw.control.action_rate_hz)
            if f_mag > hw.safety.wrench_limit_N or t_mag > hw.safety.wrench_limit_Nm:
                if self._wrench_over_since is None:
                    self._wrench_over_since = t_now
                if t_now - self._wrench_over_since >= debounce_s:
                    events.append(SafetyEvent(t_now, "wrench_limit", max(f_mag, t_mag),
                                              SafetyAction.STOP_EPISODE))
                    action = _max(action, SafetyAction.STOP_EPISODE)
            else:
                self._wrench_over_since = None
                # adapt the baseline ONLY when calm — never track an event in
                if dt > 0.0:
                    alpha = min(1.0, dt / hw.safety.wrench_baseline_tau_s)
                    self._wrench_base += alpha * (ft - self._wrench_base)

        # fingertip force / indentation e-stop (teacher rigs always record tactile)
        for s in hw.tactile.sensors:
            ring = self.rings.get(f"tactile_{s.name}")
            if ring is None:
                continue
            ts_t, tac = ring.latest(1)
            if not len(ts_t):
                continue
            if t_now - ts_t[0] > self._ring_stale_s:
                events.append(SafetyEvent(t_now, f"tactile_{s.name}_stale",
                                          t_now - ts_t[0], SafetyAction.STOP_EPISODE))
                action = _max(action, SafetyAction.STOP_EPISODE)
                continue
            fields = np.asarray(tac["fields_ds"][0], dtype=np.float32)
            from phantom.data.derived import channel_slices
            ch = channel_slices(hw.tactile)
            if hw.tactile.force_calibrated:
                peak = float(np.abs(fields[..., ch["dist_force"]][..., 2]).max()
                             * hw.tactile.dist_force_unit_to_N)
                limit, kind = hw.safety.tactile_fz_limit_N, "tactile_fz"
            else:
                peak = float(np.abs(fields[..., ch["depth"]]).max())
                limit, kind = hw.safety.tactile_depth_limit, "tactile_depth"
            if peak > limit:
                events.append(SafetyEvent(t_now, kind, peak, SafetyAction.STOP_EPISODE))
                action = _max(action, SafetyAction.STOP_EPISODE)

        # workspace clamp on the commanded target
        if not hw.safety.workspace_m.contains(tcp_target[:3]):
            events.append(SafetyEvent(t_now, "workspace_clamp",
                                      float(np.linalg.norm(tcp_target[:3])),
                                      SafetyAction.CLAMP))
            action = _max(action, SafetyAction.CLAMP)

        self.log_events.extend(events)
        for e in events:
            log.warning("SAFETY %s value=%.3f -> %s", e.kind, e.value, e.action.value)
        return SafetyVerdict(action=action, events=events)

    def recovered(self, frac: float = 0.8) -> bool:
        """Hysteresis gate for resuming after a STOP_EPISODE: True when the
        protective stop is clear and wrist wrench + fingertip peaks are all
        below frac * their limits (avoids stop/resume chatter at the limit)."""
        hw = self.hw
        ts, arm = self.rings["arm"].latest(1)
        if len(ts):
            if arm["protective_stop"][0]:
                return False
            ft = np.asarray(arm["ft"][0], dtype=np.float64).reshape(-1)
            dev = ft - self._wrench_base if self._wrench_base is not None else ft
            if (np.linalg.norm(dev[:3]) > frac * hw.safety.wrench_limit_N
                    or np.linalg.norm(dev[3:]) > frac * hw.safety.wrench_limit_Nm):
                return False
        from phantom.data.derived import channel_slices
        ch = channel_slices(hw.tactile)
        for s in hw.tactile.sensors:
            ring = self.rings.get(f"tactile_{s.name}")
            if ring is None:
                continue
            ts_t, tac = ring.latest(1)
            if not len(ts_t):
                continue
            if time.perf_counter() - ts_t[0] > self._ring_stale_s:
                return False   # dead/stale sensor: no resume without tactile safety
            fields = np.asarray(tac["fields_ds"][0], dtype=np.float32)
            if hw.tactile.force_calibrated:
                peak = float(np.abs(fields[..., ch["dist_force"]][..., 2]).max()
                             * hw.tactile.dist_force_unit_to_N)
                limit = hw.safety.tactile_fz_limit_N
            else:
                peak = float(np.abs(fields[..., ch["depth"]]).max())
                limit = hw.safety.tactile_depth_limit
            if peak > frac * limit:
                return False
        return True

    def clamp_target(self, tcp_target: np.ndarray) -> np.ndarray:
        ws = self.hw.safety.workspace_m
        out = tcp_target.copy()
        out[0] = np.clip(out[0], *ws.x)
        out[1] = np.clip(out[1], *ws.y)
        out[2] = np.clip(out[2], *ws.z)
        return out


_ORDER = [SafetyAction.OK, SafetyAction.CLAMP, SafetyAction.STOP_EPISODE,
          SafetyAction.PROTECTIVE_STOP]


def _max(a: SafetyAction, b: SafetyAction) -> SafetyAction:
    return a if _ORDER.index(a) >= _ORDER.index(b) else b
