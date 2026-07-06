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
            ft = arm["ft"][0]
            f_mag = float(np.linalg.norm(ft[:3]))
            t_mag = float(np.linalg.norm(ft[3:]))
            if f_mag > hw.safety.wrench_limit_N or t_mag > hw.safety.wrench_limit_Nm:
                events.append(SafetyEvent(t_now, "wrench_limit", max(f_mag, t_mag),
                                          SafetyAction.STOP_EPISODE))
                action = _max(action, SafetyAction.STOP_EPISODE)

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
