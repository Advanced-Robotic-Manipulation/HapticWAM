"""Safety layer (pipeline.md §6e): checked EVERY executor tick before a
command is sent, strict priority over the governor. All events are
t_master-logged for the eval harness."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from phantom.config.hardware import HardwareConfig, WorkspaceBox
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
    # A sustained condition (a clamp, a stale ring) re-fires on EVERY executor
    # tick. log_events keeps one event per CONDITION and counts the ticks here,
    # so EpisodeResult.safety_events reads "1 clamp" instead of "15000 clamps".
    count: int = 1
    t_last: float = 0.0


@dataclass
class SafetyVerdict:
    action: SafetyAction
    events: list[SafetyEvent] = field(default_factory=list)


def camera_stale_s(hw: HardwareConfig) -> float:
    """How old the newest scene frame may be before the episode is stopped.

    Deliberately looser than the tactile ring threshold: dropping a few USB
    frames is routine on the rig, while a wedged librealsense pipeline is a
    multi-second condition. 0.5 s is >=15 frames at 30 fps."""
    return max(0.5, 10.0 / hw.cameras.scene.fps)


def arm_stale_s(hw: HardwareConfig) -> float:
    """How old the newest arm sample may be before the episode is stopped.

    The arm ring is the ONLY source of tcp_pose, tcp_speed, the wrist F/T
    window and the protective-stop flag. If the RTDE-receive worker dies or
    its stream stalls, the ring keeps serving the pre-stall sample and the
    policy replans, the governor scales and the wrench guard all run on a
    frozen robot state with nothing detecting it (Codex review 2026-08-27) —
    the same failure the camera guard above was written for.

    Same shape as camera_stale_s: a generous floor over a rate-derived bound.
    The 0.5 s floor is 250 missed samples at the 500 Hz e-series default (62
    on a cb3 at 125 Hz) — deliberately loose so a GIL-starved mock poller or a
    scheduling hiccup never false-trips, while the real condition (a dead
    worker process / a lost RTDE stream) is unbounded. Below 100 Hz the
    rate-derived term takes over at 50 samples."""
    return max(0.5, 50.0 / hw.arm.rtde_receive_hz)


# A sustained condition logs on its rising edge and then at most this often.
_EVENT_LOG_PERIOD_S = 1.0
# Hard ceiling on retained distinct events (an episode that produces this many
# separate conditions is already pathological; counts still accumulate).
_MAX_LOG_EVENTS = 1000


class SafetyMonitor:
    def __init__(self, hw: HardwareConfig, rings: dict[str, SharedRingBuffer]):
        self.hw = hw
        self.rings = rings
        self.log_events: list[SafetyEvent] = []
        self.dropped_events = 0
        # kind -> the retained SafetyEvent for the condition currently active,
        # plus when it was last logged. Cleared as soon as a tick passes
        # without that kind, so the next occurrence is a fresh rising edge.
        self._active: dict[str, tuple[SafetyEvent, float]] = {}
        self._ring_stale_s = 3.0 / min(hw.recording.field_ds_rate_hz,
                                       hw.cameras.scene.fps)
        self._cam_stale_s = camera_stale_s(hw)
        self._arm_stale_s = arm_stale_s(hw)
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
            # Arm-stream freshness FIRST: everything below (protective stop,
            # wrench deviation) and everything the executor does this tick
            # reads this one frozen sample if the RTDE-receive worker died or
            # its stream stalled. Same treatment as a stale tactile ring or a
            # wedged scene camera.
            arm_age = t_now - float(ts[0])
            if arm_age > self._arm_stale_s:
                events.append(SafetyEvent(t_now, "arm_stale", arm_age,
                                          SafetyAction.STOP_EPISODE))
                action = _max(action, SafetyAction.STOP_EPISODE)
            if arm["protective_stop"][0]:
                events.append(SafetyEvent(t_now, "protective_stop", 1.0,
                                          SafetyAction.PROTECTIVE_STOP))
                action = SafetyAction.PROTECTIVE_STOP
            # singularity whip detector: measured joint speed, not commanded —
            # see SafetyConfig.joint_speed_stop_rad_s. No debounce: one tick
            # over 3 rad/s is already a whip, and a spurious stop is benign
            # next to 400 ms of uncontrolled wrist at 7 rad/s.
            # wrist-extension (elbow-straight) guard — see
            # SafetyConfig.wrist_extension_stop_m. Fires 0.3-1.4 s BEFORE an
            # IK branch flip can even be attempted; non-letgo.
            wd_max = hw.safety.wrist_extension_stop_m
            q_raw = arm.get("q") if hasattr(arm, "get") else None
            if wd_max is not None and q_raw is not None:
                a2, a3, d4 = hw.safety.ur_dh_a2_a3_d4_m
                q_elbow = float(np.asarray(q_raw[0]).reshape(-1)[2])
                wd = float(np.sqrt(a2 * a2 + a3 * a3
                                   + 2.0 * a2 * a3 * np.cos(q_elbow)
                                   + d4 * d4))
                if wd > wd_max:
                    events.append(SafetyEvent(t_now, "wrist_extension", wd,
                                              SafetyAction.STOP_EPISODE))
                    action = _max(action, SafetyAction.STOP_EPISODE)
            qd_raw = arm.get("qd") if hasattr(arm, "get") else None
            qd = (np.asarray(qd_raw[0], dtype=np.float64).reshape(-1)
                  if qd_raw is not None else np.zeros(0))
            qd_max = float(np.max(np.abs(qd))) if qd.size else 0.0
            if qd_max > hw.safety.joint_speed_stop_rad_s:
                events.append(SafetyEvent(t_now, "joint_speed", qd_max,
                                          SafetyAction.STOP_EPISODE))
                action = _max(action, SafetyAction.STOP_EPISODE)
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

        # scene-camera freshness: a wedged RealSense pipeline stops pushing and
        # the ring keeps serving the pre-stall frame, so every replan conditions
        # on a frozen image while the arm keeps being driven (blocker
        # 2026-08-26). Same treatment as a stale tactile ring. Read the ring's
        # timestamp ONLY — latest(1) here would copy ~1 MB of pixels per tick.
        cam_ring = self.rings.get("camera_scene")
        if cam_ring is not None:
            ts_c = cam_ring.latest_ts()
            if ts_c is not None and t_now - ts_c > self._cam_stale_s:
                events.append(SafetyEvent(t_now, "camera_scene_stale",
                                          t_now - ts_c, SafetyAction.STOP_EPISODE))
                action = _max(action, SafetyAction.STOP_EPISODE)

        # reach clamp on the commanded target (see clamp_target)
        if (hw.safety.reach_clamp_m is not None
                and float(np.linalg.norm(tcp_target[:3])) > hw.safety.reach_clamp_m):
            events.append(SafetyEvent(t_now, "reach_clamp",
                                      float(np.linalg.norm(tcp_target[:3])),
                                      SafetyAction.CLAMP))
            action = _max(action, SafetyAction.CLAMP)
        # workspace clamp on the commanded target
        if not hw.safety.workspace_m.contains(tcp_target[:3]):
            events.append(SafetyEvent(t_now, "workspace_clamp",
                                      float(np.linalg.norm(tcp_target[:3])),
                                      SafetyAction.CLAMP))
            action = _max(action, SafetyAction.CLAMP)
        # per-task hitbox: leaving the demo envelope is not something to clamp
        # and continue from — the policy is already lost; stop the episode
        # Evaluated on the CLAMPED target: the z floor is a clamp, not a stop —
        # a policy that finally descends deep enough must be pinned at the
        # floor and allowed to close there, never stopped for it (review
        # 2026-08-28: floor == hitbox lower edge turned every deep descent
        # into a STOP and would have biased the A/B against the better arm)
        hb = hw.safety.hitbox_m
        if hb is not None:
            p3 = self.clamp_target(np.asarray(tcp_target, dtype=np.float64))[:3]
            if not hb.contains(p3):
                # A pure TOP-face exit is a successful lift outgrowing the
                # demo envelope, not the policy getting lost sideways — stop
                # the episode but do NOT let go: "hitbox_exit" is a let-go
                # reason in the executor, and opening the fingers half a
                # metre up drops a held object (rig 2026-09-01: the first
                # tactile-confirmed grasp was dropped exactly this way).
                only_top = (p3[2] > hb.z[1]
                            and hb.x[0] <= p3[0] <= hb.x[1]
                            and hb.y[0] <= p3[1] <= hb.y[1])
                kind = "hitbox_exit_top" if only_top else "hitbox_exit"
                events.append(SafetyEvent(t_now, kind,
                                          float(np.linalg.norm(tcp_target[:3])),
                                          SafetyAction.STOP_EPISODE))
                action = _max(action, SafetyAction.STOP_EPISODE)

        self._record(t_now, events)
        return SafetyVerdict(action=action, events=events)

    def _record(self, t_now: float, events: list[SafetyEvent]) -> None:
        """Retain and log CONDITIONS, not ticks.

        check() runs every executor tick (500 Hz), so a sustained clamp used to
        append ~15k events per episode and issue ~15k warnings — synchronous
        stderr writes from inside the 2 ms servo loop, and a safety_events count
        that reported ticks instead of problems. Each kind is now stored once on
        its rising edge, its `count` incremented while it persists, and re-logged
        at most every _EVENT_LOG_PERIOD_S."""
        seen = set()
        for e in events:
            seen.add(e.kind)
            prev = self._active.get(e.kind)
            if prev is None:
                if len(self.log_events) < _MAX_LOG_EVENTS:
                    self.log_events.append(e)
                else:
                    self.dropped_events += 1
                e.t_last = t_now
                self._active[e.kind] = (e, t_now)
                log.warning("SAFETY %s value=%.3f -> %s", e.kind, e.value,
                            e.action.value)
                continue
            held, last_log = prev
            held.count += 1
            held.t_last = t_now
            held.value = e.value            # latest magnitude of the condition
            if t_now - last_log >= _EVENT_LOG_PERIOD_S:
                self._active[e.kind] = (held, t_now)
                log.warning("SAFETY %s value=%.3f -> %s (sustained %.1fs, %d ticks)",
                            held.kind, held.value, held.action.value,
                            t_now - held.t, held.count)
        for kind in [k for k in self._active if k not in seen]:
            self._active.pop(kind, None)    # condition cleared: re-arm the edge

    def recovered(self, frac: float = 0.8) -> bool:
        """Hysteresis gate for resuming after a STOP_EPISODE: True when the
        protective stop is clear and wrist wrench + fingertip peaks are all
        below frac * their limits (avoids stop/resume chatter at the limit)."""
        hw = self.hw
        ts, arm = self.rings["arm"].latest(1)
        if len(ts):
            if time.perf_counter() - float(ts[0]) > self._arm_stale_s:
                return False   # dead/stale arm stream: no resume without it
            if arm["protective_stop"][0]:
                return False
            ft = np.asarray(arm["ft"][0], dtype=np.float64).reshape(-1)
            dev = ft - self._wrench_base if self._wrench_base is not None else ft
            if (np.linalg.norm(dev[:3]) > frac * hw.safety.wrench_limit_N
                    or np.linalg.norm(dev[3:]) > frac * hw.safety.wrench_limit_Nm):
                return False
            # the two 09-01 STOP kinds must also clear, or the shared
            # teleop/collection loop livelocks in safety_hold after the
            # operator lifts near full extension (verification 09-01):
            # wrist back inside the guard with the same `frac` hysteresis,
            # joints back to quasi-static speed
            wd_max = hw.safety.wrist_extension_stop_m
            q_raw = arm.get("q") if hasattr(arm, "get") else None
            if wd_max is not None and q_raw is not None:
                a2, a3, d4 = hw.safety.ur_dh_a2_a3_d4_m
                q_elbow = float(np.asarray(q_raw[0]).reshape(-1)[2])
                wd = float(np.sqrt(a2 * a2 + a3 * a3
                                   + 2.0 * a2 * a3 * np.cos(q_elbow)
                                   + d4 * d4))
                if wd > wd_max - (1.0 - frac) * 0.05:   # ~10 mm hysteresis
                    return False
            qd_raw = arm.get("qd") if hasattr(arm, "get") else None
            if qd_raw is not None:
                qd = np.asarray(qd_raw[0], dtype=np.float64).reshape(-1)
                if qd.size and float(np.max(np.abs(qd))) \
                        > frac * hw.safety.joint_speed_stop_rad_s:
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
        # Reach clamp FIRST (rig 2026-09-01 #2): commanded TCP radius from the
        # base is capped below the arm's singular zone — near full extension
        # servoL's IK turns a legal Cartesian step into a wrist whip
        # (measured: whip at 623 mm on the UR3; demos p95 = 605, tail 636).
        # Radial scaling, so the direction of the command is preserved and
        # the policy simply cannot extend further.
        r_max = self.hw.safety.reach_clamp_m
        r = float(np.linalg.norm(out[:3]))
        if r_max is not None and r > r_max:
            out[:3] *= r_max / r
        out[0] = np.clip(out[0], *ws.x)
        out[1] = np.clip(out[1], *ws.y)
        out[2] = np.clip(out[2], *ws.z)
        return out


_ORDER = [SafetyAction.OK, SafetyAction.CLAMP, SafetyAction.STOP_EPISODE,
          SafetyAction.PROTECTIVE_STOP]


def _max(a: SafetyAction, b: SafetyAction) -> SafetyAction:
    return a if _ORDER.index(a) >= _ORDER.index(b) else b


def apply_z_floor(hw: HardwareConfig, floor_m: float) -> HardwareConfig:
    """Copy of `hw` whose workspace z lower bound is RAISED to floor_m.

    The per-task no-go floor (rig 2026-08-28: a policy drove the gripper
    40 mm below the lowest demo height and slammed the table): the executor
    clamps every commanded target to the workspace box, so a floor at
    (demo z_min - margin) stops the descent there and lets x/y continue.
    Never lowers the configured bound."""
    ws = hw.safety.workspace_m
    lo = max(float(ws.z[0]), float(floor_m))
    if not np.isfinite(lo) or lo >= ws.z[1]:
        raise ValueError(f"z floor {floor_m} m is not below the workspace ceiling {ws.z[1]} m")
    box = WorkspaceBox(x=ws.x, y=ws.y, z=(lo, float(ws.z[1])))
    return hw.model_copy(update={"safety": hw.safety.model_copy(update={"workspace_m": box})})


def apply_hitbox(hw: HardwareConfig, lo, hi, margin_m: float) -> HardwareConfig:
    """Copy of `hw` with a STOP hitbox = [lo - margin, hi + margin] per axis,
    intersected with the workspace box (the hitbox can only be tighter).
    Apply BEFORE apply_z_floor so the hitbox floor sits below the clamp floor."""
    lo = np.asarray(lo, dtype=np.float64) - float(margin_m)
    hi = np.asarray(hi, dtype=np.float64) + float(margin_m)
    ws = hw.safety.workspace_m
    box = {}
    for i, ax in enumerate("xyz"):
        wlo, whi = getattr(ws, ax)
        a, b = max(float(lo[i]), float(wlo)), min(float(hi[i]), float(whi))
        if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
            raise ValueError(f"hitbox {ax} bounds collapse: ({a}, {b})")
        box[ax] = (a, b)
    hb = WorkspaceBox(**box)
    return hw.model_copy(update={"safety": hw.safety.model_copy(update={"hitbox_m": hb})})


def apply_tcp_speed_limit(hw: HardwareConfig, v_m_s: float) -> HardwareConfig:
    """Copy of `hw` whose executor TCP speed cap is LOWERED to v_m_s (never raised)."""
    lim = hw.arm.limits
    v = min(float(lim.tcp_speed_m_s), float(v_m_s))
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"bad tcp speed limit {v_m_s}")
    new_lim = lim.model_copy(update={"tcp_speed_m_s": v})
    return hw.model_copy(update={"arm": hw.arm.model_copy(update={"limits": new_lim})})
