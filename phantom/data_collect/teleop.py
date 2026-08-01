"""Device-rate teleop command path — the fix for the "awful teleop".

The reference stack (phantom/scripts/record_episodes.py) updated the servo
streamer's target from the 10 Hz record loop, so the arm chased a 100 ms
staircase through an acceleration tracker capped at 1 rad/s behind a 1 Hz
one-euro filter: 200-400 ms of stacked lag plus 10 Hz surge/brake lurch.
The known-good vendored main11.py sends every fresh leader sample straight
to servoJ(lookahead=0.1, gain=200) — the controller does the smoothing.

DirectServoStreamer restores that: a thread at control_rate_hz pulls
EchoTeleop.latest_q_target() DIRECTLY every cycle (the leader reader caches
at ~100 Hz), so the freshest filtered sample reaches servoJ every cycle. The
record loop is not in the command path at all — it only reads `last_cmd`
for logging the absolute action.

TRACK uses the sqrt-braking AccelLimitedTracker (v_max above human motion,
a_max sized for a small steady lag). It provably never overshoots, so it
never rings — the earlier deadbeat vel+accel clamp overshot ~2x and oscillated
whenever the dropout extrapolation snapped the target back, felt on the rig as
a ~1-2 Hz "wiggle". The accel cap also keeps every servoJ command feasible
(commanded accel <= a_max, under the CB3 C153A3 ceiling), taming the leader's
~0.5 deg / ~360 Hz rest jitter.

Engaging (session start, safeguard resume, post-protective-stop) uses the
slow AccelLimitedTracker glide (engage_v_max) from the measured arm pose to
the leader pose and hands over to TRACK only once converged — no jump by
construction, unlike main11 which trusts the operator to zero the glove.

Safety interactions (thread-safe; the safeguard thread calls hold()):
 - hold()/resume(): freeze at the current commanded pose / re-engage.
   ALL phase transitions inside the loop are compare-and-set under the lock
   so a concurrent hold() can never be stomped by an in-flight ENGAGE->TRACK
   or reseed write, and the servo write itself re-checks HOLD.
 - workspace hold: freezes at the last commanded pose whose measured TCP was
   INSIDE the box by a safety margin (freezing at the crossing pose leaves
   the commanded target OUTSIDE and can deadlock the resume check); the arm
   therefore retreats to a known-inside pose and resumes when the TCP has
   been back inside for a few consecutive cycles.
 - protective stop: while active, no commands; when it clears, the control
   script is dead — reconnect_control() is invoked (when the driver has it)
   before re-engaging from the measured pose.
 - ctl_lock serializes every control-interface call (servo_j here, zero_ft
   in the session loop) — ur_rtde's RTDEControlInterface is not thread-safe.

Arm state (workspace hold, protective stop, measured q) is read from the
session's shared-memory "arm" ring; the 8-getter RTDE get_state() is never
called from this thread. Without rings (unit tests) it falls back to
arm.get_state().
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from enum import Enum

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.teleop.filters import AccelLimitedTracker

from phantom.data_collect.config import TeleopTuning

log = logging.getLogger(__name__)


def enable_fine_timer() -> None:
    """Windows quantizes sub-16 ms sleeps to the 15.6 ms system tick, which
    would turn the 125 Hz streamer into ~60 Hz — request 1 ms resolution.
    (The rig NUC runs Linux; this matters for Windows bench runs.)"""
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)


class TrackPhase(str, Enum):
    ENGAGE = "engage"     # slow glide toward the leader pose
    TRACK = "track"       # transparent device-rate tracking (slew-limited)
    HOLD = "hold"         # frozen (safeguard / operator hold)
    FAULT = "fault"       # control script lost - awaits an operator re-engage


class DirectServoStreamer:
    def __init__(self, hw: HardwareConfig, arm, leader, tuning: TeleopTuning,
                 *, rings: dict | None = None):
        """leader: object with latest_q_target() -> np.ndarray | None
        (EchoTeleop, or any TeleopDevice exposing the same)."""
        assert hw.teleop is not None and hw.teleop.echo is not None
        self.hw = hw
        self.arm = arm
        self.leader = leader
        self.tuning = tuning
        self.rings = rings
        self.rate_hz = hw.teleop.echo.control_rate_hz
        # ENGAGE-only glide (sqrt-braking is fine for engaging, laggy for tracking)
        self.tracker = AccelLimitedTracker(v_max=tuning.engage_v_max_rad_s,
                                           a_max=tuning.a_max_rad_s2)
        # TRACK path: sqrt-braking accel-limited tracker. It CANNOT overshoot, so
        # it never rings — a plain deadbeat vel+accel clamp overshoots ~2x and
        # oscillated on the rig ("wiggle") whenever the dropout extrapolation
        # snapped the target back. a_max doubles as the servoJ-feasibility cap
        # (commanded accel <= a_max, well under the CB3 C153A3 ceiling) and sets
        # the small steady tracking lag v^2/(2*a_max); see track_a_max_rad_s2.
        self.track = AccelLimitedTracker(v_max=tuning.v_max_rad_s,
                                         a_max=tuning.track_a_max_rad_s2)
        # serializes control-interface calls (servo_j vs zero_ft etc.)
        self.ctl_lock = threading.Lock()
        self._phase = TrackPhase.ENGAGE
        self._hold_q: np.ndarray | None = None
        self._safe_q: np.ndarray | None = None      # last commanded pose whose
        self._ws_hold = False                       #   TCP was inside (marginal)
        self._ws_ok_cycles = 0
        self._last_cmd: np.ndarray | None = None
        # leader drop-out extrapolation (Echo firmware stalls ~100 ms ~2x/s):
        # predict the target forward at its last filtered velocity across a gap
        # instead of freezing then jumping.
        self._pred_q: np.ndarray | None = None
        self._pred_v: np.ndarray | None = None
        self._last_sample_t = 0.0
        self._stale_since: float | None = None
        self._stale_reengage = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        # bounded control-reconnect after a protective stop (don't spin forever
        # rebuilding the RTDE control interface if the robot never comes back)
        self._reconnect_fails = 0
        self._max_reconnect_fails = 20
        # set when a servo_j is rejected because the control script died
        self._ctl_lost = False

    # -- state read (ring-first, driver fallback) -----------------------
    def _arm_sample(self):
        """(q, tcp_pose, protective_stop) from the arm ring, else driver."""
        if self.rings is not None and "arm" in self.rings:
            ts, d = self.rings["arm"].latest(1)
            if len(ts):
                return (np.asarray(d["q"][0], dtype=np.float64),
                        np.asarray(d["tcp_pose"][0], dtype=np.float64),
                        bool(d["protective_stop"][0]))
        st = self.arm.get_state()
        return st.q, st.tcp_pose, st.protective_stop

    # -- leader target with drop-out extrapolation ----------------------
    def _leader_target(self, now: float, period: float) -> np.ndarray | None:
        """Freshest leader target, bridged across serial drop-outs.

        While a fresh device sample keeps arriving we return it verbatim. When
        the Echo stalls (no new sample this cycle) we keep gliding the target at
        its last filtered velocity, easing off linearly over extrap_cap_s so we
        neither freeze (the "stall") nor overshoot if the hand actually stopped
        during the gap. Past stale_reengage_s we hold the prediction (no
        runaway) and flag a re-engage so the arm glides — not lurches — back to
        the leader when data returns. Leaders without a stamped API (tests) fall
        back to the plain latest target (no extrapolation)."""
        stamped = getattr(self.leader, "latest_target_stamped", None)
        if stamped is None:
            return self.leader.latest_q_target()
        s = stamped()
        if s is None:
            return None
        q, v, t = s
        if t != self._last_sample_t:                       # fresh sample
            gap = (t - self._last_sample_t) if self._last_sample_t else 0.0
            self._last_sample_t = t
            self._pred_q = q
            self._pred_v = v
            self._stale_since = None
            if gap >= self.tuning.stale_reengage_s:
                self._stale_reengage = True
            return q
        # stale: no new sample this cycle -> extrapolate (or hold past the cap)
        if self._stale_since is None:
            self._stale_since = now
        dt_stale = now - self._stale_since
        cap = self.tuning.extrap_cap_s
        if dt_stale < cap and self._pred_q is not None and self._pred_v is not None:
            decay = 1.0 - dt_stale / cap
            self._pred_q = self._pred_q + self._pred_v * period * decay
        return self._pred_q

    # -- public control (thread-safe) -----------------------------------
    @property
    def phase(self) -> TrackPhase:
        return self._phase

    @property
    def in_workspace_hold(self) -> bool:
        return self._ws_hold

    @property
    def last_cmd(self) -> np.ndarray | None:
        """The absolute joint target commanded on the most recent cycle —
        what the record loop logs as the (absolute) action."""
        with self._lock:
            return None if self._last_cmd is None else self._last_cmd.copy()

    def hold(self) -> None:
        """Freeze NOW at the last commanded pose (safeguard trip).
        Takes priority over every in-loop transition (compare-and-set there)."""
        with self._lock:
            self._hold_q = (self._last_cmd.copy() if self._last_cmd is not None
                            else None)
            self._phase = TrackPhase.HOLD

    def reengage(self) -> tuple[bool, str]:
        """Operator-driven recovery from FAULT (panel "Re-engage arm").

        Verifies the robot is genuinely ready, restores + VERIFIES the control
        script, then glides in from the CURRENT measured pose (never a jump -
        the arm was probably re-positioned by hand on the pendant). The session
        keeps running: recording can continue with the next episode."""
        with self._lock:
            if self._phase is TrackPhase.HOLD:
                # HOLD is the latched tactile-safeguard freeze: re-engaging
                # here would move the arm with the e-stop latched and blind
                return False, ("safeguard hold latched — press Resume "
                               "collection to clear it first")
        ready = getattr(self.arm, "is_ready_for_control", None)
        if ready is not None:
            ok, why = ready()
            if not ok:
                return False, why
        try:
            with self.ctl_lock:
                self.arm.reconnect_control()
        except Exception as e:
            return False, str(e)
        try:
            q_meas, _, _ = self._arm_sample()
            self.tracker.reset_to(q_meas)
        except Exception as e:
            return False, f"cannot read the arm pose: {e}"
        with self._lock:
            if self._phase is TrackPhase.HOLD:
                # a safeguard trip landed while we were reconnecting: HOLD wins
                return False, "safeguard tripped during re-engage — arm stays held"
            self._ctl_lost = False
            self._hold_q = None
            self._phase = TrackPhase.ENGAGE
            self._reseed_flag = True
        self._reconnect_fails = 0
        log.info("operator re-engage OK - gliding to the leader pose")
        return True, ""

    def resume(self) -> None:
        """Leave HOLD via a fresh slow engage toward the current leader pose."""
        with self._lock:
            if self._phase is TrackPhase.HOLD:
                self._phase = TrackPhase.ENGAGE
                self._hold_q = None
                self._reseed_flag = True
        # tracker re-seeds from the measured pose inside the loop

    def start(self) -> None:
        # Refuse a double start: a second live thread would servo the same arm
        # AND run its own reconnect, i.e. two control interfaces on one robot
        # (the double-engage that preceded the rtde_control segfault).
        if self._thread is not None and self._thread.is_alive():
            log.warning("streamer already running — ignoring duplicate start()")
            return
        q, _, _ = self._arm_sample()
        self.tracker.reset_to(q)
        self.track.reset_to(q)
        self._reseed_flag = False
        self._reconnect_fails = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="collect-streamer")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Join LONGER than the RTDE control-script start timeout (~5 s): a
            # thread blocked inside reconnect_control must finish and exit before
            # teardown disconnects the arm / a new session opens a second control
            # interface. A thread still alive here is a bug we must see, not
            # silently orphan (an orphan + a new session = two controllers ->
            # rtde_control segfault).
            self._thread.join(8.0)
            if self._thread.is_alive():
                log.error("collect-streamer did not exit within 8 s — possible "
                          "stuck RTDE reconnect; NOT starting another controller")
            else:
                self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- phase transitions (compare-and-set: HOLD always wins) ----------
    def _transition(self, expect: TrackPhase, to: TrackPhase) -> bool:
        """Move expect -> to under the lock; refuse if a concurrent hold()
        (or anything else) changed the phase since it was read."""
        with self._lock:
            if self._phase is expect:
                self._phase = to
                return True
            return False

    # -- the loop --------------------------------------------------------
    def _run(self) -> None:
        try:
            enable_fine_timer()
            self._run_inner()
        except Exception as e:   # surface loudly — a dead streamer must not look
            log.exception("collect streamer died")   # like "the arm stopped following"
            self.error = str(e)

    @staticmethod
    def _is_ctl_fault(e: Exception) -> bool:
        """Is this the UR control script dying (protective stop / pendant
        popup) rather than a genuinely dead streamer?

        A protective stop kills the control script, so a servo_j already in
        flight this cycle is rejected. That is RECOVERABLE - reconnect once the
        pendant is cleared - and must never propagate, because the session loop
        treats a streamer exception as fatal and ends the whole session."""
        m = str(e).lower()
        return any(k in m for k in ("servoj rejected", "control script",
                                    "protective stop",
                                    "control interface not connected",
                                    "rtde control stream lost"))

    def _servo(self, q: np.ndarray, period: float) -> bool:
        """Send servo_j UNLESS a concurrent hold() latched since this cycle's
        decision. Returns False when the send was suppressed."""
        servoj = self.hw.arm.servoj
        with self._lock:
            if self._phase is TrackPhase.HOLD:
                return False
            self._last_cmd = q.copy()
        try:
            with self.ctl_lock:
                self.arm.servo_j(q, period, servoj.lookahead_time_s, servoj.gain)
        except Exception as e:
            if not self._is_ctl_fault(e):
                raise
            # the pstop landed mid-cycle. Flag it ONCE and let the loop
            # park in FAULT; recovery is operator-driven (panel Re-engage).
            if not self._ctl_lost:
                log.warning("control fault during servo (%s) - entering FAULT; "
                            "clear the pendant, then press Re-engage arm", e)
            self._ctl_lost = True
            return False
        return True

    def _reconnect_after_pstop(self) -> None:
        """The UR control script dies with a protective stop; the first
        servo_j after the pendant clear would raise and kill the thread."""
        reconnect = getattr(self.arm, "reconnect_control", None)
        if reconnect is not None:
            try:
                with self.ctl_lock:
                    reconnect()
                log.info("control interface reconnected after protective stop")
            except Exception:
                log.exception("reconnect_control failed — retrying next cycle")
                raise

    def _run_inner(self) -> None:
        period = 1.0 / self.rate_hz
        servoj = self.hw.arm.servoj
        eps = self.tuning.engage_eps_rad
        ws = self.hw.safety.workspace_m
        was_pstop = False
        while not self._stop.is_set():
            t0 = time.perf_counter()
            q_meas, tcp, pstop = self._arm_sample()
            if pstop or self._ctl_lost:
                # Control is gone. Park in FAULT and do NOTHING until the
                # operator presses Re-engage: auto-reconnecting here span a
                # 125 Hz reconnect/reject loop, and the arm has usually been
                # jogged by hand on the pendant, so a silent re-engage would
                # snap it back to the pre-fault pose.
                if pstop:
                    self._ctl_lost = True
                with self._lock:
                    if self._phase is not TrackPhase.FAULT:
                        self._phase = TrackPhase.FAULT
                        self._hold_q = None
                        log.warning("ARM FAULT (protective stop / control script "
                                    "lost) - teleop parked; clear the pendant "
                                    "and press Re-engage arm on the panel")
                self._stop.wait(period)
                continue

            with self._lock:
                phase = self._phase
                hold_q = self._hold_q
                reseed = getattr(self, "_reseed_flag", False)
                self._reseed_flag = False

            if phase is TrackPhase.HOLD:
                # keep servoing the frozen pose (a stationary stream holds
                # position firmly; dropping the stream would leave servo mode)
                if hold_q is not None:
                    try:
                        with self.ctl_lock:
                            self.arm.servo_j(hold_q, period,
                                             servoj.lookahead_time_s, servoj.gain)
                    except Exception as e:
                        if not self._is_ctl_fault(e):
                            raise
                        self._ctl_lost = True
                self._stop.wait(period)
                continue

            if reseed:   # first cycle after resume(): engage from measured pose
                self.tracker.reset_to(q_meas)

            target = self._leader_target(t0, period)
            if target is None:       # leader not streaming yet — do nothing
                self._stop.wait(period)
                continue
            if self._stale_reengage and phase is TrackPhase.TRACK:
                # a long drop-out (> reengage window) just ended: don't let the
                # accel clamp chase a large accumulated delta — glide back in
                self._transition(TrackPhase.TRACK, TrackPhase.ENGAGE)
                self.tracker.reset_to(self._last_cmd if self._last_cmd is not None
                                      else q_meas)
                phase = self._phase
                self._stale_reengage = False

            # -- workspace guard: remember the last SAFELY-inside command and
            #    retreat to it on a boundary crossing --------------------------
            inside = ws.contains(tcp[:3], margin=0.0)
            safely_inside = ws.contains(tcp[:3], margin=-0.02)   # 2 cm inside
            if self._ws_hold:
                if inside:
                    self._ws_ok_cycles += 1
                    if self._ws_ok_cycles >= 5:      # debounced re-entry
                        self._ws_hold = False
                        log.info("TCP back inside the workspace box — resuming")
                else:
                    self._ws_ok_cycles = 0
                if self._ws_hold and self._safe_q is not None:
                    target = self._safe_q            # retreat toward safety
            else:
                if inside:
                    if safely_inside and self._last_cmd is not None:
                        self._safe_q = self._last_cmd
                else:
                    self._ws_hold = True
                    self._ws_ok_cycles = 0
                    log.warning("TCP left the workspace box — retreating to the "
                                "last safely-inside pose")
                    if self._safe_q is not None:
                        target = self._safe_q

            if phase is TrackPhase.ENGAGE:
                q_cmd = self.tracker.step(target, period)
                if float(np.max(np.abs(q_cmd - target))) < eps:
                    if self._transition(TrackPhase.ENGAGE, TrackPhase.TRACK):
                        # hand off at the converged pose (engage is slow + within
                        # eps here, so starting the track tracker at rest is a
                        # sub-degree, sub-tick transient)
                        self.track.reset_to(self.tracker.q)
                        log.info("teleop engaged — direct tracking at %.0f Hz",
                                 self.rate_hz)
                # a concurrent hold() wins: _servo() suppresses the send below
            else:
                # TRACK: sqrt-braking tracker — no overshoot/ring, small steady
                # lag, commanded accel bounded (servoJ-feasible / C153A3-safe).
                q_cmd = self.track.step(target, period)
                self.tracker.reset_to(q_cmd)   # keep the engage tracker current

            self._servo(np.asarray(q_cmd, dtype=np.float64), period)
            wait = period - (time.perf_counter() - t0)
            if wait > 0:
                self._stop.wait(wait)
