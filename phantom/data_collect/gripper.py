"""GripperPilot — continuous proportional gripper teleop on its own thread.

Why a dedicated thread: RobotiqGripper.move() blocks on the URCap socket
round-trip ('ack'), and in the reference stack that round-trip sat inside the
record loop, competing for the socket lock with a 100 Hz status poller. Here
the pilot PULLS the freshest leader squeeze value directly from the Echo
reader cache at its own rate (<=rate_hz), sends only when the change exceeds
the deadband, and nothing upstream ever blocks on the socket. Pulling at
device rate (not via the 10 Hz session loop) keeps squeeze-to-motion latency
at ~10-30 ms — main11-class feel, but proportional instead of binary.

Continuity: the leader's raw squeeze ticks are already continuous — the lab
stack merely binarized them. The Echo layer maps ticks -> 0..1 via the
open/closed tick calibration (data_collect.yaml gripper.open_tick/closed_tick;
the panel's Calibrate button re-measures them live). The pilot forwards that
0..1 position with a SNAPPY speed (default 1.0 — main11 used 255/255) and
the hardware-config force, which the driver clamps to the DM-Tac pad ceiling
(cmd_force_limit_N=30 N) and close ceiling (max_close_cmd) — pad safety is
enforced at the driver boundary no matter what this class sends.

Safeguard hook: open_now() jumps the queue — full open, sent immediately,
and the pilot ignores the leader until release() (the panel Resume flow).
On release() the pilot follows the CURRENT leader value (freshly pulled),
never a stale pre-trip squeeze, and the first command is deadband-exempt.

Force-limited grasp (optional `tactile_force` reader): each tick the pilot
reads the max resultant tactile force across both DM-Tac pads; once it reaches
grasp_stop_n it stops closing FURTHER (holds the current position — "grabbed")
while the leader keeps pressing, so the operator can squeeze the Echo leader
fully without overgripping. Opening is always obeyed immediately and clears the
ceiling; a grabbed hold only re-closes once force relaxes below
grasp_resume_frac*grasp_stop_n (anti-chatter hysteresis). If tactile data goes
stale (freshest sample older than grasp_stale_s — worker lag/death) the ceiling
is dropped and the pilot degrades to plain proportional control; it NEVER blocks
the gripper on dead data. Without a reader (unit tests / mock) the feature is
inert. This working limit sits below the pad safeguard (15 N) and the driver
pad ceiling (30 N).

The close is plain proportional at `speed` — overshoot past grasp_stop_n (the
gripper closes faster than the ~10 Hz tactile updates) is accepted as natural
grasp dynamics. A freeze logs at INFO for post-hoc debugging; a one-shot INFO
notes if the limit goes transparent (no sample / stale) in the field.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from phantom.config.hardware import HardwareConfig

from phantom.data_collect.config import GripperTuning
from phantom.data_collect.teleop import enable_fine_timer

log = logging.getLogger(__name__)


class GripperPilot:
    def __init__(self, hw: HardwareConfig, gripper, tuning: GripperTuning,
                 *, leader=None, tactile_force=None, ring=None,
                 feedback_rate_hz: float | None = None):
        """leader: object with latest_gripper01() -> float | None (CollectEcho);
        without one (unit tests) targets come from set_target().
        tactile_force: callable() -> (force_n, age_s) | None enabling the
        force-limited grasp; None disables it (plain proportional control).
        ring / feedback_rate_hz: recording sink for gripper.get_state() — see
        _run_inner docstring for why this lives HERE instead of a separate
        ThreadPoller (phantom/recording/workers.py used to run one at
        hw.gripper.feedback_rate_hz; measured 2026-07-31 to periodically stall
        20-40 ms, tracing to two threads racing the SAME URCap socket lock).
        feedback_rate_hz defaults to tuning.rate_hz (no separate recording
        rate) when not given, matching the pre-existing unit-test behavior."""
        self.hw = hw
        self.gripper = gripper
        self.tuning = tuning
        self.leader = leader
        self.tactile_force = tactile_force
        self._ring = ring
        self._feedback_rate_hz = feedback_rate_hz
        self._target: float | None = None      # mailbox (leaderless mode)
        self._last_sent: float | None = None
        self._suspended = False                # safeguard: ignore the leader
        self._open_pending = False
        # force-limited grasp state (pilot thread only, reset under _lock):
        # _grasp_frozen -> currently holding at _freeze_pos because force hit
        # the ceiling while the leader was still closing.
        self._grasp_frozen = False
        self._freeze_pos = 0.0
        self._transparent_logged = False   # one-shot log when the limit goes inert
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    # -- inputs (any thread) --------------------------------------------
    def set_target(self, position01: float) -> None:
        with self._lock:
            self._target = float(min(max(position01, 0.0), 1.0))

    def open_now(self) -> None:
        """Safeguard trip: command full open immediately, then hold open
        (leader ignored) until release()."""
        with self._lock:
            self._suspended = True
            self._open_pending = True
            self._grasp_frozen = False        # opening clears any grasp hold

    def release(self) -> None:
        """Panel Resume: follow the leader again — the FRESH leader value,
        never the pre-trip squeeze. First command is deadband-exempt."""
        with self._lock:
            self._suspended = False
            self._open_pending = False
            self._target = None        # drop any stale mailbox value
            self._last_sent = None
            self._grasp_frozen = False

    @property
    def suspended(self) -> bool:
        return self._suspended

    @property
    def last_sent(self) -> float | None:
        """Last position command actually sent (0..1) — the absolute gripper
        action the record loop logs."""
        with self._lock:
            return self._last_sent

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="gripper-pilot")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- the loop --------------------------------------------------------
    def _run(self) -> None:
        try:
            enable_fine_timer()
            self._run_inner()
        except Exception as e:
            log.exception("gripper pilot died")
            self.error = str(e)

    def _fresh_target(self) -> float | None:
        """Freshest leader value when a leader is attached, else the mailbox."""
        if self.leader is not None:
            v = self.leader.latest_gripper01()
            if v is not None:
                return float(min(max(v, 0.0), 1.0))
            return None
        with self._lock:
            return self._target

    def _note_transparent(self, reason: str) -> None:
        """Log ONCE (until fresh tactile returns) that the force limit is inert,
        so the operator can tell 'no ceiling' from 'ceiling active' in the field.
        """
        if not self._transparent_logged:
            log.info("grasp force-limit INACTIVE (transparent): %s", reason)
            self._transparent_logged = True

    def _apply_grasp_ceiling(self, target: float, last: float | None) -> float:
        """Force-limited grasp (simple position ceiling). Returns the position to
        command: the leader `target`, unless tactile force has reached
        grasp_stop_n while closing further — then the current position is held
        ("grabbed"). Opening is always obeyed; stale/missing tactile degrades to
        transparent (returns `target`) and NEVER blocks the gripper. The close is
        plain proportional at `speed`; overshoot past grasp_stop_n is accepted as
        natural grasp dynamics. Runs on the pilot thread; position 0=open,
        1=closed."""
        reader = self.tactile_force
        if reader is None:
            return target
        try:
            reading = reader()
        except Exception:
            # a reader failure must NEVER block the gripper — go transparent
            log.debug("tactile force reader failed", exc_info=True)
            reading = None
        if reading is None:
            self._note_transparent("no tactile sample yet")
            self._grasp_frozen = False
            return target
        force_n, age_s = reading
        if age_s > self.tuning.grasp_stale_s:
            self._note_transparent(f"tactile stale ({age_s * 1e3:.0f} ms old)")
            self._grasp_frozen = False
            return target
        self._transparent_logged = False        # fresh data -> re-arm the log
        stop_n = self.tuning.grasp_stop_n
        resume_n = self.tuning.grasp_resume_frac * stop_n
        if self._grasp_frozen:
            if target < self._freeze_pos:
                # leader opening past the grabbed point -> obey, clear ceiling
                self._grasp_frozen = False
                return target
            if force_n < resume_n and target > self._freeze_pos:
                # force relaxed AND leader still pressing -> allow re-close (дожим)
                self._grasp_frozen = False
                return target
            return self._freeze_pos            # hold the grabbed position
        # not yet grabbed: freeze only if the leader is closing FURTHER
        if last is not None and target > last and force_n >= stop_n:
            self._grasp_frozen = True
            self._freeze_pos = last
            log.info("grasp stop at %.1f N, pos %.3f", force_n, last)
            return last
        return target

    def _run_inner(self) -> None:
        """Single thread, single socket: this used to be TWO threads (this
        pilot sending move() commands, plus a separate ThreadPoller calling
        get_state() for recording) racing RobotiqGripper's one TCP connection
        + lock. Under load that produced periodic 20-40 ms stalls on the
        recording side (measured 2026-07-31, ~16 missed samples/episode) —
        each thread blocking on the other's in-flight socket round-trip.
        Folding the state read into this loop removes the lock entirely: one
        thread, one owner of the socket, no contention possible.

        cmd_period (tuning.rate_hz) is the ORIGINAL move()-dispatch cadence —
        unchanged, still the URCap socket command cap. state_period is the
        (>=) faster tick the loop actually runs at so the recorded `gripper`
        stream keeps hw.gripper.feedback_rate_hz resolution; open_now() is
        still dispatched on the very next tick regardless of cmd_period (never
        throttled — it is the safeguard's emergency-open path)."""
        cmd_period = 1.0 / self.tuning.rate_hz
        state_hz = max(self._feedback_rate_hz or self.tuning.rate_hz, self.tuning.rate_hz)
        state_period = 1.0 / state_hz
        speed = self.tuning.speed
        force = self.hw.gripper.default_force   # driver clamps to the pad ceiling
        next_cmd = 0.0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            if self._ring is not None:
                try:
                    st = self.gripper.get_state()
                    self._ring.push(st.t_host, state=np.array(
                        [st.position, st.obj], dtype=np.float32))
                except Exception:
                    log.exception("gripper state read failed")
            with self._lock:
                open_pending = self._open_pending
                self._open_pending = False
                suspended = self._suspended
            if open_pending:
                # full open at max speed — pad protection, no deadband, no
                # cmd_period throttle (safety-critical: fire on the next tick)
                self.gripper.move(0.0, 1.0, force)
                with self._lock:
                    self._last_sent = 0.0
                log.warning("gripper OPENED by safeguard")
                next_cmd = t0 + cmd_period
            elif not suspended and t0 >= next_cmd:
                target = self._fresh_target()
                if target is not None:
                    with self._lock:
                        last = self._last_sent
                    cmd_pos = self._apply_grasp_ceiling(target, last)
                    if last is None or abs(cmd_pos - last) > self.tuning.deadband:
                        self.gripper.move(cmd_pos, speed, force)
                        with self._lock:
                            self._last_sent = cmd_pos
                next_cmd = t0 + cmd_period
            wait = state_period - (time.perf_counter() - t0)
            if wait > 0:
                self._stop.wait(wait)
