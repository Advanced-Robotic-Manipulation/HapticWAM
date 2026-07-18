"""JointServoStreamer: high-rate smooth joint servoing for teleop leaders.

Decouples CONTROL from the action grid (same pattern as deploy's
ChunkExecutor): the 10 Hz record loop only updates the target; this thread
streams servo_j at teleop.echo.control_rate_hz (default 125 Hz, CB3-safe)
with a matched dt, tracking the target through an AccelLimitedTracker
(bounded velocity from arm.limits.joint_speed_rad_s, bounded acceleration
from config). Result: no 10 Hz step-and-hold, no engage jump, no bang-bang
clamping — every commanded step is an S-curve.

Safety (mirrors the old in-loop guards, now at control rate):
 - reactive workspace hold — if the measured TCP leaves safety.workspace_m
   the target freezes at the tracker's last in-bounds pose (joint targets
   cannot be clipped in EE space without IK); it resumes when the operator
   steers back;
 - protective stop -> streamer parks (keeps thread alive, stops commanding).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.drivers.base import Arm
from phantom.teleop.filters import AccelLimitedTracker

log = logging.getLogger(__name__)


class JointServoStreamer:
    def __init__(self, hw: HardwareConfig, arm: Arm):
        assert hw.teleop is not None and hw.teleop.echo is not None
        cfg = hw.teleop.echo
        self.hw = hw
        self.arm = arm
        self.rate_hz = cfg.control_rate_hz
        self.tracker = AccelLimitedTracker(
            v_max=hw.arm.limits.joint_speed_rad_s,
            a_max=cfg.joint_accel_rad_s2)
        self._target: np.ndarray | None = None
        self._hold_target: np.ndarray | None = None
        self._in_hold = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None   # set when the thread dies — the record
                                        # loop checks it and fails the session LOUDLY

    # ------------------------------------------------------------------
    def set_target(self, q: np.ndarray) -> None:
        """Called from the record loop (any rate); the streamer glides there."""
        with self._lock:
            self._target = np.asarray(q, dtype=np.float64).copy()

    def park(self) -> None:
        """Freeze at the tracker's current pose — called by the record loop's
        safety circuit on a force/e-stop event (glides to rest under the
        accel bound; the loop resumes by set_target after recovered())."""
        with self._lock:
            self._target = self.tracker.q

    def start(self) -> None:
        """Engage at the CURRENT measured pose — no jump by construction."""
        self.tracker.reset_to(self.arm.get_state().q)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="joint-streamer")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def in_hold(self) -> bool:
        """True while frozen at the workspace boundary (surfaced in the panel)."""
        return self._in_hold

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:
            # a silently-dead streamer looks like "the arm just stopped
            # following" — record the reason so the session fails visibly
            log.exception("joint streamer died")
            self.error = str(e)

    def _run_inner(self) -> None:
        period = 1.0 / self.rate_hz
        servoj = self.hw.arm.servoj
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                target = self._target
            if target is None:
                self._stop.wait(period)
                continue
            state = self.arm.get_state()
            if state.protective_stop:
                self._stop.wait(period)   # park; operator clears on the pendant
                continue
            # reactive workspace hold at control rate. Resume with a 2 cm
            # hysteresis margin: the frozen pose sits marginally OUTSIDE the
            # boundary it stopped at — without the margin it never resumes.
            if self.hw.safety.workspace_m.contains(state.tcp_pose[:3],
                                                   margin=0.02 if self._in_hold else 0.0):
                self._hold_target = None
                if self._in_hold:
                    self._in_hold = False
                    log.info("TCP back in the workspace box — resuming teleop")
            else:
                if self._hold_target is None:
                    self._hold_target = self.tracker.q   # last in-bounds pose
                    self._in_hold = True
                    log.warning("TCP left the workspace box — holding last in-bounds pose")
                target = self._hold_target
            q_cmd = self.tracker.step(target, period)
            self.arm.servo_j(q_cmd, period, servoj.lookahead_time_s, servoj.gain)
            wait = period - (time.perf_counter() - t0)
            if wait > 0:
                self._stop.wait(wait)
