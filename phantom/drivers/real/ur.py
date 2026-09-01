"""Real UR arm driver over `ur_rtde`.

- RTDEReceiveInterface is ALWAYS opened (recording, safety, wrist F/T,
  master-clock timestamps).
- RTDEControlInterface is opened only when connect(control=True): it claims
  exclusive control-script ownership and must not exist in record-only
  sessions (it would fight teleop-through-pendant or other controllers).
- Protective stops kill the servo session; reconnect_control() restores it
  after manual unlock on the pendant.

`ur_rtde` is imported lazily. CB3 vs e-Series rate constraints are enforced by
the hardware config validators before this driver is ever constructed.
"""

from __future__ import annotations

import logging
import threading
import sys
import time

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.drivers.base import Arm, ArmState

log = logging.getLogger(__name__)


class URArm(Arm):
    # A UR robot runs exactly ONE control script; two RTDEControlInterface
    # objects pointed at it (an orphaned old session/thread overlapping a new
    # one) fight for that script and segfault rtde_control.so. This process-wide
    # count enforces the single-controller invariant across URArm instances:
    # opening a second interface while one is live is REFUSED (a loud error the
    # streamer surfaces) instead of crashing the server.
    _live_ctrl_lock = threading.Lock()
    _live_ctrl_count = 0
    # Settle time between releasing the old control script and starting a new
    # one on reconnect — the robot needs a moment to free the script, else the
    # new RTDEControlInterface hits "Failed to start control script".
    _reconnect_settle_s = 0.2

    # IK branch guard (rig 2026-09-01): host-side getInverseKinematics with
    # no qnear seed returned a DIFFERENT solution branch at the elbow-straight
    # boundary (wrist centre 470.5 mm) and servoJ swept a ballistic joint-space
    # arc to it (wrists 3.4-6.9 rad/s, four episodes). Seed every solve with
    # the previous solution and reject any solution that jumps a branch.
    IK_BRANCH_TOL_RAD = 0.35          # >> one 8 ms tick of motion, << any flip
    IK_REJECT_LIMIT = 25              # consecutive rejects (~0.2 s) -> give up

    def __init__(self, hw: HardwareConfig):
        super().__init__(hw)
        self._servo_active = False
        self._recv = None
        self._ctrl = None
        self._seq = 0
        self._want_control = False
        self._last_qsol: list[float] | None = None
        self._ik_rejects = 0
        self._ik_dev_max = 0.0            # max per-tick |q_ik - q_seed| seen
        self._ik_rejects_total = 0
        # RTDEControlInterface is NOT thread-safe and its C++ object is freed on
        # disconnect(). The teleop streamer calls servo_j from its own thread
        # while the record loop (zero_ft) and teardown (disconnect /
        # reconnect_control) touch the same interface — a servoJ racing a
        # disconnect is a use-after-free that segfaults inside rtde_control.so.
        # Every control-interface call is serialized through this lock (the
        # RECEIVE interface is separate and read-only, so it is not covered).
        self._ctrl_lock = threading.RLock()

    # ------------------------------------------------------------------
    def connect(self, *, control: bool = False) -> None:
        try:
            import rtde_receive
        except ImportError as e:
            raise RuntimeError("ur_rtde not installed — pip install ur-rtde, or set "
                               "mode.overrides.arm: mock") from e
        self._recv = rtde_receive.RTDEReceiveInterface(
            self.hw.arm.ip, frequency=self.hw.arm.rtde_receive_hz)
        self._want_control = control
        if control:
            self._connect_control()

    def _teardown_ctrl(self) -> None:
        """Release self._ctrl (caller MUST hold _ctrl_lock). Drops the reference
        FIRST so no concurrent locked call invokes a method on the object being
        destroyed (use-after-free), fully stops the control script, and frees the
        process-wide controller slot."""
        old, self._ctrl = self._ctrl, None
        # A servo session belongs to ONE control script. Tearing that script
        # down (or rebuilding it in reconnect_control) ends the session by
        # definition, so the guard flag must clear here — leaving it True
        # made move_l refuse start-pose homing for the rest of the session
        # even though the fresh script has no servo stream at all.
        self._servo_active = False
        if old is None:
            return
        for op in ("servoStop", "stopScript", "disconnect"):
            try:
                getattr(old, op)()
            except Exception:
                pass
        del old
        with URArm._live_ctrl_lock:
            URArm._live_ctrl_count = max(0, URArm._live_ctrl_count - 1)

    def _preflight_stop_old_script(self) -> None:
        """Force the robot's old control script dead BEFORE constructing a new
        RTDEControlInterface (dashboard 'stop' + wait for runtime_state
        STOPPED). Best-effort: any failure just falls through to the existing
        retry loop."""
        try:
            import dashboard_client
            db = dashboard_client.DashboardClient(self.hw.arm.ip)
            db.connect()
            try:
                db.stop()
            finally:
                db.disconnect()
        except Exception as e:
            log.debug("dashboard preflight stop unavailable: %s", e)
        r = self._recv
        if r is None:
            return
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            try:
                if int(r.getRuntimeState()) != URArm._RT_PLAYING:
                    break
            except Exception:
                break
            time.sleep(0.1)
        time.sleep(URArm._reconnect_settle_s)

    def _probe_construct(self) -> None:
        """Construct+release a control interface in a THROWAWAY subprocess.
        If ur_rtde's constructor segfaults (dying-script race), only the probe
        dies; we settle and let the in-process retry loop proceed. rc==0 in
        the common case costs ~1-2 s per session start."""
        import subprocess
        code = ("import sys, rtde_control\n"
                "c = rtde_control.RTDEControlInterface(sys.argv[1], frequency=float(sys.argv[2]))\n"
                "c.disconnect()\n")
        try:
            rc = subprocess.run(
                [sys.executable, "-c", code, str(self.hw.arm.ip),
                 str(self.hw.arm.rtde_control_hz)],
                timeout=20, capture_output=True).returncode
        except Exception as e:
            log.warning("probe construct errored (%s) - continuing", e)
            return
        if rc != 0:
            log.warning("probe construct died rc=%s (segfault=-11 means the old "
                        "script was still dying) - settling before real attempt", rc)
            time.sleep(2.0)
            self._preflight_stop_old_script()

    def _connect_control(self) -> None:
        """Build a fresh control interface. Reserves the single process-wide
        controller slot FIRST (refusing if one is already live — a second
        controller on one robot segfaults), builds into a local, and publishes
        to self._ctrl only once fully configured; any failure releases the slot
        and any half-built C++ interface."""
        import rtde_control
        with URArm._live_ctrl_lock:
            if URArm._live_ctrl_count > 0:
                raise RuntimeError(
                    "refusing to open a SECOND RTDE control interface in this "
                    "process — a previous controller is still live (it would "
                    "fight for the robot's control script and segfault). Fully "
                    "stop the old session/streamer first.")
            URArm._live_ctrl_count += 1        # reserve the slot
        ctrl = None
        try:
            # ---- preflight (added 2026-08-01, segfix): the "1st session start
            # after a fault fails, 2nd works" pattern was not a polite failure -
            # kernel log shows the FIRST process SEGFAULTING inside
            # rtde_control.so's asio thread while constructing against a robot
            # whose previous control script is still dying (3x on 2026-08-01,
            # identical offset). Deterministically kill the old script first
            # (dashboard stop), wait until the runtime state is actually
            # STOPPED, and probe-construct once in a sacrificial subprocess so
            # a residual constructor crash can never take the recorder down.
            self._preflight_stop_old_script()
            self._probe_construct()
            # "failed to start in 5s" right after a fault means the old
            # control script has not finished dying; a retry succeeds (observed
            # on this rig: 1st session start after a fault fails, 2nd works).
            last = None
            for attempt in range(3):
                try:
                    ctrl = rtde_control.RTDEControlInterface(
                        self.hw.arm.ip, frequency=self.hw.arm.rtde_control_hz)
                    break
                except Exception as e:
                    last, ctrl = e, None
                    log.warning("RTDE control start failed (%d/3): %s",
                                attempt + 1, e)
                    time.sleep(2.0)
            if ctrl is None:
                raise RuntimeError(f"RTDE control would not start: {last}")
            ctrl.setTcp(list(self.hw.arm.tcp_offset_m))
            ctrl.setPayload(self.hw.arm.payload_kg, [0.0, 0.0, 0.05])
        except Exception:
            with URArm._live_ctrl_lock:
                URArm._live_ctrl_count = max(0, URArm._live_ctrl_count - 1)
            if ctrl is not None:
                try:
                    ctrl.disconnect()
                except Exception:
                    pass
            raise
        self._ctrl = ctrl

    def reconnect_control(self) -> None:
        """Rebuild the control interface after a protective stop was cleared.

        Single-flight and reference-safe (all under _ctrl_lock, so it can never
        race a servo_j / disconnect). If another caller already reconnected while
        this one waited, it returns without stacking a second control script. The
        old interface is fully torn down and the robot given a moment to release
        the control script BEFORE the new one starts."""
        with self._ctrl_lock:
            # DO NOT probe an interface whose control script may be dead.
            # isProgramRunning()/reuploadScript() on such an interface
            # SEGFAULTED the whole process (ec=139, observed 2026-07-27 the
            # moment the operator pressed Re-engage) - rtde_control.so is not
            # safe to poke once its script has gone.
            #
            # Equally, do NOT early-return on isConnected(): that is only the
            # SOCKET, which stays up while the script is dead, and returning
            # "already reconnected" there is what made every following servo_j
            # get rejected in a 125 Hz reconnect/reject loop.
            #
            # Blind teardown -> settle -> fresh build is the only safe path.
            # _connect_control() is itself the verification: ur_rtde raises
            # "Failed to start control script" if the script does not come up.
            self._teardown_ctrl()
            time.sleep(URArm._reconnect_settle_s)   # let the robot free the script
            self._connect_control()
            log.info("RTDE control interface rebuilt (control script started)")

    # UR RTDE runtime_state values (UR RTDE guide): 0 STOPPING, 1 STOPPED,
    # 2 PLAYING, 3 PAUSING, 4 PAUSED, 5 RESUMING
    _RT_STOPPED = 1
    _RT_PLAYING = 2

    def program_running(self) -> bool:
        """Is the control SCRIPT actually running (not just the socket)?

        Answered from the RECEIVE interface (read-only, thread-safe) — NEVER
        by probing the control object: isProgramRunning() on an interface
        whose script died SEGFAULTS the process (observed 2026-07-27; the
        reconnect_control docstring documents it, this method used to do that
        exact probe — fixed 2026-08-01)."""
        r = self._recv
        if r is None or self._ctrl is None:
            return False
        try:
            return int(r.getRuntimeState()) == URArm._RT_PLAYING
        except Exception:
            return False

    def is_ready_for_control(self) -> tuple[bool, str]:
        """(ok, why_not) - may the control script be started right now?"""
        r = self._recv
        if r is None:
            return False, "RTDE receive not connected"
        try:
            if r.isProtectiveStopped():
                return False, ("robot is STILL protective-stopped - clear it on "
                               "the pendant first")
            mode, safety = int(r.getRobotMode()), int(r.getSafetyMode())
        except Exception as e:
            return False, f"cannot read robot state: {e}"
        if mode != 7:
            return False, (f"robot mode {mode} (need 7=RUNNING: power on and "
                           "release the brakes on the pendant)")
        if safety != 1:
            return False, f"safety mode {safety} (need 1=NORMAL)"
        return True, ''

    def disconnect(self) -> None:
        # serialize against a concurrent servo_j (streamer thread) so we never
        # free the control interface out from under an in-flight servoJ
        with self._ctrl_lock:
            self._teardown_ctrl()
        if self._recv is not None:
            self._recv.disconnect()
            self._recv = None

    def _require_ctrl(self):
        if self._ctrl is None:
            raise RuntimeError("control interface not connected (connect(control=True))")
        if not self._ctrl.isConnected():
            raise RuntimeError("RTDE control stream lost (robot rebooted / protective "
                               "stop / network drop) — reconnect_control() after fixing")
        return self._ctrl

    # ------------------------------------------------------------------
    def get_state(self) -> ArmState:
        r = self._recv
        if r is None:
            raise RuntimeError("get_state() before connect()")
        if not r.isConnected():
            # fail LOUD instead of blocking the whole teleop loop forever —
            # observed on the CB3 after a mid-motion RTDE stream drop
            raise RuntimeError("RTDE receive stream lost — robot rebooted or "
                               "network dropped; restart the session")
        st = ArmState(
            t_host=time.perf_counter(), seq=self._seq,
            t_rtde=float(r.getTimestamp()),
            q=np.asarray(r.getActualQ(), dtype=np.float64),
            qd=np.asarray(r.getActualQd(), dtype=np.float64),
            tcp_pose=np.asarray(r.getActualTCPPose(), dtype=np.float64),
            tcp_speed=np.asarray(r.getActualTCPSpeed(), dtype=np.float64),
            ft=np.asarray(r.getActualTCPForce(), dtype=np.float64),
            protective_stop=bool(r.isProtectiveStopped()),
            robot_mode=int(r.getRobotMode()),
        )
        self._seq += 1
        return st

    def servo_j(self, q: np.ndarray, dt: float, lookahead: float, gain: int) -> None:
        with self._ctrl_lock:
            self._servo_active = True
            ok = self._require_ctrl().servoJ(list(np.asarray(q, dtype=float)), 0.0,
                                             0.0, dt, lookahead, gain)
        if ok is False:
            # ur_rtde returns False SILENTLY when the control script is no
            # longer running on the robot (a protective stop kills it) — the
            # arm just stops following while everything else keeps working
            raise RuntimeError("servoJ rejected — the RTDE control script is not "
                               "running (clear the pendant popup / protective stop "
                               "and restart the session)")

    def servo_l(self, tcp_pose: np.ndarray, dt: float, lookahead: float, gain: int) -> None:
        with self._ctrl_lock:
            self._servo_active = True
            ctrl = self._require_ctrl()
            qref = self._last_qsol
            if qref is None and self._recv is not None:
                qref = list(self._recv.getActualQ())
            if qref is not None:
                q = ctrl.getInverseKinematics(
                    list(np.asarray(tcp_pose, dtype=float)), list(qref))
            else:
                q = ctrl.getInverseKinematics(
                    list(np.asarray(tcp_pose, dtype=float)))
            dev = (max(abs(a - b) for a, b in zip(q, qref))
                   if (qref is not None and q) else 0.0)
            if dev > self._ik_dev_max:
                self._ik_dev_max = dev
            if qref is not None and q and dev > self.IK_BRANCH_TOL_RAD:
                # branch flip / degenerate solve: DO NOT stream it. Holding
                # the previous setpoint for a tick is safe; a sustained
                # inability to solve on-branch ends the episode instead of
                # whipping the arm.
                self._ik_rejects += 1
                self._ik_rejects_total += 1
                if self._ik_rejects == 1:
                    log.warning("IK solution jumped %.2f rad off the current "
                                "branch — holding pose (kinematic boundary?)",
                                dev)
                if self._ik_rejects >= self.IK_REJECT_LIMIT:
                    raise RuntimeError(
                        "IK cannot stay on the current joint branch "
                        f"({self._ik_rejects} consecutive solves rejected) — "
                        "the target is at/beyond a kinematic boundary")
                return
            self._ik_rejects = 0
            if q:
                self._last_qsol = list(q)
            ok = ctrl.servoJ(q, 0.0, 0.0, dt, lookahead, gain)
        if ok is False:
            raise RuntimeError("servoJ rejected — the RTDE control script is not "
                               "running (clear the pendant popup / protective stop "
                               "and restart the session)")

    def speed_l(self, xd: np.ndarray, accel: float, dt: float) -> None:
        with self._ctrl_lock:
            self._require_ctrl().speedL(list(np.asarray(xd, dtype=float)), accel, dt)

    def move_j(self, q: np.ndarray, speed: float, accel: float, blocking: bool = True) -> None:
        with self._ctrl_lock:
            self._require_ctrl().moveJ(list(np.asarray(q, dtype=float)), speed, accel,
                                       not blocking)

    def move_l(self, tcp_pose: np.ndarray, speed: float, accel: float,
               blocking: bool = True) -> None:
        with self._ctrl_lock:
            if self._servo_active:
                # same guard MockArm enforces: a moveL while a servo stream is
                # (possibly) live is an illegal mode mix on the controller —
                # the homing caller must run before executor.start() or after
                # servo_stop()
                raise RuntimeError("move_l while servo mode may be active; "
                                   "call servo_stop() first")
            ok = self._require_ctrl().moveL(list(np.asarray(tcp_pose, dtype=float)),
                                            speed, accel, not blocking)
        if ok is False:
            raise RuntimeError("moveL rejected - control script not running "
                               "(pendant popup / protective stop / Local mode)")

    def stop(self, decel: float) -> None:
        # stopJ, not stopL (run analysis 09-01): a safety stop is most often
        # fired DURING joint-space trouble (a whip, a singular configuration),
        # exactly where the tool-space controller behind stopL is ill-posed.
        # Joint-space deceleration is well-defined in every configuration.
        try:
            with self._ctrl_lock:
                self._require_ctrl().stopJ(decel)
        except Exception:
            log.exception("stopJ failed")

    def zero_ft(self) -> None:
        with self._ctrl_lock:
            self._require_ctrl().zeroFtSensor()

    def is_protective_stopped(self) -> bool:
        if self._recv is None:
            raise RuntimeError("is_protective_stopped() before connect()")
        return bool(self._recv.isProtectiveStopped())

    def servo_stop(self) -> None:
        """End the servo stream.

        Also drops the IK branch seed: after servo mode ends the arm may be
        moved by moveJ/moveL/hand, so the next servo session must re-seed
        from the MEASURED q, not a stale solution.

        INVARIANT for `_servo_active`: it is True only while a servo session
        may still be live on the control script that is running RIGHT NOW. It
        must never outlive that script — a sticky True is what refuses every
        subsequent move_l (start-pose homing), which is the OOD-start failure
        mode start_pose.py exists to prevent.

        So the flag is cleared whenever the servo session is provably over:
        after a successful servoStop, in _teardown_ctrl (script replaced), and
        when servoStop could not be delivered AND the receive interface
        confirms the control script is not running (a dead script cannot be
        servoing). The one case that keeps the guard up is a servoStop failure
        against a script that IS still playing: there the stream may genuinely
        still be live, so the caller gets the exception and move_l stays
        refused."""
        # IK-guard telemetry (run analysis 09-01 note 1: without the IK
        # OUTPUT distribution the 0.35 rad threshold cannot be judged —
        # measured q is the servo's smoothed response and hides a flip)
        if self._ik_dev_max > 0.0 or self._ik_rejects_total:
            log.info("IK guard: max per-tick deviation %.4f rad, %d rejects "
                     "this servo session", self._ik_dev_max,
                     self._ik_rejects_total)
        self._last_qsol = None
        self._ik_rejects = 0
        self._ik_dev_max = 0.0
        self._ik_rejects_total = 0
        with self._ctrl_lock:
            try:
                self._require_ctrl().servoStop()
            except Exception:
                # program_running() reads the RECEIVE interface only — safe to
                # call on a control interface whose script has died (probing
                # the control object there segfaults; see reconnect_control).
                if self.program_running():
                    raise
                self._servo_active = False
                log.warning("servoStop failed and the control script is not "
                            "running — the servo session is already over; "
                            "cleared servo guard so homing can move_l")
                return
            self._servo_active = False
