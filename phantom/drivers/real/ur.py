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
import math
import sys
import threading
import time

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.data.derived import rotvec_nearest
from phantom.drivers import servo_limiter
from phantom.drivers.base import (Arm, ArmState, ControlLost, ServoBranchFault,
                                  ServoHoldTimeout, ServoResult)
from phantom.drivers.real import rig_lease

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
    IK_REJECT_LIMIT = 25              # consecutive ik_branch rejects (~0.2 s) -> give up
    # A target with NO solution (or a limiter hold) is a HOLD, not a fault: the
    # arm keeps its last setpoint while the planner gets several replans
    # (0.8-1.1 s each) to move the target back inside reach. Rig 09-07: the
    # 25-tick counter (0.2 s, shorter than one replan) was applied to
    # ik_invalid too and ended 12 of 43 episodes as "servo cannot stream"
    # while the arm stood still at the reach boundary; before issue #7 those
    # ticks were a silent unbounded hold. Bounded now by elapsed time.
    HOLD_BUDGET_S = 4.0
    # Reach / joint-speed limiter (rig 2026-09-04, see SafetyConfig.elbow_min_rad)
    LIMITER_BISECT = 3                # step fractions down to 1/8
    LIMITER_LOG_PERIOD_S = 1.0

    def __init__(self, hw: HardwareConfig):
        super().__init__(hw)
        self._servo_active = False
        self._recv = None
        self._ctrl = None
        self._seq = 0
        self._want_control = False
        self._last_qsol: list[float] | None = None
        self._ik_rejects = 0              # consecutive rejected ticks (any reason)
        self._branch_rejects = 0          # consecutive ik_branch rejects
        self._hold_since: float | None = None   # start of the current hold streak
        self._ik_dev_max = 0.0            # max per-tick |q_ik - q_seed| seen
        self._ik_rejects_total = 0
        self._last_cmd_pose: np.ndarray | None = None   # last pose actually streamed
        self._lease = None                # rig_lease file object (control sessions)
        self.control_loss_last: dict = {}  # why the controller dropped the stream (stop.json)
        self._limiter_hits = 0
        self._limiter_holds = 0
        self._limiter_log_t = 0.0
        self.limiter_last: dict = {}
        self._constraint_hold_budget = None
        self.constraint_hold_last: dict = {}
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
        # a hold/branch streak belongs to the script being torn down: a
        # rebuilt script starts clean (a 60 s pendant pause must not become a
        # 60 s "hold" on the first rejected tick after reconnect)
        self._ik_rejects = 0
        self._branch_rejects = 0
        self._hold_since = None
        # A servo session belongs to ONE control script. Tearing that script
        # down (or rebuilding it in reconnect_control) ends the session by
        # definition, so the guard flag must clear here — leaving it True
        # made move_l refuse start-pose homing for the rest of the session
        # even though the fresh script has no servo stream at all.
        self._servo_active = False
        if self._constraint_hold_budget is not None:
            self._constraint_hold_budget.reset()
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
        # exclusive per-host lease FIRST — before the dashboard "stop old
        # script" preflight, which is itself a robot call that would kill
        # the OWNER's control script (issue #8). A Ctrl-Z'd owner keeps the
        # lease and the error names that process; nothing is sent to the
        # robot if this raises.
        if self._lease is None:
            self._lease = rig_lease.acquire(str(self.hw.arm.ip))
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

    def _control_script_dead(self) -> bool:
        """True only when the receive side POSITIVELY reports the control
        script not PLAYING. Unreadable state (no receive, a test double
        without getRuntimeState, a transport error) is NOT a verdict —
        the stale-stream guard covers that path."""
        r = self._recv
        if r is None:
            return False
        try:
            return int(r.getRuntimeState()) != URArm._RT_PLAYING
        except Exception:
            return False

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
        if self._lease is not None:
            try:
                self._lease.close()       # releases the flock (issue #8)
            except Exception:
                pass
            self._lease = None

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
            diag = self._diagnose_control_loss(list(np.asarray(q, dtype=float)), None)
            raise ControlLost("servoJ rejected — the RTDE control script is not "
                               "running (clear the pendant popup / protective stop "
                               f"and restart the session); robot: {diag.get('summary', 'n/a')}")

    def _solve_ik(self, ctrl, pose, qref):
        pose = list(np.asarray(pose, dtype=float))
        if qref is not None:
            return ctrl.getInverseKinematics(pose, list(qref))
        return ctrl.getInverseKinematics(pose)

    def _servo_limits(self) -> servo_limiter.ServoLimits:
        sf = getattr(getattr(self, "hw", None), "safety", None)
        return servo_limiter.ServoLimits(
            elbow_min_rad=getattr(sf, "elbow_min_rad", None),
            joint_speed_max_rad_s=getattr(sf, "servo_joint_speed_max_rad_s", None),
            branch_tolerance_rad=self.IK_BRANCH_TOL_RAD,
            bisection_iterations=self.LIMITER_BISECT,
            shoulder_height_m=float(getattr(sf, "ur_dh_d1_m", 0.1519)),
        )

    def _limit_violation(self, q, qref, dt: float) -> str | None:
        return servo_limiter.limit_violation(q, qref, dt, self._servo_limits())

    def _speed_violation(self, q, qref, dt: float) -> bool:
        return servo_limiter.speed_violation(q, qref, dt, self._servo_limits())

    def _feasible(self, q, qref, dt: float) -> bool:
        return servo_limiter.feasible(q, qref, dt, self._servo_limits())

    def _limited_step(self, ctrl, prev: np.ndarray, target: np.ndarray, qref, dt: float):
        return servo_limiter.limited_step(
            lambda pose, seed: self._solve_ik(ctrl, pose, seed),
            prev, target, qref, dt, self._servo_limits(),
        )

    @staticmethod
    def _ik_valid(q) -> bool:
        """6 finite joints — an empty / short / NaN solve must never reach
        servoJ (issue #7)."""
        try:
            return len(q) == 6 and all(math.isfinite(float(v)) for v in q)
        except Exception:
            return False

    def _reject(self, why: str, detail: str, *, strict_fault: bool = False) -> ServoResult:
        """Hold the previous setpoint for this tick; a sustained run of
        rejects ends the episode (executor crash net) instead of streaming
        anything doubtful."""
        self._ik_rejects += 1
        self._ik_rejects_total += 1
        now = time.monotonic()
        if self._ik_rejects == 1:
            self._hold_since = now
            log.warning("servo hold (%s): %s", why, detail)
        # Rig 09-09: 24 of 61 episodes ended as a 4 s "hold" that was really a
        # DEAD CONTROL SCRIPT (E-stop): ur_rtde's getInverseKinematics returns
        # an EMPTY solution, never an error, once the script is down, so the
        # rejects looked like an unreachable target. Ask the receive side
        # (runtime_state, read-only) on the first reject and every 25 ticks
        # and name the real cause immediately.
        if self._ik_rejects == 1 or self._ik_rejects % 25 == 0:
            if self._control_script_dead():
                diag = self._diagnose_control_loss(None, None)
                raise ControlLost(
                    "control script is not running (runtime_state != PLAYING) "
                    f"— E-stop / protective stop / pendant popup; robot: "
                    f"{diag.get('summary', 'n/a')}")
        if why == "ik_branch":
            # a solution on another branch: streaming it would whip the arm;
            # a sustained run means the seed is lost — give up fast
            self._branch_rejects += 1
            if self._branch_rejects >= self.IK_REJECT_LIMIT:
                raise ServoBranchFault(
                    f"servo cannot stream ({self._branch_rejects} consecutive "
                    f"ticks rejected, last: {why}) — the target is at/beyond a "
                    "kinematic boundary")
        if strict_fault and self._ik_rejects >= self.IK_REJECT_LIMIT:
            # The opt-in verified-hold path grants extra time only to audited
            # finite/on-branch constraints. Its mixed invalid/branch solves
            # retain the original stricter fault counter, even if verified
            # anchor holds were streamed between the rejected ticks. The
            # default path still grants no-solution holds its 4 s budget.
            raise RuntimeError(
                f"servo cannot stream ({self._ik_rejects} consecutive "
                f"ticks rejected, last: {why}) — unverified IK or constraint fault")
        # (a no-solution tick between two off-branch ticks does not forgive
        # the branch streak; only a SENT tick does)
        since = getattr(self, "_hold_since", None)
        held_s = now - (since if since is not None else now)
        if held_s >= self.HOLD_BUDGET_S:
            raise ServoHoldTimeout(
                f"servo held for {held_s:.1f} s ({self._ik_rejects} ticks, last: "
                f"{why}) — the target stayed beyond reach across several replans")
        return ServoResult(False, getattr(self, "_last_cmd_pose", None), why)

    def servo_l(self, tcp_pose: np.ndarray, dt: float, lookahead: float,
                gain: int, *, target_guard=None) -> ServoResult:
        with self._ctrl_lock:
            self._servo_active = True
            ctrl = self._require_ctrl()
            safety = getattr(getattr(self, "hw", None), "safety", None)
            if getattr(safety, "servo_constraint_hold_s", None) is not None:
                return self._servo_l_bounded_hold(ctrl, tcp_pose, dt, lookahead, gain,
                                                  target_guard=target_guard)
            if target_guard is not None:
                raise ValueError("boundary projection requires the verified bounded servo path")
            qref = self._last_qsol
            if qref is None and self._recv is not None:
                qref = list(self._recv.getActualQ())
            q = self._solve_ik(ctrl, tcp_pose, qref)
            if not self._ik_valid(q):
                # no solution (unreachable target) or garbage: with the
                # limiter on, try to shorten towards the anchor below;
                # otherwise hold.
                if not (qref is not None and self._limiter_enabled()):
                    return self._reject("ik_invalid",
                                        f"IK returned {q!r} for {np.round(tcp_pose, 4).tolist()}")
                q = []
            dev = (max(abs(a - b) for a, b in zip(q, qref))
                   if (qref is not None and q) else 0.0)
            if dev > self._ik_dev_max:
                self._ik_dev_max = dev
            if qref is not None and q and dev > self.IK_BRANCH_TOL_RAD:
                # branch flip / degenerate solve: DO NOT stream it. Holding
                # the previous setpoint for a tick is safe; a sustained
                # inability to solve on-branch ends the episode instead of
                # whipping the arm.
                return self._reject("ik_branch",
                                    f"IK solution jumped {dev:.2f} rad off the "
                                    "current branch (kinematic boundary?)")
            # ---- reach / joint-speed limiter (09-04): shorten the step
            # instead of letting the safety monitor stop the episode at the
            # elbow-straight boundary. Only when a previous streamed pose (or
            # the measured TCP) gives a feasible anchor to shorten towards.
            viol = self._limit_violation(q, qref, dt) if qref is not None else None
            if viol is not None:
                prev = self._last_cmd_pose
                if prev is None and self._recv is not None:
                    prev = np.asarray(self._recv.getActualTCPPose(), dtype=float)
                if prev is not None:
                    tgt = np.asarray(tcp_pose, dtype=float).copy()
                    tgt[3:6] = rotvec_nearest(prev[3:6], tgt[3:6])
                    best = self._limited_step(ctrl, prev, tgt, qref, dt)
                    self._limiter_hits += 1
                    now = time.perf_counter()
                    if now - self._limiter_log_t >= self.LIMITER_LOG_PERIOD_S:
                        self._limiter_log_t = now
                        log.warning("servo limiter (%s): elbow %s -> %s "
                                    "(hits %d, holds %d this session)",
                                    viol,
                                    ("%.1f deg" % np.degrees(float(q[2])))
                                    if q else "n/a (no IK solution)",
                                    ("%s %.0f%%" % (best[3], 100 * best[2]))
                                    if best else "HOLD",
                                    self._limiter_hits, self._limiter_holds)
                    if best is None:
                        self._limiter_holds += 1
                        # counted in the same streak as IK rejects: a
                        # sustained hold must end the episode, not freeze
                        # the arm for the 150 s budget (verify 09-05 #3)
                        return self._reject("limiter_hold",
                                            f"no feasible fraction of the step ({viol})")
                    tcp_pose, q = best[0], best[1]
                else:
                    return self._reject("limiter_no_anchor", "no anchor pose to shorten towards")
            if not self._ik_valid(q):
                return self._reject("ik_invalid", f"no streamable solution ({q!r})")
            ok = ctrl.servoJ(list(q), 0.0, 0.0, dt, lookahead, gain)
            if ok is not False:
                self._last_qsol = list(q)
                self._last_cmd_pose = np.asarray(tcp_pose, dtype=float).copy()
                self._ik_rejects = 0          # the streak ends only on a SENT tick
                self._branch_rejects = 0
                self._hold_since = None
        if ok is False:
            diag = self._diagnose_control_loss(q, tcp_pose)
            raise ControlLost("servoJ rejected — the RTDE control script is not "
                               "running (clear the pendant popup / protective stop "
                               f"and restart the session); robot: {diag.get('summary', 'n/a')}")
        return ServoResult(True, tcp_pose, "sent")

    _SAFETY_BITS = ("normal", "reduced", "protective_stopped", "recovery",
                    "safeguard_stop", "system_estop", "robot_estop",
                    "emergency_stopped", "violation", "fault", "stopped_due_to_safety")

    def _diagnose_control_loss(self, q_cmd, pose_cmd) -> dict:
        """Snapshot WHY the controller dropped the servo stream (rig 09-07: 15
        of 43 episodes ended as 'servoJ rejected' with nothing on our side —
        no guard, no IK event — and stop.json carried no robot-side cause).
        Reads the safety status bits, modes, measured state and the force at
        the trip, plus the last commanded joint target vs the measured one.
        Never raises; kept in `control_loss_last` for stop.json."""
        d: dict = {"t": time.time()}
        r = self._recv
        try:
            if r is not None:
                d["protective_stop"] = bool(r.isProtectiveStopped())
                d["robot_mode"] = int(r.getRobotMode())
                d["safety_mode"] = int(r.getSafetyMode())
                try:
                    d["safety_status_bits"] = int(r.getSafetyStatusBits())
                except Exception:
                    pass
                q_act = np.asarray(r.getActualQ(), dtype=float)
                d["q_actual"] = np.round(q_act, 4).tolist()
                d["qd_actual"] = np.round(np.asarray(r.getActualQd(), dtype=float), 4).tolist()
                d["tcp_actual"] = np.round(np.asarray(r.getActualTCPPose(), dtype=float), 4).tolist()
                d["tcp_force"] = np.round(np.asarray(r.getActualTCPForce(), dtype=float), 2).tolist()
                if q_cmd is not None and len(q_cmd) == 6:
                    qc = np.asarray(q_cmd, dtype=float)
                    d["q_cmd"] = np.round(qc, 4).tolist()
                    d["q_cmd_minus_actual_max_rad"] = float(np.max(np.abs(qc - q_act)))
                if pose_cmd is not None:
                    d["tcp_cmd"] = np.round(np.asarray(pose_cmd, dtype=float), 4).tolist()
        except Exception as e:
            d["read_error"] = repr(e)
        try:
            import dashboard_client
            db = dashboard_client.DashboardClient(self.hw.arm.ip)
            db.connect()
            try:
                d["dashboard_safety"] = str(db.safetystatus()).strip()
                d["dashboard_robotmode"] = str(db.robotmode()).strip()
                d["dashboard_program"] = str(db.programState()).strip()
            finally:
                db.disconnect()
        except Exception as e:
            d["dashboard_error"] = repr(e)
        bits = d.get("safety_status_bits")
        names = [n for i, n in enumerate(self._SAFETY_BITS)
                 if bits is not None and bits & (1 << i)]
        d["safety_status_names"] = names
        f = d.get("tcp_force")
        parts = [f"safety={d.get('dashboard_safety', d.get('safety_mode'))}",
                 f"bits={names or bits}", f"mode={d.get('robot_mode')}"]
        if f:
            parts.append(f"|F|={float(np.linalg.norm(f[:3])):.1f}N")
        if "q_cmd_minus_actual_max_rad" in d:
            parts.append(f"dq_cmd={d['q_cmd_minus_actual_max_rad']:.3f}rad")
        d["summary"] = " ".join(parts)
        self.control_loss_last = d
        log.error("servo control lost: %s", d["summary"])
        return d

    def _servo_l_bounded_hold(self, ctrl, tcp_pose, dt, lookahead, gain, *, target_guard=None):
        """Opt-in: stream a verified held anchor without fabricating progress.

        Invalid solves retain the historical fault counter. Only explicit
        finite/on-branch envelope holds get the separate elapsed-time budget.
        The caller owns the control lock; native measured safety still runs.
        """
        from phantom.drivers.servo_hold import (
            ConstraintHoldBudget,
            ServoHoldTimeout as ConstraintHoldTimeout,
            verified_constraint_hold,
        )

        limits = self._servo_limits()
        if not limits.enabled:
            raise ValueError("servo_constraint_hold_s requires an enabled servo limiter")
        if self._constraint_hold_budget is None:
            self._constraint_hold_budget = ConstraintHoldBudget(
                self.hw.safety.servo_constraint_hold_s
            )
        qref = self._last_qsol
        prev = self._last_cmd_pose
        if qref is None and self._recv is not None:
            qref = list(self._recv.getActualQ())
        if prev is None and self._recv is not None:
            prev = np.asarray(self._recv.getActualTCPPose(), dtype=float)
        selection = None if target_guard is None else target_guard.terminal_selection(prev, qref, dt, limits)
        if selection is None:
            selection = servo_limiter.select_servo_step(
                tcp_pose, prev, qref, dt,
                lambda pose, seed: self._solve_ik(ctrl, pose, seed), limits,
            )
        held = verified_constraint_hold(selection, qref, dt, limits)
        if not selection.accepted and not held:
            return self._reject(
                selection.reason, "unverified constrained hold or IK fault",
                strict_fault=True,
            )
        sent_q = np.asarray(qref if held else selection.q, dtype=float)
        sent_pose = np.asarray(prev if held else selection.pose, dtype=float)
        if np.array_equal(sent_q, np.asarray(qref)):
            # IK tolerance can accept a nearby pose with identical joints.
            # Re-sending those joints cannot advance the Cartesian anchor.
            sent_pose = np.asarray(prev, dtype=float)
        state = self._constraint_hold_budget.check(time.perf_counter(), sent_q, held=held)
        self.constraint_hold_last = {
            **state, "held": held, "mode": selection.mode,
            "violation": selection.violation, "ik_calls": selection.ik_calls,
            "all_ik_valid": selection.all_ik_valid,
            "all_ik_on_branch": selection.all_ik_on_branch,
        }
        if state["timed_out"]:
            raise ConstraintHoldTimeout("verified constraint hold exceeded its fixed deadline")
        if target_guard is not None:
            # Calibrated controller FK (active TCP), not a nominal UR model.
            # A missing/failed FK method fails before any joint submission.
            sent_pose = np.asarray(ctrl.getForwardKinematics(sent_q.tolist()), dtype=float)
            target_guard.verify_final(
                time.perf_counter(), sent_pose, sent_q,
                verified=bool(selection.all_ik_valid and selection.all_ik_on_branch),
                dt=dt, previous_pose=prev,
            )
        ok = ctrl.servoJ(sent_q.tolist(), 0.0, 0.0, dt, lookahead, gain)
        if ok is False:
            diag = self._diagnose_control_loss(sent_q, sent_pose)
            raise ControlLost(
                "servoJ rejected while streaming constrained servo target; "
                f"robot: {diag.get('summary', 'n/a')}"
            )
        self._last_qsol = sent_q.tolist()
        self._last_cmd_pose = sent_pose.copy()
        if target_guard is not None:
            target_guard.note_submission(time.perf_counter(), sent_pose, sent_q, mechanism="native_servoJ")
            target_guard.acknowledge(time.perf_counter(), sent_pose)
        if selection.violation is not None:
            self._limiter_hits += 1
        if held:
            self._limiter_holds += 1
            # Streamed hold is not IK recovery; preserve preceding fault count.
        else:
            self._ik_rejects = 0
            self._branch_rejects = 0
            self._hold_since = None
        return ServoResult(True, sent_pose, "constraint_hold" if held else "sent")

    def _limiter_enabled(self) -> bool:
        sf = getattr(getattr(self, "hw", None), "safety", None)
        return sf is not None and (getattr(sf, "elbow_min_rad", None) is not None
                                   or getattr(sf, "servo_joint_speed_max_rad_s", None) is not None)

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
        if self._limiter_hits:
            log.info("servo limiter: %d ticks shortened, %d held this servo "
                     "session", self._limiter_hits, self._limiter_holds)
        # keep the finished session's counters readable (stop.json) after reset
        self.limiter_last = {"hits": self._limiter_hits, "holds": self._limiter_holds,
                             "ik_rejects": self._ik_rejects_total,
                             "ik_dev_max_rad": round(self._ik_dev_max, 4)}
        if self.constraint_hold_last:
            self.limiter_last["constraint_hold"] = self.constraint_hold_last.copy()
        self._last_qsol = None
        self._last_cmd_pose = None
        if self._constraint_hold_budget is not None:
            self._constraint_hold_budget.reset()
        self._limiter_hits = 0
        self._limiter_holds = 0
        self._ik_rejects = 0
        self._branch_rejects = 0
        self._hold_since = None
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
