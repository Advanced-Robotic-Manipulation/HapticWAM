"""Chunk executor: streams the active Plan to the arm at executor_rate_hz.

- interpolates the action-rate delta grid to a smooth pose target (cumulative
  deltas from the plan's reference pose, linear interp between grid points);
- the speed governor reparametrizes PLAYBACK TIME (path preserved);
- plan swap: accepted only if the new plan covers now + replan_min_lead_s;
  blended over chunk_blend_s in command space;
- stale plan (no fresh chunk within stale_plan_timeout_s): decelerate to hold,
  keep the servo session alive;
- SafetyMonitor verdict checked every tick BEFORE the command is sent.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.data.derived import rotvec_nearest
from phantom.deploy.governor import SpeedGovernor
from phantom.deploy.safety import SafetyAction, SafetyMonitor
from phantom.drivers.base import Arm, Gripper
from phantom.inference.policy import Plan

log = logging.getLogger(__name__)


class ChunkExecutor:
    def __init__(self, hw: HardwareConfig, arm: Arm, gripper: Gripper,
                 safety: SafetyMonitor, *, record_action=None, gripper_ring=None):
        self.hw = hw
        self.arm = arm
        self.gripper = gripper
        self.safety = safety
        self.governor = SpeedGovernor(hw.safety.governor)
        self.record_action = record_action     # recorder.record_action hook
        # The executor thread is the gripper's ONLY user at deploy, so gripper
        # state polling lives HERE (same single-owner rule as GripperPilot on
        # the collection path — a separate poller thread races move()/
        # get_state() on RobotiqGripper's one TCP socket). Root cause
        # 2026-08-14: no deploy-path poller existed at all, the snapshot's
        # gripper dims were zeros(2) every rig episode.
        self.gripper_ring = gripper_ring
        self._grip_poll_period = 1.0 / hw.gripper.feedback_rate_hz
        self._grip_target: float | None = None
        self._grip_thread: threading.Thread | None = None
        self._last_grip_poll = 0.0
        self._plan: Plan | None = None
        self._prev_plan: Plan | None = None
        self._prev_play_time = 0.0             # prev plan keeps playing during blend
        self._swap_t = 0.0
        self._play_time = 0.0                  # governed playback clock
        self._last_action_k = -1               # last action index reported to record_action
        self._last_tick = 0.0
        self._last_cmd: np.ndarray | None = None   # last pose actually commanded
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stopped_reason: str | None = None
        # SESSION-fatal, not episode-fatal: set when stop() could not join a
        # worker. See stop() — the old worker may still own the Robotiq socket
        # / the servo session, so no further episode may run in this process.
        self.join_failed: bool = False

    # ------------------------------------------------------------------
    def submit(self, plan: Plan) -> bool:
        """Accept a new plan if enough of it lies ahead of playback.

        The plan is REBASED at swap (field root-cause 2026-08-14): its
        t0_pose is the TCP measured at snapshot time, one full inference
        latency ago, and playback used to restart at index 0 — together every
        replan re-commanded already-elapsed motion from an already-left pose,
        rewinding ~all of the previous cycle's advance (measured: 100% of rig
        replan boundaries landed 22-26mm behind the commanded ramp).
        Rebasing anchors the plan so pose_at(elapsed) == the pose we are
        commanding RIGHT NOW, and playback starts at the action index whose
        time it actually is."""
        now = time.perf_counter()
        with self._lock:
            lead_ok = (plan.action_times[-1]
                       > now + self.hw.control.replan_min_lead_s)
            if not lead_ok:
                log.warning("plan rejected: does not cover replan_min_lead_s")
                return False
            rate = self.hw.control.action_rate_hz
            H = plan.actions.shape[0]
            # Playback anchors at action_times[0] = obs.t + inference latency —
            # the SAME origin the lead check above uses. Anchoring at t_created
            # (obs.t) skipped `latency*rate` actions of EVERY chunk: at the
            # rig's ~1.4 s replans that discarded ~14 of 16 actions, so the arm
            # only ever played chunk tails (review find 2026-08-20; the
            # postmortem's 35 executed actions / 17 replans).
            u0 = float(np.clip((now - float(plan.action_times[0])) * rate,
                               0.0, H - 1e-6))
            k0 = int(u0)
            cum = np.cumsum(plan.actions[:, :6], axis=0)
            prev0 = cum[k0 - 1] if k0 > 0 else np.zeros(6)
            c0 = prev0 + (u0 - k0) * (cum[k0] - prev0)
            if self._last_cmd is not None:
                # continuity: target(u0) = t0_pose + c0 == _last_cmd
                plan.t0_pose = self._last_cmd.copy() - c0
            self._prev_plan = self._plan
            self._prev_play_time = self._play_time
            self._plan = plan
            self._swap_t = now
            self._play_time = u0 / rate
            # steps < k0 were never executed — don't report them as executed
            self._last_action_k = k0 - 1
            return True

    def active_plan(self) -> Plan | None:
        with self._lock:
            return self._plan

    def _set_reason(self, reason: str) -> None:
        with self._lock:
            if self.stopped_reason is None:
                self.stopped_reason = reason

    def _halt(self, reason: str) -> None:
        """Stop BOTH threads and invalidate the gripper mailbox. Safety stops
        used to only break the servo loop — the gripper worker could still
        consume the previous tick's target (possibly a close) on an arm that
        had already stopped (review 2026-08-20)."""
        self._set_reason(reason)
        self._grip_target = None
        self._stop.set()

    def request_stop(self, reason: str) -> None:
        """External stop (planner watchdog): record the reason (first writer
        wins) and halt the servo loop now — not at episode teardown."""
        self._halt(reason)

    def last_cmd(self) -> np.ndarray | None:
        """Last commanded TCP pose (copy) - the stall watchdog's reference."""
        with self._lock:
            return None if self._last_cmd is None else self._last_cmd.copy()

    # ------------------------------------------------------------------
    def _pose_at(self, plan: Plan, play_time: float) -> tuple[np.ndarray, float]:
        """Pose target from cumulative deltas at governed playback time."""
        rate = self.hw.control.action_rate_hz
        H = plan.actions.shape[0]
        u = np.clip(play_time * rate, 0.0, H - 1e-6)
        k = int(u)
        frac = u - k
        cum = np.cumsum(plan.actions[:, :6], axis=0)
        prev = cum[k - 1] if k > 0 else np.zeros(6)
        target = plan.t0_pose + prev + frac * (cum[k] - prev)
        grip = plan.actions[min(k, H - 1), 6]
        return target, float(grip)

    def _run(self) -> None:
        hw = self.hw
        period = 1.0 / hw.control.executor_rate_hz
        dt_servo = period
        self._last_tick = time.perf_counter()
        hold_pose: np.ndarray | None = None
        while not self._stop.is_set():
            t0 = time.perf_counter()
            dt = t0 - self._last_tick
            self._last_tick = t0

            with self._lock:
                plan = self._plan
            if plan is None:
                time.sleep(period)
                continue

            stale = (t0 - self._swap_t) > (plan.actions.shape[0]
                                           / hw.control.action_rate_hz
                                           + hw.safety.stale_plan_timeout_s)
            if stale:
                if hold_pose is None:
                    hold_pose = self.arm.get_state().tcp_pose.copy()
                target, grip = hold_pose, None
            else:
                hold_pose = None
                scale = self.governor.scale_profile(plan.sigma)
                with self._lock:
                    self._play_time += dt * scale
                target, grip = self._pose_at(plan, self._play_time)

                # chunk blending: the previous plan keeps playing on its own
                # governed clock and the command is cross-faded over
                # chunk_blend_s (documented contract; removes the swap jerk)
                blend_s = hw.control.chunk_blend_s
                since_swap = t0 - self._swap_t
                with self._lock:
                    prev_plan = self._prev_plan
                if prev_plan is not None:
                    if since_swap < blend_s:
                        prev_scale = self.governor.scale_profile(prev_plan.sigma)
                        self._prev_play_time += dt * prev_scale
                        prev_target, _ = self._pose_at(prev_plan, self._prev_play_time)
                        beta = since_swap / blend_s
                        target = (1.0 - beta) * prev_target + beta * target
                    else:
                        with self._lock:
                            self._prev_plan = None

                # report each newly-entered action-grid step (DAgger rollout
                # episodes need the executed STREAM_ACTIONS like teleop demos)
                if self.record_action is not None:
                    k = min(int(self._play_time * hw.control.action_rate_hz),
                            plan.actions.shape[0] - 1)
                    while self._last_action_k < k:
                        self._last_action_k += 1
                        self.record_action(t0, plan.actions[self._last_action_k])

            verdict = self.safety.check(t0, target)
            if verdict.action == SafetyAction.PROTECTIVE_STOP:
                self._halt("protective_stop")
                break
            if verdict.action == SafetyAction.STOP_EPISODE:
                self.arm.stop(2.0)
                self._halt("safety_stop")
                break
            if verdict.action == SafetyAction.CLAMP:
                target = self.safety.clamp_target(target)

            # Kinematic rate limit on the COMMANDED pose — the model/plan side
            # has no dynamics bound, and replan-boundary jumps otherwise get
            # executed as whips (field 2026-08-11: joint speeds grew to
            # 5.5 rad/s over an episode). Cap the per-tick displacement so any
            # target is approached at arm.limits speeds; the executor converges
            # to the plan whenever the plan itself is feasible.
            target = np.array(target, dtype=np.float64)
            dt_eff = period
            if self._last_cmd is not None:
                # Rate limit per MEASURED tick, not the nominal 8 ms: the old
                # per-period cap silently scaled the speed ceiling by
                # (period / actual dt) — with synchronous gripper I/O in this
                # loop the tick ran ~40 ms and the arm topped out at ~1/5 of
                # tcp_speed_m_s (audit 2026-08-20). Bounded so a stalled tick
                # can never license a jump.
                dt_eff = float(np.clip(dt, period, 2.0 * period))
                v_lin = hw.arm.limits.tcp_speed_m_s
                v_rot = hw.arm.limits.joint_speed_rad_s
                dp = target[:3] - self._last_cmd[:3]
                n = float(np.linalg.norm(dp))
                if n > v_lin * dt_eff:
                    target[:3] = self._last_cmd[:3] + dp * (v_lin * dt_eff / n)
                rv = rotvec_nearest(self._last_cmd[3:6], target[3:6])
                dr = rv - self._last_cmd[3:6]
                rn = float(np.linalg.norm(dr))
                if rn > v_rot * dt_eff:
                    target[3:6] = self._last_cmd[3:6] + dr * (v_rot * dt_eff / rn)
                else:
                    target[3:6] = rv
            with self._lock:
                self._last_cmd = target.copy()

            # servoJ's time parameter = the interval the controller is asked
            # to reach the setpoint in; give it the interval the setpoint was
            # actually sized for, so a delayed tick never doubles the
            # commanded speed (review 2026-08-20)
            self.arm.servo_l(target, dt_eff, hw.arm.servoj.lookahead_time_s,
                             hw.arm.servoj.gain)
            if grip is not None:
                # hand the gripper target to the gripper thread (non-blocking):
                # a synchronous socket round-trip here throttled the servo loop
                self._grip_target = float(np.clip(grip, 0, 1))

            wait = period - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    # ------------------------------------------------------------------
    def _run_guarded(self) -> None:
        """_run with a crash net: an executor that dies (e.g. servoJ rejected
        after a UR protective stop) MUST surface through stopped_reason —
        field-debugged 2026-08-14: an uncaught servo exception killed the
        thread silently and the planner kept replanning into a stopped arm
        for 30+ cycles."""
        try:
            self._run()
        except Exception:
            log.exception("executor thread crashed")
            self._halt("executor_crash")
            self._set_reason("executor_crash")

    GRIP_DEADBAND = 0.008  # ~2/255 counts, same as collection (GripperTuning):
                           # re-sending an unchanged target makes the Robotiq
                           # report OBJ=0 ("moving") for a cycle — the flicker
                           # the model never saw in training
    GRIP_FLUSH_CYCLES = 5  # a target stable this many cycles is sent even if
                           # inside the deadband (no permanent residual)

    def _grip_worker(self) -> None:
        """Single owner of gripper I/O: sends the latest target (deadbanded)
        and polls state into the ring at feedback_rate_hz, off the servo
        thread. Runs while idle so the ring is warm before the first plan."""
        hw = self.hw
        last_sent: float | None = None
        stable_for, last_seen = 0, None
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                tgt = self._grip_target
                stable_for = stable_for + 1 if tgt == last_seen else 0
                last_seen = tgt
                due = tgt is not None and (
                    last_sent is None
                    or abs(tgt - last_sent) > self.GRIP_DEADBAND
                    or (stable_for == self.GRIP_FLUSH_CYCLES and tgt != last_sent))
                if due and not self._stop.is_set():
                    # re-check the stop flag right before I/O: a stop between
                    # reading the mailbox and move() must not close the gripper
                    self.gripper.move(tgt, hw.gripper.default_speed,
                                      hw.gripper.default_force)
                    last_sent = tgt
                if self.gripper_ring is not None:
                    gs = self.gripper.get_state()
                    self.gripper_ring.push(gs.t_host, state=np.array(
                        [gs.position, gs.obj], dtype=np.float32))
            except Exception:
                log.exception("gripper worker error")
                self._halt("executor_crash")
                break
            wait = self._grip_poll_period - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    def start(self) -> None:
        self._stop.clear()
        self.stopped_reason = None
        self._last_cmd = None                  # re-seed the rate limit per episode
        self._grip_target = None
        self._grip_thread = threading.Thread(target=self._grip_worker, daemon=True,
                                             name="gripper")
        self._grip_thread.start()
        self._thread = threading.Thread(target=self._run_guarded, daemon=True,
                                        name="executor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._grip_target = None
        # the gripper worker may be inside a socket transaction (2 s timeout
        # x2 per get_state) — wait long enough to be CONCLUSIVE, and never
        # hand the gripper to a new executor while the old worker is alive
        for th, name, budget in ((self._thread, "executor", 2.0),
                                 (getattr(self, "_grip_thread", None), "gripper", 6.0)):
            if th is None:
                continue
            th.join(budget)
            if th.is_alive():
                # FATAL for the deployment session (Codex review 2026-08-27):
                # a stale gripper worker still owns the one Robotiq socket and
                # a stale servo thread still owns the servo session, so
                # starting a second executor here races both. recover_control()
                # rebuilds the RTDE control script but cannot evict a thread —
                # only a fresh process can. run_deploy refuses to start
                # another episode once this is set.
                log.error("%s thread did not exit within %.0fs — the device is "
                          "still owned by a stale worker. This deployment "
                          "session is over: no further episode may start in "
                          "this process (restart it).", name, budget)
                self.join_failed = True
                self._set_reason("executor_crash")
        try:
            self.arm.servo_stop()
        except Exception:
            log.exception("servo_stop failed")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
