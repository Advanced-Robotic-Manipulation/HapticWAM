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
import traceback
import threading
import time
from collections import deque

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.data.derived import rotvec_nearest
from phantom.deploy.governor import SpeedGovernor
from phantom.deploy.safety import SafetyAction, SafetyMonitor
from phantom.drivers.base import Arm, Gripper
from phantom.inference.policy import Plan

log = logging.getLogger(__name__)


#: Stop conditions that mean "let go of whatever is in the fingers".
#: `_halt` commands the gripper open on these before it kills the workers: the
#: tactile guards exist to protect the gel fingertips, and leaving the fingers
#: squeezing at the pad ceiling through the label prompt and the "clear the
#: arm's path" prompt — minutes, unattended — is the opposite of that
#: (validation_0830/safety-final.md §4).
LETGO_STOP_REASONS = ("wrench_limit", "hitbox_exit", "veto_retry_cap")


def halt_reason_for(events) -> str:
    """Executor stop reason for a STOP_EPISODE verdict: a stop made ONLY of
    lift_complete events is the SUCCESS end (never a let-go); anything else
    stays the generic safety_stop whose events carry the let-go granularity."""
    kinds = {getattr(e, "kind", "") for e in (events or [])}
    return "lift_complete" if kinds and kinds == {"lift_complete"} else "safety_stop"


def is_letgo_reason(name: str | None) -> bool:
    return bool(name) and (str(name).startswith("tactile_")
                           or str(name) in LETGO_STOP_REASONS)


class ChunkExecutor:
    def __init__(self, hw: HardwareConfig, arm: Arm, gripper: Gripper,
                 safety: SafetyMonitor, *, record_action=None, gripper_ring=None,
                 open_aperture: float = 0.0, max_play_steps: int | None = None):
        self.hw = hw
        # Chunk-tail cap (run analysis 09-01 P1 #5): steps beyond
        # HEAD_STEPS are never validated by a subsequent replan, and ALL
        # four whips began in that unsupervised tail, 0.08-0.43 s after the
        # last trace row, while the planner was blocked in inference.
        # Playback holds at the cap point until the next plan lands.
        # None = uncapped (legacy behaviour, and every pre-09-01 test).
        self.max_play_steps = max_play_steps
        # aperture the gripper is commanded to on a "let go" stop (the task's
        # demo START aperture; 0.0 = fully open when the task is unknown)
        self.open_aperture = float(open_aperture)
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
        # aperture latch (SafetyConfig.grip_latch_fz_n): once both pads carry
        # load, the commanded closure may not DECREASE until an intended
        # release — clear_grip_latch() (veto recovery) or the episode end
        self._grip_latch: float | None = None
        self._grip_thread: threading.Thread | None = None
        # single lock over ALL gripper.move calls (worker + release): the
        # ordering swap in _halt alone leaves a window where a latched close
        # lands AFTER the release (measured 1/30..23/70 depending on socket
        # RTT — revalidation 2026-08-31 §2 #5); the lock is the load-bearing
        # half of the fix
        self._grip_io_lock = threading.Lock()
        self._last_grip_poll = 0.0
        self._plan: Plan | None = None
        self._prev_plan: Plan | None = None
        self._prev_play_time = 0.0             # prev plan keeps playing during blend
        self._swap_t = 0.0
        self._play_time = 0.0                  # governed playback clock
        self._last_action_k = -1               # last action index reported to record_action
        # (t, gripper command) of every action-grid step the playback ENTERED —
        # the deploy-side stand-in for the recorded STREAM_ACTIONS gripper
        # channel that WindowSampler reads into prev_chunk (parity fix P2).
        # Only entered steps land here, so it is the command actually sent, not
        # the proposal: rejected plans and never-reached chunk tails never
        # appear. Bounded; ~1 entry per 0.1 s at action_rate_hz.
        self._grip_hist: deque[tuple[float, float]] = deque(maxlen=256)
        self._last_tick = 0.0
        self._last_cmd: np.ndarray | None = None   # last pose the ARM RECEIVED (driver result)
        self._held_ticks = 0                       # consecutive driver holds (telemetry)
        self.halt_state: dict = {}                 # arm state captured AT the halt (review item 7)
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
            u_hi = (min(H, self.max_play_steps)
                    if self.max_play_steps else H) - 1e-6
            u0 = float(np.clip((now - float(plan.action_times[0])) * rate,
                               0.0, u_hi))
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

    def _halt(self, reason: str, *, events=None) -> None:
        """Stop BOTH threads and invalidate the gripper mailbox. Safety stops
        used to only break the servo loop — the gripper worker could still
        consume the previous tick's target (possibly a close) on an arm that
        had already stopped (review 2026-08-20).

        On a stop that means "let go" (`is_letgo_reason` over the reason and
        over the safety events behind a generic `safety_stop`) the gripper is
        commanded OPEN once, synchronously, BEFORE `_stop` is set — after that
        the gripper worker exits without sending anything and the next
        `gripper.move` in the whole deploy path is the NEXT episode's homing
        (start_pose.py:208)."""
        self._set_reason(reason)
        self._grip_latch = None
        # order matters: kill the mailbox and the worker's while-condition
        # BEFORE releasing, or a latched close can land after the release
        self._grip_target = None
        self._stop.set()
        kinds = [getattr(e, "kind", "") for e in (events or [])]
        if is_letgo_reason(reason) or any(is_letgo_reason(k) for k in kinds):
            self._release_gripper(reason if is_letgo_reason(reason)
                                  else next(k for k in kinds if is_letgo_reason(k)))
        # AFTER the let-go: a driver read must never delay a release
        if not self.halt_state:
            self.halt_state = self._snapshot_arm(reason)

    def _snapshot_arm(self, reason: str) -> dict:
        """Arm state AT the halt (before stopJ settles it): stop.json used to
        sample after teardown, so qd read ~0 and the pose was post-stop
        (review 09-05 item 7). Best effort, never raises."""
        try:
            st = self.arm.get_state()
            q = np.asarray(st.q, dtype=float).reshape(-1)
            return {"reason": reason, "t": time.time(),
                    "tcp_pose": np.asarray(st.tcp_pose, dtype=float).reshape(-1).tolist(),
                    "q_deg": np.degrees(q).tolist(),
                    "qd_max": float(np.max(np.abs(np.asarray(st.qd, dtype=float)))),
                    "held_ticks": int(self._held_ticks)}
        except Exception:
            return {"reason": reason, "t": time.time()}

    def _release_gripper(self, why: str) -> None:
        """One blocking `move` to the open aperture. Best-effort: a stop must
        never be lost because the Robotiq socket is unhappy."""
        if self.gripper is None:
            return
        try:
            with self._grip_io_lock:
                self.gripper.move(self.open_aperture, self.hw.gripper.default_speed,
                                  self.hw.gripper.default_force)
            log.warning("stop (%s): gripper RELEASED to %.2f — if it is still "
                        "closed, run ./GRIPPER_OPEN.sh", why, self.open_aperture)
        except Exception:
            log.exception("stop (%s): could not command the gripper open — open "
                          "it by hand / with ./GRIPPER_OPEN.sh", why)

    def request_stop(self, reason: str) -> None:
        """External stop (planner watchdog): record the reason (first writer
        wins) and halt the servo loop now — not at episode teardown."""
        self._halt(reason)

    def last_cmd(self) -> np.ndarray | None:
        """Last commanded TCP pose (copy) - the stall watchdog's reference."""
        with self._lock:
            return None if self._last_cmd is None else self._last_cmd.copy()

    def gripper_cmd_at(self, times) -> np.ndarray | None:
        """Gripper command in force at each of `times` (zero-order hold).

        The parity prev_chunk (P2) needs the gripper channel of the EXECUTED
        action grid, exactly as `WindowSampler.sample()` reads it out of the
        recorded STREAM_ACTIONS rows. Returns None while nothing has been
        executed yet (first replan), or NaN for grid times that precede the
        first entered step — the caller substitutes the measured aperture."""
        times = np.asarray(times, dtype=np.float64)
        with self._lock:
            hist = list(self._grip_hist)
        if not hist:
            return None
        ts = np.asarray([h[0] for h in hist], dtype=np.float64)
        gs = np.asarray([h[1] for h in hist], dtype=np.float64)
        idx = np.searchsorted(ts, times, side="right") - 1
        out = np.where(idx >= 0, gs[np.clip(idx, 0, len(gs) - 1)], np.nan)
        return out.astype(np.float64)

    def entered_grip_after(self, t: float) -> list[tuple[float, float]]:
        """(t, gripper command) of every action-grid step playback ENTERED
        strictly after `t`.

        The terminal veto latches its phantom-grasp recovery on this, not on
        plan acceptance: an accepted chunk whose close lives in a tail the
        playback never reached was never commanded (VALIDATION_0830 P0 #3)."""
        with self._lock:
            return [(ts, g) for ts, g in self._grip_hist if ts > t]

    # ------------------------------------------------------------------
    def _pose_at(self, plan: Plan, play_time: float) -> tuple[np.ndarray, float]:
        """Pose target from cumulative deltas at governed playback time."""
        rate = self.hw.control.action_rate_hz
        H = plan.actions.shape[0]
        u_hi = (min(H, self.max_play_steps)
                if self.max_play_steps else H) - 1e-6
        u = np.clip(play_time * rate, 0.0, u_hi)
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
                # and remember its gripper command (parity prev_chunk, P2).
                # capped by max_play_steps like _pose_at: steps past the cap
                # were NEVER commanded and must not be reported as executed
                # (verification 09-01: an uncapped k fed never-sent closes to
                # _grip_hist / record_action, arming the veto's phantom-grasp
                # recovery and corrupting STREAM_ACTIONS)
                _H = plan.actions.shape[0]
                _cap = (min(_H, self.max_play_steps)
                        if self.max_play_steps else _H)
                k = min(int(self._play_time * hw.control.action_rate_hz),
                        _cap - 1)
                while self._last_action_k < k:
                    self._last_action_k += 1
                    a = plan.actions[self._last_action_k]
                    # report the command that actually GOES OUT: the aperture
                    # latch floors the gripper channel (verification 09-04:
                    # recording the raw proposal wrote "open mid-carry" into
                    # STREAM_ACTIONS and confused the veto's close detector)
                    g_sent = float(np.clip(a[6], 0.0, 1.0))
                    if self._grip_latch is not None:
                        g_sent = max(g_sent, self._grip_latch)
                    if g_sent != a[6]:
                        a = np.array(a, copy=True); a[6] = g_sent
                    if self.record_action is not None:
                        self.record_action(t0, a)
                    # locked: the planner thread reads this deque, and a full
                    # maxlen deque pops-left on append — an unsynchronised
                    # list() over it can raise "mutated during iteration"
                    with self._lock:
                        self._grip_hist.append((t0, g_sent))

            verdict = self.safety.check(t0, target)
            if verdict.action == SafetyAction.PROTECTIVE_STOP:
                # a hard press into the table raises tactile_fz AND the UR
                # protective stop in the same tick; PROTECTIVE_STOP outranks
                # STOP_EPISODE, so without the events the letgo release never
                # ran on exactly that coincidence (revalidation §2 #5)
                self._halt("protective_stop", events=verdict.events)
                break
            if verdict.action == SafetyAction.STOP_EPISODE:
                self.arm.stop(2.0)
                # the EVENTS carry the granularity `safety_stop` loses: a
                # tactile/wrench/hitbox stop must also open the fingers
                self._halt(halt_reason_for(verdict.events), events=verdict.events)
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
            # servoJ's time parameter = the interval the controller is asked
            # to reach the setpoint in; give it the interval the setpoint was
            # actually sized for, so a delayed tick never doubles the
            # commanded speed (review 2026-08-20)
            res = self.arm.servo_l(target, dt_eff, hw.arm.servoj.lookahead_time_s,
                                   hw.arm.servoj.gain)
            # The anchor is what the arm RECEIVED, not what we proposed
            # (issue #7): a driver hold (IK branch reject, invalid IK, limiter
            # hold) keeps the previous anchor, a limiter-shortened step
            # becomes the anchor. Simple drivers return None = sent as given.
            if res is None or (res.sent and res.pose is None):
                streamed = target
            elif res.sent:
                streamed = np.asarray(res.pose, dtype=np.float64)
            else:
                streamed = None
                self._held_ticks += 1
                if self._held_ticks == 1:
                    log.info("servo tick held by the driver (%s) — anchor not advanced",
                             res.reason)
            if streamed is not None:
                self._held_ticks = 0
                with self._lock:
                    self._last_cmd = streamed.copy()
            if grip is not None:
                grip = self._latched_grip(float(np.clip(grip, 0, 1)))
                # hand the gripper target to the gripper thread (non-blocking):
                # a synchronous socket round-trip here throttled the servo loop
                self._grip_target = grip

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
            self.crash_text = traceback.format_exc()[-3000:]
            self._halt("executor_crash")
            self._set_reason("executor_crash")

    crash_text: str | None = None  # traceback of an executor_crash (stop.json)
    def _latched_grip(self, grip: float) -> float:
        """Aperture latch (rig 09-04): in 4 of the 5 objects lost mid-carry
        the policy was commanding the fingers OPEN while carrying. Once both
        pads register `grip_latch_fz_n` of load (trailing-window, dropout
        tolerant) the commanded closure is floored at its value at that
        moment; it can still close further, never open, until an intended
        release clears the latch."""
        thr = float(getattr(self.hw.safety, "grip_latch_fz_n", 0.0) or 0.0)
        if thr <= 0:
            return grip
        loads = getattr(self.safety, "contact_load", None) or {}
        if self._grip_latch is None:
            if len(loads) >= 2 and all(v > thr for v in loads.values()):
                self._grip_latch = grip
                log.info("aperture latched at %.2f (both pads loaded: %s)",
                         grip, {k: round(v, 1) for k, v in loads.items()})
            return grip
        # RUNNING MAX since contact (09-04 analysis): contact registers at
        # ~0.58 and the fingers then close a further ~0.05 into the object;
        # latching at the contact value would under-grip on the way back.
        if grip > self._grip_latch:
            self._grip_latch = grip
        return self._grip_latch

    def clear_grip_latch(self) -> None:
        """An INTENDED release (veto recovery / end of episode) drops the latch."""
        self._grip_latch = None

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
                    # re-check the stop flag INSIDE the shared I/O lock: a stop
                    # between reading the mailbox and move() must not close the
                    # gripper, and without the lock a move already past the
                    # check lands after _halt's release
                    with self._grip_io_lock:
                        if not self._stop.is_set():
                            self.gripper.move(tgt, hw.gripper.default_speed,
                                              hw.gripper.default_force)
                            last_sent = tgt
                if self.gripper_ring is not None:
                    gs = self.gripper.get_state()
                    self.gripper_ring.push(gs.t_host, state=np.array(
                        [gs.position, gs.obj], dtype=np.float32))
            except Exception:
                log.exception("gripper worker error")
                self.crash_text = "gripper worker: " + traceback.format_exc()[-3000:]
                self._halt("executor_crash")
                break
            wait = self._grip_poll_period - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    def start(self) -> None:
        self._stop.clear()
        self.stopped_reason = None
        self._last_cmd = None                  # re-seed the rate limit per episode
        self._held_ticks = 0
        self.halt_state = {}
        self._grip_target = None
        self._grip_hist.clear()                # executed-gripper history is per-episode
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
            # URArm.servo_stop() only re-raises when the control script is
            # STILL RUNNING (a dead script clears the guard itself, per the
            # ur.py invariant): the servo stream may genuinely still be live,
            # so `_servo_active` stays set and every later move_l — i.e. the
            # next episode's start-pose homing — is refused. Swallowed as a log
            # line, that downgraded the whole rest of the campaign to
            # hand-jogged OOD starts (rig 2026-08-27). Surface it as a stop
            # reason run_deploy treats as control-dead, so the RTDE control
            # script is rebuilt (which clears the guard) before the next
            # episode. First-writer-wins: a real episode stop reason
            # (protective_stop, ...) is never overwritten by this.
            log.exception("servo_stop failed — the servo session may still be "
                          "live on a running control script; the arm's move_l "
                          "guard stays up until the script is rebuilt")
            self._set_reason("servo_stop_failed")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
