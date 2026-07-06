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
from phantom.deploy.governor import SpeedGovernor
from phantom.deploy.safety import SafetyAction, SafetyMonitor
from phantom.drivers.base import Arm, Gripper
from phantom.inference.policy import Plan

log = logging.getLogger(__name__)


class ChunkExecutor:
    def __init__(self, hw: HardwareConfig, arm: Arm, gripper: Gripper,
                 safety: SafetyMonitor, *, record_action=None):
        self.hw = hw
        self.arm = arm
        self.gripper = gripper
        self.safety = safety
        self.governor = SpeedGovernor(hw.safety.governor)
        self.record_action = record_action     # recorder.record_action hook
        self._plan: Plan | None = None
        self._prev_plan: Plan | None = None
        self._prev_play_time = 0.0             # prev plan keeps playing during blend
        self._swap_t = 0.0
        self._play_time = 0.0                  # governed playback clock
        self._last_action_k = -1               # last action index reported to record_action
        self._last_tick = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stopped_reason: str | None = None

    # ------------------------------------------------------------------
    def submit(self, plan: Plan) -> bool:
        """Accept a new plan if enough of it lies ahead of playback."""
        with self._lock:
            lead_ok = (plan.action_times[-1]
                       > time.perf_counter() + self.hw.control.replan_min_lead_s)
            if not lead_ok:
                log.warning("plan rejected: does not cover replan_min_lead_s")
                return False
            self._prev_plan = self._plan
            self._prev_play_time = self._play_time
            self._plan = plan
            self._swap_t = time.perf_counter()
            self._play_time = 0.0
            self._last_action_k = -1
            return True

    def active_plan(self) -> Plan | None:
        with self._lock:
            return self._plan

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
                self.stopped_reason = "protective_stop"
                break
            if verdict.action == SafetyAction.STOP_EPISODE:
                self.arm.stop(2.0)
                self.stopped_reason = "safety_stop"
                break
            if verdict.action == SafetyAction.CLAMP:
                target = self.safety.clamp_target(target)

            self.arm.servo_l(target, dt_servo, hw.arm.servoj.lookahead_time_s,
                             hw.arm.servoj.gain)
            if grip is not None:
                self.gripper.move(float(np.clip(grip, 0, 1)),
                                  hw.gripper.default_speed, hw.gripper.default_force)

            wait = period - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self.stopped_reason = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="executor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        try:
            self.arm.servo_stop()
        except Exception:
            log.exception("servo_stop failed")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
