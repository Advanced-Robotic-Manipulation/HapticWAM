"""Planner: builds ObsSnapshots from the ring buffers and runs the
receding-horizon replan loop (pipeline.md §6e steps 1-3).

System modes (pipeline.md §8 comparative systems) select which streams enter
the model's snapshot — the tactile workers RUN AND RECORD in every mode (the
rig physically wears the sensors; load-bearing for DAgger relabeling and
student-trial force metrics); student modes merely exclude them from the
model input.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.data import derived as dv
from phantom.deploy.safety import arm_stale_s, camera_stale_s
from phantom.inference.policy import ObsSnapshot, Plan, PhantomPolicy
from phantom.recording.workers import SensorSession

log = logging.getLogger(__name__)

SYSTEM_MODES = ("teacher", "student", "vision_only", "no_distill", "drop_tactile")
TACTILE_INPUT_MODES = ("teacher",)   # modes whose MODEL consumes tactile streams


class SnapshotBuilder:
    def __init__(self, hw: HardwareConfig, session: SensorSession, mode: str):
        assert mode in SYSTEM_MODES, f"unknown system mode {mode}"
        self.hw = hw
        self.session = session
        self.mode = mode
        self._prev_fields: np.ndarray | None = None
        self._cam_stale_s = camera_stale_s(hw)
        self._arm_stale_s = arm_stale_s(hw)
        # Enough arm rows to SPAN the wrist window in TIME (the ring runs at
        # rtde_receive_hz, which is not wrist_ft.rate_hz on a cb3), with 2x
        # margin — sampling below is by timestamp, not by row count, and a ring
        # that somehow does not reach back a full window_s only makes np.interp
        # clamp (flat-extend) its head rather than fail.
        self._n_arm = max(2, int(np.ceil(2.0 * hw.wrist_ft.window_s
                                         * hw.arm.rtde_receive_hz)) + 8)

    def build(self) -> ObsSnapshot:
        hw = self.hw
        rings = self.session.rings
        t_now = time.perf_counter()

        cam_name = "camera_scene"
        ts, cam = rings[cam_name].latest(1)
        assert len(ts), "no camera frames yet"
        # Freshness is a HARD requirement, like the gripper state below: a
        # wedged RealSense pipeline leaves this ring frozen on the pre-stall
        # frame and every later plan would be conditioned on it with nothing
        # detecting the stall (blocker 2026-08-26). _wait_rings_warm treats the
        # AssertionError as "not warm yet" during start-up; after that it ends
        # the episode loudly. SafetyMonitor carries the same threshold and stops
        # the executor first, within one tick.
        cam_age = t_now - float(ts[0])
        assert cam_age < self._cam_stale_s, (
            f"camera_scene stale by {cam_age:.2f}s (>{self._cam_stale_s:.2f}s) — "
            "the scene pipeline is wedged; the policy must not replan on a "
            "frozen frame")
        rgb = cam["color"][0]

        ts_a, arm = rings["arm"].latest(self._n_arm)
        # Same hard requirement as the camera above: a dead RTDE-receive worker
        # freezes tcp_pose / tcp_speed / the F/T window / protective_stop all at
        # once, and every later plan is conditioned on a robot that is no longer
        # where the snapshot says. SafetyMonitor carries the same threshold and
        # stops the executor within one tick; _wait_rings_warm treats the
        # AssertionError as "not warm yet" during start-up.
        assert len(ts_a), "no arm samples yet"
        arm_age = t_now - float(ts_a[-1])
        assert arm_age < self._arm_stale_s, (
            f"arm ring stale by {arm_age:.2f}s (>{self._arm_stale_s:.2f}s) — "
            "the RTDE receive stream is dead; the policy must not replan on a "
            "frozen robot state")

        # Wrist F/T window, TRAINING PARITY (Codex review 2026-08-27):
        # WindowSampler.sample() resamples the recorded F/T stream onto
        # np.linspace(t0 - window_s, t0, window_len) with np.interp
        # (phantom/data/windows.py). Taking the last window_len ROWS instead
        # made the deployed window a sample COUNT, not a duration: it silently
        # equals the trained window only while the ring runs at exactly
        # wrist_ft.rate_hz with no dropped samples (a cb3 at 125 Hz would feed
        # the WristTCN a 4x-long window; RTDE jitter warps it either way).
        # Anchor at the newest arm SAMPLE time, not t_now: t_now is later, and
        # np.interp would clamp — i.e. flat-extrapolate — the window's tail.
        L = hw.wrist_ft.window_len
        t_ft = float(ts_a[-1])
        ft = np.asarray(arm["ft"], dtype=np.float64).reshape(len(ts_a), -1)
        grid = np.linspace(t_ft - hw.wrist_ft.window_s, t_ft, L)
        # Warm-up (fewer samples than the window spans) needs no special case:
        # np.interp clamps to ft[0] below the first timestamp, exactly the
        # leading repeat-pad the row-slice version applied.
        wrist_window = np.stack(
            [np.interp(grid, ts_a, ft[:, k]) for k in range(ft.shape[1])],
            axis=-1).astype(np.float32)

        _, arm1 = rings["arm"].latest(1)
        _, grip = rings["gripper"].latest(1)
        # HARD requirement, not a fallback: a zeros(2) substitute is a frozen
        # -1.85sigma gripper-position input that collapses the sampled action
        # magnitude ~3x (field root-cause 2026-08-14 — the deploy path never
        # polled the gripper, so every rig episode ran on this substitute).
        # _wait_rings_warm covers the startup window; a raise after that
        # means the gripper feed stalled and the episode must end loudly.
        assert len(grip["state"]), "no gripper state yet"
        gr = grip["state"][0]
        ur_state = np.concatenate([
            arm1["q"][0], arm1["qd"][0], arm1["tcp_pose"][0], arm1["tcp_speed"][0], gr,
        ]).astype(np.float32)

        snap = ObsSnapshot(t=t_now, rgb=rgb, wrist_window=wrist_window,
                           ur_state=ur_state)

        if self.mode in TACTILE_INPUT_MODES:
            fields, gels, contact_states = [], [], []
            dt_field = 1.0 / hw.recording.field_ds_rate_hz
            for s in hw.tactile.sensors:
                _, kf = rings[f"tactile_{s.name}_kf"].latest(1)
                fields.append(np.asarray(kf["keyframe"][0], dtype=np.float32))
                img_ring = rings.get(f"tactile_{s.name}_img")
                if img_ring is not None:
                    _, img = img_ring.latest(1)
                    gels.append(img["infer_img"][0])
                _, tac = rings[f"tactile_{s.name}"].latest(2)
                cur = np.asarray(tac["fields_ds"][-1], dtype=np.float32)
                prev = np.asarray(tac["fields_ds"][0], dtype=np.float32)
                d = dv.derive_timestep(cur, prev, dt_field, hw)
                contact_states.append(np.concatenate([
                    tac["wrench"][-1], [tac["area"][-1]],
                    np.nan_to_num(d["cop"], nan=0.0), [d["slip"]], [d["mask_frac"]],
                ]).astype(np.float32))
            snap.fields = np.stack(fields)
            snap.gel = np.stack(gels) if gels else None
            snap.contact_state = np.stack(contact_states)
            ds_now = np.stack([np.asarray(
                rings[f"tactile_{s.name}"].latest(1)[1]["fields_ds"][0],
                dtype=np.float32) for s in hw.tactile.sensors])
            if self._prev_fields is not None:
                snap.reactive = dv.reactive_score(ds_now, self._prev_fields)
            self._prev_fields = ds_now
        return snap


class PlannerLoop:
    """Runs replan cycles; publishes plans to the executor; keeps a trace."""

    def __init__(self, hw: HardwareConfig, policy: PhantomPolicy,
                 snapshots: SnapshotBuilder, executor, *, trace: list | None = None,
                 session: SensorSession | None = None):
        self.hw = hw
        self.policy = policy
        self.snapshots = snapshots
        self.executor = executor
        self.trace = trace if trace is not None else []
        # optional: when given, a dead sensor/arm worker ends the episode as
        # `worker_died` instead of the loop replanning on whatever the abandoned
        # ring last held (workers.py: a dead worker "aborts the in-progress
        # episode ... rather than silently corrupting or gapping the stream")
        self.session = session
        self._stop = threading.Event()

    # Stall watchdog (postmortem 2026-08-20, ep ...1999): a protective stop
    # kills the control script, the arm freezes, and the planner replans blind
    # forever - actual motion was 2-10% of commanded for 18 straight replans
    # with zero detection. If actual displacement < STALL_FRACTION x commanded
    # for STALL_STRIKES consecutive replan windows (and the command was big
    # enough to measure), stop the episode loudly.
    STALL_FRACTION = 0.2
    STALL_MIN_CMD_M = 0.005
    STALL_STRIKES = 2

    def run(self, max_replans: int | None = None) -> None:
        n = 0
        prev_plan: Plan | None = None
        prev_tcp = prev_cmd = None
        strikes = 0
        while not self._stop.is_set():
            # BEFORE building a snapshot: a safety stop raised by the executor
            # (e.g. camera_scene_stale) must end the episode through this clean
            # path, not through the staleness assert inside build().
            if self._executor_stopped() or self._workers_dead():
                break
            snap = self.snapshots.build()
            tcp_pose = snap.ur_state[2 * self.hw.arm.dof:2 * self.hw.arm.dof + 6]
            cmd = self.executor.last_cmd()
            if prev_cmd is not None and cmd is not None and prev_tcp is not None:
                cmd_d = float(np.linalg.norm(cmd[:3] - prev_cmd[:3]))
                act_d = float(np.linalg.norm(tcp_pose[:3] - prev_tcp[:3]))
                if cmd_d > self.STALL_MIN_CMD_M and act_d < self.STALL_FRACTION * cmd_d:
                    strikes += 1
                    log.warning("stall watchdog: actual %.1fmm vs commanded %.1fmm "
                                "(strike %d/%d)", act_d * 1000, cmd_d * 1000,
                                strikes, self.STALL_STRIKES)
                    if strikes >= self.STALL_STRIKES:
                        log.error("MOTION STALL: arm is not following commands - "
                                  "protective stop / Local mode suspected. Check "
                                  "the pendant. Ending episode.")
                        self.executor.request_stop("motion_stall")
                        break
                else:
                    strikes = 0
            prev_tcp, prev_cmd = tcp_pose.copy(), cmd
            plan = self.policy.replan(snap, prev_plan, tcp_pose)
            accepted = self.executor.submit(plan)
            self.trace.append({
                "t": snap.t, "latency_s": plan.latency_s, "gate": plan.gate,
                "p_evt": plan.p_evt.tolist(), "sigma": plan.sigma.tolist(),
                "accepted": accepted,
                "actions": plan.actions.tolist(),
                # provenance: nfe/guidance/... per replan (audit 2026-08-20 —
                # the A/B condition lived only in the operator's memory)
                "diag": {k: v for k, v in (getattr(plan, "diag", None) or {}).items()
                         if isinstance(v, (int, float, str, bool))},
            })
            from phantom.config.model import EVENTS
            k_evt = int(np.argmax(plan.p_evt))
            log.info("replan %d: latency=%.2fs gate=%.2f sigma_max=%.2f "
                     "p_evt=%s:%.2f accepted=%s",
                     n, plan.latency_s, plan.gate, float(np.max(plan.sigma)),
                     EVENTS[k_evt], float(plan.p_evt[k_evt]), accepted)
            if accepted:
                # Feedback state advances ONLY on a plan the executor took.
                # A rejected plan (too late to cover replan_min_lead_s) is
                # never commanded, so carrying it into the next replan's
                # prev_chunk / prev_cpk conditions the policy on motion that
                # never happened (Codex review 2026-08-27).
                # DEFERRED (deliberately, not an oversight): prev_chunk should
                # be the last H EXECUTED action-grid samples, the way
                # WindowSampler.sample() builds it from recorded actions
                # (phantom/data/windows.py) and the way model/acc.py documents
                # it ("previously committed action chunk") — an accepted plan
                # is still only a PROPOSAL, and the governor/blend/rate-limit
                # reshape it before the arm sees it.
                prev_plan = plan
            n += 1
            if max_replans is not None and n >= max_replans:
                break
            if self._executor_stopped():
                break

    def _workers_dead(self) -> bool:
        """A dead sensor/arm worker leaves its ring frozen (or gapped) with no
        one writing it: fail closed, and make it the executor's stop reason so
        the runtime can end the whole deployment session on it."""
        if self.session is None or self.session.all_alive():
            return False
        log.error("SENSOR WORKER DIED mid-episode — the rings it feeds are "
                  "abandoned; stopping the episode. The session cannot be "
                  "restarted in this process.")
        self.executor.request_stop("worker_died")
        return True

    def _executor_stopped(self) -> bool:
        reason = self.executor.stopped_reason
        if reason is None:
            return False
        log.warning("executor stopped (%s); ending episode", reason)
        return True

    def stop(self) -> None:
        self._stop.set()
