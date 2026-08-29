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
from dataclasses import dataclass

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.data import derived as dv
from phantom.deploy.safety import arm_stale_s, camera_stale_s
from phantom.inference.policy import ObsSnapshot, Plan, PhantomPolicy
from phantom.recording.workers import SensorSession

log = logging.getLogger(__name__)

SYSTEM_MODES = ("teacher", "student", "vision_only", "no_distill", "drop_tactile")
TACTILE_INPUT_MODES = ("teacher",)   # modes whose MODEL consumes tactile streams
# Modes whose model must ALSO not see the wrist F/T window (P10A, review
# 2026-08-28). Without this the "tactile-free" arms still received
# WristTCN(wrist F/T) through OBS_PROPRIO — vision_only, no_distill and
# drop_tactile were input-identical to `student`, so eval/aggregate.py's
# recovery_ratio = (student - vision_only)/(teacher - vision_only) had no
# producible denominator and any student gain could be attributed to the
# surviving wrist signal. The window is still RECORDED (the rig wears the
# sensors in every mode) — it is zeroed on the way into the model only.
WRIST_MASKED_MODES = ("vision_only", "drop_tactile")


class StaleStreamError(AssertionError):
    """A ring the snapshot HARD-REQUIRES is stale (or not flowing yet).

    Subclasses AssertionError deliberately: `_wait_rings_warm` treats it as
    "not warm yet" during start-up and every existing caller that catches
    AssertionError keeps working unchanged.

    `reason` is the executor stop reason it becomes once the episode is under
    way. Before 2026-08-27 these were bare asserts: raised mid-episode from
    inside `SnapshotBuilder.build()` they escaped `run_episode`'s try/finally
    and reached run_deploy as an rc-1 traceback — no operator label prompt, and
    an episode directory left behind with nothing in it. They now end the
    episode through the SAME path as any other stop (executor halted, trace
    written, label prompt shown)."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def snapshot_stop_reason(exc: BaseException) -> str:
    """Executor stop reason for a snapshot that could not be built.

    Anything that is not a StaleStreamError (a plain assert somewhere in
    build(), a malformed ring row) still ends the episode cleanly rather than
    crashing the process — it just gets the generic reason."""
    return getattr(exc, "reason", "snapshot_invalid")


class SnapshotBuilder:
    """Builds one ObsSnapshot per replan.

    `parity_fixes` (P2, review 2026-08-28) switches three deploy-vs-training
    mismatches to the training construction. Default OFF so the rig A/B can
    attribute the effect; every one of them is a *verified* difference between
    what `WindowSampler.sample()` writes and what this builder used to read:

    1. `prev_chunk` — was the policy's own last PROPOSAL (see PlannerLoop.run's
       former DEFERRED note); training writes 16 MEASURED `derived.pose_delta`
       rows on the grid `t0 - (H-k)/rate` plus the gripper command that was
       actually sent (`data_collect/session.py` records exactly that pair into
       STREAM_ACTIONS, `windows.py:277` samples it nearest-neighbour).
    2. contact-state `dt` — was the NOMINAL `1/field_ds_rate_hz`; training's
       `WindowSampler._field_frame` returns the MEASURED `ts[i] - ts[i-1]`,
       and `derived.derive_timestep` divides the tangential flow by it, so a
       jittery ring rescaled `slip` at deploy only.
    3. `reactive` — was the frame difference between the last two REPLANS
       (~1 s apart at the rig's cadence); training differences the two
       consecutive `fields_ds` frames at t0 (~1/30 s apart), i.e. a ~30x
       larger dt fed the same CASA `psi_react` MLP.
    """

    def __init__(self, hw: HardwareConfig, session: SensorSession, mode: str,
                 *, parity_fixes: bool = False, executor=None):
        assert mode in SYSTEM_MODES, f"unknown system mode {mode}"
        self.hw = hw
        self.session = session
        self.mode = mode
        self.parity_fixes = bool(parity_fixes)
        # only used by the parity prev_chunk: the gripper command per EXECUTED
        # action-grid step (ChunkExecutor.gripper_cmd_at)
        self.executor = executor
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
        if self.parity_fixes:
            # ... and enough to reach back one full prev_chunk + the extra
            # step the first delta is chained from: (H + 1) / action_rate_hz
            # (1.7 s at 16/10 Hz => ~213 rows at 125 Hz), with 20% margin.
            span_s = (hw.control.chunk_horizon + 1) / hw.control.action_rate_hz
            self._n_arm = max(self._n_arm,
                              int(np.ceil(1.2 * span_s * hw.arm.rtde_receive_hz)) + 8)

    def prev_chunk_from_history(self, t_now: float, ts_a: np.ndarray,
                                arm: dict, grip_now: float) -> np.ndarray | None:
        """The EXECUTED past as `WindowSampler.sample()` builds `prev_chunk`.

        Training (`windows.py:277`) samples the recorded STREAM_ACTIONS rows
        nearest the grid `t0 - (H - k)/rate`, k = 0..H-1; those rows are
        `derived.pose_delta(prev_tcp, tcp)` between consecutive MEASURED TCP
        poses one action-grid tick apart, with the gripper command that was
        sent alongside (`data_collect/session.py:733-750`, teleop loop period
        == 1/action_rate_hz). Here: pick the measured `tcp_pose` nearest each
        of those H+1 grid times (one extra step back so the first delta has a
        predecessor) and chain `pose_delta` over them — identical convention,
        identical rotvec continuity guard.

        Returns None when the ring cannot supply two distinct rows; the caller
        then keeps the normalized-zero first-replan conditioning.
        """
        H = self.hw.control.chunk_horizon
        rate = self.hw.control.action_rate_hz
        ts_a = np.asarray(ts_a, dtype=np.float64)
        if ts_a.size < 2:
            return None
        poses = np.asarray(arm["tcp_pose"], dtype=np.float64).reshape(ts_a.size, -1)
        # k = -1 .. H-1: the k = -1 row is only the predecessor of delta 0
        grid = t_now - (H - np.arange(-1, H)) / rate
        idx = np.abs(ts_a[None, :] - grid[:, None]).argmin(axis=1)
        p = poses[idx]
        deltas = np.stack([dv.pose_delta(p[k], p[k + 1]) for k in range(H)])
        # gripper: the command the executor actually entered at that grid time;
        # before the first executed step (first replan / warm-up) the measured
        # aperture is the honest stand-in — the gripper is holding it.
        g = None
        if self.executor is not None:
            g = self.executor.gripper_cmd_at(grid[1:])
        g = np.full(H, np.nan) if g is None else np.asarray(g, dtype=np.float64)
        g = np.where(np.isfinite(g), g, float(grip_now))
        return np.concatenate([deltas, g[:, None]], axis=1).astype(np.float32)

    def build(self) -> ObsSnapshot:
        hw = self.hw
        rings = self.session.rings
        t_now = time.perf_counter()

        cam_name = "camera_scene"
        ts, cam = rings[cam_name].latest(1)
        if not len(ts):
            raise StaleStreamError("camera_scene_stale", "no camera frames yet")
        # Freshness is a HARD requirement, like the gripper state below: a
        # wedged RealSense pipeline leaves this ring frozen on the pre-stall
        # frame and every later plan would be conditioned on it with nothing
        # detecting the stall (blocker 2026-08-26). _wait_rings_warm treats the
        # AssertionError as "not warm yet" during start-up; after that it ends
        # the episode loudly. SafetyMonitor carries the same threshold and stops
        # the executor first, within one tick.
        cam_age = t_now - float(ts[0])
        if cam_age >= self._cam_stale_s:
            raise StaleStreamError(
                "camera_scene_stale",
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
        if not len(ts_a):
            raise StaleStreamError("arm_stale", "no arm samples yet")
        arm_age = t_now - float(ts_a[-1])
        if arm_age >= self._arm_stale_s:
            raise StaleStreamError(
                "arm_stale",
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
        if self.mode in WRIST_MASKED_MODES:
            # input ablation for the sensor-free comparative arms — see
            # WRIST_MASKED_MODES. Zeroed here (not skipped) so the snapshot
            # shape, the staleness checks above and the recording are all
            # unchanged; a model built with mc.mask_wrist=True zeroes it again
            # inside HHT, so the two paths agree.
            wrist_window = np.zeros_like(wrist_window)

        _, arm1 = rings["arm"].latest(1)
        _, grip = rings["gripper"].latest(1)
        # HARD requirement, not a fallback: a zeros(2) substitute is a frozen
        # -1.85sigma gripper-position input that collapses the sampled action
        # magnitude ~3x (field root-cause 2026-08-14 — the deploy path never
        # polled the gripper, so every rig episode ran on this substitute).
        # _wait_rings_warm covers the startup window; a raise after that
        # means the gripper feed stalled and the episode must end loudly.
        if not len(grip["state"]):
            raise StaleStreamError("gripper_stale", "no gripper state yet")
        gr = grip["state"][0]
        ur_state = np.concatenate([
            arm1["q"][0], arm1["qd"][0], arm1["tcp_pose"][0], arm1["tcp_speed"][0], gr,
        ]).astype(np.float32)

        snap = ObsSnapshot(t=t_now, rgb=rgb, wrist_window=wrist_window,
                           ur_state=ur_state)
        if self.parity_fixes:
            snap.prev_chunk = self.prev_chunk_from_history(
                t_now, ts_a, arm, float(gr[0]))

        if self.mode in TACTILE_INPUT_MODES:
            fields, gels, contact_states = [], [], []
            dt_field = 1.0 / hw.recording.field_ds_rate_hz
            frame_pairs = []
            for s in hw.tactile.sensors:
                _, kf = rings[f"tactile_{s.name}_kf"].latest(1)
                fields.append(np.asarray(kf["keyframe"][0], dtype=np.float32))
                img_ring = rings.get(f"tactile_{s.name}_img")
                if img_ring is not None:
                    _, img = img_ring.latest(1)
                    gels.append(img["infer_img"][0])
                ts_t, tac = rings[f"tactile_{s.name}"].latest(2)
                cur = np.asarray(tac["fields_ds"][-1], dtype=np.float32)
                prev = np.asarray(tac["fields_ds"][0], dtype=np.float32)
                frame_pairs.append((cur, prev))
                dt_use = dt_field
                if self.parity_fixes and len(ts_t) >= 2:
                    # training reads the MEASURED inter-frame dt out of the
                    # stream (windows.py `_field_frame`); derive_timestep
                    # divides the tangential flow by it
                    dt_use = float(max(float(ts_t[-1]) - float(ts_t[0]), 1e-6))
                d = dv.derive_timestep(cur, prev, dt_use, hw)
                contact_states.append(np.concatenate([
                    tac["wrench"][-1], [tac["area"][-1]],
                    np.nan_to_num(d["cop"], nan=0.0), [d["slip"]], [d["mask_frac"]],
                ]).astype(np.float32))
            snap.fields = np.stack(fields)
            snap.gel = np.stack(gels) if gels else None
            snap.contact_state = np.stack(contact_states)
            ds_now = np.stack([c for c, _ in frame_pairs])
            if self.parity_fixes:
                # training: the two CONSECUTIVE fields_ds frames at t0, not
                # the last two replans (~1 s apart on the rig)
                snap.reactive = dv.reactive_score(
                    ds_now, np.stack([p for _, p in frame_pairs]))
            elif self._prev_fields is not None:
                snap.reactive = dv.reactive_score(ds_now, self._prev_fields)
            self._prev_fields = ds_now
        return snap


@dataclass
class TerminalVeto:
    """Deploy-time terminal commitment guard (review P3, 2026-08-28).

    The ACC gate is read-only at deploy — `plan.gate` / `plan.p_evt` are logged
    and nothing in `phantom/deploy/` lets them modify a chunk — while the rig
    closes on air at 65-120 mm and lifts anyway. This is the scripted override
    (2503.23835 gets 100% disturbance resilience from exactly this shape of
    rule; 2410.13124 reports 11.5% phantom grasps even for a tactile-equipped
    diffusion policy, so a scripted veto is defensible rather than a patch).

    Two rules, both OFF unless `--terminal-veto` is passed:

    close-mask   a commanded gripper-close transition is rewritten to HOLD the
                 current aperture unless p_contact > p_close, or the TCP is
                 already within `z_margin` of the task's z floor (where closing
                 is what a demo would do).
    recovery     if a close WAS commanded and the very next replan reports
                 p_evt[none] > p_none, the grasp is phantom: command the task
                 open aperture and forbid any upward z in the chunk, so the
                 policy re-descends instead of lifting nothing. Capped at
                 `max_retries` cycles per episode, then the episode ends with
                 reason `veto_retry_cap` (an uncapped open/re-descend loop is
                 the safety gap the review's completeness critic flagged).
    """
    p_close: float = 0.5              # theta_close on p_contact = 1 - p_evt[none]
    p_none: float = 0.9               # p_evt[none] above which a close is phantom
    max_retries: int = 3
    z_floor: float | None = None      # m; from start_poses.yaml tcp_z_min
    z_margin: float = 0.015           # "already at the floor" band (15 mm)
    open_aperture: float = 0.0        # task open aperture (demo start mean)
    # Close detection reuses the TRAINING rule (train/common.close_index):
    # aperture past CLOSE_ABS_POS after rising CLOSE_ABS_RISE from where it is.
    close_pos: float = 0.45
    close_rise: float = 0.15


class PlannerLoop:
    """Runs replan cycles; publishes plans to the executor; keeps a trace."""

    def __init__(self, hw: HardwareConfig, policy: PhantomPolicy,
                 snapshots: SnapshotBuilder, executor, *, trace: list | None = None,
                 session: SensorSession | None = None,
                 veto: "TerminalVeto | None" = None):
        self.hw = hw
        self.policy = policy
        self.snapshots = snapshots
        self.executor = executor
        self.veto = veto
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

    # ------------------------------------------------------------------
    def _apply_veto(self, plan, tcp_pose: np.ndarray, grip_now: float,
                    state: dict) -> dict | None:
        """Rewrite `plan.actions` in place per TerminalVeto. Returns the trace
        record (or None when the veto is off).

        Both rules only ever touch the gripper channel and the z channel, and
        both are expressed as a rewrite of the CHUNK, not of the executor: the
        executor keeps its single contract (play the chunk it was given), and
        the trace records exactly the chunk the arm was asked to follow."""
        v = self.veto
        if v is None:
            return None
        a = plan.actions
        z_now = float(tcp_pose[2])
        p_none = float(plan.p_evt[0]) if np.size(plan.p_evt) else 0.0
        p_contact = 1.0 - p_none
        rec = {"p_contact": round(p_contact, 4), "retries": state["retries"]}

        # ---- (b) phantom-grasp recovery ------------------------------------
        # checked FIRST: it reacts to the close committed one replan ago.
        if state["closed_at"] is not None and p_none > v.p_none:
            state["closed_at"] = None
            state["retries"] += 1
            rec["retries"] = state["retries"]
            if state["retries"] > v.max_retries:
                rec["action"] = "retry_cap"
                return rec
            a[:, 6] = v.open_aperture
            # no lift. NB: REVIEW_SYNTHESIS P3 writes "clamp the next chunk to
            # z >= z_now (no lift)"; taken literally that clamp PERMITS exactly
            # the upward motion its own parenthesis forbids, so the intent —
            # never command a z above where we are, i.e. re-descend or hold —
            # is what is implemented.
            cum = np.cumsum(a[:, 2])
            cum = np.minimum(cum, 0.0)
            a[:, 2] = np.diff(np.concatenate([[0.0], cum]))
            rec["action"] = "recovery_open"
            log.warning("terminal veto: phantom grasp (p_none=%.2f) — opening to "
                        "%.2f and forbidding lift (retry %d/%d)",
                        p_none, v.open_aperture, state["retries"], v.max_retries)
            return rec

        # ---- (a) close mask -------------------------------------------------
        g_max = float(np.max(a[:, 6]))
        closing = g_max > v.close_pos and (g_max - grip_now) > v.close_rise
        if not closing:
            rec["action"] = "none"
            return rec
        at_floor = (v.z_floor is not None and z_now <= v.z_floor + v.z_margin)
        if p_contact > v.p_close or at_floor:
            # `closed_at` is committed by run() only if the executor ACCEPTS
            # this plan — a rejected chunk was never commanded, so it cannot
            # be the close the recovery rule reacts to.
            rec["action"] = "close_allowed"
            rec["at_floor"] = bool(at_floor)
            return rec
        a[:, 6] = grip_now                       # hold the current aperture
        rec["action"] = "close_masked"
        log.warning("terminal veto: close masked (p_contact=%.2f <= %.2f, "
                    "z=%.0f mm) — holding aperture %.2f",
                    p_contact, v.p_close, z_now * 1000, grip_now)
        return rec

    def run(self, max_replans: int | None = None) -> None:
        n = 0
        prev_plan: Plan | None = None
        prev_tcp = prev_cmd = None
        strikes = 0
        # per-episode terminal-veto state: when the last ACCEPTED chunk carried
        # a commanded close, and how many open->re-descend cycles have run
        veto_state = {"closed_at": None, "retries": 0}
        while not self._stop.is_set():
            # BEFORE building a snapshot: a safety stop raised by the executor
            # (e.g. camera_scene_stale) must end the episode through this clean
            # path, not through the staleness assert inside build().
            if self._executor_stopped() or self._workers_dead():
                break
            try:
                snap = self.snapshots.build()
            except AssertionError as e:
                # A hard-required ring went stale mid-episode (the SafetyMonitor
                # usually gets there first, within one executor tick, but the
                # planner must not depend on that). Turn it into a normal stop:
                # letting it propagate killed run_episode's return path, so the
                # operator lost the label prompt and the episode directory was
                # left empty (rig 2026-08-27).
                reason = snapshot_stop_reason(e)
                log.error("snapshot rejected mid-episode (%s): %s — ending the "
                          "episode through the normal stop path", reason, e)
                self.executor.request_stop(reason)
                break
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
            grip_now = float(snap.ur_state[-2]) if np.size(snap.ur_state) >= 2 else 0.0
            veto_rec = self._apply_veto(plan, tcp_pose, grip_now, veto_state)
            if veto_rec is not None and veto_rec.get("action") == "retry_cap":
                self.trace.append({"t": snap.t, "latency_s": plan.latency_s,
                                   "gate": plan.gate, "p_evt": plan.p_evt.tolist(),
                                   "sigma": plan.sigma.tolist(), "accepted": False,
                                   "actions": plan.actions.tolist(), "diag": {},
                                   "terminal_veto": veto_rec})
                log.error("terminal veto: %d open/re-descend retries exhausted — "
                          "ending the episode", self.veto.max_retries)
                self.executor.request_stop("veto_retry_cap")
                break
            accepted = self.executor.submit(plan)
            if veto_rec is not None and veto_rec.get("action") == "close_allowed" \
                    and accepted:
                veto_state["closed_at"] = plan.t_created
            self.trace.append({
                "t": snap.t, "latency_s": plan.latency_s, "gate": plan.gate,
                "p_evt": plan.p_evt.tolist(), "sigma": plan.sigma.tolist(),
                "accepted": accepted,
                "actions": plan.actions.tolist(),
                # terminal-veto decision for this replan (None => veto off)
                "terminal_veto": veto_rec,
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
                # NOTE: this fallback is only what runs WITHOUT --parity-fixes.
                # prev_chunk should be the last H EXECUTED action-grid samples,
                # the way WindowSampler.sample() builds it from recorded
                # actions (phantom/data/windows.py:277) and the way
                # model/acc.py documents it ("previously committed action
                # chunk") — an accepted plan is still only a PROPOSAL, and the
                # governor/blend/rate-limit reshape it before the arm sees it.
                # SnapshotBuilder.prev_chunk_from_history now builds exactly
                # that from the MEASURED arm ring; when it is on, the snapshot
                # carries prev_chunk and PhantomPolicy ignores prev_plan for
                # this channel (prev_plan is still the prev_cpk carrier).
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
