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
from typing import Callable

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
# veto actions that REWRITE the chunk (the trace then carries actions_pre_veto and
# the replay tools must not score the veto's arithmetic as a model sample — F9)
VETO_REWRITE_ACTIONS = ("close_masked", "recovery_open", "recovery_tactile")


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

        # SAME fetch as the wrist anchor above: a second latest(1) could be a
        # row newer than ts_a[-1] (125 Hz sample landing between the reads),
        # leaving ur_state inconsistent with its own wrist window — the race
        # behind the flaky snapshot-parity test AND a real deploy-side skew
        # (revalidation 2026-08-31 §2 #7)
        arm1 = {k: v[-1:] for k, v in arm.items()}
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
                 already inside the task's demo CLOSE band (where closing is
                 what a demo would do): `z <= z_ref + z_margin`, with `z_ref`
                 the demo `tcp_z_min` and `z_margin` sized so the band reaches
                 the per-task demo close p95 (eval/grasp_label.Z_MAX_MM).
                 Measuring it from the already-lowered SAFETY floor with a
                 15 mm margin put the hatch 30-90 mm below every demo close and
                 ~50 mm below the model's own predicted close height, so it
                 never fired (VALIDATION_0830 P0 #5).
    recovery     if a close was EXECUTED and this or the very next replan
                 reports p_evt[none] > p_none, the grasp is phantom: command
                 the task open aperture and forbid any upward z in the chunk,
                 so the policy re-descends instead of lifting nothing. Capped
                 at `max_retries` cycles per episode, then the episode ends
                 with reason `veto_retry_cap` (an uncapped open/re-descend loop
                 is the safety gap the review's completeness critic flagged).

    Both rules read the aperture through the TRAINING close rule
    (train/common.close_index): a rise of `close_rise` above the episode's
    RUNNING MINIMUM, not above the current sample. The rig's terminal phase
    ramps 0.31 -> 0.52 over several replans (per-replan rise 0.02-0.06), so the
    old per-replan rate test never fired on the failure this exists to catch.
    """
    p_close: float = 0.5              # theta_close on p_contact = 1 - p_evt[none]
    p_none: float = 0.9               # p_evt[none] above which a close is phantom
    max_retries: int = 3
    z_floor: float | None = None      # m; the SAFETY floor (tcp_z_min - margin)
    z_ref: float | None = None        # m; demo tcp_z_min the band is measured
                                      # from (falls back to z_floor)
    z_margin: float = 0.015           # "already in the demo close band" margin
    open_aperture: float = 0.0        # task open aperture (demo start mean)
    # Close detection reuses the TRAINING rule (train/common.close_index):
    # aperture past CLOSE_ABS_POS after rising CLOSE_ABS_RISE from the running
    # minimum of the measured aperture.
    close_pos: float = 0.45
    close_rise: float = 0.15
    # Tactile-grounded phantom / lost-object recovery (rig 09-04, evaluated
    # on all 17 closes of the session: 6/6 phantoms + 5/5 lost objects flagged,
    # 0/6 carried grasps): gripper measured closed AND the TCP has risen more
    # than `phantom_dz` above its height at the close AND the trailing
    # `phantom_window` max of the baseline-corrected pad force is below
    # `phantom_f` on BOTH pads, sustained `phantom_t` -> open + re-descend
    # through the same recovery arm as the p_none rule. A time-only gate has
    # no clean cell (force ramp after a real close spans 0.4-3.2 s); the lift
    # gate defers the question to the moment it becomes answerable.
    phantom_dz: float = 0.03
    phantom_f: float = 2.5
    phantom_t: float = 0.3
    phantom_window: float = 1.0


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
        # why the LOOP ended when the executor has no stop reason of its own
        # (replan cap / wall-clock budget); runtime.run_episode surfaces it
        self.stop_reason: str | None = None
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
    # how many replans after an EXECUTED close the phantom-grasp recovery may
    # still fire (the docstring's "the very next replan"). Before 2026-08-30
    # the latch was unbounded and reopened a real grasp any time later in the
    # episode (VALIDATION_0830 P0 #3).
    VETO_RECOVERY_REPLANS = 1

    def _note_executed_close(self, state: dict, n: int) -> None:
        """Latch the replan index of a close the EXECUTOR actually entered.

        Plan acceptance is not execution: a close living in the tail of a chunk
        playback never reached was never commanded, and a chunk the veto itself
        rewrote carries the held aperture rather than the proposal. The
        executor's `_grip_hist` is the deploy-side stand-in for the recorded
        STREAM_ACTIONS gripper channel, i.e. exactly the commands that went
        out."""
        v = self.veto
        fn = getattr(self.executor, "entered_grip_after", None)
        if v is None or fn is None:
            return
        try:
            steps = fn(state["grip_seen_t"])
        except Exception:               # never let telemetry kill an episode
            log.exception("entered_grip_after failed — the veto latch stays cold")
            return
        g_min = state["g_min"]
        for t_step, g in steps:
            state["grip_seen_t"] = max(state["grip_seen_t"], float(t_step))
            if g_min is None:
                continue
            # the TRANSITION into the close is the event, not the state: the
            # running minimum keeps every later step of a held grasp above the
            # rise threshold, and re-arming on those would make the latch
            # unbounded again by another route.
            closed = g > v.close_pos and (g - g_min) > v.close_rise
            if closed and not state["in_close"]:
                # arm the latch only for a close the veto PERMITTED: after
                # `close_masked` the executor holds the aperture, but on an
                # open-loop ramp the MEASURED aperture can rise anyway and
                # this read it back as an executed close, arming the
                # phantom-grasp recovery on a close the veto itself prevented
                # (revalidation 2026-08-31 §2 #3).
                if state.get("close_permitted"):
                    state["closed_idx"] = n
                    # a close allowed ONLY by the at_floor hatch carries no
                    # gate evidence, and on the 08-28 statistics (p_none>0.9 on
                    # 80% of replans) the recovery would reopen 5/18 real
                    # grasps: never fire it on a floor-only close.
                    state["closed_floor_only"] = state.get("allowed_floor_only",
                                                           False)
            state["in_close"] = closed

    def _apply_veto(self, plan, tcp_pose: np.ndarray, grip_now: float,
                    state: dict, n: int = 0) -> dict | None:
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

        # running minimum of the MEASURED aperture (train/common.close_index's
        # rule), and the executed-close latch it feeds.
        state["g_min"] = (grip_now if state["g_min"] is None
                          else min(state["g_min"], grip_now))
        self._note_executed_close(state, n)

        # ---- (b) phantom-grasp recovery ------------------------------------
        # checked FIRST: it reacts to the close executed one replan ago, and
        # only inside that window — outside it the latch is dropped, so a real
        # grasp is never reopened by a late high-p_none frame (a release, a
        # transport frame where the gel loses the object, an egg held lightly).
        idx = state["closed_idx"]
        if idx is not None and (n - idx) > self.VETO_RECOVERY_REPLANS:
            state["closed_idx"] = idx = None
        if idx is not None and p_none > v.p_none and state.get("closed_floor_only"):
            # the only door open on the 08-28 data is the floor hatch; a
            # recovery keyed on p_none would self-cancel those closes (the
            # voters split on whether that is protection or an anti-grasp —
            # unresolvable without a successful rig grasp, so the guard takes
            # the tail risk off the table either way)
            state["closed_idx"] = None
            rec["action"] = "recovery_skipped_floor_close"
            log.info("terminal veto: high p_none (%.2f) after a FLOOR-ONLY "
                     "close — recovery suppressed, latch cleared", p_none)
            return rec
        if idx is not None and p_none > v.p_none:
            # tactile veto over the model's opinion (verification 09-04): with
            # both pads loaded the object IS held — never open on p_none alone
            held = self._pad_loads(state, v.phantom_window)
            if len(held) >= 2 and all(f >= v.phantom_f for f in held.values()):
                state["closed_idx"] = None
                rec["action"] = "recovery_skipped_loaded"
                rec["pad_load"] = {k: round(f, 2) for k, f in held.items()}
                log.info("terminal veto: high p_none (%.2f) but both pads loaded "
                         "(%s) — recovery suppressed", p_none, rec["pad_load"])
                return rec
            state["closed_idx"] = None
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
            self._invalidate_cpk(plan)
            if hasattr(self.executor, "clear_grip_latch"):
                self.executor.clear_grip_latch()
            rec["action"] = "recovery_open"
            log.warning("terminal veto: phantom grasp (p_none=%.2f) — opening to "
                        "%.2f and forbidding lift (retry %d/%d)",
                        p_none, v.open_aperture, state["retries"], v.max_retries)
            return rec

        # ---- (b2) tactile-grounded phantom / lost-object recovery -----------
        # z at the moment the MEASURED aperture crossed close_pos upward
        if grip_now > v.close_pos:
            if state.get("close_z") is None:
                state["close_z"] = z_now
        else:
            state["close_z"] = None
            state["phantom_since"] = None
        loads = self._pad_loads(state, v.phantom_window)
        empty = (state.get("close_z") is not None
                 and z_now - state["close_z"] > v.phantom_dz
                 and len(loads) >= 2 and all(f < v.phantom_f for f in loads.values()))
        t_rep = time.perf_counter()
        if empty:
            if state.get("phantom_since") is None:
                state["phantom_since"] = t_rep
        else:
            state["phantom_since"] = None
        if empty and (t_rep - state["phantom_since"]) >= v.phantom_t:
            rise_mm = (z_now - state["close_z"]) * 1000.0
            state["close_z"] = None
            state["phantom_since"] = None
            state["closed_idx"] = None
            state["retries"] += 1
            rec["retries"] = state["retries"]
            rec["pad_load"] = {k: round(f, 2) for k, f in loads.items()}
            if state["retries"] > v.max_retries:
                rec["action"] = "retry_cap"
                return rec
            a[:, 6] = v.open_aperture
            cum = np.minimum(np.cumsum(a[:, 2]), 0.0)
            a[:, 2] = np.diff(np.concatenate([[0.0], cum]))
            self._invalidate_cpk(plan)
            if hasattr(self.executor, "clear_grip_latch"):
                self.executor.clear_grip_latch()
            rec["action"] = "recovery_tactile"
            log.warning("terminal veto: gripper closed and lifted %.0f mm with "
                        "NO pad load (%s) — opening to %.2f and re-descending "
                        "(retry %d/%d)", rise_mm, rec["pad_load"],
                        v.open_aperture, state["retries"], v.max_retries)
            return rec

        # ---- (a) close mask -------------------------------------------------
        g_max = float(np.max(a[:, 6]))
        g_min = state["g_min"] if state["g_min"] is not None else grip_now
        closing = g_max > v.close_pos and (g_max - g_min) > v.close_rise
        if not closing:
            rec["action"] = "none"
            return rec
        z_ref = v.z_ref if v.z_ref is not None else v.z_floor
        at_floor = (z_ref is not None and z_now <= z_ref + v.z_margin)
        if p_contact > v.p_close or at_floor:
            # the latch is armed later, by `_note_executed_close`, from the
            # gripper steps the executor actually ENTERED — a plan that is
            # accepted but never played is not the close the recovery reacts to.
            rec["action"] = "close_allowed"
            rec["at_floor"] = bool(at_floor)
            state["close_permitted"] = True
            state["allowed_floor_only"] = bool(at_floor and not (p_contact > v.p_close))
            return rec
        a[:, 6] = grip_now                       # hold the current aperture
        self._invalidate_cpk(plan)
        state["close_permitted"] = False
        rec["action"] = "close_masked"
        log.warning("terminal veto: close masked (p_contact=%.2f <= %.2f, "
                    "z=%.0f mm) — holding aperture %.2f",
                    p_contact, v.p_close, z_now * 1000, grip_now)
        return rec

    def _pad_loads(self, state: dict, window_s: float) -> dict[str, float]:
        """Trailing-`window_s` max of the baseline-corrected |fz| per pad from
        the tactile rings; {} when no session/rings (tests, student stubs).
        The baseline is the first replan's reading (pads untouched at the
        start pose) — the left pad idles 0.7-2.2 N above zero."""
        session = self.session or getattr(self.snapshots, "session", None)
        rings = getattr(session, "rings", None)
        if not rings:
            return {}
        rate = float(getattr(self.hw.tactile, "rate_hz", 8.0) or 8.0)
        k = max(1, int(round(window_s * rate)))
        out = {}
        base = state.setdefault("pad_base", {})
        for s_ in self.hw.tactile.sensors:
            ring = rings.get(f"tactile_{s_.name}")
            if ring is None:
                continue
            try:
                ts_t, tac = ring.latest(k)
                w = tac.get("wrench") if hasattr(tac, "get") else None
            except Exception:
                continue
            if w is None or not len(ts_t):
                continue
            fz = np.asarray(w, dtype=np.float64).reshape(len(ts_t), -1)[:, 2]
            if s_.name not in base:
                base[s_.name] = float(fz[-1])
            out[s_.name] = float(np.max(np.abs(fz - base[s_.name])))
        return out

    @staticmethod
    def _invalidate_cpk(plan) -> None:
        """Drop the contact package of a chunk the veto rewrote.

        `plan.cpk` is the model's IMAGINED contact for the chunk it proposed,
        and the next replan feeds it back as `prev_cpk` (policy.py:244). After
        a rewrite it describes motion the arm was never asked to make, so it
        must not condition the next chunk (Codex, 2026-08-30).

        With a policy server the package lives SERVER-side, addressed by
        `_cpk_token` — clearing only `cpk` (always None on the remote path)
        would leave the token alive and the invalidation a silent no-op
        (verification 09-01)."""
        plan.cpk = None
        if hasattr(plan, "_cpk_token"):
            plan._cpk_token = None

    def run(self, max_replans: int | None = None,
            max_episode_s: float | None = None,
            stop_check: Callable[[], bool] | None = None,
            success_check: Callable[[], bool] | None = None) -> None:
        """Replan until a cap, a stop or the wall-clock budget.

        `max_episode_s` is the budget the replan COUNT was standing in for:
        episode wall-time is `max_replans x latency`, so the same cap of 40 is
        a 35 s episode at the NFE-5 cadence (865 ms) and a 7 s one at `--nfe 1`
        (172 ms) — against demos that run 16-31 s (VALIDATION_0830 P0 #4)."""
        n = 0
        t_start = time.perf_counter()
        self.stop_reason = None
        prev_plan: Plan | None = None
        prev_tcp = prev_cmd = None
        strikes = 0
        # per-episode terminal-veto state: the replan index at which an
        # EXECUTED close was observed, how many open->re-descend cycles have
        # run, the running minimum of the measured aperture, and how far the
        # executed-gripper-step scan has read.
        veto_state = {"closed_idx": None, "retries": 0, "g_min": None,
                      "in_close": False, "grip_seen_t": float("-inf"),
                      "close_permitted": False, "allowed_floor_only": False,
                      "closed_floor_only": False}
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
            # the PROPOSAL, before the veto rewrites the chunk in place: on a
            # vetoed replan `actions` is the veto's arithmetic, and G0's
            # trace_in_spread / trace_head_dz score against it (P1 #12)
            pre_veto = (np.array(plan.actions, copy=True)
                        if self.veto is not None else None)
            tcp_row = [float(x) for x in np.asarray(tcp_pose).ravel()]
            veto_rec = self._apply_veto(plan, tcp_pose, grip_now, veto_state, n)
            if veto_rec is not None and veto_rec.get("action") == "retry_cap":
                self.trace.append({"t": snap.t, "latency_s": plan.latency_s,
                                   "gate": plan.gate, "p_evt": plan.p_evt.tolist(),
                                   "sigma": plan.sigma.tolist(), "accepted": False,
                                   "actions": plan.actions.tolist(), "diag": {},
                                   "tcp_pose": tcp_row,
                                   "terminal_veto": veto_rec})
                log.error("terminal veto: %d open/re-descend retries exhausted — "
                          "ending the episode", self.veto.max_retries)
                self.executor.request_stop("veto_retry_cap")
                break
            accepted = self.executor.submit(plan)
            row = {
                "t": snap.t, "latency_s": plan.latency_s, "gate": plan.gate,
                "p_evt": plan.p_evt.tolist(), "sigma": plan.sigma.tolist(),
                "accepted": accepted,
                "actions": plan.actions.tolist(),
                # measured TCP pose this chunk was planned from: without it
                # even "z at close" needed a clock-calibrated join against
                # arm_tcp_pose.zarr (P3 #32)
                "tcp_pose": tcp_row,
                # terminal-veto decision for this replan (None => veto off)
                "terminal_veto": veto_rec,
                # provenance: nfe/guidance/head_dz_mm/... per replan (audit
                # 2026-08-20 — the A/B condition lived only in the operator's
                # memory). Numeric LISTS are kept too: _select_seed's per-seed
                # head_dz_mm is the only record of what the K seeds proposed
                # and the quantity replay_rig's spread is validated against.
                "diag": {k: v for k, v in (getattr(plan, "diag", None) or {}).items()
                         if isinstance(v, (int, float, str, bool))
                         or (isinstance(v, (list, tuple))
                             and all(isinstance(x, (int, float)) for x in v))},
            }
            if pre_veto is not None and veto_rec is not None \
                    and veto_rec.get("action") in VETO_REWRITE_ACTIONS:
                row["actions_pre_veto"] = pre_veto.tolist()
            self.trace.append(row)
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
                # NOT silent any more: this used to break with stopped_reason
                # None, so a capped episode was indistinguishable from a
                # completed one in the ledger (P0 #4).
                self.stop_reason = "replan_cap"
                log.warning("replan cap reached (%d replans, %.1f s) — ending "
                            "the episode", n, time.perf_counter() - t_start)
                break
            if max_episode_s is not None and \
                    (time.perf_counter() - t_start) >= max_episode_s:
                self.stop_reason = "episode_time_cap"
                log.warning("episode budget reached (%.1f s of %.1f s, %d "
                            "replans) — ending the episode",
                            time.perf_counter() - t_start, max_episode_s, n)
                break
            if success_check is not None and success_check():
                # Grasp confirmed on tactile AND lifted clear (rig 2026-09-01:
                # demos END right after the lift; every extra second of "carry"
                # is out-of-distribution and twice ran the arm into the
                # full-extension singularity). Clean exit, gripper held.
                self.stop_reason = "lift_complete"
                log.info("LIFT COMPLETE — tactile-confirmed grasp held above "
                         "the lift height (%.1f s, %d replans). Episode ends "
                         "as a SUCCESS candidate; gripper stays closed.",
                         time.perf_counter() - t_start, n)
                break
            if stop_check is not None and stop_check():
                # Operator ended the episode. Same clean exit as the caps:
                # the replan loop breaks, the executor is stopped by the
                # runtime teardown, and the gripper is NOT touched — an
                # operator stop after a successful grasp must not drop the
                # object the way a let-go safety stop would.
                self.stop_reason = "operator_stop"
                log.info("operator stop (%.1f s, %d replans) — ending the "
                         "episode", time.perf_counter() - t_start, n)
                break
            if self._executor_stopped():
                break

        self._log_gate_calibration()
    def _log_gate_calibration(self) -> None:
        """One line per episode: the p_none distribution the veto thresholds
        were never fitted on (revalidation 2026-08-31 §2 #3 — Session 4 must
        produce this calibration whether or not the veto fires)."""
        try:
            pn = [float(r["p_evt"][0]) for r in self.trace if r.get("p_evt")]
            if not pn:
                return
            q = np.percentile(np.asarray(pn), [0, 25, 50, 75, 100])
            log.info("gate calibration: p_none over %d replans min/q25/med/q75/max "
                     "= %.2f/%.2f/%.2f/%.2f/%.2f  (veto p_close=%s p_none=%s)",
                     len(pn), *q,
                     getattr(self.veto, "p_close", None),
                     getattr(self.veto, "p_none", None))
        except Exception:
            log.exception("gate calibration failed (telemetry only)")

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
