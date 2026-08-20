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

    def build(self) -> ObsSnapshot:
        hw = self.hw
        rings = self.session.rings
        t_now = time.perf_counter()

        cam_name = "camera_scene"
        ts, cam = rings[cam_name].latest(1)
        assert len(ts), "no camera frames yet"
        rgb = cam["color"][0]

        L = hw.wrist_ft.window_len
        _, arm = rings["arm"].latest(max(L, 2))
        ft = arm["ft"]
        if ft.shape[0] < L:
            ft = np.concatenate([np.repeat(ft[:1], L - ft.shape[0], axis=0), ft])
        wrist_window = ft[-L:].astype(np.float32)

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
                 snapshots: SnapshotBuilder, executor, *, trace: list | None = None):
        self.hw = hw
        self.policy = policy
        self.snapshots = snapshots
        self.executor = executor
        self.trace = trace if trace is not None else []
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
            })
            from phantom.config.model import EVENTS
            k_evt = int(np.argmax(plan.p_evt))
            log.info("replan %d: latency=%.2fs gate=%.2f sigma_max=%.2f "
                     "p_evt=%s:%.2f accepted=%s",
                     n, plan.latency_s, plan.gate, float(np.max(plan.sigma)),
                     EVENTS[k_evt], float(plan.p_evt[k_evt]), accepted)
            prev_plan = plan
            n += 1
            if max_replans is not None and n >= max_replans:
                break
            if self.executor.stopped_reason is not None:
                log.warning("executor stopped (%s); ending episode",
                            self.executor.stopped_reason)
                break

    def stop(self) -> None:
        self._stop.set()
