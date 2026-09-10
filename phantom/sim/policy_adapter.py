"""Deterministic bridge from simulated sensors to the existing PHANTOM policy.

All times are simulation seconds. No driver, sensor worker, or hardware launch
path is imported or started. A supplied ``PhantomPolicy`` or ``RemotePolicy``
owns inference and the existing image resize/normalization pipeline.

Loop contract::

    adapter.observe(t, rgb=rgb, q=q, qd=qd, tcp_pose=tcp, tcp_speed=twist,
                    gripper_state=[closure, robotiq_obj], wrist_ft=wrench)
    if adapter.ready_for_replan(t):
        adapter.replan(latency_s=recorded_latency)
    command = adapter.step(t)
    if command is not None:
        # Solve IK seeded from previous joints; apply bounded joint drives.
        adapter.report_execution(t, accepted=ik_ok, tcp_pose=accepted_target,
                                 gripper_command=sent_closure)

TCP is in the UR base frame, metres + rotation vector in radians; it includes
the configured tool offset. Joint order is shoulder_pan, shoulder_lift, elbow,
wrist_1, wrist_2, wrist_3. Closure is 0=open, 1=closed (not jaw gap in metres).
``gripper_state[1]`` is measured Robotiq OBJ (0 moving, 1 opening contact,
2 closing contact, 3 requested position), never the commanded closure.

This reproduces the deploy executor's additive delta/rotvec action semantics,
governor, plan rebasing/blending, safety checks, rate limits, and aperture latch.
The optional terminal-veto/planner watchdog heuristics are not enabled here;
``plan_filter`` can apply an experiment's explicit planner transform. Physical
IK, joint limits, motor/contact dynamics and tactile rendering belong to the
simulator. Supplying synthetic tactile signals does not validate their fidelity.
The simulator records accepted gripper targets directly; the real gripper's
100 Hz mailbox, command deadband and Robotiq firmware remain approximations.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field

import numpy as np

from phantom.data import derived as dv
from phantom.deploy.governor import SpeedGovernor
from phantom.deploy.safety import (
    SafetyAction,
    SafetyMonitor,
    arm_stale_s,
    camera_stale_s,
    is_letgo_reason,
)

log = logging.getLogger(__name__)


@dataclass
class SimulationObservation:
    """Numpy-only structural equivalent of inference.policy.ObsSnapshot.

    Both local and remote policy implementations consume these attributes;
    keeping this type here also permits import in an Isaac Python environment
    without importing the training stack merely to construct observations.
    """

    t: float
    rgb: np.ndarray
    wrist_window: np.ndarray
    ur_state: np.ndarray
    gel: np.ndarray | None = None
    fields: np.ndarray | None = None
    contact_state: np.ndarray | None = None
    reactive: float = 0.0
    prev_chunk: np.ndarray | None = None
    sensor_times: dict = field(default_factory=dict)


@dataclass
class SimulationCommand:
    t: float
    tcp_pose: np.ndarray
    gripper: float
    dt: float
    stopped: bool = False
    reason: str | None = None
    diagnostics: dict = field(default_factory=dict)


class _History:
    """Small in-process, simulation-clock equivalent of the ring read API."""

    def __init__(self, capacity: int):
        self.rows: deque = deque(maxlen=capacity)

    def push(self, t: float, **fields) -> None:
        if not np.isfinite(t):
            raise ValueError("sensor timestamp must be finite")
        if self.rows and t < self.rows[-1][0]:
            raise ValueError("sensor timestamps must be monotonic")
        if self.rows and fields.keys() != self.rows[-1][1].keys():
            raise ValueError(
                "sensor field names must remain constant within an episode"
            )
        row = (float(t), {k: np.array(v, copy=True) for k, v in fields.items()})
        if self.rows and t == self.rows[-1][0]:
            self.rows[-1] = row
        else:
            self.rows.append(row)

    def latest(self, n: int = 1):
        rows = list(self.rows)[-n:]
        if not rows:
            return np.zeros(0), {}
        return (
            np.asarray([r[0] for r in rows]),
            {k: np.stack([r[1][k] for r in rows]) for k in rows[0][1]},
        )

    def latest_ts(self):
        return self.rows[-1][0] if self.rows else None

    def through(self, t):
        """Causal view for inference transport delay; safety keeps live rings."""
        view = _History(self.rows.maxlen)
        view.rows.extend(row for row in self.rows if row[0] <= t + 1e-9)
        return view


def _vector(value, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (n,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite ({n},) vector")
    return arr


class PolicyTimingError(ValueError):
    """Invalid timing instrumentation/input; never a policy task outcome."""

    infrastructure_invalid = True


class SimulationPolicyAdapter:
    """Single-threaded deployment semantics driven explicitly by sim time.

    Default ``native`` delivery uses the policy's reported inference latency.
    A nonnegative ``latency_s`` override retains the existing idealized timing
    experiment. Opt-in ``rpc_wall`` instead measures the complete synchronous
    policy call, preserving the native action grid and latency for conditioning.
    Its later delivery skips expired samples through the usual submit path.
    Rendering/physics continue on sim time; no wall-clock sleeps are used.

    ``step`` returns None while awaiting the first accepted plan: retain the
    existing drive targets without reporting execution. Returned commands
    require feedback before the next tick. Commands rejected by IK must never
    be presented as measured motion or executed-past actions.
    Teacher mode requires complete tactile data for every configured pad.
    """

    def __init__(
        self,
        hw,
        policy,
        *,
        mode: str = "student",
        parity_fixes: bool = True,
        max_play_steps: int | None = 10,
        grip_play_steps: int | None = None,
        open_aperture: float = 0.0,
        ik_reject_limit: int = 25,
        plan_filter: Callable | None = None,
        observation_callback: Callable | None = None,
        delivered_plan_callback: Callable | None = None,
        release_config=None,
        boundary_config=None,
        controller_profile: str | None = None,
        policy_delivery_clock: str = "native",
        replan_clock: Callable[[], float] | None = None,
    ):
        if mode not in (
            "teacher",
            "student",
            "vision_only",
            "no_distill",
            "drop_tactile",
        ):
            raise ValueError(f"unknown policy mode {mode!r}")
        if hw.arm.dof != 6 or hw.control.action_dim != 7:
            raise ValueError(
                "this adapter requires a six-joint UR and seven-channel actions"
            )
        if max_play_steps is not None and max_play_steps < 1:
            raise ValueError("max_play_steps must be positive or None")
        if grip_play_steps is not None and grip_play_steps < 1:
            raise ValueError("grip_play_steps must be positive or None")
        if not 0 <= open_aperture <= hw.gripper.max_close_cmd:
            raise ValueError("open_aperture is outside the configured closure limits")
        if ik_reject_limit < 1:
            raise ValueError("ik_reject_limit must be positive")
        if policy_delivery_clock not in ("native", "rpc_wall"):
            raise ValueError("policy_delivery_clock must be native or rpc_wall")
        if replan_clock is not None and not callable(replan_clock):
            raise TypeError("replan_clock must be callable")
        self.hw, self.policy, self.mode = hw, policy, mode
        self.wrench_baseline_rows = int(
            getattr(policy, "wrench_baseline_rows", 0) or 0
        )
        self.parity_fixes = bool(parity_fixes)
        self.max_play_steps = max_play_steps
        self.grip_play_steps = grip_play_steps
        self.open_aperture = float(open_aperture)
        self.ik_reject_limit = int(ik_reject_limit)
        self.plan_filter = plan_filter
        self.observation_callback = observation_callback
        self.delivered_plan_callback = delivered_plan_callback
        self.policy_delivery_clock = policy_delivery_clock
        self._replan_clock = time.perf_counter if replan_clock is None else replan_clock
        from phantom.deploy.release_controller import make_release_controller

        self.release_controller = make_release_controller(release_config, hw)
        from phantom.deploy.boundary_projection import boundary_config as parse_boundary_config
        self.boundary_config = parse_boundary_config(boundary_config)
        from phantom.deploy.minimal_v5 import validate_profile

        self.controller_profile = controller_profile
        self.controller_profile_metadata = validate_profile(
            controller_profile,
            release_config=self.release_controller.config if self.release_controller else None,
            veto=getattr(plan_filter, "veto", None), mode=mode, hw=hw,
        )
        if controller_profile != getattr(plan_filter, "controller_profile", None):
            raise ValueError("adapter and terminal veto controller profiles must agree")
        if self.controller_profile_metadata is not None:
            self.controller_profile_metadata = {
                **self.controller_profile_metadata,
                "sensor_inputs": "Simulator observations with native checkpoint preprocessing; tactile and wrist transfer remain unvalidated",
            }
        self.governor = SpeedGovernor(hw.safety.governor)
        self.reset(reset_policy=False)

    def reset(self, *, reset_policy: bool = True, seed: int | None = None) -> None:
        """Reset before sensor warm-up; never invoke a hardware reset."""
        if reset_policy:
            if hasattr(self.policy, "remote_reset"):
                self.policy.remote_reset(seed)
            else:
                if seed is not None:
                    import torch

                    if not hasattr(self.policy, "rf"):
                        raise ValueError("policy does not expose a seedable generator")
                    self.policy.rf._gen = torch.Generator().manual_seed(int(seed))
                self.policy.reset_episode()
        span = max(
            self.hw.wrist_ft.window_s,
            (self.hw.control.chunk_horizon + 1) / self.hw.control.action_rate_hz,
        )
        self._history_capacity = max(
            16, int(2 * span * self.hw.arm.rtde_receive_hz) + 8
        )
        self.rings = {
            "arm": _History(self._history_capacity),
            "gripper": _History(self._history_capacity),
            "camera_scene": _History(64),
        }
        self.safety = SafetyMonitor(self.hw, self.rings, boundary_config=self.boundary_config)
        self._plan = self._prev_plan = self._pending = None
        self._play_time = self._prev_play_time = self._swap_t = 0.0
        self._last_tick = self._last_observe_t = None
        self._last_replan_t = None
        self._last_cmd = self._hold_pose = self._awaiting_feedback = None
        self._last_grip = self.open_aperture
        self._grip_ack = None
        self._grip_latch = None
        if self.release_controller is not None:
            self.release_controller.reset()
        self._grip_hist: deque = deque(maxlen=self._history_capacity)
        self._prev_fields = None
        self.reset_baseline()
        self.stopped_reason = None
        self.completed_reason = None
        self.completed_at_s = None
        self._finish_pose = self._finish_grip = None
        self.ik_rejects = self.ik_rejects_total = 0
        if self.plan_filter is not None and hasattr(self.plan_filter, "reset"):
            self.plan_filter.reset()

    def reset_baseline(self) -> None:
        """Capture each pad's zero offset again at the next teacher snapshot."""
        self.wrench_base: dict[str, np.ndarray] = {}

    def _baseline_for(self, name: str, ring) -> np.ndarray:
        """Match native SnapshotBuilder's first-build, latest-N median.

        The caller must hold the pads unloaded during startup. Like native
        deployment, an early first snapshot uses the available rows and warns;
        it neither waits for N rows nor updates the baseline during contact.
        Only model contact-state wrench is corrected; safety keeps raw sensors.
        """
        rows = self.wrench_baseline_rows
        if rows <= 0:
            return np.zeros(6, np.float32)
        if name not in self.wrench_base:
            try:
                _, tac = ring.latest(rows)
                wrench = np.asarray(tac["wrench"], dtype=np.float32).reshape(-1, 6)
                if len(wrench) < rows:
                    log.warning(
                        "tactile %s: wrench baseline from %d rows (< %d): ring not warm yet",
                        name, len(wrench), rows,
                    )
                self.wrench_base[name] = np.median(wrench, axis=0).astype(np.float32)
            except Exception:
                # Native fallback preserves the raw pad wrench, not zero model
                # observations. Normal snapshots validate the ring beforehand.
                log.exception(
                    "tactile %s: could not read the wrench baseline — using ZERO "
                    "(the model input keeps this pad's offset)", name,
                )
                self.wrench_base[name] = np.zeros(6, np.float32)
            log.info(
                "tactile %s wrench baseline (%d rows) %s",
                name, rows, np.round(self.wrench_base[name], 2).tolist(),
            )
        return self.wrench_base[name]

    def observe(
        self,
        t: float,
        *,
        rgb,
        q,
        qd,
        tcp_pose,
        tcp_speed,
        gripper_state,
        wrist_ft,
        camera_t: float | None = None,
        protective_stop: bool = False,
        tactile: dict | None = None,
    ) -> None:
        """Ingest measured simulation sensors (RGB/RGBA uint8, no BGR conversion).

        RGBA alpha is removed, preserving RGB values and camera resolution.
        ``rgb=None`` retains the preceding frame and its original timestamp.
        If a frame is repeated, pass its capture time as ``camera_t``.
        Tactile maps pad names to {t, fields_ds, keyframe, infer_img, wrench,
        area}; t defaults to this observation's t. These are the same physical
        units/channels as the recorded rig streams. Student modes can provide
        pad wrench/fields for safety without feeding gels into the policy.
        """
        t = float(t)
        if not np.isfinite(t) or (
            self._last_observe_t is not None and t < self._last_observe_t
        ):
            raise ValueError("observation time must be finite and monotonic")
        q = _vector(q, 6, "q")
        qd = _vector(qd, 6, "qd")
        pose = _vector(tcp_pose, 6, "tcp_pose")
        speed = _vector(tcp_speed, 6, "tcp_speed")
        ft = _vector(wrist_ft, 6, "wrist_ft")
        grip = _vector(gripper_state, 2, "gripper_state")
        if not 0 <= grip[0] <= 1 or grip[1] not in (0, 1, 2, 3):
            raise ValueError("gripper_state must be [measured closure 0..1, OBJ 0..3]")
        self.rings["arm"].push(
            t,
            q=q,
            qd=qd,
            tcp_pose=pose,
            tcp_speed=speed,
            ft=ft,
            protective_stop=protective_stop,
        )
        self.rings["gripper"].push(t, state=grip)
        if rgb is not None:
            frame = np.asarray(rgb)
            if (
                frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[-1] not in (3, 4)
            ):
                raise ValueError("camera frame must be HxWx3 RGB or HxWx4 RGBA uint8")
            capture_t = t if camera_t is None else float(camera_t)
            if capture_t > t:
                raise ValueError(
                    "camera timestamp cannot be in the observation's future"
                )
            self.rings["camera_scene"].push(capture_t, color=frame[..., :3])
        known_pads = {s.name for s in self.hw.tactile.sensors}
        for name, data in (tactile or {}).items():
            if name not in known_pads:
                raise ValueError(f"unknown tactile pad {name!r}")
            fields = dict(data)
            tt = float(fields.pop("t", t))
            if tt > t:
                raise ValueError(
                    "tactile timestamp cannot be in the observation's future"
                )
            if "wrench" in fields:
                fields["wrench"] = _vector(
                    fields["wrench"], 6, f"tactile {name} wrench"
                )
            for k, v in fields.items():
                if not np.isfinite(v).all():
                    raise ValueError(f"tactile {name} {k} contains nonfinite values")
            for key, expected in (
                ("fields_ds", (*self.hw.recording.field_ds.hw, 8)),
                ("keyframe", (*self.hw.recording.keyframe_ds.hw, 8)),
            ):
                if key in fields and np.shape(fields[key]) != expected:
                    raise ValueError(f"tactile {name} {key} must have shape {expected}")
            if "area" in fields and (
                np.ndim(fields["area"]) != 0 or float(fields["area"]) < 0
            ):
                raise ValueError(f"tactile {name} area must be a nonnegative scalar")
            if "infer_img" in fields:
                gel = np.asarray(fields["infer_img"])
                ih, iw, _ = self.hw.tactile.infer_img.hwc
                if gel.dtype != np.uint8 or gel.shape not in ((ih, iw), (ih, iw, 3)):
                    raise ValueError(
                        f"tactile {name} infer_img must be grayscale or RGB uint8 at {(ih, iw)}"
                    )
            self.rings.setdefault(
                f"tactile_{name}", _History(max(32, self.wrench_baseline_rows))
            ).push(tt, **fields)
        if self._last_cmd is None:
            self._last_cmd = pose.copy()
            self._last_grip = float(grip[0])
        self._last_observe_t = t

    def gripper_cmd_at(self, times):
        times = np.asarray(times, dtype=np.float64)
        if not self._grip_hist:
            return None
        hist = np.asarray(self._grip_hist)
        idx = np.searchsorted(hist[:, 0], times, side="right") - 1
        return np.where(idx >= 0, hist[np.clip(idx, 0, len(hist) - 1), 1], np.nan)

    def snapshot(
        self, t: float | None = None, *, observation_delay_s: float = 0.0
    ) -> SimulationObservation:
        """Deploy-compatible snapshot from measured history on the sim clock."""
        if self._last_observe_t is None:
            raise RuntimeError(
                "observe measured sensors before constructing a snapshot"
            )
        now = self._last_observe_t if t is None else float(t)
        if not np.isfinite(now) or now < self._last_observe_t:
            raise ValueError("snapshot time cannot precede the newest observation")
        if not np.isfinite(observation_delay_s) or observation_delay_s < 0:
            raise ValueError("observation_delay_s must be finite and nonnegative")
        capture_cutoff = now - observation_delay_s
        rings = (
            {name: ring.through(capture_cutoff) for name, ring in self.rings.items()}
            if observation_delay_s
            else self.rings
        )
        for name, limit in (
            ("arm", arm_stale_s(self.hw)),
            ("camera_scene", camera_stale_s(self.hw)),
        ):
            stamp = rings[name].latest_ts()
            if stamp is None or now - stamp >= limit:
                raise RuntimeError(f"{name}_stale: missing or stale simulated sensor")
        ts, arm = rings["arm"].latest(self._history_capacity)
        gr = rings["gripper"].latest(1)[1]["state"][0]
        rgb = rings["camera_scene"].latest(1)[1]["color"][0]
        grid = np.linspace(
            ts[-1] - self.hw.wrist_ft.window_s, ts[-1], self.hw.wrist_ft.window_len
        )
        wrist = np.stack([np.interp(grid, ts, arm["ft"][:, k]) for k in range(6)], -1)
        if self.mode in ("vision_only", "drop_tactile"):
            wrist[:] = 0
        ur = np.concatenate(
            [arm[k][-1] for k in ("q", "qd", "tcp_pose", "tcp_speed")] + [gr]
        ).astype(np.float32)
        snap = SimulationObservation(now, rgb, wrist.astype(np.float32), ur)
        snap.sensor_times = {name: ring.latest_ts() for name, ring in rings.items()}
        if self.parity_fixes and len(ts) >= 2:
            H, rate = self.hw.control.chunk_horizon, self.hw.control.action_rate_hz
            past_grid = now - (H - np.arange(-1, H)) / rate
            ids = np.abs(ts[None, :] - past_grid[:, None]).argmin(axis=1)
            poses = arm["tcp_pose"][ids]
            deltas = np.stack([dv.pose_delta(poses[k], poses[k + 1]) for k in range(H)])
            gs = self.gripper_cmd_at(past_grid[1:])
            gs = (
                np.full(H, gr[0])
                if gs is None
                else np.where(np.isfinite(gs), gs, gr[0])
            )
            snap.prev_chunk = np.concatenate([deltas, gs[:, None]], 1).astype(
                np.float32
            )
        if self.mode == "teacher":
            fields, gels, contact, pairs = [], [], [], []
            for sensor in self.hw.tactile.sensors:
                ring = rings.get(f"tactile_{sensor.name}")
                tt, tac = ring.latest(2) if ring is not None else ([], {})
                required = {"fields_ds", "keyframe", "infer_img", "wrench", "area"}
                if not len(tt) or not required.issubset(tac):
                    raise RuntimeError(
                        f"teacher requires explicit tactile data for {sensor.name}: {sorted(required)}"
                    )
                if now - tt[-1] > 3 / min(
                    self.hw.recording.field_ds_rate_hz, self.hw.cameras.scene.fps
                ):
                    raise RuntimeError(f"tactile_{sensor.name}_stale")
                cur, prev = tac["fields_ds"][-1], tac["fields_ds"][0]
                dt = (
                    max(tt[-1] - tt[0], 1e-6)
                    if self.parity_fixes and len(tt) >= 2
                    else 1 / self.hw.recording.field_ds_rate_hz
                )
                derived = dv.derive_timestep(cur, prev, dt, self.hw)
                baseline = self._baseline_for(sensor.name, ring)
                contact.append(
                    np.concatenate(
                        [
                            np.asarray(tac["wrench"][-1], dtype=np.float32) - baseline,
                            [float(tac["area"][-1])],
                            np.nan_to_num(derived["cop"], nan=0.0),
                            [derived["slip"], derived["mask_frac"]],
                        ]
                    )
                )
                fields.append(tac["keyframe"][-1])
                gels.append(tac["infer_img"][-1])
                pairs.append((cur, prev))
            snap.fields = np.stack(fields).astype(np.float32)
            snap.gel = np.stack(gels)
            snap.contact_state = np.stack(contact).astype(np.float32)
            cur = np.stack([p[0] for p in pairs])
            prev = (
                np.stack([p[1] for p in pairs])
                if self.parity_fixes
                else self._prev_fields
            )
            if prev is not None:
                snap.reactive = dv.reactive_score(cur, prev)
            self._prev_fields = cur.copy()
        return snap

    def ready_for_replan(self, t: float) -> bool:
        return (
            self.stopped_reason is None
            and self.completed_reason is None
            and self._pending is None
            and self._last_observe_t is not None
            and (
                self._last_replan_t is None
                or t - self._last_replan_t >= 1 / self.hw.control.model_tick_hz
            )
        )

    def replan(
        self,
        *,
        t: float | None = None,
        latency_s: float | None = None,
        inference_delay_add_s: float = 0.0,
        observation_delay_s: float = 0.0,
    ):
        """Run the existing policy; activate the result after inference latency."""
        if self.stopped_reason or self.completed_reason:
            raise RuntimeError(
                f"episode ended: {self.stopped_reason or self.completed_reason}"
            )
        if self._pending is not None:
            raise RuntimeError("a policy plan is already pending activation")
        if self.policy_delivery_clock == "rpc_wall" and latency_s is not None:
            raise PolicyTimingError(
                "rpc_wall cannot be combined with a latency override"
            )
        now = self._last_observe_t if t is None else float(t)
        if not np.isfinite(inference_delay_add_s) or inference_delay_add_s < 0:
            raise ValueError("inference_delay_add_s must be finite and nonnegative")
        snap = self.snapshot(t, observation_delay_s=observation_delay_s)
        if self.observation_callback is not None:
            self.observation_callback(snap)
        rpc_wall = None
        if self.policy_delivery_clock == "rpc_wall":
            started = self._read_replan_clock()
        plan = self.policy.replan(
            snap, self._plan, snap.ur_state[12:18].astype(np.float64)
        )
        if self.policy_delivery_clock == "rpc_wall":
            rpc_wall = self._read_replan_clock() - started
            try:
                native_latency = float(plan.latency_s)
            except (ValueError, TypeError, OverflowError) as error:
                raise PolicyTimingError(
                    "native latency is not a finite duration"
                ) from error
            if (
                not np.isfinite(rpc_wall)
                or rpc_wall < 0
                or not np.isfinite(native_latency)
                or native_latency < 0
                or rpc_wall + 1e-9 < native_latency
            ):
                raise PolicyTimingError(
                    "rpc_wall requires finite monotonic timing and complete client wall "
                    "duration at least the reported native latency"
                )
        delay = float(plan.latency_s if latency_s is None else latency_s)
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("latency_s must be finite and nonnegative")
        delivery_delay = (
            delay if rpc_wall is None else rpc_wall
        ) + inference_delay_add_s
        if rpc_wall is not None and not np.isfinite(delivery_delay):
            raise PolicyTimingError("rpc_wall delivery duration overflowed")
        plan = copy(
            plan
        )  # preserve remote CPK token; never mutate the policy's proposal
        plan.actions = np.array(plan.actions, copy=True)
        plan.t0_pose = np.array(plan.t0_pose, copy=True)
        if self.policy_delivery_clock == "rpc_wall":
            # Preserve native temporal conditioning. Only delivery is later;
            # submit() skips expired samples and rebases the surviving head.
            plan.action_times = np.array(plan.action_times, copy=True)
        else:
            plan.action_times = (
                now
                + delay
                + np.arange(len(plan.actions)) / self.hw.control.action_rate_hz
            )
        plan.diag = {
            **plan.diag,
            "sim_native_inference_latency_s": float(plan.latency_s),
            "sim_effective_inference_latency_s": delivery_delay,
            "sim_inference_delay_add_s": inference_delay_add_s,
            "sim_observation_delay_s": observation_delay_s,
            "sim_observation_capture_t": now - observation_delay_s,
            "sim_inference_request_t": now,
            "sim_observation_sensor_timestamps": snap.sensor_times,
        }
        if rpc_wall is not None:
            plan.diag.update(
                sim_policy_delivery_clock="rpc_wall",
                sim_policy_replan_wall_time_s=rpc_wall,
                sim_policy_delivery_base_s=rpc_wall,
                sim_effective_delivery_delay_s=delivery_delay,
                sim_rpc_minus_native_latency_s=rpc_wall - native_latency,
            )
        plan.latency_s = delay
        plan.t_created = snap.t
        self._validate_plan(plan)
        # Transport delay occurs after native policy selection. Keep the native
        # action grid: the deployment submit path skips expired head samples
        # and rebases at delivery, exactly as a late remote response does.
        self._pending = (now + delivery_delay, plan, snap)
        self._last_replan_t = now
        return plan

    def _read_replan_clock(self):
        try:
            value = float(self._replan_clock())
        except (ValueError, TypeError, OverflowError) as error:
            raise PolicyTimingError(
                "replan clock must return finite seconds"
            ) from error
        if not np.isfinite(value):
            raise PolicyTimingError("replan clock must return finite seconds")
        return value

    def _validate_plan(self, plan):
        actions = np.asarray(plan.actions)
        if (
            actions.ndim != 2
            or actions.shape[1] != 7
            or not len(actions)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                "plan actions must be finite nonempty Hx7 denormalized deltas"
            )
        _vector(plan.t0_pose, 6, "plan.t0_pose")
        times = _vector(plan.action_times, len(actions), "plan.action_times")
        if len(times) > 1 and not np.allclose(
            np.diff(times), 1 / self.hw.control.action_rate_hz
        ):
            raise ValueError(
                "plan action_times must use the configured action-rate grid"
            )
        if not np.isfinite(plan.sigma).all():
            raise ValueError("plan sigma must be finite")

    def _grip_play_limit(self) -> int | None:
        """Match native: gripper cannot outlive its own or the pose cap."""
        limits = [int(limit) for limit in (self.max_play_steps, self.grip_play_steps) if limit]
        return min(limits) if limits else None

    def _pose_at(self, plan, play_time):
        H = len(plan.actions)
        cap = H if self.max_play_steps is None else min(H, self.max_play_steps)
        u = float(np.clip(play_time * self.hw.control.action_rate_hz, 0, cap - 1e-6))
        k = int(u)
        cum = np.cumsum(plan.actions[:, :6], axis=0)
        prev = cum[k - 1] if k else np.zeros(6)
        grip_limit = self._grip_play_limit()
        kg = k if grip_limit is None else min(k, grip_limit - 1)
        return plan.t0_pose + prev + (u - k) * (cum[k] - prev), float(plan.actions[kg, 6])

    def submit(self, plan, t: float) -> bool:
        """Accept a ready plan, rebased at the last confirmed command."""
        self._validate_plan(plan)
        if not np.isfinite(t) or (
            self._last_observe_t is not None and t < self._last_observe_t
        ):
            raise ValueError("plan submission time must cover the newest observation")
        if (
            self.stopped_reason
            or self.completed_reason
            or self._last_cmd is None
            or plan.action_times[0] > t + 1e-9
            or plan.action_times[-1] <= t + self.hw.control.replan_min_lead_s
        ):
            return False
        new = copy(plan)
        new.actions = np.array(plan.actions, copy=True)
        elapsed = max(0, t - float(plan.action_times[0]))
        offset, _ = self._pose_at(new, elapsed)
        new.t0_pose = self._last_cmd.copy() - (offset - plan.t0_pose)
        self._prev_plan, self._prev_play_time = self._plan, self._play_time
        self._plan, self._play_time, self._swap_t = new, elapsed, float(t)
        return True

    def clear_grip_latch(self):
        self._grip_latch = None
        if self.release_controller is not None:
            self.release_controller.reset()

    def placement_release_opening_mask(self, proposed_grip):
        """Permit original opening samples through a close mask inside the gate.

        Called at plan delivery, using current measured TCP feedback. This
        grants no release by itself: the executor must play a sustained opening
        while still inside the volume, after the native latch was armed.
        """
        values = np.asarray(proposed_grip)
        if (
            self.release_controller is None
            or self.stopped_reason
            or self.completed_reason
        ):
            return np.zeros(values.shape, dtype=bool)
        tcp = self.rings["arm"].latest(1)[1]["tcp_pose"][0]
        return (
            np.isfinite(values)
            & (values <= self.release_controller.config.open_command_max)
            & self.release_controller.window_active(tcp)
        )

    def _original_policy_grip(self):
        from phantom.deploy.release_controller import original_policy_grip

        return original_policy_grip(
            self._plan,
            self._play_time,
            self.hw.control.action_rate_hz,
            self._grip_play_limit(),
        )

    def entered_grip_after(self, t):
        """Actually issued gripper transitions, strictly after the given time."""
        return [(stamp, value) for stamp, value in self._grip_hist if stamp > t]

    def request_stop(self, reason: str):
        if self.stopped_reason is None:
            self.stopped_reason = str(reason)
        self._pending = None

    def step(self, t: float) -> SimulationCommand | None:
        """Compute a safe command, or None while no plan has been accepted.

        A missing plan must not turn measured closure into a new drive target.
        Native ChunkExecutor leaves the previous hardware command active while
        waiting. Simulator safety checks still run, and explicit safety stops
        retain their existing hold/release behavior before the first plan.
        """
        t = float(t)
        if self._last_cmd is None:
            raise RuntimeError("observe measured sensors before stepping")
        if (
            not np.isfinite(t)
            or t < self._last_observe_t
            or (self._last_tick is not None and t <= self._last_tick)
        ):
            raise ValueError("executor time must increase and cover newest observation")
        if self._awaiting_feedback is not None:
            raise RuntimeError("report_execution is required before the next step")
        period = 1 / self.hw.control.executor_rate_hz
        dt = period if self._last_tick is None else t - self._last_tick
        self._last_tick = t
        activated = False
        if self._pending is not None and t + 1e-9 >= self._pending[0]:
            _ready, plan, captured_snapshot = self._pending
            self._pending = None
            if self.plan_filter is not None:
                plan = self.plan_filter(plan, captured_snapshot, self)
                self._validate_plan(plan)
            if not self.stopped_reason:
                activated = self.submit(plan, t)
            if self.delivered_plan_callback is not None:
                self.delivered_plan_callback(plan, captured_snapshot, t, activated)
        target, grip = self._last_cmd.copy(), self._last_grip
        stale = False
        if (
            self._plan is not None
            and not self.stopped_reason
            and not self.completed_reason
        ):
            stale = (
                t - self._swap_t
                > len(self._plan.actions) / self.hw.control.action_rate_hz
                + self.hw.safety.stale_plan_timeout_s
            )
            if stale:
                if self._hold_pose is None:
                    self._hold_pose = (
                        self.rings["arm"].latest(1)[1]["tcp_pose"][0].copy()
                    )
                target = self._hold_pose.copy()
            else:
                self._hold_pose = None
                # ChunkExecutor advances every servo tick, including the first
                # tick after submit has rebased at the elapsed action index.
                self._play_time += dt * self.governor.scale_profile(
                    np.asarray(self._plan.sigma)
                )
                target, grip = self._pose_at(self._plan, self._play_time)
                since = t - self._swap_t
                blend = self.hw.control.chunk_blend_s
                if self._prev_plan is not None and since < blend:
                    self._prev_play_time += dt * self.governor.scale_profile(
                        np.asarray(self._prev_plan.sigma)
                    )
                    prev_target, _ = self._pose_at(
                        self._prev_plan, self._prev_play_time
                    )
                    beta = since / blend
                    target = (1 - beta) * prev_target + beta * target
                else:
                    self._prev_plan = None
        if self.completed_reason and not self.stopped_reason:
            target, grip = self._finish_pose.copy(), self._finish_grip
        verdict = self.safety.check(t, target)
        kinds = [e.kind for e in verdict.events]
        # A STOP replaces target with the last accepted command below. Keep
        # the checked proposal so a rejected next target cannot be mistaken
        # for measured motion outside the workspace.
        safety_target = (
            self.safety.target_diagnostics(t, target) if kinds else None
        )
        if self.rings["camera_scene"].latest_ts() is None:
            kinds.append("camera_scene_stale")
            self.request_stop("camera_scene_stale")
        if verdict.action in (SafetyAction.STOP_EPISODE, SafetyAction.PROTECTIVE_STOP):
            reason = (
                "protective_stop"
                if verdict.action == SafetyAction.PROTECTIVE_STOP
                else "lift_complete"
                if set(kinds) == {"lift_complete"}
                else "safety_stop"
            )
            self.request_stop(reason)
        if self._plan is None and not self.stopped_reason:
            # In particular, no feedback token, gripper-history entry or latch
            # transition is created during sensor warmup / first inference.
            return None
        if self.stopped_reason:
            target = self._last_cmd.copy()
            if any(is_letgo_reason(k) for k in kinds + [self.stopped_reason]):
                self.clear_grip_latch()
                grip = self.open_aperture
            else:
                # report_execution retains the accepted drive command here.
                # Replacing it with measured closure would unload a finite-P
                # gripper at a boundary halt; the arm remains stopped.
                grip = self._last_grip
            if self.release_controller is not None:
                self.release_controller.stop()
        else:
            if getattr(verdict, "selected_target", None) is not None:
                target = verdict.selected_target
            if verdict.action == SafetyAction.CLAMP:
                target = self.safety.clamp_target(target)
            grip = float(np.clip(grip, 0, self.hw.gripper.max_close_cmd))
            threshold = self.hw.safety.grip_latch_fz_n
            loads = self.safety.contact_load
            suppress_latch = False
            if self.release_controller is not None:
                from phantom.deploy.unlatched_finish import feedback_capture_times

                arm_times, arm_feedback = self.rings["arm"].latest(1)
                grip_times, grip_feedback = self.rings["gripper"].latest(1)
                measured = arm_feedback["tcp_pose"][0]
                measured_grip = grip_feedback["state"][0, 0]
                self.release_controller.note_latch(self._grip_latch)
                suppress_latch = self.release_controller.update(
                    t,
                    tcp=measured,
                    policy_grip=grip,
                    measured_grip=float(measured_grip),
                    pad_loads=loads,
                    eligible=not stale and self._original_policy_grip(),
                    accepted_grip=self._last_grip,
                    accepted_grip_ack=self._grip_ack,
                    feedback_times=feedback_capture_times(
                        arm_times[0], grip_times[0], self.safety.contact_load_times, self.hw.tactile.sensors,
                    ) if self.release_controller.unlatched_observer is not None else None,
                    finish_permitted=np.allclose(
                        self.safety.clamp_target(measured),
                        measured,
                        atol=1e-9,
                        rtol=0,
                    ),
                )
                if suppress_latch:
                    self._grip_latch = None
            if (
                threshold > 0
                and not suppress_latch
                and len(loads) >= 2
                and all(v > threshold for v in loads.values())
                and self._grip_latch is None
            ):
                self._grip_latch = grip
            if self._grip_latch is not None:
                self._grip_latch = max(self._grip_latch, grip)
                grip = self._grip_latch
            if self.release_controller is not None:
                self.release_controller.note_latch(self._grip_latch)
                if self.release_controller.finished:
                    if self.completed_reason is None:
                        self.completed_reason = "placement_release_finished"
                        self.completed_at_s = float(t)
                        self._finish_pose = measured.copy()
                        self._finish_grip = self._last_grip
                        if self.safety.boundary_projection is not None:
                            from phantom.deploy.boundary_projection import BoundaryProjectionStop
                            try:
                                self._finish_pose = self.safety.boundary_projection.request_finish(
                                    t, self._finish_grip, self._grip_ack)
                            except BoundaryProjectionStop as error:
                                self.request_stop(error.reason)
                                kinds.append(error.reason)
                                self._finish_pose = self._last_cmd.copy()
                        self._pending = self._prev_plan = None
                    target, grip = self._finish_pose.copy(), self._finish_grip
                    target = self.safety.clamp_target(target)
        dt_eff = float(np.clip(dt, period, 2 * period))
        if not self.stopped_reason and self.safety.boundary_projection is not None:
            from phantom.deploy.boundary_projection import BoundaryProjectionStop
            try:
                self.safety.boundary_projection.check_grip(grip)
            except BoundaryProjectionStop as error:
                self.request_stop(error.reason)
                kinds.append(error.reason)
                target, grip = self._last_cmd.copy(), self._last_grip
        target = np.asarray(target, dtype=np.float64).copy()
        rate_reference = (
            self._finish_pose
            if self.completed_reason and not self.stopped_reason
            else self._last_cmd
        )
        for sl, vmax in (
            (slice(0, 3), self.hw.arm.limits.tcp_speed_m_s),
            (slice(3, 6), self.hw.arm.limits.joint_speed_rad_s),
        ):
            if sl.start == 3:
                target[sl] = dv.rotvec_nearest(rate_reference[sl], target[sl])
            delta = target[sl] - rate_reference[sl]
            norm = np.linalg.norm(delta)
            if norm > vmax * dt_eff:
                target[sl] = rate_reference[sl] + delta * (vmax * dt_eff / norm)
        command = SimulationCommand(
            t,
            target,
            float(grip),
            dt_eff,
            self.stopped_reason is not None,
            self.stopped_reason,
            {
                "safety_events": kinds,
                "safety_target": safety_target,
                "wrist_guard": self.safety.wrench_diagnostics(),
                "completed_reason": self.completed_reason,
                "completed_at_s": self.completed_at_s,
                "completion_hold": self.completed_reason is not None,
                "stale_plan_hold": stale,
                "plan_activated": activated,
                "play_time_s": self._play_time,
                "ik_rejects": self.ik_rejects,
                "ik_rejects_total": self.ik_rejects_total,
            },
        )
        if self.release_controller is not None:
            measured = self.rings["arm"].latest(1)[1]["tcp_pose"][0]
            command.diagnostics["placement_release"] = (
                self.release_controller.diagnostics(measured)
            )
            command.diagnostics["grip_latch"] = self._grip_latch
            if self.release_controller.unlatched_observer is not None:
                command.diagnostics["original_policy_gripper_eligible"] = bool(
                    not self.stopped_reason and not self.completed_reason
                    and not stale and self._original_policy_grip()
                )
        self._awaiting_feedback = command
        if self.safety.boundary_projection is not None:
            # Mutable per-tick diagnostic receives final FK/ACK after the
            # runner verifies it, before this exact trace row is serialized.
            command.diagnostics["boundary_projection"] = self.safety.boundary_projection.last
        return command

    def report_execution(
        self,
        t: float,
        *,
        accepted: bool,
        tcp_pose=None,
        gripper_command: float | None = None,
        reason: str = "ik_rejected",
        held: bool = False,
        controller_stop: bool = False,
    ) -> None:
        """Confirm accepted setpoints after the simulator's IK/drive stage.

        ``tcp_pose`` is the actually accepted target after any additional IK
        shortening, not a fictitious measurement; subsequent ``observe`` calls
        provide measured physics state. When omitted on acceptance it defaults
        to the requested target. On rejection the prior confirmed target is
        retained. Only explicitly supplied gripper commands enter history,
        whether or not arm IK succeeded. Omit if no gripper command was sent.
        """
        if held and (not accepted or tcp_pose is None):
            raise ValueError("a streamed hold requires an explicit accepted pose")
        if controller_stop and (accepted or self.stopped_reason != reason):
            raise ValueError("controller-stop feedback requires the matching requested stop")
        cmd = self._awaiting_feedback
        if cmd is None or not np.isclose(t, cmd.t, rtol=0, atol=1e-9):
            raise ValueError(
                "execution feedback must match the pending command timestamp"
            )
        accepted_pose = (
            _vector(
                cmd.tcp_pose if tcp_pose is None else tcp_pose, 6, "accepted tcp_pose"
            ).copy()
            if accepted
            else None
        )
        if gripper_command is not None:
            grip = float(gripper_command)
            if not np.isfinite(grip) or not 0 <= grip <= self.hw.gripper.max_close_cmd:
                raise ValueError(
                    "executed gripper command is outside configured limits"
                )
        if accepted:
            self._last_cmd = accepted_pose
            if not held:
                self.ik_rejects = 0
        elif not controller_stop:
            self.ik_rejects += 1
            self.ik_rejects_total += 1
            if self.ik_rejects >= self.ik_reject_limit:
                self.request_stop(reason)
        if gripper_command is not None:
            self._last_grip = grip
            if ((self.release_controller is not None and self.release_controller.unlatched_observer is not None)
                    or self.safety.boundary_projection is not None):
                from phantom.deploy.unlatched_finish import acknowledge_gripper

                self._grip_ack = acknowledge_gripper(
                    self._grip_ack, t, grip,
                    bool(cmd.diagnostics.get("original_policy_gripper_eligible", False)
                         and np.isclose(grip, cmd.gripper, atol=1e-12, rtol=0)),
                )
            if not self._grip_hist or self._grip_hist[-1][1] != grip:
                self._grip_hist.append((float(t), grip))
        self._awaiting_feedback = None
