"""Hardware configuration schema.

Loads and validates configs/hardware.yaml — the single source of truth for
every hardware-related hyperparameter (shapes, rates, unit scales, thresholds).
Ranks of shapes are fixed by this schema; values are expected to change after
the day-1 hardware bench (docs/hardware_bench_day1.md).

Rule enforced by review, aided by this module: no literal shape/rate/threshold
anywhere else in phantom/ — everything derives from HardwareConfig.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

log = logging.getLogger(__name__)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ImageShape(_Frozen):
    h: int = Field(gt=0)
    w: int = Field(gt=0)
    c: int = Field(gt=0)

    @property
    def hwc(self) -> tuple[int, int, int]:
        return (self.h, self.w, self.c)


class FieldShape(_Frozen):
    h: int = Field(gt=0)
    w: int = Field(gt=0)

    @property
    def hw(self) -> tuple[int, int]:
        return (self.h, self.w)


class MetaConfig(_Frozen):
    schema_version: Literal[1] = 1
    rig_name: str
    bench_verified: bool = False


class ModeConfig(_Frozen):
    drivers: Literal["mock", "real"] = "mock"
    overrides: dict[Literal["arm", "gripper", "tactile", "cameras"], Literal["mock", "real"]] = {}

    def resolve(self, device: str) -> str:
        return self.overrides.get(device, self.drivers)  # type: ignore[arg-type]


class ArmLimits(_Frozen):
    joint_speed_rad_s: float = Field(gt=0)
    tcp_speed_m_s: float = Field(gt=0)
    tcp_accel_m_s2: float = Field(gt=0)


class ServoJParams(_Frozen):
    lookahead_time_s: float = Field(gt=0)
    gain: int = Field(ge=100, le=2000)


class ArmConfig(_Frozen):
    model: str
    generation: Literal["e-series", "cb3"]
    ip: str
    rtde_receive_hz: float = Field(gt=0)
    rtde_control_hz: float = Field(gt=0)
    dof: int = Field(ge=6, le=7)
    tcp_offset_m: tuple[float, float, float, float, float, float]
    payload_kg: float = Field(ge=0)
    limits: ArmLimits
    servoj: ServoJParams

    @model_validator(mode="after")
    def _check_generation_rates(self) -> "ArmConfig":
        if self.generation == "cb3":
            if self.rtde_receive_hz > 125 or self.rtde_control_hz > 125:
                raise ValueError(
                    "CB3-generation arms support at most 125 Hz RTDE; got "
                    f"receive={self.rtde_receive_hz}, control={self.rtde_control_hz}"
                )
        else:
            for name, hz in (("rtde_receive_hz", self.rtde_receive_hz),
                             ("rtde_control_hz", self.rtde_control_hz)):
                if hz not in (125.0, 250.0, 500.0):
                    raise ValueError(f"e-series {name} must be one of 125/250/500 Hz, got {hz}")
        return self


class WristFTConfig(_Frozen):
    source: Literal["ur_internal", "ft300s"]
    rate_hz: float = Field(gt=0)
    dim: Literal[6] = 6
    window_s: float = Field(gt=0)
    bias_on_episode_start: bool = True
    bias_window_s: float = Field(gt=0)

    @property
    def window_len(self) -> int:
        """Number of F/T samples in the ACC/TCN window."""
        return max(2, round(self.rate_hz * self.window_s))


class GripperConfig(_Frozen):
    type: Literal["robotiq_2f85", "robotiq_2f140"]
    interface: Literal["urcap_socket"] = "urcap_socket"
    port: int = 63352
    stroke_mm: float = Field(gt=0)
    feedback_rate_hz: float = Field(gt=0)
    default_speed: float = Field(ge=0, le=1)
    default_force: float = Field(ge=0, le=1)
    # Normalized force command 0..1 -> physical grip force in N (linear over
    # the FOR register; 2F-85 datasheet range). BENCH: verify endpoints.
    force_range_N: tuple[float, float] = (20.0, 235.0)
    # Hard ceiling on COMMANDED grip force — the DM-Tac pad crushes above
    # 30 N total (docs/sensor_sdk.md). Enforced at the driver boundary:
    # every move() clamps its force arg to max_force_cmd.
    cmd_force_limit_N: float = Field(default=30.0, gt=0)
    # Hard ceiling on the COMMANDED close position (0=open..1=closed). With
    # tactile pads mounted on both fingers a full close presses pad into pad
    # — the lab's proven grip position was 155/255 ≈ 0.61. Enforced at the
    # driver boundary like the force clamp.
    max_close_cmd: float = Field(default=1.0, gt=0, le=1)

    @property
    def max_force_cmd(self) -> float:
        """Highest allowed normalized force command under cmd_force_limit_N."""
        lo, hi = self.force_range_N
        return max(0.0, min(1.0, (self.cmd_force_limit_N - lo) / (hi - lo)))

    @model_validator(mode="after")
    def _check_force(self) -> "GripperConfig":
        lo, hi = self.force_range_N
        if hi <= lo or lo < 0:
            raise ValueError(f"gripper.force_range_N must be increasing and >= 0, got ({lo}, {hi})")
        if self.default_force > self.max_force_cmd:
            est = lo + self.default_force * (hi - lo)
            raise ValueError(
                f"gripper.default_force={self.default_force} ≈ {est:.0f} N exceeds the "
                f"DM-Tac pad ceiling cmd_force_limit_N={self.cmd_force_limit_N} N "
                f"(max normalized command {self.max_force_cmd:.3f}) — lower default_force")
        return self


class TactileSensorEntry(_Frozen):
    name: str
    # int = SDK device index (quick single-sensor tests); str = device serial
    # ("识别码", e.g. "L26050098", printed on the yellow cable label) —
    # vendor-recommended for multi-sensor rigs (stable across replug).
    dev_id: int | str
    # Network streaming options (SensorOptions fields, verified against the
    # Denmark-rig scripts in incoming/DM-Tac-SDK — used there even with
    # cpu/cuda backends). None = leave the SDK default.
    remote_addr: str | None = None   # sensor's gRPC endpoint, e.g. "192.168.127.10:50051"
    pc_port: int | None = None       # port on the PC this sensor streams frames to

    @field_validator("dev_id")
    @classmethod
    def _dev_id_valid(cls, v: int | str) -> int | str:
        if isinstance(v, int) and v < 0:
            raise ValueError("dev_id index must be >= 0")
        if isinstance(v, str) and not v.strip():
            raise ValueError("dev_id serial must be non-empty")
        return v


class TactileFieldChannels(_Frozen):
    deformation2d: int = 2
    depth: int = 1
    shear: int = 2
    dist_force: int = 3

    @property
    def total(self) -> int:
        return self.deformation2d + self.depth + self.shear + self.dist_force

    @model_validator(mode="after")
    def _check_total(self) -> "TactileFieldChannels":
        if self.total != 8:
            raise ValueError(
                f"tactile field channel stack must sum to 8 (D||F per pipeline.md §1), got {self.total}"
            )
        return self


class TactileConfig(_Frozen):
    sdk: Literal["dmrobotics"] = "dmrobotics"
    # SDK runtime options (SDK_Publish_1.2.10: backends are lowercase; channel
    # enable flags default to False in the real SDK — the driver sets them).
    sdk_backend: Literal["cpu", "cuda", "flux"] = "cuda"
    sdk_mode: Literal["standard", "high"] = "high"
    rate_hz: float = Field(gt=0)
    raw_img: ImageShape
    infer_img: ImageShape
    field: FieldShape
    hw_order: Literal["hw", "wh"] = "hw"
    field_channels: TactileFieldChannels
    wrench_dim: Literal[6] = 6
    force_unit_to_N: float = Field(ge=0)       # 0.0 = UNCALIBRATED sentinel (getForce F, manual: N -> 1.0)
    torque_unit_to_Nm: float = Field(ge=0)     # 0.0 = UNCALIBRATED sentinel (getForce M, manual: 1e-2 N*m -> 0.01)
    dist_force_unit_to_N: float = Field(ge=0)  # 0.0 = UNCALIBRATED sentinel (getDistributeForce, units unverified)
    offline_recompute_ok: bool = False
    # PC-side host address the sensors stream frames back to (SensorOptions
    # pc_host; shared by all sensors on the rig). None = SDK default "0.0.0.0".
    pc_host: str | None = None
    sensors: tuple[TactileSensorEntry, ...]

    @property
    def field_ch(self) -> int:
        return self.field_channels.total

    @property
    def wrench_calibrated(self) -> bool:
        """Resultant 6-axis wrench (getForce) is in SI units after driver scaling."""
        return self.force_unit_to_N > 0 and self.torque_unit_to_Nm > 0

    @property
    def force_calibrated(self) -> bool:
        """Full force calibration incl. the distributed field (gates fz-based
        contact/safety paths and the calibrated HID-S penalty)."""
        return self.force_unit_to_N > 0 and self.dist_force_unit_to_N > 0

    @field_validator("sensors")
    @classmethod
    def _unique_sensors(cls, v: tuple[TactileSensorEntry, ...]) -> tuple[TactileSensorEntry, ...]:
        names = [s.name for s in v]
        ids = [s.dev_id for s in v]
        if len(set(names)) != len(names) or len(set(ids)) != len(ids):
            raise ValueError("tactile sensor names and dev_ids must be unique")
        if len(v) < 1:
            raise ValueError("at least one tactile sensor must be configured")
        return v


class CameraEntry(_Frozen):
    enabled: bool = True
    serial: str = ""
    color: ImageShape
    fps: float = Field(gt=0)
    depth_enabled: bool = False


class CamerasConfig(_Frozen):
    scene: CameraEntry
    wrist: CameraEntry


class DerivedConfig(_Frozen):
    tau_contact_depth: float = Field(gt=0)
    # frame-level contact requires this FRACTION of pad pixels above
    # tau_contact_depth. max-over-110k-pixels statistics saturate (one hot
    # pixel = "contact"): measured on the v3 dataset, gate labels were 100%
    # positive and events 96% "hold" across every task — the tactile heads
    # trained on constants (issue #1, verified 2026-08-16). Calibrated from
    # data: free-frame mask_frac max 0.022 vs contact p10 0.031 — 0.025 sits
    # in the empirical gap (0% false-contact, ~95% true-contact).
    tau_contact_area: float = Field(gt=0, le=1, default=0.025)
    tau_contact_fz: float = Field(gt=0)
    contact_source: Literal["depth", "fz"] = "depth"
    tau_slip: float = Field(gt=0)
    slip_flow_weight: float = Field(ge=0, le=1)
    friction_cone_mu: float = Field(gt=0)
    cop_min_contact_cells: int = Field(gt=0)
    area_crosscheck_tol: float = Field(gt=0)
    cpk_downsample: int = Field(gt=0)
    event_lookahead_s: float = Field(gt=0)
    event_min_hold_ticks: int = Field(ge=1)


class RecordingConfig(_Frozen):
    field_ds: FieldShape
    field_ds_rate_hz: float = Field(gt=0)
    # Downsampled keyframe resolution (mirrors field_ds: must divide
    # tactile.field exactly — validated). Keyframes used to be stored at the
    # native tactile.field resolution, which TactileFieldEncoder immediately
    # crushed down through several stride-2 stages anyway (nothing downstream
    # consumes native res) — so keyframe_ds cuts keyframe bytes/compression
    # cost (and the encoder's stage count) with no loss to any consumer.
    keyframe_ds: FieldShape
    keyframe_rate_hz: float = Field(gt=0)
    field_dtype: Literal["float16", "float32"] = "float16"
    save_infer_img: bool = True
    infer_img_rate_hz: float = Field(gt=0)
    archive_raw_img: bool = False
    ring_seconds: float = Field(gt=0)
    zarr_chunk_frames: int = Field(gt=0)
    # Upper bound on ONE zarr chunk's uncompressed bytes. zarr rewrites the
    # ENTIRE partially-filled chunk on every append, so append cost scales with
    # chunk BYTES, not rows: a flat 120-frame chunk is 212 MB for the
    # 288x384x8 keyframe stream, which drove drains to 5.8 s, overflowed the
    # 5 s ring and SILENTLY dropped samples (ringbuffer.drain resumes from the
    # oldest surviving row). Rows/chunk = min(zarr_chunk_frames,
    # zarr_chunk_mb / row_bytes).
    zarr_chunk_mb: float = Field(default=8.0, gt=0)
    drain_interval_s: float = Field(gt=0)
    # Scene/camera RGB is JPEG-encoded on disk at this quality (0-100). Raw
    # 640x480x3 @ 30 fps is ~1.6 GB/min; JPEG cuts that ~20-50x. Storage quality
    # is kept above the rerun viz quality (90). Set 0 to store raw uint8 frames.
    scene_jpeg_quality: int = Field(default=92, ge=0, le=100)


class ControlConfig(_Frozen):
    model_tick_hz: float = Field(gt=0)
    action_rate_hz: float = Field(gt=0)
    action_dim: Literal[7] = 7  # Δ-EE pose (6) + gripper (1); deliberate schema tripwire
    chunk_horizon: int = Field(gt=0)
    executor_rate_hz: float = Field(gt=0)
    replan_min_lead_s: float = Field(gt=0)
    chunk_blend_s: float = Field(gt=0)

    @property
    def chunk_duration_s(self) -> float:
        return self.chunk_horizon / self.action_rate_hz


class GovernorConfig(_Frozen):
    sigma_lo: float = Field(ge=0)
    sigma_hi: float = Field(gt=0)
    min_scale: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _check_order(self) -> "GovernorConfig":
        if self.sigma_hi <= self.sigma_lo:
            raise ValueError("governor sigma_hi must exceed sigma_lo")
        return self


class WorkspaceBox(_Frozen):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @model_validator(mode="after")
    def _check_bounds(self) -> "WorkspaceBox":
        for axis in ("x", "y", "z"):
            lo, hi = getattr(self, axis)
            if hi <= lo:
                raise ValueError(f"workspace {axis} bounds must be increasing, got ({lo}, {hi})")
        return self

    def contains(self, p, margin: float = 0.0) -> bool:
        """margin > 0 expands the box — used as resume-hysteresis so a pose
        frozen marginally outside the boundary still counts as back inside."""
        return (self.x[0] - margin <= p[0] <= self.x[1] + margin
                and self.y[0] - margin <= p[1] <= self.y[1] + margin
                and self.z[0] - margin <= p[2] <= self.z[1] + margin)


class SafetyConfig(_Frozen):
    # ArmGuard crash detector. On the CB3 (no real F/T sensor) getActualTCPForce
    # is a current-based ESTIMATE with a large, POSE-dependent static bias
    # (~18 N / ~5 Nm measured at rest) — so these limits are NOT absolute, they
    # are the allowed DEVIATION from a slow rolling baseline that tracks the
    # bias out. A sudden, SUSTAINED deviation (held wrench_debounce_ticks
    # consecutive checks) is a collision; brief acceleration spikes are ignored.
    wrench_limit_N: float = Field(gt=0)
    wrench_limit_Nm: float = Field(gt=0)
    wrench_baseline_tau_s: float = Field(default=2.0, gt=0)   # rolling-baseline time constant
    wrench_debounce_ticks: int = Field(default=3, ge=1)       # consecutive over-limit checks to trip
    # Deployment SafetyMonitor only; data-collection ArmGuard remains rolling.
    # Fixed mode requires a reviewed unloaded episode start. It does not model
    # the CB3's pose-dependent current-estimate bias or alter teacher inputs.
    wrench_baseline_mode: Literal["rolling_calm", "episode_fixed"] = "rolling_calm"
    # Measured joint-speed stop (rig 2026-09-01): near a singular configuration
    # servoL turns a LEGAL Cartesian step into a joint whip (wrists hit
    # 5-7 rad/s carrying a grasped object toward full extension) — the
    # Cartesian rate limiter cannot see it, only the measured qd can. Normal
    # deploy motion stays under ~1 rad/s (max 0.73 measured over 19 whip-free
    # rig episodes 09-01, so 1.2 is free); a whip crosses it within ~2-4
    # ticks of onset. Last-ditch only — the wd guard and the IK branch guard
    # lead it by 0.3-1.4 s. The stop is NOT a let-go (a held object stays
    # held).
    joint_speed_stop_rad_s: float = Field(default=1.2, gt=0)
    # The guard that actually leads the whips (run analysis 09-01): the wrist
    # CENTRE's distance from the shoulder is pure elbow geometry —
    # sqrt(a2^2 + a3^2 + 2*a2*a3*cos(q_elbow) + d4^2), boundary at full
    # extension sqrt((a2+a3)^2 + d4^2) = 470.5 mm on the UR3. All four 09-01
    # whips touched it. Threshold set from DATA on both sides: every demo ever
    # recorded stays <= 0.461 (waffles p95 0.451, 7% above 0.450 — a 0.45
    # guard would fight demo-faithful lifts), every 09-01 whip/near-miss sat
    # at 0.465+. None disables (non-UR3 arms until DH is set).
    wrist_extension_stop_m: float | None = Field(default=0.462)
    # Tick-rate SUCCESS stop (rig 09-04): both pads loaded over their
    # per-episode baseline AND tcp z above the lift height, sustained ->
    # the episode ends HOLDING. Evaluated in SafetyMonitor.check() at the
    # executor rate so it wins the race against wrist_extension (the
    # per-replan planner check lost 4 real grasps to the guard on 09-04).
    # Thresholds from the 29-episode 09-04 evaluation: trailing-window max of
    # the baseline-corrected pad force (the pads drop out for up to 40% of
    # frames on a real carry — an instantaneous test is unsatisfiable on one
    # of the six carried grasps), 4.0 N (weakest carry 7.3 N, strongest
    # touch-lost already at noise by 0.32 m), z 0.32 m (0.30 false-fires on
    # one empty lift with a trailing window). 6/6 carried, 0/11 empty lifts.
    lift_complete_z_m: float = Field(default=0.32, ge=0)      # 0 disables
    lift_complete_fz_n: float = Field(default=4.0, ge=0)
    lift_complete_hold_s: float = Field(default=0.4, ge=0)
    lift_complete_window_s: float = Field(default=0.6, ge=0)  # trailing-max window
    # Aperture latch (09-04 touch-lost mechanism): in 4 of 5 lost objects the
    # policy was commanding the fingers OPEN while carrying (demos never
    # contain a post-lift release). Once both pads register this much load
    # the commanded closure can no longer decrease until an intended release
    # (episode end or a veto recovery). 0 disables.
    grip_latch_fz_n: float = Field(default=2.5, ge=0)
    # PLACEMENT RELEASE of the latch (rig 09-08 seed 110: the student carried
    # the packet to the box, lowered to z 0.10 and commanded the fingers from
    # 0.64 down to 0.35 for five seconds — the latch never let go, because
    # its only clears were lost-object recovery and episode end). The latch
    # clears when the policy asks to open by >= grip_latch_release_drop below
    # the latched closure for >= grip_latch_release_s continuously, the TCP
    # is below grip_latch_release_z_m (placement height), and the TCP has
    # been above grip_latch_release_z_m + grip_latch_release_lift_m since the
    # latch armed (a carry happened; never at the grasp site). The 09-04
    # mid-carry drops were commanded high, so they stay blocked. After a
    # release the latch re-arms only once both pads have unloaded.
    # release_drop 0 disables the release (latch behaves as before).
    grip_latch_release_drop: float = Field(default=0.15, ge=0)
    grip_latch_release_s: float = Field(default=0.5, ge=0)
    grip_latch_release_z_m: float = Field(default=0.16)
    grip_latch_release_lift_m: float = Field(default=0.08, ge=0)
    ur_dh_a2_a3_d4_m: tuple[float, float, float] = (0.24365, 0.21325, 0.11235)
    ur_dh_d1_m: float = 0.1519            # shoulder height above the base frame
    # Servo-level LIMITER (rig 2026-09-04) — the answer to "the arm stops at
    # the apex of every carry": with lift_complete off, 4/5 carries ended on
    # wrist_extension / joint_speed at elbow 10-20 deg. That was not a whip:
    # the elbow speed ramped smoothly 0.5 -> 1.3 rad/s while the TCP held
    # 100 mm/s, i.e. the policy asked for a TCP beyond the arm's reach. A stop
    # cannot fix that; a clamp can. Per servo tick the driver solves IK for the
    # requested pose and, if the elbow would fold below `elbow_min_rad` or any
    # joint would exceed `servo_joint_speed_max_rad_s`, shortens the step
    # (bisection, then a tangential slide along the reach sphere) so the arm
    # hugs the boundary instead of stopping. 0.40 rad = 22.9 deg = wrist
    # centre 0.4616 m, just inside every demo (max 0.461) and 7 mm inside the
    # 0.468 wrist_extension stop, which stays as the last net. None disables.
    # DEFAULT OFF (09-05 review, docs/review_servo_limiter_0905.md): the
    # limiter ran for six 09-04 episodes (19:50-19:56, no guard fired, apex
    # 0.34-0.36 m) but three defects reproduce in its own harness (IndexError
    # on an unreachable target — fixed,
    # deadlock + 25-reject crash when the anchor already violates emin, the
    # executor never learning a step was shortened) block it. Ilya's values
    # were 0.40 rad / 1.0 rad/s — set them in the NUC yaml to enable once fixed.
    elbow_min_rad: float | None = Field(default=None)
    servo_joint_speed_max_rad_s: float | None = Field(default=None, gt=0)
    # Opt-in constrained-setpoint hold budget; separate from invalid-IK faults.
    # A held joint target is streamed at servo cadence. No-op sends or plan
    # arrivals do not refresh this deadline. Requires an enabled servo limiter.
    servo_constraint_hold_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    # RETIRED as a default (run analysis 09-01): the TCP-radius clamp fired in
    # 0 of 4 whips (their commanded radius peaked just under it) and in the one
    # episode it did engage it dragged commanded z down 32 mm mid-lift. The
    # wrist_extension guard above measures the actual boundary. None = off.
    reach_clamp_m: float | None = Field(default=None)
    tactile_fz_limit_N: float = Field(gt=0)
    tactile_depth_limit: float = Field(gt=0)
    workspace_m: WorkspaceBox
    # Optional per-task HITBOX (set by run_deploy from the task's demo TCP
    # envelope + margin, never from yaml by default): a commanded target
    # outside it ends the episode with a stop — unlike workspace_m, which only
    # clamps. Rig 2026-08-28: rollouts wandered 300 mm past any demo pose.
    hitbox_m: WorkspaceBox | None = None
    governor: GovernorConfig
    stale_plan_timeout_s: float = Field(gt=0)


class EchoTeleopConfig(_Frozen):
    """Echo exoskeleton leader (STM32 over USB serial; protocol facts verified
    against the lab's working stack, third_party/echo_teleop/PROVENANCE.md)."""
    vid: int = 1603
    pid: int = 1868
    baud: int = 115200
    # Joint-space reference: q_target = base_pose + exo_offset / divisor.
    base_pose: tuple[float, float, float, float, float, float]
    # Indexed by the device's sense_flag (0 / 1 / 2).
    sensitivity_divisors: tuple[float, float, float] = (1.0, 1.25, 1.75)
    # Raw exoskeleton gripper ticks mapped to 0 (open) .. 1 (closed).
    # BENCH: calibrate on the rig (read ticks at full open / full close).
    gripper_open_tick: int
    gripper_closed_tick: int
    # --- smoothing / control (teleop/filters.py + teleop/streamer.py) ------
    # High-rate servo streaming, decoupled from control.action_rate_hz
    # (recording stays on the action grid). 125 Hz is CB3-safe.
    control_rate_hz: float = Field(default=125.0, gt=0)
    # One-euro filter on the leader joint signal (applied at device rate):
    # min_cutoff Hz at rest (lower = stronger tremor suppression), beta adds
    # cutoff per rad/s of motion (higher = less lag when moving fast), d_cutoff
    # Hz smooths the SPEED estimate that drives beta (lower = the filter is
    # slow to "open up" at motion onset -> mushy starts; raise for snappier
    # engagement). See collect/config.py TeleopTuning for the tuned rig values.
    filter_min_cutoff: float = Field(default=1.0, gt=0)
    filter_beta: float = Field(default=0.3, ge=0)
    filter_d_cutoff: float = Field(default=1.0, gt=0)
    # Acceleration bound of the joint tracker (velocity bound comes from
    # arm.limits.joint_speed_rad_s); turns steps/engage into smooth S-curves.
    joint_accel_rad_s2: float = Field(default=4.0, gt=0)
    # Gripper conditioning: EMA factor at device rate + command deadband
    # (a new socket command is sent only when the change exceeds this).
    gripper_ema_alpha: float = Field(default=0.2, gt=0, le=1)
    gripper_deadband: float = Field(default=0.02, ge=0, le=1)

    @model_validator(mode="after")
    def _check_gripper_ticks(self) -> "EchoTeleopConfig":
        if self.gripper_open_tick == self.gripper_closed_tick:
            raise ValueError("gripper_open_tick and gripper_closed_tick must differ")
        return self


class TeleopConfig(_Frozen):
    echo: EchoTeleopConfig | None = None


class HardwareConfig(_Frozen):
    meta: MetaConfig
    mode: ModeConfig
    arm: ArmConfig
    wrist_ft: WristFTConfig
    gripper: GripperConfig
    tactile: TactileConfig
    cameras: CamerasConfig
    derived: DerivedConfig
    recording: RecordingConfig
    control: ControlConfig
    safety: SafetyConfig
    teleop: TeleopConfig | None = None

    # ----- cross-section validation ---------------------------------------

    @model_validator(mode="after")
    def _cross_checks(self) -> "HardwareConfig":
        t, r, c = self.tactile, self.recording, self.control
        if self.tactile.field.h % r.field_ds.h or self.tactile.field.w % r.field_ds.w:
            raise ValueError(
                f"recording.field_ds {r.field_ds.hw} must divide tactile.field {t.field.hw} exactly"
            )
        if self.tactile.field.h % r.keyframe_ds.h or self.tactile.field.w % r.keyframe_ds.w:
            raise ValueError(
                f"recording.keyframe_ds {r.keyframe_ds.hw} must divide tactile.field "
                f"{t.field.hw} exactly"
            )
        if r.field_ds_rate_hz > t.rate_hz or r.keyframe_rate_hz > t.rate_hz:
            raise ValueError("recording field/keyframe rates cannot exceed tactile.rate_hz")
        if r.infer_img_rate_hz > t.rate_hz:
            raise ValueError("recording.infer_img_rate_hz cannot exceed tactile.rate_hz")
        if abs(c.executor_rate_hz - self.arm.rtde_control_hz) > 1e-9:
            raise ValueError(
                f"control.executor_rate_hz ({c.executor_rate_hz}) must equal "
                f"arm.rtde_control_hz ({self.arm.rtde_control_hz})"
            )
        if self.wrist_ft.source == "ur_internal" and self.wrist_ft.rate_hz > self.arm.rtde_receive_hz:
            raise ValueError("wrist_ft.rate_hz cannot exceed arm.rtde_receive_hz for ur_internal source")
        if self.arm.generation == "cb3" and self.wrist_ft.source == "ur_internal":
            raise ValueError(
                "CB3 arms only estimate TCP force from joint currents; pipeline.md §1 requires "
                "an external FT-300S (set wrist_ft.source: ft300s)"
            )
        if self.derived.contact_source == "fz" and not self.tactile.force_calibrated:
            raise ValueError(
                "derived.contact_source=fz requires calibrated force units "
                "(tactile.force_unit_to_N and dist_force_unit_to_N > 0); use contact_source=depth"
            )
        if t.field.h % self.derived.cpk_downsample or t.field.w % self.derived.cpk_downsample:
            raise ValueError(
                f"derived.cpk_downsample ({self.derived.cpk_downsample}) must divide "
                f"tactile.field {t.field.hw} exactly"
            )
        if c.replan_min_lead_s >= c.chunk_duration_s:
            raise ValueError(
                f"control.replan_min_lead_s ({c.replan_min_lead_s}) must be smaller than the chunk "
                f"duration ({c.chunk_duration_s:.2f}s = chunk_horizon/action_rate_hz)"
            )
        return self

    # ----- derived quantities (single definitions, used everywhere) --------

    @property
    def n_fingers(self) -> int:
        return len(self.tactile.sensors)

    @property
    def cpk_shape(self) -> tuple[int, int]:
        """Contact-package (generated Δ-field) resolution."""
        d = self.derived.cpk_downsample
        return (self.tactile.field.h // d, self.tactile.field.w // d)

    @property
    def ur_state_dim(self) -> int:
        """qpos(dof) + qvel(dof) + tcp pose(6) + tcp speed(6) + gripper (pos, obj)."""
        return 2 * self.arm.dof + 6 + 6 + 2

    @property
    def contact_state_dim(self) -> int:
        """Per-finger contact-state vector: wrench(6) + area(1) + CoP(2) + slip(1) + mask_frac(1)."""
        return self.tactile.wrench_dim + 1 + 2 + 1 + 1

    def field_bytes_per_s(self) -> float:
        """Estimated on-disk field throughput per sensor (pre-compression)."""
        r = self.recording
        itemsize = 2 if r.field_dtype == "float16" else 4
        ds = r.field_ds.h * r.field_ds.w * self.tactile.field_ch * itemsize * r.field_ds_rate_hz
        kf = (r.keyframe_ds.h * r.keyframe_ds.w * self.tactile.field_ch
              * itemsize * r.keyframe_rate_hz)
        return ds + kf

    def snapshot_yaml(self) -> str:
        """Canonical YAML dump for episode/checkpoint provenance."""
        data = self.model_dump(mode="json")
        # Preserve legacy hashes for the unchanged default controller. Full
        # model_dump metadata still records the effective mode explicitly.
        if self.safety.wrench_baseline_mode == "rolling_calm":
            data["safety"].pop("wrench_baseline_mode", None)
        if self.safety.servo_constraint_hold_s is None:
            data["safety"].pop("servo_constraint_hold_s", None)
        return yaml.safe_dump(data, sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.snapshot_yaml().encode()).hexdigest()[:16]

    def shape_relevant_fields(self) -> dict:
        """The subset of fields that change tensor shapes; checkpoint-compat is asserted on these."""
        t, r = self.tactile, self.recording
        return {
            "field": t.field.hw,
            "field_ch": t.field_ch,
            "keyframe_ds": r.keyframe_ds.hw,
            "n_fingers": self.n_fingers,
            "wrench_dim": t.wrench_dim,
            "wrist_ft_dim": self.wrist_ft.dim,
            "wrist_window_len": self.wrist_ft.window_len,
            "ur_state_dim": self.ur_state_dim,
            "contact_state_dim": self.contact_state_dim,
            "action_dim": self.control.action_dim,
            "chunk_horizon": self.control.chunk_horizon,
            "cpk_shape": self.cpk_shape,
        }


DEFAULT_HARDWARE_YAML = Path(__file__).resolve().parents[2] / "configs" / "hardware.yaml"


def load_hardware(path: str | Path | None = None, *, quiet: bool = False) -> HardwareConfig:
    """Load + validate the hardware config; emit provenance warnings."""
    path = Path(path) if path is not None else DEFAULT_HARDWARE_YAML
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = HardwareConfig.model_validate(raw)

    if not quiet:
        any_real = cfg.mode.drivers == "real" or "real" in cfg.mode.overrides.values()
        if any_real and not cfg.meta.bench_verified:
            warnings.warn(
                "hardware.yaml has real drivers enabled but meta.bench_verified=false — "
                "run docs/hardware_bench_day1.md and fill the BENCH fields first.",
                stacklevel=2,
            )
        if not cfg.tactile.force_calibrated:
            log.info(
                "tactile distributed-force units UNCALIBRATED (dist_force_unit_to_N=0, "
                "bench item a): running on depth-based contact + deformation fallback "
                "(pipeline.md §5). Resultant wrench calibrated: %s.",
                cfg.tactile.wrench_calibrated,
            )
        per_sensor = cfg.field_bytes_per_s() / 1e6
        log.info(
            "recording throughput estimate: %.1f MB/s/sensor fields (x%d sensors) pre-compression",
            per_sensor, cfg.n_fingers,
        )
    return cfg
