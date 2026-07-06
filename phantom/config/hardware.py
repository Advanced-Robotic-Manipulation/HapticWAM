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


class TactileSensorEntry(_Frozen):
    name: str
    # int = SDK device index (quick single-sensor tests); str = device serial
    # ("识别码", e.g. "X26040565", printed on the yellow cable label) —
    # vendor-recommended for multi-sensor rigs (stable across replug).
    dev_id: int | str

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
    keyframe_rate_hz: float = Field(gt=0)
    field_dtype: Literal["float16", "float32"] = "float16"
    save_infer_img: bool = True
    infer_img_rate_hz: float = Field(gt=0)
    archive_raw_img: bool = False
    ring_seconds: float = Field(gt=0)
    zarr_chunk_frames: int = Field(gt=0)
    drain_interval_s: float = Field(gt=0)


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

    def contains(self, p) -> bool:
        return (self.x[0] <= p[0] <= self.x[1]
                and self.y[0] <= p[1] <= self.y[1]
                and self.z[0] <= p[2] <= self.z[1])


class SafetyConfig(_Frozen):
    wrench_limit_N: float = Field(gt=0)
    wrench_limit_Nm: float = Field(gt=0)
    tactile_fz_limit_N: float = Field(gt=0)
    tactile_depth_limit: float = Field(gt=0)
    workspace_m: WorkspaceBox
    governor: GovernorConfig
    stale_plan_timeout_s: float = Field(gt=0)


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

    # ----- cross-section validation ---------------------------------------

    @model_validator(mode="after")
    def _cross_checks(self) -> "HardwareConfig":
        t, r, c = self.tactile, self.recording, self.control
        if self.tactile.field.h % r.field_ds.h or self.tactile.field.w % r.field_ds.w:
            raise ValueError(
                f"recording.field_ds {r.field_ds.hw} must divide tactile.field {t.field.hw} exactly"
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
        """qpos(dof) + qvel(dof) + tcp pose(6) + tcp speed(6) + gripper (pos, current)."""
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
        kf = (self.tactile.field.h * self.tactile.field.w * self.tactile.field_ch
              * itemsize * r.keyframe_rate_hz)
        return ds + kf

    def snapshot_yaml(self) -> str:
        """Canonical YAML dump for episode/checkpoint provenance."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.snapshot_yaml().encode()).hexdigest()[:16]

    def shape_relevant_fields(self) -> dict:
        """The subset of fields that change tensor shapes; checkpoint-compat is asserted on these."""
        t = self.tactile
        return {
            "field": t.field.hw,
            "field_ch": t.field_ch,
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
