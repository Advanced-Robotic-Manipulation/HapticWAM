"""Abstract driver interfaces + frame dataclasses.

Every device has a real implementation (phantom/drivers/real/) and a mock
(phantom/drivers/mock/); both implement these ABCs, selected by
phantom.drivers.factory.make_rig from the hardware config's mode section.

Conventions:
 - every frame carries t_host = time.perf_counter() at read; the sync layer
   fills t_master (RTDE master clock) downstream;
 - all shapes are validated against the hardware config once, on the first
   frame after connect() (drivers call self._check_first_frame);
 - all tactile arrays are in CANONICAL (field.h, field.w) order — the real
   driver transposes at the boundary if hw_order == "wh".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from phantom.config.hardware import (CameraEntry, GripperConfig, HardwareConfig,
                                     TactileConfig, TactileSensorEntry)


# ---------------------------------------------------------------------------
# frame dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TactileFrame:
    t_host: float
    seq: int
    deformation: np.ndarray          # (H, W, 2) f32 — getDeformation2D
    depth: np.ndarray                # (H, W)    f32 — getDepth
    shear: np.ndarray                # (H, W, 2) f32 — getShear
    dist_force: np.ndarray           # (H, W, 3) f32 — getDistributeForce (fx, fy, fz)
    wrench: np.ndarray               # (6,)      f32 — getForce, already * force_unit_to_N (if calibrated)
    contact_area_mm2: float          # getContactArea
    infer_img: np.ndarray | None = None   # uint8 (h, w[, c]) — getInferImg
    raw_img: np.ndarray | None = None     # uint8 (h, w[, c]) — getRawImg

    def field_stack(self) -> np.ndarray:
        """Canonical 8-ch stack [disp_x, disp_y, depth, shear_x, shear_y, fx, fy, fz]."""
        return np.concatenate(
            [self.deformation, self.depth[..., None], self.shear, self.dist_force],
            axis=-1, dtype=np.float32)


@dataclass(frozen=True)
class ArmState:
    t_host: float
    seq: int
    t_rtde: float                    # controller timestamp (master clock source)
    q: np.ndarray                    # (dof,) joint positions, rad
    qd: np.ndarray                   # (dof,) joint velocities, rad/s
    tcp_pose: np.ndarray             # (6,) x,y,z + axis-angle rx,ry,rz
    tcp_speed: np.ndarray            # (6,)
    ft: np.ndarray                   # (6,) wrist wrench (tool frame)
    protective_stop: bool
    robot_mode: int


@dataclass(frozen=True)
class GripperState:
    t_host: float
    seq: int
    position: float                  # 0 (open) .. 1 (closed)
    current: float                   # motor current, normalized
    moving: bool
    object_detected: bool


@dataclass(frozen=True)
class CameraFrame:
    t_host: float
    seq: int
    color: np.ndarray                # (h, w, 3) uint8
    depth: np.ndarray | None = None  # (h, w) uint16
    t_device: float | None = None


# ---------------------------------------------------------------------------
# ABCs
# ---------------------------------------------------------------------------

class TactileSensor(ABC):
    def __init__(self, cfg: TactileConfig, sensor: TactileSensorEntry):
        self.cfg = cfg
        self.sensor = sensor
        self._shape_checked = False

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read(self) -> TactileFrame:
        """Block until the next sample at native rate; return canonical frame."""

    @abstractmethod
    def reset_reference(self) -> None:
        """Re-zero the no-contact reference (getBaseFrame)."""

    @property
    def field_shape(self) -> tuple[int, int]:
        return self.cfg.field.hw

    def _check_first_frame(self, f: TactileFrame) -> None:
        if self._shape_checked:
            return
        H, W = self.cfg.field.hw
        expected = {"deformation": (H, W, 2), "depth": (H, W),
                    "shear": (H, W, 2), "dist_force": (H, W, 3),
                    "wrench": (self.cfg.wrench_dim,)}
        for name, shape in expected.items():
            got = getattr(f, name).shape
            if tuple(got) != shape:
                raise RuntimeError(
                    f"tactile[{self.sensor.name}] {name} shape {got} != configured {shape}; "
                    "fix configs/hardware.yaml (tactile.field / hw_order)")
        self._shape_checked = True


class Arm(ABC):
    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    @abstractmethod
    def connect(self, *, control: bool = False) -> None:
        """control=True also opens the (exclusive) control interface."""

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_state(self) -> ArmState:
        """Latest state, non-blocking."""

    @abstractmethod
    def servo_j(self, q: np.ndarray, dt: float, lookahead: float, gain: int) -> None: ...

    @abstractmethod
    def servo_l(self, tcp_pose: np.ndarray, dt: float, lookahead: float, gain: int) -> None:
        """Cartesian servo (pose target); implementations may IK + servo_j."""

    @abstractmethod
    def speed_l(self, xd: np.ndarray, accel: float, dt: float) -> None: ...

    @abstractmethod
    def move_j(self, q: np.ndarray, speed: float, accel: float, blocking: bool = True) -> None:
        """Setup moves only — never inside the control loop."""

    @abstractmethod
    def stop(self, decel: float) -> None: ...

    @abstractmethod
    def zero_ft(self) -> None: ...

    @abstractmethod
    def is_protective_stopped(self) -> bool: ...

    @abstractmethod
    def servo_stop(self) -> None:
        """Exit servo mode cleanly (must be called before move_j)."""


class Gripper(ABC):
    def __init__(self, cfg: GripperConfig):
        self.cfg = cfg

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def activate(self) -> None: ...

    @abstractmethod
    def move(self, position: float, speed: float, force: float) -> None:
        """All args normalized 0..1; non-blocking."""

    @abstractmethod
    def get_state(self) -> GripperState: ...

    def open(self) -> None:
        self.move(0.0, self.cfg.default_speed, self.cfg.default_force)

    def close(self) -> None:
        self.move(1.0, self.cfg.default_speed, self.cfg.default_force)


class Camera(ABC):
    def __init__(self, cfg: CameraEntry, name: str):
        self.cfg = cfg
        self.name = name

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read(self) -> CameraFrame:
        """Block until the next frame at configured fps."""


# ---------------------------------------------------------------------------
# rig aggregate
# ---------------------------------------------------------------------------

@dataclass
class Rig:
    hw: HardwareConfig
    arm: Arm
    gripper: Gripper
    tactile: dict[str, TactileSensor] = field(default_factory=dict)
    cameras: dict[str, Camera] = field(default_factory=dict)
    control: bool = False

    def connect_all(self) -> None:
        self.arm.connect(control=self.control)
        self.gripper.connect()
        for s in self.tactile.values():
            s.connect()
        for c in self.cameras.values():
            c.connect()

    def disconnect_all(self) -> None:
        for c in self.cameras.values():
            c.disconnect()
        for s in self.tactile.values():
            s.disconnect()
        self.gripper.disconnect()
        self.arm.disconnect()

    def __enter__(self) -> "Rig":
        self.connect_all()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect_all()
