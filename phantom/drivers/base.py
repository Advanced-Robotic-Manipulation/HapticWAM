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

import time
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


class ServoFault(RuntimeError):
    """A servo stream the driver cannot continue. `stop_reason` names the
    class in stop.json (rig 09-07: every driver raise collapsed into
    `executor_crash`, so a 0.2 s reach-boundary hold and a UR protective stop
    were indistinguishable in the episode record)."""
    stop_reason = "executor_crash"


class ServoHoldTimeout(ServoFault):
    """The target stayed beyond reach for HOLD_BUDGET_S (arm stationary)."""
    stop_reason = "servo_hold_timeout"


class ServoBranchFault(ServoFault):
    """25 consecutive off-branch IK solutions — the seed is lost."""
    stop_reason = "servo_branch_fault"


class ControlLost(ServoFault):
    """servoJ rejected: the controller dropped the script (protective stop,
    e-stop, popup). `diag` = URArm._diagnose_control_loss snapshot."""
    stop_reason = "control_lost"


class ServoResult:
    """Outcome of one servo tick — the executor anchors its rate limit,
    its recorded action and every replan on `pose`, never on the proposal."""
    __slots__ = ("sent", "pose", "reason")

    def __init__(self, sent: bool, pose, reason: str = "sent"):
        self.sent = bool(sent)
        self.pose = None if pose is None else np.asarray(pose, dtype=np.float64).copy()
        self.reason = reason

    def __repr__(self) -> str:
        return f"ServoResult(sent={self.sent}, reason={self.reason!r})"


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
    # Robotiq gOBJ status, raw 0..3: 0 moving, 1 contact while opening,
    # 2 contact while closing (= holding an object), 3 at requested position.
    # This replaced the motor-current channel on 2026-07-31: no route to a
    # real current/force magnitude exists on this rig (GET CUR answers "?",
    # the undocumented COU/MSC/PCO/DST registers stay 0 or carry only the
    # same stall bit, and RTDE tool_output_current measured a flat 0.0 mA
    # through a genuine 8 s motor stall). OBJ is the one contact-bearing
    # gripper channel that actually responds, and it costs nothing extra --
    # get_state() already queried it for the booleans below.
    obj: float
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

    def read_raw_img(self) -> np.ndarray:
        """Full raw camera frame (uint8 grayscale) — the offline-recompute
        archive path (recording.archive_raw_img; docs/sensor_sdk.md)."""
        raise NotImplementedError(f"{type(self).__name__} does not provide raw frames")

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
    def servo_l(self, tcp_pose: np.ndarray, dt: float, lookahead: float,
                gain: int) -> "ServoResult | None":
        """Cartesian servo (pose target); implementations may IK + servo_j.

        Returns what was ACTUALLY streamed (issue #7, review 09-05): a
        `ServoResult` with `sent=True` and the streamed pose (possibly
        shortened by a driver-side limiter), or `sent=False` + reason when
        the driver held the previous setpoint (IK branch reject, invalid IK,
        limiter hold). `None` = "sent as given" (simple drivers/mocks)."""

    @abstractmethod
    def speed_l(self, xd: np.ndarray, accel: float, dt: float) -> None: ...

    def move_l(self, tcp_pose: np.ndarray, speed: float, accel: float,
               blocking: bool = True) -> None:
        """Linear TCP move (moveL). Slow homing moves only - never the servo
        path. Default raises: drivers without a linear move (teleop stubs)
        simply can't home - run_deploy catches and asks for a manual jog."""
        raise NotImplementedError(f"{type(self).__name__} has no move_l")

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
        """All args normalized 0..1; non-blocking. Implementations MUST pass
        force through clamp_force() — the pad-safe ceiling is enforced at
        this boundary, not by callers."""

    def clamp_force(self, force: float) -> float:
        """Hard cap: commanded force never exceeds gripper.cmd_force_limit_N
        (DM-Tac pads crush above 30 N — docs/sensor_sdk.md)."""
        return min(max(float(force), 0.0), self.cfg.max_force_cmd)

    def clamp_position(self, position: float) -> float:
        """Hard cap on close: with pads on both fingers a full close is
        pad-into-pad — never exceed gripper.max_close_cmd."""
        return min(max(float(position), 0.0), self.cfg.max_close_cmd)

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

    # -- health contract (a wedged camera must never look like a working one) --
    @property
    def healthy(self) -> bool:
        """False whenever the driver is not currently serving frames. Drivers
        that can wedge (RealSenseCamera) override this; a frozen stream is
        otherwise indistinguishable from a slow one.

        NOT a death signal: it is also False during a recoverable rebuild (the
        RealSense driver retries a USB stall up to _REBUILD_MAX_ATTEMPTS times
        and routinely comes back). Use `dead_reason` to decide whether to kill
        a worker or a session."""
        return True

    @property
    def dead_reason(self) -> str | None:
        """Non-None once the driver KNOWS its stream is unrecoverable IN THIS
        PROCESS — i.e. its own bounded self-healing has given up.

        This is the signal a poller/session may act on: `healthy` False is a
        stall the driver may still heal, `dead_reason` set never heals."""
        return None

    def last_frame_age(self) -> float:
        """Seconds since the last frame this driver actually produced (inf
        before the first). The planner sees the same age via the camera ring's
        newest timestamp."""
        t = getattr(self, "_last_frame_t", 0.0)
        return float("inf") if not t else time.perf_counter() - t


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
    # In the recording/deploy path the tactile sensors are opened by the
    # SensorSession's per-sensor WORKER PROCESSES (they need their own
    # process for the heavy SDK inference). A real DM-Tac device is single-
    # open, so the parent must NOT also open it here or the workers get
    # "device is occupied". Set True for those paths; leave False when the
    # parent reads tactile directly (tactile_monitor, tests).
    worker_owned_tactile: bool = False

    def connect_all(self) -> None:
        self.arm.connect(control=self.control)
        self.gripper.connect()
        # activation is part of bringing the rig up: the real URCap gripper
        # rejects move() until activated (activate() is a no-op when the
        # gripper is already active — GET STA returns 3 immediately)
        self.gripper.activate()
        if not self.worker_owned_tactile:
            for s in self.tactile.values():
                s.connect()
        for c in self.cameras.values():
            c.connect()

    def disconnect_all(self) -> None:
        for c in self.cameras.values():
            c.disconnect()
        if not self.worker_owned_tactile:
            for s in self.tactile.values():
                s.disconnect()
        self.gripper.disconnect()
        self.arm.disconnect()

    def __enter__(self) -> "Rig":
        self.connect_all()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect_all()
