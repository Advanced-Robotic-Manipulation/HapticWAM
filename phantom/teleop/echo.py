"""Echo exoskeleton leader — the lab's proven joint-space teleop device.

Clean-room implementation of the serial protocol from the working stack
vendored at third_party/echo_teleop/ (see PROVENANCE.md): an STM32 device on
USB serial (VID/PID from config) answering the "r2" command with a 34-byte
frame:

    bytes  0..31   16 x int16  joint ticks (300 deg / 4096 ticks per unit;
                   indices 7 and 15 are the LEFT / RIGHT gripper — raw ticks,
                   NOT converted to radians)
    byte   32      uint8       sense_flag (sensitivity mode 0 / 1 / 2)
    byte   33      uint8       start_collection flag (LEVEL — held high while
                   the operator wants recording on)

Right-arm joints are ticks[8:14]; the joint-space target follows the lab's
control law: q_target = base_pose + offset_rad / sensitivity_divisor.

Semantics at the phantom boundary:
 - poll() returns a TeleopCommand with q_target set (joint-space leader) and
   dpose zeroed — the record loop drives servo_j and derives the recorded
   Δ-EE action from measured TCP poses;
 - the exo gripper ticks are normalized to 0..1 via the configured
   open/closed tick calibration (continuous, not the lab's binary mode);
 - the device start_collection LEVEL is edge-detected HERE and emitted as a
   one-shot buttons["start_stop"], matching what the record loop expects.

A background reader thread polls the device at ~100 Hz (its native teleop
rate) and caches the latest sample; poll() at action_rate_hz never blocks on
serial I/O. `pyserial` is imported lazily inside start() so this module
imports fine on machines without it (same pattern as drivers/real/dmtac.py).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from phantom.config.hardware import EchoTeleopConfig
from phantom.teleop.base import TeleopCommand, TeleopDevice
from phantom.teleop.filters import OneEuroFilter

# 300 degrees over 4096 ticks (device joint encoder resolution).
TICK_TO_RAD = np.deg2rad(300.0) / 4096.0

_FRAME_LEN = 34
_N_TICKS = 16
_RIGHT_ARM = slice(8, 14)     # 6 right-arm joints
_RIGHT_GRIPPER = 15           # raw ticks, stays un-converted


def parse_r2_frame(data: bytes) -> tuple[np.ndarray, int, bool]:
    """Parse one 34-byte "r2" response -> (ticks int16 (16,), sense_flag, start_flag)."""
    if len(data) != _FRAME_LEN:
        raise ValueError(f"echo r2 frame must be {_FRAME_LEN} bytes, got {len(data)}")
    ticks = np.frombuffer(data[:32], dtype=np.int16)
    sense_flag = data[32]
    start_flag = bool(data[33])
    return ticks, sense_flag, start_flag


class EchoTeleop(TeleopDevice):
    def __init__(self, cfg: EchoTeleopConfig):
        self.cfg = cfg
        self._port = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # signal conditioning, applied at device rate in the reader thread
        self._filter = OneEuroFilter(min_cutoff=cfg.filter_min_cutoff,
                                     beta=cfg.filter_beta,
                                     d_cutoff=cfg.filter_d_cutoff)
        # latest sample (under _lock)
        self._q_target = np.asarray(cfg.base_pose, dtype=np.float64).copy()
        self._q_target_v = np.zeros(6, dtype=np.float64)   # filtered joint velocity
        self._q_target_t = 0.0                             # perf_counter of the sample
        self._v_ema = np.zeros(6, dtype=np.float64)         # EMA'd output velocity
        self._gripper = 0.0
        self._start_level = False
        self._have_sample = False
        self._prev_start_level = False  # edge detector state (poll-side)

    # ------------------------------------------------------------------
    def start(self) -> None:
        try:
            import serial  # lazy: rig machines only
            import serial.tools.list_ports
        except ImportError as e:
            raise RuntimeError(
                "pyserial not installed — pip install 'phantom[hw]' on the rig "
                "machine, or use --teleop keyboard") from e
        device = None
        for p in serial.tools.list_ports.comports():
            if p.vid == self.cfg.vid and p.pid == self.cfg.pid:
                device = p.device
                break
        if device is None:
            raise RuntimeError(
                f"Echo device not found (USB VID {self.cfg.vid} / PID {self.cfg.pid}) "
                "— check the cable and teleop.echo in configs/hardware.yaml")
        self._port = serial.Serial(device, baudrate=self.cfg.baud, timeout=0.1)
        self._thread = threading.Thread(target=self._reader, daemon=True, name="echo-teleop")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
        port, self._port = self._port, None
        if port is not None:
            port.close()

    # ------------------------------------------------------------------
    def _gripper_01(self, tick: float) -> float:
        span = self.cfg.gripper_closed_tick - self.cfg.gripper_open_tick
        return float(np.clip((tick - self.cfg.gripper_open_tick) / span, 0.0, 1.0))

    def _reader(self) -> None:
        """~100 Hz: request a frame, parse, condition, cache the latest sample.

        Conditioning happens HERE at device rate (not at poll rate) so the
        one-euro filter sees every sample: encoder quantization (0.073 deg /
        tick) and operator tremor are removed before anything downstream —
        the high-rate JointServoStreamer and the 10 Hz poll() both read the
        same filtered signal."""
        base = np.asarray(self.cfg.base_pose, dtype=np.float64)
        divisors = self.cfg.sensitivity_divisors
        alpha = self.cfg.gripper_ema_alpha
        while not self._stop.is_set():
            try:
                self._port.reset_input_buffer()
                self._port.write(b"r2")
                data = self._port.read(_FRAME_LEN)
                ticks, sense_flag, start_flag = parse_r2_frame(data)
            except (ValueError, OSError):
                # The Echo firmware (DIY STM32) intermittently fails to answer
                # r2 and read() times out (~100 ms) about twice a second. Retry
                # IMMEDIATELY — the blocking read's own timeout paces the loop,
                # and the streamer extrapolates the target across the gap
                # (latest_target_stamped). The old fixed 50 ms wait only made
                # each drop-out longer (~150 ms target freeze -> visible jump).
                self._stop.wait(0.002)
                continue
            divisor = divisors[sense_flag] if sense_flag < len(divisors) else divisors[0]
            offset = ticks[_RIGHT_ARM].astype(np.float64) * TICK_TO_RAD
            q_raw = base + offset / divisor
            now = time.perf_counter()
            q_filt = self._filter.filter(q_raw, now)
            # velocity of the FILTERED OUTPUT (not the one-euro's internal _dx,
            # which is inflated by filter lag/dt): the true target speed, lightly
            # EMA'd, so the streamer can extrapolate it across a serial drop-out
            if self._have_sample and now > self._q_target_t:
                v_inst = (q_filt - self._q_target) / (now - self._q_target_t)
                self._v_ema += 0.2 * (v_inst - self._v_ema)
            grip_raw = self._gripper_01(float(ticks[_RIGHT_GRIPPER]))
            with self._lock:
                self._q_target = q_filt
                self._q_target_v = self._v_ema.copy()
                self._q_target_t = now
                self._gripper = (grip_raw if not self._have_sample
                                 else self._gripper + alpha * (grip_raw - self._gripper))
                self._start_level = start_flag
                self._have_sample = True
            # The "r2" protocol is request/response: reset_input_buffer + read()
            # self-paces the loop at the device's true update rate — measured
            # ~360 Hz on the rig (NOT the ~100 Hz the old docstring assumed;
            # verified by a raw-tick capture, 0.3% of consecutive polls repeat).
            # The old fixed 10 ms wait stacked on top of the round-trip, capping
            # the reader near ~50 Hz and adding a straight latency tax. Yield
            # only enough to stay interruptible; the blocking read paces us.
            # NB: at 360 Hz each tick step is a dt~3 ms velocity spike, so the
            # one-euro d_cutoff / beta must be moderate (see collect config) or
            # rest jitter is amplified into command buzz.
            self._stop.wait(0.001)

    # ------------------------------------------------------------------
    def latest_q_target(self) -> np.ndarray | None:
        """Thread-safe filtered joint target for the high-rate streamer
        (None until the first device sample)."""
        with self._lock:
            return self._q_target.copy() if self._have_sample else None

    def latest_target_stamped(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(filtered q_target, filtered joint velocity, perf_counter timestamp)
        of the most recent device sample, or None before the first one. The
        streamer uses the timestamp to detect leader drop-outs and the velocity
        to extrapolate the target across them (no freeze-then-jump)."""
        with self._lock:
            if not self._have_sample:
                return None
            return self._q_target.copy(), self._q_target_v.copy(), self._q_target_t

    def poll(self) -> TeleopCommand:
        with self._lock:
            q_target = self._q_target.copy()
            gripper = self._gripper
            level = self._start_level
            have = self._have_sample
        buttons: dict[str, bool] = {}
        # device flag is a LEVEL; the record loop expects a one-shot toggle
        if level != self._prev_start_level:
            self._prev_start_level = level
            buttons["start_stop"] = True
        if not have:
            # no device sample yet: hold at base pose, don't move the gripper
            return TeleopCommand(dpose=np.zeros(6), gripper=0.0, buttons=buttons,
                                 q_target=None)
        return TeleopCommand(dpose=np.zeros(6), gripper=gripper, buttons=buttons,
                             q_target=q_target)
