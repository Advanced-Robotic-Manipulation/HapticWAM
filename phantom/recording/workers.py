"""Stream workers: per-device polling loops feeding shared ring buffers.

Process/thread split (throughput-driven):
 - each tactile sensor runs in its OWN spawn-safe process (2x 120 Hz full-res
   grabs + downsample + f16 cast is real CPU work, and the dmrobotics SDK's
   GIL/multi-instance behavior is unknown). The child constructs its driver
   itself from the pickled hardware config — mandatory for Windows spawn.
   Full-res frames never cross the IPC boundary except decimated keyframes.
 - REAL arm state polling also runs in its own process (ArmStateWorker /
   _arm_main below), for the same reason as tactile: measured 2026-07-31, the
   in-parent ThreadPoller shared the GIL with the camera poller, the
   recorder's drain thread and the panel HTTP server, and periodically
   stalled 20-40 ms (~9 missed 125 Hz samples/episode). It opens its OWN
   RTDEReceiveInterface — UR's RTDE protocol is designed for multiple
   simultaneous client connections (the same one a live external monitor can
   use alongside the control application), so this does not contend with the
   RTDEControlInterface the streamer drives servoJ through, nor with the
   parent's own receive interface (rig.arm._recv, used only for occasional
   is_ready_for_control()/is_protective_stopped() checks). MOCK arm keeps the
   old in-parent ThreadPoller — gaps are a real-hardware/GIL artifact that
   does not matter for the deterministic mock, and it keeps mock-mode tests
   unchanged.
 - gripper state recording is NOT a separate poller at all any more: it was
   merged into GripperPilot's own loop (phantom/data_collect/gripper.py) —
   see that module's docstring. The old ThreadPoller raced GripperPilot's
   move() calls for RobotiqGripper's single TCP socket/lock, which was the
   actual cause of its stalls (not GIL contention).
 - camera runs as a thread in the parent (its C extension releases the GIL
   and the data volume is trivial).

All rings store t_host (perf_counter — one machine-wide clock on both Windows
QPC and Linux CLOCK_MONOTONIC, so parent + children agree); the recorder maps
to t_master via MasterClock when draining.

Mock determinism across processes: children receive session_t0 and evaluate
their ContactScenario at (perf_counter() - session_t0), so mock tactile
processes stay in sync with the parent's mock arm/camera.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import yaml

from phantom.config.hardware import HardwareConfig
from phantom.recording.ringbuffer import SharedRingBuffer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ring construction (parent side, owns the shm)
# ---------------------------------------------------------------------------

def build_session_rings(hw: HardwareConfig, session_id: str) -> dict[str, SharedRingBuffer]:
    r, t = hw.recording, hw.tactile
    f_dtype = "float16" if r.field_dtype == "float16" else "float32"
    rings: dict[str, SharedRingBuffer] = {}

    def make(name: str, rate: float, fields: dict) -> None:
        cap = max(16, int(r.ring_seconds * rate))
        # shm segment names must be short (macOS 31-char limit) AND unique —
        # hash the full logical name instead of truncating it
        shm_name = "ph" + hashlib.md5(f"{session_id}_{name}".encode()).hexdigest()[:22]
        rings[name] = SharedRingBuffer(shm_name, cap, fields, create=True)

    for s in t.sensors:
        make(f"tactile_{s.name}", t.rate_hz, {
            "fields_ds": ((r.field_ds.h, r.field_ds.w, t.field_ch), f_dtype),
            "wrench": ((t.wrench_dim,), "float32"),
            "area": ((), "float32"),
        })
        make(f"tactile_{s.name}_kf", r.keyframe_rate_hz, {
            "keyframe": ((r.keyframe_ds.h, r.keyframe_ds.w, t.field_ch), f_dtype),
        })
        if r.save_infer_img:
            img_shape = ((t.infer_img.h, t.infer_img.w) if t.infer_img.c == 1
                         else t.infer_img.hwc)
            make(f"tactile_{s.name}_img", r.infer_img_rate_hz,
                 {"infer_img": (img_shape, "uint8")})
        if r.archive_raw_img:
            raw_shape = ((t.raw_img.h, t.raw_img.w) if t.raw_img.c == 1
                         else t.raw_img.hwc)
            make(f"tactile_{s.name}_raw", t.rate_hz,
                 {"raw_img": (raw_shape, "uint8")})
    make("arm", hw.arm.rtde_receive_hz, {
        "q": ((hw.arm.dof,), "float64"), "qd": ((hw.arm.dof,), "float64"),
        "tcp_pose": ((6,), "float64"), "tcp_speed": ((6,), "float64"),
        "ft": ((6,), "float64"), "t_rtde": ((), "float64"),
        "protective_stop": ((), "uint8"),
    })
    # (2,) = [position, obj] — see GripperState in phantom/drivers/base.py for
    # why channel 1 is the gOBJ status and not motor current.
    make("gripper", hw.gripper.feedback_rate_hz, {"state": ((2,), "float32")})
    for cam_name, cam in (("scene", hw.cameras.scene), ("wrist", hw.cameras.wrist)):
        if cam.enabled:
            make(f"camera_{cam_name}", cam.fps, {"color": (cam.color.hwc, "uint8")})
    return rings


# ---------------------------------------------------------------------------
# tactile worker (child process)
# ---------------------------------------------------------------------------

def _set_pdeathsig() -> None:
    """Linux: ask the kernel to SIGKILL this worker if its parent dies — so a
    hard-killed parent (kill -9, crash, OOM) can't orphan a worker that still
    holds the single-open tactile device and blocks the next session. The
    daemon=True flag only covers CLEAN parent exit (atexit); this covers the
    rest."""
    if sys.platform != "linux":
        return
    try:
        import ctypes
        import signal
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


def _tactile_main(hw_yaml: str, sensor_name: str, ring_specs: dict[str, dict],
                  session_t0: float, stop: mp.Event, finger_index: int) -> None:  # type: ignore[valid-type]
    _set_pdeathsig()
    hw = HardwareConfig.model_validate(yaml.safe_load(hw_yaml))
    t, r = hw.tactile, hw.recording
    sensor_cfg = next(s for s in t.sensors if s.name == sensor_name)

    if hw.mode.resolve("tactile") == "mock":
        from phantom.drivers.mock.dmtac import MockTactileSensor
        from phantom.drivers.mock.scenario import ContactScenario
        driver = MockTactileSensor(t, sensor_cfg, ContactScenario(seed=0),
                                   finger_index=finger_index)
        driver.connect()
        driver._t0 = session_t0  # align scenario time with the parent's mocks
    else:
        from phantom.drivers.real.dmtac import DmTacSensor
        driver = DmTacSensor(t, sensor_cfg)
        # Field-debugged 2026-08-13 (waffles deploy): both tactile workers
        # spawn together, and two SIMULTANEOUS SDK opens power-cycle the
        # second sensor off the USB bus (repeatable; solo opens and a
        # 5s-staggered dual open are clean at the full 120 Hz, incl. on a
        # shared hub). Serialize the connect window by finger index, with
        # one retry as belt-and-braces.
        time.sleep(finger_index * 8.0)
        try:
            driver.connect()
        except Exception:
            log.warning("tactile[%s] connect failed — retrying once in 5s",
                        sensor_name)
            time.sleep(5.0)
            driver.connect()

    ring = SharedRingBuffer.attach(ring_specs[f"tactile_{sensor_name}"])
    ring_kf = SharedRingBuffer.attach(ring_specs[f"tactile_{sensor_name}_kf"])
    ring_img = (SharedRingBuffer.attach(ring_specs[f"tactile_{sensor_name}_img"])
                if r.save_infer_img else None)
    ring_raw = (SharedRingBuffer.attach(ring_specs[f"tactile_{sensor_name}_raw"])
                if r.archive_raw_img else None)

    ds_h, ds_w = r.field_ds.hw
    fh, fw = t.field.h // ds_h, t.field.w // ds_w
    kf_ds_h, kf_ds_w = r.keyframe_ds.hw
    kf_fh, kf_fw = t.field.h // kf_ds_h, t.field.w // kf_ds_w
    f_dtype = np.float16 if r.field_dtype == "float16" else np.float32
    # Decimate by TIME, not by frame index. This loop free-runs at whatever
    # the SDK actually delivers, which is NOT tactile.rate_hz (measured ~6 Hz
    # against a declared 10). Dividing the DECLARED rate therefore pushed
    # keyframes/gel images below their configured rate (3.0 Hz vs 5.0 asked).
    # Wall-clock gating hits the configured rate for any source rate above it.
    kf_period = 1.0 / r.keyframe_rate_hz
    img_period = 1.0 / r.infer_img_rate_hz
    next_kf = next_img = 0.0
    import os
    try:
        while not stop.is_set():
            # orphan watchdog: if the parent died hard (segfault, kill -9) we
            # get reparented to init — exit and release the single-open device.
            # (pdeathsig is armed too, but this check is unconditional.)
            if os.getppid() == 1:
                log.warning("tactile[%s] parent died — exiting", sensor_name)
                break
            frame = driver.read()
            stack = frame.field_stack()                    # (H, W, 8) f32
            ds = stack.reshape(ds_h, fh, ds_w, fw, stack.shape[-1]).mean(axis=(1, 3))
            ring.push(frame.t_host, fields_ds=ds.astype(f_dtype),
                      wrench=frame.wrench.astype(np.float32),
                      area=np.float32(frame.contact_area_mm2))
            if frame.t_host >= next_kf:
                # mean-pool the native field stack down to keyframe_ds — the
                # encoder crushes native res through several stride-2 stages
                # anyway (see model/hht/tactile_encoder.py), so nothing
                # downstream loses information here (recording.keyframe_ds).
                kf_ds = stack.reshape(kf_ds_h, kf_fh, kf_ds_w, kf_fw,
                                      stack.shape[-1]).mean(axis=(1, 3))
                ring_kf.push(frame.t_host, keyframe=kf_ds.astype(f_dtype))
                next_kf = frame.t_host + kf_period
            if (ring_img is not None and frame.infer_img is not None
                    and frame.t_host >= next_img):
                ring_img.push(frame.t_host, infer_img=frame.infer_img)
                next_img = frame.t_host + img_period
            if ring_raw is not None:
                # raw grayscale at full rate -> offline field recompute
                # (docs/sensor_sdk.md; enable after bench item (e))
                ring_raw.push(frame.t_host, raw_img=driver.read_raw_img())
    finally:
        driver.disconnect()
        ring.close(); ring_kf.close()
        if ring_raw is not None:
            ring_raw.close()
        if ring_img is not None:
            ring_img.close()


def _arm_main(hw_yaml: str, ring_spec: dict, stop: mp.Event) -> None:  # type: ignore[valid-type]
    """Dedicated process for REAL-arm state polling -> the `arm` ring. See the
    module docstring for why this is a process rather than the ThreadPoller
    mock arm still uses. Opens its own RTDEReceiveInterface (independent of
    the parent's rig.arm._recv and of the RTDEControlInterface the streamer
    drives servoJ through) and fails LOUD on any RTDE error, exactly like
    URArm.get_state() — a dead worker aborts the in-progress episode
    (SensorSession.all_alive() -> "worker_died") rather than silently
    corrupting or gapping the stream."""
    _set_pdeathsig()
    hw = HardwareConfig.model_validate(yaml.safe_load(hw_yaml))
    try:
        import rtde_receive
    except ImportError as e:
        raise RuntimeError("ur_rtde not installed — pip install ur-rtde") from e
    recv = rtde_receive.RTDEReceiveInterface(
        hw.arm.ip, frequency=hw.arm.rtde_receive_hz)
    ring = SharedRingBuffer.attach(ring_spec)
    period = 1.0 / hw.arm.rtde_receive_hz
    next_t = time.perf_counter()
    import os
    try:
        while not stop.is_set():
            if os.getppid() == 1:
                log.warning("arm state worker: parent died — exiting")
                break
            if not recv.isConnected():
                raise RuntimeError(
                    "RTDE receive stream lost — robot rebooted or network dropped")
            t_host = time.perf_counter()
            ring.push(
                t_host,
                q=np.asarray(recv.getActualQ(), dtype=np.float64),
                qd=np.asarray(recv.getActualQd(), dtype=np.float64),
                tcp_pose=np.asarray(recv.getActualTCPPose(), dtype=np.float64),
                tcp_speed=np.asarray(recv.getActualTCPSpeed(), dtype=np.float64),
                ft=np.asarray(recv.getActualTCPForce(), dtype=np.float64),
                t_rtde=np.float64(recv.getTimestamp()),
                protective_stop=np.uint8(recv.isProtectiveStopped()))
            next_t += period
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.perf_counter()   # fell behind; don't burst
    finally:
        try:
            recv.disconnect()
        except Exception:
            pass
        ring.close()


class ArmStateWorker:
    def __init__(self, hw: HardwareConfig, ring: SharedRingBuffer):
        ctx = mp.get_context("spawn")
        self._stop = ctx.Event()
        self._proc = ctx.Process(
            target=_arm_main, args=(hw.snapshot_yaml(), ring.spec_dict(), self._stop),
            daemon=True, name="arm-state")

    def start(self) -> None:
        self._proc.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._proc.join(timeout)
        if self._proc.is_alive():
            log.warning("arm state worker did not exit; terminating")
            self._proc.terminate()

    def is_alive(self) -> bool:
        return self._proc.is_alive()


class TactileWorker:
    def __init__(self, hw: HardwareConfig, sensor_name: str, finger_index: int,
                 rings: dict[str, SharedRingBuffer], session_t0: float):
        self.sensor_name = sensor_name
        ctx = mp.get_context("spawn")
        self._stop = ctx.Event()
        specs = {k: v.spec_dict() for k, v in rings.items()
                 if k.startswith(f"tactile_{sensor_name}")}
        self._proc = ctx.Process(
            target=_tactile_main,
            args=(hw.snapshot_yaml(), sensor_name, specs, session_t0, self._stop,
                  finger_index),
            daemon=True, name=f"tactile-{sensor_name}")

    def start(self) -> None:
        self._proc.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._proc.join(timeout)
        if self._proc.is_alive():
            log.warning("tactile worker %s did not exit; terminating", self.sensor_name)
            self._proc.terminate()

    def is_alive(self) -> bool:
        return self._proc.is_alive()


# ---------------------------------------------------------------------------
# thread pollers (parent process)
# ---------------------------------------------------------------------------

class ThreadPoller:
    def __init__(self, name: str, rate_hz: float, poll_fn, ring: SharedRingBuffer,
                 *, dead_fn=None):
        self.name = name
        self.rate_hz = rate_hz
        self.poll_fn = poll_fn
        self.ring = ring
        # Optional device-death probe: returns a reason string once the DRIVER
        # knows it will never produce another sample in this process (bounded
        # self-healing exhausted), None otherwise. The poller's retry loop is
        # unbounded by design — a driver that heals itself must not lose its
        # worker over a stall — so a permanently wedged device would otherwise
        # retry forever against a ring nobody writes, and all_alive() (which
        # only sees thread liveness) would keep reporting the session healthy.
        self.dead_fn = dead_fn
        self.dead_reason: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"poll-{name}")

    def _check_dead(self) -> bool:
        if self.dead_fn is None:
            return False
        reason = self.dead_fn()
        if not reason:
            return False
        self.dead_reason = str(reason)
        log.error("poller %s: the device is DEAD (%s) — exiting the worker "
                  "thread. Its ring now has no writer, so the session is over: "
                  "the episode stops as `worker_died` and run_deploy refuses to "
                  "start another one in this process.", self.name, self.dead_reason)
        return True

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        next_t = time.perf_counter()
        while not self._stop.is_set():
            if self._check_dead():
                return
            try:
                t_host, values = self.poll_fn()
                self.ring.push(t_host, **values)
            except Exception:
                if self._check_dead():
                    return
                log.exception("poller %s failed; retrying", self.name)
                time.sleep(0.1)
            next_t += period
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.perf_counter()   # fell behind; don't burst

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def make_arm_poller(arm, hw: HardwareConfig, ring: SharedRingBuffer) -> ThreadPoller:
    """MOCK arm only in normal use (see ArmStateWorker for real hardware) —
    kept generic (takes any `arm` with get_state()) so tests can still drive
    it directly against a fake/mock arm."""
    def poll():
        st = arm.get_state()
        return st.t_host, dict(
            q=st.q, qd=st.qd, tcp_pose=st.tcp_pose, tcp_speed=st.tcp_speed,
            ft=st.ft, t_rtde=np.float64(st.t_rtde),
            protective_stop=np.uint8(st.protective_stop))
    return ThreadPoller("arm", hw.arm.rtde_receive_hz, poll, ring)


def make_gripper_poller(gripper, hw: HardwareConfig, ring: SharedRingBuffer) -> ThreadPoller:
    """NOT used by SensorSession.start() any more — the collect.py/session.py
    path folds this into GripperPilot's own loop instead (see that module's
    docstring: a separate poller thread here raced GripperPilot's move() calls
    for RobotiqGripper's one TCP socket/lock). Kept as a standalone helper for
    simpler callers with no GripperPilot-equivalent of their own (e.g.
    scripts/record_episodes.py, which sends gripper.move() straight from its
    single teleop loop and has no second thread to race)."""
    def poll():
        st = gripper.get_state()
        return st.t_host, dict(state=np.array([st.position, st.obj], dtype=np.float32))
    return ThreadPoller("gripper", hw.gripper.feedback_rate_hz, poll, ring)


def make_camera_poller(camera, name: str, fps: float, ring: SharedRingBuffer) -> ThreadPoller:
    def poll():
        f = camera.read()   # blocking read paces itself; poller rate is a backstop
        return f.t_host, dict(color=f.color)

    # `dead_reason`, not `healthy`: healthy is also False during the driver's
    # own bounded pipeline rebuild, which usually SUCCEEDS (a USB stall on the
    # DmTac hub). Killing the worker there would end a rig session over a
    # recoverable stall. dead_reason is set only once the rebuild gave up.
    return ThreadPoller(f"camera_{name}", fps * 2, poll, ring,
                        dead_fn=lambda: getattr(camera, "dead_reason", None))


# ---------------------------------------------------------------------------
# session: everything wired together
# ---------------------------------------------------------------------------

@dataclass
class SensorSession:
    """Owns rings + workers for one recording/deployment session.

    The parent keeps the shm owner handles alive (Windows frees shared memory
    when the last handle closes)."""
    hw: HardwareConfig
    rings: dict[str, SharedRingBuffer]
    tactile_workers: list[TactileWorker]
    pollers: list[ThreadPoller]
    # None in mock-arm mode (where "arm" is still filled by a ThreadPoller in
    # `pollers`, via make_arm_poller) — see module docstring.
    arm_worker: "ArmStateWorker | None" = None

    @classmethod
    def start(cls, hw: HardwareConfig, rig, session_id: str) -> "SensorSession":
        rings = build_session_rings(hw, session_id)
        session_t0 = getattr(rig.arm, "_t0", time.perf_counter())
        workers = [TactileWorker(hw, s.name, i, rings, session_t0)
                   for i, s in enumerate(hw.tactile.sensors)]
        pollers = []
        arm_worker = None
        if hw.mode.resolve("arm") == "mock":
            pollers.append(make_arm_poller(rig.arm, hw, rings["arm"]))
        else:
            arm_worker = ArmStateWorker(hw, rings["arm"])
        # NOTE: no gripper poller here any more — gripper state recording now
        # happens inside GripperPilot's own loop (phantom/data_collect/
        # gripper.py), which owns the ring reference directly.
        for name, cam in rig.cameras.items():
            pollers.append(make_camera_poller(cam, name, cam.cfg.fps,
                                              rings[f"camera_{name}"]))
        for w in workers:
            w.start()
        if arm_worker is not None:
            arm_worker.start()
        for p in pollers:
            p.start()
        return cls(hw=hw, rings=rings, tactile_workers=workers, pollers=pollers,
                   arm_worker=arm_worker)

    def all_alive(self) -> bool:
        return (all(w.is_alive() for w in self.tactile_workers)
                and all(p.is_alive() for p in self.pollers)
                and (self.arm_worker is None or self.arm_worker.is_alive()))

    def stop(self) -> None:
        for p in self.pollers:
            p.stop()
        for w in self.tactile_workers:
            w.stop()
        if self.arm_worker is not None:
            self.arm_worker.stop()
        for ring in self.rings.values():
            ring.close()
