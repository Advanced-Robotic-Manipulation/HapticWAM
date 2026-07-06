"""Stream workers: per-device polling loops feeding shared ring buffers.

Process/thread split (throughput-driven):
 - each tactile sensor runs in its OWN spawn-safe process (2x 120 Hz full-res
   grabs + downsample + f16 cast is real CPU work, and the dmrobotics SDK's
   GIL/multi-instance behavior is unknown). The child constructs its driver
   itself from the pickled hardware config — mandatory for Windows spawn.
   Full-res frames never cross the IPC boundary except decimated keyframes.
 - arm / gripper / camera run as threads in the parent (their C extensions
   release the GIL and the data volume is trivial).

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
            "keyframe": ((t.field.h, t.field.w, t.field_ch), f_dtype),
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
    make("gripper", hw.gripper.feedback_rate_hz, {"state": ((2,), "float32")})
    for cam_name, cam in (("scene", hw.cameras.scene), ("wrist", hw.cameras.wrist)):
        if cam.enabled:
            make(f"camera_{cam_name}", cam.fps, {"color": (cam.color.hwc, "uint8")})
    return rings


# ---------------------------------------------------------------------------
# tactile worker (child process)
# ---------------------------------------------------------------------------

def _tactile_main(hw_yaml: str, sensor_name: str, ring_specs: dict[str, dict],
                  session_t0: float, stop: mp.Event, finger_index: int) -> None:  # type: ignore[valid-type]
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
        driver.connect()

    ring = SharedRingBuffer.attach(ring_specs[f"tactile_{sensor_name}"])
    ring_kf = SharedRingBuffer.attach(ring_specs[f"tactile_{sensor_name}_kf"])
    ring_img = (SharedRingBuffer.attach(ring_specs[f"tactile_{sensor_name}_img"])
                if r.save_infer_img else None)
    ring_raw = (SharedRingBuffer.attach(ring_specs[f"tactile_{sensor_name}_raw"])
                if r.archive_raw_img else None)

    ds_h, ds_w = r.field_ds.hw
    fh, fw = t.field.h // ds_h, t.field.w // ds_w
    f_dtype = np.float16 if r.field_dtype == "float16" else np.float32
    kf_every = max(1, round(t.rate_hz / r.keyframe_rate_hz))
    img_every = max(1, round(t.rate_hz / r.infer_img_rate_hz))
    i = 0
    try:
        while not stop.is_set():
            frame = driver.read()
            stack = frame.field_stack()                    # (H, W, 8) f32
            ds = stack.reshape(ds_h, fh, ds_w, fw, stack.shape[-1]).mean(axis=(1, 3))
            ring.push(frame.t_host, fields_ds=ds.astype(f_dtype),
                      wrench=frame.wrench.astype(np.float32),
                      area=np.float32(frame.contact_area_mm2))
            if i % kf_every == 0:
                ring_kf.push(frame.t_host, keyframe=stack.astype(f_dtype))
            if ring_img is not None and frame.infer_img is not None and i % img_every == 0:
                ring_img.push(frame.t_host, infer_img=frame.infer_img)
            if ring_raw is not None:
                # raw grayscale at full rate -> offline field recompute
                # (docs/sensor_sdk.md; enable after bench item (e))
                ring_raw.push(frame.t_host, raw_img=driver.read_raw_img())
            i += 1
    finally:
        driver.disconnect()
        ring.close(); ring_kf.close()
        if ring_raw is not None:
            ring_raw.close()
        if ring_img is not None:
            ring_img.close()


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
    def __init__(self, name: str, rate_hz: float, poll_fn, ring: SharedRingBuffer):
        self.name = name
        self.rate_hz = rate_hz
        self.poll_fn = poll_fn
        self.ring = ring
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"poll-{name}")

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                t_host, values = self.poll_fn()
                self.ring.push(t_host, **values)
            except Exception:
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
    def poll():
        st = arm.get_state()
        return st.t_host, dict(
            q=st.q, qd=st.qd, tcp_pose=st.tcp_pose, tcp_speed=st.tcp_speed,
            ft=st.ft, t_rtde=np.float64(st.t_rtde),
            protective_stop=np.uint8(st.protective_stop))
    return ThreadPoller("arm", hw.arm.rtde_receive_hz, poll, ring)


def make_gripper_poller(gripper, hw: HardwareConfig, ring: SharedRingBuffer) -> ThreadPoller:
    def poll():
        st = gripper.get_state()
        return st.t_host, dict(state=np.array([st.position, st.current], dtype=np.float32))
    return ThreadPoller("gripper", hw.gripper.feedback_rate_hz, poll, ring)


def make_camera_poller(camera, name: str, fps: float, ring: SharedRingBuffer) -> ThreadPoller:
    def poll():
        f = camera.read()   # blocking read paces itself; poller rate is a backstop
        return f.t_host, dict(color=f.color)
    return ThreadPoller(f"camera_{name}", fps * 2, poll, ring)


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

    @classmethod
    def start(cls, hw: HardwareConfig, rig, session_id: str) -> "SensorSession":
        rings = build_session_rings(hw, session_id)
        session_t0 = getattr(rig.arm, "_t0", time.perf_counter())
        workers = [TactileWorker(hw, s.name, i, rings, session_t0)
                   for i, s in enumerate(hw.tactile.sensors)]
        pollers = [make_arm_poller(rig.arm, hw, rings["arm"]),
                   make_gripper_poller(rig.gripper, hw, rings["gripper"])]
        for name, cam in rig.cameras.items():
            pollers.append(make_camera_poller(cam, name, cam.cfg.fps,
                                              rings[f"camera_{name}"]))
        for w in workers:
            w.start()
        for p in pollers:
            p.start()
        return cls(hw=hw, rings=rings, tactile_workers=workers, pollers=pollers)

    def all_alive(self) -> bool:
        return (all(w.is_alive() for w in self.tactile_workers)
                and all(p.is_alive() for p in self.pollers))

    def stop(self) -> None:
        for p in self.pollers:
            p.stop()
        for w in self.tactile_workers:
            w.stop()
        for ring in self.rings.values():
            ring.close()
