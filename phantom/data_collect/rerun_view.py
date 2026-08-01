"""Mode-aware rerun live view for collect — isolated in a child process.

rerun-sdk 0.34's native layer (libarrow.so.2500) segfaults under a live
collect session: even with all rr.* calls serialized behind a process-wide
lock and the 10 Hz + back-off backpressure mitigations, the race between our
SDK calls and rerun's own Rust/server threads when the gRPC channel saturates
("Sender has been blocked for over 5 seconds…" precedes the crash) cannot be
fully fixed in-process. A segfault there took the whole data-collection server
down with it.

Fix: move ALL rerun activity into a dedicated spawn child process that owns
rr.init / serve and does every log. The child attaches to the SAME shared
memory rings by name (so the parent ships almost nothing) and polls them at
rate_hz itself; only event/episode markers cross an IPC queue. If the child
segfaults, the parent logs one warning, restarts it up to N times, then runs
without viz — the server / session / recording are provably unaffected.

Two layers live here:

  _CollectVizEngine   runs INSIDE the child: the mode-aware logger (the old
                      in-process CollectRerun body — subclasses the
                      hardware-proven RerunLogger for its _rr_lock thread
                      serialization and once-per-process server guard, which
                      are still correct within the child).
  CollectRerun        the parent-side proxy: same constructor/start/stop/
                      log_event surface session.py already wires, but
                      internally spawns/owns/supervises the child.

  full   everything PHANTOM needs: scene RGB, per-sensor field maps
         (depth / deformation / shear / fz), gel infer image, 6-axis wrench
         plots + contact area, arm joints / wrist F/T / TCP, gripper.
  lite   UR3 + RealSense RGB + per-sensor 6-axis wrench ONLY.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time

import numpy as np
import yaml

from phantom.config.hardware import HardwareConfig
from phantom.recording.ringbuffer import SharedRingBuffer
from phantom.recording.workers import _set_pdeathsig
from phantom.viz.rerun_logger import RerunLogger, _WRENCH

log = logging.getLogger(__name__)


class NativeViewer:
    """Launch the NATIVE (wgpu) rerun viewer window on the rig display and
    connect it to the live session's gRPC server.

    Measured on the rig NUC: the native viewer costs ~0.3 cores to render a
    640x480@15Hz stream vs ~3.5 cores for the browser WASM viewer — so when the
    operator views locally on the 4-core rig, native keeps CPU off the 125 Hz
    control loop while showing FULL resolution. It reads the same gRPC endpoint
    the web viewer would, so the operator's saved blueprint (.rbl, sent to the
    server by the child) applies here too. Never raises: a launch failure just
    leaves no native window (operator can fall back to the web viewer)."""

    def __init__(self, grpc_port: int, *, display: str = ":0"):
        # same proxy URI the browser uses; the native CLI connects to it.
        self.uri = f"rerun+http://127.0.0.1:{grpc_port}/proxy"
        self.display = display
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        # the panel is started headless (watchdog/ssh) so DISPLAY is usually
        # unset — point it at the rig's physical X display explicitly.
        env = dict(os.environ)
        env["DISPLAY"] = self.display
        env.setdefault("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "rerun", self.uri],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, start_new_session=True)
            log.info("native rerun viewer launched (pid=%s) on %s -> %s",
                     self._proc.pid, self.display, self.uri)
        except Exception:
            log.warning("native rerun viewer failed to launch — falling back "
                        "to the web viewer", exc_info=True)
            self._proc = None

    def stop(self) -> None:
        p = self._proc
        self._proc = None
        if p is None:
            return
        try:
            p.terminate()
            p.wait(2.0)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


def session_viewer_url(url: str | None, nonce: int) -> str:
    """Tag the viewer URL with a per-session nonce.

    The gRPC/web-viewer URL is byte-identical every session (fixed ports +
    fixed proxy URI), but every session spawns a FRESH child/server. The panel
    iframe only reloads when the published URL string changes, so an identical
    URL leaves the browser bound to the previous (now-dead) server — it hangs
    on the loading spinner and "rerun stops working" after an end/start cycle.
    Appending a unique nonce forces the iframe to reload and reconnect.
    """
    if not url:
        return ""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}_sid={nonce}"


# ===========================================================================
# child-side engine: the mode-aware logger (runs INSIDE the spawn child)
# ===========================================================================

class _CollectVizEngine(RerunLogger):
    """The actual rerun logger. Only ever instantiated in the child process.

    This is the former in-process CollectRerun body, unchanged: it owns
    rr.init/serve and its own 15 Hz tick thread, gated by collection mode.
    The blueprint is re-sent on every start() so a mode switch rearranges the
    viewer layout even though the rerun server (once per process) keeps
    running."""

    def __init__(self, hw, rings: dict, *, mode: str = "full",
                 rate_hz: float = 15.0, web_port: int = 9091, ws_port: int = 9878,
                 server_memory_limit: str = "512MB",
                 camera_jpeg_quality: int = 75,
                 blueprint_path: str | None = None,
                 camera_rate_hz: float | None = None,
                 camera_viz_downscale: int = 1):
        super().__init__(hw, rings, rate_hz=rate_hz, web_port=web_port,
                         ws_port=ws_port, server_memory_limit=server_memory_limit,
                         camera_jpeg_quality=camera_jpeg_quality,
                         blueprint_path=blueprint_path,
                         camera_rate_hz=camera_rate_hz,
                         camera_viz_downscale=camera_viz_downscale)
        assert mode in ("full", "lite"), mode
        self.mode = mode

    # re-send the (mode-specific) blueprint on every session start
    def start(self) -> None:
        super().start()
        if self._rr is not None:
            with RerunLogger._rr_lock:
                self._send_blueprint()

    def _send_blueprint(self) -> None:
        # the operator's saved .rbl (if any) wins over the mode-specific layout
        if self._load_saved_blueprint():
            return
        try:
            import rerun.blueprint as rrb
            rr = self._rr
            images = [rrb.Spatial2DView(origin="/camera/scene", name="scene")]
            plots = []
            if self.mode == "full":
                for s in self.hw.tactile.sensors:
                    images.append(rrb.Spatial2DView(
                        origin=f"/tactile/{s.name}/depth", name=f"{s.name} depth"))
                    images.append(rrb.Spatial2DView(
                        origin=f"/tactile/{s.name}/infer", name=f"{s.name} gel"))
            plots += [rrb.TimeSeriesView(origin=f"/tactile/{s.name}/wrench",
                                         name=f"{s.name} wrench (6-axis)")
                      for s in self.hw.tactile.sensors]
            plots += [rrb.TimeSeriesView(origin="/arm/ft", name="wrist F/T"),
                      rrb.TimeSeriesView(origin="/gripper", name="gripper")]
            rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
                rrb.Grid(*images), rrb.Vertical(*plots),
                rrb.TextLogView(origin="/events", name="events"),
                column_shares=[3, 2, 1])))
        except Exception:
            log.debug("rerun blueprint skipped (API mismatch)", exc_info=True)

    def _tick(self, ch: dict) -> None:
        if self.mode == "full":
            super()._tick(ch)
            return
        # ---- lite: camera + arm + gripper + tactile WRENCH only ----------
        got = self._fresh("camera_scene")
        if got:
            ts, d = got
            self._set_time(ts)
            self._camera_image("/camera/scene", d["color"][0])
        for s in self.hw.tactile.sensors:
            got = self._fresh(f"tactile_{s.name}")
            if got:
                ts, d = got
                self._set_time(ts)
                wrench = np.asarray(d["wrench"][0], dtype=np.float32).reshape(-1)
                for k, name in enumerate(_WRENCH[:wrench.size]):
                    self._scalar(f"/tactile/{s.name}/wrench/{name}", wrench[k])
        got = self._fresh("arm")
        if got:
            ts, d = got
            self._set_time(ts)
            for j in range(self.hw.arm.dof):
                self._scalar(f"/arm/q/j{j}", float(d["q"][0][j]))
            for k, name in enumerate(_WRENCH):
                self._scalar(f"/arm/ft/{name}", float(d["ft"][0][k]))
            tcp = d["tcp_pose"][0]
            for k, name in enumerate(("x", "y", "z")):
                self._scalar(f"/arm/tcp/{name}", float(tcp[k]))
        got = self._fresh("gripper")
        if got:
            ts, d = got
            self._set_time(ts)
            st = np.asarray(d["state"][0]).reshape(-1)
            self._scalar("/gripper/pos", float(st[0]))
            if st.size > 1:
                self._scalar("/gripper/obj", float(st[1]))


# ===========================================================================
# child entry point (spawn target)
# ===========================================================================

def _rerun_child_main(hw_yaml: str, ring_specs: dict[str, dict], mode: str,
                      rate_hz: float, web_port: int, ws_port: int,
                      server_memory_limit: str, camera_jpeg_quality: int,
                      blueprint_path: str | None, camera_rate_hz: float | None,
                      camera_viz_downscale: int,
                      event_q, result_q, stop_evt) -> None:  # type: ignore[valid-type]
    """Owns rr.init/serve + all logging. A crash here (segfault, exception)
    dies with this process and CANNOT touch the parent server.

    Attaches to the parent's rings by name and polls them itself, so the only
    thing crossing the IPC boundary is event/episode text on `event_q`. The
    engine's own daemon thread does the 15 Hz ring->rerun tick; this main
    thread only relays events (both serialized through RerunLogger._rr_lock,
    exactly as in-process)."""
    _set_pdeathsig()   # Linux: SIGKILL us if the parent dies hard
    # Viz is strictly SECONDARY to teleop/recording/tactile. On the 4-core NUC
    # the operator runs the browser viewer locally too, so the WASM decode +
    # this child + the 125 Hz control loop + two CPU-bound tactile trackers all
    # contend. Renice ourselves DOWN so the safety-critical control loop never
    # starves on our account — a laggy viewer is fine, a starved servo is not.
    try:
        os.nice(10)
    except Exception:
        pass
    engine = None
    rings: dict[str, SharedRingBuffer] = {}
    try:
        hw = HardwareConfig.model_validate(yaml.safe_load(hw_yaml))
        rings = {name: SharedRingBuffer.attach(spec)
                 for name, spec in ring_specs.items()}
        engine = _CollectVizEngine(hw, rings, mode=mode, rate_hz=rate_hz,
                                   web_port=web_port, ws_port=ws_port,
                                   server_memory_limit=server_memory_limit,
                                   camera_jpeg_quality=camera_jpeg_quality,
                                   blueprint_path=blueprint_path,
                                   camera_rate_hz=camera_rate_hz,
                                   camera_viz_downscale=camera_viz_downscale)
        engine.start()                      # rr.init + serve + tick thread
        result_q.put(("url", engine.viewer_url))
    except Exception as e:                  # rerun missing / server bring-up failed
        try:
            result_q.put(("error", repr(e)))
        except Exception:
            pass
        log.debug("rerun child failed to start", exc_info=True)
        return

    # relay episode / safety markers until told to stop (or the parent dies)
    try:
        while not stop_evt.is_set():
            if os.getppid() == 1:           # reparented to init -> parent gone
                break
            try:
                item = event_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:                # explicit shutdown sentinel
                break
            text, level = item
            engine.log_event(text, level=level)
    finally:
        try:
            engine.stop()
        except Exception:
            pass
        for r in rings.values():
            try:
                r.close()
            except Exception:
                pass


# ===========================================================================
# parent-side proxy: drop-in replacement for the old in-process CollectRerun
# ===========================================================================

class CollectRerun:
    """Thin parent-side proxy. Same surface session.py wires
    (constructor / start / stop / log_event / viewer_url) but every rerun call
    happens in a spawned child, so a rerun/libarrow segfault can never kill the
    collect server.

    NOTE: this module must never import `rerun` at parent scope — doing so
    would load libarrow into the server process, defeating the isolation.
    RerunLogger imports rerun lazily, so importing the engine class is safe."""

    _START_TIMEOUT = 15.0     # s to wait for the child to bring up the server
    _EVENT_QUEUE_MAX = 256    # events dropped (never block the record loop) above this
    _MAX_RESTARTS = 2         # per session, then run without viz

    def __init__(self, hw, rings: dict, *, mode: str = "full",
                 rate_hz: float = 15.0, web_port: int = 9091, ws_port: int = 9878,
                 server_memory_limit: str = "512MB",
                 camera_jpeg_quality: int = 75,
                 blueprint_path: str | None = None,
                 camera_rate_hz: float | None = None,
                 camera_viz_downscale: int = 1):
        # snapshot everything the child needs NOW (pickle-safe); the child
        # reconstructs hw from yaml and re-attaches the rings by name.
        self._hw_yaml = hw.snapshot_yaml()
        self._ring_specs = {name: r.spec_dict() for name, r in rings.items()}
        self.mode = mode
        self.rate_hz = rate_hz
        self.web_port = web_port
        self.ws_port = ws_port
        self.server_memory_limit = server_memory_limit
        self.camera_jpeg_quality = camera_jpeg_quality
        self.camera_rate_hz = camera_rate_hz
        self.camera_viz_downscale = camera_viz_downscale
        # resolve to absolute HERE (parent runs at repo-root cwd); the spawn
        # child may not share cwd. None if no path configured.
        self.blueprint_path = (os.path.abspath(blueprint_path)
                               if blueprint_path else None)

        self.viewer_url: str | None = None
        self._ctx = mp.get_context("spawn")
        self._child_target = _rerun_child_main    # seam: overridable in tests
        self._proc: mp.process.BaseProcess | None = None
        self._event_q = None
        self._stop_evt = None
        self._restarts = 0
        self._disabled = False    # restart cap hit -> permanent no-viz for the session
        self._death_logged = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._spawn()

    def _spawn(self) -> None:
        """(Re)spawn the child and wait for its viewer URL. Never raises — a
        failure just leaves viz disabled so the session is unaffected."""
        self._event_q = self._ctx.Queue(maxsize=self._EVENT_QUEUE_MAX)
        self._stop_evt = self._ctx.Event()
        result_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=self._child_target,
            args=(self._hw_yaml, self._ring_specs, self.mode, self.rate_hz,
                  self.web_port, self.ws_port, self.server_memory_limit,
                  self.camera_jpeg_quality, self.blueprint_path,
                  self.camera_rate_hz, self.camera_viz_downscale,
                  self._event_q, result_q, self._stop_evt),
            daemon=True, name="rerun-viz")
        self._proc.start()
        try:
            kind, payload = result_q.get(timeout=self._START_TIMEOUT)
        except queue.Empty:
            log.warning("rerun child did not come up in %.0fs — viz disabled",
                        self._START_TIMEOUT)
            self._kill()
            self.viewer_url = None
            return
        if kind == "url":
            self.viewer_url = payload
            log.info("rerun viewer (isolated child pid=%s) at %s",
                     self._proc.pid, self.viewer_url)
        else:
            log.warning("rerun disabled (child: %s)", payload)
            self._kill()
            self.viewer_url = None

    # ------------------------------------------------------------------
    def log_event(self, text: str, *, level: str = "INFO") -> None:
        """Non-blocking: drop the event rather than ever stall the record loop.

        Also the lazy liveness check — the session loop calls this at 10 Hz, so
        it doubles as the crash detector / restart trigger."""
        if self._disabled:
            return
        if self._proc is not None and not self._proc.is_alive():
            self._handle_child_death()
            if self._disabled:
                return
        q = self._event_q
        if q is None:
            return
        try:
            q.put_nowait((text, level))
        except queue.Full:
            log.debug("rerun event queue full — dropping event: %s", text)
        except (ValueError, OSError):
            pass    # queue closed during a concurrent restart/stop

    # ------------------------------------------------------------------
    def _handle_child_death(self) -> None:
        """Child died (segfault or exit). Log ONCE, restart up to _MAX_RESTARTS,
        then give up and run without viz. The server/session are untouched."""
        exit_code = self._proc.exitcode if self._proc is not None else None
        self._reap(terminate=False)
        if self._restarts >= self._MAX_RESTARTS:
            if not self._death_logged:
                log.warning("rerun child died (exit=%s) and hit the restart cap "
                            "(%d) — continuing WITHOUT viz", exit_code,
                            self._MAX_RESTARTS)
                self._death_logged = True
            self._disabled = True
            self.viewer_url = None
            return
        self._restarts += 1
        log.warning("rerun child died (exit=%s) — restarting (%d/%d)",
                    exit_code, self._restarts, self._MAX_RESTARTS)
        self._spawn()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if self._stop_evt is not None:
            try:
                self._stop_evt.set()
            except Exception:
                pass
        # nudge the child's event loop out of its blocking get()
        if self._event_q is not None:
            try:
                self._event_q.put_nowait(None)
            except Exception:
                pass
        self._reap(terminate=True)

    def _reap(self, *, terminate: bool) -> None:
        """Join the child; if terminate, escalate join -> terminate -> kill."""
        p = self._proc
        if p is None:
            return
        if terminate:
            p.join(2.0)
            if p.is_alive():
                p.terminate()
                p.join(1.0)
            if p.is_alive():
                p.kill()
                p.join(1.0)
        else:
            p.join(1.0)     # already-exited child: reap so it isn't a zombie
        self._proc = None
        self._close_queue()

    def _kill(self) -> None:
        p = self._proc
        if p is not None:
            if self._stop_evt is not None:
                try:
                    self._stop_evt.set()
                except Exception:
                    pass
            p.join(1.0)
            if p.is_alive():
                p.kill()
                p.join(1.0)
        self._proc = None
        self._close_queue()

    def _close_queue(self) -> None:
        q = self._event_q
        if q is not None:
            try:
                q.close()
            except Exception:
                pass
        self._event_q = None
        self._stop_evt = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()
