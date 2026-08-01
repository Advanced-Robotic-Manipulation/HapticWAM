"""Live Rerun visualization of the whole rig — every session ring stream.

Reads the SAME shared-memory rings the recorder drains (zero extra load on
the drivers) and logs to a rerun.io web viewer at ~15 Hz:

    /camera/scene                       RGB frames
    /tactile/<name>/depth|deform|shear|fz   field-channel images (ds res)
    /tactile/<name>/infer               sensing-area image
    /tactile/<name>/wrench/Fx..Mz       6 scalar plots  + /area
    /arm/q/j0..j5  /arm/ft/Fx..Mz      joint + wrist-wrench plots
    /arm/tcp/x|y|z                      TCP position plots
    /gripper/pos|obj                    gripper plots
    /events                             episode + safety text log

`rerun-sdk` is optional (`pip install 'phantom[viz]'`) and imported lazily;
API calls are wrapped defensively (helpers fall back across SDK versions).
The web viewer URL (`viewer_url`) is what the control panel embeds.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

from phantom.config.hardware import HardwareConfig

log = logging.getLogger(__name__)

_WRENCH = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


class RerunLogger:
    # The rerun server (rr.init + gRPC + web viewer) can only be brought up
    # ONCE per process — later sessions in the same panel process reuse it
    # (same recording; the timeline simply continues).
    _server_url: str | None = None
    # rr.log from two threads (the 15 Hz tick thread + log_event from the
    # record loop) segfaulted inside pyarrow on a real rig (libarrow.so.2500,
    # rerun 0.34) — serialize every SDK call through one process-wide lock.
    _rr_lock = threading.Lock()

    def __init__(self, hw: HardwareConfig, rings: dict, *, rate_hz: float = 15.0,
                 web_port: int = 9090, ws_port: int = 9877,
                 server_memory_limit: str = "512MB",
                 camera_jpeg_quality: int = 75,
                 blueprint_path: str | None = None,
                 camera_rate_hz: float | None = None,
                 camera_viz_downscale: int = 1):
        self.hw = hw
        self.rings = rings
        self.rate_hz = rate_hz
        # Camera poll rate. None = fall back to rate_hz (unchanged behavior for
        # the standalone logger). The collect engine passes 30 so the scene is
        # logged at its native fps; the loop runs at the faster of the two and
        # every stream self-decimates via _fresh (only logs on a new sample).
        self.camera_rate_hz = camera_rate_hz
        self.web_port = web_port
        self.ws_port = ws_port
        # cap the server-side history buffer (see RerunConfig) so neither the
        # gRPC server nor the browser viewer grow without bound -> lag/hang.
        self.server_memory_limit = server_memory_limit
        # 0 = raw camera; >0 = JPEG quality for /camera/scene (bandwidth cap).
        self.camera_jpeg_quality = int(camera_jpeg_quality)
        # integer factor to shrink the scene before logging (>=1; 1 = full res).
        # Used as a quality/CPU stopgap for the browser (WASM) viewer path; the
        # native viewer handles full res, so prefer 1 there.
        self.camera_viz_downscale = max(1, int(camera_viz_downscale))
        # cache cv2 once (present in the rig venv) for anti-aliased downscaling;
        # None -> fall back to plain decimation.
        try:
            import cv2 as _cv2
            self._cv2 = _cv2
        except Exception:
            self._cv2 = None
        # optional .rbl exported from the viewer ("Save blueprint"); loaded and
        # activated on start when the file exists, overriding the code layout.
        self.blueprint_path = blueprint_path
        self.viewer_url: str | None = None
        self._rr = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_ts: dict[str, float] = {}   # per-ring: skip unchanged samples

    # ------------------------------------------------------------------
    def start(self) -> None:
        try:
            import rerun as rr  # lazy: [viz] extra
        except ImportError as e:
            raise RuntimeError(
                "rerun-sdk not installed — pip install 'phantom[viz]'") from e
        self._rr = rr
        if RerunLogger._server_url is None:
            rr.init(f"phantom/{self.hw.meta.rig_name}", spawn=False)
            RerunLogger._server_url = self._serve_web()
            self._send_blueprint()
        self.viewer_url = RerunLogger._server_url
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="rerun-viz")
        self._thread.start()
        log.info("rerun viewer at %s", self.viewer_url)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None

    def _serve_web(self) -> str:
        rr = self._rr
        if hasattr(rr, "serve_grpc"):
            # modern API (verified on 0.34): gRPC stream + separate web viewer.
            # CORS "*" lets the control panel (another local port) iframe it.
            # server_memory_limit caps the buffer so old data is dropped and
            # the viewer stays realtime instead of drowning in history.
            from urllib.parse import quote
            kw = {"grpc_port": self.ws_port, "cors_allow_origin": ["*"]}
            if self.server_memory_limit:
                kw["server_memory_limit"] = self.server_memory_limit
            try:
                uri = rr.serve_grpc(**kw)
            except TypeError:
                # older 0.34 point release without one of the kwargs — retry
                # dropping the optional ones, then bare, so viz still comes up.
                kw.pop("server_memory_limit", None)
                try:
                    uri = rr.serve_grpc(**kw)
                except TypeError:
                    uri = rr.serve_grpc(grpc_port=self.ws_port)
            rr.serve_web_viewer(web_port=self.web_port, open_browser=False,
                                connect_to=uri)
            return f"http://localhost:{self.web_port}/?url={quote(uri, safe='')}"
        # legacy API (< 0.24)
        rr.serve_web(open_browser=False, web_port=self.web_port, ws_port=self.ws_port)
        return f"http://localhost:{self.web_port}/?url=ws://localhost:{self.ws_port}"

    def _load_saved_blueprint(self) -> bool:
        """Load+activate the operator's saved .rbl layout if it exists.

        A .rbl is a Blueprint-kind RRD exported from the viewer's
        "Save blueprint" menu. log_file_from_path replays it into this
        recording; because the child brings up a fresh server per session (no
        active blueprint yet) it becomes the active layout, and it is bound to
        our application_id (phantom/<rig>) so connected viewers apply it.
        Returns True if a file was loaded (so the caller skips the code layout).
        """
        path = self.blueprint_path
        if not path or not os.path.isfile(path):
            return False
        try:
            self._rr.log_file_from_path(path)
            log.info("rerun blueprint loaded from %s", path)
            return True
        except Exception:
            log.warning("rerun saved blueprint %s failed to load — using the "
                        "code-defined layout", path, exc_info=True)
            return False

    def _send_blueprint(self) -> None:
        """Operator-saved .rbl if present, else a tidy code-defined layout
        (harmless to skip if the blueprint API drifts)."""
        if self._load_saved_blueprint():
            return
        try:
            import rerun.blueprint as rrb
            rr = self._rr
            images = [rrb.Spatial2DView(origin="/camera/scene", name="scene")]
            for s in self.hw.tactile.sensors:
                images.append(rrb.Spatial2DView(origin=f"/tactile/{s.name}/depth",
                                                name=f"{s.name} depth"))
                images.append(rrb.Spatial2DView(origin=f"/tactile/{s.name}/infer",
                                                name=f"{s.name} infer"))
            plots = [rrb.TimeSeriesView(origin=f"/tactile/{s.name}/wrench",
                                        name=f"{s.name} wrench")
                     for s in self.hw.tactile.sensors]
            plots += [rrb.TimeSeriesView(origin="/arm/ft", name="wrist F/T"),
                      rrb.TimeSeriesView(origin="/gripper", name="gripper")]
            rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
                rrb.Grid(*images), rrb.Vertical(*plots),
                rrb.TextLogView(origin="/events", name="events"),
                column_shares=[3, 2, 1])))
        except Exception:
            log.debug("rerun blueprint skipped (API mismatch)", exc_info=True)

    # ------------------------------------------------------------------
    def log_event(self, text: str, *, level: str = "INFO") -> None:
        """Episode transitions + safety events from the record loop.

        Called from the record loop, so it must NEVER block: under rerun
        backpressure the 15 Hz tick can sit inside rr.log for seconds holding
        _rr_lock; if the record loop blocked on that lock it would stall the
        whole session (and the motion-safety bookkeeping with it). Drop the
        event rather than wait."""
        rr = self._rr
        if rr is None:
            return
        if not RerunLogger._rr_lock.acquire(timeout=0.2):
            log.debug("rerun busy (backpressure) — dropping event: %s", text)
            return
        try:
            rr.log("/events", rr.TextLog(text, level=level))
        except Exception:
            log.debug("rerun TextLog failed", exc_info=True)
        finally:
            RerunLogger._rr_lock.release()

    # ------------------------------------------------------------------
    def _set_time(self, ts: float) -> None:
        rr = self._rr
        try:
            rr.set_time("t_host", duration=ts)      # modern (0.23+)
        except (AttributeError, TypeError):
            rr.set_time_seconds("t_host", ts)       # legacy

    def _scalar(self, path: str, value: float) -> None:
        rr = self._rr
        try:
            rr.log(path, rr.Scalar(float(value)))
        except AttributeError:
            rr.log(path, rr.Scalars(float(value)))

    def _image_f(self, path: str, arr: np.ndarray) -> None:
        """Float field map — DepthImage gives a colormapped, range-sliderable view."""
        rr = self._rr
        try:
            rr.log(path, rr.DepthImage(np.ascontiguousarray(arr, dtype=np.float32)))
        except Exception:
            a = np.abs(arr).astype(np.float32)
            hi = float(a.max()) or 1.0
            rr.log(path, rr.Image((a / hi * 255).astype(np.uint8)))

    def _image_u8(self, path: str, arr: np.ndarray) -> None:
        self._rr.log(path, self._rr.Image(np.ascontiguousarray(arr)))

    def _camera_image(self, path: str, arr: np.ndarray) -> None:
        """Scene RGB — JPEG-compressed when camera_jpeg_quality > 0.

        Raw 640x480x3 at 10 Hz is ~9 MB/s, which alone saturates the gRPC
        channel and the browser viewer; JPEG q=75 is ~20x smaller. Falls back
        to raw if the SDK build lacks Image.compress or encoding fails."""
        rr = self._rr
        d = self.camera_viz_downscale
        if d > 1:
            h, w = arr.shape[:2]
            if self._cv2 is not None:
                # INTER_AREA is the correct anti-aliased downsampler; plain
                # [::d,::d] decimation aliases and looks noticeably worse.
                arr = self._cv2.resize(arr, (w // d, h // d),
                                       interpolation=self._cv2.INTER_AREA)
            else:
                arr = arr[::d, ::d]
        arr = np.ascontiguousarray(arr)
        q = self.camera_jpeg_quality
        if q > 0:
            try:
                rr.log(path, rr.Image(arr).compress(jpeg_quality=q))
                return
            except Exception:
                log.debug("rerun JPEG compress failed — logging raw", exc_info=True)
        rr.log(path, rr.Image(arr))

    def _fresh(self, ring_name: str):
        """Latest sample if newer than the previous logged one, else None."""
        ring = self.rings.get(ring_name)
        if ring is None:
            return None
        ts, data = ring.latest(1)
        if not len(ts) or ts[0] <= self._last_ts.get(ring_name, -1.0):
            return None
        self._last_ts[ring_name] = float(ts[0])
        return float(ts[0]), data

    # ------------------------------------------------------------------
    def _run(self) -> None:
        from phantom.data.derived import channel_slices
        ch = channel_slices(self.hw.tactile)
        # loop at the faster of the tactile poll rate and the camera rate;
        # slower streams self-decimate (their _fresh returns None between
        # samples), so only the camera actually logs at the higher cadence.
        loop_hz = max(self.rate_hz, self.camera_rate_hz or self.rate_hz)
        period = 1.0 / loop_hz
        slow_streak = 0
        # Back-off instrumentation: the bursty "1-2 s lag + spinner" seen in the
        # field is consistent with slow_streak pinning under CPU contention and
        # collapsing the effective tick rate. Report it every 30 s so we can
        # confirm (effective Hz far below loop_hz + high back-off count = this).
        win_t0 = time.perf_counter()
        win_ticks = win_backoff = win_max_streak = 0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                with RerunLogger._rr_lock:
                    self._tick(ch)
            except Exception:
                log.exception("rerun viz tick failed")
            # Backpressure back-off: when the viewer/gRPC channel is full, rr.log
            # blocks for seconds ("Sender has been blocked" in the rerun log).
            # Piling more data onto a full channel is what precedes the libarrow
            # segfault on rerun 0.34, so when a tick runs long, progressively
            # stretch the interval to let the channel drain instead of hammering.
            dur = time.perf_counter() - t0
            if dur > 3.0 * period:
                slow_streak = min(slow_streak + 1, 10)
                win_backoff += 1
            elif slow_streak > 0:
                slow_streak -= 1
            win_ticks += 1
            win_max_streak = max(win_max_streak, slow_streak)
            now = time.perf_counter()
            if now - win_t0 >= 30.0:
                log.info("rerun tick: %.1f Hz effective over %.0fs (target %.0f Hz); "
                         "back-off engaged %d/%d ticks, max streak %d",
                         win_ticks / (now - win_t0), now - win_t0, loop_hz,
                         win_backoff, win_ticks, win_max_streak)
                win_t0 = now
                win_ticks = win_backoff = win_max_streak = 0
            wait = period * (1 + slow_streak) - dur
            if wait > 0:
                self._stop.wait(wait)

    def _tick(self, ch: dict) -> None:
        # camera
        got = self._fresh("camera_scene")
        if got:
            ts, d = got
            self._set_time(ts)
            self._camera_image("/camera/scene", d["color"][0])
        # tactile
        for s in self.hw.tactile.sensors:
            got = self._fresh(f"tactile_{s.name}")
            if got:
                ts, d = got
                self._set_time(ts)
                fields = np.asarray(d["fields_ds"][0], dtype=np.float32)
                base = f"/tactile/{s.name}"
                self._image_f(f"{base}/depth", fields[..., ch["depth"]][..., 0])
                self._image_f(f"{base}/deform",
                              np.linalg.norm(fields[..., ch["deformation2d"]], axis=-1))
                self._image_f(f"{base}/shear",
                              np.linalg.norm(fields[..., ch["shear"]], axis=-1))
                self._image_f(f"{base}/fz", fields[..., ch["dist_force"]][..., 2])
                wrench = np.asarray(d["wrench"][0], dtype=np.float32).reshape(-1)
                for k, name in enumerate(_WRENCH[:wrench.size]):
                    self._scalar(f"{base}/wrench/{name}", wrench[k])
                self._scalar(f"{base}/area", float(np.asarray(d["area"][0]).reshape(-1)[0]))
            got = self._fresh(f"tactile_{s.name}_img")
            if got:
                ts, d = got
                self._set_time(ts)
                self._image_u8(f"/tactile/{s.name}/infer", d["infer_img"][0])
        # arm
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
        # gripper
        got = self._fresh("gripper")
        if got:
            ts, d = got
            self._set_time(ts)
            st = np.asarray(d["state"][0]).reshape(-1)
            self._scalar("/gripper/pos", float(st[0]))
            if st.size > 1:
                self._scalar("/gripper/obj", float(st[1]))
