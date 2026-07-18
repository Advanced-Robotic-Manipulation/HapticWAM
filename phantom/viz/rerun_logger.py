"""Live Rerun visualization of the whole rig — every session ring stream.

Reads the SAME shared-memory rings the recorder drains (zero extra load on
the drivers) and logs to a rerun.io web viewer at ~15 Hz:

    /camera/scene                       RGB frames
    /tactile/<name>/depth|deform|shear|fz   field-channel images (ds res)
    /tactile/<name>/infer               sensing-area image
    /tactile/<name>/wrench/Fx..Mz       6 scalar plots  + /area
    /arm/q/j0..j5  /arm/ft/Fx..Mz      joint + wrist-wrench plots
    /arm/tcp/x|y|z                      TCP position plots
    /gripper/pos|current                gripper plots
    /events                             episode + safety text log

`rerun-sdk` is optional (`pip install 'phantom[viz]'`) and imported lazily;
API calls are wrapped defensively (helpers fall back across SDK versions).
The web viewer URL (`viewer_url`) is what the control panel embeds.
"""

from __future__ import annotations

import logging
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
                 web_port: int = 9090, ws_port: int = 9877):
        self.hw = hw
        self.rings = rings
        self.rate_hz = rate_hz
        self.web_port = web_port
        self.ws_port = ws_port
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
            from urllib.parse import quote
            try:
                uri = rr.serve_grpc(grpc_port=self.ws_port, cors_allow_origin=["*"])
            except TypeError:
                uri = rr.serve_grpc(grpc_port=self.ws_port)
            rr.serve_web_viewer(web_port=self.web_port, open_browser=False,
                                connect_to=uri)
            return f"http://localhost:{self.web_port}/?url={quote(uri, safe='')}"
        # legacy API (< 0.24)
        rr.serve_web(open_browser=False, web_port=self.web_port, ws_port=self.ws_port)
        return f"http://localhost:{self.web_port}/?url=ws://localhost:{self.ws_port}"

    def _send_blueprint(self) -> None:
        """Tidy default layout; harmless to skip if the blueprint API drifts."""
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
        """Episode transitions + safety events from the record loop."""
        rr = self._rr
        if rr is None:
            return
        try:
            with RerunLogger._rr_lock:
                rr.log("/events", rr.TextLog(text, level=level))
        except Exception:
            log.debug("rerun TextLog failed", exc_info=True)

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
        period = 1.0 / self.rate_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                with RerunLogger._rr_lock:
                    self._tick(ch)
            except Exception:
                log.exception("rerun viz tick failed")
            wait = period - (time.perf_counter() - t0)
            if wait > 0:
                self._stop.wait(wait)

    def _tick(self, ch: dict) -> None:
        # camera
        got = self._fresh("camera_scene")
        if got:
            ts, d = got
            self._set_time(ts)
            self._image_u8("/camera/scene", d["color"][0])
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
                self._scalar("/gripper/current", float(st[1]))
