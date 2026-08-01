"""Episode playback + recording verification (the panel's "Episodes" card).

After a session's episodes are offloaded to the external drive the operator
can — without leaving the browser — pick any recorded episode, VERIFY that
everything expected was actually recorded (streams present, row counts,
rates, monotonic timestamps, finite values) and PLAY it back into the same
embedded rerun viewer used for live collection, every modality visualized.

Design notes:
  - Playback logs to the once-per-process rerun server (RerunLogger's server
    guard) on a SEPARATE timeline `t_episode`, so replayed data never mixes
    with the live `t_host` timeline. rerun has no per-timeline clear, so
    successive playbacks are appended after a gap (PlayerView._t_base).
  - The player thread paces a wall clock * speed; when behind it drops stale
    IMAGE frames (only the newest pending frame is logged) and strides
    scalar batches — the viewer keeps up at any speed.
  - Verification is read-only. With a HardwareConfig the expected stream set
    derives from the sensors/cameras/recording flags and the episode's mode
    tag (full | lite); without one only the recorded streams are checked.
  - Exclusive with a live session/offload: the rerun recording, the staging
    dir and the drive are shared resources. PlaybackController and
    CollectRunner serialize their start checks on the SAME gate lock.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from phantom.data.episode_store import EpisodeReader, list_episodes
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ACTIONS_ABS,
                                 STREAM_ACTIONS_QTARGET, STREAM_ARM_FT,
                                 STREAM_ARM_Q, STREAM_ARM_QD,
                                 STREAM_ARM_TCP_POSE, STREAM_ARM_TCP_SPEED,
                                 STREAM_GRIPPER, EpisodeMeta, tactile_stream)
from phantom.viz.rerun_logger import _WRENCH, RerunLogger

log = logging.getLogger(__name__)

# per-sensor stream kinds (recorded + postprocess-derived), longest-suffix
# match so sensor names may themselves contain underscores
_TACTILE_KINDS = ("fields_ds", "keyframes", "wrench", "area", "infer_img",
                  "raw_img", "mask_frac", "cop", "slip", "events")
_TCP_LABELS = ("x", "y", "z", "rx", "ry", "rz")


def _split_tactile(stream: str) -> tuple[str, str] | None:
    """"tactile_<sensor>_<kind>" -> (sensor, kind), else None."""
    if not stream.startswith("tactile_"):
        return None
    rest = stream[len("tactile_"):]
    for kind in _TACTILE_KINDS:
        if rest.endswith("_" + kind):
            return rest[:-len(kind) - 1], kind
    return None


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def expected_streams(hw, mode: str) -> list[str]:
    """What a collect episode must contain, given the rig config and the
    collection mode ("lite" drops the heavy tactile streams — session.py's
    lite_stream_filter is the recording-side mirror of this list)."""
    exp = [STREAM_ARM_Q, STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
           STREAM_ARM_TCP_SPEED, STREAM_ARM_FT, STREAM_GRIPPER,
           STREAM_ACTIONS_ABS, STREAM_ACTIONS]
    for cam_name in ("scene", "wrist"):
        if getattr(hw.cameras, cam_name).enabled:
            exp.append(f"camera_{cam_name}_color")
    for s in hw.tactile.sensors:
        exp += [tactile_stream(s.name, "wrench"), tactile_stream(s.name, "area")]
        if mode != "lite":
            exp += [tactile_stream(s.name, "fields_ds"),
                    tactile_stream(s.name, "keyframes")]
            if hw.recording.save_infer_img:
                exp.append(tactile_stream(s.name, "infer_img"))
            if hw.recording.archive_raw_img:
                exp.append(tactile_stream(s.name, "raw_img"))
    return exp


def _expect_hz(hw, stream: str) -> float | None:
    if hw is None:
        return None
    if stream.startswith("arm_"):
        return float(hw.arm.rtde_receive_hz)
    if stream == STREAM_GRIPPER:
        return float(hw.gripper.feedback_rate_hz)
    if stream in (STREAM_ACTIONS_ABS, STREAM_ACTIONS, STREAM_ACTIONS_QTARGET):
        return float(hw.control.action_rate_hz)
    if stream.startswith("camera_") and stream.endswith("_color"):
        cam = getattr(hw.cameras, stream[len("camera_"):-len("_color")], None)
        return float(cam.fps) if cam is not None else None
    tk = _split_tactile(stream)
    if tk is not None:
        kind = tk[1]
        # fields_ds/wrench/area/raw_img ride the main tactile ring, which the
        # worker pushes on EVERY driver frame (workers.py) — their on-disk
        # rate is tactile.rate_hz; only keyframes/infer_img are decimated
        if kind in ("fields_ds", "wrench", "area", "raw_img"):
            return float(hw.tactile.rate_hz)
        if kind == "keyframes":
            return float(hw.recording.keyframe_rate_hz)
        if kind == "infer_img":
            return float(hw.recording.infer_img_rate_hz)
    return None


def _stream_report(reader: EpisodeReader, stream: str,
                   expect_hz: float | None) -> dict:
    rep = {"name": stream, "rows": 0, "t0": 0.0, "t1": 0.0, "dur_s": 0.0,
           "hz": 0.0, "expect_hz": float(expect_hz or 0.0), "dt_max_ms": 0.0,
           "gap_x": 0.0, "shape": "", "dtype": "", "issues": [], "warns": []}
    try:
        ts = reader.ts(stream)
        data = reader.data(stream)
    except Exception as e:
        rep["issues"].append(f"unreadable: {e}")
        return rep
    n_ts, n_rows = len(ts), int(data.shape[0])
    rep["rows"] = n_rows
    rep["shape"] = "x".join(str(int(x)) for x in data.shape[1:]) or "scalar"
    rep["dtype"] = str(data.dtype)
    if n_rows == 0:
        rep["issues"].append("empty — no rows recorded")
        return rep
    if n_ts != n_rows:
        rep["issues"].append(f"ts/data length mismatch: {n_ts} ts vs {n_rows} rows")
    ts = ts[:min(n_ts, n_rows)]
    if not np.all(np.isfinite(ts)):
        # NaN defeats every comparison below (dt<0, dur>0, median) AND would
        # leak literal NaN into the JSON report, killing the panel's SSE
        # JSON.parse — flag hard and skip the timing math entirely
        rep["issues"].append("non-finite timestamps (NaN/Inf)")
    elif len(ts) == 1:
        rep["t0"] = rep["t1"] = float(round(ts[0], 3))
        rep["warns"].append("single row — rate/timing unverifiable")
    elif len(ts) >= 2:
        dt = np.diff(ts)
        if np.any(dt < 0):
            rep["issues"].append("timestamps not monotonic")
        dur = float(ts[-1] - ts[0])
        rep["t0"], rep["t1"] = float(round(ts[0], 3)), float(round(ts[-1], 3))
        rep["dur_s"] = float(round(dur, 2))
        if dur > 0:
            rep["hz"] = float(round((len(ts) - 1) / dur, 1))
        elif not np.any(dt < 0):
            rep["issues"].append("zero duration — all timestamps identical")
        pos = dt[dt > 0]
        if len(pos) >= 2:
            med = float(np.median(pos))
            rep["dt_max_ms"] = float(round(float(dt.max()) * 1e3, 1))
            if med > 0:
                rep["gap_x"] = float(round(float(dt.max()) / med, 1))
                if rep["gap_x"] > 5.0 and float(dt.max()) > 0.25:
                    rep["warns"].append(f"gap of {rep['dt_max_ms']:.0f} ms")
        if expect_hz and dur > 0 and rep["hz"] < 0.6 * expect_hz:
            rep["warns"].append(
                f"rate low: {rep['hz']:g} Hz vs ~{expect_hz:g} Hz expected")
    # finite scan: full for small float streams, head/tail sample for big ones
    if np.issubdtype(data.dtype, np.floating):
        try:
            if data.size <= 2_000_000:
                finite = bool(np.all(np.isfinite(data[:])))
            else:
                k = min(n_rows, 32)
                finite = bool(np.all(np.isfinite(data[:k]))
                              and np.all(np.isfinite(data[-k:])))
                rep["warns"].append("finite check sampled (large stream)")
            if not finite:
                rep["issues"].append("non-finite values (NaN/Inf)")
        except Exception as e:
            rep["warns"].append(f"finite check failed: {e}")
    return rep


def verify_episode(path: str | Path, hw=None) -> dict:
    """Read-only integrity report for one recorded episode directory.

    ok = meta finalized, no expected stream missing, and no per-stream hard
    issue (empty / length mismatch / non-monotonic ts / non-finite values).
    Rate/gap/coverage findings are warnings — visible but not fatal."""
    path = Path(path)
    rep = {"path": str(path), "name": path.name, "ok": False, "mode": "?",
           "meta": {}, "streams": [], "missing": [], "issues": [],
           "warns": [], "span_s": 0.0}
    try:
        meta = EpisodeMeta.load(path / "meta.json")
    except Exception as e:
        rep["issues"].append(f"meta.json unreadable: {e}")
        return rep
    rep["meta"] = {"task": meta.task, "text": meta.text,
                   "operator": meta.operator, "status": meta.status,
                   "success": meta.success, "notes": meta.notes,
                   "tags": list(meta.tags)}
    mode = "lite" if "lite" in meta.tags else "full"
    rep["mode"] = mode
    if meta.status != "finalized":
        rep["issues"].append(f"episode status is {meta.status!r} (not finalized)")

    reader = EpisodeReader(path)
    try:
        streams = reader.streams()
    except OSError as e:
        rep["issues"].append(f"cannot list streams: {e}")
        return rep
    if not streams:
        rep["issues"].append("no streams recorded")
        return rep
    # The expected-stream set derives from the LIVE hw config. When the
    # episode was recorded under a different config (meta.config_hash
    # mismatch — renamed sensors, other camera set, ...) a missing stream is
    # not proof of a broken recording: demote to warning instead of failing.
    hash_match = True
    if hw is not None:
        try:
            hash_match = (not meta.config_hash
                          or meta.config_hash == hw.config_hash())
        except Exception:
            hash_match = True
        exp = expected_streams(hw, mode)
        rep["missing"] = sorted(s for s in exp if s not in streams)
        if not hash_match and rep["missing"]:
            rep["warns"].append(
                "episode was recorded under a DIFFERENT hardware config — "
                "missing-stream findings are advisory only")
    for s in streams:
        rep["streams"].append(_stream_report(reader, s, _expect_hz(hw, s)))

    spans = [(s["t0"], s["t1"]) for s in rep["streams"]
             if s["rows"] >= 2 and not s["issues"]]
    if spans:
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        span = hi - lo
        rep["span_s"] = float(round(span, 2))
        if span > 2.0:
            for s in rep["streams"]:
                if s["rows"] >= 1 and not s["issues"] \
                        and s["dur_s"] < 0.5 * span:
                    s["warns"].append(
                        f"covers only {s['dur_s']:.1f} s of {span:.1f} s")
    rep["ok"] = (not rep["issues"]
                 and (not rep["missing"] or not hash_match)
                 and all(not s["issues"] for s in rep["streams"]))
    return rep


# ---------------------------------------------------------------------------
# rerun view for playback
# ---------------------------------------------------------------------------

class PlayerView(RerunLogger):
    """Rerun access for playback: the same once-per-process server as the
    live view, but NO ring-tick thread, and a dedicated `t_episode` timeline
    (normalized to 0 per playback, appended after the previous one)."""

    _t_base = 0.0            # class-level: next playback's timeline origin
    _GAP_S = 5.0

    def __init__(self, hw, *, web_port: int = 9091, ws_port: int = 9878):
        super().__init__(hw, rings={}, web_port=web_port, ws_port=ws_port)

    def start(self) -> None:
        try:
            import rerun as rr  # lazy: [viz] extra
        except ImportError as e:
            raise RuntimeError(
                "rerun-sdk not installed — pip install 'phantom[viz]'") from e
        self._rr = rr
        with RerunLogger._rr_lock:
            if RerunLogger._server_url is None:
                rr.init(f"phantom/{self.hw.meta.rig_name}", spawn=False)
                RerunLogger._server_url = self._serve_web()
        self.viewer_url = RerunLogger._server_url

    def stop(self) -> None:      # the server is process-wide; nothing to stop
        pass

    def next_base(self, dur_s: float) -> float:
        base = PlayerView._t_base
        PlayerView._t_base = base + dur_s + PlayerView._GAP_S
        return base

    def _set_time(self, ts: float) -> None:
        rr = self._rr
        try:
            rr.set_time("t_episode", duration=ts)       # modern (0.23+)
        except (AttributeError, TypeError):
            rr.set_time_seconds("t_episode", ts)        # legacy

    def send_episode_blueprint(self, streams: list[str]) -> None:
        """Layout built from what the EPISODE actually contains (the current
        hw config may differ from the one the episode was recorded under)."""
        try:
            import rerun.blueprint as rrb
            images, sensors = [], []
            for s in streams:
                if s.startswith("camera_") and s.endswith("_color"):
                    cam = s[len("camera_"):-len("_color")]
                    images.append(rrb.Spatial2DView(origin=f"/camera/{cam}",
                                                    name=cam))
            for s in streams:
                tk = _split_tactile(s)
                if tk is not None and tk[0] not in sensors:
                    sensors.append(tk[0])
            for name in sensors:
                if tactile_stream(name, "fields_ds") in streams:
                    images.append(rrb.Spatial2DView(
                        origin=f"/tactile/{name}/depth", name=f"{name} depth"))
                    images.append(rrb.Spatial2DView(
                        origin=f"/tactile/{name}/fz", name=f"{name} Fz"))
                if tactile_stream(name, "infer_img") in streams:
                    images.append(rrb.Spatial2DView(
                        origin=f"/tactile/{name}/infer", name=f"{name} gel"))
                if tactile_stream(name, "keyframes") in streams:
                    images.append(rrb.Spatial2DView(
                        origin=f"/tactile/{name}/keyframe", name=f"{name} keyframe"))
                if tactile_stream(name, "raw_img") in streams:
                    images.append(rrb.Spatial2DView(
                        origin=f"/tactile/{name}/raw", name=f"{name} raw"))
            plots = [rrb.TimeSeriesView(origin=f"/tactile/{n}/wrench",
                                        name=f"{n} wrench (6-axis)")
                     for n in sensors
                     if tactile_stream(n, "wrench") in streams]
            plots += [rrb.TimeSeriesView(origin="/arm/ft", name="wrist F/T"),
                      rrb.TimeSeriesView(origin="/arm/q", name="joints"),
                      rrb.TimeSeriesView(origin="/arm/tcp", name="TCP pose"),
                      rrb.TimeSeriesView(origin="/gripper", name="gripper")]
            # deform/shear maps, qd/tcp_speed, area and the derived streams
            # are all logged too — reachable from the viewer's entity tree
            if any(s in streams for s in (STREAM_ACTIONS_ABS, STREAM_ACTIONS,
                                          STREAM_ACTIONS_QTARGET)):
                plots.append(rrb.TimeSeriesView(origin="/actions",
                                                name="actions (abs)"))
            with RerunLogger._rr_lock:
                self._rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
                    rrb.Grid(*images), rrb.Vertical(*plots),
                    rrb.TextLogView(origin="/events", name="events"),
                    column_shares=[3, 2, 1])))
        except Exception:
            log.debug("playback blueprint skipped (API mismatch)", exc_info=True)


# ---------------------------------------------------------------------------
# the player
# ---------------------------------------------------------------------------

def _vec_logger(view, base: str, labels=None, fmt: str = "{}"):
    def _log(row) -> None:
        r = np.asarray(row, dtype=np.float32).reshape(-1)
        for k in range(r.size):
            lbl = labels[k] if labels is not None and k < len(labels) \
                else fmt.format(k)
            view._scalar(f"{base}/{lbl}", float(r[k]))
    return _log


class _StreamCursor:
    """Sequential reader over one episode stream with a small read-ahead
    block, so per-row playback never re-decompresses a whole zarr chunk."""

    def __init__(self, stream: str, ts: np.ndarray, data, log_fn, *,
                 image: bool):
        self.stream = stream
        self.log = log_fn
        self.image = image
        n = min(len(ts), int(data.shape[0]))
        # np.searchsorted needs a sorted array; a corrupt (non-monotonic)
        # episode must degrade playback, never hang it — monotonicize. The
        # verifier reports the underlying defect separately.
        self.ts = np.maximum.accumulate(
            np.asarray(ts[:n], dtype=np.float64))
        self.data = data
        self.n = n
        self.i = 0
        self._blk0 = 0
        self._blk = None
        self._blk_len = max(1, min(64, n))

    def row(self, i: int):
        if self._blk is None or not (self._blk0 <= i < self._blk0 + len(self._blk)):
            self._blk0 = i
            self._blk = self.data[i:i + self._blk_len]
        return self._blk[i - self._blk0]


class EpisodePlayer:
    """Streams a recorded episode into a rerun view, paced by wall clock *
    speed; pause/stop/speed are runtime-settable from the panel."""

    TICK_S = 0.02
    MAX_SCALAR_ROWS_PER_TICK = 16   # per stream; beyond this the batch strides

    def __init__(self, view, *, on_status=None):
        self.view = view
        self.on_status = on_status or (lambda st: None)
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # serializes snapshot+delivery in _push, so two concurrent pushes
        # (player thread vs an HTTP thread's set_speed) can never deliver
        # their status callbacks in the opposite order of their snapshots —
        # a stale 'playing' must not overwrite a terminal 'done'/'error'
        self._cb_lock = threading.Lock()
        self._st = {"episode": "", "state": "idle", "pos_s": 0.0,
                    "dur_s": 0.0, "speed": 1.0, "msg": ""}

    # ------------------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            return dict(self._st)

    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play(self, path: str | Path, *, speed: float = 1.0, hw=None) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("playback already running")
            self._stop.clear()
            self._pause.clear()
            self._st.update(episode=Path(path).name, state="loading",
                            pos_s=0.0, dur_s=0.0,
                            speed=self._clamp_speed(speed), msg="")
            self._thread = threading.Thread(
                target=self._run, args=(Path(path), hw), daemon=True,
                name="episode-player")
            self._thread.start()
        self._push()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self, join_s: float = 3.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(join_s)

    @staticmethod
    def _clamp_speed(v: float) -> float:
        return float(min(16.0, max(0.1, float(v))))

    def set_speed(self, v: float) -> None:
        with self._lock:
            self._st["speed"] = self._clamp_speed(v)
        self._push()

    # ------------------------------------------------------------------
    def _push(self, **kw) -> None:
        with self._cb_lock:
            with self._lock:
                self._st.update(kw)
                st = dict(self._st)
            try:
                self.on_status(st)
            except Exception:
                log.exception("playback status callback failed")

    def _event(self, text: str, level: str = "INFO") -> None:
        fn = getattr(self.view, "log_event", None)
        if fn is not None:
            fn(text, level=level)

    # ------------------------------------------------------------------
    def _run(self, path: Path, hw) -> None:
        try:
            cursors, skipped = self._build_cursors(EpisodeReader(path), hw)
        except Exception as e:
            log.exception("playback failed to open %s", path)
            self._push(state="error", msg=f"cannot open episode: {e}")
            return
        if not cursors:
            self._push(state="error", msg="episode has no playable streams")
            return
        t0 = min(float(c.ts[0]) for c in cursors)
        dur = max(0.0, max(float(c.ts[-1]) for c in cursors) - t0)
        base = getattr(self.view, "next_base", lambda d: 0.0)(dur)
        bp = getattr(self.view, "send_episode_blueprint", None)
        if bp is not None:
            bp([c.stream for c in cursors])
        self._event(f"PLAYBACK {path.name} ({dur:.1f} s, "
                    f"{len(cursors)} streams)")
        if skipped:
            self._event("playback skipped (no renderer): " + ", ".join(skipped),
                        "WARN")

        pos = 0.0
        last = time.perf_counter()
        self._push(state="playing", dur_s=dur)
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                if self._pause.is_set():
                    self._push(state="paused", pos_s=pos)
                    self._stop.wait(0.1)
                    # refresh AFTER the wait — otherwise up to 0.1 s of
                    # paused wall time is counted as playing on resume
                    last = time.perf_counter()
                    continue
                with self._lock:
                    speed = self._st["speed"]
                pos = min(dur, pos + (now - last) * speed)
                last = now
                t_abs = t0 + pos
                with RerunLogger._rr_lock:
                    for c in cursors:
                        self._log_pending(c, t_abs, t0, base)
                self._push(state="playing", pos_s=pos)
                if pos >= dur and all(c.i >= c.n for c in cursors):
                    break
                wait = self.TICK_S - (time.perf_counter() - now)
                if wait > 0:
                    self._stop.wait(wait)
        except Exception as e:
            log.exception("playback of %s failed", path)
            self._push(state="error", pos_s=pos, msg=str(e))
            return
        if self._stop.is_set():
            self._push(state="stopped", pos_s=pos)
            self._event(f"playback of {path.name} stopped")
        else:
            self._push(state="done", pos_s=dur)
            self._event(f"playback of {path.name} done")

    def _log_pending(self, c: _StreamCursor, t_abs: float, t0: float,
                     base: float) -> None:
        j = int(np.searchsorted(c.ts, t_abs, side="right"))
        if j <= c.i:
            return
        if c.image:                       # newest pending frame only
            idx = [j - 1]
        else:
            idx = list(range(c.i, j))
            if len(idx) > self.MAX_SCALAR_ROWS_PER_TICK:
                stride = -(-len(idx) // self.MAX_SCALAR_ROWS_PER_TICK)
                idx = idx[::stride] + ([j - 1] if idx[::stride][-1] != j - 1
                                       else [])
        for i in idx:
            self.view._set_time(base + float(c.ts[i] - t0))
            c.log(c.row(i))
        c.i = j

    # ------------------------------------------------------------------
    def _build_cursors(self, reader: EpisodeReader,
                       hw) -> tuple[list[_StreamCursor], list[str]]:
        ch = None
        if hw is not None:
            try:
                from phantom.data.derived import channel_slices
                ch = channel_slices(hw.tactile)
            except Exception:
                ch = None
        cursors: list[_StreamCursor] = []
        skipped: list[str] = []
        for stream in reader.streams():
            try:
                ts = reader.ts(stream)
                data = reader.data(stream)
            except Exception:
                log.warning("playback: unreadable stream %s — skipping", stream)
                skipped.append(stream)
                continue
            n = min(len(ts), int(data.shape[0]))
            if n == 0:
                skipped.append(stream)
                continue
            if not np.all(np.isfinite(np.asarray(ts[:n]))):
                # NaN sorts as +inf in searchsorted — the cursor would never
                # drain and the playback loop could not terminate
                log.warning("playback: non-finite timestamps in %s — skipping",
                            stream)
                skipped.append(stream)
                continue
            log_fn, image = self._renderer(stream, ch)
            if log_fn is None:
                skipped.append(stream)
                continue
            cursors.append(_StreamCursor(stream, ts, data, log_fn, image=image))
        return cursors, skipped

    def _renderer(self, stream: str, ch):
        """(log_fn(row), is_image) for a stream — None if not renderable."""
        v = self.view
        if stream.startswith("camera_") and stream.endswith("_color"):
            cam = stream[len("camera_"):-len("_color")]
            return (lambda row, p=f"/camera/{cam}":
                    v._image_u8(p, np.asarray(row))), True
        tk = _split_tactile(stream)
        if tk is not None:
            sensor, kind = tk
            base = f"/tactile/{sensor}"
            if kind == "fields_ds":
                def _fields(row, p=base, ch=ch):
                    f = np.asarray(row, dtype=np.float32)
                    if ch is not None and f.shape[-1] == 8:
                        v._image_f(f"{p}/depth", f[..., ch["depth"]][..., 0])
                        v._image_f(f"{p}/deform", np.linalg.norm(
                            f[..., ch["deformation2d"]], axis=-1))
                        v._image_f(f"{p}/shear", np.linalg.norm(
                            f[..., ch["shear"]], axis=-1))
                        v._image_f(f"{p}/fz", f[..., ch["dist_force"]][..., 2])
                    else:               # unknown channel layout: one image
                        v._image_f(f"{p}/fields", np.abs(f).mean(axis=-1))
                return _fields, True
            if kind in ("infer_img", "raw_img", "keyframes"):
                name = {"infer_img": "infer", "raw_img": "raw",
                        "keyframes": "keyframe"}[kind]
                return (lambda row, p=f"{base}/{name}":
                        v._image_u8(p, np.asarray(row))), True
            if kind == "wrench":
                return _vec_logger(v, f"{base}/wrench", labels=_WRENCH), False
            if kind == "area":
                return (lambda row, p=f"{base}/area":
                        v._scalar(p, float(np.asarray(row).reshape(-1)[0]))), False
            # derived vectors (mask_frac, cop, slip, events): generic scalars
            return _vec_logger(v, f"{base}/{kind}", fmt="c{}"), False
        if stream == STREAM_ARM_Q:
            return _vec_logger(v, "/arm/q", fmt="j{}"), False
        if stream == STREAM_ARM_QD:
            return _vec_logger(v, "/arm/qd", fmt="j{}"), False
        if stream == STREAM_ARM_TCP_POSE:
            return _vec_logger(v, "/arm/tcp", labels=_TCP_LABELS), False
        if stream == STREAM_ARM_TCP_SPEED:
            return _vec_logger(v, "/arm/tcp_speed", labels=_TCP_LABELS), False
        if stream == STREAM_ARM_FT:
            return _vec_logger(v, "/arm/ft", labels=_WRENCH), False
        if stream == STREAM_GRIPPER:
            return _vec_logger(v, "/gripper", labels=("pos", "obj")), False
        if stream in (STREAM_ACTIONS_ABS, STREAM_ACTIONS_QTARGET):
            return _vec_logger(
                v, "/actions",
                labels=("q0", "q1", "q2", "q3", "q4", "q5", "gripper")), False
        if stream == STREAM_ACTIONS:
            return _vec_logger(v, "/actions", fmt="a{}"), False
        return None, False


# ---------------------------------------------------------------------------
# panel-facing controller
# ---------------------------------------------------------------------------

class PlaybackController:
    """Bridges the panel's /api/episodes + /api/playback/* endpoints to the
    player. Mutually exclusive with sessions/offloads: both this and
    CollectRunner run their start checks under the SAME gate lock."""

    def __init__(self, cc, hw, panel, *, session_busy=None, gate_lock=None):
        self.cc = cc
        self.hw = hw
        self.panel = panel
        self._session_busy = session_busy or (lambda: False)
        self._gate = gate_lock if gate_lock is not None else threading.Lock()
        self._view = None                      # PlayerView, lazy
        self.player: EpisodePlayer | None = None
        self._verify_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def roots(self) -> list[tuple[str, Path]]:
        return [("drive", Path(self.cc.storage.external_drive)),
                ("staging", Path(self.cc.storage.staging_root))]

    def scan(self, limit: int = 500) -> list[dict]:
        """Episodes on the external drive + local staging, newest first.

        Deliberately lock-free (read-only, callable any time from the panel);
        a concurrent offload may move directories mid-walk, so every step is
        exception-tolerant — a transient half-listing self-heals on the next
        refresh, and play()/verify() re-validate under the gate anyway."""
        out = []
        for source, root in self.roots():
            try:
                if not root.exists():
                    continue
                eps = list_episodes(root, include_unfinalized=True)
            except OSError:
                log.warning("episode scan of %s failed mid-walk", root,
                            exc_info=True)
                continue
            for ep in eps:
                try:
                    meta = EpisodeMeta.load(ep / "meta.json")
                    mtime = (ep / "meta.json").stat().st_mtime
                except Exception:
                    continue
                try:
                    size_mb = sum(p.stat().st_size for p in ep.rglob("*")
                                  if p.is_file()) / 1e6
                except OSError:
                    size_mb = 0.0
                out.append({
                    "path": str(ep), "name": ep.name,
                    "session": "" if ep.parent == root else ep.parent.name,
                    "source": source, "task": meta.task,
                    "status": meta.status, "success": meta.success,
                    "mode": "lite" if "lite" in meta.tags else "full",
                    "size_mb": float(round(size_mb, 1)),
                    "mtime": float(mtime)})
        out.sort(key=lambda d: d["mtime"], reverse=True)
        return out[:limit]

    def _resolve(self, path_str: str) -> Path | None:
        """Only paths inside the drive/staging roots are playable — the panel
        must not become an arbitrary-filesystem reader."""
        if not path_str:
            return None
        try:
            p = Path(path_str).resolve()
        except OSError:
            return None
        for _, root in self.roots():
            try:
                root_r = root.resolve()
            except OSError:
                continue
            if (p == root_r or root_r in p.parents) \
                    and (p / "meta.json").is_file():
                return p
        return None

    # ------------------------------------------------------------------
    def blocking(self) -> bool:
        """True while playback or a verification is in flight — the runner
        refuses to start a session/offload meanwhile."""
        return ((self.player is not None and self.player.active())
                or (self._verify_thread is not None
                    and self._verify_thread.is_alive()))

    # ------------------------------------------------------------------
    def play(self, path: str, speed: float = 1.0) -> str | None:
        with self._gate:
            if self._session_busy():
                return "a session/offload is running — playback afterwards"
            if self.player is not None and self.player.active():
                return "playback already running — stop it first"
            p = self._resolve(path)
            if p is None:
                return "not a recorded episode under the drive/staging roots"
            try:
                self._ensure_view()
            except RuntimeError as e:
                return str(e)
            try:
                self.player.play(p, speed=speed, hw=self.hw)
            except RuntimeError as e:
                return str(e)
            return None

    def verify(self, path: str) -> str | None:
        with self._gate:
            if self._session_busy():
                return "a session/offload is running — verify afterwards"
            if self._verify_thread is not None and self._verify_thread.is_alive():
                return "a verification is already running"
            p = self._resolve(path)
            if p is None:
                return "not a recorded episode under the drive/staging roots"
            if self.panel is not None:
                self.panel.state.update(verify_running=True, verify_report=None)
            self._verify_thread = threading.Thread(
                target=self._verify_run, args=(p,), daemon=True,
                name="episode-verify")
            self._verify_thread.start()
            return None

    def _verify_run(self, p: Path) -> None:
        try:
            rep = verify_episode(p, hw=self.hw)
        except Exception as e:
            log.exception("verify of %s crashed", p)
            rep = {"path": str(p), "name": p.name, "ok": False, "mode": "?",
                   "meta": {}, "streams": [], "missing": [],
                   "issues": [f"verifier crashed: {e}"], "span_s": 0.0}
        if self.panel is not None:
            self.panel.state.update(verify_running=False, verify_report=rep)

    def pause(self) -> None:
        if self.player is not None:
            self.player.pause()

    def resume(self) -> None:
        if self.player is not None:
            self.player.resume()

    def stop(self) -> None:
        if self.player is not None:
            self.player.stop()

    def set_speed(self, v: float) -> None:
        if self.player is not None:
            self.player.set_speed(v)

    # ------------------------------------------------------------------
    def _ensure_view(self):
        if self._view is None:
            view = PlayerView(self.hw, web_port=self.cc.rerun.web_port,
                              ws_port=self.cc.rerun.grpc_port)
            view.start()               # raises RuntimeError without rerun-sdk
            self._view = view
        if self.player is None:
            self.player = EpisodePlayer(self._view, on_status=self._on_status)
        if self.panel is not None and self._view.viewer_url:
            self.panel.state.update(rerun_url=self._view.viewer_url)
        return self._view

    def _on_status(self, st: dict) -> None:
        if self.panel is None:
            return
        self.panel.state.update(
            playback_episode=st["episode"], playback_state=st["state"],
            playback_pos_s=float(round(st["pos_s"], 1)),
            playback_dur_s=float(round(st["dur_s"], 1)),
            playback_speed=float(st["speed"]),
            playback_msg=st.get("msg", ""))
