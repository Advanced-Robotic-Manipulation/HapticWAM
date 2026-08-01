"""EpisodeRecorder: drains the session ring buffers to a zarr episode on a
fixed interval, mapping t_host -> t_master via the MasterClock, with a
throughput watchdog against the config-derived estimate."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.data.episode_store import EpisodeWriter
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ARM_FT, STREAM_ARM_Q,
                                 STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                                 STREAM_ARM_TCP_SPEED, STREAM_GRIPPER,
                                 EpisodeMeta, tactile_stream)
from phantom.recording.workers import SensorSession
from phantom.timesync.clock import MasterClock

log = logging.getLogger(__name__)

# ring name (+field) -> episode stream name mapping
def _stream_map(hw: HardwareConfig) -> list[tuple[str, str, str]]:
    """(ring, ring_field, episode_stream)"""
    m: list[tuple[str, str, str]] = []
    for s in hw.tactile.sensors:
        m.append((f"tactile_{s.name}", "fields_ds", tactile_stream(s.name, "fields_ds")))
        m.append((f"tactile_{s.name}", "wrench", tactile_stream(s.name, "wrench")))
        m.append((f"tactile_{s.name}", "area", tactile_stream(s.name, "area")))
        m.append((f"tactile_{s.name}_kf", "keyframe", tactile_stream(s.name, "keyframes")))
        if hw.recording.save_infer_img:
            m.append((f"tactile_{s.name}_img", "infer_img", tactile_stream(s.name, "infer_img")))
        if hw.recording.archive_raw_img:
            m.append((f"tactile_{s.name}_raw", "raw_img", tactile_stream(s.name, "raw_img")))
    m += [("arm", "q", STREAM_ARM_Q), ("arm", "qd", STREAM_ARM_QD),
          ("arm", "tcp_pose", STREAM_ARM_TCP_POSE),
          ("arm", "tcp_speed", STREAM_ARM_TCP_SPEED), ("arm", "ft", STREAM_ARM_FT),
          ("gripper", "state", STREAM_GRIPPER)]
    for cam_name, cam in (("scene", hw.cameras.scene), ("wrist", hw.cameras.wrist)):
        if cam.enabled:
            m.append((f"camera_{cam_name}", "color", f"camera_{cam_name}_color"))
    return m


class EpisodeRecorder:
    def __init__(self, session: SensorSession, clock: MasterClock, out_root: Path,
                 *, stream_filter=None):
        """stream_filter: optional predicate on the EPISODE stream name; streams
        it rejects are not drained/recorded (collect's lite mode records only
        the tactile wrench/area, skipping fields/keyframes/infer_img)."""
        self.session = session
        self.hw = session.hw
        self.clock = clock
        self.stream_filter = stream_filter
        self.out_root = Path(out_root)
        self._writer: EpisodeWriter | None = None
        self._cursors: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._action_bufs: dict[str, list[tuple[float, np.ndarray]]] = {}
        self._action_lock = threading.Lock()
        self._bytes_written = 0
        self._t_started = 0.0

    def _map(self) -> list[tuple[str, str, str]]:
        m = _stream_map(self.hw)
        if self.stream_filter is not None:
            m = [row for row in m if self.stream_filter(row[2])]
        return m

    # ------------------------------------------------------------------
    def start(self, meta: EpisodeMeta, episode_name: str) -> Path:
        assert self._writer is None, "episode already recording"
        if self._thread is not None and self._thread.is_alive():
            # a stale drain thread appending into a NEW episode's writer would
            # interleave two episodes' data — never start over a live drain
            raise RuntimeError("previous episode's drain thread still alive")
        meta.driver_modes = {"drivers": self.hw.mode.drivers,
                             **{k: v for k, v in self.hw.mode.overrides.items()}}
        meta.clock_calibration = dict(self.clock.calibration)
        path = self.out_root / episode_name
        self._writer = EpisodeWriter(path, self.hw, meta)
        # start draining from 'now': skip everything already in the rings
        with self._action_lock:
            self._action_bufs = {}   # stale actions must not leak into this episode
        self._cursors = {}
        for ring_name, field, _ in self._map():
            ring = self.session.rings.get(ring_name)
            if ring is not None:
                self._cursors[f"{ring_name}/{field}"] = ring.write_index
        self._bytes_written = 0
        self._t_started = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain_loop, daemon=True,
                                        name="recorder-drain")
        self._thread.start()
        return path

    def record_action(self, t_host: float, action: np.ndarray,
                      stream: str = STREAM_ACTIONS) -> None:
        """Called by teleop / the executor for every commanded action.

        `stream` selects the episode stream (default the canonical Δ-EE
        actions; the Echo record loop also pushes STREAM_ACTIONS_QTARGET)."""
        with self._action_lock:
            self._action_bufs.setdefault(stream, []).append(
                (t_host, np.asarray(action, dtype=np.float32)))

    # ------------------------------------------------------------------
    def _drain_once(self) -> None:
        w = self._writer
        if w is None:
            return
        for ring_name, field, stream in self._map():
            ring = self.session.rings.get(ring_name)
            if ring is None:
                continue
            key = f"{ring_name}/{field}"
            nxt, ts, data = ring.drain(self._cursors[key])
            if len(ts):
                t_master = np.array([self.clock.host_to_master(t) for t in ts])
                w.append(stream, t_master, data[field])
                self._bytes_written += data[field].nbytes
            self._cursors[key] = nxt
        with self._action_lock:
            bufs, self._action_bufs = self._action_bufs, {}
        for stream, buf in bufs.items():
            ts = np.array([self.clock.host_to_master(t) for t, _ in buf])
            w.append(stream, ts, np.stack([a for _, a in buf]))

    def _drain_loop(self) -> None:
        interval = self.hw.recording.drain_interval_s
        est = self.hw.field_bytes_per_s() * self.hw.n_fingers
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._drain_once()
            except Exception:
                log.exception("recorder drain failed")
            took = time.perf_counter() - t0
            if took > interval:
                log.warning("recorder drain took %.3fs > interval %.3fs — disk too slow "
                            "for the configured recording plan (estimate %.1f MB/s)",
                            took, interval, est / 1e6)
            self._stop.wait(max(0.0, interval - took))

    # ------------------------------------------------------------------
    def stop(self, *, success: bool | None = None, notes: str = "",
             abort: bool = False, delete: bool = False) -> Path | None:
        if self._writer is None:
            return None
        self._stop.set()
        if self._thread is not None:
            # wait until the drain thread is REALLY dead: with a timed join a
            # stalled drain and the final flush below would append to the same
            # zarr arrays concurrently and desync data/ts
            deadline_warn = time.perf_counter() + 5.0
            while self._thread.is_alive():
                self._thread.join(1.0)
                if time.perf_counter() > deadline_warn:
                    log.warning("recorder drain thread still flushing — waiting")
                    deadline_warn = time.perf_counter() + 5.0
            self._thread = None
        self._drain_once()   # final flush
        w, self._writer = self._writer, None
        if abort:
            # abort marks the episode and KEEPS its partial data on disk
            # (arm faults, quits, teardown). Physical removal only on the
            # explicit delete flag — data destruction must never be implicit.
            w.abort()
            if delete:
                self._delete_episode(w.path)
                return None
            return w.path
        w.finalize(success=success, notes=notes)
        dur = time.perf_counter() - self._t_started
        log.info("episode %s: %.1f s, %.1f MB written (%.1f MB/s)",
                 w.path.name, dur, self._bytes_written / 1e6,
                 self._bytes_written / 1e6 / max(dur, 1e-9))
        return w.path

    def _delete_episode(self, path: Path) -> bool:
        """Physically remove a discarded episode.

        Confined to out_root: a discard must never be able to delete anything
        outside the session staging dir. Returns True if the tree is gone."""
        path = Path(path).resolve()
        root = Path(self.out_root).resolve()
        if path == root or root not in path.parents:
            log.error("refusing to delete %s - outside the staging root %s",
                      path, root)
            return False
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return True
        except OSError:
            log.exception("could not delete discarded episode %s", path)
            return False
        log.info("discarded episode DELETED: %s", path.name)
        return True

    def relabel(self, path: Path, *, success: bool | None = None,
                discard: bool = False, notes: str = "") -> None:
        """Apply an operator verdict to an ALREADY-finalized episode by
        rewriting its meta.json — used for the stop-then-judge flow (the
        episode is finalized without a verdict on stop, then marked
        success/fail/discard). The recorded data is never touched; a discard
        just flips the status to 'aborted' (kept on disk, skipped by listers,
        same as an in-recording abort)."""
        if discard:
            # The operator asked for this episode to go away: DELETE it, do
            # not merely mark it aborted. Leaving ~130 MB of zarr on disk got
            # it offloaded to the external drive and looked like "discard
            # still saves it".
            self._delete_episode(path)
            return
        meta_path = Path(path) / "meta.json"
        meta = EpisodeMeta.load(meta_path)
        if success is not None:
            meta.success = success
        if notes:
            meta.notes = notes
        meta.save(meta_path)
