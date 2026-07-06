"""WindowSampler: recorded episodes -> model window dicts (pipeline.md §6).

A window is anchored at t0 ("now"):
  - conditioning: camera frame(s) at/after t0, wrist F/T window (t0-w, t0],
    UR state at t0, prev action chunk (t0-H/rate, t0];
  - generation targets: future video frames t0+j/fps, the contact package on
    the latent temporal grid u_k = t0 + k * temporal_comp/fps (k=1..Tc), and
    the action chunk (t0, t0+H/rate].

Contact-package targets (mask/CoP/slip/events/gate) are derived ON THE FLY
from the recorded fields_ds stream via phantom.data.derived — windows never
require the offline postprocess pass to have run.

Teacher-only keys: fields (full-res keyframes), gel (infer image), raw
contact_state, reactive. Student windows drop them (the tactile *targets*
stay — they come from the rig's sensors, which the student trains against
but never consumes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from phantom.data import derived as dv
from phantom.data.episode_store import EpisodeReader, list_episodes
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ARM_FT, STREAM_ARM_Q,
                                 STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                                 STREAM_ARM_TCP_SPEED, STREAM_CAMERA_SCENE,
                                 STREAM_GRIPPER, NormStats, tactile_stream)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# resize helper (pure numpy — deploy path must not require cv2)
# ---------------------------------------------------------------------------

def bilinear_resize(a: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of (H, W) or (H, W, C) arrays; exact passthrough when
    the target equals the input size."""
    h, w = int(hw[0]), int(hw[1])
    H, W = a.shape[:2]
    if (H, W) == (h, w):
        return a
    squeeze = a.ndim == 2
    x = a[..., None].astype(np.float32) if squeeze else a.astype(np.float32)
    ys = np.linspace(0.0, H - 1.0, h)
    xs = np.linspace(0.0, W - 1.0, w)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    fy = (ys - y0).astype(np.float32)[:, None, None]
    fx = (xs - x0).astype(np.float32)[None, :, None]
    top = x[y0][:, x0] * (1 - fx) + x[y0][:, x1] * fx
    bot = x[y1][:, x0] * (1 - fx) + x[y1][:, x1] * fx
    out = top * (1 - fy) + bot * fy
    return out[..., 0] if squeeze else out


@dataclass(frozen=True)
class WindowItem:
    episode: Path
    t0: float


# ---------------------------------------------------------------------------

class _EpisodeCache:
    """Reader + per-stream timestamp cache for one episode."""

    def __init__(self, path: Path):
        self.reader = EpisodeReader(path)
        self._ts: dict[str, np.ndarray] = {}

    def ts(self, stream: str) -> np.ndarray:
        t = self._ts.get(stream)
        if t is None:
            t = self.reader.ts(stream)
            self._ts[stream] = t
        return t

    def nearest_idx(self, stream: str, t: float) -> int:
        ts = self.ts(stream)
        i = int(np.searchsorted(ts, t))
        if i <= 0:
            return 0
        if i >= len(ts):
            return len(ts) - 1
        return i if (ts[i] - t) < (t - ts[i - 1]) else i - 1

    def at(self, stream: str, t: float) -> np.ndarray:
        return np.asarray(self.reader._g(stream)["data"][self.nearest_idx(stream, t)])

    def rows(self, stream: str, idxs: list[int]) -> np.ndarray:
        data = self.reader._g(stream)["data"]
        return np.stack([np.asarray(data[i]) for i in idxs])


class WindowSampler:
    def __init__(self, hw, bb, norm: NormStats, *, student: bool = False,
                 seed: int = 0):
        self.hw = hw
        self.bb = bb
        self.norm = norm
        self.student = student
        self.rng = np.random.default_rng(seed)
        self._cache: dict[Path, _EpisodeCache] = {}
        self._warned_hash: set[str] = set()

    # ------------------------------------------------------------------
    @property
    def latent_dt(self) -> float:
        return self.bb.temporal_comp / self.bb.fps

    def _ep(self, path: Path) -> _EpisodeCache:
        path = Path(path)
        c = self._cache.get(path)
        if c is None:
            c = _EpisodeCache(path)
            self._cache[path] = c
            meta = c.reader.meta
            try:
                cur = self.hw.config_hash()
            except Exception:
                cur = ""
            if meta.config_hash and cur and meta.config_hash != cur \
                    and path.name not in self._warned_hash:
                self._warned_hash.add(path.name)
                log.warning("%s recorded under a different hardware config hash "
                            "(values drifted — shapes are asserted separately)", path.name)
        return c

    # ------------------------------------------------------------------
    def valid_range(self, ep: Path) -> tuple[float, float]:
        """(lo, hi) of admissible t0 anchors."""
        c = self._ep(ep)
        hw, bb = self.hw, self.bb
        streams = [STREAM_CAMERA_SCENE, STREAM_ARM_FT, STREAM_ACTIONS]
        streams += [tactile_stream(s.name, "fields_ds") for s in hw.tactile.sensors]
        start = max(float(c.ts(s)[0]) for s in streams)
        end = min(float(c.ts(s)[-1]) for s in streams)
        chunk_s = hw.control.chunk_horizon / hw.control.action_rate_hz
        video_span = (bb.frames_pix - 1) / bb.fps
        field_dt = 1.0 / hw.recording.field_ds_rate_hz
        past = max(hw.wrist_ft.window_s, chunk_s, 2 * field_dt)
        future = max(video_span, chunk_s, hw.derived.event_lookahead_s) + field_dt
        lo, hi = start + past, end - future
        if hi <= lo:
            raise ValueError(
                f"{Path(ep).name}: episode too short for a window "
                f"(needs > {past + future:.1f}s of stream overlap, has {end - start:.1f}s)")
        return lo, hi

    def build_index(self, root: Path, windows_per_episode: int = 8) -> list[WindowItem]:
        items: list[WindowItem] = []
        for ep in list_episodes(root):
            try:
                lo, hi = self.valid_range(ep)
            except (ValueError, KeyError, FileNotFoundError) as e:
                log.warning("skipping %s: %s", ep.name, e)
                continue
            for t0 in np.sort(self.rng.uniform(lo, hi, size=windows_per_episode)):
                items.append(WindowItem(episode=ep, t0=float(t0)))
        return items

    # ------------------------------------------------------------------
    # sampling internals
    # ------------------------------------------------------------------

    def _field_frame(self, c: _EpisodeCache, sensor: str, t: float) \
            -> tuple[np.ndarray, np.ndarray | None, float]:
        """fields_ds frame at ~t, its predecessor, and their dt."""
        stream = tactile_stream(sensor, "fields_ds")
        i = c.nearest_idx(stream, t)
        ts = c.ts(stream)
        data = c.reader._g(stream)["data"]
        cur = np.asarray(data[i], dtype=np.float32)
        if i == 0:
            return cur, None, 1.0 / self.hw.recording.field_ds_rate_hz
        prev = np.asarray(data[i - 1], dtype=np.float32)
        return cur, prev, float(max(ts[i] - ts[i - 1], 1e-6))

    def _action_grid(self, c: _EpisodeCache, times: np.ndarray) -> np.ndarray:
        idxs = [c.nearest_idx(STREAM_ACTIONS, float(t)) for t in times]
        return c.rows(STREAM_ACTIONS, idxs).astype(np.float32)

    def _rgb_at(self, c: _EpisodeCache, t: float) -> np.ndarray:
        """Camera frame -> (3, res_h, res_w) float32 in [-1, 1]."""
        img = c.at(STREAM_CAMERA_SCENE, t)
        img = bilinear_resize(np.asarray(img, dtype=np.float32),
                              (self.bb.res_h, self.bb.res_w))
        return (img / 127.5 - 1.0).transpose(2, 0, 1)

    def _gel_at(self, c: _EpisodeCache, sensor: str, t: float) -> np.ndarray:
        """Infer image -> (3, res_h, res_w) float32 in [-1, 1] (gray -> 3ch)."""
        img = np.asarray(c.at(tactile_stream(sensor, "infer_img"), t), dtype=np.float32)
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img = bilinear_resize(img, (self.bb.res_h, self.bb.res_w))
        return (img / 127.5 - 1.0).transpose(2, 0, 1)

    # ------------------------------------------------------------------
    def sample(self, ep: Path, t0: float | None = None) -> dict:
        hw, bb, norm = self.hw, self.bb, self.norm
        c = self._ep(ep)
        if t0 is None:
            lo, hi = self.valid_range(ep)
            t0 = float(self.rng.uniform(lo, hi))

        Tc = bb.t_video - 1
        cph, cpw = hw.cpk_shape
        sensors = [s.name for s in hw.tactile.sensors]
        ch = dv.channel_slices(hw.tactile)
        w: dict = {}

        # ---- video: frames_pix frames from t0 forward on the pixel-fps grid
        vid = np.stack([self._rgb_at(c, t0 + j / bb.fps)
                        for j in range(bb.frames_pix)])
        w["video"] = torch.from_numpy(np.ascontiguousarray(vid))

        # ---- wrist F/T window (t0 - window_s, t0], linearly resampled
        ts_ft = c.ts(STREAM_ARM_FT)
        ft = np.asarray(c.reader._g(STREAM_ARM_FT)["data"][:], dtype=np.float32)
        grid = np.linspace(t0 - hw.wrist_ft.window_s, t0, hw.wrist_ft.window_len)
        wrist = np.stack([np.interp(grid, ts_ft, ft[:, k]) for k in range(6)],
                         axis=-1).astype(np.float32)
        w["wrist"] = torch.from_numpy(np.asarray(norm.normalize("wrist_ft", wrist)))

        # ---- UR state at t0
        ur = np.concatenate([
            c.at(STREAM_ARM_Q, t0), c.at(STREAM_ARM_QD, t0),
            c.at(STREAM_ARM_TCP_POSE, t0), c.at(STREAM_ARM_TCP_SPEED, t0),
            c.at(STREAM_GRIPPER, t0),
        ]).astype(np.float32)
        w["ur_state"] = torch.from_numpy(np.asarray(norm.normalize("ur_state", ur)))

        # ---- action chunks on the action grid
        H = hw.control.chunk_horizon
        rate = hw.control.action_rate_hz
        fut = self._action_grid(c, t0 + (np.arange(H) + 1.0) / rate)
        prev = self._action_grid(c, t0 - (H - np.arange(H)) / rate)
        w["action_chunk"] = torch.from_numpy(np.asarray(norm.normalize("action", fut)))
        w["prev_chunk"] = torch.from_numpy(np.asarray(norm.normalize("action", prev)))

        # ---- contact package on the latent grid u_k = t0 + k*latent_dt
        u = t0 + np.arange(Tc + 1) * self.latent_dt
        F = len(sensors)
        ds = np.zeros((Tc + 1, F, cph, cpw, 8), dtype=np.float32)
        mask_frac = np.zeros((Tc + 1, F), dtype=np.float32)
        cop = np.zeros((Tc + 1, F, 2), dtype=np.float32)
        slip = np.zeros((Tc + 1, F), dtype=np.float32)
        for f, sname in enumerate(sensors):
            for k, uk in enumerate(u):
                frame, prev_frame, dt = self._field_frame(c, sname, float(uk))
                d = dv.derive_timestep(frame, prev_frame, dt, hw)
                mask_frac[k, f] = d["mask_frac"]
                cop[k, f] = np.nan_to_num(d["cop"], nan=0.0)
                slip[k, f] = d["slip"]
                ds[k, f] = bilinear_resize(frame, (cph, cpw))

        d_disp = ds[1:, ..., 0:3] - ds[:-1, ..., 0:3]           # (Tc,F,cph,cpw,3)
        d_disp = np.asarray(norm.normalize("cpk_d_disp", d_disp))
        w["cpk_d_disp"] = torch.from_numpy(
            np.ascontiguousarray(d_disp.transpose(0, 1, 4, 2, 3)))
        d_fz = (ds[1:, ..., 7:8] - ds[:-1, ..., 7:8])
        d_fz = np.asarray(norm.normalize("cpk_d_fz", d_fz))[..., 0]
        w["cpk_d_fz"] = torch.from_numpy(np.ascontiguousarray(d_fz))

        tau = hw.derived.tau_contact_depth
        w["cpk_mask"] = torch.from_numpy(
            (np.abs(ds[1:, ..., ch["depth"].start]) > tau).astype(np.float32))
        w["cpk_cop"] = torch.from_numpy(cop[1:])
        w["cpk_slip"] = torch.from_numpy(slip[1:])

        wrench = np.stack([
            np.stack([c.at(tactile_stream(s, "wrench"), float(uk))
                      for s in sensors]) for uk in u[1:]]).astype(np.float32)
        w["cpk_wrench"] = torch.from_numpy(
            np.asarray(norm.normalize("wrench", wrench)))
        wrist_fut = np.stack([np.stack([np.interp(float(uk), ts_ft, ft[:, k])
                                        for k in range(6)]) for uk in u[1:]])
        w["cpk_wrist"] = torch.from_numpy(np.asarray(
            norm.normalize("wrist_ft", wrist_fut.astype(np.float32))))

        # ---- events on the latent grid (any-finger aggregation)
        contact = (mask_frac > 0).any(axis=1)
        slip_any = slip.max(axis=1)
        from phantom.config.model import EVENT_IDX
        ev = np.full(Tc, EVENT_IDX["none"], dtype=np.int64)
        for k in range(1, Tc + 1):
            if contact[k] and not contact[k - 1]:
                ev[k - 1] = EVENT_IDX["onset"]
            elif not contact[k] and contact[k - 1]:
                ev[k - 1] = EVENT_IDX["release"]
            elif contact[k]:
                ev[k - 1] = EVENT_IDX["slip"] if slip_any[k] > hw.derived.tau_slip \
                    else EVENT_IDX["hold"]
        w["events"] = torch.from_numpy(ev)

        # ---- ACC gate label: contact within (t0, t0 + lookahead]
        gate = 0.0
        probes = t0 + np.linspace(0.15, 1.0, 6) * hw.derived.event_lookahead_s
        for sname in sensors:
            for tp in probes:
                frame, _, _ = self._field_frame(c, sname, float(tp))
                if np.abs(frame[..., ch["depth"].start]).max() > tau:
                    gate = 1.0
                    break
            if gate:
                break
        w["gate_label"] = torch.tensor(gate, dtype=torch.float32)

        # ---- teacher-only observation keys
        if not self.student:
            kf = np.stack([np.asarray(c.at(tactile_stream(s, "keyframes"), t0),
                                      dtype=np.float32) for s in sensors])
            w["fields"] = torch.from_numpy(np.asarray(norm.normalize("fields", kf)))
            w["gel"] = torch.from_numpy(np.stack(
                [self._gel_at(c, s, t0) for s in sensors]))
            cs = []
            for f, sname in enumerate(sensors):
                frame, prev_frame, dt = self._field_frame(c, sname, t0)
                d = dv.derive_timestep(frame, prev_frame, dt, hw)
                cs.append(np.concatenate([
                    c.at(tactile_stream(sname, "wrench"), t0).astype(np.float32),
                    np.atleast_1d(np.float32(c.at(tactile_stream(sname, "area"), t0))),
                    np.nan_to_num(d["cop"], nan=0.0).astype(np.float32),
                    np.float32([d["slip"], d["mask_frac"]]),
                ]))
            w["contact_state"] = torch.from_numpy(np.stack(cs))
            cur = np.stack([self._field_frame(c, s, t0)[0] for s in sensors])
            prv = np.stack([self._field_frame(c, s, t0)[1]
                            if self._field_frame(c, s, t0)[1] is not None
                            else self._field_frame(c, s, t0)[0] for s in sensors])
            w["reactive"] = torch.tensor(dv.reactive_score(cur, prv),
                                         dtype=torch.float32)

        # ---- floats + language
        for k, v in list(w.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                w[k] = v.float()
        w["text"] = c.reader.meta.text
        w["task"] = c.reader.meta.task
        return w
