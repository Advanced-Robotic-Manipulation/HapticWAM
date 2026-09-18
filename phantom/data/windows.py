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
from phantom.config.model import EVENT_IDX
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ARM_FT, STREAM_ARM_Q,
                                 STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                                 STREAM_ARM_TCP_SPEED, STREAM_CAMERA_SCENE,
                                 STREAM_GRIPPER, EpisodeMeta, NormStats,
                                 is_deliberate_failure_demo, is_failure_demo,
                                 is_policy_rollout, is_trainable_episode,
                                 needs_rederive, tactile_stream)

log = logging.getLogger(__name__)

#: `action_weight` policy for ON-POLICY (DAgger) rollouts:
#:   "failure_demo"    — the shipped rule: `is_failure_demo` fires on a
#:                       `success is False` verdict, so every judged-failed
#:                       rollout grounds NO actions. Every v5/v6 student was
#:                       distilled under this.
#:   "judged_rollouts" — a POLICY rollout that is failure-marked ONLY by the
#:                       verdict keeps its per-episode weight, so its own
#:                       re-derived actions ground the action term on
#:                       on-policy states. Deliberate failure demos (SOP tag /
#:                       `<task>_fail`) still ground nothing, and a rollout
#:                       whose actions are still the executor PROPOSAL (no
#:                       tools/rederive_rollout_actions.py) is left at 0 —
#:                       imitating the pre-clamp proposal is the defect F13
#:                       fixed.
#: The mode is named for WHICH EPISODES it re-admits, not for a supervisor:
#: it grounds the rollout's OWN re-derived (measured delta-EE) actions on the
#: episodes a verdict judged, and no teacher ever relabels them.
ROLLOUT_ACTION_WEIGHT_MODES = ("failure_demo", "judged_rollouts")


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
    # admissible anchor range for this episode, so a dataset can redraw t0
    # instead of replaying one frozen anchor for the whole run
    lo: float = 0.0
    hi: float = 0.0


# ---------------------------------------------------------------------------

# Simulated episodes with idle (unmodelled) pads: tools/sim/export_expert_episode.py
PADS_MASKED_TAG = "pads_masked"
STREAM_CONTACT_GT = "contact_gt"     # (T, 3) [any pad in contact, left N, right N]


class _EpisodeCache:
    """Reader + per-stream timestamp cache for one episode."""

    WRENCH_BASELINE_ROWS = 8      # ~1 s at the 8 Hz SDK rate, pads untouched at the start pose

    def __init__(self, path: Path):
        self.reader = EpisodeReader(path)
        self._wrench_base: dict[str, np.ndarray] = {}
        self._ts: dict[str, np.ndarray] = {}

    def wrench_baseline(self, stream: str, rows: int = WRENCH_BASELINE_ROWS) -> np.ndarray:
        """Per-episode zero offset of a pad's wrench stream: the median of its
        first WRENCH_BASELINE_ROWS rows (the pads idle untouched at the start
        pose). The left pad idles 0.7-2.2 N above zero and the offset drifts
        per session — fed raw, it is a session ID the model can fit and
        misread at deploy time (v6 data fix, docs/v6 plan B.1). The deploy
        SnapshotBuilder subtracts the same statistic from its ring."""
        if rows <= 0:
            return np.zeros(6, np.float32)
        if stream not in self._wrench_base:
            w = np.asarray(self.reader._g(stream)["data"][: int(rows)],
                           dtype=np.float32)
            self._wrench_base[stream] = (np.median(w, axis=0).astype(np.float32)
                                         if len(w) else np.zeros(6, np.float32))
        return self._wrench_base[stream]

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

    def future_idx(self, stream: str, t: float) -> int:
        """First sample at time >= t (strict-future lookup).

        Targets must never resolve backwards: `nearest_idx` can return the
        row BEFORE t, i.e. an action that already executed and whose effect is
        already visible in the conditioning — the model would be rewarded for
        'predicting' the past."""
        ts = self.ts(stream)
        i = int(np.searchsorted(ts, t, side="left"))
        return min(i, len(ts) - 1)

    def at(self, stream: str, t: float) -> np.ndarray:
        # reader.data() decodes JPEG camera streams transparently
        return np.asarray(self.reader.data(stream)[self.nearest_idx(stream, t)])

    def rows(self, stream: str, idxs: list[int]) -> np.ndarray:
        data = self.reader.data(stream)
        return np.stack([np.asarray(data[i]) for i in idxs])

    def contact_gt_at(self, t: float) -> bool:
        """Physical pad-object contact at ~t (pads-masked sim episodes)."""
        if not self.reader.has(STREAM_CONTACT_GT):
            raise KeyError(f"{self.reader.path.name}: tagged {PADS_MASKED_TAG} but has no "
                           f"{STREAM_CONTACT_GT} stream")
        return bool(float(self.at(STREAM_CONTACT_GT, t)[0]) > 0.5)


class WindowSampler:
    def __init__(self, hw, bb, norm: NormStats, *, student: bool = False,
                 seed: int = 0, wrench_baseline_rows: int = 0,
                 rollout_action_weight: str = "failure_demo"):
        if rollout_action_weight not in ROLLOUT_ACTION_WEIGHT_MODES:
            raise ValueError(
                f"rollout_action_weight={rollout_action_weight!r}: use one of "
                f"{ROLLOUT_ACTION_WEIGHT_MODES}")
        self.rollout_action_weight = rollout_action_weight
        self.hw = hw
        self.bb = bb
        self.norm = norm
        self.student = student
        # 0 = raw wrench (every checkpoint before v6); N = per-episode zero
        # offset subtracted (TeacherTrainConfig.wrench_baseline_rows)
        self.wrench_baseline_rows = int(wrench_baseline_rows or 0)
        self.rng = np.random.default_rng(seed)
        self._cache: dict[Path, _EpisodeCache] = {}
        self._warned_hash: set[str] = set()
        self.n_config_drift = 0        # episodes recorded under another config

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
                self.n_config_drift += 1
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

    def build_index(self, root: Path, windows_per_episode: int = 8,
                    episodes: list[Path] | None = None) -> list[WindowItem]:
        """Index of (episode, t0) windows.

        `episodes` overrides filesystem enumeration (used to honour a manifest
        train/val split). The t0 values are only the INITIAL draw — see
        WindowDataset, which resamples t0 per access so a long run does not
        replay one frozen set of windows."""
        items: list[WindowItem] = []
        for ep in (list_episodes(root) if episodes is None else episodes):
            # P9 (review 2026-08-28): the LAST gate before windows exist. The
            # DAgger path reaches here directly — dagger_driver hands a rollout
            # root to distill_hid --extra-data, which calls build_index with no
            # manifest at all — so an unjudged or contaminated rollout would
            # otherwise be indexed and grounded at action_weight 1.0 on exactly
            # the on-policy states the model already gets wrong.
            try:
                # read meta.json directly (not through the episode cache): a
                # rejected episode must not cost a zarr open, and the check
                # must see what is on disk now
                meta = EpisodeMeta.load(Path(ep) / "meta.json")
            except Exception as e:                       # unreadable meta.json
                log.warning("skipping %s: %s", ep.name, e)
                continue
            if not is_trainable_episode(meta):
                log.warning("skipping %s: not training-ready (status=%r "
                            "tags=%s policy=%r success=%r)", ep.name,
                            meta.status, meta.tags, meta.policy, meta.success)
                continue
            if needs_rederive(ep, meta):
                # not fatal here (the intake manifest REFUSES it — F13), but
                # the --extra-data path bypasses the manifest entirely and
                # would train on the executor's pre-clamp proposal at the
                # wrong cadence with no sign anything is wrong
                log.warning("%s: policy rollout with NO re-derived actions "
                            "(no actions_plan.zarr) — its actions stream is "
                            "the executor PROPOSAL; run "
                            "tools/rederive_rollout_actions.py before training "
                            "on it", ep.name)
            try:
                lo, hi = self.valid_range(ep)
            except (ValueError, KeyError, FileNotFoundError) as e:
                log.warning("skipping %s: %s", ep.name, e)
                continue
            for t0 in np.sort(self.rng.uniform(lo, hi, size=windows_per_episode)):
                items.append(WindowItem(episode=ep, t0=float(t0), lo=lo, hi=hi))
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

    def _action_grid(self, c: _EpisodeCache, times: np.ndarray, *,
                     future: bool = False) -> np.ndarray:
        pick = c.future_idx if future else c.nearest_idx
        idxs = [pick(STREAM_ACTIONS, float(t)) for t in times]
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
        fut = self._action_grid(c, t0 + (np.arange(H) + 1.0) / rate, future=True)
        prev = self._action_grid(c, t0 - (H - np.arange(H)) / rate)
        w["action_chunk"] = torch.from_numpy(np.asarray(norm.normalize("action", fut)))
        w["prev_chunk"] = torch.from_numpy(np.asarray(norm.normalize("action", prev)))

        # ---- contact package on the latent grid u_k = t0 + k*latent_dt
        u = t0 + np.arange(Tc + 1) * self.latent_dt
        F = len(sensors)
        ds = np.zeros((Tc + 1, F, cph, cpw, 8), dtype=np.float32)
        mask_frac = np.zeros((Tc + 1, F), dtype=np.float32)
        # NaN = "no CoP" (derived.py returns NaN below cop_min_contact_cells).
        # ContactPacker keys the CoP-bump amplitude on that NaN; zeroing it
        # here packed every no-contact timestep as a full-amplitude bump at
        # the canvas centre — a fictitious CoP target in contact_nll, worst on
        # under-grasp failure demos (Codex review 2026-08-26). Observation
        # features below still use nan_to_num (they must be finite inputs).
        cop = np.full((Tc + 1, F, 2), np.nan, dtype=np.float32)
        slip = np.zeros((Tc + 1, F), dtype=np.float32)
        for f, sname in enumerate(sensors):
            for k, uk in enumerate(u):
                frame, prev_frame, dt = self._field_frame(c, sname, float(uk))
                d = dv.derive_timestep(frame, prev_frame, dt, hw)
                mask_frac[k, f] = d["mask_frac"]
                cop[k, f] = d["cop"]
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

        # the TARGET loses the same per-episode offset as the input: a model
        # that cannot observe a session constant must not be asked to predict it
        wrench = np.stack([
            np.stack([c.at(tactile_stream(s, "wrench"), float(uk))
                      - c.wrench_baseline(tactile_stream(s, "wrench"), self.wrench_baseline_rows)
                      for s in sensors]) for uk in u[1:]]).astype(np.float32)
        w["cpk_wrench"] = torch.from_numpy(
            np.asarray(norm.normalize("wrench", wrench)))
        wrist_fut = np.stack([np.stack([np.interp(float(uk), ts_ft, ft[:, k])
                                        for k in range(6)]) for uk in u[1:]])
        w["cpk_wrist"] = torch.from_numpy(np.asarray(
            norm.normalize("wrist_ft", wrist_fut.astype(np.float32))))

        # ---- events on the latent grid (any-finger aggregation)
        # frame-level contact needs an AREA of pixels, not one: `> 0` meant a
        # single hot pixel = contact, saturating events to 96% "hold" (issue
        # #1; see DerivedConfig.tau_contact_area)
        contact = (mask_frac > hw.derived.tau_contact_area).any(axis=1)
        slip_any = slip.max(axis=1)
        ev = np.full(Tc, EVENT_IDX["none"], dtype=np.int64)
        for k in range(1, Tc + 1):
            if contact[k] and not contact[k - 1]:
                ev[k - 1] = EVENT_IDX["onset"]
            elif not contact[k] and contact[k - 1]:
                ev[k - 1] = EVENT_IDX["release"]
            elif contact[k]:
                ev[k - 1] = EVENT_IDX["slip"] if slip_any[k] > hw.derived.tau_slip \
                    else EVENT_IDX["hold"]
        # ---- ACC gate label: contact within (t0, t0 + lookahead]
        gate = 0.0
        probes = t0 + np.linspace(0.15, 1.0, 6) * hw.derived.event_lookahead_s
        pads_masked = PADS_MASKED_TAG in {str(x) for x in (c.reader.meta.tags or [])}
        if pads_masked:
            # simulated episode: the pads carry idle rows (no gel model), so the
            # contact events and the gate come from the sim's physical
            # pad-object contact (`contact_gt`, tools/sim/export_expert_episode.py)
            # and the tactile reconstruction losses are switched off for the
            # window (contact_weight 0). Slip is never labelled in sim.
            contact = np.array([c.contact_gt_at(float(uk)) for uk in u], dtype=bool)
            ev = np.full(Tc, EVENT_IDX["none"], dtype=np.int64)
            for k in range(1, Tc + 1):
                if contact[k] and not contact[k - 1]:
                    ev[k - 1] = EVENT_IDX["onset"]
                elif not contact[k] and contact[k - 1]:
                    ev[k - 1] = EVENT_IDX["release"]
                elif contact[k]:
                    ev[k - 1] = EVENT_IDX["hold"]
            gate = 1.0 if any(c.contact_gt_at(float(tp)) for tp in probes) else 0.0
        else:
            for sname in sensors:
                for tp in probes:
                    frame, _, _ = self._field_frame(c, sname, float(tp))
                    # area statistic, not max — max over 110k pixels fires on any
                    # single noisy pixel (gate labels were 100% positive; issue #1)
                    if (np.abs(frame[..., ch["depth"].start]) > tau).mean() \
                            > hw.derived.tau_contact_area:
                        gate = 1.0
                        break
                if gate:
                    break
        w["events"] = torch.from_numpy(ev)
        w["gate_label"] = torch.tensor(gate, dtype=torch.float32)
        w["contact_weight"] = 0.0 if pads_masked else 1.0

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
                wst = tactile_stream(sname, "wrench")
                cs.append(np.concatenate([
                    (c.at(wst, t0).astype(np.float32)
                     - c.wrench_baseline(wst, self.wrench_baseline_rows)),
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
        # mirror embed_task_texts.collect_texts: text falls back to the task
        # name, so an episode with the optional instruction left blank still
        # hits the cache instead of silently training unconditioned
        w["text"] = c.reader.meta.text or c.reader.meta.task
        w["task"] = c.reader.meta.task
        # Deliberate-failure demos (controlled over-squeezes / induced slips)
        # must supervise the contact, event and gate heads — that is what they
        # were recorded for — but must NEVER supervise action imitation, or the
        # teacher learns to reproduce the failure. They are marked three
        # different ways in practice: an explicit failure verdict, the SOP tag,
        # or (as recorded on this rig) a `<task>_fail` task name with
        # success=True meaning "the episode successfully captured the intended
        # failure". Any of the three disables the action loss for the window.
        # ... and a per-episode multiplier on top (D8 self-improvement,
        # validation 2026-08-30 F19): intake writes `weight` into meta.json
        # (1.0 demos and recoveries, >1 to oversample a small on-policy
        # rollout pool). It can only scale a window that is already allowed to
        # supervise actions — a failure demo stays at 0 whatever it says.
        #
        # `rollout_action_weight="judged_rollouts"` (opt-in) carves ONE
        # case out of that rule: an on-policy POLICY rollout whose only
        # failure marker is the operator's verdict is not a staged failure,
        # and zeroing it is what made the round-2 DAgger overlay inert. It
        # still needs re-derived actions to be worth grounding (F13).
        meta = c.reader.meta
        ew = getattr(meta, "weight", 1.0)
        ew = 1.0 if ew is None else max(0.0, float(ew))
        w["action_weight"] = 0.0 if is_failure_demo(meta) else ew
        if (w["action_weight"] == 0.0
                and self.rollout_action_weight == "judged_rollouts"
                and is_policy_rollout(meta)
                and not is_deliberate_failure_demo(meta)
                and not needs_rederive(c.reader.path, meta)):
            w["action_weight"] = ew
        return w
