"""Tactile prediction error (TPE): how well the policy's imagined tactile
future matched what the pads then measured.

A world-action model earns its name by predicting the observation stream it
acts on; for PHANTOM that stream is the contact package (pipeline.md §2/§4).
`phantom/deploy/cpk_trace.py` keeps every replan's predicted package in
`planner_cpk.npz`; this module rebuilds the OBSERVED package from the
recorded `tactile_<pad>_fields_ds` streams on the same latent grid — the
same construction WindowSampler uses for the training target, minus the
normalization — and scores prediction against observation per future step.

Every error is reported next to a PERSISTENCE baseline (predict "nothing
changes": zero deltas, current mask / CoP / slip / contact state). The skill
score 1 - err_model / err_persist is the quantity to put in the paper: 0 =
no better than assuming the pads stay as they are, 1 = perfect.

Clock: `planner_cpk.npz.t_host` is host time (like `planner_trace.json`);
stream `ts` are master time (`meta.clock_calibration.offset`, issue #10).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.config.model import EVENT_IDX, N_EVENTS
from phantom.data import derived as dv
from phantom.data.episode_store import EpisodeReader
from phantom.data.schema import tactile_stream
from phantom.deploy.cpk_trace import CPK_TRACE_FILE, load_cpk_trace

# per-step score keys -> (model, persistence) columns; "lower" / "higher" = better
STEP_METRICS: dict[str, str] = {
    "dfz_se": "lower",          # mean squared d_fz error over pads x cells
    "mask_iou": "higher",       # contact-mask IoU (1 when both empty)
    "mask_frac_err": "lower",   # |predicted - observed contact fraction|
    "cop_err": "lower",         # CoP L2 in [-1,1]^2 (pads where both defined)
    "slip_err": "lower",        # |predicted - observed slip|
    "event_acc": "higher",      # argmax event == derived label
    "wrench_se": "lower",       # pad wrench MSE (N, Nm), baseline-subtracted
}
WRENCH_BASELINE_ROWS = 8      # mirrors data/windows.py _EpisodeCache


# ---------------------------------------------------------------------------
# observed package (WindowSampler's target construction, unnormalized)
# ---------------------------------------------------------------------------

def _host_to_master_offset(ep_path: Path) -> float:
    try:
        meta = json.loads((Path(ep_path) / "meta.json").read_text(encoding="utf-8"))
        return float((meta.get("clock_calibration") or {}).get("offset", 0.0) or 0.0)
    except Exception:
        return 0.0


def _nearest_idx(ts: np.ndarray, t: float) -> int:
    i = int(np.searchsorted(ts, t))
    if i <= 0:
        return 0
    if i >= len(ts):
        return len(ts) - 1
    return i if (ts[i] - t) < (t - ts[i - 1]) else i - 1


class _Streams:
    """Lazy per-sensor access to fields_ds (+ wrench) with cached timestamps."""

    def __init__(self, ep: EpisodeReader, hw: HardwareConfig):
        self.ep, self.hw = ep, hw
        self.sensors = [s.name for s in hw.tactile.sensors]
        self._ts: dict[str, np.ndarray] = {}
        self._data: dict[str, object] = {}
        self._wbase: dict[str, np.ndarray] = {}

    def ts(self, stream: str) -> np.ndarray:
        if stream not in self._ts:
            self._ts[stream] = self.ep.ts(stream)
        return self._ts[stream]

    def data(self, stream: str):
        if stream not in self._data:
            self._data[stream] = self.ep._g(stream)["data"]
        return self._data[stream]

    def field_frame(self, sensor: str, t: float):
        stream = tactile_stream(sensor, "fields_ds")
        ts = self.ts(stream)
        i = _nearest_idx(ts, t)
        data = self.data(stream)
        cur = np.asarray(data[i], dtype=np.float32)
        if i == 0:
            return cur, None, 1.0 / self.hw.recording.field_ds_rate_hz
        prev = np.asarray(data[i - 1], dtype=np.float32)
        return cur, prev, float(max(ts[i] - ts[i - 1], 1e-6))

    def wrench_at(self, sensor: str, t: float) -> np.ndarray | None:
        stream = tactile_stream(sensor, "wrench")
        if not self.ep.has(stream):
            return None
        ts = self.ts(stream)
        if stream not in self._wbase:
            w = np.asarray(self.data(stream)[:WRENCH_BASELINE_ROWS], dtype=np.float32)
            self._wbase[stream] = (np.median(w, axis=0).astype(np.float32)
                                   if len(w) else np.zeros(6, np.float32))
        row = np.asarray(self.data(stream)[_nearest_idx(ts, t)], dtype=np.float32)
        return row - self._wbase[stream]

    def covers(self, t_lo: float, t_hi: float) -> bool:
        """Every pad's field stream spans [t_lo, t_hi] (within half a period)."""
        tol = 0.5 / self.hw.recording.field_ds_rate_hz
        for s in self.sensors:
            stream = tactile_stream(s, "fields_ds")
            if not self.ep.has(stream):
                return False
            ts = self.ts(stream)
            if not len(ts) or t_lo < ts[0] - tol or t_hi > ts[-1] + tol:
                return False
        return True


def observed_package(streams: _Streams, hw: HardwareConfig, t0: float,
                     latent_dt: float, Tc: int) -> dict | None:
    """The measured contact package on u_k = t0 + k*latent_dt, k = 0..Tc
    (master clock), in recorder units. None when the horizon leaves the
    recording (a replan issued in the last `Tc*latent_dt` of an episode is
    not scoreable — explicit, never padded)."""
    from phantom.data.windows import bilinear_resize   # torch-importing module

    u = t0 + np.arange(Tc + 1) * float(latent_dt)
    if not streams.covers(float(u[0]), float(u[-1])):
        return None
    cph, cpw = hw.cpk_shape
    ch = dv.channel_slices(hw.tactile)
    F = len(streams.sensors)
    ds = np.zeros((Tc + 1, F, cph, cpw, hw.tactile.field_ch), dtype=np.float32)
    mask_frac = np.zeros((Tc + 1, F), dtype=np.float32)
    cop = np.full((Tc + 1, F, 2), np.nan, dtype=np.float32)
    slip = np.zeros((Tc + 1, F), dtype=np.float32)
    wrench = np.full((Tc + 1, F, 6), np.nan, dtype=np.float32)
    for f, sname in enumerate(streams.sensors):
        for k, uk in enumerate(u):
            frame, prev_frame, dt = streams.field_frame(sname, float(uk))
            d = dv.derive_timestep(frame, prev_frame, dt, hw)
            mask_frac[k, f] = d["mask_frac"]
            cop[k, f] = d["cop"]
            slip[k, f] = d["slip"]
            ds[k, f] = bilinear_resize(frame, (cph, cpw))
            w = streams.wrench_at(sname, float(uk))
            if w is not None:
                wrench[k, f] = w
    fz = ch["dist_force"].start + 2            # channel 7 in the canonical stack
    tau = hw.derived.tau_contact_depth
    depth = ch["depth"].start
    contact = (mask_frac > hw.derived.tau_contact_area).any(axis=1)     # (Tc+1,)
    slip_any = slip.max(axis=1)
    ev = np.full(Tc, EVENT_IDX["none"], dtype=np.int64)
    for k in range(1, Tc + 1):
        if contact[k] and not contact[k - 1]:
            ev[k - 1] = EVENT_IDX["onset"]
        elif not contact[k] and contact[k - 1]:
            ev[k - 1] = EVENT_IDX["release"]
        elif contact[k]:
            ev[k - 1] = (EVENT_IDX["slip"] if slip_any[k] > hw.derived.tau_slip
                         else EVENT_IDX["hold"])
    return {
        "d_fz": ds[1:, ..., fz] - ds[:-1, ..., fz],                    # (Tc,F,cph,cpw)
        "d_disp": ds[1:, ..., 0:3] - ds[:-1, ..., 0:3],                # (Tc,F,cph,cpw,3)
        "mask": (np.abs(ds[1:, ..., depth]) > tau).astype(np.float32),
        "mask0": (np.abs(ds[0, ..., depth]) > tau).astype(np.float32),  # (F,cph,cpw)
        "cop": cop[1:], "cop0": cop[0],
        "slip": slip[1:], "slip0": slip[0],
        "wrench": wrench[1:], "wrench0": wrench[0],
        "event": ev,
        "contact": contact[1:], "contact0": bool(contact[0]),
    }


# ---------------------------------------------------------------------------
# per-replan scoring
# ---------------------------------------------------------------------------

def _nanmean(x, axis=None) -> float | np.ndarray:
    """nanmean that returns NaN silently on all-NaN input (an undefined CoP
    on every pad is a legitimate outcome, not a warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 1.0 if union == 0.0 else inter / union


def score_replan(pred: dict, obs: dict) -> dict[str, np.ndarray]:
    """Per-step (Tc,) scores of one predicted package vs the observed one;
    `<key>` = model, `<key>_persist` = persistence baseline. Also returns
    `contact` (observed any-pad contact at the step end) for conditioning."""
    Tc = obs["d_fz"].shape[0]
    out: dict[str, np.ndarray] = {}
    out["dfz_se"] = ((pred["d_fz"] - obs["d_fz"]) ** 2).reshape(Tc, -1).mean(axis=1)
    out["dfz_se_persist"] = (obs["d_fz"] ** 2).reshape(Tc, -1).mean(axis=1)

    pm = pred["mask"] > 0.5
    om = obs["mask"] > 0.5
    m0 = obs["mask0"] > 0.5
    out["mask_iou"] = np.array([_iou(pm[s], om[s]) for s in range(Tc)])
    out["mask_iou_persist"] = np.array([_iou(m0, om[s]) for s in range(Tc)])
    frac_o = om.reshape(Tc, om.shape[1], -1).mean(axis=-1)                # (Tc,F)
    out["mask_frac_err"] = np.abs(pm.reshape(Tc, pm.shape[1], -1).mean(-1) - frac_o).mean(1)
    frac_0 = m0.reshape(m0.shape[0], -1).mean(-1)                          # (F,)
    out["mask_frac_err_persist"] = np.abs(frac_0[None] - frac_o).mean(1)

    def cop_err(p: np.ndarray, o: np.ndarray) -> np.ndarray:
        ok = np.isfinite(p).all(-1) & np.isfinite(o).all(-1)              # (Tc,F)
        d = np.linalg.norm(np.nan_to_num(p) - np.nan_to_num(o), axis=-1)
        d = np.where(ok, d, np.nan)
        return _nanmean(d, axis=1)
    out["cop_err"] = cop_err(pred["cop"], obs["cop"])
    out["cop_err_persist"] = cop_err(np.repeat(obs["cop0"][None], Tc, 0), obs["cop"])

    out["slip_err"] = np.abs(pred["slip"] - obs["slip"]).mean(1)
    out["slip_err_persist"] = np.abs(obs["slip0"][None] - obs["slip"]).mean(1)

    ev_pred = np.argmax(pred["event"], axis=-1)
    ev_persist = np.full(Tc, EVENT_IDX["hold"] if obs["contact0"] else EVENT_IDX["none"])
    out["event_acc"] = (ev_pred == obs["event"]).astype(float)
    out["event_acc_persist"] = (ev_persist == obs["event"]).astype(float)

    ow = obs["wrench"]
    if np.isfinite(ow).all() and "wrench" in pred:
        out["wrench_se"] = ((pred["wrench"] - ow) ** 2).reshape(Tc, -1).mean(1)
        out["wrench_se_persist"] = ((obs["wrench0"][None] - ow) ** 2).reshape(Tc, -1).mean(1)
    else:
        out["wrench_se"] = np.full(Tc, np.nan)
        out["wrench_se_persist"] = np.full(Tc, np.nan)
    out["contact"] = np.asarray(obs["contact"], dtype=bool)
    return out


# ---------------------------------------------------------------------------
# per-episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeTPE:
    episode: str
    n_replans: int
    n_scored: int
    latent_dt: float
    Tc: int
    dist_force_unit_to_N: float
    denormalized: bool
    steps: dict[str, np.ndarray] = field(default_factory=dict)   # (n_scored, Tc)
    t_host: np.ndarray = field(default_factory=lambda: np.zeros(0))
    skipped_reason: str = ""

    def summary(self) -> dict:
        return summarize_steps(self.steps, Tc=self.Tc,
                               fz_unit_to_N=self.dist_force_unit_to_N,
                               n_total=self.n_replans, n_scored=self.n_scored)


def _skill(err_model: float, err_persist: float) -> float:
    if not np.isfinite(err_model) or not np.isfinite(err_persist) or err_persist <= 0:
        return float("nan")
    return float(1.0 - err_model / err_persist)


def summarize_steps(steps: dict[str, np.ndarray], *, Tc: int,
                    fz_unit_to_N: float = 0.0, n_total: int = 0,
                    n_scored: int = 0) -> dict:
    """Scalar digest of (n, Tc) step matrices (pooled over replans, i.e.
    every scored step weighs the same). Keys are `tpe_*` so they compose
    with `metrics.trial_metrics` / aggregate's per-system means."""
    out: dict[str, float] = {"tpe_n_replans_total": float(n_total),
                             "tpe_n_replans_scored": float(n_scored)}
    if n_scored == 0 or not steps:
        return out
    with np.errstate(all="ignore"):
        se, sep = steps["dfz_se"], steps["dfz_se_persist"]
        rm, rmp = math.sqrt(float(_nanmean(se))), math.sqrt(float(_nanmean(sep)))
        out["tpe_dfz_rmse"], out["tpe_dfz_rmse_persist"] = rm, rmp
        out["tpe_dfz_skill"] = _skill(rm, rmp)
        c = steps["contact"]
        if c.any():
            rmc = math.sqrt(float(_nanmean(se[c])))
            rmcp = math.sqrt(float(_nanmean(sep[c])))
            out["tpe_dfz_rmse_contact"], out["tpe_dfz_rmse_contact_persist"] = rmc, rmcp
            out["tpe_dfz_skill_contact"] = _skill(rmc, rmcp)
        else:
            out["tpe_dfz_rmse_contact"] = out["tpe_dfz_rmse_contact_persist"] = np.nan
            out["tpe_dfz_skill_contact"] = np.nan
        out["tpe_contact_step_frac"] = float(c.mean())
        if fz_unit_to_N and fz_unit_to_N > 0:
            out["tpe_dfz_rmse_N"] = rm * fz_unit_to_N
            out["tpe_dfz_rmse_persist_N"] = rmp * fz_unit_to_N
        for k in ("mask_iou", "mask_frac_err", "cop_err", "slip_err", "event_acc",
                  "wrench_se"):
            out[f"tpe_{k}"] = float(_nanmean(steps[k]))
            out[f"tpe_{k}_persist"] = float(_nanmean(steps[f"{k}_persist"]))
        out["tpe_wrench_rmse"] = math.sqrt(out.pop("tpe_wrench_se")) \
            if np.isfinite(out["tpe_wrench_se"]) else np.nan
        out["tpe_wrench_rmse_persist"] = math.sqrt(out.pop("tpe_wrench_se_persist")) \
            if np.isfinite(out["tpe_wrench_se_persist"]) else np.nan
        out["tpe_mask_iou_skill"] = _skill(1.0 - out["tpe_mask_iou"],
                                           1.0 - out["tpe_mask_iou_persist"])
        out["tpe_event_acc_skill"] = _skill(1.0 - out["tpe_event_acc"],
                                            1.0 - out["tpe_event_acc_persist"])
        # horizon-resolved: step s+1 ahead (s * latent_dt .. (s+1) * latent_dt)
        for s in range(Tc):
            out[f"tpe_dfz_rmse_s{s + 1}"] = math.sqrt(float(_nanmean(se[:, s])))
            out[f"tpe_mask_iou_s{s + 1}"] = float(_nanmean(steps["mask_iou"][:, s]))
            out[f"tpe_event_acc_s{s + 1}"] = float(_nanmean(steps["event_acc"][:, s]))
    return out


def episode_tactile_prediction(ep_path: Path, hw: HardwareConfig, *,
                               ep: EpisodeReader | None = None) -> EpisodeTPE | None:
    """Score every replan of an episode. None when the episode has no
    `planner_cpk.npz` (policy without a package, or pre-feature episode)."""
    ep_path = Path(ep_path)
    z = load_cpk_trace(ep_path)
    if z is None:
        return None
    n = int(z["t_host"].shape[0])
    Tc = int(z["d_fz"].shape[1])
    latent_dt = float(z["latent_dt"])
    res = EpisodeTPE(episode=ep_path.name, n_replans=n, n_scored=0, latent_dt=latent_dt,
                     Tc=Tc, dist_force_unit_to_N=float(z.get("dist_force_unit_to_N", 0.0)),
                     denormalized=bool(int(z.get("denormalized", 1))),
                     t_host=np.asarray(z["t_host"], dtype=float))
    if not np.isfinite(latent_dt) or latent_dt <= 0:
        res.skipped_reason = "latent_dt unknown"
        return res
    if not res.denormalized:
        res.skipped_reason = "package stored in model space (no norm stats at deploy)"
        return res
    ep = ep or EpisodeReader(ep_path)
    streams = _Streams(ep, hw)
    offset = _host_to_master_offset(ep_path)
    rows: list[dict[str, np.ndarray]] = []
    for i in range(n):
        obs = observed_package(streams, hw, float(z["t_host"][i]) + offset, latent_dt, Tc)
        if obs is None:
            continue
        pred = {k: np.asarray(z[k][i], dtype=np.float32)
                for k in ("d_fz", "mask", "cop", "slip", "event", "wrench")}
        rows.append(score_replan(pred, obs))
    res.n_scored = len(rows)
    if rows:
        res.steps = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    return res


def tactile_prediction_summary(ep_path: Path, hw: HardwareConfig, *,
                               ep: EpisodeReader | None = None) -> dict:
    """`trial_metrics` hook: the `tpe_*` scalars, or {} when not available."""
    try:
        r = episode_tactile_prediction(ep_path, hw, ep=ep)
    except Exception:
        return {}
    return {} if r is None else r.summary()


# ---------------------------------------------------------------------------
# across episodes: pooled curves, CIs, paired comparison (numpy only)
# ---------------------------------------------------------------------------

def pool_steps(results: list[EpisodeTPE]) -> dict[str, np.ndarray]:
    """Concatenate scored (n, Tc) matrices over episodes."""
    rs = [r for r in results if r.n_scored]
    if not rs:
        return {}
    keys = rs[0].steps.keys()
    return {k: np.concatenate([r.steps[k] for r in rs]) for k in keys}


def horizon_curves(results: list[EpisodeTPE]) -> dict:
    """Per-step means (model vs persistence) for the horizon figure."""
    pooled = pool_steps(results)
    if not pooled:
        return {}
    Tc = pooled["dfz_se"].shape[1]
    out = {"steps_ahead": list(range(1, Tc + 1)), "n_replans": int(pooled["dfz_se"].shape[0])}
    with np.errstate(all="ignore"):
        out["dfz_rmse"] = [math.sqrt(float(_nanmean(pooled["dfz_se"][:, s]))) for s in range(Tc)]
        out["dfz_rmse_persist"] = [math.sqrt(float(_nanmean(pooled["dfz_se_persist"][:, s])))
                                   for s in range(Tc)]
        for k in ("mask_iou", "mask_frac_err", "cop_err", "slip_err", "event_acc"):
            out[k] = [float(_nanmean(pooled[k][:, s])) for s in range(Tc)]
            out[f"{k}_persist"] = [float(_nanmean(pooled[f"{k}_persist"][:, s]))
                                   for s in range(Tc)]
    return out


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over episodes (NaNs dropped)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def sign_test_p(diffs) -> float:
    """Exact two-sided sign test on paired differences (zeros dropped)."""
    d = np.asarray([x for x in diffs if np.isfinite(x) and x != 0.0], dtype=float)
    n = d.size
    if n == 0:
        return float("nan")
    k = int((d > 0).sum())
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(lo + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * p))


def paired_compare(a, b, *, higher_is_better: bool = True, n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Paired A-B over matched episodes: mean difference (A − B), bootstrap
    CI of the mean difference, exact sign-test p, and how many pairs favour
    each side (`higher_is_better=False` for error metrics)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    a_wins, b_wins = (d > 0, d < 0) if higher_is_better else (d < 0, d > 0)
    return {"n_pairs": int(d.size), "mean_diff": float(d.mean()) if d.size else float("nan"),
            "diff_ci_lo": lo, "diff_ci_hi": hi, "sign_p": sign_test_p(d),
            "higher_is_better": bool(higher_is_better),
            "n_a_better": int(a_wins.sum()), "n_b_better": int(b_wins.sum()),
            "n_ties": int((d == 0).sum())}


__all__ = ["CPK_TRACE_FILE", "STEP_METRICS", "EpisodeTPE", "observed_package",
           "score_replan", "summarize_steps", "episode_tactile_prediction",
           "tactile_prediction_summary", "pool_steps", "horizon_curves",
           "bootstrap_ci", "sign_test_p", "paired_compare"]
