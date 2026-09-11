"""Per-replan record of the policy's PREDICTED contact package.

Every `PhantomPolicy.replan` returns a `ContactPackage` — the model's
imagined tactile future on the latent grid (Tc steps of `latent_dt`).
Until now only its readouts (gate, p_evt, sigma) reached
`planner_trace.json`; the package itself was consumed by the next replan's
ACC input and discarded. This module keeps it, DENORMALIZED to the units
the recorder writes, so that after the episode the prediction at step s
can be differenced against what the pads actually measured at
t + (s + 1) * latent_dt (`phantom/eval/tactile_prediction.py`).

Storage: one `planner_cpk.npz` per episode (arrays stacked over replans;
the compact per-step summary also lands in the trace row as `cpk_pred`
so a trace alone still shows what the model expected).

Time convention (same as WindowSampler's target grid): predicted step s
(0-based) describes the interval (t0 + s*dt, t0 + (s+1)*dt] where t0 is the
snapshot host time `row["t"]`; `d_*` are deltas across that interval and
`mask` / `cop` / `slip` / `event` are the state at its end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CPK_TRACE_FILE = "planner_cpk.npz"
CPK_TRACE_VERSION = 1

# npz keys that are per-replan arrays (stacked on axis 0)
_STACKED = ("t_host", "latency_s", "trace_index", "d_fz", "d_disp", "mask",
            "cop", "slip", "event", "wrench", "wrist")


def _np(x) -> np.ndarray:
    """torch tensor / array -> float32 numpy on CPU (no torch import needed)."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32)


def denormalize_package(cpk, norm, hw) -> dict[str, np.ndarray]:
    """One ContactPackage (batch row 0) -> physical-unit numpy arrays.

    Inverse of what WindowSampler.sample applies to the training target:
    `cpk_d_fz` / `cpk_d_disp` / `wrench` / `wrist_ft` keys of the norm
    stats; mask, cop, slip, event are stored unnormalized. Layouts follow
    the recorder (fields last-axis channels):
        d_fz   (Tc, F, cph, cpw)      field dist-force-z units (see meta)
        d_disp (Tc, F, cph, cpw, 3)
        mask   (Tc, F, cph, cpw)      [0, 1]
        cop    (Tc, F, 2)             [-1, 1], NaN = undefined
        slip   (Tc, F)
        event  (Tc, E)                probabilities
        wrench (Tc, F, 6)             N / Nm (baseline-subtracted, as trained)
        wrist  (Tc, 6)
    `norm=None` stores the model-space values unchanged (flagged in meta).
    """
    d_fz = _np(cpk.d_fz)[0]                       # (Tc, F, cph, cpw)
    d_disp = _np(cpk.d_disp)[0]                   # (Tc, F, 3, cph, cpw)
    d_disp = np.ascontiguousarray(np.moveaxis(d_disp, 2, -1))   # (Tc,F,cph,cpw,3)
    wrench = _np(cpk.wrench)[0]                   # (Tc, F, 6)
    wrist = _np(cpk.wrist)[0]                     # (Tc, 6)
    if norm is not None:
        d_fz = np.asarray(norm.denormalize("cpk_d_fz", d_fz[..., None]),
                          dtype=np.float32)[..., 0]
        d_disp = np.asarray(norm.denormalize("cpk_d_disp", d_disp), dtype=np.float32)
        wrench = np.asarray(norm.denormalize("wrench", wrench), dtype=np.float32)
        wrist = np.asarray(norm.denormalize("wrist_ft", wrist), dtype=np.float32)
    event = cpk.event_probs() if hasattr(cpk, "event_probs") else cpk.event
    return {
        "d_fz": np.ascontiguousarray(d_fz, dtype=np.float32),
        "d_disp": np.ascontiguousarray(d_disp, dtype=np.float32),
        "mask": np.clip(_np(cpk.mask)[0], 0.0, 1.0),
        "cop": _np(cpk.cop)[0],
        "slip": _np(cpk.slip)[0],
        "event": _np(event)[0],
        "wrench": np.ascontiguousarray(wrench, dtype=np.float32),
        "wrist": np.ascontiguousarray(wrist, dtype=np.float32),
    }


def summarize_package(pkg: dict[str, np.ndarray]) -> dict:
    """JSON-safe per-step digest of a denormalized package for the trace row:
    what the model expected each pad to feel, at a glance, without the maps."""
    d_fz, mask, cop, slip, event = (pkg["d_fz"], pkg["mask"], pkg["cop"],
                                    pkg["slip"], pkg["event"])

    def r(x, nd=4):
        return [[round(float(v), nd) for v in row] for row in np.asarray(x)]

    return {
        "mask_frac": r(mask.mean(axis=(-2, -1))),                 # (Tc, F)
        "dfz_mean": r(d_fz.mean(axis=(-2, -1))),                  # (Tc, F)
        "dfz_absmax": r(np.abs(d_fz).max(axis=(-2, -1))),         # (Tc, F)
        "slip": r(slip),                                          # (Tc, F)
        # NaN -> None: json.dumps would otherwise emit a bare NaN token
        "cop": [[[None if not np.isfinite(v) else round(float(v), 4) for v in c]
                 for c in step] for step in np.asarray(cop)],     # (Tc, F, 2)
        "event": [int(k) for k in np.argmax(event, axis=-1)],     # (Tc,)
        "p_event": r(event),                                      # (Tc, E)
    }


@dataclass
class CpkRecord:
    index: int          # row of planner_cpk.npz == index of the trace row
    summary: dict


@dataclass
class CpkTraceLog:
    """Accumulates per-replan predicted packages; `save()` writes the npz."""
    hw: object = None
    denormalized: bool = True
    _rows: list[dict] = field(default_factory=list)
    _meta: dict = field(default_factory=dict)
    _warned: bool = False

    def __len__(self) -> int:
        return len(self._rows)

    def append(self, *, t_host: float, latency_s: float, trace_index: int,
               cpk, norm, latent_dt: float) -> CpkRecord:
        if norm is None and not self._warned:
            log.warning("cpk trace: policy exposes no norm stats — storing the "
                        "package in MODEL space (planner_cpk.npz meta.denormalized=0)")
            self._warned = True
        self.denormalized = self.denormalized and norm is not None
        pkg = denormalize_package(cpk, norm, self.hw)
        pkg.update({"t_host": np.float64(t_host), "latency_s": np.float32(latency_s),
                    "trace_index": np.int64(trace_index)})
        self._rows.append(pkg)
        if not self._meta:
            self._meta = {"latent_dt": float(latent_dt)}
        return CpkRecord(index=len(self._rows) - 1, summary=summarize_package(pkg))

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.stack([r[k] for r in self._rows]) for k in _STACKED}
        hw = self.hw
        sensors = ([s.name for s in hw.tactile.sensors] if hw is not None else [])
        out.update({
            "version": np.int64(CPK_TRACE_VERSION),
            "latent_dt": np.float64(self._meta.get("latent_dt", np.nan)),
            "denormalized": np.int64(bool(self.denormalized)),
            "sensors": np.asarray(sensors, dtype="U64"),
            "dist_force_unit_to_N": np.float64(
                getattr(getattr(hw, "tactile", None), "dist_force_unit_to_N", 0.0) or 0.0),
        })
        return out

    def save(self, path: Path) -> Path:
        path = Path(path)
        np.savez_compressed(path, **self.arrays())
        return path


def load_cpk_trace(ep_path: Path) -> dict | None:
    """`planner_cpk.npz` of an episode as a dict of arrays, or None."""
    f = Path(ep_path) / CPK_TRACE_FILE
    if not f.exists():
        return None
    with np.load(f, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    return out
