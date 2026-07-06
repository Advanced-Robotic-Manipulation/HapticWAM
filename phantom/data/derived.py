"""Derived contact channels (pipeline.md §1: "derive, don't trust").

Pure numpy, config-thresholded, RECOMPUTABLE offline — nothing here depends on
undocumented SDK outputs. All functions take the canonical 8-channel field
stack [disp_x, disp_y, depth, shear_x, shear_y, fx, fy, fz]
(drivers/base.py field_stack()), at ANY spatial resolution.

Conventions:
 - CoP is in normalized [-1, 1]^2, (x, y) = (columns, rows) — the same frame
   as ContactScenario.blob_center_uv and the mock renderer's grid, which makes
   the CoP tolerance resolution-independent (tests/test_derived_closure.py).
 - The slip flow statistic subtracts the NON-contact background flow (noise
   floor), making tau_slip largely frame-rate independent.

Acceptance spec: tests/test_derived_closure.py (mock <-> derived closure).
"""

from __future__ import annotations

import numpy as np

from phantom.config.model import EVENT_IDX

_EPS = 1e-9


# ---------------------------------------------------------------------------
# channel bookkeeping
# ---------------------------------------------------------------------------

def channel_slices(tactile_cfg) -> dict[str, slice]:
    """Slices into the canonical 8-ch stack, widths from the config
    (deformation2d=2, depth=1, shear=2, dist_force=3 — validated to sum to 8)."""
    fc = tactile_cfg.field_channels
    out: dict[str, slice] = {}
    o = 0
    for name, width in (("deformation2d", fc.deformation2d), ("depth", fc.depth),
                        ("shear", fc.shear), ("dist_force", fc.dist_force)):
        out[name] = slice(o, o + width)
        o += width
    return out


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    return np.meshgrid(xs, ys)  # xx (h,w), yy (h,w) — mock-renderer convention


def _contact_mask(stack: np.ndarray, hw) -> np.ndarray:
    """(h, w) bool per-cell contact, per hw.derived.contact_source."""
    ch = channel_slices(hw.tactile)
    d = hw.derived
    if d.contact_source == "fz" and hw.tactile.force_calibrated:
        fz_N = np.abs(stack[..., ch["dist_force"]][..., 2]) * hw.tactile.dist_force_unit_to_N
        return fz_N > d.tau_contact_fz
    return np.abs(stack[..., ch["depth"]][..., 0]) > d.tau_contact_depth


# ---------------------------------------------------------------------------
# per-timestep derivation
# ---------------------------------------------------------------------------

def derive_timestep(stack: np.ndarray, prev: np.ndarray | None, dt: float,
                    hw) -> dict:
    """One field frame -> {"mask_frac": float, "cop": (2,) [-1,1] (NaN when
    undefined), "slip": float}. `prev` is the previous frame at the same
    resolution (None on the first tick -> slip 0)."""
    ch = channel_slices(hw.tactile)
    d = hw.derived
    h, w = stack.shape[:2]
    contact = _contact_mask(stack, hw)
    n_contact = int(contact.sum())
    mask_frac = float(n_contact) / float(h * w)

    # --- CoP: weight-normalized centroid over contact cells -----------------
    cop = np.full(2, np.nan, dtype=np.float32)
    if n_contact >= d.cop_min_contact_cells:
        if d.contact_source == "fz" and hw.tactile.force_calibrated:
            weight = np.abs(stack[..., ch["dist_force"]][..., 2])
        else:
            weight = np.abs(stack[..., ch["depth"]][..., 0])
        weight = weight * contact
        wsum = float(weight.sum())
        if wsum > _EPS:
            xx, yy = _grid(h, w)
            cop = np.array([float((weight * xx).sum() / wsum),
                            float((weight * yy).sum() / wsum)], dtype=np.float32)

    # --- slip: background-subtracted tangential flow (+ friction cone) ------
    slip = 0.0
    if prev is not None and n_contact > 0:
        dt = max(float(dt), 1e-6)
        flow = np.abs(stack[..., ch["deformation2d"]]
                      - prev[..., ch["deformation2d"]]).mean(axis=-1) / dt
        bg = float(flow[~contact].mean()) if (~contact).any() else 0.0
        flow_stat = max(float(flow[contact].mean()) - bg, 0.0)
        if hw.tactile.force_calibrated:
            fxy = np.linalg.norm(stack[..., ch["dist_force"]][..., :2], axis=-1)
            fz = np.abs(stack[..., ch["dist_force"]][..., 2])
            cone = float((fxy[contact] / (fz[contact] + _EPS)).mean()) / d.friction_cone_mu
            slip = d.slip_flow_weight * flow_stat + (1.0 - d.slip_flow_weight) * cone
        else:
            slip = flow_stat
    return {"mask_frac": mask_frac, "cop": cop, "slip": float(slip)}


# ---------------------------------------------------------------------------
# event ontology (auto labels — zero manual tagging)
# ---------------------------------------------------------------------------

def _debounced_contact(contact: np.ndarray, min_ticks: int) -> np.ndarray:
    """Merge sub-min gaps, then drop sub-min runs."""
    c = contact.copy()
    for target, fill in ((False, True), (True, False)):
        # find runs of value `target` shorter than min_ticks and flip them
        i = 0
        T = len(c)
        while i < T:
            if c[i] == target:
                j = i
                while j < T and c[j] == target:
                    j += 1
                interior = i > 0 and j < T
                if (j - i) < min_ticks and (interior if target is False else True):
                    c[i:j] = fill
                i = j
            else:
                i += 1
    return c


def event_labels(mask_frac: np.ndarray, slip: np.ndarray, derived_cfg) -> np.ndarray:
    """(T,) int labels over {none, onset, hold, slip, release} (EVENT_IDX).

    onset = first tick of a (debounced) contact run; release = first tick
    after it ends; slip = in-contact tick with slip score > tau_slip;
    hold = other in-contact ticks; none = everything else."""
    mask_frac = np.asarray(mask_frac)
    slip = np.asarray(slip)
    T = len(mask_frac)
    contact = _debounced_contact(mask_frac > 0, max(1, int(derived_cfg.event_min_hold_ticks)))
    ev = np.full(T, EVENT_IDX["none"], dtype=np.int64)
    i = 0
    while i < T:
        if contact[i]:
            j = i
            while j < T and contact[j]:
                j += 1
            ev[i] = EVENT_IDX["onset"]
            for k in range(i + 1, j):
                ev[k] = EVENT_IDX["slip"] if slip[k] > derived_cfg.tau_slip \
                    else EVENT_IDX["hold"]
            if j < T:
                ev[j] = EVENT_IDX["release"]
            i = j + 1
        else:
            i += 1
    return ev


# ---------------------------------------------------------------------------
# small pure helpers used across train/eval/deploy
# ---------------------------------------------------------------------------

def contact_within(mask_frac: np.ndarray, i: int, n: int) -> bool:
    """Contact anywhere in the strictly-future window (i, i+n]. The ACC gate
    BCE label (pipeline.md §3)."""
    return bool((np.asarray(mask_frac)[i + 1:i + 1 + n] > 0).any())


def calibrate_tau_obj(peaks: np.ndarray, quantile: float) -> float:
    """'The dataset defines gentle enough' (pipeline.md §5, HID-S)."""
    return float(np.quantile(np.asarray(peaks, dtype=np.float64), quantile))


def force_safety_penalty(x: np.ndarray, tau: float, lam: float) -> np.ndarray:
    """r = -lam * max(0, x - tau), elementwise."""
    return -lam * np.maximum(0.0, np.asarray(x, dtype=np.float64) - tau)


def peak_normal_force(fields: np.ndarray, hw) -> float:
    """Peak |fz| in N over a (T?, h, w, 8) stack when the distributed force is
    calibrated; peak |depth| indentation (deformation proxy) otherwise."""
    ch = channel_slices(hw.tactile)
    if hw.tactile.force_calibrated:
        return float(np.abs(fields[..., ch["dist_force"]][..., 2]).max()
                     * hw.tactile.dist_force_unit_to_N)
    return float(np.abs(fields[..., ch["depth"]][..., 0]).max())


def reactive_score(cur: np.ndarray, prev: np.ndarray) -> float:
    """CASA-style reactive tactile-change score (mean abs frame difference
    over all fingers/cells/channels). Finite scalar."""
    return float(np.mean(np.abs(np.asarray(cur, dtype=np.float32)
                                - np.asarray(prev, dtype=np.float32))))
