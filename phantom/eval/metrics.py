"""Per-trial metrics (pipeline.md §8): pure functions over a recorded episode
(EpisodeReader + planner_trace.json) and its ledger row."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.config.model import EVENT_IDX
from phantom.data import derived as dv
from phantom.data.episode_store import EpisodeReader
from phantom.data.schema import tactile_stream


def peak_normal_force(ep: EpisodeReader, hw: HardwareConfig) -> float:
    """Peak |fz| in N (calibrated) or peak indentation (fallback) across
    fingers, keyframes AND downsampled fields."""
    peak = 0.0
    for s in hw.tactile.sensors:
        for kind in ("keyframes", "fields_ds"):
            stream = tactile_stream(s.name, kind)
            if ep.has(stream):
                fields = np.asarray(ep._g(stream)["data"][:], dtype=np.float32)
                peak = max(peak, dv.peak_normal_force(fields, hw))
    return peak


def threshold_violations(ep: EpisodeReader, hw: HardwareConfig,
                         tau_obj: float) -> int:
    """Number of field-rate frames whose peak normal force exceeds tau_obj."""
    n = 0
    for s in hw.tactile.sensors:
        stream = tactile_stream(s.name, "fields_ds")
        if not ep.has(stream):
            continue
        fields = np.asarray(ep._g(stream)["data"][:], dtype=np.float32)
        ch = dv.channel_slices(hw.tactile)
        if hw.tactile.force_calibrated:
            per_frame = np.abs(fields[..., ch["dist_force"]][..., 2]).max(axis=(1, 2)) \
                * hw.tactile.dist_force_unit_to_N
        else:
            per_frame = np.abs(fields[..., ch["depth"]]).max(axis=(1, 2, 3))
        n += int((per_frame > tau_obj).sum())
    return n


def slip_events_and_recovery(ep: EpisodeReader, hw: HardwareConfig,
                             recovery_window_s: float = 1.0) -> tuple[int, float]:
    """(n slip events, recovery rate): a slip event recovers if contact is
    regained (hold) within the window after the slip run ends."""
    n_events, n_recovered = 0, 0
    for s in hw.tactile.sensors:
        stream = tactile_stream(s.name, "events")
        if not ep.has(stream):
            continue
        ev = np.asarray(ep._g(stream)["data"][:])
        ts = ep.ts(stream)
        slip = ev == EVENT_IDX["slip"]
        starts = np.flatnonzero(slip & ~np.concatenate([[False], slip[:-1]]))
        for i0 in starts:
            n_events += 1
            i1 = i0
            while i1 < len(ev) and slip[i1]:
                i1 += 1
            t_end = ts[min(i1, len(ts) - 1)]
            after = (ts > t_end) & (ts <= t_end + recovery_window_s)
            if (ev[after] == EVENT_IDX["hold"]).any():
                n_recovered += 1
    return n_events, (n_recovered / n_events if n_events else float("nan"))


def acc_lead_times(ep_path: Path, ep: EpisodeReader, hw: HardwareConfig,
                   gate_threshold: float = 0.5) -> list[float]:
    """ACC lead time (s) per contact onset: (derived onset t) - (last gate
    crossing before it). Positive = the gate fired BEFORE the physical onset —
    the §8 money-plot data."""
    trace_file = ep_path / "planner_trace.json"
    if not trace_file.exists():
        return []
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    gate_t = np.array([r["t"] for r in trace])
    gate_v = np.array([r["gate"] for r in trace])
    if len(gate_v) > 1:
        rising = (gate_v[1:] >= gate_threshold) & (gate_v[:-1] < gate_threshold)
        crossings = gate_t[1:][rising]
    else:
        crossings = np.array([])
    if not len(crossings):
        return []

    leads: list[float] = []
    for s in hw.tactile.sensors:
        stream = tactile_stream(s.name, "events")
        if not ep.has(stream):
            continue
        ev = np.asarray(ep._g(stream)["data"][:])
        ts = ep.ts(stream)
        onsets = ts[ev == EVENT_IDX["onset"]]
        for t_on in onsets:
            before = crossings[crossings <= t_on]
            if len(before):
                leads.append(float(t_on - before[-1]))
    return leads


def event_f1(ep: EpisodeReader, hw: HardwareConfig, ep_path: Path) -> float:
    """F1 of the planner's predicted next-step event vs derived labels at the
    replan times (macro over non-none classes)."""
    trace_file = ep_path / "planner_trace.json"
    if not trace_file.exists():
        return float("nan")
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    if not trace:
        return float("nan")
    s0 = hw.tactile.sensors[0].name
    stream = tactile_stream(s0, "events")
    if not ep.has(stream):
        return float("nan")
    ev = np.asarray(ep._g(stream)["data"][:])
    ts = ep.ts(stream)
    y_true, y_pred = [], []
    for r in trace:
        i = int(np.clip(np.searchsorted(ts, r["t"]), 0, len(ev) - 1))
        y_true.append(int(ev[i]))
        y_pred.append(int(np.argmax(r["p_evt"])))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in range(1, len(EVENT_IDX)):     # skip 'none'
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def latency_stats(ep_path: Path) -> dict:
    trace_file = ep_path / "planner_trace.json"
    if not trace_file.exists():
        return {}
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    if not trace:
        return {}
    lat = np.array([r["latency_s"] for r in trace])
    return {"replan_latency_mean_s": float(lat.mean()),
            "replan_latency_p95_s": float(np.percentile(lat, 95)),
            "n_replans": len(trace)}


def trial_metrics(ep_path: Path, hw: HardwareConfig, tau_obj: float) -> dict:
    ep = EpisodeReader(ep_path)
    n_slip, recovery = slip_events_and_recovery(ep, hw)
    leads = acc_lead_times(ep_path, ep, hw)
    out = {
        "peak_normal_force": peak_normal_force(ep, hw),
        "threshold_violations": threshold_violations(ep, hw, tau_obj),
        "slip_events": n_slip,
        "slip_recovery_rate": recovery,
        "acc_lead_time_mean_s": float(np.mean(leads)) if leads else float("nan"),
        "event_f1": event_f1(ep, hw, ep_path),
    }
    out.update(latency_stats(ep_path))
    return out
