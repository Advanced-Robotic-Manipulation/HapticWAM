"""Aggregation of the eval ledger into the §8 tables:

    recovery ratio = (student - vision_only) / (teacher - vision_only)
    retention      = student / teacher

per (task, occlusion) with per-seed bootstrap CIs, plus per-cell success /
damage / metric means. Emits CSV + a markdown table, and the ACC lead-time
distribution data (the money-plot input).
"""

from __future__ import annotations

import csv
import json
import math
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.eval import metrics as M

log = logging.getLogger(__name__)


def load_ledger(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — non-empty at 0/n
    and n/n, unlike a percentile bootstrap (issue #10: 0/16 gave [0, 0])."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cell_stats(rows: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Success/damage means with TWO intervals on success:
    - `success_ci_lo/hi`: Wilson 95% on the pooled trials (always defined
      for n>0, honest at all-zero / all-one);
    - `success_boot_ci_lo/hi`: 95% percentile bootstrap resampling SEEDS
      (the unit of independent repetition — trials within a seed share a
      policy checkpoint) then trials within each seed; NaN with fewer than
      two seed groups, where a cluster bootstrap is meaningless."""
    succ = np.array([r["success"] == "True" for r in rows], dtype=float)
    dmg = np.array([r["damage"] == "True" for r in rows], dtype=float)
    out = {"n": len(rows), "success": float(succ.mean()) if len(rows) else np.nan,
           "damage": float(dmg.mean()) if len(rows) else np.nan,
           "success_ci_lo": np.nan, "success_ci_hi": np.nan,
           "success_boot_ci_lo": np.nan, "success_boot_ci_hi": np.nan,
           "n_seeds": 0, "ci_method": "wilson"}
    if not len(rows):
        return out
    out["success_ci_lo"], out["success_ci_hi"] = wilson_interval(int(succ.sum()), len(rows))
    by_seed: dict[str, list] = {}
    for r, s in zip(rows, succ):
        by_seed.setdefault(r.get("seed", "0"), []).append(s)
    groups = [np.asarray(v) for v in by_seed.values()]
    out["n_seeds"] = len(groups)
    if len(groups) < 2:
        return out
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(groups), size=len(groups))
        vals = [groups[i][rng.integers(0, len(groups[i]), size=len(groups[i]))]
                for i in picked]
        boots[b] = np.concatenate(vals).mean()
    out["success_boot_ci_lo"] = float(np.quantile(boots, 0.025))
    out["success_boot_ci_hi"] = float(np.quantile(boots, 0.975))
    return out


def aggregate(ledger_path: Path, hw: HardwareConfig, *, tau_obj: float = 1.0,
              compute_episode_metrics: bool = False) -> dict:
    rows = load_ledger(ledger_path)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["task"], r["system"], r["occlusion"])].append(r)

    table = {key: cell_stats(v) for key, v in cells.items()}

    # headline ratios per (task, occlusion)
    headline = {}
    keys = {(t, o) for (t, _s, o) in cells}
    for (task, occl) in keys:
        s = table.get((task, "student", occl), {}).get("success", np.nan)
        te = table.get((task, "teacher", occl), {}).get("success", np.nan)
        vo = table.get((task, "vision_only", occl), {}).get("success", np.nan)
        denom = te - vo
        headline[(task, occl)] = {
            "recovery_ratio": (s - vo) / denom if denom and not np.isnan(denom)
            and abs(denom) > 1e-9 else np.nan,
            "retention": s / te if te and not np.isnan(te) and te > 0 else np.nan,
        }

    lead_times: dict[str, list[float]] = defaultdict(list)
    ep_metrics: dict[str, list[dict]] = defaultdict(list)
    tpe_results: dict[str, list] = defaultdict(list)
    if compute_episode_metrics:
        from phantom.data.episode_store import EpisodeReader
        from phantom.eval import tactile_prediction as TP
        for r in rows:
            ep_path = Path(r["episode_path"])
            if not ep_path.exists():
                continue
            try:
                m = M.trial_metrics(ep_path, hw, tau_obj)
                ep_metrics[r["system"]].append(m)
                ep = EpisodeReader(ep_path)
                lead_times[r["system"]] += M.acc_lead_times(ep_path, ep, hw)
                tpe = TP.episode_tactile_prediction(ep_path, hw, ep=ep)
                if tpe is not None:
                    tpe_results[r["system"]].append(tpe)
            except Exception:
                log.exception("metrics failed for %s", ep_path)

    # tactile prediction error per system: POOLED over every scored replan
    # (not a mean of episode means) + the horizon curves for the figure
    tactile_prediction = {}
    if tpe_results:
        from phantom.eval import tactile_prediction as TP
        for system, res in tpe_results.items():
            pooled = TP.pool_steps(res)
            Tc = res[0].Tc
            tactile_prediction[system] = {
                "n_episodes": len(res),
                "summary": TP.summarize_steps(
                    pooled, Tc=Tc, fz_unit_to_N=res[0].dist_force_unit_to_N,
                    n_total=sum(r.n_replans for r in res),
                    n_scored=sum(r.n_scored for r in res)),
                "horizon": TP.horizon_curves(res),
                "skill_ci": TP.bootstrap_ci([r.summary().get("tpe_dfz_skill", np.nan)
                                             for r in res]),
            }

    return {"cells": table, "headline": headline,
            "lead_times": dict(lead_times),
            "episode_metrics": {k: _mean_dicts(v) for k, v in ep_metrics.items()},
            "tactile_prediction": tactile_prediction}


def _mean_dicts(dicts: list[dict]) -> dict:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    import warnings
    with warnings.catch_warnings():
        # a metric undefined on every episode (no trace, no contact, ...) is
        # a NaN mean, not a warning
        warnings.simplefilter("ignore", RuntimeWarning)
        return {k: float(np.nanmean([d[k] for d in dicts if k in d])) for k in keys}


def write_report(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # csv
    with open(out_dir / "cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "system", "occlusion", "n", "success",
                    "success_ci_lo", "success_ci_hi", "ci_method",
                    "success_boot_ci_lo", "success_boot_ci_hi", "n_seeds", "damage"])
        for (task, system, occl), st in sorted(result["cells"].items()):
            w.writerow([task, system, occl, st["n"], st["success"],
                        st.get("success_ci_lo", ""), st.get("success_ci_hi", ""),
                        st.get("ci_method", ""), st.get("success_boot_ci_lo", ""),
                        st.get("success_boot_ci_hi", ""), st.get("n_seeds", ""),
                        st["damage"]])
    # markdown
    lines = ["# PHANTOM evaluation report", "", "## Headline (per task)",
             "", "| task | occlusion | recovery ratio | retention |", "|---|---|---|---|"]
    for (task, occl), h in sorted(result["headline"].items()):
        lines.append(f"| {task} | {occl} | {h['recovery_ratio']:.3f} "
                     f"| {h['retention']:.3f} |")
    lines += ["", "## Success / damage per cell (95% Wilson CI on pooled trials; "
              "seed-cluster bootstrap CI alongside, NaN under 2 seeds)", "",
              "| task | system | occlusion | n | success | 95% Wilson CI | seed-bootstrap CI | seeds | damage |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (task, system, occl), st in sorted(result["cells"].items()):
        ci = (f"[{st['success_ci_lo']:.2f}, {st['success_ci_hi']:.2f}]"
              if not np.isnan(st.get("success_ci_lo", np.nan)) else "—")
        bci = (f"[{st['success_boot_ci_lo']:.2f}, {st['success_boot_ci_hi']:.2f}]"
               if not np.isnan(st.get("success_boot_ci_lo", np.nan)) else "—")
        lines.append(f"| {task} | {system} | {occl} | {st['n']} "
                     f"| {st['success']:.3f} | {ci} | {bci} | {st.get('n_seeds', '')} "
                     f"| {st['damage']:.3f} |")
    if result["episode_metrics"]:
        lines += ["", "## Episode metrics (means per system)", ""]
        for system, m in sorted(result["episode_metrics"].items()):
            lines.append(f"- **{system}**: "
                         + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    tpe = result.get("tactile_prediction") or {}
    if tpe:
        lines += ["", "## Tactile prediction error (predicted contact package vs "
                  "measured pads; pooled over scored replans; skill = 1 - model/persistence)",
                  "",
                  "| system | episodes | replans scored | d_fz RMSE model / persist | d_fz skill "
                  "[95% CI over episodes] | mask IoU model / persist | event acc model / persist |",
                  "|---|---|---|---|---|---|---|"]
        for system, t in sorted(tpe.items()):
            s = t["summary"]
            lo, hi = t["skill_ci"]
            ci = f"[{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else "—"
            lines.append(
                f"| {system} | {t['n_episodes']} | {int(s.get('tpe_n_replans_scored', 0))}"
                f"/{int(s.get('tpe_n_replans_total', 0))} "
                f"| {s.get('tpe_dfz_rmse', np.nan):.4f} / {s.get('tpe_dfz_rmse_persist', np.nan):.4f} "
                f"| {s.get('tpe_dfz_skill', np.nan):.3f} {ci} "
                f"| {s.get('tpe_mask_iou', np.nan):.3f} / {s.get('tpe_mask_iou_persist', np.nan):.3f} "
                f"| {s.get('tpe_event_acc', np.nan):.3f} / {s.get('tpe_event_acc_persist', np.nan):.3f} |")
        (out_dir / "tactile_prediction.json").write_text(
            json.dumps(tpe, indent=1, default=_json_default), encoding="utf-8")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "lead_times.json").write_text(json.dumps(result["lead_times"]),
                                             encoding="utf-8")
    return out_dir / "report.md"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        v = o.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
