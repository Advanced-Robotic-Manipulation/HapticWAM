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


def cell_stats(rows: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Success/damage means + a 95% bootstrap CI on success, resampling SEEDS
    (the unit of independent repetition in the protocol — trials within a seed
    share a policy checkpoint) and trials within each seed."""
    succ = np.array([r["success"] == "True" for r in rows], dtype=float)
    dmg = np.array([r["damage"] == "True" for r in rows], dtype=float)
    out = {"n": len(rows), "success": float(succ.mean()) if len(rows) else np.nan,
           "damage": float(dmg.mean()) if len(rows) else np.nan,
           "success_ci_lo": np.nan, "success_ci_hi": np.nan}
    if not len(rows):
        return out
    by_seed: dict[str, np.ndarray] = {}
    for r, s in zip(rows, succ):
        by_seed.setdefault(r.get("seed", "0"), []).append(s)
    groups = [np.asarray(v) for v in by_seed.values()]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(groups), size=len(groups))
        vals = [groups[i][rng.integers(0, len(groups[i]), size=len(groups[i]))]
                for i in picked]
        boots[b] = np.concatenate(vals).mean()
    out["success_ci_lo"] = float(np.quantile(boots, 0.025))
    out["success_ci_hi"] = float(np.quantile(boots, 0.975))
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
    if compute_episode_metrics:
        from phantom.data.episode_store import EpisodeReader
        for r in rows:
            ep_path = Path(r["episode_path"])
            if not ep_path.exists():
                continue
            try:
                m = M.trial_metrics(ep_path, hw, tau_obj)
                ep_metrics[r["system"]].append(m)
                ep = EpisodeReader(ep_path)
                lead_times[r["system"]] += M.acc_lead_times(ep_path, ep, hw)
            except Exception:
                log.exception("metrics failed for %s", ep_path)

    return {"cells": table, "headline": headline,
            "lead_times": dict(lead_times),
            "episode_metrics": {k: _mean_dicts(v) for k, v in ep_metrics.items()}}


def _mean_dicts(dicts: list[dict]) -> dict:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: float(np.nanmean([d[k] for d in dicts if k in d])) for k in keys}


def write_report(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # csv
    with open(out_dir / "cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "system", "occlusion", "n", "success",
                    "success_ci_lo", "success_ci_hi", "damage"])
        for (task, system, occl), st in sorted(result["cells"].items()):
            w.writerow([task, system, occl, st["n"], st["success"],
                        st.get("success_ci_lo", ""), st.get("success_ci_hi", ""),
                        st["damage"]])
    # markdown
    lines = ["# PHANTOM evaluation report", "", "## Headline (per task)",
             "", "| task | occlusion | recovery ratio | retention |", "|---|---|---|---|"]
    for (task, occl), h in sorted(result["headline"].items()):
        lines.append(f"| {task} | {occl} | {h['recovery_ratio']:.3f} "
                     f"| {h['retention']:.3f} |")
    lines += ["", "## Success / damage per cell (95% bootstrap CI over seeds)", "",
              "| task | system | occlusion | n | success | 95% CI | damage |",
              "|---|---|---|---|---|---|---|"]
    for (task, system, occl), st in sorted(result["cells"].items()):
        ci = (f"[{st['success_ci_lo']:.2f}, {st['success_ci_hi']:.2f}]"
              if not np.isnan(st.get("success_ci_lo", np.nan)) else "—")
        lines.append(f"| {task} | {system} | {occl} | {st['n']} "
                     f"| {st['success']:.3f} | {ci} | {st['damage']:.3f} |")
    if result["episode_metrics"]:
        lines += ["", "## Episode metrics (means per system)", ""]
        for system, m in sorted(result["episode_metrics"].items()):
            lines.append(f"- **{system}**: "
                         + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "lead_times.json").write_text(json.dumps(result["lead_times"]),
                                             encoding="utf-8")
    return out_dir / "report.md"
