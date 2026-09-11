"""Tactile prediction error across episodes: per-episode rows, per-system
pooled summaries with bootstrap CIs, horizon curves, and paired comparisons.

    python tools/tactile_prediction_eval.py data/episodes/deploy/20260912 \\
        --hardware configs/hardware.nuc.yaml --out-dir out/tpe
    python tools/tactile_prediction_eval.py data/episodes/deploy/2026091* \\
        --ledger runs/eval/ledger.csv --pair teacher vision_only --out-dir out/tpe
    python tools/tactile_prediction_eval.py <ep_dir> ... --group-by tag:ckpt: --png

Arguments are episode dirs (with meta.json) or roots searched for `ep_*`.
Only episodes carrying `planner_cpk.npz` (recorded with the predicted
contact package) are scored; the rest are listed as skipped.

Grouping (`--group-by`): `ledger` (default when --ledger is given: the
ledger's `system` column, matched on episode_path), `policy` (meta.policy),
`tag:<prefix>` (the first episode tag starting with <prefix>, e.g.
`tag:ckpt:`), or `none`.

`--pair A B` compares two groups on matched episodes: by (task, seed, trial)
from the ledger, else by (task, seed:<n> tag). Reports the mean difference
of per-episode d_fz skill / mask IoU / event accuracy with a bootstrap CI
and an exact sign test.

Outputs in --out-dir: episodes.csv, summary.json, summary.md,
horizon.json, pairs.json (with --pair), horizon.png (with --png, needs
matplotlib).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from phantom.config.hardware import load_hardware
from phantom.data.episode_store import list_episodes
from phantom.eval import tactile_prediction as TP

HEADLINE = ("tpe_dfz_skill", "tpe_dfz_skill_contact", "tpe_mask_iou", "tpe_event_acc",
            "tpe_dfz_rmse", "tpe_dfz_rmse_persist", "tpe_cop_err", "tpe_slip_err")


def collect(paths: list[str], include_unfinalized: bool) -> list[Path]:
    eps: list[Path] = []
    for p in paths:
        p = Path(p)
        if (p / "meta.json").exists():
            eps.append(p)
        else:
            eps += list_episodes(p, include_unfinalized=include_unfinalized)
    seen, out = set(), []
    for e in eps:
        r = e.resolve()
        if r not in seen:
            seen.add(r)
            out.append(e)
    return out


def _meta(ep: Path) -> dict:
    try:
        return json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ledger_index(path: Path | None) -> dict[Path, dict]:
    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {Path(r["episode_path"]).resolve(): r for r in rows if r.get("episode_path")}


def group_key(ep: Path, meta: dict, ledger: dict[Path, dict], mode: str) -> str:
    if mode == "ledger":
        row = ledger.get(ep.resolve())
        return row["system"] if row else "unlabelled"
    if mode == "policy":
        return str(meta.get("policy") or "unknown")
    if mode.startswith("tag:"):
        prefix = mode[4:]
        for t in meta.get("tags") or []:
            if str(t).startswith(prefix):
                return str(t)
        return "untagged"
    return "all"


def pair_key(ep: Path, meta: dict, ledger: dict[Path, dict]) -> tuple | None:
    row = ledger.get(ep.resolve())
    if row:
        return (row.get("task"), row.get("seed"), row.get("trial"))
    seed = next((t[5:] for t in (meta.get("tags") or []) if str(t).startswith("seed:")), None)
    if seed is None:
        return None
    return (meta.get("task"), seed)


def _fmt(v) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--hardware", default=None, help="hardware yaml (default: repo default)")
    ap.add_argument("--ledger", type=Path, default=None, help="run_eval ledger.csv")
    ap.add_argument("--group-by", default=None,
                    help="ledger | policy | tag:<prefix> | none (default: ledger if "
                         "--ledger else policy)")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("out/tactile_prediction"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-unfinalized", action="store_true")
    ap.add_argument("--png", action="store_true", help="also draw horizon.png (matplotlib)")
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware, quiet=True)
    ledger = _ledger_index(args.ledger)
    mode = args.group_by or ("ledger" if args.ledger else "policy")
    eps = collect(args.paths, args.include_unfinalized)

    per_ep: list[dict] = []
    results: dict[str, list[TP.EpisodeTPE]] = defaultdict(list)
    keyed: dict[str, dict[tuple, TP.EpisodeTPE]] = defaultdict(dict)
    skipped: list[tuple[str, str]] = []
    for ep in eps:
        meta = _meta(ep)
        g = group_key(ep, meta, ledger, mode)
        try:
            r = TP.episode_tactile_prediction(ep, hw)
        except Exception as e:                                   # keep going; report
            skipped.append((ep.name, f"{type(e).__name__}: {e}"))
            continue
        if r is None:
            skipped.append((ep.name, "no planner_cpk.npz"))
            continue
        if r.skipped_reason:
            skipped.append((ep.name, r.skipped_reason))
            continue
        results[g].append(r)
        s = r.summary()
        per_ep.append({"episode": str(ep), "group": g, "task": meta.get("task", ""),
                       "success": meta.get("success"), **s})
        pk = pair_key(ep, meta, ledger)
        if pk is not None:
            keyed[g][pk] = r

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # per-episode csv
    if per_ep:
        keys = ["episode", "group", "task", "success"] + sorted(
            {k for row in per_ep for k in row if k.startswith("tpe_")})
        with open(args.out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(per_ep)

    # per-group pooled summary + CI over episode means
    summary: dict[str, dict] = {}
    horizon: dict[str, dict] = {}
    for g, res in results.items():
        pooled = TP.pool_steps(res)
        Tc = res[0].Tc
        pooled_summary = TP.summarize_steps(
            pooled, Tc=Tc, fz_unit_to_N=res[0].dist_force_unit_to_N,
            n_total=sum(r.n_replans for r in res), n_scored=sum(r.n_scored for r in res))
        ep_summaries = [r.summary() for r in res]
        ci = {}
        for k in HEADLINE:
            vals = [s.get(k, np.nan) for s in ep_summaries]
            lo, hi = TP.bootstrap_ci(vals, n_boot=args.boot, seed=args.seed)
            ci[k] = {"mean_of_episodes": float(np.nanmean(vals)) if np.isfinite(vals).any()
                     else float("nan"), "ci_lo": lo, "ci_hi": hi}
        summary[g] = {"n_episodes": len(res), "pooled": pooled_summary, "episode_ci": ci,
                      "latent_dt": res[0].latent_dt, "Tc": Tc}
        horizon[g] = TP.horizon_curves(res)

    pairs = None
    if args.pair:
        a, b = args.pair
        common = sorted(set(keyed.get(a, {})) & set(keyed.get(b, {})), key=str)
        pairs = {"a": a, "b": b, "n_pairs": len(common), "metrics": {}}
        for k, higher in (("tpe_dfz_skill", True), ("tpe_dfz_skill_contact", True),
                          ("tpe_mask_iou", True), ("tpe_event_acc", True),
                          ("tpe_dfz_rmse", False), ("tpe_cop_err", False)):
            va = [keyed[a][c].summary().get(k, np.nan) for c in common]
            vb = [keyed[b][c].summary().get(k, np.nan) for c in common]
            pairs["metrics"][k] = TP.paired_compare(va, vb, higher_is_better=higher,
                                                    n_boot=args.boot, seed=args.seed)

    def dflt(o):
        if isinstance(o, (np.floating, np.integer)):
            v = o.item()
            return None if isinstance(v, float) and not math.isfinite(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        raise TypeError(type(o).__name__)

    (args.out_dir / "summary.json").write_text(
        json.dumps({"groups": summary, "skipped": skipped}, indent=1, default=dflt),
        encoding="utf-8")
    (args.out_dir / "horizon.json").write_text(json.dumps(horizon, indent=1, default=dflt),
                                               encoding="utf-8")
    if pairs is not None:
        (args.out_dir / "pairs.json").write_text(json.dumps(pairs, indent=1, default=dflt),
                                                 encoding="utf-8")

    # markdown
    lines = ["# Tactile prediction error", "",
             f"{len(per_ep)} scored episodes, {len(skipped)} skipped; grouping = `{mode}`.",
             "Skill = 1 - err_model / err_persistence (0 = no better than 'the pads stay as "
             "they are', 1 = perfect); pooled over scored replans, CI = percentile bootstrap "
             "over per-episode means.", "",
             "| group | eps | replans | d_fz RMSE model / persist | d_fz skill [CI] | "
             "d_fz skill in contact | mask IoU model / persist | event acc model / persist | "
             "CoP err | slip err |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g, s in sorted(summary.items()):
        p, c = s["pooled"], s["episode_ci"]
        sk = c["tpe_dfz_skill"]
        lines.append(
            f"| {g} | {s['n_episodes']} | {int(p['tpe_n_replans_scored'])}/"
            f"{int(p['tpe_n_replans_total'])} | {_fmt(p.get('tpe_dfz_rmse'))} / "
            f"{_fmt(p.get('tpe_dfz_rmse_persist'))} | {_fmt(p.get('tpe_dfz_skill'))} "
            f"[{_fmt(sk['ci_lo'])}, {_fmt(sk['ci_hi'])}] | {_fmt(p.get('tpe_dfz_skill_contact'))} "
            f"| {_fmt(p.get('tpe_mask_iou'))} / {_fmt(p.get('tpe_mask_iou_persist'))} "
            f"| {_fmt(p.get('tpe_event_acc'))} / {_fmt(p.get('tpe_event_acc_persist'))} "
            f"| {_fmt(p.get('tpe_cop_err'))} | {_fmt(p.get('tpe_slip_err'))} |")
    if horizon:
        lines += ["", "## Horizon (steps ahead: d_fz RMSE model / persistence)", ""]
        for g, h in sorted(horizon.items()):
            if not h:
                continue
            cells = " · ".join(f"s{k}: {m:.4f}/{pm:.4f}" for k, m, pm in
                               zip(h["steps_ahead"], h["dfz_rmse"], h["dfz_rmse_persist"]))
            lines.append(f"- **{g}** ({h['n_replans']} replans): {cells}")
    if pairs is not None:
        lines += ["", f"## Paired: {pairs['a']} − {pairs['b']} ({pairs['n_pairs']} matched episodes)",
                  "", "| metric | mean diff | 95% CI | sign-test p | A better / B better / ties |",
                  "|---|---|---|---|---|"]
        for k, m in pairs["metrics"].items():
            lines.append(f"| {k} | {_fmt(m['mean_diff'])} | [{_fmt(m['diff_ci_lo'])}, "
                         f"{_fmt(m['diff_ci_hi'])}] | {_fmt(m['sign_p'])} | "
                         f"{m['n_a_better']} / {m['n_b_better']} / {m['n_ties']} |")
    if skipped:
        lines += ["", "## Skipped", ""] + [f"- {n}: {why}" for n, why in skipped]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if args.png and horizon:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                                   # optional dependency
            print(f"[png skipped: {e}]", file=sys.stderr)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            for g, h in sorted(horizon.items()):
                if not h:
                    continue
                x = h["steps_ahead"]
                axes[0].plot(x, h["dfz_rmse"], marker="o", label=f"{g}")
                axes[0].plot(x, h["dfz_rmse_persist"], ls="--", color="gray", alpha=0.6)
                axes[1].plot(x, h["mask_iou"], marker="o", label=g)
                axes[1].plot(x, h["mask_iou_persist"], ls="--", color="gray", alpha=0.6)
                axes[2].plot(x, h["event_acc"], marker="o", label=g)
                axes[2].plot(x, h["event_acc_persist"], ls="--", color="gray", alpha=0.6)
            for ax, title in zip(axes, ("d_fz RMSE (dashed: persistence)", "contact-mask IoU",
                                        "event accuracy")):
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("latent steps ahead")
            axes[0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.out_dir / "horizon.png", dpi=150)
    return 0


if __name__ == "__main__":
    sys.exit(main())
