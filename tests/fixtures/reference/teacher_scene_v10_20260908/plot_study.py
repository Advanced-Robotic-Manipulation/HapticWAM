#!/usr/bin/env python3
"""Render the frozen study's complete outcome grid and stage counts from JSON."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SUSTAINED = "sustained_supported_placement_through_horizon"
CLEAN = "clean_placement_through_horizon"
STAGES = ("acquired", "lifted", "carried", "released_in_bin", "full_task", SUSTAINED, CLEAN)
LABELS = ("Acquired", "Lifted", "Carried", "Released in bin", "Initial placement", "Supported through 60 s", "Clean through 60 s")
CATEGORIES = (
    ("unresolved", "Unresolved", "#dedede"),
    ("no_acquisition", "No acquisition", "#efb0a5"),
    ("acquired", "Acquired only", "#f1ca90"),
    ("lifted", "Lifted", "#f4e3a0"),
    ("carried", "Carried", "#a8c8e7"),
    ("released_in_bin", "Released in bin", "#a8d8cf"),
    ("full_task", "Initial placement", "#9db7e6"),
    (SUSTAINED, "Supported through 60 s", "#83bdba"),
    (CLEAN, "Clean through 60 s", "#79bb88"),
)


def category(row):
    """Missing/invalid scores cannot inherit a successful stage color."""
    if row.get("status") != "scored":
        return "unresolved"
    outcomes = row["outcomes"]
    for stage in reversed(STAGES):
        if outcomes.get(stage) is True:
            return stage
    return "no_acquisition" if outcomes.get("acquired") is False else "unresolved"


def prepare(summary):
    variants = list(summary["variants"])
    trials = summary["trials"]
    if len(trials) != summary["planned_trials"]:
        raise ValueError("Every planned trial must appear in the summary")
    rows = list(dict.fromkeys((r["condition"], r["seed"]) for r in trials))
    # Keep each authentic start's two seeds together for paired inspection.
    starts = list(dict.fromkeys(condition for condition, _ in rows))
    rows.sort(key=lambda row: (starts.index(row[0]), row[1]))
    lookup = {(r["condition"], r["seed"], r["variant"]): r for r in trials}
    if len(lookup) != len(trials):
        raise ValueError("Duplicate trial identities")
    if set(lookup) != {(c, s, v) for c, s in rows for v in variants}:
        raise ValueError("Outcome grid is not the complete planned matched design")
    counts = {}
    for variant in variants:
        selected = [r for r in trials if r["variant"] == variant]
        counts[variant] = {}
        for stage in STAGES:
            values = [r["outcomes"].get(stage) if r.get("status") == "scored" else None for r in selected]
            if any(value is not True and value is not False and value is not None for value in values):
                raise ValueError("Outcomes must be true, false or unresolved")
            count = {"true": sum(v is True for v in values), "false": sum(v is False for v in values), "unresolved": sum(v is None for v in values)}
            expected = summary["variants"][variant]["outcomes"][stage]
            if count != {key: expected[key] for key in count} or len(values) != expected["planned"]:
                raise ValueError(f"Aggregate disagrees with planned rows: {variant}/{stage}")
            counts[variant][stage] = count
    return variants, rows, lookup, counts


def render(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    variants, rows, lookup, counts = prepare(summary)
    colors = {key: color for key, _, color in CATEGORIES}
    short = {key: label for key, label, _ in CATEGORIES}
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, max(9, len(rows) * .77)))
    for y, (condition, seed) in enumerate(rows):
        for x, variant in enumerate(variants):
            row = lookup[condition, seed, variant]
            level = category(row)
            ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor=colors[level], edgecolor="white", linewidth=2))
            text = short[level]
            if level == "unresolved":
                text += "\n" + row["status"].replace("_", " ")
            elif row.get("stop_reason"):
                guards = (row.get("first_stop_diagnostic") or {}).get("safety_events")
                text += "\nSTOP: " + ("\n".join(guards) if guards else row["stop_reason"])
            if row.get("status") == "scored" and row["outcomes"].get("dropped") is True:
                text += "\nDROP"
            if level == "full_task" and row["outcomes"].get(SUSTAINED) is None:
                text += "\n60 s unresolved"
            ax.text(x, y, text, ha="center", va="center", fontsize=8, wrap=True)
    ax.set(xlim=(-.5, len(variants)-.5), ylim=(len(rows)-.5, -.5),
           xticks=range(len(variants)), yticks=range(len(rows)))
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=10)
    ax.set_yticklabels([f"{c}\nseed {s}" for c, s in rows], fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    resolved = sum(r["status"] == "scored" for r in summary["trials"])
    fig.suptitle(f"Teacher fixed-object study: {resolved}/{summary['planned_trials']} valid trials", fontsize=16, y=.99)
    fig.text(.5, .016, "Color = furthest confirmed stage. STOP and DROP are retained independently.\n"
             "Initial placement is a latched score; supported/clean placement requires the full 60 s observation.\n"
             "Six selected arm starts × two seeds; tactile/contact transfer remains uncalibrated.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .95))
    for extension in ("png", "svg"):
        fig.savefig(destination / f"outcome_matrix.{extension}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(variants), figsize=(16, 5), sharey=True, squeeze=False)
    for ax, variant in zip(axes[0], variants):
        for y, stage in enumerate(STAGES):
            left = 0
            count = counts[variant][stage]
            for key, color in (("true", "#378653"), ("false", "#d18c7d"), ("unresolved", "#dedede")):
                width = count[key]
                ax.barh(y, width, left=left, color=color, height=.67)
                if width:
                    ax.text(left+width/2, y, str(width), ha="center", va="center", fontsize=9)
                left += width
        ax.set_title(variant.replace("_", "\n"), fontsize=10)
        ax.set_xlim(0, len(rows))
        ax.set_xticks([0, len(rows)//2, len(rows)])
        ax.set_xlabel("Planned trials")
        ax.set_yticks(range(len(STAGES)), LABELS)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Stage counts retain unresolved cases in the planned denominator", fontsize=14)
    fig.legend(handles=[Patch(color=c, label=k) for k, c in (("Achieved", "#378653"), ("Not achieved", "#d18c7d"), ("Unresolved", "#dedede"))], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .075, 1, .9))
    for extension in ("png", "svg"):
        fig.savefig(destination / f"stage_counts.{extension}", dpi=160)
    plt.close(fig)
    return {"planned_trials": len(summary["trials"]), "valid_trials": resolved,
            "rows": [{"condition": c, "seed": s} for c, s in rows], "variants": variants,
            "category_counts": dict(Counter(category(r) for r in summary["trials"])), "stage_counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    manifest = render(json.loads(raw), args.out)
    manifest.update(summary_sha256=hashlib.sha256(raw).hexdigest(), plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"planned": manifest["planned_trials"], "valid": manifest["valid_trials"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
