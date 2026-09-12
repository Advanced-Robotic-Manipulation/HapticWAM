#!/usr/bin/env python3
"""Aggregate zoo trial results into the stage table per (model, recipe); pre-registered metrics only.

Primary: placed rate (stage 4) with Wilson 95% CI. Secondary: stage>=3 rate,
mean ordinal stage. Latency-confounded and scorer-invalid trials are counted
separately and excluded from the primary denominator; a sensitivity row keeps
them. Paired comparisons use the exact sign test on ordinal stage over shared
identities (start, seed).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

STAGES = ("no_grab", "grab", "pick", "in_box_gripped", "placed")


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"n": 0, "p_two_sided": None, "wins": pos, "losses": neg}
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return {"n": n, "wins": pos, "losses": neg, "p_two_sided": round(min(1.0, p), 4)}


def load(study, raw_root):
    rows = []
    for t in study["trials"]:
        folder = Path(raw_root) / "rollouts" / t["id"]
        status = json.loads((folder / "run_status.json").read_text()) if (folder / "run_status.json").exists() else {}
        result = json.loads((folder / "trial_result.json").read_text()) if (folder / "trial_result.json").exists() else None
        # post-hoc partial score of an integrity-stopped rollout (original trial_result.json is kept untouched)
        if (result or {}).get("status") == "missing_inputs" and (folder / "trial_result_integrity.json").exists():
            result = json.loads((folder / "trial_result_integrity.json").read_text())
        row = dict(t, run_status=status.get("status", "unrun"), exit_code=status.get("exit_code"))
        if result and result.get("status") == "integrity_stopped":
            row.update(stage=None, stage_name=None, valid=False, latency_confounded=result["latency"]["latency_confounded"],
                       invalid_reasons=["integrity_stopped"], integrity_stopped=True,
                       integrity_stage_reached=result["stage_name"], integrity_stop_t_s=(result.get("integrity_stop") or {}).get("t_s"),
                       stop_reason=result.get("stop_reason"), rpc_p95_s=result["latency"].get("rpc_wall_p95_s"))
        elif result and result.get("status") in ("scored", "invalid"):
            row.update(stage=result["stage"], stage_name=result["stage_name"], dropped=result["dropped"],
                       stop_reason=result.get("stop_reason"), duration_s=result.get("duration_s"),
                       valid=result["status"] == "scored", invalid_reasons=result.get("invalid_reasons", []),
                       latency_confounded=result["latency"]["latency_confounded"],
                       rpc_p95_s=result["latency"].get("rpc_wall_p95_s"), capped=result["latency"].get("capped_lead_rejections"),
                       reach_error_m=result.get("reach_error_m"), min_pad_object_m=result.get("min_pad_object_m"))
        else:
            row.update(stage=None, stage_name=None, valid=False, latency_confounded=None,
                       invalid_reasons=[(result or {}).get("status", status.get("status", "unrun"))])
        rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["recipe"])].append(r)
    table = []
    for (model, recipe), members in sorted(groups.items()):
        scored = [r for r in members if r["stage"] is not None]
        primary = [r for r in scored if r["valid"] and not r["latency_confounded"]]
        counts = {s: sum(1 for r in scored if r["stage_name"] == s) for s in STAGES}
        placed = sum(1 for r in primary if r["stage"] == 4)
        box = sum(1 for r in primary if r["stage"] >= 3)
        stops = defaultdict(int)
        stage_by_stop = defaultdict(lambda: defaultdict(int))
        for r in scored:
            stops[str(r.get("stop_reason"))] += 1
            stage_by_stop[r["stage_name"]][str(r.get("stop_reason"))] += 1
        table.append(dict(model=model, recipe=recipe, planned=len(members), scored=len(scored), primary_n=len(primary),
                          placed=placed, placed_rate=round(placed / len(primary), 3) if primary else None,
                          placed_ci95=wilson(placed, len(primary)), stage3plus=box,
                          stage3plus_rate=round(box / len(primary), 3) if primary else None,
                          mean_stage=round(sum(r["stage"] for r in primary) / len(primary), 2) if primary else None,
                          mean_stage_all_scored=round(sum(r["stage"] for r in scored) / len(scored), 2) if scored else None,
                          stage_counts_all_scored=counts, dropped=sum(1 for r in scored if r.get("dropped")),
                          invalid=sum(1 for r in scored if not r["valid"]),
                          latency_confounded=sum(1 for r in scored if r["latency_confounded"]),
                          unrun_or_failed=len(members) - len(scored), stop_reasons=dict(stops),
                          stage_by_stop_reason={k: dict(v) for k, v in stage_by_stop.items()},
                          integrity_stopped=sum(1 for r in members if r.get("integrity_stopped")),
                          integrity_stage_reached=[r["integrity_stage_reached"] for r in members if r.get("integrity_stopped")],
                          failed_statuses=dict(defaultdict(int, {str(r["run_status"]): sum(1 for m in members if m["run_status"] == r["run_status"]) for r in members if r["stage"] is None}))))
    return table


def paired(rows, a, b):
    """a, b are (model, recipe) tuples; identities shared by (start, seed)."""
    by = {}
    for r in rows:
        if r["stage"] is None or not r["valid"] or r["latency_confounded"]:
            continue
        by.setdefault((r["model"], r["recipe"]), {})[(r["start"], r["seed"])] = r["stage"]
    common = sorted(set(by.get(a, {})) & set(by.get(b, {})))
    diffs = [by[a][k] - by[b][k] for k in common]
    return dict(a=a, b=b, shared_identities=len(common), a_better=sum(d > 0 for d in diffs), b_better=sum(d < 0 for d in diffs),
                ties=sum(d == 0 for d in diffs), sign_test=sign_test(diffs))


def markdown(table):
    lines = ["| model | recipe | primary n / scored / planned | placed (CI95) | stage≥3 | mean stage | no grab / grab / pick / in-box / placed | dropped | confounded | invalid | integrity-stopped (stage reached) | stops |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        c = r["stage_counts_all_scored"]
        lines.append(f"| {r['model']} | {r['recipe']} | {r['primary_n']} / {r['scored']} / {r['planned']} | "
                     f"{r['placed']} ({r['placed_rate']}, {r['placed_ci95']}) | {r['stage3plus']} ({r['stage3plus_rate']}) | {r['mean_stage']} | "
                     f"{c['no_grab']} / {c['grab']} / {c['pick']} / {c['in_box_gripped']} / {c['placed']} | {r['dropped']} | "
                     f"{r['latency_confounded']} | {r['invalid']} | {r['integrity_stopped']} {r['integrity_stage_reached'] or ''} | {json.dumps(r['stop_reasons'])} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="local copy of the stage raw root (rollouts/*)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                        help="paired comparison of two model:recipe setups")
    parser.add_argument("--baseline", nargs=2, action="append", default=[], metavar=("STUDY", "RAW"),
                        help="extra (study, raw) pairs whose trials join the table and the paired tests")
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    rows = load(study, args.raw)
    for extra_study, extra_raw in args.baseline:
        rows += load(json.loads(Path(extra_study).read_text()), extra_raw)
    table = summarize(rows)
    pairs = [paired(rows, tuple(a.split(":")), tuple(b.split(":"))) for a, b in args.pair]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "trials.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    (args.output / "table.json").write_text(json.dumps({"table": table, "pairs": pairs}, indent=2, default=str) + "\n")
    md = markdown(table)
    if pairs:
        md += "\n\nPaired (exact sign test on ordinal stage, shared identities):\n" + "\n".join(
            f"- {p['a'][0]}:{p['a'][1]} vs {p['b'][0]}:{p['b'][1]}: n={p['shared_identities']}, wins {p['a_better']} / losses {p['b_better']} / ties {p['ties']}, p={p['sign_test']['p_two_sided']}" for p in pairs)
    (args.output / "table.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
