#!/usr/bin/env python3
"""Does the imagined-future agreement score know an episode is going wrong?

Every rig launch since 328a93c runs with `--agreement-shadow`: at each replan
the policy scores the K imagined futures (distance of each candidate to their
elementwise median) and writes the result into the plan diag, which the planner
keeps in the episode's `planner_trace.json` rows under "diag". The default rule
still picks the chunk; the shadow is a pure record of what the agreement rule
would have said. When the selector itself is on (`--select-by video_agreement`)
the same numbers are there under the un-shadowed key names and the pick IS the
agreement rule's pick.

This tool joins those per-replan scores to the EPISODE outcome (the ordinal
0 reach / 1 close / 2 lift / 3 placed of `phantom.eval.stats`, rule-derived
where streams allow) and to the stop reason, and asks three questions:

1. separation — does a per-episode summary of the agreement distance (mean /
   percentile / max of the chosen chunk's distance, of the K spread, or the
   fraction of replans above a pooled threshold) rank the failed episodes above
   the placed ones? Reported as AUC with a percentile bootstrap CI that
   resamples EPISODES (replans within an episode are not independent draws).
2. pre-stop window — in the last few accepted replans before a hard stop
   (safety_stop / control_lost), is the distance higher than in every other
   replan? Mann-Whitney U (scipy when installed, otherwise a permutation test
   on the same statistic) plus a rank-biserial effect size.
3. disagreement — how often would the agreement rule have kept a different
   candidate than the default rule, and is that more frequent in the episodes
   that failed?

READ THE LENGTH SECTION FIRST. Episode duration is confounded with outcome: a
failure is stopped a few replans into the reach while a placement runs through
grasp, lift and place, and the late phases are where the imagined futures
diverge anyway. So a whole-episode feature partly measures "how long did this
run". The report always prints duration alone scored as if it were a feature,
plus the whole AUC table recomputed on the first N replans of every episode;
`--first-n N` length-matches every table in the report instead.

Nothing here is an arm: it is hindsight over recorded diags. An AUC near 0.5 is
the honest answer that the score carries no episode-level signal, and the
bootstrap CI is what says whether ~20 episodes could tell the difference.

    tools/rig/analysis/agreement_outcome.py data/episodes/deploy/20260915 \
        --hw configs/hardware.nuc.yaml [--jsonl replay_records.jsonl] \
        [--json-out agreement_outcome.json] [--percentile 90] [--first-n 5]

Deploy days that ran before `--agreement-shadow` shipped carry no agreement
numbers at all; for those the replay tool's --jsonl is the only source, and it
may be given with no day folder.

`episode_features`, `auc_ci` and `prestop_test` are pure (numpy only, scipy
optional) so the replay tool can import them and feed its own records, which
share this module's per-replan schema with source "replay".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:                                   # run as a script
    sys.path.insert(0, str(_REPO))

#: stop reasons that mean the episode was cut short by the hardware or the
#: guards, not by the operator or a finished placement
HARD_STOPS = ("safety_stop", "control_lost")

#: the per-replan record schema, shared with the replay tool
RECORD_KEYS = ("episode", "day", "task", "seed", "ckpt", "label", "outcome",
               "stop", "i", "t", "k_seeds", "agreement", "pick",
               "agreement_of_pick", "spread", "trace_pick", "source")

#: per-episode features scored against the outcome (higher = worse, by
#: construction: every one is a distance or a rate of large distances)
FEATURES = ("mean_of_pick", "pctl_of_pick", "max_of_pick",
            "mean_spread", "pctl_spread", "max_spread",
            "frac_above", "disagree_rate")

_FEATURE_ABBR = {"mean_of_pick": "mean_p", "pctl_of_pick": "pct_p",
                 "max_of_pick": "max_p", "mean_spread": "mean_s",
                 "pctl_spread": "pct_s", "max_spread": "max_s",
                 "frac_above": "frac>t", "disagree_rate": "disag"}


# --------------------------------------------------------------- extraction
def _num_list(v) -> list[float] | None:
    if not isinstance(v, (list, tuple)) or not v:
        return None
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return out if all(np.isfinite(out)) else None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def diag_agreement(diag: dict) -> dict | None:
    """The agreement numbers of ONE plan diag, or None when the replan carries
    none (shadow off, a scoring error, or a run older than 328a93c).

    Shadow rows: `agreement_of_pick` is the DEFAULT rule's pick distance
    (`video_agreement_shadow_of_pick`) and `trace_pick` the default rule's
    `k_pick`, while `pick` is the candidate the agreement rule would have kept.
    Selector rows: the rule IS the pick, so `pick`, `trace_pick` and the
    distance coincide."""
    if not isinstance(diag, dict):
        return None
    k_pick = diag.get("k_pick")
    k_pick = int(k_pick) if isinstance(k_pick, (int, float)) and not isinstance(k_pick, bool) else None
    scores = _num_list(diag.get("video_agreement_shadow"))
    if scores is not None:
        pick = diag.get("video_agreement_shadow_pick")
        pick = int(pick) if isinstance(pick, (int, float)) and not isinstance(pick, bool) else int(np.argmin(scores))
        of_pick = _num(diag.get("video_agreement_shadow_of_pick"))
        if of_pick is None and k_pick is not None and 0 <= k_pick < len(scores):
            of_pick = float(scores[k_pick])
        spread = _num(diag.get("video_agreement_shadow_spread"))
        source = "shadow"
    else:
        scores = _num_list(diag.get("video_agreement"))
        if scores is None:
            return None
        pick = k_pick if k_pick is not None else int(np.argmin(scores))
        of_pick = _num(diag.get("video_agreement_pick"))
        if of_pick is None and 0 <= pick < len(scores):
            of_pick = float(scores[pick])
        spread = _num(diag.get("video_agreement_spread"))
        k_pick = pick
        source = "selector"
    if of_pick is None:
        return None
    if spread is None:
        spread = float(max(scores) - min(scores))
    k_seeds = diag.get("k_seeds")
    k_seeds = int(k_seeds) if isinstance(k_seeds, (int, float)) and not isinstance(k_seeds, bool) else len(scores)
    return {"k_seeds": k_seeds, "agreement": scores, "pick": int(pick),
            "agreement_of_pick": float(of_pick), "spread": float(spread),
            "trace_pick": k_pick, "source": source}


def trace_records(trace, base: dict) -> list[dict]:
    """One record per ACCEPTED replan of a planner trace that carries agreement
    numbers. `i` is the row's index in the trace (what replay_rig's `t_master`
    indexes), NOT a count of the accepted rows."""
    out = []
    if not isinstance(trace, list):
        return out
    for i, row in enumerate(trace):
        if not isinstance(row, dict) or not row.get("accepted"):
            continue
        got = diag_agreement(row.get("diag"))
        if got is None:
            continue
        rec = {**base, "i": int(i), "t": _num(row.get("t"))}
        rec.update(got)
        out.append({k: rec.get(k) for k in RECORD_KEYS})
    return out


def _tag_value(tags, prefix: str) -> str | None:
    for t in sorted(tags or ()):
        if isinstance(t, str) and t.startswith(prefix):
            return t[len(prefix):] or None
    return None


def day_records(day_dir, hw=None) -> tuple[list[dict], dict]:
    """(records, diagnostics) for one deploy day folder. Episodes without a
    readable `planner_trace.json`, and episodes whose replans carry no
    agreement numbers, are counted, never fatal."""
    from phantom.eval import stats as S

    root = Path(day_dir)
    day = root.name
    recs: list[dict] = []
    diag = {"day": day, "n_episodes": 0, "no_trace": [], "unreadable_trace": [],
            "no_agreement": [], "with_agreement": 0}
    for ep in S.load_episodes(root, hw):
        diag["n_episodes"] += 1
        path = Path(ep["path"])
        tags = ep.get("arm_tags") or set()
        base = {"episode": path.name, "day": day, "task": ep.get("task"),
                "seed": ep.get("seed"), "ckpt": _tag_value(tags, "ckpt:"),
                "label": _tag_value(tags, "label:"), "outcome": ep.get("outcome"),
                "stop": ep.get("stop")}
        tp = path / "planner_trace.json"
        if not tp.exists():
            diag["no_trace"].append(path.name)
            continue
        try:
            trace = json.loads(tp.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            diag["unreadable_trace"].append(f"{path.name}: {type(e).__name__}")
            continue
        got = trace_records(trace, base)
        if got:
            recs.extend(got)
            diag["with_agreement"] += 1
        else:
            diag["no_agreement"].append(path.name)
    return recs, diag


def read_jsonl(path) -> tuple[list[dict], dict]:
    """Extra records (same schema, e.g. a replay tool's) — one JSON object per
    line. Lines that are not an object with the numbers this tool needs are
    counted and skipped."""
    out, bad = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if not isinstance(r, dict) or _num(r.get("agreement_of_pick")) is None:
            bad += 1
            continue
        rec = {k: r.get(k) for k in RECORD_KEYS}
        rec["agreement"] = _num_list(r.get("agreement")) or []
        rec["agreement_of_pick"] = _num(r.get("agreement_of_pick"))
        rec["spread"] = _num(r.get("spread"))
        if rec["spread"] is None:
            rec["spread"] = (float(max(rec["agreement"]) - min(rec["agreement"]))
                             if rec["agreement"] else float("nan"))
        rec["source"] = r.get("source") or "replay"
        out.append(rec)
    return out, {"file": str(path), "n": len(out), "n_skipped": bad}


# ------------------------------------------------------------------ features
def episode_key(rec: dict) -> tuple:
    """Episodes are identified by (day, episode): cell seeds and folder names
    repeat across deploy days."""
    return (rec.get("day"), rec.get("episode"))


def truncate_records(records, first_n: int | None, drop_short: bool = False):
    """The first `first_n` accepted replans of every episode (by trace index).

    LENGTH MATCHING. Episode duration is confounded with outcome: a failure is
    stopped a few replans into the reach, a placement runs through grasp, lift
    and place, and the late phases are where the imagined futures diverge
    anyway. Any feature pooled over a whole episode therefore reads partly as
    "how long did this episode last". Comparing the SAME opening window of
    every episode removes that channel.

    Two rules, and NEITHER is clean — the report prints both:

    * truncate (default) keeps every episode, so an 8-replan failure and a
      92-replan placement both contribute, but the short one contributes fewer
      replans: exposure is still unequal, which is the very confound the
      matching is for.
    * `drop_short` removes the episodes with fewer than `first_n` replans, so
      exposure matches exactly — at the cost of deleting the short episodes,
      which are overwhelmingly the FAILURES. The AUC is then computed on a
      different population, and the positive count falls as N grows.

    They agree wherever nearly every episode reaches `first_n`, which is what
    makes a small N the honest operating point. `n_short` / `n_reaching` in the
    report say how far apart the two samples are."""
    if first_n is None:
        return list(records)
    n = max(0, int(first_n))
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by[episode_key(r)].append(r)
    out = []
    for rs in by.values():
        if drop_short and len(rs) < n:
            continue
        rs = sorted(rs, key=lambda r: (r.get("i") if r.get("i") is not None else 0))
        out.extend(rs[:n])
    return out


def episode_features(records, percentile: float = 90.0, threshold: float | None = None,
                     first_n: int | None = None, drop_short: bool = False) -> list[dict]:
    """One row per episode: the per-replan agreement distances summarised.

    `threshold` is the POOLED cut `frac_above` counts against — pass the value
    from `pooled_threshold` so the fraction is comparable across episodes; the
    default derives it from `records` themselves. `first_n` (with `drop_short`)
    length-matches the episodes first — see `truncate_records`, which documents
    why that matters and why the two rules disagree. Episodes keep their
    outcome, stop, task and label, including unlabeled ones (outcome None),
    which every caller is free to list and every AUC drops."""
    records = truncate_records(records, first_n, drop_short)
    if threshold is None:
        threshold = pooled_threshold(records, percentile)
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by[episode_key(r)].append(r)
    out = []
    for key in sorted(by, key=lambda k: (str(k[0]), str(k[1]))):
        rs = sorted(by[key], key=lambda r: (r.get("i") if r.get("i") is not None else 0))
        first = rs[0]
        ofp = np.asarray([_num(r["agreement_of_pick"]) for r in rs], dtype=float)
        spr = np.asarray([_num(r["spread"]) if _num(r["spread"]) is not None else np.nan
                          for r in rs], dtype=float)
        # only rows where the two rules CAN differ: a selector row's pick is
        # the agreement rule's own, so counting it would dilute the rate
        dis = [int(r["pick"] != r["trace_pick"]) for r in rs
               if r.get("pick") is not None and r.get("trace_pick") is not None
               and str(r.get("source")) != "selector"]
        row = {"day": key[0], "episode": key[1], "task": first.get("task"),
               "seed": first.get("seed"), "ckpt": first.get("ckpt"),
               "label": first.get("label"), "outcome": first.get("outcome"),
               "stop": first.get("stop"),
               "sources": sorted({str(r.get("source")) for r in rs}),
               "n_replans": len(rs), "percentile": float(percentile),
               "threshold": float(threshold) if np.isfinite(threshold) else float("nan"),
               "mean_of_pick": float(np.mean(ofp)), "max_of_pick": float(np.max(ofp)),
               "pctl_of_pick": float(np.percentile(ofp, percentile)),
               "mean_spread": float(np.nanmean(spr)) if np.isfinite(spr).any() else float("nan"),
               "max_spread": float(np.nanmax(spr)) if np.isfinite(spr).any() else float("nan"),
               "pctl_spread": (float(np.percentile(spr[np.isfinite(spr)], percentile))
                               if np.isfinite(spr).any() else float("nan")),
               "frac_above": (float(np.mean(ofp > threshold))
                              if np.isfinite(threshold) else float("nan")),
               "n_disagree": int(sum(dis)),
               "disagree_rate": float(np.mean(dis)) if dis else float("nan")}
        out.append(row)
    return out


def pooled_threshold(records, percentile: float = 90.0) -> float:
    """The cut `frac_above` counts against: the given percentile of EVERY
    replan's `agreement_of_pick`, pooled over the episodes given."""
    x = np.asarray([_num(r["agreement_of_pick"]) for r in records
                    if _num(r.get("agreement_of_pick")) is not None], dtype=float)
    return float(np.percentile(x, percentile)) if x.size else float("nan")


# ----------------------------------------------------------------- statistics
def rankdata(x) -> np.ndarray:
    """Average ranks (scipy's `rankdata` without scipy)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    ranks = np.empty(n, dtype=float)
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(scores, labels) -> float:
    """P(score of a positive > score of a negative) + half the ties — the
    Mann-Whitney U statistic normalised. NaN when either class is empty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_ci(scores, labels, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """AUC with a percentile bootstrap CI resampling the ITEMS given (call it
    with one score per EPISODE — replans of one episode are not independent).
    Resamples that lose a class are dropped and counted."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    n_pos, n_neg = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    point = auc(scores, labels)
    out = {"auc": point, "lo": float("nan"), "hi": float("nan"),
           "n_pos": n_pos, "n_neg": n_neg, "n_boot": int(n_boot), "n_boot_used": 0,
           "n_dropped": int(np.sum(~ok))}
    if not np.isfinite(point) or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    n = scores.size
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        v = auc(scores[idx], labels[idx])
        if np.isfinite(v):
            boots.append(v)
    if boots:
        b = np.asarray(boots, dtype=float)
        out.update(lo=float(np.quantile(b, alpha / 2)),
                   hi=float(np.quantile(b, 1 - alpha / 2)), n_boot_used=len(boots))
    return out


def mannwhitney(a, b, n_perm: int = 5000, seed: int = 0) -> dict:
    """Two-sided Mann-Whitney U of a vs b. scipy when importable, otherwise a
    permutation test on the same U (ties handled by average ranks either way).
    `rank_biserial` = 2 * P(a > b) - 1, in [-1, 1]; positive = a is larger."""
    a = np.asarray([v for v in np.asarray(a, dtype=float) if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in np.asarray(b, dtype=float) if np.isfinite(v)], dtype=float)
    out = {"n_a": int(a.size), "n_b": int(b.size), "u": float("nan"),
           "p": float("nan"), "rank_biserial": float("nan"), "method": "none",
           "median_a": float(np.median(a)) if a.size else float("nan"),
           "median_b": float(np.median(b)) if b.size else float("nan"),
           "mean_a": float(np.mean(a)) if a.size else float("nan"),
           "mean_b": float(np.mean(b)) if b.size else float("nan")}
    if a.size == 0 or b.size == 0:
        return out
    both = np.concatenate([a, b])
    lab = np.concatenate([np.ones(a.size, int), np.zeros(b.size, int)])
    u_obs = auc(both, lab) * a.size * b.size
    out["u"] = float(u_obs)
    out["rank_biserial"] = float(2.0 * (u_obs / (a.size * b.size)) - 1.0)
    try:
        from scipy.stats import mannwhitneyu           # noqa: PLC0415 — optional
    except ImportError:
        rng = np.random.default_rng(seed)
        centre = a.size * b.size / 2.0
        hits = 0
        for _ in range(int(n_perm)):
            perm = rng.permutation(lab)
            u = auc(both, perm) * a.size * b.size
            if abs(u - centre) >= abs(u_obs - centre) - 1e-12:
                hits += 1
        out.update(p=float((hits + 1) / (int(n_perm) + 1)),
                   method=f"permutation({int(n_perm)})")
    else:
        res = mannwhitneyu(a, b, alternative="two-sided")
        out.update(u=float(res.statistic), p=float(res.pvalue), method="scipy")
        out["rank_biserial"] = float(2.0 * (float(res.statistic) / (a.size * b.size)) - 1.0)
    return out


def prestop_test(records, window: int = 3, n_perm: int = 5000, seed: int = 0) -> dict:
    """`agreement_of_pick` in the last `window` accepted replans before a HARD
    stop (safety_stop / control_lost) vs every other replan.

    The "rest" is deliberately every other replan, the earlier ones of the same
    episodes included: the claim under test is that the score rises INTO the
    stop, not that hard-stopped episodes are worse overall."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by[episode_key(r)].append(r)
    pre, rest, hard_eps = [], [], []
    for key, rs in by.items():
        rs = sorted(rs, key=lambda r: (r.get("i") if r.get("i") is not None else 0))
        vals = [_num(r["agreement_of_pick"]) for r in rs]
        vals = [v for v in vals if v is not None]
        if str(rs[0].get("stop")) in HARD_STOPS and vals:
            hard_eps.append(key)
            cut = max(0, len(vals) - int(window))
            pre.extend(vals[cut:])
            rest.extend(vals[:cut])
        else:
            rest.extend(vals)
    res = mannwhitney(pre, rest, n_perm=n_perm, seed=seed)
    return {"window": int(window), "n_hard_episodes": len(hard_eps),
            "n_episodes": len(by), "n_pre": res["n_a"], "n_rest": res["n_b"],
            "median_pre": res["median_a"], "median_rest": res["median_b"],
            "mean_pre": res["mean_a"], "mean_rest": res["mean_b"],
            "u": res["u"], "p": res["p"], "rank_biserial": res["rank_biserial"],
            "method": res["method"]}


# ---------------------------------------------------------------- aggregation
def _targets(feats) -> dict[str, list[int | None]]:
    """Per-episode labels. None = the episode cannot answer this question and
    is dropped from that AUC (unlabeled outcome; no recorded stop reason)."""
    not_placed, hard = [], []
    for f in feats:
        o = f.get("outcome")
        not_placed.append(None if o is None else int(int(o) < 3))
        s = f.get("stop")
        hard.append(None if s is None else int(str(s) in HARD_STOPS))
    return {"not_placed": not_placed, "hard_stop": hard}


def group_auc(feats, n_boot: int = 1000, seed: int = 0) -> dict:
    """{target: {feature: auc_ci(...)}} over the episodes given."""
    tg = _targets(feats)
    out = {}
    for name, labels in tg.items():
        keep = [i for i, v in enumerate(labels) if v is not None]
        y = [labels[i] for i in keep]
        per_feature = {}
        for f in FEATURES:
            x = [feats[i].get(f, float("nan")) for i in keep]
            per_feature[f] = auc_ci(x, y, n_boot=n_boot, seed=seed)
        out[name] = {"n_episodes": len(keep),
                     "n_pos": int(sum(y)), "n_neg": int(len(y) - sum(y)),
                     "n_unusable": len(labels) - len(keep),
                     "features": per_feature}
    return out


def groups(records) -> list[tuple[str, list[dict]]]:
    """(name, records) for pooled and for every day / task / label / source."""
    out = [("pooled", list(records))]
    for field, prefix in (("day", "day"), ("task", "task"),
                          ("label", "label"), ("source", "source")):
        by: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            v = r.get(field)
            if v is not None:
                by[str(v)].append(r)
        if len(by) <= 1:
            # a single day / task / label / source adds nothing over "pooled"
            continue
        for k in sorted(by):
            out.append((f"{prefix}:{k}", by[k]))
    return out


def disagreement_summary(records, feats) -> dict:
    """How often the agreement rule would have kept another candidate, overall
    and split by outcome.

    Counted over every row where the two rules CAN differ — shadow rows, and a
    replay tool's rows, whose `trace_pick` is what the rig actually played.
    Selector rows are excluded, never mixed in: there the agreement rule IS the
    pick, so it agrees with itself by construction."""
    shad = [r for r in records if r.get("source") != "selector"
            and r.get("pick") is not None and r.get("trace_pick") is not None]
    d = np.asarray([int(r["pick"] != r["trace_pick"]) for r in shad], dtype=float)
    # CHANCE. Two rules picking independently among K candidates coincide 1/K
    # of the time, so the disagreement rate to beat is 1 - mean(1/K) — 0.75 at
    # K=4. A raw rate near three quarters is not "the rules nearly always
    # differ", it is "the shadow pick is unrelated to the default pick".
    ks = np.asarray([float(r.get("k_seeds") or len(r.get("agreement") or ())) for r in shad],
                    dtype=float)
    ks = ks[ks >= 1]
    out = {"n_comparable_replans": int(d.size),
           "n_shadow_replans": int(sum(1 for r in shad if r.get("source") == "shadow")),
           "n_selector_replans": int(sum(1 for r in records if r.get("source") == "selector")),
           "n_other_replans": int(sum(1 for r in shad if r.get("source") != "shadow")),
           "disagree_rate": float(np.mean(d)) if d.size else float("nan"),
           "chance_disagree_rate": float(np.mean(1.0 - 1.0 / ks)) if ks.size else float("nan"),
           "k_seeds": sorted({int(k) for k in ks})}
    # per-episode rates by outcome (episode-level, so one long episode cannot
    # dominate the comparison)
    fail = [f["disagree_rate"] for f in feats
            if f.get("outcome") is not None and int(f["outcome"]) < 3
            and np.isfinite(f.get("disagree_rate", np.nan))]
    placed = [f["disagree_rate"] for f in feats
              if f.get("outcome") is not None and int(f["outcome"]) == 3
              and np.isfinite(f.get("disagree_rate", np.nan))]
    out.update(n_failed_episodes=len(fail), n_placed_episodes=len(placed),
               mean_rate_failed=float(np.mean(fail)) if fail else float("nan"),
               mean_rate_placed=float(np.mean(placed)) if placed else float("nan"),
               median_rate_failed=float(np.median(fail)) if fail else float("nan"),
               median_rate_placed=float(np.median(placed)) if placed else float("nan"))
    out["diff_failed_minus_placed"] = (out["mean_rate_failed"] - out["mean_rate_placed"]
                                       if fail and placed else float("nan"))
    if fail and placed:
        out["test"] = mannwhitney(fail, placed)
    # the same split on the OTHER failure definition, so a null on one cannot be
    # mistaken for a null on both
    hard = [f["disagree_rate"] for f in feats if str(f.get("stop")) in HARD_STOPS
            and np.isfinite(f.get("disagree_rate", np.nan))]
    soft = [f["disagree_rate"] for f in feats
            if f.get("stop") is not None and str(f.get("stop")) not in HARD_STOPS
            and np.isfinite(f.get("disagree_rate", np.nan))]
    out.update(n_hard_stop_episodes=len(hard), n_no_hard_stop_episodes=len(soft),
               mean_rate_hard_stop=float(np.mean(hard)) if hard else float("nan"),
               mean_rate_no_hard_stop=float(np.mean(soft)) if soft else float("nan"),
               median_rate_hard_stop=float(np.median(hard)) if hard else float("nan"),
               median_rate_no_hard_stop=float(np.median(soft)) if soft else float("nan"))
    out["diff_hard_minus_rest"] = (out["mean_rate_hard_stop"] - out["mean_rate_no_hard_stop"]
                                   if hard and soft else float("nan"))
    if hard and soft:
        out["test_hard_stop"] = mannwhitney(hard, soft)
    # pooled replan-level rates, for the record
    for key, sel in (("pooled_rate_failed", lambda o: o is not None and o < 3),
                     ("pooled_rate_placed", lambda o: o is not None and o == 3)):
        v = [int(r["pick"] != r["trace_pick"]) for r in shad if sel(r.get("outcome"))]
        out[key] = float(np.mean(v)) if v else float("nan")
    return out


def length_summary(records, feats, percentile: float = 90.0, n_boot: int = 1000,
                   seed: int = 0, matched=(5, 10)) -> dict:
    """How much of every AUC above is really episode DURATION, and what the
    same numbers look like length-matched.

    Two readings: the replan count alone scored as if it were a feature (an AUC
    far from 0.5 means duration by itself separates the outcomes, so any
    feature correlated with it inherits that separation), and the whole AUC
    table recomputed on the first N replans of every episode."""
    n = np.asarray([f["n_replans"] for f in feats], dtype=float)
    out = {"n_episodes": len(feats), "matched_n": [int(v) for v in matched]}
    for key, sel in (("failed", lambda f: f["outcome"] is not None and int(f["outcome"]) < 3),
                     ("placed", lambda f: f["outcome"] is not None and int(f["outcome"]) == 3),
                     ("hard_stop", lambda f: str(f["stop"]) in HARD_STOPS),
                     ("other_stop", lambda f: f["stop"] is not None
                      and str(f["stop"]) not in HARD_STOPS)):
        v = [f["n_replans"] for f in feats if sel(f)]
        out[f"median_replans_{key}"] = float(np.median(v)) if v else float("nan")
        out[f"n_{key}"] = len(v)
    # duration alone, scored exactly like a feature (higher n = LONGER, so an
    # AUC well below 0.5 is the early-stop signature of a failure)
    tg = _targets(feats)
    out["auc_n_replans"] = {}
    for target, labels in tg.items():
        keep = [i for i, v in enumerate(labels) if v is not None]
        out["auc_n_replans"][target] = auc_ci([n[i] for i in keep],
                                              [labels[i] for i in keep],
                                              n_boot=n_boot, seed=seed)
    out["matched"] = {}
    for N in matched:
        N = int(N)
        entry = {"n": N,
                 "n_short": int(sum(1 for f in feats if f["n_replans"] < N)),
                 "n_reaching": int(sum(1 for f in feats if f["n_replans"] >= N)),
                 "rules": {}}
        # BOTH matching rules, because neither is clean: "truncate" keeps every
        # episode at unequal exposure, "drop_short" matches exposure by deleting
        # the short episodes — which are the failures. Where the two agree, the
        # choice does not matter; where they do not, the reader must see both.
        for rule, drop in (("truncate", False), ("drop_short", True)):
            recs = truncate_records(records, N, drop_short=drop)
            thr = pooled_threshold(recs, percentile)
            r_entry = {"threshold": thr, "n_records": len(recs),
                       "n_episodes": len({episode_key(r) for r in recs}),
                       "n_episodes_dropped": len(feats) - len({episode_key(r) for r in recs}),
                       "groups": {}}
            for name, rs in groups(recs):
                r_entry["groups"][name] = {
                    "n_episodes": len({episode_key(r) for r in rs}), "n_records": len(rs),
                    "auc": group_auc(episode_features(rs, percentile, thr),
                                     n_boot=n_boot, seed=seed)}
            entry["rules"][rule] = r_entry
        out["matched"][str(N)] = entry
    return out


def analyze(records, percentile: float = 90.0, n_boot: int = 1000, seed: int = 0,
            window: int = 3, first_n: int | None = None, matched=(5, 10),
            drop_short: bool = False) -> dict:
    """The whole report as plain JSON-able data. With `first_n` every table is
    computed on the first N replans of each episode instead of whole episodes
    (`drop_short` picks the matching rule); the length-matched section, which
    prints both rules, is reported either way."""
    records = truncate_records(records, first_n, drop_short)
    thr = pooled_threshold(records, percentile)
    feats = episode_features(records, percentile, thr)
    res = {"n_records": len(records), "n_episodes": len(feats),
           "percentile": float(percentile), "threshold": thr,
           "n_boot": int(n_boot), "seed": int(seed),
           "first_n": None if first_n is None else int(first_n),
           "match_rule": ("drop_short" if drop_short else "truncate") if first_n else None,
           "sources": {s: int(sum(1 for r in records if r.get("source") == s))
                       for s in sorted({str(r.get("source")) for r in records})},
           "episodes": feats,
           "groups": {}, "prestop": {}, "disagreement": disagreement_summary(records, feats)}
    for name, recs in groups(records):
        gf = episode_features(recs, percentile, thr)
        res["groups"][name] = {"n_episodes": len(gf), "n_records": len(recs),
                               "auc": group_auc(gf, n_boot=n_boot, seed=seed)}
    res["prestop"]["pooled"] = prestop_test(records, window=window, seed=seed)
    for s in res["sources"]:
        if len(res["sources"]) > 1:
            res["prestop"][f"source:{s}"] = prestop_test(
                [r for r in records if str(r.get("source")) == s], window=window, seed=seed)
    res["length"] = length_summary(records, feats, percentile=percentile,
                                   n_boot=n_boot, seed=seed,
                                   matched=[int(v) for v in matched
                                            if first_n is None or int(v) < int(first_n)])
    return res


# --------------------------------------------------------------------- report
def _f(v, w=6, p=3) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-".rjust(w)
    return ("-" if not np.isfinite(v) else f"{v:.{p}f}").rjust(w)


_TARGETS = (("not_placed", "not placed (outcome < 3; unlabeled dropped)"),
            ("hard_stop", f"hard stop ({'/'.join(HARD_STOPS)})"))


def _print_auc_table(groups_dict: dict, target: str, only=None, indent: str = "") -> None:
    head = " ".join(f"{_FEATURE_ABBR[f]:>6s}" for f in FEATURES)
    print(f"{indent}{'group':22s} {'n+':>3s} {'n-':>3s} {head}")
    for name, g in groups_dict.items():
        if only is not None and not only(name):
            continue
        a = g["auc"][target]
        if a["n_pos"] == 0 or a["n_neg"] == 0:
            print(f"{indent}{name[:22]:22s} {a['n_pos']:3d} {a['n_neg']:3d} "
                  f"  (one class only — no AUC)")
            continue
        cells = " ".join(_f(a["features"][f]["auc"]) for f in FEATURES)
        print(f"{indent}{name[:22]:22s} {a['n_pos']:3d} {a['n_neg']:3d} {cells}")


#: the features the side-by-side length-matched table shows (the full set for
#: both rules is in --json-out)
_COMPARE = ("mean_of_pick", "pctl_of_pick", "frac_above", "disagree_rate")


def _print_rule_comparison(trunc: dict, drop: dict, target: str, indent: str = "") -> None:
    """One row per group: the same AUCs under both length-matching rules, with
    each rule's own n+/n- so a shrinking positive count is visible, not
    inferred."""
    cols = " ".join(f"{_FEATURE_ABBR[f]:>6s}" for f in _COMPARE)
    print(f"{indent}{'':22s} {'--- truncate: keep all ---':>{9 + len(cols)}s}   "
          f"{'--- drop short ---':>{9 + len(cols)}s}")
    print(f"{indent}{'group':22s} {'n+':>3s} {'n-':>3s} {cols}   {'n+':>3s} {'n-':>3s} {cols}")
    for name in trunc:
        cells = []
        for g in (trunc.get(name), drop.get(name)):
            if g is None:
                cells.append(f"{'-':>3s} {'-':>3s} " + " ".join(f"{'-':>6s}" for _ in _COMPARE))
                continue
            a = g["auc"][target]
            body = (" ".join(_f(a["features"][f]["auc"]) for f in _COMPARE)
                    if a["n_pos"] and a["n_neg"]
                    else f"{'(one class — no AUC)':>{len(cols)}s}")
            cells.append(f"{a['n_pos']:3d} {a['n_neg']:3d} {body}")
        print(f"{indent}{name[:22]:22s} {cells[0]}   {cells[1]}")


def _print_length(res: dict) -> None:
    """The duration confound and the length-matched tables."""
    L = res.get("length")
    if not L:
        return
    print(f"\nepisode length vs outcome: failed episodes ran a median "
          f"{L['median_replans_failed']:.0f} replans (n={L['n_failed']}), placed "
          f"{L['median_replans_placed']:.0f} (n={L['n_placed']}); hard-stopped "
          f"{L['median_replans_hard_stop']:.0f} (n={L['n_hard_stop']}), other stops "
          f"{L['median_replans_other_stop']:.0f} (n={L['n_other_stop']})")
    tail = (f"— already length-matched to the first {res['first_n']} replans, so this is "
            f"the residual" if res.get("first_n") else
            "— every feature above is correlated with it, so read the whole-episode "
            "tables with that in mind")
    for target, _title in _TARGETS:
        a = L["auc_n_replans"][target]
        if not np.isfinite(a["auc"]):
            continue
        print(f"  duration ALONE as a predictor of {target}: AUC {a['auc']:.3f} "
              f"[{a['lo']:.3f}, {a['hi']:.3f}] {tail}")
    for key in sorted(L.get("matched", {}), key=int):
        m = L["matched"][key]
        tr, ds = m["rules"]["truncate"], m["rules"]["drop_short"]
        print(f"\n  length-matched: first {m['n']} accepted replans of every episode "
              f"— {m['n_reaching']} of {L['n_episodes']} episodes reach {m['n']}, "
              f"{m['n_short']} are shorter")
        print(f"  two rules, neither clean: TRUNC keeps all {tr['n_episodes']} episodes at "
              f"unequal exposure; DROP matches exposure over {ds['n_episodes']} episodes by "
              f"deleting the {ds['n_episodes_dropped']} short ones, which are mostly the "
              f"failures — watch n+ fall")
        for target, title in _TARGETS:
            print(f"  AUC vs {title}")
            _print_rule_comparison(tr["groups"], ds["groups"], target, indent="    ")


def print_report(res: dict, load_diag: list[dict] | None = None) -> None:
    src = ", ".join(f"{k} {v}" for k, v in res["sources"].items()) or "none"
    print(f"agreement records: {res['n_records']} accepted replans over "
          f"{res['n_episodes']} episodes ({src})"
          + (f" — LENGTH-MATCHED to the first {res['first_n']} replans of each episode "
             f"({res.get('match_rule')})" if res.get("first_n") else ""))
    pct = res["percentile"]
    print(f"pooled p{pct:g} threshold on agreement_of_pick = {res['threshold']:.6g}")
    for d in load_diag or []:
        extra = []
        for k in ("no_trace", "unreadable_trace", "no_agreement"):
            if d.get(k):
                extra.append(f"{k}={len(d[k])}")
        print(f"  {d.get('day')}: {d.get('n_episodes')} episodes, "
              f"{d.get('with_agreement')} with agreement"
              + (f" ({', '.join(extra)})" if extra else ""))
    if not res["n_records"]:
        print("no agreement numbers found — nothing to score")
        return

    print(f"\nper episode (p{pct:g} of the chosen chunk's distance; frac>t against the pooled cut)")
    print(f"{'episode':38s} {'day':9s} {'task':9s} {'label':12s} {'out':>3s} "
          f"{'stop':15s} {'n':>3s} {'mean':>7s} {'p'+f'{pct:g}':>7s} {'max':>7s} "
          f"{'m_spr':>7s} {'frac>t':>6s} {'disag':>6s}")
    for f in res["episodes"]:
        print(f"{str(f['episode'])[:38]:38s} {str(f['day'])[:9]:9s} "
              f"{str(f['task'])[:9]:9s} {str(f['label'])[:12]:12s} "
              f"{('-' if f['outcome'] is None else f['outcome']):>3} "
              f"{str(f['stop'] or '-')[:15]:15s} {f['n_replans']:3d} "
              f"{_f(f['mean_of_pick'], 7, 4)} {_f(f['pctl_of_pick'], 7, 4)} "
              f"{_f(f['max_of_pick'], 7, 4)} {_f(f['mean_spread'], 7, 4)} "
              f"{_f(f['frac_above'])} {_f(f['disagree_rate'])}")

    for target, title in _TARGETS:
        print(f"\nAUC vs {title} — higher feature = more failure; "
              f"{res['n_boot']} bootstrap resamples over episodes")
        _print_auc_table(res["groups"], target)
        pooled = res["groups"]["pooled"]["auc"][target]
        if pooled["n_pos"] and pooled["n_neg"]:
            print("  pooled 95% CI: " + "  ".join(
                f"{_FEATURE_ABBR[f]} {pooled['features'][f]['auc']:.2f} "
                f"[{pooled['features'][f]['lo']:.2f}, {pooled['features'][f]['hi']:.2f}]"
                for f in FEATURES if np.isfinite(pooled["features"][f]["auc"])))

    _print_length(res)

    for name, p in res["prestop"].items():
        if p["n_pre"] == 0 or p["n_rest"] == 0:
            print(f"\npre-stop window ({name}): no hard stops with scored replans "
                  f"(n_pre={p['n_pre']}, n_rest={p['n_rest']})")
            continue
        print(f"\npre-stop window ({name}): last {p['window']} accepted replans before a hard "
              f"stop ({p['n_hard_episodes']} episodes) vs the rest")
        print(f"  n {p['n_pre']} vs {p['n_rest']}  median {p['median_pre']:.4g} vs "
              f"{p['median_rest']:.4g}  rank-biserial {p['rank_biserial']:+.3f}  "
              f"p = {p['p']:.4g} ({p['method']})")

    d = res["disagreement"]
    tail = (f"; {d['n_selector_replans']} selector replans are excluded — there the "
            f"agreement rule IS the pick" if d["n_selector_replans"] else "")
    if not d["n_comparable_replans"]:
        print("\nagreement vs default pick: no replan can answer it — every record is a "
              "selector row, or none carries both picks" + tail)
    else:
        ch = d.get("chance_disagree_rate", float("nan"))
        ks = "/".join(str(k) for k in d.get("k_seeds") or ()) or "?"
        print(f"\nagreement vs default pick: disagreed on {d['disagree_rate']:.1%} of "
              f"{d['n_comparable_replans']} replans where the two rules can differ "
              f"({d['n_shadow_replans']} shadow, {d['n_other_replans']} replayed)" + tail)
        if np.isfinite(ch):
            print(f"  chance for K={ks} is {ch:.1%} — two rules picking independently among K "
                  f"candidates coincide 1/K of the time, so read this rate against that, "
                  f"not against 0")
    for key, label, n_a, n_b, diff, test in (
            ("failed_minus_placed", ("failed", "placed"), "n_failed_episodes",
             "n_placed_episodes", "diff_failed_minus_placed", "test"),
            ("hard_minus_rest", ("hard stop", "other stops"), "n_hard_stop_episodes",
             "n_no_hard_stop_episodes", "diff_hard_minus_rest", "test_hard_stop")):
        if not np.isfinite(d.get(diff, np.nan)):
            continue
        a_key = "mean_rate_failed" if key.startswith("failed") else "mean_rate_hard_stop"
        b_key = "mean_rate_placed" if key.startswith("failed") else "mean_rate_no_hard_stop"
        t = d.get(test, {})
        print(f"  per-episode rate: {label[0]} {d[a_key]:.1%} (n={d[n_a]}) vs {label[1]} "
              f"{d[b_key]:.1%} (n={d[n_b]}), diff {d[diff]:+.1%}, "
              f"p = {t.get('p', float('nan')):.4g} ({t.get('method', '-')})")
    if not np.isfinite(d.get("diff_failed_minus_placed", np.nan)):
        print("  outcome split unavailable (need both failed and placed episodes "
              "with shadow replans)")


# ------------------------------------------------------------------------ cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("day_dir", nargs="*", type=Path,
                    help="deploy day folder(s), e.g. data/episodes/deploy/20260915. Days that "
                         "ran before --agreement-shadow shipped carry no agreement numbers and "
                         "contribute nothing; for those, --jsonl alone is a complete input")
    ap.add_argument("--hw", type=Path, default=None,
                    help="hardware yaml: outcome stages 0-2 come from the tactile rule on the "
                         "recorded streams (omit = operator notes only)")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="extra per-replan records in this module's schema (one JSON object "
                         "per line), e.g. a replay tool's — counted with source 'replay'")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--percentile", type=float, default=90.0,
                    help="percentile for the per-episode summary and the pooled frac>t cut")
    ap.add_argument("--window", type=int, default=3,
                    help="replans before a hard stop in the pre-stop window test")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first-n", type=int, default=None,
                    help="length-match EVERY table to the first N accepted replans of each "
                         "episode — failures are stopped early, so whole-episode features "
                         "partly measure duration (see the length section, always printed)")
    ap.add_argument("--matched", type=int, nargs="*", default=(5, 10),
                    help="the N values the length-matched section reports (default 5 10). Prefer "
                         "an N nearly every episode reaches: there the two matching rules agree "
                         "and the choice between them cannot change the answer")
    ap.add_argument("--drop-short", action="store_true",
                    help="with --first-n, DROP the episodes shorter than N instead of keeping "
                         "them truncated — exposure then matches exactly, at the cost of "
                         "deleting episodes that are mostly failures (both rules are always "
                         "printed side by side in the length section)")
    args = ap.parse_args(argv)
    if not args.day_dir and args.jsonl is None:
        ap.error("give at least one deploy day folder, or --jsonl")

    hw = None
    if args.hw is not None:
        from phantom.config.hardware import load_hardware
        hw = load_hardware(args.hw)
    records, load_diag = [], []
    for d in args.day_dir:
        recs, diag = day_records(d, hw)
        records.extend(recs)
        load_diag.append(diag)
    if args.jsonl is not None:
        extra, jd = read_jsonl(args.jsonl)
        records.extend(extra)
        load_diag.append({"day": str(args.jsonl), "n_episodes": len({episode_key(r) for r in extra}),
                          "with_agreement": jd["n"], "no_trace": [], "unreadable_trace": [],
                          "no_agreement": [], "n_skipped_lines": jd["n_skipped"]})

    res = analyze(records, percentile=args.percentile, n_boot=args.n_boot,
                  seed=args.seed, window=args.window, first_n=args.first_n,
                  matched=args.matched, drop_short=args.drop_short)
    res["inputs"] = {"day_dirs": [str(d) for d in args.day_dir],
                     "hw": str(args.hw) if args.hw else None,
                     "jsonl": str(args.jsonl) if args.jsonl else None,
                     "load": load_diag}
    print_report(res, load_diag)
    if args.json_out:
        out = dict(res)
        out["records"] = records
        args.json_out.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
