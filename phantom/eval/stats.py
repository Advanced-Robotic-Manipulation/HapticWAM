"""Paired closed-loop statistics and offline bootstrap intervals for the paper.

Closed-loop (rig) episodes are paired by (task, seed): PICK.sh derives the
seed from the cell number (100 + cell) and a multi-episode process increments
it per episode, so the SAME seed on two arms means the same start jitter and,
by protocol, the same object placement. An arm is identified by its checkpoint
tag (`ckpt:<basename>`) or, when given, the menu label.

Outcome per episode is ORDINAL: 0 = fail, 1 = grasp (operator note contains
"grasp"), 2 = lift ("lift"), 3 = placed (operator verdict `s`). Episodes with
no verdict (status != finalized, success None) are excluded — they are the
"unlabeled" ones run_deploy leaves as aborted.

    python -m phantom.eval.stats pairs data/episodes/deploy/20260911 \
        --arm-a ckpt:BEST.pt --arm-b ckpt:student_002000.pt [--task waffles]

    python -m phantom.eval.stats offline eval4/ftA_teacher_nfe1_s4.json eval4/stu_ftA_r2_nfe1_s4.json

The paired test is the exact two-sided sign test on discordant pairs (ties
dropped) — no distributional assumption, which is what ordinal outcomes and
n ≈ 20 allow. Offline: percentile bootstrap over the val windows for mean and
median endpoint error, and a paired bootstrap for the difference between two
checkpoints scored on the same windows.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDINAL = {"fail": 0, "grasp": 1, "lift": 2, "placed": 3}


# ----------------------------------------------------------------- outcomes
def episode_outcome(meta: dict) -> int | None:
    """Ordinal outcome from an episode's meta.json, or None when unlabeled."""
    if meta.get("status") != "finalized" or meta.get("success") is None:
        return None
    if meta.get("success"):
        return ORDINAL["placed"]
    note = str(meta.get("notes") or "").lower()
    if "lift" in note:
        return ORDINAL["lift"]
    if "grasp" in note:
        return ORDINAL["grasp"]
    return ORDINAL["fail"]


def episode_seed(meta: dict) -> int | None:
    for t in meta.get("tags", []) or []:
        if isinstance(t, str) and t.startswith("seed:"):
            try:
                return int(t.split(":", 1)[1])
            except ValueError:
                return None
    d = meta.get("deploy_overrides")
    if isinstance(d, dict) and d.get("seed") is not None:
        return int(d["seed"])
    return None


def episode_arm(meta: dict) -> str | None:
    for t in meta.get("tags", []) or []:
        if isinstance(t, str) and t.startswith("ckpt:"):
            return t
    return None


def load_episodes(root: Path) -> list[dict]:
    """Every ep_*/meta.json under root, with derived seed / arm / outcome."""
    out = []
    for mp in sorted(root.rglob("meta.json")):
        if not mp.parent.name.startswith("ep_"):
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"path": str(mp.parent), "task": meta.get("task"),
                    "seed": episode_seed(meta), "arm": episode_arm(meta),
                    "outcome": episode_outcome(meta),
                    "stop": (meta.get("deploy_overrides") or {}).get("stop_reason")
                    if isinstance(meta.get("deploy_overrides"), dict) else None})
    return out


# ------------------------------------------------------------------- pairing
def pair_episodes(eps: list[dict], arm_a: str, arm_b: str, task: str | None = None
                  ) -> tuple[list[dict], dict]:
    """Pairs = (task, seed) present on both arms with a labeled outcome each.
    When an arm has several labeled episodes for one seed (a re-run), the LAST
    one counts — the protocol says a censored cell is re-run under the same
    number. Returns (pairs, diagnostics)."""
    by_key: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    unlabeled = 0
    for e in eps:
        if task and e["task"] != task:
            continue
        if e["arm"] not in (arm_a, arm_b) or e["seed"] is None:
            continue
        if e["outcome"] is None:
            unlabeled += 1
            continue
        by_key[(e["task"], e["seed"])][e["arm"]].append(e)
    pairs, unpaired = [], []
    for key in sorted(by_key, key=lambda k: (str(k[0]), k[1])):
        arms = by_key[key]
        if arm_a in arms and arm_b in arms:
            a, b = arms[arm_a][-1], arms[arm_b][-1]
            pairs.append({"task": key[0], "seed": key[1], "a": a["outcome"], "b": b["outcome"],
                          "a_path": a["path"], "b_path": b["path"],
                          "reruns": len(arms[arm_a]) + len(arms[arm_b]) - 2})
        else:
            unpaired.append({"task": key[0], "seed": key[1], "have": sorted(arms)})
    return pairs, {"unlabeled_skipped": unlabeled, "unpaired": unpaired}


# ---------------------------------------------------------------- sign test
def sign_test(pairs: list[dict]) -> dict:
    """Exact two-sided sign test: B > A wins vs A > B wins, ties dropped."""
    b_wins = sum(1 for p in pairs if p["b"] > p["a"])
    a_wins = sum(1 for p in pairs if p["a"] > p["b"])
    ties = len(pairs) - b_wins - a_wins
    n = b_wins + a_wins
    if n == 0:
        p_two = 1.0
    else:
        k = min(a_wins, b_wins)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
        p_two = min(1.0, 2 * tail)
    return {"n_pairs": len(pairs), "b_wins": b_wins, "a_wins": a_wins, "ties": ties,
            "p_two_sided": p_two,
            "b_win_rate_of_discordant": (b_wins / n) if n else float("nan")}


def outcome_table(pairs: list[dict]) -> dict:
    names = {v: k for k, v in ORDINAL.items()}
    cnt_a = {names[v]: sum(1 for p in pairs if p["a"] == v) for v in sorted(names)}
    cnt_b = {names[v]: sum(1 for p in pairs if p["b"] == v) for v in sorted(names)}
    return {"a": cnt_a, "b": cnt_b,
            "a_placed_rate": np.mean([p["a"] == 3 for p in pairs]) if pairs else float("nan"),
            "b_placed_rate": np.mean([p["b"] == 3 for p in pairs]) if pairs else float("nan"),
            "a_grasp_or_better": np.mean([p["a"] >= 1 for p in pairs]) if pairs else float("nan"),
            "b_grasp_or_better": np.mean([p["b"] >= 1 for p in pairs]) if pairs else float("nan")}


# ------------------------------------------------------------ offline boots
def _rows_errors(path: Path, metric: str = "endpoint_err_mm") -> np.ndarray:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d.get("rows", [])
    vals = np.asarray([r[metric] for r in rows if r.get(metric) is not None], dtype=float)
    return vals[np.isfinite(vals)]


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 5000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap over windows."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx])
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def offline_summary(path: Path, metric: str = "endpoint_err_mm", n_boot: int = 5000) -> dict:
    x = _rows_errors(path, metric)
    m, mlo, mhi = bootstrap_ci(x, np.mean, n_boot)
    md, mdlo, mdhi = bootstrap_ci(x, np.median, n_boot)
    return {"file": str(path), "n_windows": int(x.size), "mean": m, "mean_ci": [mlo, mhi],
            "median": md, "median_ci": [mdlo, mdhi]}


def paired_offline_diff(path_a: Path, path_b: Path, metric: str = "endpoint_err_mm",
                        n_boot: int = 5000, seed: int = 0) -> dict:
    """B − A on the same windows (rows matched by order; both files must come
    from the same split/seed count). Negative = B better."""
    a, b = _rows_errors(path_a, metric), _rows_errors(path_b, metric)
    n = min(a.size, b.size)
    if n == 0:
        return {"n": 0}
    d = b[:n] - a[:n]
    dm, lo, hi = bootstrap_ci(d, np.mean, n_boot, seed)
    dmd, mlo, mhi = bootstrap_ci(d, np.median, n_boot, seed)
    return {"n": int(n), "mean_diff": dm, "mean_diff_ci": [lo, hi],
            "median_diff": dmd, "median_diff_ci": [mlo, mhi],
            "frac_b_better": float(np.mean(d < 0))}


# ---------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs", help="paired sign test over rig episodes")
    p.add_argument("root", type=Path)
    p.add_argument("--arm-a", required=True, help="ckpt tag of arm A, e.g. ckpt:BEST.pt")
    p.add_argument("--arm-b", required=True, help="ckpt tag of arm B, e.g. ckpt:student_002000.pt")
    p.add_argument("--task", default=None)
    p.add_argument("--json-out", type=Path, default=None)
    o = sub.add_parser("offline", help="bootstrap CIs for terminal_eval json files (2 files = paired diff)")
    o.add_argument("files", nargs="+", type=Path)
    o.add_argument("--metric", default="endpoint_err_mm")
    args = ap.parse_args(argv)

    if args.cmd == "pairs":
        eps = load_episodes(args.root)
        pairs, diag = pair_episodes(eps, args.arm_a, args.arm_b, args.task)
        res = {"arm_a": args.arm_a, "arm_b": args.arm_b, "task": args.task,
               "n_episodes_seen": len(eps), **diag, "pairs": pairs,
               "sign_test": sign_test(pairs), "outcomes": outcome_table(pairs)}
        st = res["sign_test"]
        print(f"pairs: {st['n_pairs']}  B>A: {st['b_wins']}  A>B: {st['a_wins']}  ties: {st['ties']}  "
              f"p(two-sided sign) = {st['p_two_sided']:.3f}")
        print("A outcomes:", res["outcomes"]["a"], " B outcomes:", res["outcomes"]["b"])
        if diag["unlabeled_skipped"]:
            print(f"unlabeled (skipped): {diag['unlabeled_skipped']}")
        if diag["unpaired"]:
            print("unpaired seeds:", [(u['task'], u['seed'], u['have']) for u in diag["unpaired"]])
        if args.json_out:
            args.json_out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        return 0
    for f in args.files:
        s = offline_summary(f, args.metric)
        print(f"{f.name}: n={s['n_windows']} mean {s['mean']:.2f} [{s['mean_ci'][0]:.2f}, {s['mean_ci'][1]:.2f}]  "
              f"median {s['median']:.2f} [{s['median_ci'][0]:.2f}, {s['median_ci'][1]:.2f}]")
    if len(args.files) == 2:
        d = paired_offline_diff(args.files[0], args.files[1], args.metric)
        print(f"paired B−A: mean {d['mean_diff']:.2f} [{d['mean_diff_ci'][0]:.2f}, {d['mean_diff_ci'][1]:.2f}]  "
              f"median {d['median_diff']:.2f} [{d['median_diff_ci'][0]:.2f}, {d['median_diff_ci'][1]:.2f}]  "
              f"B better on {d['frac_b_better']:.0%} of {d['n']} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
