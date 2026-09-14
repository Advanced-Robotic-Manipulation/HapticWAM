"""Paired closed-loop statistics and offline bootstrap intervals for the paper.

Closed-loop (rig) episodes are paired by (task, seed): PICK.sh derives the
seed from the cell number (100 + cell) and a multi-episode process increments
it per episode, so the SAME seed on two arms means the same start jitter and,
by protocol, the same object placement. An arm is identified by its checkpoint
tag (`ckpt:<basename>`) or, when given, the menu label.

Outcome per episode is ORDINAL: 0 = reach (no held grasp), 1 = close (a held,
contacting grasp by the tactile rule — z_close <= Z_MAX, hold >= 2 s, contact
fraction >= 0.8 — whose lift stayed < 50 mm), 2 = lift (the rule's `grasp_ok`,
i.e. contact sustained through a >= 50 mm lift), 3 = placed (operator verdict
`s`). Stages 0-2 come from the recorded streams via the validated P8 rule
(`phantom/eval/grasp_label.py`) whenever a hardware config is given; only
"placed" is the operator's word. Without a hardware config the ordinal falls
back to the operator notes ("f grasp" / "f lift"), which is what the pre-09-12
sheets recorded. Episodes with no verdict (status != finalized, success None)
are excluded — they are the "unlabeled" ones run_deploy leaves as aborted.

    python -m phantom.eval.stats pairs data/episodes/deploy/20260911 \
        --arm-a ckpt:BEST.pt --arm-b ckpt:student_002000.pt [--task waffles] \
        [--hw configs/hardware.nuc.yaml]

    python -m phantom.eval.stats offline eval4/ftA_teacher_nfe1_s4.json eval4/stu_ftA_r2_nfe1_s4.json

The paired test is the exact two-sided sign test on discordant pairs (ties
dropped) — no distributional assumption, which is what ordinal outcomes and
n ≈ 20 allow. Offline: percentile bootstrap for mean and median endpoint
error resampling EPISODES (windows of one episode are not independent), and a
paired bootstrap for the difference between two checkpoints joined on the
same (episode, t0, seed) windows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDINAL = {"reach": 0, "close": 1, "lift": 2, "placed": 3}
# legacy names still accepted in operator notes
_NOTE_STAGE = (("lift", 2), ("grasp", 1), ("close", 1))
_NO_LIFT = re.compile(r"\b(no|never|not|without)[ -]?lift")
# rule reasons that mean "closed and held with contact, but did not lift"
_LIFT_ONLY = {"lift"}
# rule reasons that mean the recording could not be measured at all
_UNMEASURABLE = {"missing_stream", "unreadable_stream", "empty_streams",
                 "no_tactile_stream", "z_max_unknown_task"}


# ----------------------------------------------------------------- outcomes
def episode_outcome(meta: dict) -> int | None:
    """Ordinal outcome from an episode's meta.json ALONE (operator notes), or
    None when unlabeled. Kept as the fallback for episodes without streams."""
    if meta.get("status") != "finalized" or meta.get("success") is None:
        return None
    if meta.get("success"):
        return ORDINAL["placed"]
    note = str(meta.get("notes") or "").lower()
    for word, stage in _NOTE_STAGE:
        if word in note and not (stage == 2 and _NO_LIFT.search(note)):
            return stage
    return ORDINAL["reach"]


def rule_stage(lab) -> int | None:
    """Stages 0-2 from a GraspLabel; None when the rule has no verdict: it
    abstained (the recording ended inside the hold window with nothing else
    against the grasp) or the episode could not be measured (missing or
    unreadable streams, a task without a Z_MAX)."""
    from phantom.eval.grasp_label import reason_key
    if lab.grasp_ok:
        return ORDINAL["lift"]
    if lab.inconclusive:
        return None
    keys = {reason_key(r) for r in lab.reasons}
    if keys & _UNMEASURABLE:
        return None
    if keys and keys <= _LIFT_ONLY:
        return ORDINAL["close"]
    return ORDINAL["reach"]


def episode_outcome_rule(ep_dir: Path, meta: dict, hw) -> tuple[int | None, dict]:
    """(ordinal, info). Stages 0-2 from the tactile rule on the recorded
    streams; 3 only from the operator. Unlabeled episodes stay None (the
    protocol requires a verdict at every prompt; a missing one is a gap, not a
    stage). `info` records the source and any rule/operator disagreement."""
    from phantom.eval.grasp_label import label_episode
    op = episode_outcome(meta)
    info: dict = {"source": "operator"}
    if op is None:
        return None, info
    try:
        lab = label_episode(ep_dir, hw, task=meta.get("task"))
    except Exception as e:  # noqa: BLE001 — a broken recording is reported, not fatal
        info.update(source="operator", rule_error=f"{type(e).__name__}: {e}")
        return op, info
    stage = rule_stage(lab)
    info.update(lift_mm=lab.lift_mm, z_close_mm=lab.z_close_mm, hold_s=lab.hold_s,
                c_hold=lab.c_hold, stall=lab.stall, n_close_attempts=lab.n_close_attempts,
                rule_reasons=list(lab.reasons))
    if op == ORDINAL["placed"]:
        info["source"] = "operator"
        if stage is not None and stage < ORDINAL["lift"]:
            # the human saw a placement the rule cannot see a held lift for
            info["conflict"] = f"operator placed, rule stage {stage}"
        return op, info
    if stage is None:
        info.update(source="operator", rule_inconclusive=True)
        return op, info
    info["source"] = "rule"
    if op != stage:
        info["operator_stage"] = op
    return stage, info


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


ARM_TAG_PREFIXES = ("label:", "ckpt_sha:", "ckpt:")


def episode_arm(meta: dict) -> str | None:
    """The `ckpt:<basename>` tag (legacy identity of an arm)."""
    for t in meta.get("tags", []) or []:
        if isinstance(t, str) and t.startswith("ckpt:"):
            return t
    return None


def episode_arm_tags(meta: dict) -> set[str]:
    """Every tag that can name an arm: `label:<menu row>` (PICK.sh, 09-12),
    `ckpt_sha:<12 hex>`, `ckpt:<basename>`. An --arm-a/--arm-b spec matches an
    episode when it is one of these. Prefer label: or ckpt_sha: — the three
    LeRobot rows (pi0.5, diffusion, x-vla) all load a directory named
    `pretrained_model`, and two students are both `student_002000.pt`."""
    return {t for t in (meta.get("tags", []) or [])
            if isinstance(t, str) and t.startswith(ARM_TAG_PREFIXES)}


def episode_stop(meta: dict) -> str | None:
    """Stop reason from the `stop:<reason>` tag run_deploy files at episode end."""
    for t in meta.get("tags", []) or []:
        if isinstance(t, str) and t.startswith("stop:"):
            return t[5:] or None
    d = meta.get("deploy_overrides")
    return d.get("stop_reason") if isinstance(d, dict) else None


def load_episodes(root: Path, hw=None) -> list[dict]:
    """Every ep_*/meta.json under root, with derived seed / arm / outcome.
    With `hw` (a loaded hardware config) stages 0-2 come from the tactile rule."""
    out = []
    for mp in sorted(root.rglob("meta.json")):
        if not mp.parent.name.startswith("ep_"):
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if hw is not None:
            outcome, info = episode_outcome_rule(mp.parent, meta, hw)
        else:
            outcome, info = episode_outcome(meta), {"source": "operator"}
        out.append({"path": str(mp.parent), "task": meta.get("task"),
                    "seed": episode_seed(meta), "arm": episode_arm(meta),
                    "arm_tags": episode_arm_tags(meta),
                    "outcome": outcome, "outcome_info": info,
                    "stop": episode_stop(meta)})
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
    if arm_a == arm_b:
        raise ValueError("arm A and arm B are the same spec")
    for e in eps:
        if task and e["task"] != task:
            continue
        tags = e.get("arm_tags") or ({e["arm"]} if e.get("arm") else set())
        arm = arm_a if arm_a in tags else (arm_b if arm_b in tags else None)
        if arm is None or e["seed"] is None:
            continue
        if e["outcome"] is None:
            unlabeled += 1
            continue
        by_key[(e["task"], e["seed"])][arm].append(e)
    pairs, unpaired = [], []
    for key in sorted(by_key, key=lambda k: (str(k[0]), k[1])):
        arms = by_key[key]
        if arm_a in arms and arm_b in arms:
            a, b = arms[arm_a][-1], arms[arm_b][-1]
            pairs.append({"task": key[0], "seed": key[1], "a": a["outcome"], "b": b["outcome"],
                          "a_path": a["path"], "b_path": b["path"],
                          "a_source": a.get("outcome_info", {}).get("source"),
                          "b_source": b.get("outcome_info", {}).get("source"),
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
            "b_grasp_or_better": np.mean([p["b"] >= 1 for p in pairs]) if pairs else float("nan"),
            "a_lift_or_better": np.mean([p["a"] >= 2 for p in pairs]) if pairs else float("nan"),
            "b_lift_or_better": np.mean([p["b"] >= 2 for p in pairs]) if pairs else float("nan")}


# ------------------------------------------------------------ offline boots
def window_key(r: dict) -> tuple:
    """Identity of one scored window: (episode, t0, seed). Rows lacking any part
    get an order-free key of None and are never joined across files."""
    if r.get("episode") is None or r.get("t0") is None or r.get("seed") is None:
        return (None,)
    return (str(r["episode"]), round(float(r["t0"]), 4), int(r["seed"]))


def _rows(path: Path, metric: str) -> list[tuple[tuple, str | None, float]]:
    """[(key, episode, value)] for every row with a finite metric."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for r in d.get("rows", []):
        v = r.get(metric)
        if v is None:
            continue
        v = float(v)
        if np.isfinite(v):
            out.append((window_key(r), r.get("episode"), v))
    return out


def _rows_errors(path: Path, metric: str = "endpoint_err_mm") -> np.ndarray:
    return np.asarray([v for _, _, v in _rows(path, metric)], dtype=float)


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 5000, seed: int = 0,
                 alpha: float = 0.05, clusters=None) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap. With `clusters` (one label per
    value, e.g. the episode) whole clusters are resampled with replacement —
    the windows of one episode share its scene and trajectory and are not
    independent draws. Without clusters every value is its own cluster."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, x.size, size=(n_boot, x.size))
        boots = np.apply_along_axis(stat, 1, x[idx])
    else:
        labels = np.asarray(clusters)
        if labels.shape != x.shape:
            raise ValueError("clusters must have one label per value")
        uniq, inv = np.unique(labels, return_inverse=True)
        members = [np.flatnonzero(inv == c) for c in range(uniq.size)]
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, uniq.size, size=uniq.size)
            boots[b] = stat(x[np.concatenate([members[c] for c in pick])])
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def offline_summary(path: Path, metric: str = "endpoint_err_mm", n_boot: int = 5000) -> dict:
    rows = _rows(path, metric)
    x = np.asarray([v for _, _, v in rows], dtype=float)
    eps = [e if e is not None else i for i, (_, e, _) in enumerate(rows)]
    m, mlo, mhi = bootstrap_ci(x, np.mean, n_boot, clusters=eps)
    md, mdlo, mdhi = bootstrap_ci(x, np.median, n_boot, clusters=eps)
    return {"file": str(path), "n_windows": int(x.size),
            "n_episodes": len({e for e in eps}), "mean": m, "mean_ci": [mlo, mhi],
            "median": md, "median_ci": [mdlo, mdhi]}


def paired_offline_diff(path_a: Path, path_b: Path, metric: str = "endpoint_err_mm",
                        n_boot: int = 5000, seed: int = 0) -> dict:
    """B − A on the same windows, joined on (episode, t0, seed) — never by row
    order. Windows present in only one file are counted, not paired. The CI
    resamples episodes. Negative = B better."""
    ra = {k: (e, v) for k, e, v in _rows(path_a, metric) if k != (None,)}
    rb = {k: (e, v) for k, e, v in _rows(path_b, metric) if k != (None,)}
    keys = sorted(set(ra) & set(rb), key=str)
    n = len(keys)
    out = {"n": n, "n_only_a": len(set(ra) - set(rb)), "n_only_b": len(set(rb) - set(ra))}
    if n == 0:
        return out
    d = np.asarray([rb[k][1] - ra[k][1] for k in keys], dtype=float)
    eps = [ra[k][0] for k in keys]
    dm, lo, hi = bootstrap_ci(d, np.mean, n_boot, seed, clusters=eps)
    dmd, mlo, mhi = bootstrap_ci(d, np.median, n_boot, seed, clusters=eps)
    out.update({"n_episodes": len(set(eps)), "mean_diff": dm, "mean_diff_ci": [lo, hi],
                "median_diff": dmd, "median_diff_ci": [mlo, mhi],
                "frac_b_better": float(np.mean(d < 0))})
    return out


# ---------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs", help="paired sign test over rig episodes")
    p.add_argument("root", type=Path, nargs="+",
                   help="deploy day folder(s); several folders join across days by (task, seed) "
                        "— cell seeds repeat across days, so only give the days that ran the SAME cells")
    p.add_argument("--arm-a", required=True, help="arm A tag: label:<menu row> (preferred), ckpt_sha:<12hex> or ckpt:<basename>")
    p.add_argument("--arm-b", required=True, help="arm B tag, e.g. label:stu_ftA_r2")
    p.add_argument("--task", default=None)
    p.add_argument("--hw", type=Path, default=None,
                   help="hardware yaml: stages 0-2 from the tactile rule on the recorded streams "
                        "(omit = operator notes only)")
    p.add_argument("--json-out", type=Path, default=None)
    o = sub.add_parser("offline", help="bootstrap CIs for terminal_eval json files (2 files = paired diff)")
    o.add_argument("files", nargs="+", type=Path)
    o.add_argument("--metric", default="endpoint_err_mm")
    args = ap.parse_args(argv)

    if args.cmd == "pairs":
        hw = None
        if args.hw is not None:
            from phantom.config.hardware import load_hardware
            hw = load_hardware(args.hw)
        eps = [e for r in args.root for e in load_episodes(r, hw)]
        pairs, diag = pair_episodes(eps, args.arm_a, args.arm_b, args.task)
        res = {"arm_a": args.arm_a, "arm_b": args.arm_b, "task": args.task,
               "ordinal": "rule (0-2) + operator (3)" if hw is not None else "operator notes",
               "n_episodes_seen": len(eps), **diag, "pairs": pairs,
               "sign_test": sign_test(pairs), "outcomes": outcome_table(pairs)}
        srcs = [e["outcome_info"] for e in eps if e["outcome"] is not None]
        res["rule_conflicts"] = [{"path": e["path"], **{k: e["outcome_info"][k] for k in ("conflict",)}}
                                 for e in eps if "conflict" in e["outcome_info"]]
        res["rule_inconclusive"] = sum(1 for s in srcs if s.get("rule_inconclusive"))
        res["rule_errors"] = sum(1 for s in srcs if s.get("rule_error"))
        st = res["sign_test"]
        print(f"ordinal: {res['ordinal']}")
        print(f"pairs: {st['n_pairs']}  B>A: {st['b_wins']}  A>B: {st['a_wins']}  ties: {st['ties']}  "
              f"p(two-sided sign) = {st['p_two_sided']:.3f}")
        print("A outcomes:", res["outcomes"]["a"], " B outcomes:", res["outcomes"]["b"])
        if diag["unlabeled_skipped"]:
            print(f"unlabeled (skipped): {diag['unlabeled_skipped']}")
        if diag["unpaired"]:
            per_arm = defaultdict(int)
            for u in diag["unpaired"]:
                per_arm[",".join(u["have"])] += 1
            print(f"unpaired cells: {len(diag['unpaired'])} "
                  f"({'; '.join(f'{k}: {v}' for k, v in sorted(per_arm.items()))}) — full list in --json-out")
        if hw is not None:
            print(f"rule: inconclusive (operator fallback) {res['rule_inconclusive']}, "
                  f"unreadable {res['rule_errors']}, operator-placed-but-rule-lower {len(res['rule_conflicts'])}")
        if args.json_out:
            args.json_out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        return 0
    for f in args.files:
        s = offline_summary(f, args.metric)
        print(f"{f.name}: n={s['n_windows']} ({s['n_episodes']} episodes) "
              f"mean {s['mean']:.2f} [{s['mean_ci'][0]:.2f}, {s['mean_ci'][1]:.2f}]  "
              f"median {s['median']:.2f} [{s['median_ci'][0]:.2f}, {s['median_ci'][1]:.2f}]")
    if len(args.files) == 2:
        d = paired_offline_diff(args.files[0], args.files[1], args.metric)
        if d["n"] == 0:
            print(f"paired B−A: no shared (episode, t0, seed) windows "
                  f"(only in A: {d['n_only_a']}, only in B: {d['n_only_b']})")
            return 1
        print(f"paired B−A on {d['n']} shared windows ({d['n_episodes']} episodes; "
              f"unmatched A {d['n_only_a']}, B {d['n_only_b']}): "
              f"mean {d['mean_diff']:.2f} [{d['mean_diff_ci'][0]:.2f}, {d['mean_diff_ci'][1]:.2f}]  "
              f"median {d['median_diff']:.2f} [{d['median_diff_ci'][0]:.2f}, {d['median_diff_ci'][1]:.2f}]  "
              f"B better on {d['frac_b_better']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
