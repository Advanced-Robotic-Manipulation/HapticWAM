"""Label episodes with the P8 tactile grasp rule (rule + operator, dual label).

    python tools/label_grasps.py data/val_eval/tasks/waffles
    python tools/label_grasps.py data/episodes/deploy/20260828 --confusion
    python tools/label_grasps.py <ep_dir> --json-out labels.json

Arguments are episode dirs (containing meta.json) or roots to search for
`ep_*` recursively. Prints one row per episode, a per-task positive rate, the
condition-failure histogram, and — with --confusion — the rule-vs-operator
contingency table. The rule is `phantom/eval/grasp_label.py`; Robotiq OBJ is
reported as an over-squeeze flag only and never enters `grasp_ok`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from phantom.config.hardware import load_hardware
from phantom.data.episode_store import list_episodes
from phantom.eval.grasp_label import (CONTACT_RATE_HZ, MIN_C_HOLD, MIN_HOLD_S,
                                      MIN_LIFT_MM, Z_MAX_MM, confusion,
                                      label_episode, reason_key)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="episode dirs or roots holding ep_*")
    ap.add_argument("--hw", default=None, help="hardware.yaml (default: configs/)")
    ap.add_argument("--task", default=None, help="only episodes with this meta.task")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--confusion", action="store_true",
                    help="print the rule-vs-operator contingency table")
    ap.add_argument("--contact-rate", type=float, default=CONTACT_RATE_HZ,
                    help="Hz at which contact/z are sampled over the hold window")
    ap.add_argument("--min-hold", type=float, default=MIN_HOLD_S)
    ap.add_argument("--min-c-hold", type=float, default=MIN_C_HOLD)
    ap.add_argument("--min-lift", type=float, default=MIN_LIFT_MM)
    ap.add_argument("--z-max", action="append", default=[], metavar="TASK=MM",
                    help="override Z_MAX for a task (repeatable)")
    ap.add_argument("--include-unfinalized", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    hw = load_hardware(args.hw, quiet=True)
    table = dict(Z_MAX_MM)
    for spec in args.z_max:
        k, _, v = spec.partition("=")
        table[k] = float(v)

    eps = collect(args.paths, args.include_unfinalized)
    if args.limit:
        eps = eps[:args.limit]
    if not eps:
        print("no episodes found", file=sys.stderr)
        return 1

    hdr = (f"{'episode':44s} {'task':12s} {'ok':3s} {'op':3s} {'t_cl':>6s} "
           f"{'z_cl':>6s} {'hold':>5s} {'c_hold':>6s} {'lift':>6s} {'stall':>5s}  reasons")
    if not args.quiet:
        print(hdr)
    labels = []
    for ep in eps:
        try:
            lab = label_episode(ep, hw, z_max_table=table,
                                contact_rate_hz=args.contact_rate,
                                min_hold_s=args.min_hold,
                                min_c_hold=args.min_c_hold,
                                min_lift_mm=args.min_lift)
        except Exception as e:                             # noqa: BLE001
            print(f"{ep.name:44s} ERROR {e}", file=sys.stderr)
            continue
        if args.task and lab.task != args.task:
            continue
        labels.append(lab)
        if not args.quiet:
            op = {True: "s", False: "f", None: "-"}[lab.operator_success]
            print(f"{lab.episode:44s} {lab.task:12s} "
                  f"{'OK' if lab.grasp_ok else '..':3s} {op:3s} "
                  f"{(lab.t_close if lab.t_close is not None else float('nan')):6.1f} "
                  f"{(lab.z_close_mm if lab.z_close_mm is not None else float('nan')):6.0f} "
                  f"{lab.hold_s:5.1f} {lab.c_hold:6.2f} {lab.lift_mm:6.0f} "
                  f"{'Y' if lab.stall else '.':5s}  {'; '.join(lab.reasons)}")

    # ---- summary --------------------------------------------------------
    by_task: dict[str, list] = defaultdict(list)
    for l in labels:
        by_task[l.task].append(l)
    print(f"\n== {len(labels)} episodes")
    print(f"{'task':14s} {'n':>5s} {'grasp_ok':>9s} {'rate':>6s} {'stall':>6s} "
          f"{'op_s':>5s} {'op_f':>5s} {'op_-':>5s}")
    for task in sorted(by_task):
        ls = by_task[task]
        ok = sum(l.grasp_ok for l in ls)
        print(f"{task:14s} {len(ls):5d} {ok:9d} {ok / len(ls):6.1%} "
              f"{sum(l.stall for l in ls):6d} "
              f"{sum(l.operator_success is True for l in ls):5d} "
              f"{sum(l.operator_success is False for l in ls):5d} "
              f"{sum(l.operator_success is None for l in ls):5d}")

    cnt = Counter(reason_key(r) for l in labels for r in l.reasons)
    if cnt:
        print("\nfailing conditions (episodes):")
        for k, v in cnt.most_common():
            print(f"  {k:26s} {v:5d}  ({v / len(labels):5.1%})")

    if args.confusion:
        c = confusion(labels)
        print("\nrule vs operator (dual label — the rule has no validated "
              "positive class on a policy grasp):")
        print(f"{'':12s} {'op=s':>6s} {'op=f':>6s} {'op=none':>8s}")
        print(f"{'grasp_ok':12s} {c['ok_s']:6d} {c['ok_f']:6d} {c['ok_none']:8d}")
        print(f"{'not ok':12s} {c['no_s']:6d} {c['no_f']:6d} {c['no_none']:8d}")
        # NOT negatives: the recording ended inside the hold window, so the
        # rule abstains (grasp_label.hold_truncated)
        print(f"{'truncated':12s} {c['trunc_s']:6d} {c['trunc_f']:6d} "
              f"{c['trunc_none']:8d}")
        lab_n = c["ok_s"] + c["ok_f"] + c["no_s"] + c["no_f"]
        if lab_n:
            agree = c["ok_s"] + c["no_f"]
            print(f"  agreement on the {lab_n} operator-labeled episodes: "
                  f"{agree / lab_n:.1%} (truncated episodes excluded — the "
                  f"rule abstains on them)")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([l.to_dict() for l in labels], indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
