"""Write the P8 grasp-rule verdicts into episode meta (`success`) so rollouts
become trainable (schema.is_trainable_episode refuses a policy rollout with
`success is None` — a rollout nobody judged is exactly the on-policy state
where the model is already wrong).

    python tools/label_grasps.py data/episodes/deploy/20260904 --include-unfinalized --json-out labels.json
    python tools/apply_grasp_labels.py labels.json data/episodes/deploy/20260904 --dry-run
    python tools/apply_grasp_labels.py labels.json data/episodes/deploy/20260904

Rules (phantom/eval/grasp_label.py): `grasp_ok` is the tactile rule — contact
sustained through a >= 50 mm lift, hold >= 2 s, c_hold >= 0.8. A
`hold_truncated` / `inconclusive` episode is ABSTAINED (left as is), never
coerced to False. An operator verdict already in meta wins over the rule
unless --overwrite. Every write is recorded as tags `label:rule` and
`rule:grasp_ok` / `rule:fail` so the origin of the label stays auditable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phantom.data.schema import EpisodeMeta


def apply(labels_path: Path, roots: list[Path], *, dry_run: bool, overwrite: bool) -> dict:
    rows = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    by_name = {r["episode"]: r for r in rows}
    stats = {"written": 0, "abstained": 0, "kept_operator": 0, "missing": 0, "unchanged": 0}
    for root in roots:
        for meta_path in sorted(Path(root).rglob("meta.json")):
            ep = meta_path.parent
            r = by_name.get(ep.name)
            if r is None:
                stats["missing"] += 1
                continue
            if r.get("hold_truncated") or r.get("inconclusive"):
                stats["abstained"] += 1
                print(f"abstain  {ep.name}: {'hold_truncated' if r.get('hold_truncated') else 'inconclusive'}")
                continue
            meta = EpisodeMeta.load(meta_path)
            rule_labelled = "label:rule" in (meta.tags or [])
            if meta.success is not None and not rule_labelled and not overwrite:
                stats["kept_operator"] += 1        # a HUMAN verdict wins over the rule
                continue
            verdict = bool(r["grasp_ok"])
            # mirror recorder.relabel: a judged rollout is FINALIZED and no
            # longer `unlabeled` — is_trainable_episode refuses either
            tags = [t for t in (meta.tags or [])
                    if not t.startswith("rule:") and t not in ("label:rule", "unlabeled")]
            tags += ["label:rule", "rule:grasp_ok" if verdict else "rule:fail"]
            if (meta.success == verdict and set(tags) == set(meta.tags or [])
                    and meta.status == "finalized"):
                stats["unchanged"] += 1
                continue
            print(f"{'DRY ' if dry_run else ''}write    {ep.name}: success={verdict} "
                  f"(c_hold {r.get('c_hold')}, lift {r.get('lift_mm')} mm)")
            if not dry_run:
                meta.success = verdict
                meta.tags = tags
                meta.status = "finalized"
                meta.save(meta_path)
            stats["written"] += 1
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing operator verdict with the rule's")
    a = ap.parse_args(argv)
    st = apply(a.labels, a.roots, dry_run=a.dry_run, overwrite=a.overwrite)
    print(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
