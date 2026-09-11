"""Apply (or fix) an operator verdict on already-recorded rig episodes.

The deploy prompt is the normal path; this is the rescue for episodes that
ended without an answer (status 'aborted' + tag 'unlabeled' — they are
dropped from every pair). Same tokens as the live prompt: s / f / c, with a
trailing d for damage, then free notes ("f grasp", "f lift" feed the ordinal
outcome in phantom.eval.stats).

    python tools/verdict.py data/episodes/deploy/20260911/ep_student_Carton_1789125209_000 f grasp
    python tools/verdict.py --root data/episodes/deploy/20260911 --seed 104 --arm ckpt:student_002000.pt s
    python tools/verdict.py --root data/episodes/deploy/20260911 --list          # who is unlabeled

Only meta.json is rewritten (EpisodeRecorder.relabel); recorded data is never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phantom.recording.recorder import EpisodeRecorder
from phantom.scripts.run_deploy import label_episode


class _MetaOnlyRecorder:
    """Recorder.relabel needs no live session; route the verdict through it."""

    def relabel(self, path: Path, **kw) -> None:
        EpisodeRecorder.relabel(self, path, **kw)  # type: ignore[arg-type]


def _episodes(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("meta.json") if p.parent.name.startswith("ep_"))


def _summary(ep: Path) -> str:
    m = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    tags = m.get("tags", []) or []
    seed = next((t for t in tags if str(t).startswith("seed:")), "seed:?")
    arm = next((t for t in tags if str(t).startswith("ckpt:")), "ckpt:?")
    return (f"{ep.name}  {m.get('task')}  {seed}  {arm}  status={m.get('status')}  "
            f"success={m.get('success')}  notes={m.get('notes')!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*",
                    help="[episode dirs...] verdict [notes...]  — verdict = s | f | c (+d for damage); "
                         "with --root the dirs are selected by --seed/--arm/--task instead")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arm", default=None, help="ckpt tag, e.g. ckpt:student_002000.pt")
    ap.add_argument("--task", default=None)
    ap.add_argument("--list", action="store_true", help="list unlabeled episodes under --root and exit")
    ap.add_argument("--force", action="store_true", help="overwrite an existing verdict")
    args = ap.parse_args(argv)
    # positionals: leading existing episode dirs, then the verdict token, then notes
    words = list(args.words)
    args.episodes = []
    while words and Path(words[0]).is_dir():
        args.episodes.append(words.pop(0))
    args.verdict = words[0] if words else None
    args.notes = words[1:]

    targets: list[Path] = []
    if args.root is not None:
        eps = _episodes(args.root)
        if args.list:
            for ep in eps:
                m = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
                if m.get("success") is None:
                    print(_summary(ep))
            return 0
        for ep in eps:
            m = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
            tags = [str(t) for t in m.get("tags", []) or []]
            if args.seed is not None and f"seed:{args.seed}" not in tags:
                continue
            if args.arm is not None and args.arm not in tags:
                continue
            if args.task is not None and m.get("task") != args.task:
                continue
            targets.append(ep)
    targets += [Path(e) for e in args.episodes]
    if not targets:
        print("no episodes matched", file=sys.stderr)
        return 1
    if not args.verdict:
        print("verdict required (s / f / c, +d)", file=sys.stderr)
        return 1
    ans = " ".join([args.verdict] + list(args.notes)).strip()
    rec = _MetaOnlyRecorder()
    rc = 0
    for ep in targets:
        m = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
        if m.get("success") is not None and not args.force:
            print(f"SKIP (already labeled, use --force): {_summary(ep)}")
            rc = 2
            continue
        print("before:", _summary(ep))
        code = label_episode(rec, ep, ans)
        print(f"after : {_summary(ep)}   [verdict {code!r}]")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
