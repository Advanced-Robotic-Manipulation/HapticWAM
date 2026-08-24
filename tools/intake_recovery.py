"""Intake for the 2026-08-22 recovery-demo sessions (and future ones).

    python tools/intake_recovery.py normalize <root-with-sessions>   # fix meta.json in place
    python tools/intake_recovery.py place <archive_root> <tasks_root> # symlink eps into tasks/<task>/
    python tools/intake_recovery.py manifest <tasks_root> <manifests/all.jsonl>  # append rows

Why each rule exists:
- task names: sessions were recorded as `carton` / `carton_fail_undergrasp`; the
  dataset (and the text-embedding cache the teacher conditions on) uses
  `Carton` / `Carton_fail`. An unknown text key silently falls back to the
  empty-string embedding.
- `*_fail_undergrasp` -> `<Task>_fail` + tags [deliberate_failure, undergrasp]:
  is_failure_demo() (phantom/data/schema.py) matches success is False, the
  `deliberate_failure` tag, or a task ending in `_fail`. The under-grasp
  episodes continue the task with an EMPTY gripper ("phantom carry") — if
  they were ever imitated they would teach exactly the rig failure mode.
  windows.py sets action_weight=0 for failure demos; contact/event heads
  still learn from their (true) tactile labels.
- successful recovery episodes get the tag `recovery` (start deliberately
  off the demo manifold) so they can be selected/weighted later.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

CANON = {"carton": "Carton", "waffles": "waffles", "egg": "egg", "whiteboard": "whiteboard"}


def canon_task(raw: str) -> tuple[str, list[str]]:
    """raw task -> (canonical task, extra tags)."""
    t = raw.strip()
    tags: list[str] = []
    base = t
    if "_fail" in t:
        base = t.split("_fail")[0]
        tags.append("deliberate_failure")
        suffix = t.split("_fail", 1)[1].strip("_")
        if suffix:
            tags.append(suffix)               # e.g. 'undergrasp'
        return CANON.get(base.lower(), base) + "_fail", tags
    return CANON.get(base.lower(), base), tags


def normalize(root: Path) -> int:
    n = changed = 0
    for mp in sorted(root.rglob("ep_*/meta.json")):
        m = json.loads(mp.read_text())
        n += 1
        raw = m.get("task", "")
        task, extra = canon_task(raw)
        tags = list(m.get("tags") or [])
        if not task.endswith("_fail") and "recovery" not in tags:
            extra.append("recovery")
        new_tags = tags + [t for t in extra if t not in tags]
        new = dict(m)
        new["task"] = task
        if m.get("text") in (raw, "", None):
            new["text"] = task
        new["tags"] = new_tags
        if task.endswith("_fail"):
            # historical encoding: the EPISODE succeeded at capturing the
            # failure; the manipulation did not. Keep success as recorded,
            # the `_fail` task + tag already gate the action loss.
            new["failure_demo"] = True
        if new != m:
            mp.write_text(json.dumps(new, indent=1))
            changed += 1
            print(f"fixed {mp.parent.name}: {raw} -> {task} tags={new_tags}")
    print(f"normalize: {changed}/{n} meta.json changed")
    return 0


def place(archive: Path, tasks_root: Path) -> int:
    n = 0
    for mp in sorted(archive.rglob("ep_*/meta.json")):
        ep = mp.parent
        task = json.loads(mp.read_text())["task"]
        dst = tasks_root / task / ep.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(ep.resolve(), dst)
            n += 1
    print(f"place: {n} episodes linked under {tasks_root}")
    return 0


def manifest(tasks_root: Path, manifest_path: Path) -> int:
    rows = []
    existing = set()
    if manifest_path.exists():
        for l in manifest_path.read_text().splitlines():
            if l.strip():
                existing.add(json.loads(l)["episode"])
    added = 0
    with manifest_path.open("a") as f:
        for mp in sorted(tasks_root.rglob("ep_*/meta.json")):
            ep = mp.parent
            if ep.name in existing:
                continue
            m = json.loads(mp.read_text())
            if "recovery" not in (m.get("tags") or []) and not m["task"].endswith("_fail"):
                continue                       # only new intake episodes
            row = {"episode": ep.name, "task": m["task"], "success": m.get("success"),
                   "failure_demo": bool(m.get("failure_demo")) or m["task"].endswith("_fail"),
                   "tactile_contact": None, "split": "train",   # v3 val stays frozen
                   "session": ep.resolve().parent.name, "operator": m.get("operator", ""),
                   "duration_s": None, "peak_force_N": None, "max_contact_mm2": None,
                   "path": f"tasks/{m['task']}/{ep.name}",
                   "tags": m.get("tags") or []}
            f.write(json.dumps(row) + "\n")
            added += 1
    print(f"manifest: appended {added} rows to {manifest_path}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "normalize":
        raise SystemExit(normalize(Path(sys.argv[2])))
    if cmd == "place":
        raise SystemExit(place(Path(sys.argv[2]), Path(sys.argv[3])))
    if cmd == "manifest":
        raise SystemExit(manifest(Path(sys.argv[2]), Path(sys.argv[3])))
    raise SystemExit(f"unknown command {cmd}")
