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
- every intake episode gets a provenance tag `batch_<YYYYMMDD>` derived from
  its session directory (e.g. `batch_20260822`). Successful episodes are
  ordinary positive demos — appended, weight 1, nothing else. (An earlier
  revision tagged them `recovery`; QC showed their start poses are inside
  the demo distribution, so the tag was wrong and is removed here.)
- `<Task>_fail` episodes get success=False: the manipulation failed. The
  v4 `_fail` rows carry success=True ("episode captured the failure") —
  training treats both identically (is_failure_demo short-circuits on any
  of the three encodings); only student-side tau calibration reads success.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

CANON = {"carton": "Carton", "waffles": "waffles", "egg": "egg", "whiteboard": "whiteboard"}


def batch_tag(session_dir_name: str) -> str:
    """`20260822_130412_waffles` -> `batch_20260822` (provenance, training-inert)."""
    head = session_dir_name.split("_", 1)[0]
    return f"batch_{head}" if head.isdigit() and len(head) == 8 else f"batch_{session_dir_name}"


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
        tags = [t for t in (m.get("tags") or []) if t != "recovery"]
        extra.append(batch_tag(mp.parent.parent.name))
        new_tags = tags + [t for t in extra if t not in tags]
        new = dict(m)
        new["task"] = task
        if m.get("text") in (raw, "", None):
            new["text"] = task
        new["tags"] = new_tags
        if task.endswith("_fail"):
            new["failure_demo"] = True
            new["success"] = False        # the manipulation failed (by design)
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
            if m.get("status", "finalized") != "finalized":
                print(f"manifest: skipping {ep.name} (status={m.get('status')!r})")
                continue                       # crashed/in-flight/aborted takes never train
            # anything not already in the manifest is a new intake episode
            # (v4 rows are all present; val stays frozen — new rows -> train)
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
