#!/usr/bin/env python3
"""Census of all PHANTOM episodes across the NUC roots.

Read-only. Groups episodes by task (from meta.json), reports success/failure
counts per task and per session. Dedups by (session, episode) name across
roots -- the external drive is canonical, local dirs are live staging.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOTS = [
    Path("/media/nuc/kostya_drive/phantom_episodes"),   # canonical, first wins
    Path.home() / "phantom-data" / "collect",
    Path.home() / "phantom-data" / "episodes",
]

seen = set()           # (session_name, ep_name)
by_task = defaultdict(lambda: {"eps": 0, "T": 0, "F": 0, "None": 0,
                               "status": Counter(), "sessions": set()})
by_session = defaultdict(lambda: Counter())
sess_task = {}
unreadable = []

for root in ROOTS:
    if not root.is_dir():
        continue
    for sess in sorted(root.iterdir()):
        if not sess.is_dir():
            continue
        for ep in sorted(sess.glob("ep_*")):
            key = (sess.name, ep.name)
            if key in seen:
                continue
            seen.add(key)
            meta_p = ep / "meta.json"
            try:
                meta = json.loads(meta_p.read_text())
            except Exception as e:  # noqa: BLE001
                unreadable.append(f"{sess.name}/{ep.name}: {e}")
                continue
            task = meta.get("task") or sess.name.split("_", 2)[-1]
            t = by_task[task]
            t["eps"] += 1
            t["sessions"].add(sess.name)
            t["status"][str(meta.get("status"))] += 1
            s = meta.get("success")
            bucket = "T" if s is True else ("F" if s is False else "None")
            t[bucket] += 1
            by_session[sess.name][bucket] += 1
            by_session[sess.name]["eps"] += 1
            sess_task[sess.name] = task

print(f"{'TASK':<16} {'eps':>4} {'succ':>5} {'fail':>5} {'unset':>6}  statuses")
print("-" * 70)
tot = Counter()
for task, t in sorted(by_task.items(), key=lambda kv: -kv[1]["eps"]):
    st = ",".join(f"{k}:{v}" for k, v in t["status"].items())
    print(f"{task:<16} {t['eps']:>4} {t['T']:>5} {t['F']:>5} {t['None']:>6}  {st}"
          f"  ({len(t['sessions'])} sessions)")
    for k in ("eps", "T", "F", "None"):
        tot[k] += t[k]
print("-" * 70)
print(f"{'TOTAL':<16} {tot['eps']:>4} {tot['T']:>5} {tot['F']:>5} {tot['None']:>6}")

print("\nPer-session (chronological):")
for sess in sorted(by_session):
    c = by_session[sess]
    flag = "" if c["F"] == 0 and c["None"] == 0 else "   <-- has fail/unset"
    print(f"  {sess:<34} task={sess_task[sess]:<14} n={c['eps']:>3} "
          f"T={c['T']:>3} F={c['F']:>2} ?={c['None']:>2}{flag}")

if unreadable:
    print("\nUNREADABLE meta.json:")
    for u in unreadable:
        print("  ", u)
