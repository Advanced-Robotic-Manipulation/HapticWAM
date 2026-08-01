#!/usr/bin/env python3
"""Finalize episodes stranded with meta status "recording" after a crash.
Streams are append-only zarr, so all data up to the crash is intact -- only the
meta.json finalization is missing. Marks them status=finalized, success=None,
notes=recovered so downstream tooling sees them (operator decides success later).

  python recover_episodes.py <session_dir_or_root> [--dry-run]
"""
import json, sys, time
from pathlib import Path

dry = "--dry-run" in sys.argv
root = Path([a for a in sys.argv[1:] if not a.startswith("--")][0]).expanduser()
n = 0
for meta_p in sorted(root.rglob("meta.json")):
    d = json.loads(meta_p.read_text())
    if d.get("status") != "recording":
        continue
    print(("WOULD RECOVER " if dry else "RECOVER ") + str(meta_p.parent))
    if not dry:
        d["status"] = "finalized"
        stamp = time.strftime("%Y-%m-%d %H:%M")
        prev = d.get("notes") or ""
        d["notes"] = (prev + " | " if prev else "") + "recovered after crash " + stamp
        tmp = meta_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(meta_p)
    n += 1
print(f"{n} episode(s) {'found' if dry else 'recovered'}")
