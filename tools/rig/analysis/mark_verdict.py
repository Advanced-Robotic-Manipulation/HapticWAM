"""Re-file a take's verdict after the fact on the operator's word.
usage: mark_verdict.py <episode dir name> <s|f|c> "<note>" [tag_to_add ...]"""
import json
import os
import sys
from pathlib import Path

from phantom.recording.recorder import EpisodeRecorder

D = Path(os.path.expanduser("~/phantom-icra-2027/data/episodes/deploy/20260915_experiment"))
ep, v, note = D / sys.argv[1], sys.argv[2], sys.argv[3]
extra = sys.argv[4:]
rec = EpisodeRecorder.__new__(EpisodeRecorder)
rec.out_root = D
if v == "s":
    rec.relabel(ep, success=True, status="finalized", remove_tags=["unlabeled", "crushed"], tags=extra, notes=note)
elif v == "c":
    rec.relabel(ep, success=True, status="finalized", remove_tags=["unlabeled"], tags=["crushed"] + extra, notes=note)
else:
    rec.relabel(ep, success=False, status="finalized", remove_tags=["unlabeled", "crushed"], tags=extra, notes=note)
m = json.load(open(ep / "meta.json"))
print(ep.name, m["status"], "success=", m["success"], "tags=", [t for t in m["tags"] if not t.startswith(("nfe", "g1", "pnoise", "ckpt", "git", "seed", "zfloor", "hitbox", "vmax", "descent", "parity", "veto", "kseeds", "sel:", "aveto", "ashadow", "label:"))], m["notes"])
