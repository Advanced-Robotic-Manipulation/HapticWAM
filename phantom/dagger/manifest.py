"""DAgger training manifests: which episodes (demos + relabeled rollouts) make
up round k's training mix. Supports the DAgger-on/off ablation trivially
(manifest with/without rollouts)."""

from __future__ import annotations

import json
from pathlib import Path

from phantom.data.episode_store import list_episodes


def write_manifest(path: Path, *, demos: Path, rollouts: Path | None,
                   round_k: int, demo_weight: float = 1.0,
                   rollout_weight: float = 1.0) -> Path:
    entries = [{"episode": str(p), "weight": demo_weight, "source": "demo"}
               for p in list_episodes(demos)]
    if rollouts is not None:
        entries += [{"episode": str(p), "weight": rollout_weight,
                     "source": f"rollout_r{round_k}"}
                    for p in list_episodes(rollouts)]
    payload = {"round": round_k, "entries": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_manifest(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["entries"]
