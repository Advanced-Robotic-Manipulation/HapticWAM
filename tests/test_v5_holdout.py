"""batch_20260822 intake: whole-session val holdout + idempotent re-run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import intake_recovery as IR  # noqa: E402


def _rows(spec):
    out = []
    for task, sess, n in spec:
        for i in range(n):
            out.append({"episode": f"ep_{task}_{sess}_{i:03d}", "task": task, "session": f"20260822_{sess}_{task}",
                        "split": "train", "path": f"tasks/{task}/ep_{task}_{sess}_{i:03d}"})
    return out


def test_holdout_takes_last_whole_sessions_until_min_eps():
    rows = _rows([("waffles", "130000", 10), ("waffles", "135140", 9), ("waffles", "140049", 1),
                  ("Carton", "141637", 10), ("Carton", "150330", 10),
                  ("waffles_fail", "140150", 15)])
    hold = IR.holdout_sessions(rows, 10)
    assert hold == {("waffles", "20260822_140049_waffles"), ("waffles", "20260822_135140_waffles"),
                    ("Carton", "20260822_150330_Carton")}
    assert IR.holdout_sessions(rows, 0) == set()
    assert not any(t.endswith("_fail") for t, _ in hold)


def _ep(root: Path, sess: str, task: str, name: str):
    real = root / "archive" / sess / name
    real.mkdir(parents=True)
    (real / "meta.json").write_text(json.dumps({"episode_id": name, "task": task, "success": True,
                                                "status": "finalized", "tags": ["full", "batch_20260822"]}))
    link = root / "tasks" / task / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)


def test_manifest_holdout_marks_val_and_is_idempotent(tmp_path):
    for sess, n in (("20260822_130000_waffles", 3), ("20260822_140000_waffles", 2)):
        for i in range(n):
            _ep(tmp_path, sess, "waffles", f"ep_waffles_{sess[9:15]}_{i:03d}")
    mf = tmp_path / "manifests" / "all.jsonl"
    mf.parent.mkdir()
    mf.write_text(json.dumps({"episode": "ep_old", "task": "waffles", "split": "val", "path": "tasks/waffles/ep_old"}) + "\n")
    IR.manifest(tmp_path / "tasks", mf, val_min_eps=2)
    rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    val = sorted(r["episode"] for r in rows if r["split"] == "val")
    assert val == ["ep_old", "ep_waffles_140000_000", "ep_waffles_140000_001"]
    info = json.loads((tmp_path / "manifests" / "intake_holdout.json").read_text())
    assert info == {"added": 5, "val": 2, "train": 3, "holdout_sessions": ["waffles/20260822_140000_waffles"]}
    IR.manifest(tmp_path / "tasks", mf, val_min_eps=2)          # re-run: no new rows, record kept
    assert len([l for l in mf.read_text().splitlines() if l.strip()]) == 6
    assert json.loads((tmp_path / "manifests" / "intake_holdout.json").read_text()) == info
