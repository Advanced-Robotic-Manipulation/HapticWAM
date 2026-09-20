"""tools/verdict.py — post-hoc operator verdicts rewrite meta.json exactly like the live prompt."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verdict  # noqa: E402


def _ep(root: Path, name: str, *, seed: int, ckpt: str, task="Carton"):
    d = root / name
    d.mkdir(parents=True)
    meta = {"task": task, "status": "aborted", "success": None, "notes": "",
            "tags": ["nfe1", f"seed:{seed}", f"ckpt:{ckpt}", "unlabeled"],
            "operator": "op", "damage": None, "weight": 1.0}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def _meta(d: Path) -> dict:
    return json.loads((d / "meta.json").read_text())


def test_label_by_path_success_with_notes(tmp_path):
    ep = _ep(tmp_path, "ep_student_Carton_1_000", seed=101, ckpt="student_002000.pt")
    assert verdict.main([str(ep), "s"]) == 0
    m = _meta(ep)
    assert m["success"] is True and m["status"] == "finalized"
    assert "unlabeled" not in m["tags"] and m["notes"].startswith("operator: s")


def test_label_by_root_seed_arm_with_ordinal_note(tmp_path):
    a = _ep(tmp_path, "ep_student_Carton_2_000", seed=104, ckpt="student_002000.pt")
    b = _ep(tmp_path, "ep_teacher_Carton_3_000", seed=104, ckpt="BEST.pt")
    c = _ep(tmp_path, "ep_student_Carton_4_000", seed=105, ckpt="student_002000.pt")
    assert verdict.main(["--root", str(tmp_path), "--seed", "104", "--arm", "ckpt:student_002000.pt",
                         "f", "grasp"]) == 0
    assert _meta(a)["success"] is False and "grasp" in _meta(a)["notes"]
    assert _meta(b)["success"] is None and _meta(c)["success"] is None   # untouched


def test_existing_label_is_not_overwritten_without_force(tmp_path, capsys):
    ep = _ep(tmp_path, "ep_teacher_Carton_5_000", seed=101, ckpt="BEST.pt")
    assert verdict.main([str(ep), "f"]) == 0
    assert verdict.main([str(ep), "s"]) == 2
    assert _meta(ep)["success"] is False
    assert verdict.main([str(ep), "s", "--force"]) == 0
    assert _meta(ep)["success"] is True


def test_unrecognized_verdict_leaves_episode_unlabeled(tmp_path):
    ep = _ep(tmp_path, "ep_teacher_Carton_6_000", seed=101, ckpt="BEST.pt")
    assert verdict.main([str(ep), "grasp"]) == 0   # 'g' is not a verdict token
    m = _meta(ep)
    assert m["success"] is None and "unlabeled" in m["tags"]


def test_list_shows_only_unlabeled(tmp_path, capsys):
    a = _ep(tmp_path, "ep_student_Carton_7_000", seed=101, ckpt="student_002000.pt")
    _ep(tmp_path, "ep_student_Carton_8_000", seed=102, ckpt="student_002000.pt")
    verdict.main([str(a), "f"])
    capsys.readouterr()
    assert verdict.main(["--root", str(tmp_path), "--list"]) == 0
    out = capsys.readouterr().out
    assert "ep_student_Carton_8_000" in out and "ep_student_Carton_7_000" not in out
