"""v5 intake fixes: close detector fallback, symlink-aware episode listing,
intake tagging semantics."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from phantom.data.episode_store import list_episodes
from phantom.train.common import close_index

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import intake_recovery as IR  # noqa: E402


def _ramp(start, end, n):
    return np.linspace(start, end, n)


def test_close_index_primary_rule_unchanged():
    # open 0.30 -> closes to 0.55 : the historical rule fires at the 0.45 crossing
    pos = np.concatenate([np.full(20, 0.30), _ramp(0.30, 0.55, 26), np.full(20, 0.55)])
    i = close_index(pos)
    old = np.nonzero((pos > 0.45) & (pos - np.minimum.accumulate(pos) > 0.15))[0][0]
    assert i == old


def test_close_index_fallback_catches_wide_grasp():
    # Carton-like: closes only to 0.42 -> primary never fires, fallback must
    pos = np.concatenate([np.full(20, 0.28), _ramp(0.28, 0.42, 14), np.full(30, 0.42)])
    assert not np.any(pos > 0.45)
    i = close_index(pos)
    assert i is not None
    assert pos[i] >= 0.42 - 0.05 and pos[i] - 0.28 > 0.10
    assert 20 <= i < 34                       # on the closing ramp, not the plateau


def test_close_index_fallback_ignores_small_aperture_adjustment():
    # egg-like: small adjustment 0.20->0.30, then the real grasp 0.30->0.44
    pos = np.concatenate([np.full(10, 0.20), _ramp(0.20, 0.30, 10), np.full(20, 0.30),
                          _ramp(0.30, 0.44, 14), np.full(20, 0.44)])
    i = close_index(pos)
    assert i >= 40, "fallback must anchor on the tightest closure, not the adjustment"


def test_close_index_never_closes():
    assert close_index(np.full(50, 0.25)) is None
    assert close_index(np.array([])) is None


def _fake_episode(d: Path, task: str = "waffles", **meta):
    d.mkdir(parents=True)
    m = {"episode_id": d.name, "task": task, "success": True, "tags": ["full"],
         "finalized": True}
    m.update(meta)
    (d / "meta.json").write_text(json.dumps(m))


def test_list_episodes_follows_symlinked_episode_dirs(tmp_path):
    real = tmp_path / "archive" / "20260822_130412_waffles" / "ep_waffles_1_000"
    _fake_episode(real)
    tasks = tmp_path / "tasks" / "waffles"
    tasks.mkdir(parents=True)
    os.symlink(real, tasks / real.name)
    found = list_episodes(tmp_path / "tasks", include_unfinalized=True)
    assert [p.name for p in found] == ["ep_waffles_1_000"]


def test_intake_normalize_tags_and_failure_encoding(tmp_path):
    ok = tmp_path / "20260822_130412_waffles" / "ep_waffles_1_000"
    _fake_episode(ok, task="waffles", tags=["full", "recovery"])
    bad = tmp_path / "20260822_151116_carton_fail_undergrasp" / "ep_carton_fail_undergrasp_2_000"
    _fake_episode(bad, task="carton_fail_undergrasp", success=True)
    IR.normalize(tmp_path)
    m_ok = json.loads((ok / "meta.json").read_text())
    m_bad = json.loads((bad / "meta.json").read_text())
    assert m_ok["task"] == "waffles" and m_ok["success"] is True
    assert m_ok["tags"] == ["full", "batch_20260822"]         # 'recovery' removed
    assert m_bad["task"] == "Carton_fail" and m_bad["text"] == "Carton_fail"
    assert m_bad["success"] is False and m_bad["failure_demo"] is True
    assert set(m_bad["tags"]) >= {"deliberate_failure", "undergrasp", "batch_20260822"}
    assert IR.batch_tag("20260822_130412_waffles") == "batch_20260822"


def test_intake_manifest_appends_only_new_rows_as_train(tmp_path):
    tasks = tmp_path / "tasks"
    old = tasks / "waffles" / "ep_old"
    new = tasks / "waffles" / "ep_new"
    _fake_episode(old)
    _fake_episode(new, tags=["full", "batch_20260822"])
    mf = tmp_path / "manifests" / "all.jsonl"
    mf.parent.mkdir()
    mf.write_text(json.dumps({"episode": "ep_old", "task": "waffles", "split": "val",
                              "path": "tasks/waffles/ep_old"}) + "\n")
    IR.manifest(tasks, mf)
    rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    assert [r["episode"] for r in rows] == ["ep_old", "ep_new"]
    assert rows[0]["split"] == "val"                          # frozen
    assert rows[1]["split"] == "train" and rows[1]["path"] == "tasks/waffles/ep_new"
