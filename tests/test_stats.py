"""phantom.eval.stats — pairing by (task, seed), ordinal outcomes, exact sign test, bootstraps."""
import json
import math
from pathlib import Path

import numpy as np

from phantom.eval import stats as S


def _ep(root: Path, name: str, *, task, seed, ckpt, status="finalized", success=False, notes=""):
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "task": task, "status": status, "success": success, "notes": notes,
        "tags": ["nfe1", f"seed:{seed}", f"ckpt:{ckpt}", "start:-1,-2,3mm/g0.1"],
        "deploy_overrides": {"max_play_steps": 16}}))
    return d


def test_outcome_is_ordinal_and_unlabeled_is_none():
    assert S.episode_outcome({"status": "finalized", "success": True}) == 3
    assert S.episode_outcome({"status": "finalized", "success": False, "notes": "operator: f lift"}) == 2
    assert S.episode_outcome({"status": "finalized", "success": False, "notes": "operator: f grasp"}) == 1
    assert S.episode_outcome({"status": "finalized", "success": False, "notes": "operator: f"}) == 0
    assert S.episode_outcome({"status": "aborted", "success": None}) is None


def test_pairs_by_task_and_seed_last_rerun_wins(tmp_path):
    A, B = "ckpt:BEST.pt", "ckpt:student_002000.pt"
    _ep(tmp_path, "ep_teacher_waffles_1_000", task="waffles", seed=101, ckpt="BEST.pt", success=True)
    _ep(tmp_path, "ep_student_waffles_2_000", task="waffles", seed=101, ckpt="student_002000.pt",
        success=False, notes="operator: f grasp")
    # seed 102: teacher labeled, student unlabeled -> unpaired
    _ep(tmp_path, "ep_teacher_waffles_3_000", task="waffles", seed=102, ckpt="BEST.pt", success=False)
    _ep(tmp_path, "ep_student_waffles_4_000", task="waffles", seed=102, ckpt="student_002000.pt",
        status="aborted", success=None)
    # seed 103: student run twice (censored re-run) -> the last one counts
    _ep(tmp_path, "ep_teacher_waffles_5_000", task="waffles", seed=103, ckpt="BEST.pt", success=False)
    _ep(tmp_path, "ep_student_waffles_6_000", task="waffles", seed=103, ckpt="student_002000.pt", success=False)
    _ep(tmp_path, "ep_student_waffles_7_000", task="waffles", seed=103, ckpt="student_002000.pt", success=True)
    # a different task with the same seed must not pair with waffles
    _ep(tmp_path, "ep_teacher_egg_8_000", task="egg", seed=101, ckpt="BEST.pt", success=True)

    eps = S.load_episodes(tmp_path)
    pairs, diag = S.pair_episodes(eps, A, B, task="waffles")
    assert [(p["seed"], p["a"], p["b"]) for p in pairs] == [(101, 3, 1), (103, 0, 3)]
    assert diag["unlabeled_skipped"] == 1
    assert [(u["seed"]) for u in diag["unpaired"]] == [102]
    assert pairs[1]["reruns"] == 1
    st = S.sign_test(pairs)
    assert (st["a_wins"], st["b_wins"], st["ties"]) == (1, 1, 0)
    assert st["p_two_sided"] == 1.0


def test_sign_test_exact_binomial():
    # 9 B-wins, 1 A-win, 2 ties -> two-sided p = 2 * P(X <= 1 | n=10, 0.5) = 22/1024
    pairs = [{"a": 0, "b": 3}] * 9 + [{"a": 3, "b": 0}] + [{"a": 1, "b": 1}] * 2
    st = S.sign_test(pairs)
    assert st["n_pairs"] == 12 and st["ties"] == 2
    assert math.isclose(st["p_two_sided"], 22 / 1024)
    assert S.sign_test([])["p_two_sided"] == 1.0


def test_offline_bootstrap_and_paired_diff(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(20, 5, size=200).clip(1)
    b = a - 2.0 + rng.normal(0, 0.5, size=200)
    fa, fb = tmp_path / "a.json", tmp_path / "b.json"
    fa.write_text(json.dumps({"summary": {}, "rows": [{"endpoint_err_mm": float(v)} for v in a]}))
    fb.write_text(json.dumps({"summary": {}, "rows": [{"endpoint_err_mm": float(v)} for v in b]}))
    sa = S.offline_summary(fa, n_boot=500)
    assert sa["n_windows"] == 200 and sa["mean_ci"][0] < sa["mean"] < sa["mean_ci"][1]
    d = S.paired_offline_diff(fa, fb, n_boot=500)
    assert d["n"] == 200 and d["mean_diff_ci"][1] < 0 and d["frac_b_better"] > 0.95
    # NaN / missing rows are ignored, not crashed on
    fa.write_text(json.dumps({"rows": [{"endpoint_err_mm": 1.0}, {"endpoint_err_mm": None}, {"other": 2}]}))
    assert S.offline_summary(fa, n_boot=10)["n_windows"] == 1
