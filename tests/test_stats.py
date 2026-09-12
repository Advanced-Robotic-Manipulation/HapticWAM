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


def _rows(vals, seeds=(0, 1), episodes=("e1", "e2", "e3", "e4")):
    rows, i = [], 0
    for v in vals:
        rows.append({"episode": episodes[i % len(episodes)], "t0": 1.0 + (i // len(episodes)) * 0.5,
                     "seed": seeds[i % len(seeds)], "endpoint_err_mm": float(v)})
        i += 1
    return rows


def test_offline_bootstrap_and_paired_diff_joined_on_window_key(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(20, 5, size=200).clip(1)
    b = a - 2.0 + rng.normal(0, 0.5, size=200)
    ra, rb = _rows(a), _rows(b)
    rb_shuffled = [rb[i] for i in rng.permutation(len(rb))]      # order must not matter
    fa, fb = tmp_path / "a.json", tmp_path / "b.json"
    fa.write_text(json.dumps({"summary": {}, "rows": ra}))
    fb.write_text(json.dumps({"summary": {}, "rows": rb_shuffled}))
    sa = S.offline_summary(fa, n_boot=500)
    assert sa["n_windows"] == 200 and sa["n_episodes"] == 4
    assert sa["mean_ci"][0] < sa["mean"] < sa["mean_ci"][1]
    d = S.paired_offline_diff(fa, fb, n_boot=500)
    assert d["n"] == 200 and d["n_only_a"] == 0 and d["n_only_b"] == 0
    assert d["mean_diff_ci"][1] < 0 and d["frac_b_better"] > 0.95
    assert abs(d["mean_diff"] - float(np.mean(b - a))) < 1e-9     # joined, not order-matched
    # a window missing on one side is reported, never paired by position
    fb.write_text(json.dumps({"rows": rb[:-1]}))
    d2 = S.paired_offline_diff(fa, fb, n_boot=50)
    assert d2["n"] == 199 and d2["n_only_a"] == 1 and d2["n_only_b"] == 0
    # rows without a window identity cannot be joined at all
    fa.write_text(json.dumps({"rows": [{"endpoint_err_mm": 1.0}, {"endpoint_err_mm": None}, {"other": 2}]}))
    fb.write_text(json.dumps({"rows": [{"endpoint_err_mm": 1.0}]}))
    assert S.offline_summary(fa, n_boot=10)["n_windows"] == 1
    assert S.paired_offline_diff(fa, fb, n_boot=10)["n"] == 0


def test_bootstrap_resamples_clusters_not_windows():
    # two episodes with very different levels: the clustered CI must be far wider
    # than the per-window CI (which treats 200 correlated windows as independent)
    x = np.concatenate([np.full(100, 10.0), np.full(100, 30.0)])
    clusters = ["e1"] * 100 + ["e2"] * 100
    m, lo, hi = S.bootstrap_ci(x, np.mean, n_boot=400, clusters=clusters)
    m2, lo2, hi2 = S.bootstrap_ci(x, np.mean, n_boot=400)
    assert m == m2 == 20.0
    assert (hi - lo) > 5 * (hi2 - lo2)
    assert {round(lo, 6), round(hi, 6)} <= {10.0, 20.0, 30.0}      # only whole-episode draws
    # one label per value is required
    try:
        S.bootstrap_ci(x, np.mean, n_boot=5, clusters=["e1"])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched clusters must raise")


class _Lab:
    def __init__(self, grasp_ok=False, inconclusive=False, reasons=(), lift_mm=0.0):
        self.grasp_ok, self.inconclusive, self.reasons = grasp_ok, inconclusive, list(reasons)
        self.lift_mm, self.z_close_mm, self.hold_s, self.c_hold = lift_mm, 120.0, 2.5, 0.9
        self.stall, self.n_close_attempts = False, 1


def test_rule_stage_mapping():
    assert S.rule_stage(_Lab(grasp_ok=True, lift_mm=80)) == 2
    assert S.rule_stage(_Lab(reasons=["lift 12mm < 50mm"])) == 1           # held with contact, no lift
    assert S.rule_stage(_Lab(reasons=["never_closed"])) == 0
    assert S.rule_stage(_Lab(reasons=["c_hold 0.12 < 0.80", "lift 0mm < 50mm"])) == 0
    assert S.rule_stage(_Lab(reasons=["z_close 210mm > 160mm"])) == 0
    assert S.rule_stage(_Lab(inconclusive=True, reasons=["hold_truncated 0.4s < 2.0s"])) is None
    # unmeasurable recordings have no rule verdict (they fall back to the operator, flagged)
    for bad in (["missing_stream:gripper"], ["unreadable_stream:KeyError"], ["empty_streams"],
                ["no_tactile_stream", "c_hold 0.00 < 0.80", "lift 0mm < 50mm"],
                ["z_max_unknown_task:smoke", "lift 0mm < 50mm"]):
        assert S.rule_stage(_Lab(reasons=bad)) is None, bad
    assert S.rule_stage(_Lab(reasons=["hold 1.2s < 2.0s", "lift 3mm < 50mm"])) == 0   # hold not held


def test_operator_note_fallback_reads_no_lift_as_not_lifted():
    m = {"status": "finalized", "success": False}
    assert S.episode_outcome({**m, "notes": "operator: f lift"}) == 2
    assert S.episode_outcome({**m, "notes": "auto-relabel: outcome=grasp-contact-no-lift"}) == 1
    assert S.episode_outcome({**m, "notes": "operator: f no lift, no grasp"}) == 1
    assert S.episode_outcome({**m, "notes": "f: closed but never lifted"}) == 1
    assert S.episode_outcome({**m, "notes": "f"}) == 0


def test_episode_outcome_rule_uses_rule_for_0_2_and_operator_for_placed(monkeypatch, tmp_path):
    import phantom.eval.grasp_label as G
    labs = {}
    monkeypatch.setattr(G, "label_episode", lambda ep, hw, task=None: labs[Path(ep).name])
    meta_f = {"status": "finalized", "success": False, "notes": "operator: f grasp", "task": "waffles"}
    meta_s = {"status": "finalized", "success": True, "notes": "", "task": "waffles"}
    labs["e_lift"] = _Lab(grasp_ok=True, lift_mm=70)
    labs["e_close"] = _Lab(reasons=["lift 3mm < 50mm"])
    labs["e_air"] = _Lab(reasons=["c_hold 0.00 < 0.80", "lift 0mm < 50mm"])
    labs["e_trunc"] = _Lab(inconclusive=True, reasons=["hold_truncated 0.3s < 2.0s"])
    labs["e_placed"] = _Lab(grasp_ok=True, lift_mm=90)
    labs["e_placed_conflict"] = _Lab(reasons=["never_closed"])
    hw = object()
    assert S.episode_outcome_rule(tmp_path / "e_lift", meta_f, hw)[0] == 2      # operator said grasp, rule says lift
    o, info = S.episode_outcome_rule(tmp_path / "e_close", meta_f, hw)
    assert o == 1 and info["source"] == "rule" and "operator_stage" not in info
    o, info = S.episode_outcome_rule(tmp_path / "e_air", meta_f, hw)
    assert o == 0 and info["operator_stage"] == 1                              # note said grasp, pads say air
    o, info = S.episode_outcome_rule(tmp_path / "e_trunc", meta_f, hw)
    assert o == 1 and info["source"] == "operator" and info["rule_inconclusive"]
    o, info = S.episode_outcome_rule(tmp_path / "e_placed", meta_s, hw)
    assert o == 3 and info["source"] == "operator" and "conflict" not in info
    o, info = S.episode_outcome_rule(tmp_path / "e_placed_conflict", meta_s, hw)
    assert o == 3 and "conflict" in info
    # unlabeled stays unlabeled even if the streams show a lift
    assert S.episode_outcome_rule(tmp_path / "e_lift", {"status": "aborted", "success": None}, hw)[0] is None
    # an unreadable recording falls back to the operator note, flagged
    def boom(ep, hw, task=None):
        raise OSError("no streams")
    monkeypatch.setattr(G, "label_episode", boom)
    o, info = S.episode_outcome_rule(tmp_path / "e_lift", meta_f, hw)
    assert o == 1 and "rule_error" in info


def test_load_episodes_with_hw_uses_rule(monkeypatch, tmp_path):
    import phantom.eval.grasp_label as G
    monkeypatch.setattr(G, "label_episode", lambda ep, hw, task=None: _Lab(grasp_ok=True, lift_mm=60))
    _ep(tmp_path, "ep_teacher_waffles_1_000", task="waffles", seed=101, ckpt="BEST.pt", notes="operator: f")
    eps = S.load_episodes(tmp_path, hw=object())
    assert eps[0]["outcome"] == 2 and eps[0]["outcome_info"]["source"] == "rule"
    assert S.load_episodes(tmp_path)[0]["outcome"] == 0
