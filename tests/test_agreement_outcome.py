"""tools/rig/analysis/agreement_outcome.py — per-replan agreement (shadow or
selector) joined to the episode outcome.

The fixtures are synthetic deploy days: meta.json in the style of
tests/test_stats.py's `_ep` helper (tags label:/ckpt:/seed:/stop:, status +
success) plus a planner_trace.json whose accepted rows carry the diag the
policy's `_shadow_video_agreement` writes. High agreement distances and a high
disagreement rate are PLANTED in the failed episodes, so the direction of every
statistic here is known in advance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.rig.analysis.agreement_outcome import (
    RECORD_KEYS,
    auc,
    auc_ci,
    day_records,
    episode_features,
    main,
    pooled_threshold,
    prestop_test,
    read_jsonl,
    trace_records,
)

REPO = Path(__file__).resolve().parents[1]
HW_NUC = REPO / "configs" / "hardware.nuc.yaml"


# --------------------------------------------------------------- fixtures
def _meta(d: Path, *, task, seed, label, ckpt, stop, status="finalized",
          success=False, notes=""):
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({
        "task": task, "status": status, "success": success, "notes": notes,
        "tags": ["nfe1", f"seed:{seed}", f"ckpt:{ckpt}", f"label:{label}",
                 f"stop:{stop}", "start:-1,-2,3mm/g0.1"],
        "deploy_overrides": {"max_play_steps": 16}}), encoding="utf-8")


def _row(t, diag, accepted=True):
    return {"t": float(t), "latency_s": 0.3, "gate": 1.0, "p_evt": [1.0, 0, 0, 0, 0],
            "sigma": [0.1], "accepted": bool(accepted), "actions": [[0.0] * 7],
            "tcp_pose": None, "terminal_veto": None, "diag": diag}


def _shadow_diag(of_pick: float, *, disagree: bool, k_seeds=3):
    """A diag as `_shadow_video_agreement` writes it: the default rule's pick
    scores `of_pick`; when `disagree`, some other candidate is closer to the
    K-median, so the agreement rule would have kept that one instead."""
    if disagree:
        scores = [round(of_pick - 0.05, 6), float(of_pick), float(of_pick) + 0.2]
        k_pick = 1
    else:
        scores = [float(of_pick), float(of_pick) + 0.1, float(of_pick) + 0.2]
        k_pick = 0
    scores = scores[:k_seeds] if k_seeds <= 3 else scores
    spick = int(np.argmin(scores))
    return {"nfe": 1, "guidance": 1.0, "k_seeds": len(scores), "k_pick": k_pick,
            "head_dz_mm": [1.0] * len(scores), "k_rejected": 0,
            "video_agreement_shadow": scores,
            "video_agreement_shadow_pick": spick,
            "video_agreement_shadow_agrees": int(spick == k_pick),
            "video_agreement_shadow_of_pick": float(scores[k_pick]),
            "video_agreement_shadow_spread": float(max(scores) - min(scores))}


def _selector_diag(scores):
    """A diag as `_select_video_agreement` writes it: the pick IS the argmin."""
    scores = [float(s) for s in scores]
    pick = int(np.argmin(scores))
    return {"nfe": 1, "k_seeds": len(scores), "k_pick": pick,
            "k_selection": "video_agreement",
            "video_agreement": scores,
            "video_agreement_pick": float(scores[pick]),
            "video_agreement_spread": float(max(scores) - min(scores))}


def _trace(d: Path, rows):
    (d / "planner_trace.json").write_text(json.dumps(rows), encoding="utf-8")


#: the planted per-replan distances of the chosen chunk
FAIL1 = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
FAIL1_DIS = [True, True, False, True, True, True]
FAIL2 = [0.5, 0.7, 0.9, 1.1, 1.3]
OK1 = [0.05, 0.10, 0.15, 0.10]
OK2 = [0.08, 0.12, 0.06]


def _day(tmp_path: Path, day="20260915") -> Path:
    root = tmp_path / day
    # 1. failed (reach), hard stop, high distances rising into the stop
    d = root / "ep_v6_waffles_1_000"
    _meta(d, task="waffles", seed=101, label="v6_simft2k", ckpt="v6.pt",
          stop="safety_stop", success=False, notes="operator: f")
    _trace(d, [_row(10.0 + i, _shadow_diag(v, disagree=g))
               for i, (v, g) in enumerate(zip(FAIL1, FAIL1_DIS))]
           # a rejected replan is not a record, however high it scores
           + [_row(20.0, _shadow_diag(9.0, disagree=True), accepted=False)])
    # 2. failed (close), hard stop
    d = root / "ep_v6_waffles_2_000"
    _meta(d, task="waffles", seed=102, label="v6_simft2k", ckpt="v6.pt",
          stop="control_lost", success=False, notes="operator: f grasp")
    _trace(d, [_row(10.0 + i, _shadow_diag(v, disagree=True))
               for i, v in enumerate(FAIL2)])
    # 3-4. placed, clean stop, low distances, the agreement rule agrees
    d = root / "ep_v6_waffles_3_000"
    _meta(d, task="waffles", seed=103, label="v6_simft2k", ckpt="v6.pt",
          stop="finish", success=True)
    _trace(d, [_row(10.0 + i, _shadow_diag(v, disagree=False))
               for i, v in enumerate(OK1)])
    d = root / "ep_v6_waffles_4_000"
    _meta(d, task="waffles", seed=104, label="v6_simft2k", ckpt="v6.pt",
          stop="finish", success=True)
    _trace(d, [_row(10.0 + i, _shadow_diag(v, disagree=False))
               for i, v in enumerate(OK2)])
    return root


def _extras(root: Path):
    """The awkward episodes: unlabeled, selector-mode, no trace, no shadow."""
    d = root / "ep_v6_waffles_5_000"          # unlabeled (aborted, no verdict)
    _meta(d, task="waffles", seed=105, label="v6_simft2k", ckpt="v6.pt",
          stop="operator_abort", status="aborted", success=None)
    _trace(d, [_row(10.0 + i, _shadow_diag(v, disagree=False))
               for i, v in enumerate([0.30, 0.35])])
    d = root / "ep_sel_egg_6_000"             # selector mode: no shadow keys
    _meta(d, task="egg", seed=106, label="v6_selector", ckpt="v6.pt",
          stop="finish", success=True)
    _trace(d, [_row(10.0, _selector_diag([0.30, 0.10, 0.50])),
               _row(11.0, _selector_diag([0.20, 0.40, 0.90]))])
    d = root / "ep_v6_waffles_7_000"          # no planner_trace.json at all
    _meta(d, task="waffles", seed=107, label="v6_simft2k", ckpt="v6.pt",
          stop="finish", success=True)
    d = root / "ep_v6_waffles_8_000"          # trace without any agreement key
    _meta(d, task="waffles", seed=108, label="v6_simft2k", ckpt="v6.pt",
          stop="finish", success=False, notes="operator: f")
    _trace(d, [_row(10.0, {"nfe": 1, "k_seeds": 3, "k_pick": 0,
                           "head_dz_mm": [1.0, 2.0, 3.0]}),
               _row(11.0, {"nfe": 1, "video_agreement_shadow_error": "no video frames"})])


# ------------------------------------------------------------------- schema
def test_records_have_exactly_the_shared_schema_and_the_shadow_semantics(tmp_path):
    root = _day(tmp_path)
    recs, diag = day_records(root)
    assert diag["with_agreement"] == 4 and diag["n_episodes"] == 4
    assert len(recs) == len(FAIL1) + len(FAIL2) + len(OK1) + len(OK2)
    for r in recs:
        assert set(r) == set(RECORD_KEYS)
    r0 = [r for r in recs if r["episode"] == "ep_v6_waffles_1_000"][0]
    assert (r0["day"], r0["task"], r0["seed"], r0["label"], r0["ckpt"]) == \
        ("20260915", "waffles", 101, "v6_simft2k", "v6.pt")
    assert (r0["outcome"], r0["stop"], r0["source"]) == (0, "safety_stop", "shadow")
    assert (r0["i"], r0["t"], r0["k_seeds"]) == (0, 10.0, 3)
    # shadow: agreement_of_pick is the DEFAULT rule's pick distance, and
    # trace_pick is its k_pick — not the agreement rule's argmin
    assert r0["agreement_of_pick"] == FAIL1[0] and r0["trace_pick"] == 1
    assert r0["pick"] == int(np.argmin(r0["agreement"])) == 0
    assert r0["spread"] == max(r0["agreement"]) - min(r0["agreement"])
    # the rejected replan never becomes a record
    assert max(r["i"] for r in recs if r["episode"] == "ep_v6_waffles_1_000") == 5


def test_selector_rows_are_parsed_and_pick_coincides_with_the_default(tmp_path):
    root = _day(tmp_path)
    _extras(root)
    recs, _ = day_records(root)
    sel = [r for r in recs if r["source"] == "selector"]
    assert len(sel) == 2
    assert [r["agreement"] for r in sel] == [[0.30, 0.10, 0.50], [0.20, 0.40, 0.90]]
    for r in sel:
        assert r["pick"] == r["trace_pick"] == int(np.argmin(r["agreement"]))
        assert r["agreement_of_pick"] == min(r["agreement"])
        assert r["task"] == "egg" and r["label"] == "v6_selector"
    assert sel[0]["spread"] == 0.4
    # a selector-only report has nothing to say about disagreement — the rule
    # is the pick there — and says so instead of dividing by zero
    from tools.rig.analysis.agreement_outcome import analyze, print_report
    res = analyze(sel, n_boot=50)
    assert res["disagreement"]["n_comparable_replans"] == 0
    assert res["disagreement"]["n_selector_replans"] == 2
    assert np.isnan(res["episodes"][0]["disagree_rate"])
    print_report(res)


def test_missing_shadow_keys_and_missing_trace_yield_no_records_not_a_crash(tmp_path):
    root = tmp_path / "20260915"
    _extras(root)
    recs, diag = day_records(root)
    assert diag["no_trace"] == ["ep_v6_waffles_7_000"]
    assert diag["no_agreement"] == ["ep_v6_waffles_8_000"]
    assert {r["episode"] for r in recs} == {"ep_v6_waffles_5_000", "ep_sel_egg_6_000"}
    # an unreadable trace is reported, not raised
    (root / "ep_v6_waffles_8_000" / "planner_trace.json").write_text("{not json", encoding="utf-8")
    _, diag = day_records(root)
    assert len(diag["unreadable_trace"]) == 1
    # and a trace that is not even a list of rows scores nothing
    assert trace_records({"rows": []}, {}) == [] and trace_records([None, 3], {}) == []


# ----------------------------------------------------------------- features
def test_episode_features_are_the_planted_summaries(tmp_path):
    recs, _ = day_records(_day(tmp_path))
    feats = episode_features(recs)                      # the one-argument form
    assert [f["episode"] for f in feats] == [f"ep_v6_waffles_{i}_000" for i in (1, 2, 3, 4)]
    f1 = feats[0]
    assert f1["n_replans"] == 6 and f1["outcome"] == 0 and f1["stop"] == "safety_stop"
    assert f1["mean_of_pick"] == 0.9 and f1["max_of_pick"] == 1.4
    assert f1["pctl_of_pick"] == pytest.approx(1.3)     # p90 of six points, interpolated
    assert f1["n_disagree"] == 5 and f1["disagree_rate"] == 5 / 6
    assert f1["sources"] == ["shadow"]
    f3 = feats[2]
    assert f3["outcome"] == 3 and f3["disagree_rate"] == 0.0
    assert f3["mean_of_pick"] == float(np.mean(OK1))
    # frac_above counts against the POOLED cut, so the failures carry it all
    thr = pooled_threshold(recs, 90.0)
    assert f1["frac_above"] > 0 and f3["frac_above"] == 0.0
    assert all(f["threshold"] == thr for f in feats)
    # a different percentile moves the cut and the fraction, not the mean
    hi = episode_features(recs, percentile=99.0)
    assert hi[0]["frac_above"] <= f1["frac_above"] and hi[0]["mean_of_pick"] == 0.9


def test_auc_separates_the_planted_case_and_sits_at_half_for_shuffled_labels():
    # perfectly separated: AUC 1.0 and a CI that excludes chance
    scores = np.concatenate([np.linspace(1.0, 2.0, 12), np.linspace(3.0, 4.0, 12)])
    labels = np.r_[np.zeros(12, int), np.ones(12, int)]
    a = auc_ci(scores, labels, n_boot=1000, seed=0)
    assert a["auc"] == 1.0 and a["lo"] > 0.5 and a["n_pos"] == a["n_neg"] == 12
    # the same scores with labels shuffled carry no signal
    rng = np.random.default_rng(7)
    shuffled = rng.permutation(labels)
    b = auc_ci(scores, shuffled, n_boot=1000, seed=0)
    assert abs(b["auc"] - 0.5) < 0.25 and b["lo"] < 0.5 < b["hi"]
    # ties are half-credit, and an absent class has no AUC
    assert auc([1.0, 1.0], [1, 0]) == 0.5
    assert np.isnan(auc_ci([1.0, 2.0], [1, 1], n_boot=10)["auc"])
    assert np.isnan(auc_ci([np.nan, 2.0], [1, 0], n_boot=10)["auc"])


def test_auc_over_episodes_ranks_the_failed_episodes_above_the_placed_ones(tmp_path):
    from tools.rig.analysis.agreement_outcome import analyze
    root = _day(tmp_path)
    _extras(root)
    recs, _ = day_records(root)
    res = analyze(recs, n_boot=200)
    pooled = res["groups"]["pooled"]["auc"]
    np_ = pooled["not_placed"]
    # the two planted failures rank above the three placed episodes on every
    # distance feature; the unlabeled episode is dropped from the AUC
    assert np_["n_pos"] == 2 and np_["n_neg"] == 3 and np_["n_unusable"] == 1
    for feat in ("mean_of_pick", "pctl_of_pick", "max_of_pick", "frac_above"):
        assert np_["features"][feat]["auc"] == 1.0, feat
        assert np_["features"][feat]["n_boot_used"] > 0
    hard = pooled["hard_stop"]
    assert hard["n_pos"] == 2 and hard["n_neg"] == 4
    assert hard["features"]["mean_of_pick"]["auc"] == 1.0
    # ...but the unlabeled episode IS listed, with its features
    ep5 = [f for f in res["episodes"] if f["episode"] == "ep_v6_waffles_5_000"]
    assert len(ep5) == 1 and ep5[0]["outcome"] is None and ep5[0]["n_replans"] == 2
    assert res["n_episodes"] == 6
    # groups exist per task, per label and per source (one day => no day split)
    assert "task:waffles" in res["groups"] and "task:egg" in res["groups"]
    assert "label:v6_selector" in res["groups"] and "source:selector" in res["groups"]
    assert "day:20260915" not in res["groups"]
    # disagreement: planted only in the failures, which are also the hard stops
    d = res["disagreement"]
    assert d["n_selector_replans"] == 2
    assert d["mean_rate_failed"] > d["mean_rate_placed"] == 0.0
    assert d["diff_failed_minus_placed"] > 0
    assert d["median_rate_failed"] > d["median_rate_placed"] == 0.0
    # 2 hard stops vs 3, not 4: the selector episode has no comparable replan,
    # so it has no disagreement rate to average
    assert (d["n_hard_stop_episodes"], d["n_no_hard_stop_episodes"]) == (2, 3)
    assert d["mean_rate_hard_stop"] > d["mean_rate_no_hard_stop"] == 0.0
    assert d["diff_hard_minus_rest"] > 0 and "test_hard_stop" in d


def test_prestop_window_finds_the_rise_into_a_hard_stop(tmp_path):
    recs, _ = day_records(_day(tmp_path))
    p = prestop_test(recs, window=3)
    assert p["n_hard_episodes"] == 2 and p["n_episodes"] == 4
    assert p["n_pre"] == 6 and p["n_rest"] == len(recs) - 6
    assert p["median_pre"] > p["median_rest"]
    assert p["rank_biserial"] > 0.9 and p["p"] < 0.05
    assert p["method"].startswith("scipy") or p["method"].startswith("permutation")
    # no hard stop at all => nothing to compare, still no crash
    q = prestop_test([r for r in recs if r["stop"] == "finish"], window=3)
    assert q["n_pre"] == 0 and np.isnan(q["p"])
    # a window longer than the episode takes every replan of it
    w = prestop_test(recs, window=99)
    assert w["n_pre"] == len(FAIL1) + len(FAIL2)


def test_length_matching_removes_the_duration_confound(tmp_path):
    """The trap the pooled AUC walks into: a failure is stopped after a few
    reaching replans, a placement runs on through the phases where the
    imaginations diverge anyway. Here the PLACED episodes are long and their
    LATE replans score high, so whole-episode features call them the failures;
    the same episodes compared over their first 4 replans do not."""
    from tools.rig.analysis.agreement_outcome import analyze, truncate_records

    root = tmp_path / "20260915"
    for i, (seed, ok) in enumerate([(201, False), (202, False), (203, True), (204, True)]):
        d = root / f"ep_len_waffles_{i}_000"
        _meta(d, task="waffles", seed=seed, label="v6_simft2k", ckpt="v6.pt",
              stop="finish" if ok else "safety_stop", success=ok,
              notes="" if ok else "operator: f")
        if ok:      # long: 4 quiet replans, then 16 loud ones
            vals = [0.20, 0.22, 0.24, 0.26] + [2.0 + 0.01 * j for j in range(16)]
        else:       # short: stopped early, and already noisier than the openings
            vals = [0.50, 0.55, 0.60, 0.65]
        _trace(d, [_row(10.0 + j, _shadow_diag(v, disagree=False))
                   for j, v in enumerate(vals)])
    recs, _ = day_records(root)

    whole = analyze(recs, n_boot=200)["groups"]["pooled"]["auc"]["not_placed"]
    assert whole["features"]["mean_of_pick"]["auc"] == 0.0        # exactly backwards
    matched = analyze(recs, n_boot=200, first_n=4)["groups"]["pooled"]["auc"]["not_placed"]
    assert matched["features"]["mean_of_pick"]["auc"] == 1.0
    # the same comparison is reported without --first-n, under "length"
    res = analyze(recs, n_boot=200, matched=(4,))
    L = res["length"]
    assert L["median_replans_failed"] == 4 and L["median_replans_placed"] == 20
    assert L["auc_n_replans"]["not_placed"]["auc"] == 0.0         # duration alone separates
    m = L["matched"]["4"]
    assert m["n_short"] == 0 and m["n_reaching"] == 4
    for rule in ("truncate", "drop_short"):
        r = m["rules"][rule]
        assert r["n_records"] == 16 and r["n_episodes"] == 4
        assert r["groups"]["pooled"]["auc"]["not_placed"]["features"]["mean_of_pick"]["auc"] == 1.0

    # truncation keeps the FIRST replans by trace index, and short episodes whole
    t = truncate_records(recs, 4)
    assert len(t) == 16
    assert sorted(r["i"] for r in t if r["episode"] == "ep_len_waffles_2_000") == [0, 1, 2, 3]
    assert len(truncate_records(recs, 99)) == len(recs)
    assert truncate_records(recs, None) == recs
    feats = episode_features(recs, first_n=4)
    assert [f["n_replans"] for f in feats] == [4, 4, 4, 4]


def test_both_length_matching_rules_are_reported_and_drop_short_loses_the_failures(tmp_path):
    """The second fork under length matching: truncate-and-keep leaves exposure
    unequal, drop-short matches it by deleting the SHORT episodes — which are
    the failures. At an N nearly every episode reaches the two rules coincide,
    which is what makes a small N the honest operating point; at a large N the
    positive count collapses and the report must show that, not hide it."""
    from tools.rig.analysis.agreement_outcome import analyze, truncate_records

    root = tmp_path / "20260915"
    # 6 short failures (4 replans) and 4 long placements (20 replans)
    for i in range(6):
        d = root / f"ep_short_waffles_{i}_000"
        _meta(d, task="waffles", seed=300 + i, label="v6_simft2k", ckpt="v6.pt",
              stop="safety_stop", success=False, notes="operator: f")
        _trace(d, [_row(10.0 + j, _shadow_diag(0.5 + 0.01 * j, disagree=False))
                   for j in range(4)])
    for i in range(4):
        d = root / f"ep_long_waffles_{i}_000"
        _meta(d, task="waffles", seed=400 + i, label="v6_simft2k", ckpt="v6.pt",
              stop="finish", success=True)
        _trace(d, [_row(10.0 + j, _shadow_diag(0.2 + 0.01 * j, disagree=False))
                   for j in range(20)])
    recs, _ = day_records(root)
    L = analyze(recs, n_boot=100, matched=(4, 10))["length"]

    # N=4: every episode reaches it, so the two rules are the SAME sample
    m4 = L["matched"]["4"]
    assert m4["n_reaching"] == 10 and m4["n_short"] == 0
    a4 = {k: m4["rules"][k]["groups"]["pooled"]["auc"]["not_placed"] for k in m4["rules"]}
    assert a4["truncate"]["n_pos"] == a4["drop_short"]["n_pos"] == 6
    assert (a4["truncate"]["features"]["mean_of_pick"]["auc"]
            == a4["drop_short"]["features"]["mean_of_pick"]["auc"] == 1.0)
    assert m4["rules"]["drop_short"]["n_episodes_dropped"] == 0

    # N=10: drop-short deletes every failure, so its AUC cannot be computed at
    # all — the whole positive class is gone, and the counts say so
    m10 = L["matched"]["10"]
    assert m10["n_reaching"] == 4 and m10["n_short"] == 6
    t10 = m10["rules"]["truncate"]["groups"]["pooled"]["auc"]["not_placed"]
    d10 = m10["rules"]["drop_short"]["groups"]["pooled"]["auc"]["not_placed"]
    assert t10["n_pos"] == 6 and t10["n_neg"] == 4
    assert d10["n_pos"] == 0 and d10["n_neg"] == 4
    assert np.isnan(d10["features"]["mean_of_pick"]["auc"])
    assert m10["rules"]["drop_short"]["n_episodes_dropped"] == 6

    # the rule is selectable for the primary tables too
    assert len(truncate_records(recs, 10, drop_short=True)) == 40
    assert len(truncate_records(recs, 10, drop_short=False)) == 4 * 10 + 6 * 4
    prim = analyze(recs, n_boot=50, first_n=10, drop_short=True)
    assert prim["match_rule"] == "drop_short" and prim["n_episodes"] == 4
    assert analyze(recs, n_boot=50, first_n=10)["n_episodes"] == 10
    assert [f["n_replans"] for f in episode_features(recs, first_n=10, drop_short=True)] == [10] * 4


def test_the_permutation_fallback_matches_scipy_when_scipy_is_absent(tmp_path, monkeypatch):
    """scipy is optional: without it the same U statistic is tested by
    permutation, and both paths must tell the same story."""
    import builtins

    recs, _ = day_records(_day(tmp_path))
    with_scipy = prestop_test(recs, window=3)
    real_import = builtins.__import__

    def no_scipy(name, *a, **kw):
        if name.startswith("scipy"):
            raise ImportError("no scipy")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    without = prestop_test(recs, window=3, n_perm=2000, seed=0)
    assert without["method"] == "permutation(2000)"
    assert without["u"] == pytest.approx(with_scipy["u"])
    assert without["rank_biserial"] == pytest.approx(with_scipy["rank_biserial"])
    assert without["p"] < 0.05 and abs(without["p"] - with_scipy["p"]) < 0.05


# ---------------------------------------------------------------------- cli
def test_jsonl_records_merge_as_a_replay_source(tmp_path):
    from tools.rig.analysis.agreement_outcome import analyze
    root = _day(tmp_path)
    extra = tmp_path / "replay.jsonl"
    rows = [{"episode": "ep_replay_1", "day": "replay", "task": "waffles", "seed": 201,
             "ckpt": "v6.pt", "label": "v6_replayed", "outcome": 3, "stop": "finish",
             "i": i, "t": 10.0 + i, "k_seeds": 3, "agreement": [0.1, 0.2, 0.3],
             "pick": 0, "agreement_of_pick": 0.1, "spread": 0.2, "trace_pick": 0,
             "source": "replay"} for i in range(3)]
    extra.write_text("\n".join(json.dumps(r) for r in rows)
                     + "\n\nnot json\n" + json.dumps({"episode": "x"}) + "\n",
                     encoding="utf-8")
    got, diag = read_jsonl(extra)
    assert len(got) == 3 and diag["n_skipped"] == 2
    assert all(set(r) == set(RECORD_KEYS) and r["source"] == "replay" for r in got)
    recs, _ = day_records(root)
    res = analyze(recs + got, n_boot=100)
    assert res["sources"] == {"replay": 3, "shadow": len(recs)}
    assert "source:replay" in res["groups"] and "day:replay" in res["groups"]
    assert res["n_episodes"] == 5
    # a replayed row CAN disagree (its trace_pick is what the rig played), so
    # it is counted with the shadow rows, unlike a selector row
    d = res["disagreement"]
    assert d["n_comparable_replans"] == len(recs) + 3
    assert d["n_shadow_replans"] == len(recs) and d["n_other_replans"] == 3
    # K=3 everywhere here, so two independent rules would already differ on 2/3
    # of the replans: the rate means nothing read against 0
    assert d["k_seeds"] == [3] and d["chance_disagree_rate"] == pytest.approx(2 / 3)


def test_cli_prints_a_table_and_writes_json(tmp_path, capsys):
    root = _day(tmp_path)
    _extras(root)
    out = tmp_path / "agreement.json"
    argv = [str(root), "--json-out", str(out), "--percentile", "90", "--n-boot", "100"]
    if HW_NUC.exists():
        # the rig invocation: episodes with no streams fall back to the
        # operator verdict instead of raising
        argv += ["--hw", str(HW_NUC)]
    assert main(argv) == 0
    txt = capsys.readouterr().out
    assert "agreement records:" in txt and "per episode" in txt
    assert "AUC vs not placed" in txt and "pre-stop window" in txt
    assert "ep_v6_waffles_1_000" in txt and "ep_v6_waffles_5_000" in txt
    res = json.loads(out.read_text())
    assert res["n_records"] == len(FAIL1) + len(FAIL2) + len(OK1) + len(OK2) + 2 + 2
    assert res["n_episodes"] == 6 and res["percentile"] == 90.0
    assert len(res["records"]) == res["n_records"]
    assert set(res["records"][0]) == set(RECORD_KEYS)
    assert res["groups"]["pooled"]["auc"]["not_placed"]["features"]["mean_of_pick"]["auc"] == 1.0
    assert res["inputs"]["load"][0]["no_trace"] == ["ep_v6_waffles_7_000"]
    # an empty day is reported, not a crash
    empty = tmp_path / "20260916"
    empty.mkdir()
    assert main([str(empty)]) == 0
    assert "nothing to score" in capsys.readouterr().out
