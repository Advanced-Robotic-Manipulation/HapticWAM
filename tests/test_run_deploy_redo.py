"""Rig 09-15: the 'r' (redo) verdict — the take is recorded for the logs but
counted nowhere, and the same episode index (same seed, same cell) runs again."""
from pathlib import Path

from phantom.data.schema import NON_TRAINING_TAGS
from phantom.eval import stats as S
from phantom.scripts.run_deploy import VERDICT_PROMPT, label_episode, parse_verdict


class FakeRecorder:
    def __init__(self):
        self.calls = []

    def relabel(self, path, **kw):
        self.calls.append((path, kw))


def test_parse_verdict_accepts_redo_with_and_without_damage():
    assert parse_verdict("r") == ("r", False)
    assert parse_verdict("redo wrong placement") == ("r", False)
    assert parse_verdict("rd bumped the crate") == ("r", True)
    assert parse_verdict("s") == ("s", False)
    assert parse_verdict("x") == ("", False)


def test_prompt_offers_redo():
    assert "[r]edo" in VERDICT_PROMPT and "NOT counted" in VERDICT_PROMPT


def test_label_episode_redo_is_recorded_but_not_counted():
    rec = FakeRecorder()
    ep = Path("/tmp/ep_teacher_waffles_1_000")
    assert label_episode(rec, ep, "r false start") == "r"
    (path, kw), = rec.calls
    assert path == ep
    assert kw["success"] is None and kw["status"] == "aborted"
    assert "redo" in kw["tags"] and "unlabeled" in kw["remove_tags"]
    assert kw["damage"] is False and "REDO" in kw["notes"]


def test_redo_tag_keeps_the_episode_out_of_training():
    assert "redo" in NON_TRAINING_TAGS


def test_stats_sees_no_outcome_for_a_redo_take():
    meta = {"status": "aborted", "success": None, "task": "waffles",
            "tags": ["redo", "label:v6_simft2k"], "notes": "operator: r REDO"}
    assert S.episode_outcome(meta) is None


def test_pairs_skip_the_redo_take_and_count_the_rerun():
    def ep(arm, outcome, path):
        return {"task": "waffles", "seed": 101, "arm": arm, "arm_tags": {arm},
                "outcome": outcome, "outcome_info": {"source": "operator"}, "path": path}
    eps = [ep("label:t", None, "t_redo"), ep("label:t", 3, "t_ok"),
           ep("label:s", None, "s_redo"), ep("label:s", 0, "s_ok")]
    pairs, diag = S.pair_episodes(eps, "label:t", "label:s")
    assert len(pairs) == 1 and pairs[0]["a"] == 3 and pairs[0]["b"] == 0
    assert pairs[0]["a_path"] == "t_ok" and pairs[0]["b_path"] == "s_ok"
    assert pairs[0]["reruns"] == 0 and diag["unlabeled_skipped"] == 2


def test_redo_requeues_the_same_index():
    # the loop discipline of run_deploy.main: a redo puts the same index back
    # at the head, so the same seed runs again before the next episode
    pending = list(range(3))
    order, verdicts = [], iter(["s", "r", "f", "s"])
    while pending:
        i = pending.pop(0)
        order.append(i)
        if next(verdicts) == "r":
            pending.insert(0, i)
    assert order == [0, 1, 1, 2]


def test_preset_tags_can_name_an_arm_on_reserved_cells():
    # Block N (rig 09-15): the same model under two presets on cells nobody
    # else ran; the veto:/sel:/aveto: tags identify the arms
    on = {"tags": ["label:v6_simft2k", "ckpt_sha:1df9fc93510c", "veto:pc0.50/pn0.90/r3", "sel:default"]}
    off = {"tags": ["label:v6_simft2k", "ckpt_sha:1df9fc93510c", "veto:off", "sel:default"]}
    assert "veto:pc0.50/pn0.90/r3" in S.episode_arm_tags(on) and "veto:off" in S.episode_arm_tags(off)
    def ep(meta, outcome, seed):
        return {"task": "waffles", "seed": seed, "arm": "label:v6_simft2k", "arm_tags": S.episode_arm_tags(meta),
                "outcome": outcome, "outcome_info": {"source": "operator"}, "path": f"p{seed}{outcome}"}
    eps = [ep(on, 3, 131), ep(off, 0, 131), ep(on, 0, 132), ep(off, 0, 132)]
    pairs, _ = S.pair_episodes(eps, "veto:pc0.50/pn0.90/r3", "veto:off")
    assert [(p["seed"], p["a"], p["b"]) for p in pairs] == [(131, 3, 0), (132, 0, 0)]
