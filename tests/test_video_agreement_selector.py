"""`--select-by video_agreement`: rank the K sampled chunks by how far each
one's IMAGINED future is from the elementwise median of the K imaginations.

The score needs no ground truth, so unlike the hindsight `video_err` of
tools/terminal_eval.py it is computable at deploy — it is that tool's
`video_err_to_median` column, and these tests pin the two definitions against
each other rather than restating the formula.

Offline (124 val windows x 4 seeds per checkpoint, scratchpad/wm_*.json) the
argmin of that score moved mean endpoint error 20.04 -> 19.63 mm (v6),
16.25 -> 15.36 (v6 nfe2), 19.57 -> 19.20 (ftA), 18.68 -> 18.43 (sensor-free
student); scratchpad/wm_selector_check.py re-derives those four numbers with
the selector implemented here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phantom.deploy.planner import PlannerLoop
from phantom.inference.policy import (
    PhantomPolicy,
    select_by_video_agreement,
    video_agreement_scores,
    video_gen_latents,
)
from phantom.model.sequence import FrameGroup
from phantom_test_utils import make_small_hw

from test_deploy_levers import _Ex, _Snaps, _cpk, _fake_policy, _snap


# ---------------------------------------------------------------------------
# fakes: a layout whose VIDEO_GEN frames are a known slice of x_final
# ---------------------------------------------------------------------------

class _Layout:
    """Minimal SequenceLayout surface the selector reads."""

    def __init__(self, gen=slice(1, 3), student=True, has_gen=True):
        self.student, self._gen, self._has = student, gen, has_gen

    def has(self, g):
        return bool(self._has) if g is FrameGroup.VIDEO_GEN else True

    def frame_slice(self, g):
        assert g is FrameGroup.VIDEO_GEN
        return self._gen


def _pred_with_video(x_final, *, H=None, A=None, hw=None, dz=None):
    """A PhantomPrediction whose K rows carry the given x_final latents."""
    from phantom.model.rf import PhantomPrediction
    K = x_final.shape[0]
    H = H if H is not None else hw.control.chunk_horizon
    A = A if A is not None else hw.control.action_dim
    actions = torch.zeros(K, H, A)
    for j in range(K):
        # a per-candidate signature the test can read back off the plan
        actions[j, :, 0] = float(j + 1)
    if dz is not None:
        for j, v in enumerate(dz):
            actions[j, :, 2] = v
    return PhantomPrediction(
        actions_B_H_A=actions, cpk=_cpk(K), event_logits_B_Tc_E=torch.zeros(K, 3, 5),
        log_sigma_B_Tc_K=torch.zeros(K, 3, 1),
        governor_sigma_B_Tc=torch.zeros(K, 3),
        acc=SimpleNamespace(g=torch.zeros(K),
                            p_evt=torch.tensor([[1.0, 0, 0, 0, 0]] * K)),
        x_final_B_C_T_H_W=torch.as_tensor(x_final, dtype=torch.float32))


def _agreement_policy(hw, x_final, *, k_seeds=None, dz=None, **kw):
    K = x_final.shape[0]
    pol = _fake_policy(hw, lambda batch, **k: _pred_with_video(x_final, hw=hw, dz=dz),
                       k_seeds=k_seeds if k_seeds is not None else K)
    pol.pm = SimpleNamespace(layout=_Layout())
    pol.select_by = kw.pop("select_by", "video_agreement")
    pol.agreement_veto = kw.pop("agreement_veto", None)
    assert not kw
    return pol


def _latents(K=4, C=2, T=4, h=3, w=3, seed=0):
    return np.random.default_rng(seed).normal(size=(K, C, T, h, w))


def _expected(x, gen=slice(1, 3)):
    """The scores the policy must produce: computed on the VIDEO_GEN frames
    only, never on the whole (conditioning + action + contact) sequence."""
    return video_agreement_scores(list(np.asarray(x, np.float32)[:, :, gen]))


# ---------------------------------------------------------------------------
# 1. the score is tools/terminal_eval.py's definition, not a second one
# ---------------------------------------------------------------------------

def test_the_score_is_bit_identical_to_the_offline_definition():
    """`video_agreement_scores` IS terminal_eval's `seed_agreement` — the
    column the 19.63 / 15.36 / 19.20 / 18.43 numbers were measured on. If the
    two ever drift, the rig arm stops being the thing that was validated."""
    from tools.terminal_eval import seed_agreement                  # noqa: E402
    for seed in range(3):
        vids = list(_latents(K=4, seed=seed))
        assert video_agreement_scores(vids) == seed_agreement(vids)


def test_a_single_candidate_has_zero_distance_by_construction():
    assert video_agreement_scores(list(_latents(K=1))) == [0.0]


def test_the_selector_is_the_argmin_of_the_distances():
    """Pure-function check on fake latents: five constant candidates, so the
    elementwise median is the middle constant (0.30) and every distance is
    known in closed form. The candidate ON the median wins; the outlier that
    imagines a different future is ranked last."""
    base = np.zeros((5, 2, 4, 3, 3))
    for j, v in enumerate([0.10, 0.20, 0.30, 0.31, 9.00]):
        base[j] += v
    scores = video_agreement_scores(list(base))
    assert scores == pytest.approx([0.04, 0.01, 0.0, 0.0001, 75.69])
    assert select_by_video_agreement(scores) == int(np.argmin(scores)) == 2
    # the outlier is ranked worst, and by a wide margin
    assert scores[4] == max(scores) and scores[4] > 10 * max(scores[:4])


def test_ties_take_the_lowest_index_like_the_offline_pick():
    assert select_by_video_agreement([0.5, 0.5, 0.5]) == 0


def test_video_gen_latents_reads_the_gen_slice_of_every_candidate():
    x = torch.arange(2 * 3 * 5 * 2 * 2, dtype=torch.float32).reshape(2, 3, 5, 2, 2)
    vids = video_gen_latents(x, _Layout(gen=slice(1, 4)))
    assert len(vids) == 2 and vids[0].shape == (3, 3, 2, 2)
    assert np.array_equal(vids[1], x[1, :, 1:4].numpy())


def test_a_layout_without_imagined_frames_is_refused():
    with pytest.raises(ValueError, match="VIDEO_GEN"):
        video_gen_latents(torch.zeros(2, 1, 3, 1, 1), _Layout(has_gen=False))


# ---------------------------------------------------------------------------
# 2. the policy: selection, diagnostics, refusals
# ---------------------------------------------------------------------------

def test_replan_returns_the_agreeing_candidates_chunk():
    hw = make_small_hw()
    x = _latents(K=4, seed=7)
    x[2] *= 0.001                    # candidate 2 sits nearest the median
    pol = _agreement_policy(hw, x)
    expect = int(np.argmin(_expected(x)))
    plan = pol.replan(_snap(hw), None, np.zeros(6))
    assert plan.diag["k_selection"] == "video_agreement"
    assert plan.diag["k_pick"] == expect
    # actions[:, 0] == j + 1 identifies which candidate was returned
    assert plan.actions[0, 0] == pytest.approx(expect + 1)
    assert plan.diag["video_agreement"] == pytest.approx(_expected(x))
    assert plan.diag["video_agreement_pick"] == pytest.approx(
        min(plan.diag["video_agreement"]))


def test_the_K_distances_reach_the_planner_trace():
    """The per-seed distances are the only record of how much the imaginations
    disagreed on a replan — a veto threshold is read off them."""
    hw = make_small_hw()
    x = _latents(K=3, seed=1)
    pol = _agreement_policy(hw, x)
    loop = PlannerLoop(hw, pol, _Snaps(hw), _Ex())
    loop.run(max_replans=1)
    d = loop.trace[0]["diag"]
    assert d["k_selection"] == "video_agreement" and d["k_seeds"] == 3
    assert d["video_agreement"] == pytest.approx(_expected(x))
    assert loop.trace[0]["accepted"] is True


def test_one_candidate_or_no_imagination_is_refused_not_silently_ranked():
    hw = make_small_hw()
    pol = _agreement_policy(hw, _latents(K=1), k_seeds=1)
    with pytest.raises(ValueError, match="k-seeds >= 2"):
        pol.replan(_snap(hw), None, np.zeros(6))

    pol = _agreement_policy(hw, _latents(K=4))
    pol.drop_video = True
    with pytest.raises(ValueError, match="drop_video"):
        pol.replan(_snap(hw), None, np.zeros(6))


def test_the_selector_name_is_validated():
    hw = make_small_hw()
    pol = _agreement_policy(hw, _latents(K=2))
    with pytest.raises(ValueError, match="select_by"):
        pol.select_by = "vibes"
    with pytest.raises(ValueError, match="agreement_veto"):
        pol.agreement_veto = -1.0


# ---------------------------------------------------------------------------
# 3. the default path is untouched
# ---------------------------------------------------------------------------

def test_select_by_default_changes_neither_the_seed_nor_the_plan():
    """The lever is opt-in: with select_by='default' the contact-consistent
    rule still runs and the plan is identical, field by field, to the one a
    policy that has never heard of the selector produces."""
    hw = make_small_hw()
    x = _latents(K=4, seed=3)
    x[3] *= 0.001                    # agreement would pick 3 ...
    dz = [-0.005, -0.001, -0.004, -0.0005]   # ... the default rule picks 0

    legacy = _fake_policy(hw, lambda batch, **k: _pred_with_video(x, hw=hw, dz=dz),
                          k_seeds=4)
    legacy.pm = SimpleNamespace(layout=_Layout())
    explicit = _agreement_policy(hw, x, dz=dz, select_by="default")

    assert legacy.select_by == "default"          # the default for a policy
    a = legacy.replan(_snap(hw), None, np.zeros(6))   # built without the flag
    b = explicit.replan(_snap(hw), None, np.zeros(6))
    assert a.diag["k_pick"] == b.diag["k_pick"] == 0
    assert "k_selection" not in a.diag
    assert np.array_equal(a.actions, b.actions)
    assert np.array_equal(a.sigma, b.sigma) and a.gate == b.gate
    assert np.array_equal(a.p_evt, b.p_evt)
    assert a.diag.keys() == b.diag.keys()
    assert "video_agreement" not in a.diag        # no score computed at all
    assert a.diag["head_dz_mm"] == b.diag["head_dz_mm"]


def test_the_default_selector_does_not_touch_the_imagined_frames():
    """Proof the default path pays nothing: a layout that raises on any
    VIDEO_GEN access still replans."""
    hw = make_small_hw()

    class _Explode(_Layout):
        def has(self, g):
            raise AssertionError("default selection must not read the layout")

    pol = _fake_policy(hw, lambda batch, **k: _pred_with_video(
        _latents(K=2), hw=hw, dz=[-0.005, -0.001]), k_seeds=2)
    pol.pm = SimpleNamespace(layout=_Explode())
    assert pol.replan(_snap(hw), None, np.zeros(6)).actions.shape == (
        hw.control.chunk_horizon, hw.control.action_dim)


# ---------------------------------------------------------------------------
# 4. the agreement veto
# ---------------------------------------------------------------------------

def _veto_loop(hw, threshold, x):
    pol = _agreement_policy(hw, x, agreement_veto=threshold)
    ex = _Ex()
    loop = PlannerLoop(hw, pol, _Snaps(hw), ex)
    loop.run(max_replans=1)
    return loop, ex


def test_a_plan_above_the_threshold_is_never_submitted():
    """Rejection reuses the path a plan submit() refuses already takes: the
    executor keeps the previous chunk and the loop replans. No new arm
    behaviour, no rewritten actions."""
    hw = make_small_hw()
    x = _latents(K=4, seed=11)
    loop, ex = _veto_loop(hw, 1e-9, x)          # every distance exceeds it
    assert ex.submitted == []                    # the executor saw nothing
    row = loop.trace[0]
    assert row["accepted"] is False
    assert row["diag"]["agreement_vetoed"] is True
    assert row["diag"]["agreement_veto_threshold"] == pytest.approx(1e-9)


def test_a_plan_below_the_threshold_is_submitted_normally():
    hw = make_small_hw()
    x = _latents(K=4, seed=11)
    loop, ex = _veto_loop(hw, 1e6, x)
    assert len(ex.submitted) == 1
    assert loop.trace[0]["accepted"] is True
    assert loop.trace[0]["diag"]["agreement_vetoed"] is False


def test_the_veto_never_cancels_a_scripted_terminal_veto_recovery():
    """A chunk the terminal veto rewrote is its arithmetic, not a model sample
    — and the phantom-grasp recovery has already spent one of its retries. The
    agreement score describes the PROPOSAL, so it must not drop that plan."""
    hw = make_small_hw()
    x = _latents(K=4, seed=11)
    pol = _agreement_policy(hw, x, agreement_veto=1e-9)   # would veto everything
    ex = _Ex()
    loop = PlannerLoop(hw, pol, _Snaps(hw), ex)
    loop._apply_veto = lambda *a, **k: {"action": "recovery_open", "retries": 1}
    loop.run(max_replans=1)
    assert len(ex.submitted) == 1                        # the recovery went out
    assert loop.trace[0]["accepted"] is True
    assert loop.trace[0]["diag"]["agreement_veto_suppressed"] == "recovery_open"


def test_the_veto_is_off_unless_a_threshold_is_given():
    hw = make_small_hw()
    loop, ex = _veto_loop(hw, None, _latents(K=4, seed=11))
    assert len(ex.submitted) == 1
    assert "agreement_vetoed" not in loop.trace[0]["diag"]


def test_a_vetoed_plan_does_not_become_the_next_replans_conditioning():
    """A rejected plan was never commanded, so it must not be carried forward
    as prev_plan — the same invariant the executor-rejection path holds."""
    hw = make_small_hw()
    seen = []
    x = _latents(K=4, seed=5)

    def sample(batch, **kw):
        seen.append(kw.get("prev_cpk"))
        return _pred_with_video(x, hw=hw)

    pol = _fake_policy(hw, sample, k_seeds=4)
    pol.pm = SimpleNamespace(layout=_Layout())
    pol.select_by, pol.agreement_veto = "video_agreement", 1e-9
    loop = PlannerLoop(hw, pol, _Snaps(hw), _Ex())
    loop.run(max_replans=3)
    assert len(seen) == 3 and all(p is None for p in seen)
    assert [r["accepted"] for r in loop.trace] == [False, False, False]


# ---------------------------------------------------------------------------
# 5. wiring: CLI flags, provenance, policy-server pass-through
# ---------------------------------------------------------------------------

def _parse(*argv):
    from phantom.scripts.run_deploy import build_parser
    return build_parser().parse_args(["--system", "teacher", "--task", "x", *argv])


def test_the_selector_defaults_to_the_old_rule():
    a = _parse()
    assert (a.select_by, a.agreement_veto) == ("default", None)


def test_the_flags_are_refused_where_the_score_does_not_exist(monkeypatch):
    from phantom.scripts import run_deploy
    from phantom.scripts.run_deploy import main
    monkeypatch.setattr(run_deploy, "_attach_file_log", lambda: None)
    assert main(["--system", "teacher", "--task", "x",
                 "--select-by", "video_agreement"]) == 2
    assert main(["--system", "teacher", "--task", "x", "--select-by",
                 "video_agreement", "--k-seeds", "4", "--drop-video"]) == 2
    assert main(["--system", "teacher", "--task", "x",
                 "--agreement-veto", "0.02"]) == 2
    assert main(["--system", "teacher", "--task", "x", "--select-by",
                 "video_agreement", "--k-seeds", "4",
                 "--agreement-veto", "0"]) == 2


def test_the_selector_is_configurable_on_the_policy_server():
    """PICK attaches to a warm server, so the lever has to travel in the
    client's `configure` call — the server itself takes no new flag."""
    from phantom.inference.remote import CONFIGURABLE, PolicyServer
    assert "select_by" in CONFIGURABLE and "agreement_veto" in CONFIGURABLE

    pol = PhantomPolicy.__new__(PhantomPolicy)
    pol.select_by, pol.agreement_veto, pol._k_seeds = "default", None, 4
    srv = PolicyServer(SimpleNamespace(policy_kind="phantom",
                                       wrench_baseline_rows=0, select_by="default",
                                       agreement_veto=None), ckpt="x.pt", ckpt_sha="a")
    srv.policy = pol
    out = srv.handle(("configure", {"select_by": "video_agreement",
                                    "agreement_veto": 0.02}))
    assert out["effective"]["select_by"] == "video_agreement"
    assert out["effective"]["agreement_veto"] == pytest.approx(0.02)


def test_an_unset_threshold_disarms_a_warm_server():
    """nfe=None means 'the checkpoint default', but agreement_veto=None means
    OFF: a client that does not ask for the veto must not inherit the previous
    arm's threshold from a server that stayed warm between cells."""
    from phantom.inference.remote import RESET_WHEN_NONE
    import inspect
    from phantom.inference import remote
    assert "agreement_veto" in RESET_WHEN_NONE
    src = inspect.getsource(remote.RemotePolicy.__init__)
    assert "RESET_WHEN_NONE" in src

    pol = PhantomPolicy.__new__(PhantomPolicy)
    pol.select_by, pol.agreement_veto, pol._k_seeds = "video_agreement", 0.02, 4
    srv = remote.PolicyServer(SimpleNamespace(policy_kind="phantom",
                                              wrench_baseline_rows=0), ckpt="x", ckpt_sha="a")
    srv.policy = pol
    out = srv.handle(("configure", {"select_by": "default", "agreement_veto": None}))
    assert out["effective"]["agreement_veto"] is None
    assert out["effective"]["select_by"] == "default"


def test_a_lerobot_server_refuses_the_selector():
    """The pi0.5 adapter imagines no video: configure would set the attribute,
    the adapter would ignore it and the trace would still say sel:... ."""
    import inspect
    from phantom.scripts import run_deploy
    src = inspect.getsource(run_deploy.main)
    assert 'policy.info.get("policy_kind") == "lerobot"' in src
    assert 'args.select_by != "default"' in src


def test_the_selector_reaches_the_condition_tags_and_the_overrides():
    """An arm that ran the new selector must be reconstructable from the
    recording alone, like every other deploy lever."""
    import inspect
    from phantom.scripts import run_deploy
    src = inspect.getsource(run_deploy)
    assert 'f"sel:{getattr(args, \'select_by\', \'default\')}"' in src
    assert '"aveto:off"' in src
    assert 'deploy_overrides["select_by"]' in src


def test_the_policy_constructor_takes_the_selector():
    import inspect
    sig = inspect.signature(PhantomPolicy.__init__)
    assert sig.parameters["select_by"].default == "default"
    assert sig.parameters["agreement_veto"].default is None
