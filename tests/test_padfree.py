"""Pad-free deploy (rig 09-18): `--pad-free` never opens, reads or records the
tactile pads, so a gripper WITHOUT pads runs and no pad-driven safeguard acts."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.deploy.runtime import DeploymentRuntime
from phantom_test_utils import make_small_hw
from test_p9_p10_fixes import _CrawlPolicy


def _tactile_keys(rings):
    return sorted(k for k in rings if k.startswith("tactile_"))


def test_padfree_episode_runs_with_no_pad_worker_ring_or_stream(tmp_path):
    hw = make_small_hw()
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="student", out_root=tmp_path,
                           pad_free=True) as rt:
        assert rt.session.tactile_workers == []
        assert _tactile_keys(rt.session.rings) == []
        assert rt.session.all_alive()
        res = rt.run_episode(task="whiteboard", max_replans=2, policy_name="student")
    assert res.episode_path is not None and res.stopped_reason != "worker_died"
    recorded = sorted(p.name for p in res.episode_path.iterdir())
    assert not [n for n in recorded if "tactile" in n], recorded
    meta = json.loads((res.episode_path / "meta.json").read_text())
    assert meta["deploy_overrides"]["pad_mode"] == "padfree"


def test_padfree_episode_runs_with_the_rig_levers_on(tmp_path):
    """The rig preset (PICK.sh preset 1 + SESSION_EXTRA): terminal veto, parity
    fixes, the descend-then-release supervisor and the aperture latch threshold
    all configured — every one of them must tolerate the absent pad rings."""
    from phantom.deploy.descend_then_release import DescendThenReleaseConfig
    from phantom.deploy.planner import TerminalVeto
    hw = make_small_hw()
    assert hw.safety.grip_latch_fz_n > 0          # the latch is configured, just never armed
    veto = TerminalVeto(p_close=0.5, p_none=0.9, max_retries=3, z_floor=0.10,
                        z_margin=0.015, open_aperture=0.25)
    descent = DescendThenReleaseConfig(release_z_m=0.081, release_y_min_m=0.0,
                                       task="whiteboard")
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="student", out_root=tmp_path,
                           pad_free=True, veto=veto, parity_fixes=True,
                           placement_descent=descent, max_play_steps=16,
                           grip_play_steps=10, open_aperture=0.25) as rt:
        res = rt.run_episode(task="whiteboard", max_replans=4, policy_name="student")
    assert res.episode_path is not None
    assert res.stopped_reason not in ("worker_died", "executor_crash", "planner_crash"), res
    assert not [p.name for p in res.episode_path.iterdir() if "tactile" in p.name]


def test_padfree_never_reopens_a_grasp_on_p_none_alone():
    """Replan 0 closes (p_contact 0.8), replan 1 reports p_none 0.95. With pads
    that opens only when the pads are unloaded; pad-free has no pad-load veto,
    so the recovery must NOT rewrite the chunk (09-04: it reopened real grasps)."""
    from test_deploy_levers import VETO, _Ex, _Pol, _loop
    hw = make_small_hw()
    ex = _Ex()
    loop = _loop(hw, _Pol(hw, [0.2, 0.95]), ex, VETO)
    loop.pad_free = True
    loop.run(max_replans=2)
    rec = loop.trace[1]["terminal_veto"]
    assert rec["action"] == "recovery_skipped_padfree" and rec["retries"] == 0
    assert np.allclose(ex.submitted[1][:, 6], 0.8)      # the policy's own closure, untouched
    assert np.allclose(ex.submitted[1][:, 2], 0.01)     # and its lift


def test_the_runtime_hands_pad_free_to_the_planner(tmp_path, monkeypatch):
    import phantom.deploy.runtime as R
    seen = {}
    real = R.PlannerLoop.run
    def run(self, *a, **k):
        seen["pad_free"] = self.pad_free
        return real(self, *a, **k)
    monkeypatch.setattr(R.PlannerLoop, "run", run)
    hw = make_small_hw()
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="student", out_root=tmp_path,
                           pad_free=True) as rt:
        rt.run_episode(task="whiteboard", max_replans=1, policy_name="student")
    assert seen["pad_free"] is True


def test_default_deploy_still_runs_the_pads(tmp_path):
    hw = make_small_hw()
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="student", out_root=tmp_path) as rt:
        assert len(rt.session.tactile_workers) == len(hw.tactile.sensors) > 0
        assert _tactile_keys(rt.session.rings)
        assert "pad_mode" not in rt.deploy_overrides


def test_padfree_is_refused_for_a_pad_input_policy_before_any_device_opens(tmp_path, monkeypatch):
    import phantom.deploy.runtime as R
    monkeypatch.setattr(R, "make_rig", lambda *a, **k: pytest.fail("rig built"))
    hw = make_small_hw()
    with pytest.raises(ValueError, match="pad-free"):
        DeploymentRuntime(hw, _CrawlPolicy(hw), mode="teacher", out_root=tmp_path,
                          pad_free=True)


# --- CLI level: the flag as the rig launches it ------------------------------

def _main(monkeypatch, tmp_path, argv):
    from phantom.deploy import start_pose as sp
    from phantom.deploy.runtime import EpisodeResult
    from phantom.scripts import run_deploy as RD

    hw = make_small_hw(mode={"drivers": "real"})
    seen: dict = {}

    class _StubRuntime:
        def __init__(self, hw_, policy, mode, out_root, **kw):
            seen["runtime_kw"] = kw
            self.rig = SimpleNamespace(
                arm=SimpleNamespace(program_running=lambda: True,
                                    get_state=lambda: SimpleNamespace(tcp_pose=np.zeros(6))),
                gripper=SimpleNamespace(get_state=lambda: SimpleNamespace(
                    position=0.0, obj=3.0)))
            self.recorder = SimpleNamespace(relabel=lambda *a, **k: None)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run_episode(self, **kw):
            seen["episode_kw"] = kw
            return EpisodeResult(episode_path=None, stopped_reason="finish",
                                 n_replans=1, safety_events=0, trace_path=None)

    monkeypatch.setattr(RD, "load_hardware", lambda *a, **k: hw)
    monkeypatch.setattr(RD, "load_paths", lambda *a, **k: SimpleNamespace(
        validate=lambda **k: None, episodes_root=lambda: tmp_path))
    monkeypatch.setattr(RD, "build_policy", lambda *a, **k: SimpleNamespace(
        nfe=1, guidance=1.0, rf=SimpleNamespace()))
    monkeypatch.setattr(RD, "DeploymentRuntime", _StubRuntime)
    monkeypatch.setattr(RD.time, "sleep", lambda *_: None)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    stats = SimpleNamespace(tcp_z_min=0.052, gripper_mean=0.25, task="waffles",
                            tcp_min=None, tcp_max=None)
    monkeypatch.setattr(sp, "load_start_stats", lambda: {"waffles": stats})
    monkeypatch.setattr(sp, "start_sigma_report",
                        lambda *a, **k: (np.zeros(7), "in distribution"))
    monkeypatch.setattr(sp, "move_to_start", lambda *a, **k: (
        np.array([0.111, 0.222, 0.333, 0.0, 0.0, 0.0]), 0.33))
    rc = RD.main(["--task", "waffles", "--tiny", "--episodes", "1", "--nfe", "1", *argv])
    return rc, seen


def test_cli_padfree_reaches_the_runtime_and_tags_the_episode(monkeypatch, tmp_path):
    rc, seen = _main(monkeypatch, tmp_path, ["--system", "student", "--pad-free"])
    assert rc == 0
    assert seen["runtime_kw"]["pad_free"] is True
    assert "padfree:on" in seen["episode_kw"]["tags"]


def test_cli_default_is_not_padfree_and_carries_no_tag(monkeypatch, tmp_path):
    rc, seen = _main(monkeypatch, tmp_path, ["--system", "student"])
    assert rc == 0
    assert seen["runtime_kw"]["pad_free"] is False
    assert "padfree:on" not in seen["episode_kw"]["tags"]


@pytest.mark.parametrize("argv", [
    ["--system", "teacher", "--pad-free"],
    ["--system", "student", "--pad-free", "--lift-complete-z", "0.3"],
    ["--system", "student", "--pad-free", "--placement-release-config",
     "configs/placement_release_descent_sim_waffles.json"],
])
def test_cli_padfree_conflicts_exit_2_before_the_runtime_is_built(monkeypatch, tmp_path, argv):
    rc, seen = _main(monkeypatch, tmp_path, argv)
    assert rc == 2 and "runtime_kw" not in seen


def test_a_padfree_take_never_enters_a_training_index():
    from phantom.data.schema import NON_TRAINING_TAGS, is_trainable_episode
    assert "padfree:on" in NON_TRAINING_TAGS
    meta = SimpleNamespace(tags=["label:stu", "padfree:on"], status="finalized",
                           policy="student", success=True)
    assert not is_trainable_episode(meta)
    meta.tags = ["label:stu"]
    assert is_trainable_episode(meta)
