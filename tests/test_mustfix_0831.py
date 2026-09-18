"""MUST-FIX batch from docs review REVALIDATION_0831 §2 (2026-08-31).

Each test first REPRODUCES what the revalidation measured on this tree, then
pins the fix.

#1 `--max-episode-s` was dead: `PlannerLoop.run` checks the replan COUNT
   before the wall clock and `--max-replans` defaulted to 40, so the
   documented arm-B line (`--nfe 1`, 172 ms replans, `--max-episode-s 35`)
   stopped at `replan_cap` after 7.2 s against 16-31 s demos.
#8 `--resume` silently rewrote the recipe: `grasp_frac`, `photo_aug`,
   `commit_band_weight` and `split` reached no part of the checkpoint, and
   `ema_decay` was recorded but never re-applied — so a resume with only the
   model flags re-passed reverted the recipe to the dataclass defaults and
   then wrote those defaults into the next checkpoint as if they had governed
   the whole run.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.config.paths import load_paths
from phantom.config.training import TeacherTrainConfig
from phantom.deploy.planner import PlannerLoop
from phantom.deploy.runtime import EpisodeResult
from phantom.inference.policy import ObsSnapshot, Plan
from phantom_test_utils import make_small_hw

# ===========================================================================
# #1 — the wall-clock budget must govern when no replan count was named
# ===========================================================================


def _busy(seconds: float) -> None:
    """Latency that a monkeypatched `time.sleep` cannot erase (the CLI harness
    stubs out sleep so the operator prompts do not stall the test)."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


class _Ex:
    """Executor stub: accepts every chunk, plays no steps back."""

    def __init__(self):
        self.stopped_reason = None
        self.submitted: list[np.ndarray] = []

    def last_cmd(self):
        return None

    def request_stop(self, reason):
        if self.stopped_reason is None:
            self.stopped_reason = reason

    def submit(self, plan):
        self.submitted.append(np.array(plan.actions, copy=True))
        return True

    def entered_grip_after(self, t):
        return []


class _Snaps:
    def __init__(self, hw):
        self.hw = hw

    def build(self):
        hw = self.hw
        s = ObsSnapshot(t=time.perf_counter(),
                        rgb=np.zeros((4, 4, 3), np.uint8),
                        wrist_window=np.zeros((hw.wrist_ft.window_len, 6), np.float32),
                        ur_state=np.zeros(2 * hw.arm.dof + 14, np.float32))
        s.ur_state[2 * hw.arm.dof + 2] = 0.25
        s.ur_state[-2] = 0.30
        return s


class _Pol:
    """A replan that costs `latency_s` of real wall clock (the `--nfe 1` lever
    buys 172 ms; the default cap of 40 makes that a 7 s episode)."""

    def __init__(self, hw, latency_s):
        self.hw, self.latency_s = hw, latency_s

    def replan(self, snap, prev_plan, tcp_pose):
        _busy(self.latency_s)
        H, A = self.hw.control.chunk_horizon, self.hw.control.action_dim
        a = np.zeros((H, A))
        a[:, 6] = 0.30
        p = Plan(t_created=time.perf_counter(),
                 t0_pose=np.asarray(tcp_pose, float).copy(), actions=a,
                 action_times=np.arange(H) / 10.0, sigma=np.zeros(3), gate=0.0,
                 p_evt=np.array([0.5, 0.5, 0, 0, 0]), cpk=None,
                 latency_s=self.latency_s)
        p.diag = {}
        return p


class _StubArm:
    def is_ready_for_control(self):
        return True, ""

    def reconnect_control(self):
        pass

    def program_running(self):
        return True

    def get_state(self):
        return SimpleNamespace(tcp_pose=np.zeros(6))


def _run_main_over_a_real_loop(monkeypatch, tmp_path, argv, *, latency_s=0.002):
    """run_deploy.main end to end with every device stubbed, but a REAL
    PlannerLoop inside run_episode — so the stop reason is the planner's."""
    from phantom.deploy import start_pose as sp
    from phantom.scripts import run_deploy as RD

    hw = make_small_hw(mode={"drivers": "real"})
    seen: dict = {}

    class _StubRuntime:
        def __init__(self, hw_, policy, mode, out_root, **kw):
            self.rig = SimpleNamespace(
                arm=_StubArm(),
                gripper=SimpleNamespace(get_state=lambda: SimpleNamespace(
                    position=0.0, obj=3.0)))
            self.recorder = SimpleNamespace(relabel=lambda *a, **k: None)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run_episode(self, **kw):
            seen.update(kw)
            loop = PlannerLoop(hw, _Pol(hw, latency_s), _Snaps(hw), _Ex(), veto=None)
            loop.run(max_replans=kw["max_replans"],
                     max_episode_s=kw["max_episode_s"])
            seen["stop_reason"] = loop.stop_reason
            seen["n_replans"] = len(loop.trace)
            return EpisodeResult(episode_path=None, stopped_reason=loop.stop_reason,
                                 n_replans=len(loop.trace), safety_events=0,
                                 trace_path=None)

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
    assert RD.main(["--system", "teacher", "--task", "waffles", "--tiny",
                    "--episodes", "1", "--nfe", "1", *argv]) == 0
    return seen


def test_the_wall_clock_budget_governs_when_no_replan_count_was_named(
        monkeypatch, tmp_path):
    """Reproduction: at 2 ms replans the default cap of 40 ended the episode
    in 0.08 s with `stop='replan_cap'`, and `--max-episode-s` never fired —
    the arm-B failure, scaled down. The budget must be the binding cap."""
    seen = _run_main_over_a_real_loop(monkeypatch, tmp_path,
                                      ["--max-episode-s", "0.3"])
    assert seen["max_replans"] is None, seen["max_replans"]
    assert seen["stop_reason"] == "episode_time_cap", seen
    assert seen["n_replans"] > 40, seen["n_replans"]


def test_an_explicit_replan_cap_still_wins_over_the_budget(monkeypatch, tmp_path):
    seen = _run_main_over_a_real_loop(
        monkeypatch, tmp_path, ["--max-episode-s", "30", "--max-replans", "3"])
    assert seen["max_replans"] == 3
    assert seen["stop_reason"] == "replan_cap" and seen["n_replans"] == 3


def test_the_replan_cap_survives_when_the_budget_is_disabled(monkeypatch, tmp_path):
    """`--max-episode-s 0` disables the clock, so the count is all there is."""
    seen = _run_main_over_a_real_loop(monkeypatch, tmp_path,
                                      ["--max-episode-s", "0"], latency_s=0.0)
    assert seen["max_episode_s"] is None
    assert seen["max_replans"] == 40 and seen["stop_reason"] == "replan_cap"


def test_resolve_max_replans_is_the_whole_rule():
    from phantom.scripts.run_deploy import (DEFAULT_MAX_REPLANS, build_parser,
                                            resolve_max_replans)
    ap = build_parser()

    def r(argv):
        return resolve_max_replans(ap.parse_args(
            ["--system", "teacher", "--task", "waffles", *argv]))

    assert r([]) is None                                  # 35 s default budget
    assert r(["--max-episode-s", "35"]) is None
    assert r(["--max-episode-s", "0"]) == DEFAULT_MAX_REPLANS
    assert r(["--max-episode-s", "-1"]) == DEFAULT_MAX_REPLANS
    assert r(["--max-replans", "200"]) == 200
    assert r(["--max-episode-s", "0", "--max-replans", "20"]) == 20
    # the arm-B line the recipe documents, with the stopgap no longer needed
    assert r(["--nfe", "1", "--max-episode-s", "35"]) is None


# ===========================================================================
# #8 — --resume restores the recipe it was given, or refuses
# ===========================================================================

def _cosmos_available() -> bool:
    try:
        from phantom.backbone import loader as bl
        bl.setup_cosmos(load_paths())
        return True
    except Exception:                                        # noqa: BLE001
        return False


needs_cosmos = pytest.mark.skipif(not _cosmos_available(),
                                  reason="cosmos repo not importable")

#: the recipe the base run below is launched with — none of it is model config
RECIPE = ["--ema-decay", "0.995", "--photo-aug", "0.7",
          "--commit-band-weight", "1.6", "--split", "all",
          "--event-band-weight", "0"]


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _run_cli(runs_root, argv) -> list[str]:
    from phantom.train import train_teacher as TT
    p = dataclasses.replace(load_paths(), runs_root=Path(runs_root))
    orig, h = TT.load_paths, _Records()
    TT.load_paths = lambda: p
    root = logging.getLogger()
    prev = root.level
    root.addHandler(h)
    root.setLevel(logging.INFO)
    try:
        assert TT.main(argv) == 0
    finally:
        TT.load_paths = orig
        root.removeHandler(h)
        root.setLevel(prev)
    return h.msgs


TINY = ["--tiny", "--synthetic", "--device", "cpu", "--acc-two-pass"]


@pytest.fixture(scope="module")
def runs_root(tmp_path_factory):
    return tmp_path_factory.mktemp("mustfix0831_runs")


@pytest.fixture(scope="module")
def recipe_ckpt(runs_root):
    """A 1-step checkpoint that carries the whole RECIPE in configs.train."""
    if not _cosmos_available():
        pytest.skip("cosmos repo not importable")
    _run_cli(runs_root, [*TINY, "--max-steps", "1", "--ckpt-every", "1",
                         "--run-name", "mf0831_base", *RECIPE])
    ck = Path(runs_root) / "teacher" / "mf0831_base" / "teacher_000001.pt"
    assert ck.exists()
    return ck


def test_the_recipe_knobs_are_train_config_fields():
    """Reproduction: `grasp_frac`, `photo_aug`, `commit_band_weight` and
    `split` were read straight off `args` and reached no part of the
    checkpoint, so `configs.train` had no record of them at all."""
    import argparse

    from phantom.train.train_teacher import apply_overrides

    d = TeacherTrainConfig().to_dict()
    for k, default in (("split", "train"), ("grasp_frac", 0.0),
                       ("photo_aug", 0.0), ("commit_band_weight", 1.0),
                       ("ema_decay", 0.999)):
        assert d[k] == default, k

    ap = argparse.ArgumentParser()
    from phantom.train.train_teacher import add_common_args
    add_common_args(ap)
    for flag, kind in (("--split", str), ("--grasp-frac", float),
                       ("--photo-aug", float), ("--commit-band-weight", float),
                       ("--ema-decay", float)):
        ap.add_argument(flag, type=kind, default=None)
    # unspoken stays the dataclass default ...
    base = apply_overrides(TeacherTrainConfig(), ap.parse_args([]))
    assert (base.split, base.grasp_frac, base.photo_aug,
            base.commit_band_weight) == ("train", 0.0, 0.0, 1.0)
    # ... and a named value lands in configs.train
    cfg = apply_overrides(TeacherTrainConfig(), ap.parse_args(
        ["--split", "all", "--grasp-frac", "0.4", "--photo-aug", "0.7",
         "--commit-band-weight", "1.6"]))
    got = cfg.to_dict()
    assert (got["split"], got["grasp_frac"], got["photo_aug"],
            got["commit_band_weight"]) == ("all", 0.4, 0.7, 1.6)


def test_the_hid_programs_carry_the_data_recipe_too():
    """`apply_overrides` is shared. Until 2026-09-18 HIDConfig had NO
    data-recipe fields, so distill_hid read `--grasp-frac` / `--photo-aug` /
    `--commit-band-weight` / `--split` straight off `args` and a `--resume`
    that forgot them reverted the window recipe with no warning (code-defect
    review (c)). The fields now live on HIDConfig with the SAME defaults, so
    the lock below covers them for the student as well."""
    from phantom.config.training import HIDConfig
    from phantom.train.train_teacher import apply_overrides
    d = HIDConfig().to_dict()
    assert (d["split"], d["grasp_frac"], d["photo_aug"],
            d["commit_band_weight"]) == ("train", 0.0, 0.0, 1.0)
    args = SimpleNamespace(synthetic=False, tiny=True, device="cpu",
                           run_name=None, max_steps=None, batch_size=None,
                           grad_accum=None, num_workers=None, split=None)
    cfg = apply_overrides(HIDConfig(), args)
    assert cfg.to_dict()["run_name"] == "hid" and cfg.split == "train"
    # ... and a named value still lands in configs.train
    args.split = "all"
    assert apply_overrides(HIDConfig(), args).split == "all"


@needs_cosmos
def test_resume_restores_every_recipe_key_the_cli_did_not_name(recipe_ckpt,
                                                              runs_root,
                                                              monkeypatch):
    """Reproduction (revalidation §2 #8): resuming with only the model flags
    gave `ema_decay 0.999` (checkpoint 0.995), `photo_aug 0.0`,
    `commit_band_weight 1.0` — and the NEW checkpoint recorded the reverted
    values as if they had held for the whole run."""
    import torch

    from phantom.train import common as C

    base = torch.load(str(recipe_ckpt), map_location="cpu", weights_only=False)
    saved = base["configs"]["train"]
    assert (saved["ema_decay"], saved["photo_aug"], saved["commit_band_weight"],
            saved["split"], saved["event_band_weight"]) == (
        0.995, 0.7, 1.6, "all", 0.0)

    decays: list[float] = []

    class _EMA(C.EMA):
        def __init__(self, model, decay):
            decays.append(float(decay))
            super().__init__(model, decay)

    monkeypatch.setattr(C, "EMA", _EMA)
    msgs = _run_cli(runs_root, [*TINY, "--max-steps", "2", "--ckpt-every", "2",
                                "--run-name", "mf0831_resume",
                                "--resume", str(recipe_ckpt)])
    # the decay the run actually trained with, not the dataclass default
    assert decays == [0.995], decays
    assert any("ema_decay" in m and "restored from the checkpoint" in m
               for m in msgs), msgs
    assert any("event-band weight" in m and "restored from the checkpoint" in m
               for m in msgs), msgs

    out = Path(runs_root) / "teacher" / "mf0831_resume" / "teacher_000002.pt"
    got = torch.load(str(out), map_location="cpu",
                     weights_only=False)["configs"]["train"]
    # the checkpoint written AFTER the resume records what governed it
    for k in ("ema_decay", "photo_aug", "commit_band_weight", "split",
              "grasp_frac", "event_band_weight", "lr", "lr_new_modules"):
        assert got[k] == saved[k], (k, got[k], saved[k])
    # ... while the operational keys are free to change
    assert got["max_steps"] == 2 and got["run_name"] == "mf0831_resume"


@needs_cosmos
@pytest.mark.parametrize("flag,value,label", [
    ("--ema-decay", "0.999", "ema_decay"),
    ("--photo-aug", "0.0", "photo_aug"),
    ("--commit-band-weight", "1.0", "commit_band_weight"),
    ("--split", "train", "split"),
    ("--event-band-weight", "0.5", "event-band weight"),
    ("--grasp-frac", "0.4", "grasp_frac"),
])
def test_resume_refuses_an_explicitly_different_recipe_value(
        recipe_ckpt, runs_root, flag, value, label):
    from phantom.train import train_teacher as TT
    p = dataclasses.replace(load_paths(), runs_root=Path(runs_root))
    orig, TT.load_paths = TT.load_paths, lambda: p
    try:
        with pytest.raises(SystemExit, match=f"{label} drift"):
            TT.main([*TINY, "--max-steps", "2", "--run-name", "mf0831_refuse",
                     "--resume", str(recipe_ckpt), flag, value])
    finally:
        TT.load_paths = orig


@needs_cosmos
def test_resume_accepts_the_recipe_passed_back_verbatim(recipe_ckpt, runs_root):
    msgs = _run_cli(runs_root, [*TINY, "--max-steps", "2", "--ckpt-every", "2",
                                "--run-name", "mf0831_same",
                                "--resume", str(recipe_ckpt), *RECIPE])
    assert not any("drift" in m for m in msgs), msgs
