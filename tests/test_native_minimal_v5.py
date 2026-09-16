"""Opt-in native historical controller parity; no model, GPU or hardware calls."""

import argparse
import ast
import json
import logging
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from phantom_test_utils import make_small_hw
from test_sim_placement_release import INSIDE, observe, plan
from test_sim_teacher_anchor_finish import adapter as frozen_adapter
from test_sim_teacher_anchor_finish import module
from test_sim_teacher_anchor_finish import trees as _frozen_trees

from phantom.deploy import executor as executor_module
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.minimal_v5 import (
    planner_class,
    validate_profile,
)
from phantom.deploy.release_controller import (
    PlacementReleaseConfig,
    PlacementReleaseController,
    restore_policy_openings,
)
from phantom.deploy.safety import SafetyAction, SafetyMonitor

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/teacher_anchor_v1"
VETO = {
    "p_close": 0.5,
    "p_none": 0.9,
    "max_retries": 3,
    "z_floor": 0.0315,
    "z_ref": 0.0415,
    "z_margin": 0.0615,
    "open_aperture": 0.232,
    "close_pos": 0.45,
    "close_rise": 0.15,
}
RELEASE = PlacementReleaseConfig(
    tcp_min_m=(-0.55, -0.076, 0.058),
    tcp_max_m=(-0.23, 0.149, 0.274),
    finish_after_release=True,
)


@pytest.fixture
def trees(tmp_path):
    return _frozen_trees.__wrapped__(tmp_path)


def native_planner(clock=None):
    """Execute the actual checked-in class without its unrelated torch imports."""
    clock = clock or SimpleNamespace(perf_counter=lambda: 1.0, sleep=lambda _: None)
    tree = ast.parse((ROOT / "phantom/deploy/planner.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlannerLoop"
    )
    namespace = {
        "np": np,
        "threading": threading,
        "time": clock,
        "log": logging.getLogger(__name__),
        "snapshot_stop_reason": lambda e: "snapshot_invalid",
        "VETO_REWRITE_ACTIONS": ("close_masked", "recovery_open", "recovery_tactile"),
    }
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 - trusted checked-in control AST
        compile(ast.fix_missing_locations(code), "native PlannerLoop", "exec"),
        namespace,
    )
    return namespace["PlannerLoop"]


def state():
    return {
        "closed_idx": None,
        "retries": 0,
        "g_min": None,
        "in_close": False,
        "grip_seen_t": float("-inf"),
        "close_permitted": False,
        "allowed_floor_only": False,
        "closed_floor_only": False,
    }


def test_opt_in_is_explicit_and_default_class_identity_is_unchanged():
    parent = native_planner()
    assert planner_class(None, parent) is parent
    assert not hasattr(parent, "request_snapshot_veto")
    assert (
        validate_profile(None, release_config=None, veto=None, mode="student", hw=None)
        is None
    )
    for kwargs in [
        {"mode": "replay"},   # sim zoo 2026-09-12: student and vision_only are supported modes now
        {"veto": None},
        {"release_config": None},
        {
            "release_config": PlacementReleaseConfig(
                tcp_min_m=(-1, -1, 0), tcp_max_m=(0, 0, 1)
            )
        },
    ]:
        args = {
            "release_config": RELEASE,
            "veto": SimpleNamespace(**VETO),
            "mode": "teacher",
            "hw": make_small_hw(),
        }
        args.update(kwargs)
        with pytest.raises(ValueError):
            validate_profile("minimal_v5", **args)
    metadata = validate_profile(
        "minimal_v5",
        release_config=RELEASE,
        veto=SimpleNamespace(**VETO),
        mode="teacher",
        hw=make_small_hw(),
    )
    assert metadata["veto_feedback_source"] == "request_snapshot_historical"
    assert metadata["effective_veto"] == VETO
    assert metadata["effective_release"]["finish_after_release"] is True
    with pytest.raises(ValueError):
        planner_class("unknown", parent)


def test_historical_method_matches_immutable_fixture_ast():
    def method(path):
        return next(
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "_apply_veto_historical"
        )

    assert ast.dump(
        method(ROOT / "phantom/deploy/minimal_v5.py"), include_attributes=False
    ) == ast.dump(method(FIXTURES / "deployment_filters.py"), include_attributes=False)


def test_all_37_saved_delivered_plans_match_frozen_historical_bridge(monkeypatch):
    fixture = json.loads((FIXTURES / "saved_native_port_plans.json").read_text())
    frozen = module(monkeypatch, "frozen_port_veto", FIXTURES / "deployment_filters.py")
    hw = make_small_hw()
    filter_ = frozen.TerminalVetoFilter(hw, VETO, implementation="fd4a032")
    feedback = SimpleNamespace(t=0.0, pose=np.zeros(6))
    controller = PlacementReleaseController(RELEASE)

    def mask(values):
        return (
            np.isfinite(values)
            & (values <= 0.45)
            & controller.window_active(feedback.pose)
        )

    ex = SimpleNamespace(
        release_controller=controller,
        placement_release_opening_mask=mask,
        rings={},
        clear_grip_latch=lambda: pytest.fail(
            "historical recovery must not clear native latch"
        ),
        request_stop=lambda _: pytest.fail("saved run did not reach retry cap"),
        entered_grip_after=lambda t: [
            (ts, g) for ts, g in fixture["accepted_grip_history"] if t < ts < feedback.t
        ],
    )
    native = planner_class("minimal_v5", native_planner())(
        hw, None, None, ex, veto=SimpleNamespace(**VETO)
    )
    vstate = state()
    actions_seen = set()
    for row in fixture["samples"]:
        feedback.t = row["delivery_t"]
        feedback.pose = np.array(row["delivery_tcp"])
        ex._last_observe_t = feedback.t
        controller.phase = row["prior_release_phase"]
        proposal = SimpleNamespace(
            actions=np.array(row["actions"], dtype=np.float32),
            p_evt=np.array(row["p_evt"], dtype=np.float32),
            cpk="package",
            _cpk_token=42,
            diag={},
        )
        original = proposal.actions.copy()
        snap = SimpleNamespace(
            t=row["request_t"], ur_state=np.array(row["ur_state"], dtype=np.float32)
        )
        sim = filter_(deepcopy(proposal), snap, ex)
        rec = native._apply_veto(
            proposal,
            snap.ur_state[12:18],
            float(snap.ur_state[-2]),
            vstate,
            row["replan_id"],
        )
        restore_policy_openings(proposal, original, rec, ex)
        np.testing.assert_array_equal(proposal.actions, sim.actions)
        np.testing.assert_array_equal(
            proposal.actions, np.array(row["delivered_actions"], dtype=np.float32)
        )
        assert (
            rec["action"]
            == row["veto"]["action"]
            == sim.diag["terminal_veto"]["action"]
        )
        assert rec["retries"] == sim.diag["terminal_veto"]["retries"]
        assert rec.get("placement_release_passthrough_indices") == sim.diag[
            "terminal_veto"
        ].get("placement_release_passthrough_indices")
        assert (proposal.cpk, proposal._cpk_token) == (sim.cpk, sim._cpk_token)
        actions_seen.add(rec["action"])
    assert len(fixture["samples"]) == 37
    assert actions_seen == {"none", "close_allowed", "close_masked", "recovery_open"}


@pytest.mark.parametrize(
    "profile,expected", [("minimal_v5", "close_masked"), (None, "close_allowed")]
)
def test_actual_planner_run_uses_request_feedback_only_for_opt_in(profile, expected):
    hw = make_small_hw()
    now = [0.0]
    clock = SimpleNamespace(
        perf_counter=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt)
    )
    request_pose = INSIDE.copy()
    request_pose[2] = 0.2
    current_pose = INSIDE.copy()
    current_pose[2] = 0.07
    ur = np.zeros(hw.ur_state_dim)
    ur[12:18] = request_pose
    ur[-2] = 0.2
    snap = SimpleNamespace(t=0.0, ur_state=ur)
    saved = deepcopy(ur)
    proposal = plan(0, 0.65)
    proposal.p_evt = np.array([0.99, 0.01, 0, 0, 0])
    original = proposal.actions.copy()
    controller = PlacementReleaseController(RELEASE)
    feedback_calls = []
    submitted = []

    def feedback():
        feedback_calls.append(True)
        return current_pose, 0.35, now[0]

    def infer(obs, *_):
        assert obs is snap
        np.testing.assert_array_equal(obs.ur_state, saved)
        now[0] = 0.8
        return deepcopy(proposal)

    ex = SimpleNamespace(
        stopped_reason=None,
        completed_reason=None,
        release_controller=controller,
        _release_feedback=feedback,
        last_cmd=lambda: request_pose.copy(),
        entered_grip_after=lambda _t: [],
        submit=lambda p: submitted.append(p) or True,
        placement_release_opening_mask=lambda g: np.zeros(len(g), bool),
        release_diagnostics=lambda: {"phase": "unarmed"},
    )
    cls = planner_class(profile, native_planner(clock))
    veto = frozen_veto_config()
    loop = cls(
        hw,
        SimpleNamespace(replan=infer),
        SimpleNamespace(build=lambda: snap),
        ex,
        veto=veto,
    )
    loop.run(max_replans=1)
    assert loop.trace[0]["terminal_veto"]["action"] == expected
    assert len(feedback_calls) == (0 if profile else 1)
    np.testing.assert_array_equal(snap.ur_state, saved)
    if profile:
        np.testing.assert_array_equal(loop.trace[0]["actions_pre_veto"], original)
        assert submitted[0].cpk is None and submitted[0]._cpk_token is None
        assert loop.trace[0]["terminal_veto"]["feedback_gripper"] == 0.2
    else:
        np.testing.assert_array_equal(submitted[0].actions, original)


def frozen_veto_config():
    # Extra fields are only read by unchanged default live rules.
    return SimpleNamespace(
        **VETO, phantom_window=1.0, phantom_f=2.5, phantom_dz=0.03, phantom_t=0.3
    )


def test_native_release_sequence_matches_frozen_minimal_adapter_and_safety(
    trees, monkeypatch
):
    _, overlay, _ = trees
    ad = frozen_adapter(monkeypatch, overlay, finish=True)
    safety = SafetyMonitor(ad.hw, ad.rings)
    fake_arm = SimpleNamespace(
        stop=lambda *_: None,
        get_state=lambda: SimpleNamespace(
            q=np.zeros(6), qd=np.zeros(6), tcp_pose=INSIDE
        ),
    )
    ex = ChunkExecutor(
        ad.hw,
        fake_arm,
        SimpleNamespace(),
        safety,
        gripper_ring=ad.rings["gripper"],
        release_config=ad.release_controller.config.to_dict(),
        max_play_steps=10,
    )
    ex._last_cmd = INSIDE.copy()
    ex._last_grip_command = 0.55
    saw = set()
    for i in range(260):
        t = i * 0.008
        load = 0 if i == 0 or i >= 100 else 3.0
        measured = 0.3 if i >= 100 else 0.55
        observe(ad, t, load=load, measured=measured)
        if i in (0, 40):
            p = plan(t, 0.63 if i == 0 else 0.4)
            assert ad.submit(deepcopy(p), t)
            ex._plan = p
        cmd = ad.step(t)
        ex._play_time = ad._play_time
        target, grip = ex._pose_at(ex._plan, ex._play_time)
        verdict = safety.check(t, target)
        assert verdict.action not in (
            SafetyAction.STOP_EPISODE,
            SafetyAction.PROTECTIVE_STOP,
        )
        target, grip = ex._apply_release_control(t, target, grip, ex._plan, False)
        np.testing.assert_array_equal(target, cmd.tcp_pose)
        assert grip == cmd.gripper
        assert ex.release_controller.phase == ad.release_controller.phase
        assert ex._grip_latch == ad._grip_latch
        saw.add(ex.release_controller.phase)
        # Ideal acknowledged mailbox: no I/O delay is invented by this test.
        ex._last_grip_command = grip
        ad.report_execution(
            t, accepted=True, tcp_pose=cmd.tcp_pose, gripper_command=cmd.gripper
        )
        if ex.completed_reason:
            break
    assert {"holding", "releasing", "finished"} <= saw
    assert ex.completed_reason == ad.completed_reason == "placement_release_finished"
    assert ex.completed_at_s == ad.completed_at_s
    np.testing.assert_array_equal(ex._finish_pose, INSIDE)
    assert ex._finish_grip == 0.4 and not cmd.stopped
    # Real native loop checks safety before issuing a FINISH hold command.
    next_t = t + 0.008
    observe(ad, next_t, load=0, measured=0.3, qd=np.ones(6) * 3)
    stopped = ad.step(next_t)
    assert stopped.stopped and "joint_speed" in stopped.diagnostics["safety_events"]
    fake_arm.servo_l = lambda *_: pytest.fail("safety must preempt a new hold command")
    monkeypatch.setattr(executor_module.time, "perf_counter", lambda: next_t)
    ex._run()
    assert ex.stopped_reason is not None
    assert ex.completed_reason == "placement_release_finished"
    assert ex.release_controller.last_event == "safety_preempted"


def test_cli_parser_defaults_and_explicit_profile_flag():
    source = ast.parse((ROOT / "phantom/scripts/run_deploy.py").read_text())
    node = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_parser"
    )
    namespace = {
        "argparse": argparse,
        "Path": Path,
        "SYSTEM_MODES": ("teacher", "student"),
        "DEFAULT_MAX_REPLANS": 40,
        # phantom.inference.policy.NULL_IMAGINATION_MODES, mirrored here for
        # the same reason SYSTEM_MODES is: build_parser is exec'd in isolation
        "NULL_IMAGINATION_MODES": ("none", "prev_cpk", "contact_zero"),
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    }
    exec(  # noqa: S102 - trusted checked-in control AST
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "actual CLI parser",
            "exec",
        ),
        namespace,
    )
    parser = namespace["build_parser"]()
    base = ["--system", "teacher", "--task", "waffles"]
    assert parser.parse_args(base).placement_controller_profile is None
    assert (
        parser.parse_args(
            base + ["--placement-controller-profile", "minimal_v5"]
        ).placement_controller_profile
        == "minimal_v5"
    )


def test_runtime_profile_validation_precedes_any_rig_construction():
    source = ast.parse((ROOT / "phantom/deploy/runtime.py").read_text())
    cls = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "DeploymentRuntime"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    cls.body = [init]
    calls = []
    parent = native_planner()
    namespace = {
        "PlannerLoop": parent,
        "Path": Path,
        "make_rig": lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace(),
    }
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 - trusted checked-in control AST
        compile(ast.fix_missing_locations(code), "actual runtime constructor", "exec"),
        namespace,
    )
    runtime = namespace["DeploymentRuntime"]
    hw = make_small_hw()
    args = (hw, SimpleNamespace(), "teacher", Path("/unused"))
    with pytest.raises(ValueError, match="terminal-veto"):
        runtime(*args, release_config=RELEASE, controller_profile="minimal_v5")
    assert calls == []
    for aperture in (-0.1, 1.1, hw.gripper.max_close_cmd + 0.001):
        with pytest.raises(ValueError, match="open_aperture"):
            runtime(
                *args,
                release_config=RELEASE,
                veto=SimpleNamespace(**{**VETO, "open_aperture": aperture}),
                controller_profile="minimal_v5",
            )
        assert calls == []
    current = runtime(*args)
    assert current.planner_class is parent
    assert "placement_controller_profile" not in current.deploy_overrides
    selected = runtime(
        *args,
        release_config=RELEASE,
        veto=SimpleNamespace(**VETO),
        controller_profile="minimal_v5",
    )
    assert issubclass(selected.planner_class, parent)
    assert selected.planner_class.request_snapshot_veto is True
    assert (
        selected.deploy_overrides["placement_veto_feedback"]
        == "request_snapshot_historical"
    )
    assert len(calls) == 2


@pytest.mark.parametrize("corrupt", ["actions", "pose", "grip"])
def test_opt_in_rejects_nonfinite_plan_feedback_before_rewriting(corrupt):
    cls = planner_class("minimal_v5", native_planner())
    loop = cls(
        make_small_hw(), None, None, SimpleNamespace(), veto=frozen_veto_config()
    )
    proposal = plan(0, 0.65)
    pose = INSIDE.copy()
    grip = 0.2
    if corrupt == "actions":
        proposal.actions[0, 0] = np.nan
    elif corrupt == "pose":
        pose[0] = np.nan
    else:
        grip = np.nan
    before = state()
    actual = deepcopy(before)
    with pytest.raises(ValueError, match="finite native plan"):
        loop._apply_veto(proposal, pose, grip, actual)
    assert actual == before
    assert proposal.cpk == "contact" and proposal._cpk_token == 123
