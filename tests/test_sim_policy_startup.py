"""Startup observations must not replace an existing gripper drive target."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_sim_policy_adapter import POSE, adapter, observe, plan, tick

from phantom.deploy.executor import ChunkExecutor


def test_warmup_and_pending_inference_preserve_measured_feedback_without_commands():
    ad = adapter()
    for t, measured in ((0.0, 0.452525), (0.128, 0.4526), (0.256, 0.4527)):
        observe(ad, t, grip=measured, ft=np.arange(6) * 0.1)
        assert ad.step(t) is None
        assert ad._awaiting_feedback is None
        assert ad.gripper_cmd_at([t]) is None
        assert ad.entered_grip_after(-1) == []
        assert ad._grip_latch is None
        snap = ad.snapshot()
        assert snap.ur_state[24] == np.float32(measured)
        np.testing.assert_allclose(snap.wrist_window[-1], np.arange(6) * 0.1)
        if snap.prev_chunk is not None:
            # Native prehistory fallback uses measured closure, with no
            # fabricated warmup command in the action history.
            np.testing.assert_array_equal(snap.prev_chunk[:, 6], np.float32(measured))
    with pytest.raises(ValueError, match="pending command timestamp"):
        ad.report_execution(0.256, accepted=True, gripper_command=0.4527)
    ad.replan(t=0.256)
    observe(ad, 0.448, grip=0.4528)
    assert ad.step(0.448) is None
    observe(ad, 0.456, grip=0.4529)
    first = tick(ad, 0.456)
    assert first.diagnostics["plan_activated"]
    assert first.gripper == 0.4
    assert ad.entered_grip_after(-1) == [(0.456, 0.4)]
    assert ad.snapshot().ur_state[24] == np.float32(0.4529)
    # Reset must reinstate no-command startup for the next trial.
    ad.reset()
    observe(ad, grip=0.08)
    assert ad.step(0) is None
    assert ad.gripper_cmd_at([0]) is None


def test_rejected_first_plan_still_does_not_submit_a_hold():
    ad = adapter()
    observe(ad)
    expired = plan(t=-10)
    ad._pending = (0.0, expired, ad.snapshot())
    assert ad.step(0) is None
    assert ad._pending is None and ad._plan is None
    assert ad.gripper_cmd_at([0]) is None


def test_native_executor_leaves_existing_gripper_mailbox_untouched_without_plan(monkeypatch):
    native = ChunkExecutor.__new__(ChunkExecutor)
    native.hw = adapter().hw
    native._lock = nullcontext()
    native._plan = None
    native._grip_target = 0.4274509847164154
    stop_checks = iter([False, False, True])
    native._stop = SimpleNamespace(is_set=lambda: next(stop_checks))
    sleeps = []
    monkeypatch.setattr("phantom.deploy.executor.time.sleep", sleeps.append)
    native._run()
    assert native._grip_target == 0.4274509847164154
    assert sleeps == [1 / native.hw.control.executor_rate_hz] * 2


def test_runner_keeps_all_drive_references_without_ik_or_execution_feedback():
    # Execute the runner's actual post-step dispatch block without importing
    # Isaac. An absent command must not invoke IK, map measured closure into
    # finger targets, alter the passive springs, or create an execution row.
    source = Path(__file__).resolve().parents[1] / "tools/sim/run_waffles.py"
    tree = ast.parse(source.read_text())
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "command is not None"
                 and any(isinstance(child, ast.Name) and child.id == "pending_execution"
                         for child in ast.walk(node)))
    # Preserve the runner's lexical binding for its nested verified-submit
    # callback. Compiling this guard at module scope rejects its nonlocal;
    # never strip or rewrite that statement merely to make extraction pass.
    scope = ast.parse(
        "def dispatch(command, desired, pending_execution):\n"
        "    submitted_this_tick = False\n"
        "    return desired, pending_execution, submitted_this_tick\n"
    ).body[0]
    scope.body.insert(1, guard)
    module = ast.fix_missing_locations(ast.Module(body=[scope], type_ignores=[]))
    desired = np.array([1., 2., 3., 4., 5., 6., .38, 2.62, 0., .38, 2.62, 0., 0., 0.])
    original = desired.copy()
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    returned, pending, submitted = namespace["dispatch"](None, desired, None)
    np.testing.assert_array_equal(desired, original)
    assert returned is desired and pending is None and submitted is False


def test_startup_safety_stop_remains_an_explicit_command():
    ad = adapter()
    observe(ad, retain_rgb=True)
    command = tick(ad, 0)
    assert command.stopped and command.reason == "camera_scene_stale"
    np.testing.assert_array_equal(command.tcp_pose, POSE)
