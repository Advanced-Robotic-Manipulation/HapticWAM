"""Measured release completion shared by the simulator and native executor."""

import ast
import logging
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_sim_placement_release import CONFIG, INSIDE, arm, observe, plan, setup, tick

from phantom.deploy import executor as executor_module
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.release_controller import (
    PlacementReleaseConfig,
    PlacementReleaseController,
    restore_policy_openings,
)
from phantom.deploy.safety import SafetyAction

FINISH = {**CONFIG, "finish_after_release": True, "finish_observation_s": 2.0}


def update(controller, t, *, accepted=0.4, permitted=True, loads=0.0, measured=0.3):
    return controller.update(
        t,
        tcp=INSIDE,
        policy_grip=0.4,
        measured_grip=measured,
        pad_loads={"left": loads, "right": loads},
        eligible=True,
        accepted_grip=accepted,
        finish_permitted=permitted,
    )


def test_finish_requires_committed_release_measured_unload_and_sent_open():
    controller = PlacementReleaseController(PlacementReleaseConfig(**FINISH))
    update(controller, 0)
    update(controller, 1)
    assert not controller.finished
    controller.note_latch(0.63)
    update(controller, 2, loads=3.0, measured=0.55)
    update(controller, 2.2, loads=3.0, measured=0.55)
    assert controller.phase == "releasing"
    update(controller, 2.4)
    update(controller, 2.6, accepted=None)
    update(controller, 2.8, accepted=0.63)
    update(controller, 3.0, permitted=False)
    assert not controller.finished
    # I/O contention may delay finishing; it must not reset measured dwell.
    update(controller, 3.008)
    assert controller.finished and controller.finished_at == 3.008
    controller.note_latch(0.8)
    update(controller, 4, accepted=0.8, loads=3.0, measured=0.7)
    assert controller.finished  # no new task/re-grasp after configured finish
    controller.stop()
    assert controller.last_event == "safety_preempted"
    assert controller.finished_at == 3.008  # retain event history


def test_completed_sim_holds_measured_pose_and_sent_grip_rejects_new_plan():
    ad = setup()
    ad.release_controller = PlacementReleaseController(PlacementReleaseConfig(**FINISH))
    arm(ad)
    assert ad.submit(plan(0.008, 0.4), 0.008)
    for i in range(2, 28):
        tick(ad, i * 0.008)
    for i in range(28, 160):
        # Accepted setpoint may lead actual motion. Finish must freeze the
        # measured pose, not continue moving towards that prior setpoint.
        ad._last_cmd = INSIDE + np.array([0.01, 0, 0, 0, 0, 0])
        cmd = tick(ad, i * 0.008, load=0, measured=0.3)
        if ad.completed_reason:
            break
    assert ad.completed_reason == "placement_release_finished"
    assert ad.completed_at_s == cmd.t
    assert not cmd.stopped and cmd.reason is None
    np.testing.assert_allclose(cmd.tcp_pose, INSIDE, atol=1e-12)
    assert cmd.gripper == pytest.approx(0.4)
    assert not ad.ready_for_replan(cmd.t + 1)
    assert not ad.submit(plan(cmd.t, 0.8), cmd.t)
    with pytest.raises(RuntimeError, match="episode ended"):
        ad.replan(t=cmd.t)
    # Even after a long hold, a measured joint-speed violation overrides it.
    stopped = tick(ad, cmd.t + 0.008, load=0, measured=0.3, qd=np.ones(6) * 3)
    assert stopped.stopped and stopped.reason == "safety_stop"
    assert "joint_speed" in stopped.diagnostics["safety_events"]
    assert stopped.diagnostics["completed_reason"] == "placement_release_finished"


def test_safety_prevents_same_tick_sim_finish():
    ad = setup()
    ad.release_controller = PlacementReleaseController(PlacementReleaseConfig(**FINISH))
    observe(ad, 0, load=0, measured=0.3)
    ad.release_controller.phase = "releasing"
    ad.release_controller.unloaded_since = -1.0
    ad._last_grip = 0.4
    ad.submit(plan(0, 0.4), 0)
    cmd = tick(ad, 0.008, load=0, measured=0.3, qd=np.ones(6) * 3)
    assert cmd.stopped and ad.completed_reason is None
    assert not ad.release_controller.finished


def native():
    ad = setup()
    observe(ad, 0, load=0, measured=0.3)
    fake_arm = SimpleNamespace()
    fake_gripper = SimpleNamespace()
    ex = ChunkExecutor(
        ad.hw,
        fake_arm,
        fake_gripper,
        ad.safety,
        gripper_ring=ad.rings["gripper"],
        release_config=FINISH,
        max_play_steps=10,
    )
    ex._last_cmd = INSIDE.copy()
    ex._plan = plan(0, 0.4)
    ex._last_grip_command = 0.4
    return ex, ad


def test_native_uses_measured_finish_and_nonblocking_mailbox_commit():
    ex, ad = native()
    ex.release_controller.note_latch(0.63)
    ex.release_controller.phase = "releasing"
    ex.release_controller.committed_at = 0.0
    ex.release_controller.unloaded_since = 0.0
    ex._grip_target = 0.8  # an older mailbox entry must be invalidated
    target = INSIDE + np.array([0.02, 0, 0, 0, 0, 0])
    observe(ad, 0.3, load=0, measured=0.3)
    ad.safety.check(0.3, target)
    ex._grip_io_lock.acquire()
    try:
        returned, _ = ex._apply_release_control(0.3, target, 0.8, ex._plan, False)
        np.testing.assert_array_equal(returned, target)
        assert ex.completed_reason is None
    finally:
        ex._grip_io_lock.release()
    observe(ad, 0.308, load=0, measured=0.3)
    ad.safety.check(0.308, target)
    returned, grip = ex._apply_release_control(0.308, target, 0.8, ex._plan, False)
    assert ex.completed_reason == "placement_release_finished"
    np.testing.assert_allclose(returned, INSIDE)
    assert grip == ex._grip_target == 0.4
    assert not ex.submit(plan(0.308, 0.8))


def test_native_stale_gripper_cannot_complete_release():
    ex, _ = native()
    ex.release_controller.phase = "releasing"
    ex.release_controller.unloaded_since = 0.0
    ex._apply_release_control(1.0, INSIDE, 0.4, ex._plan, False)
    assert ex.completed_reason is None
    assert ex.release_controller.unloaded_since is None


def test_native_cannot_finish_at_pose_requiring_workspace_clamp():
    ex, ad = native()
    ex.release_controller.phase = "releasing"
    ex.release_controller.unloaded_since = 0.0
    outside = INSIDE.copy()
    outside[0] = -0.75
    observe(ad, 0.3, tcp=outside, load=0, measured=0.3)
    ad.safety.check(0.3, INSIDE)
    ex._apply_release_control(0.3, INSIDE, 0.4, ex._plan, False)
    assert ex.completed_reason is None


def test_native_worker_rechecks_mailbox_after_lock_handoff():
    ex, _ = native()
    ex._grip_target = 0.8
    sent = []

    class FinishDuringWait:
        def __enter__(self):
            # Worker already captured the old close, then FINISH takes the
            # I/O lock before it and replaces the pending command.
            ex._grip_target = 0.4

        def __exit__(self, *_):
            pass

    def move(grip, *_):
        sent.append(grip)
        ex._stop.set()

    ex._grip_io_lock = FinishDuringWait()
    ex.gripper.move = move
    ex.gripper_ring = None
    ex._grip_worker()
    assert sent == [0.4]
    assert ex.entered_grip_after(float("-inf"))[0][1] == 0.4


def test_shared_mask_restores_policy_openings_only_and_invalidates_cpk():
    ex, _ = native()
    ex.release_controller.note_latch(0.63)
    proposal = plan(0, 0.8)
    proposal.actions[:3, 6] = [0.4, 0.3, 0.2]
    original = proposal.actions.copy()
    proposal.actions[:, 6] = 0.55
    rec = {"action": "close_masked"}
    restore_policy_openings(proposal, original, rec, ex)
    np.testing.assert_array_equal(proposal.actions[:3, 6], original[:3, 6])
    np.testing.assert_array_equal(proposal.actions[3:, 6], 0.55)
    assert rec["placement_release_passthrough_indices"] == [0, 1, 2]
    assert rec["placement_release_variant"] == "placement_policy_release_finish_v2"
    assert proposal.cpk is None and proposal._cpk_token is None
    # Recovery remains authoritative; it cannot create a completion opening.
    unchanged = deepcopy(proposal.actions)
    restore_policy_openings(proposal, original, {"action": "recovery_tactile"}, ex)
    np.testing.assert_array_equal(proposal.actions, unchanged)


def test_native_finished_loop_keeps_safety_active_and_holds_achieved_pose(monkeypatch):
    ex, ad = native()
    ex.completed_reason = "placement_release_finished"
    ex.completed_at_s = 0.0
    ex._finish_pose, ex._finish_grip = INSIDE.copy(), 0.4
    ex._last_cmd = INSIDE + np.array([0.05, 0, 0, 0, 0, 0])
    calls = []

    def servo(target, *_args):
        calls.append(np.array(target))
        ex._stop.set()

    ex.arm.servo_l = servo
    monkeypatch.setattr(executor_module.time, "perf_counter", lambda: 0.008)
    observe(ad, 0.008, load=0, measured=0.3)
    ex._run()
    np.testing.assert_allclose(calls, [INSIDE])
    assert ex._grip_target == 0.4

    ex._stop.clear()
    ex.safety = SimpleNamespace(
        check=lambda *_: SimpleNamespace(action=SafetyAction.STOP_EPISODE, events=[]),
    )
    ex.arm.stop = lambda *_: None
    ex.arm.get_state = lambda: SimpleNamespace(
        q=np.zeros(6), qd=np.zeros(6), tcp_pose=INSIDE
    )
    ex._run()
    assert len(calls) == 1  # safety did not issue another servo target
    assert ex.stopped_reason == "safety_stop"


@pytest.mark.parametrize("interrupt", [None, "safety", "sensor", "operator"])
def test_native_planner_observes_finish_without_more_inference(interrupt):
    # Execute the real PlannerLoop class from source. Its module-level model
    # imports need torch; this CPU-only control-flow test does not load a model.
    source = Path(executor_module.__file__).with_name("planner.py")
    tree = ast.parse(source.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlannerLoop"
    )
    clock = [0.0]
    builds = []

    def sleep(seconds):
        clock[0] += seconds

    namespace = {
        "np": np,
        "threading": threading,
        "time": SimpleNamespace(perf_counter=lambda: clock[0], sleep=sleep),
        "log": logging.getLogger(__name__),
        "snapshot_stop_reason": lambda exc: "snapshot_invalid",
    }
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), namespace)  # noqa: S102 -- trusted checked-in class
    ex, ad = native()
    ex.completed_reason = "placement_release_finished"
    ex.completed_at_s = 0.0
    ex.arm.get_state = lambda: SimpleNamespace(
        q=np.zeros(6), qd=np.zeros(6), tcp_pose=INSIDE
    )

    def build():
        builds.append(clock[0])
        if clock[0] >= 0.1:
            if interrupt == "sensor":
                raise AssertionError("required sensor stream stopped")
            if interrupt == "safety":
                ex.stopped_reason = "safety_stop"
        return SimpleNamespace()

    def infer(*_):
        pytest.fail("controller completion must stop further inference")

    loop = namespace["PlannerLoop"](
        ad.hw,
        SimpleNamespace(replan=infer),
        SimpleNamespace(build=build),
        ex,
    )
    loop.run(stop_check=lambda: interrupt == "operator" and clock[0] >= 0.1)
    assert len(builds) > 2  # sensors remained observed during the hold
    if interrupt is None:
        assert loop.stop_reason == "placement_release_finished"
        assert 2.0 <= clock[0] < 2.01
    elif interrupt == "operator":
        assert loop.stop_reason == "operator_stop"
    else:
        assert (
            ex.stopped_reason
            == {"sensor": "snapshot_invalid", "safety": "safety_stop"}[interrupt]
        )
        assert loop.stop_reason is None
    assert loop.trace == []


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_invalid_finish_observation_duration(value):
    with pytest.raises(ValueError):
        PlacementReleaseConfig(**{**FINISH, "finish_observation_s": value})
