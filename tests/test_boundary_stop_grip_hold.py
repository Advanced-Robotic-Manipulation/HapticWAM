"""Boundary halts stop motion while preserving the accepted gripper drive."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from phantom.deploy.executor import ChunkExecutor, halt_reason_for
from phantom.deploy import executor as executor_module
from phantom.deploy.safety import SafetyAction, SafetyEvent
from phantom.sim.policy_adapter import SimulationPolicyAdapter
from phantom_test_utils import make_small_hw
from test_sim_policy_adapter import POSE, Policy, observe, plan

ACCEPTED = .612588
MEASURED = .74
OPEN = .232
RELEASE = {"tcp_min_m": [-.55, -.076, .058], "tcp_max_m": [-.23, .149, .274]}


def adapter_with_accepted_command(*, workspace_floor=.03):
    hw = make_small_hw(safety={
        "wrist_extension_stop_m": None, "reach_clamp_m": None,
        "workspace_m": {"x": [-.7, .15], "y": [-.5, .3], "z": [workspace_floor, .8]},
        "hitbox_m": {"x": [-.5, -.2], "y": [-.2, .1], "z": [.1, .35]},
    })
    ad = SimulationPolicyAdapter(hw, Policy(), open_aperture=OPEN, release_config=RELEASE)
    observe(ad, 0, grip=MEASURED)
    assert ad.submit(plan(0, delta=0, grip=ACCEPTED), 0)
    command = ad.step(0)
    assert command.gripper == pytest.approx(ACCEPTED)
    ad.report_execution(0, accepted=True, tcp_pose=POSE, gripper_command=command.gripper)
    observe(ad, .008, grip=MEASURED)
    ad._grip_latch = ACCEPTED
    return ad


class Gripper:
    """The encoder differs from the accepted motor target under load."""
    def __init__(self):
        self.moves = [ACCEPTED]
        self.reads = 0

    def move(self, target, speed, force):
        self.moves.append(float(target))

    def get_state(self):
        self.reads += 1
        return SimpleNamespace(position=MEASURED, obj=2)


def native_with_accepted_command(ad):
    stops = []
    arm = SimpleNamespace(servo_stop=lambda: stops.append("servo_stop"),
                          get_state=lambda: SimpleNamespace(q=np.zeros(6), qd=np.zeros(6), tcp_pose=POSE))
    grip = Gripper()
    ex = ChunkExecutor(ad.hw, arm, grip, ad.safety, open_aperture=OPEN, release_config=RELEASE)
    ex._last_grip_command = ACCEPTED
    ex._grip_latch = ACCEPTED
    ex._grip_target = .15  # An unaccepted mailbox opening must not win the halt.
    return ex, grip, stops


def assert_native_halted(ex, grip, stops, expected):
    assert ex._stop.is_set() and ex._grip_target is None
    ex._grip_worker()  # A stopped worker cannot consume the old mailbox.
    assert not ex.submit(SimpleNamespace())
    assert ex.completed_reason is None and not ex.release_controller.finished
    ex.stop()
    assert stops == ["servo_stop"]
    assert grip.moves == ([ACCEPTED] if expected == ACCEPTED else [ACCEPTED, OPEN])
    assert grip.reads == 0  # No measured-closure substitution.


@pytest.mark.parametrize("axis,bound", [(0, -.5), (0, -.2), (1, -.2), (1, .1), (2, .1), (2, .35)])
def test_every_geometric_boundary_stops_native_and_sim_without_unloading(axis, bound):
    ad = adapter_with_accepted_command()
    target = POSE.copy()
    lower = bound == (-.5, -.2, .1)[axis]
    target[axis] = bound + (-1 if lower else 1) * .0002856
    verdict = ad.safety.check(.008, target)
    expected_kind = "hitbox_exit_top" if axis == 2 and not lower else "hitbox_exit"
    assert verdict.action == SafetyAction.STOP_EPISODE
    assert [event.kind for event in verdict.events] == [expected_kind]
    assert halt_reason_for(verdict.events) == "safety_stop"

    ex, grip, stops = native_with_accepted_command(ad)
    ex._halt(halt_reason_for(verdict.events), events=verdict.events,
             safety_target=(.008, target))
    assert ex.stopped_reason == "safety_stop"
    assert ex._last_grip_command == ACCEPTED
    assert_native_halted(ex, grip, stops, ACCEPTED)
    np.testing.assert_array_equal(ex.halt_state["tcp_pose"], POSE)
    np.testing.assert_array_equal(
        ex.halt_state["safety_target"]["proposed_tcp_pose"], target
    )

    ad._pose_at = lambda *_: (target.copy(), .15)
    command = ad.step(.008)
    assert command.stopped and command.reason == "safety_stop"
    assert command.diagnostics["safety_events"] == [expected_kind]
    assert command.gripper == ACCEPTED and command.gripper != MEASURED
    np.testing.assert_array_equal(command.tcp_pose, POSE)
    # The halted/held command remains inside while the checked next proposal
    # identifies the actual rejected boundary. Native and sim preserve it.
    assert command.diagnostics["safety_target"] == ex.halt_state["safety_target"]
    assert not ad.hw.safety.hitbox_m.contains(
        command.diagnostics["safety_target"]["hitbox_checked_tcp_pose"][:3]
    )
    assert ad.completed_reason is None and not command.diagnostics["completion_hold"]
    assert not ad.release_controller.finished
    ad.report_execution(.008, accepted=True, gripper_command=command.gripper)
    observe(ad, .016, grip=MEASURED)
    assert not ad.submit(plan(.016, delta=0, grip=.15), .016)
    again = ad.step(.016)
    assert again.stopped and again.gripper == ACCEPTED
    np.testing.assert_array_equal(again.tcp_pose, POSE)


@pytest.mark.parametrize("boundary", ["hitbox_exit", "hitbox_exit_top"])
@pytest.mark.parametrize("release_kind", ["wrench_limit", "tactile_fz", "tactile_depth", "veto_retry_cap"])
def test_simultaneous_release_stop_still_wins_over_boundary(boundary, release_kind):
    ad = adapter_with_accepted_command()
    target = POSE.copy()
    target[2 if boundary.endswith("_top") else 1] = .3502856 if boundary.endswith("_top") else .1002856
    verdict = ad.safety.check(.008, target)
    assert boundary in [event.kind for event in verdict.events]
    verdict.events.append(SafetyEvent(.008, release_kind, 99., SafetyAction.STOP_EPISODE))
    ex, grip, stops = native_with_accepted_command(ad)
    ex._halt(halt_reason_for(verdict.events), events=verdict.events)
    assert_native_halted(ex, grip, stops, OPEN)
    ad.safety.check = lambda *_: verdict
    command = ad.step(.008)
    assert command.stopped and command.gripper == OPEN
    assert ad._grip_latch is None and ad.completed_reason is None
    assert not ad.release_controller.finished
    np.testing.assert_array_equal(command.tcp_pose, POSE)


@pytest.mark.parametrize("reason,expected", [("operator_stop", ACCEPTED), ("camera_scene_stale", ACCEPTED),
                                          ("wrench_limit", OPEN), ("tactile_depth", OPEN), ("veto_retry_cap", OPEN)])
def test_existing_explicit_stop_semantics_are_preserved(reason, expected):
    ad = adapter_with_accepted_command()
    ex, grip, stops = native_with_accepted_command(ad)
    ex._halt(reason)
    assert_native_halted(ex, grip, stops, expected)
    ad.request_stop(reason)
    command = ad.step(.008)
    assert command.stopped and command.reason == reason
    assert command.gripper == expected and ad.completed_reason is None


def test_boundary_halt_blocks_worker_waiting_to_send_a_new_opening():
    ad = adapter_with_accepted_command()
    ex, grip, stops = native_with_accepted_command(ad)
    attempted = threading.Event()
    lock = threading.Lock()

    class ObservedLock:
        def __enter__(self):
            attempted.set()
            lock.acquire()

        def __exit__(self, *_):
            lock.release()

    ex._grip_io_lock = ObservedLock()
    lock.acquire()
    worker = threading.Thread(target=ex._grip_worker)
    worker.start()
    try:
        assert attempted.wait(1), "Worker did not reach the I/O lock"
        ex._halt("safety_stop", events=[SimpleNamespace(kind="hitbox_exit")])
        assert ex._stop.is_set()
    finally:
        lock.release()
        worker.join(2)
    assert not worker.is_alive()
    assert_native_halted(ex, grip, stops, ACCEPTED)


def test_native_servo_loop_records_rejected_target_without_sending_it(monkeypatch):
    ad = adapter_with_accepted_command()
    ex, grip, stops = native_with_accepted_command(ad)
    target = POSE.copy()
    target[1] = .1002856
    ex._last_cmd = POSE.copy()
    ex._plan = plan(0, delta=0, grip=.15)
    ex._pose_at = lambda *_: (target.copy(), .15)
    sent = []
    ex.arm.servo_l = lambda *args: sent.append(args)
    ex.arm.stop = lambda *_: stops.append("arm_stop")
    monkeypatch.setattr(executor_module.time, "perf_counter", lambda: .008)
    ex._run()
    assert ex.stopped_reason == "safety_stop" and not sent
    assert stops == ["arm_stop"] and grip.moves == [ACCEPTED]
    assert ex._grip_target is None
    np.testing.assert_array_equal(ex.halt_state["tcp_pose"], POSE)
    captured = ex.halt_state["safety_target"]
    assert captured["checked_at_s"] == .008
    np.testing.assert_array_equal(captured["proposed_tcp_pose"], target)
    np.testing.assert_array_equal(captured["hitbox_checked_tcp_pose"], target)
    target[:] = 0  # Diagnostic owns its saved values.
    assert captured["proposed_tcp_pose"][1] == .1002856


def test_hitbox_diagnostic_preserves_the_floor_clamp_semantics():
    ad = adapter_with_accepted_command(workspace_floor=.1)
    target = POSE.copy()
    target[2] = .05
    verdict = ad.safety.check(.008, target)
    assert verdict.action == SafetyAction.CLAMP
    assert "hitbox_exit" not in [event.kind for event in verdict.events]
    captured = ad.safety.target_diagnostics(.008, target)
    assert captured["proposed_tcp_pose"][2] == .05
    assert captured["hitbox_checked_tcp_pose"][2] == .1
    assert ad.hw.safety.hitbox_m.contains(captured["hitbox_checked_tcp_pose"][:3])
