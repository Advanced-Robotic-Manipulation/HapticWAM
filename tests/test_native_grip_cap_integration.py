"""Native gripper caps must match recorded feedback and release permission."""
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.safety import SafetyAction, SafetyVerdict


RELEASE = {"tcp_min_m": [-.55, -.076, .058], "tcp_max_m": [-.23, .149, .274]}
INSIDE = np.array([-.38, .05, .10, 0., 0., 0.])
OUTSIDE = np.array([-.38, -.25, .10, 0., 0., 0.])


def rig(monkeypatch, *, release=False, grip_cap=10, pose_cap=16):
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    clock = [100.]
    monkeypatch.setattr("phantom.deploy.executor.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("phantom.deploy.executor.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    measured = [OUTSIDE.copy()]
    ring = SimpleNamespace(latest=lambda n: (np.array([clock[0]]), {"tcp_pose": np.array([measured[0]])}))
    safety = SimpleNamespace(contact_load={"left": 0., "right": 0.}, rings={"arm": ring},
                             check=lambda *args: SafetyVerdict(action=SafetyAction.OK),
                             clamp_target=lambda target: target)
    arm = SimpleNamespace(servo_l=lambda *args: None)
    recorded = []
    ex = ChunkExecutor(hw, arm, None, safety, max_play_steps=pose_cap, grip_play_steps=grip_cap,
                       record_action=lambda t, a: recorded.append((t, a.copy())),
                       release_config=RELEASE if release else None)
    ex._release_feedback = lambda: (measured[0].copy(), .64, clock[0])
    return ex, clock, measured, recorded


def plan(grips=None):
    actions = np.zeros((16, 7))
    actions[:, 2] = .001
    actions[:, 6] = .6
    actions[10:, 6] = .1
    if grips is not None:
        actions[:, 6] = grips
    return SimpleNamespace(actions=actions, t0_pose=np.zeros(6), sigma=np.zeros(4), diag={})


def play(ex, clock, active, duration=2.1):
    ex._plan = active
    ex._swap_t = clock[0]
    stop_at = clock[0] + duration
    streamed = []

    def servo(target, *args):
        streamed.append(target.copy())
        if clock[0] >= stop_at:
            ex._stop.set()
        return None

    ex.arm.servo_l = servo
    ex._run()  # Production loop, deterministic clock; no threads or devices.
    return streamed


@pytest.mark.parametrize("latch", [None, .7])
def test_pose_tail_does_not_leak_unplayed_opening_into_recording_or_feedback(monkeypatch, latch):
    ex, clock, _, recorded = rig(monkeypatch)
    ex._grip_latch = latch
    active = plan()
    original = active.actions.copy()
    streamed = play(ex, clock, active)
    expected = .6 if latch is None else latch
    assert len(recorded) == 16 and len(ex._grip_hist) == 16
    assert ex._grip_target == pytest.approx(expected)
    np.testing.assert_allclose([a[6] for _, a in recorded], expected)
    np.testing.assert_allclose(ex.gripper_cmd_at([t for t, _ in recorded]), expected)
    np.testing.assert_allclose([g for _, g in ex.entered_grip_after(0)], expected)
    np.testing.assert_array_equal(np.array([a[:6] for _, a in recorded]), original[:, :6])
    np.testing.assert_array_equal(active.actions, original)
    assert streamed[-1][2] > .015  # Pose still plays all sixteen steps.


@pytest.mark.parametrize("grip_cap,pose_cap,expected_count,expected_index", [
    (None, 16, 16, 15), (0, 16, 16, 15), (1, 16, 16, 0),
    (10, 16, 16, 9), (32, 16, 16, 15), (10, 6, 6, 5), (10, None, 16, 9),
])
def test_grip_cap_boundaries_match_pose_mailbox_and_recorded_history(monkeypatch, grip_cap, pose_cap, expected_count, expected_index):
    ex, clock, _, recorded = rig(monkeypatch, grip_cap=grip_cap, pose_cap=pose_cap)
    active = plan(np.arange(1, 17)/20.)
    play(ex, clock, active)
    expected = active.actions[expected_index, 6]
    assert len(recorded) == expected_count
    assert recorded[-1][1][6] == pytest.approx(expected)
    assert ex._grip_target == pytest.approx(expected)
    assert ex._pose_at(active, 10.)[1] == pytest.approx(expected)
    assert ex.gripper_cmd_at([clock[0]])[0] == pytest.approx(expected)


def test_explicit_release_recording_retains_last_acknowledged_worker_command(monkeypatch):
    ex, clock, _, recorded = rig(monkeypatch, release=True)
    ex._last_grip_command = .52
    ex._grip_hist.append((99., .52))
    active = plan()
    play(ex, clock, active)
    assert ex._grip_target == .6  # Capped mailbox input, not yet sent by a worker.
    np.testing.assert_allclose([a[6] for _, a in recorded], .52)
    assert list(ex._grip_hist) == [(99., .52)]  # Only actual worker sends append here.


def test_explicit_release_volume_cannot_be_bypassed_by_low_z_heuristic(monkeypatch):
    ex, clock, measured, _ = rig(monkeypatch, release=True)
    ex.safety.contact_load = {"left": 9., "right": 9.}
    active = plan()
    for target_z in (.07, .30):
        measured[0][2] = target_z
        clock[0] += .6
        _, command = ex._apply_release_control(clock[0], measured[0], .64, active, False)
        assert command == .64 and ex._grip_latch == .64
    measured[0] = OUTSIDE.copy()
    for _ in range(100):  # Longer than the legacy 0.5 s low-Z opening dwell.
        clock[0] += .008
        _, command = ex._apply_release_control(clock[0], measured[0], .35, active, False)
        assert command == .64 and ex._grip_latch == .64
    assert ex.release_controller.phase == "holding"
    measured[0] = INSIDE.copy()
    for _ in range(30):
        clock[0] += .008
        _, command = ex._apply_release_control(clock[0], measured[0], .35, active, False)
    assert command == .35 and ex._grip_latch is None
    assert ex.release_controller.phase == "releasing"


@pytest.mark.parametrize("passthrough,should_release", [([9], True), ([15], False)])
def test_release_provenance_uses_capped_grip_step_not_later_pose_step(monkeypatch, passthrough, should_release):
    ex, clock, measured, _ = rig(monkeypatch, release=True)
    measured[0] = INSIDE.copy()
    ex.safety.contact_load = {"left": 9., "right": 9.}
    ex._grip_latch = .64
    ex._last_grip_command = .64
    active = plan()
    active.actions[9, 6] = .35
    active.diag = {"terminal_veto": {"action": "close_masked", "placement_release_passthrough_indices": passthrough}}
    ex._play_time = 15.5 / ex.hw.control.action_rate_hz
    target, grip = ex._pose_at(active, ex._play_time)
    assert grip == .35
    for _ in range(30):
        clock[0] += .008
        _, command = ex._apply_release_control(clock[0], target, grip, active, False)
    assert (ex.release_controller.phase == "releasing") is should_release
    assert command == (.35 if should_release else .64)
