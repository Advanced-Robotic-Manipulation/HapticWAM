"""Unlatched completion consumes real feedback and accepted command provenance."""

import numpy as np
import pytest

from test_sim_placement_release import CONFIG, INSIDE, observe, plan, setup
from test_sim_release_finish import native

from phantom.deploy.release_controller import PlacementReleaseConfig, PlacementReleaseController
from phantom.deploy.unlatched_finish import UnlatchedFinishConfig, acknowledge_gripper


SPEC = {**CONFIG, "finish_after_release": True, "unlatched_finish": {
    "tcp_min_m": [-.55, -.076, .03], "tcp_max_m": [-.23, .149, .8],
}}


def make():
    return PlacementReleaseController(PlacementReleaseConfig(**SPEC))


def update(c, t, command=.6, measured=.57, loads=.7, *, ack_t=None,
           serial=None, eligible=True, tcp=INSIDE, captures=None, ack_eligible=True,
           ack=True, accepted=None, permitted=True):
    if serial is None:
        serial = int(round(t * 1000)) + 1
    if ack_t is None:
        ack_t = t - .001
    if captures is None:
        captures = dict(arm=t, gripper=t, tactile_left=t, tactile_right=t)
    return c.update(t, tcp=tcp, policy_grip=command, measured_grip=measured,
                    pad_loads={"left": loads, "right": loads} if np.isscalar(loads) else loads,
                    eligible=eligible, accepted_grip=command if accepted is None else accepted,
                    finish_permitted=permitted,
                    accepted_grip_ack=(serial, ack_t, command, ack_eligible) if ack else None,
                    feedback_times=captures)


def grasp(c):
    update(c, 0, .1, .1, 0, ack=False)
    update(c, .1)
    update(c, .2)
    update(c, .3)
    assert c.unlatched_observer.phase == "grasp_activity"
    assert c.phase == "unarmed"
    assert not c.window_active(INSIDE) and not c.suppress_latch


def opening(c):
    grasp(c)
    update(c, .4, .53, .535, 0)
    assert c.unlatched_observer.phase == "opening_observed"


def test_relative_unlatched_finish_holds_sent_command_without_release_permission():
    c = make()
    opening(c)
    update(c, .5, .53, .535, 0)
    assert not c.finished
    update(c, .6, .53, .535, 0)
    assert c.finished and c.finished_at == .6
    assert c.last_event == "unlatched_release_finished"
    assert c.committed_at is None  # no latch opening was permitted
    assert c.diagnostics(INSIDE)["task_success"] == "not inferred by this controller"


@pytest.mark.parametrize("kind", ["startup_open", "empty_close_open", "small_adjustment",
                                  "command_only_open", "unaccepted_open", "no_actual_close"])
def test_non_release_sequences_cannot_finish(kind):
    c = make()
    if kind == "startup_open":
        for t in np.arange(0, 2, .1):
            update(c, t, .1, .1, 0)
    elif kind == "empty_close_open":
        update(c, 0, .1, .1, 0)
        for t in (.1, .2, .3):
            update(c, t, .6, .57, 0)
        for t in (.4, .5, .6, .7):
            update(c, t, .3, .3, 0)
    elif kind == "no_actual_close":
        for t in np.arange(0, 1, .1):
            update(c, t, .6, .57, .7)
        for t in (1., 1.1, 1.2, 1.3):
            update(c, t, .3, .3, 0)
    else:
        grasp(c)
        command, measured = (.58, .555) if kind == "small_adjustment" else (.53, .535)
        if kind == "command_only_open":
            measured = .57
        for t in (.4, .5, .6, .7):
            update(c, t, command, measured, 0, ack=kind != "unaccepted_open")
    assert not c.finished


@pytest.mark.parametrize("kind", ["reclose", "outside", "recovery",
                                  "ack_recovery", "missing_tactile", "nan", "negative",
                                  "stale_arm", "stale_grip", "stale_tactile", "future_sample",
                                  "load_returns", "regressed_ack", "regressed_sample"])
def test_bad_feedback_and_interruptions_cancel_pending_completion(kind):
    c = make()
    opening(c)
    kw = {}
    command, measured, loads = .53, .535, 0
    if kind == "reclose":
        command = .6
    if kind == "outside":
        kw["tcp"] = INSIDE + np.array([0, -.4, 0, 0, 0, 0])
    if kind == "recovery":
        kw["eligible"] = False
    if kind == "ack_recovery":
        kw["ack_eligible"] = False
    if kind == "missing_tactile":
        loads = {"left": 0}
    if kind == "nan":
        measured = float("nan")
    if kind == "negative":
        loads = -1
    if kind == "load_returns":
        loads = .7
    if kind == "regressed_ack":
        kw.update(serial=0, ack_t=0)
    if kind.startswith("stale") or kind in ("future_sample", "regressed_sample"):
        captures = dict(arm=.5, gripper=.5, tactile_left=.5, tactile_right=.5)
        key = {"stale_arm": "arm", "stale_grip": "gripper", "stale_tactile": "tactile_left"}.get(kind, "arm")
        captures[key] = .2 if kind.startswith("stale") else (.6 if kind == "future_sample" else .35)
        kw["captures"] = captures
    update(c, .5, command, measured, loads, **kw)
    for t in (.6, .7, .8, .9):
        update(c, t, .53, .535, 0)
    assert not c.finished


def test_duplicate_frames_and_ack_delay_do_not_advance_dwell():
    c = make()
    opening(c)
    captures = dict(arm=.4, gripper=.4, tactile_left=.4, tactile_right=.4)
    for t in (.5, .6):
        update(c, t, .53, .535, 0, captures=captures)
    assert not c.finished
    update(c, .61, .53, .535, 0, ack_t=.62)
    assert not c.finished and c.unlatched_observer.phase == "cold"


def test_native_latch_supersedes_observer_and_legacy_config_is_unchanged():
    c = make()
    grasp(c)
    c.note_latch(.63)
    assert c.phase == "holding" and c.window_active(INSIDE)
    assert c.unlatched_observer.phase == "cold"
    for t in (.4, .5, .6, .7):
        update(c, t, .53, .535, 0)
    assert not c.finished  # existing absolute 0.45 path still applies
    legacy = PlacementReleaseConfig(**CONFIG)
    assert "unlatched_finish" not in legacy.to_dict()
    assert PlacementReleaseController(legacy).variant == "placement_policy_release_v1"


def test_busy_mailbox_or_inconsistent_accepted_hold_delays_finish():
    c = make()
    opening(c)
    update(c, .6, .53, .535, 0, permitted=False)
    update(c, .7, .53, .535, 0, accepted=.6)
    assert not c.finished
    update(c, .8, .53, .535, 0)
    assert c.finished and c.finished_at == .8


def test_unloaded_carry_gap_without_opening_keeps_bounded_activity_history():
    c = make()
    grasp(c)
    outside = INSIDE + np.array([0, -.4, 0, 0, 0, 0])
    for t in (.4, .5, .6):
        update(c, t, .6, .57, 0, tcp=outside)
    assert c.unlatched_observer.phase == "grasp_activity"
    for t in (.7, .8, .9):
        update(c, t, .53, .535, 0)
    assert c.finished
    # A real measured/accepted opening during carry discards that history;
    # later entering the box while open must not create a normal FINISH.
    c = make()
    grasp(c)
    update(c, .4, .53, .535, 0, tcp=outside)
    for t in (.5, .6, .7):
        update(c, t, .53, .535, 0)
    assert not c.finished


def test_grasp_history_cannot_arm_unbounded_future_opening():
    c = make()
    grasp(c)
    for t in (10.4, 10.5, 10.6, 10.7):
        update(c, t, .53, .535, 0)
    assert not c.finished


def test_opening_event_survives_unloaded_equilibrium_rebound_but_actual_reclose_cancels():
    c = make()
    opening(c)
    update(c, .5, .54, .535, 0)
    assert c.unlatched_observer.phase == "opening_observed"
    update(c, .6, .54, .57, 0)
    assert c.finished
    c = make()
    opening(c)
    update(c, .5, .54, .59, 0)
    assert not c.finished
    assert c.unlatched_observer.reason == "measured_closed_past_grasp_reference"
    for t in (.6, .7, .8):
        update(c, t, .54, .535, 0)
    assert not c.finished


def test_pre_open_ack_tactile_frames_cannot_prove_unload_even_when_fresh():
    c = make()
    grasp(c)
    for t, tactile_t in ((.55, .31), (.65, .41), (.75, .51)):
        update(c, t, .53, .535, 0, serial=400, ack_t=.549,
               captures=dict(arm=t, gripper=t, tactile_left=tactile_t, tactile_right=tactile_t))
    assert not c.finished
    assert c.unlatched_observer.unloaded_since is None
    assert c.unlatched_observer.reason == "awaiting_post_ack_feedback"


def test_duplicate_drive_ack_does_not_starve_delayed_but_fresh_sensors():
    c = make()
    ack = None
    for i in range(151):
        t = .04 + i * .008
        command, measured, load = (.1, .1, 0) if t < .2 else (
            (.6, .57, .7) if t < .6 else (.53, .535, 0)
        )
        sample_t = t - .02  # every sensor lags more than one executor tick
        captures = dict(arm=sample_t, gripper=sample_t,
                        tactile_left=sample_t, tactile_right=sample_t)
        c.update(t, tcp=INSIDE, policy_grip=command, measured_grip=measured,
                 pad_loads={"left": load, "right": load}, eligible=True,
                 accepted_grip=None if ack is None else ack[2],
                 accepted_grip_ack=ack, feedback_times=captures)
        ack = acknowledge_gripper(ack, t, command, True)
        if c.finished:
            break
    assert c.finished
    first = acknowledge_gripper(None, 1., .53, True)
    assert acknowledge_gripper(first, 2., .53, True) is first
    changed = acknowledge_gripper(first, 2., .53, False)
    assert changed == (2, 2., .53, False)


@pytest.mark.parametrize("changes", [{"command_delta_min": 0}, {"measured_delta_min": float("nan")},
                                    {"reclose_tolerance": .5}, {"feedback_max_age_s": -1},
                                    {"tcp_max_m": [-.6, .1, .8]}])
def test_observer_invalid_thresholds_rejected(changes):
    with pytest.raises(ValueError):
        UnlatchedFinishConfig(**{**SPEC["unlatched_finish"], **changes})


def test_sim_ack_only_records_original_sent_gripper_and_never_proposal():
    ad = setup()
    ad.release_controller = make()
    observe(ad, 0, load=0, measured=.1)
    ad.submit(plan(0, .6), 0)
    cmd = ad.step(.008)
    assert ad._grip_ack is None
    ad.report_execution(.008, accepted=False, gripper_command=None)
    assert ad._grip_ack is None
    cmd = ad.step(.016)
    ad.report_execution(.016, accepted=True, gripper_command=.4)
    assert ad._grip_ack[2:] == (.4, False)  # execution differed from policy
    cmd = ad.step(.024)
    ad.report_execution(.024, accepted=True, gripper_command=cmd.gripper)
    assert ad._grip_ack[3]


def test_native_ack_is_written_after_move_and_uses_atomic_mailbox_provenance():
    ex, _ = native()
    ex.release_controller = make()
    ex._grip_observer_mailbox = (.6, True)
    ex._grip_target = .2  # observer mailbox is an atomic value/provenance pair
    ex.gripper_ring = None
    sent = []

    def move(value, *_):
        assert ex._grip_ack is None
        sent.append(value)
        ex._stop.set()

    ex.gripper.move = move
    ex._grip_worker()
    assert sent == [.6]
    assert ex._grip_ack[2:] == (.6, True)


def test_native_observer_finish_invalidates_old_mailbox_and_requires_fresh_capture_times():
    ex, ad = native()
    ex.release_controller = make()
    opening(ex.release_controller)
    ex._grip_ack = (601, .59, .53, True)
    ex._last_grip_command = .53
    ex._grip_observer_mailbox = (.8, True)
    ex._grip_target = .8
    ex._plan = plan(.6, .53)
    observe(ad, .6, load=0, measured=.535)
    ad.safety.check(.6, INSIDE)
    # The first real safety tick initializes its zero offset; use its names in
    # observer evidence, as production does. A synthetic prior graph has left/
    # right keys, so clear only its sample marker before crossing callers.
    ex.release_controller.unlatched_observer.last_samples = None
    ex._grip_io_lock.acquire()
    try:
        ex._apply_release_control(.6, INSIDE, .53, ex._plan, False)
        assert ex.completed_reason is None
    finally:
        ex._grip_io_lock.release()
    observe(ad, .608, load=0, measured=.535)
    ad.safety.check(.608, INSIDE)
    ex._apply_release_control(.608, INSIDE, .53, ex._plan, False)
    assert ex.completed_reason == "placement_release_finished"
    assert ex._grip_observer_mailbox == (.53, False)
    np.testing.assert_allclose(ex._finish_pose, INSIDE)


def test_native_same_tick_future_ack_defers_without_erasing_grasp_evidence():
    ex, ad = native()
    ex.release_controller = make()
    opening(ex.release_controller)
    ex._grip_ack = (601, .601, .53, True)  # I/O finished after servo t0=.6
    ex._last_grip_command = .53
    ex._plan = plan(.6, .53)
    observe(ad, .6, load=0, measured=.535)
    ad.safety.check(.6, INSIDE)
    ex._apply_release_control(.6, INSIDE, .53, ex._plan, False)
    assert ex.completed_reason is None
    assert ex.release_controller.unlatched_observer.phase == "opening_observed"
    assert ex.release_controller.unlatched_observer.reason == "awaiting_causal_executor_tick"
    observe(ad, .608, load=0, measured=.535)
    ad.safety.check(.608, INSIDE)
    ex._apply_release_control(.608, INSIDE, .53, ex._plan, False)
    assert ex.completed_reason == "placement_release_finished"


def test_native_uses_timestamp_paired_with_measured_value_not_later_ring_timestamp(monkeypatch):
    ex, ad = native()
    ex.release_controller = make()
    opening(ex.release_controller)
    ex._grip_ack = (601, .59, .53, True)
    ex._last_grip_command = .53
    ex._plan = plan(.6, .53)
    observe(ad, .6, load=0, measured=.535)
    ad.safety.check(.6, INSIDE)
    arm = ad.rings["arm"]
    _, values = arm.latest(1)
    monkeypatch.setattr(arm, "latest", lambda count: (np.array([.3]), values))
    monkeypatch.setattr(arm, "latest_ts", lambda: .6)
    ex._apply_release_control(.6, INSIDE, .53, ex._plan, False)
    assert ex.completed_reason is None
    assert ex.release_controller.unlatched_observer.reason == "stale_or_missing_feedback"


def test_native_and_sim_observe_identical_ack_sensor_streams_and_safety_preempts():
    ex, native_ad = native()
    ex.release_controller = make()
    ad = setup()
    ad.release_controller = make()
    observe(ad, 0, load=0, measured=.1)
    assert ad.submit(plan(0, .1), 0)
    finishes = []
    for i in range(21):
        t = i * .1
        command, measured, load = (.1, .1, 0) if i < 2 else (
            (.6, .57, .7) if i < 7 else (.53, .535, 0)
        )
        ad._plan.actions[:, 6] = command
        ex._plan = ad._plan
        ex._play_time = ad._play_time
        observe(ad, t, load=load, measured=measured)
        observe(native_ad, t, load=load, measured=measured)
        native_ad.safety.check(t, INSIDE)
        _, native_grip = ex._apply_release_control(t, INSIDE, command, ex._plan, False)
        sim_command = ad.step(t)
        assert ex.release_controller.phase == ad.release_controller.phase
        assert ex.release_controller.unlatched_observer.phase == ad.release_controller.unlatched_observer.phase
        assert ex.completed_at_s == ad.completed_at_s
        ad.report_execution(t, accepted=True, gripper_command=sim_command.gripper)
        # Emulate successful native I/O acknowledgment after this same tick;
        # earlier tests exercise the actual native worker and its mailbox.
        ex._grip_ack = (i + 1, t, native_grip, not bool(ex.completed_reason))
        ex._last_grip_command = native_grip
        if ex.completed_reason:
            finishes.append(t)
            break
    assert finishes
    t = finishes[0] + .1
    observe(ad, t, load=0, measured=.535, qd=np.ones(6) * 3)
    cmd = ad.step(t)
    assert cmd.stopped and "joint_speed" in cmd.diagnostics["safety_events"]
    assert ad.completed_at_s == finishes[0]


def test_safety_letgo_cannot_supply_unlatched_finish_opening():
    ad = setup()
    ad.release_controller = make()
    opening(ad.release_controller)
    observe(ad, .6, load=0, measured=.535, qd=np.ones(6) * 3)
    ad._last_grip = .53
    ad._grip_ack = (600, .59, .53, True)
    ad.submit(plan(.6, .53), .6)
    cmd = ad.step(.6)
    assert cmd.stopped and ad.completed_reason is None
    assert ad.release_controller.unlatched_observer.phase == "cold"
