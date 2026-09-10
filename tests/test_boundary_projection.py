"""Fail-closed boundary continuation and irrevocable FINISH hold, CPU only."""

from dataclasses import replace

import numpy as np
import pytest
from phantom_test_utils import make_small_hw

from phantom.deploy.boundary_projection import (
    BoundaryProjectionConfig, BoundaryProjectionStop, UpperYBoundaryProjection,
)
from phantom.deploy.safety import SafetyAction, SafetyMonitor
from phantom.sim.boundary_audit import BoundaryPhysicsAudit
from phantom.sim.kinematics import forward_pose

Q = np.array([-.908, -1.4, 1.5, -1.7, 1.4, 0.0])
P = forward_pose(Q)


def config():
    return BoundaryProjectionConfig(
        "upper_y_projection_v1", .010, .002, .0122, .016, .0002, .016, .25, 2.5,
    )


def hardware():
    return make_small_hw(safety={
        "wrist_extension_stop_m": None, "reach_clamp_m": None,
        "servo_constraint_hold_s": 2.5, "elbow_min_rad": .4,
        "servo_joint_speed_max_rad_s": 1.0,
        "workspace_m": {"x": [-.7, .15], "y": [-.5, .3], "z": [.03, .8]},
        "hitbox_m": {"x": [-.6, .1], "y": [-.4, .1758], "z": [.03, .75]},
    })


class Ring:
    def __init__(self):
        self.update(0, P)

    def update(self, t, pose, *, q=Q, qd=None, force=None):
        self.ts = np.array([t])
        self.arm = {"tcp_pose": np.asarray([pose]), "q": np.asarray([q]),
                    "qd": np.zeros((1, 6)) if qd is None else np.asarray([qd]),
                    "ft": np.zeros((1, 6)) if force is None else np.asarray([force]),
                    "protective_stop": np.array([False])}

    def latest(self, _n):
        return self.ts, self.arm


def guard():
    ring = Ring()
    return UpperYBoundaryProjection(config(), hardware(), {"arm": ring}), ring


def accept(g, ring, t, raw, *, measured=None, sent=None, q=Q, previous=None, dt=.008):
    measured = raw if measured is None else measured
    ring.update(t, measured, q=q)
    selected = g.select(t, raw, raw)
    sent = selected if sent is None else sent
    if previous is None:
        previous = measured if g.anchor is None else g.anchor.copy()
    g.verify_final(t, sent, q, verified=True, dt=dt, previous_pose=previous)
    g.note_submission(t, sent, q, mechanism="fake_cpu_submission")
    g.acknowledge(t, sent)
    return selected


def seeded():
    g, ring = guard()
    accept(g, ring, 0, P)
    return g, ring


def projected(g, ring, t=.008):
    raw = P.copy()
    raw[1] = .176
    return accept(g, ring, t, raw, measured=g.anchor.copy())


def test_config_is_explicit_default_off_and_does_not_alter_hardware_model():
    hw = hardware()
    before = hw.model_dump()
    monitor = SafetyMonitor(hw, {"arm": Ring()})
    assert monitor.boundary_projection is None
    assert BoundaryProjectionConfig.from_dict(config().to_dict()) == config()
    with pytest.raises(ValueError):
        BoundaryProjectionConfig.from_dict({"variant": "upper_y_projection_v1"})
    for name, value in (("budget_s", 3), ("feedback_max_age_s", float("nan")),
                        ("maximum_excursion_m", .003), ("max_tick_s", .02)):
        with pytest.raises(ValueError):
            replace(config(), **{name: value})
    assert hw.model_dump() == before


def test_projection_preserves_descent_rotation_and_raw_clamped_provenance():
    g, ring = seeded()
    raw = P.copy()
    raw[1], raw[2], raw[5] = .176, P[2] - .0005, P[5] - .001
    selected = accept(g, ring, .008, raw, measured=P)
    assert selected[1] == pytest.approx(.1656)
    np.testing.assert_array_equal(selected[[0, 2, 3, 4, 5]], raw[[0, 2, 3, 4, 5]])
    assert g.last["raw_tcp_pose"] == raw.tolist()
    assert g.last["clamped_tcp_pose"] == raw.tolist()
    assert g.last["selected_tcp_pose"] == selected.tolist()
    assert g.last["accepted_tcp_pose"] == selected.tolist()


@pytest.mark.parametrize("failure", ["large", "second_face", "no_anchor", "stale", "future", "nan", "measured_exit"])
def test_refuses_unverified_or_outside_recovery_conditions(failure):
    g, ring = seeded()
    raw, measured = P.copy(), P.copy()
    raw[1] = .176
    if failure == "large":
        raw[1] = .178
    elif failure == "second_face":
        raw[0] = -.61
    elif failure == "no_anchor":
        g.anchor = None
    elif failure == "nan":
        measured[3] = np.nan
    elif failure == "measured_exit":
        measured[1] = .182
        raw[1] = .170  # Same mechanism as the independent wrench-stop trial.
    capture = -.1 if failure == "stale" else .02 if failure == "future" else .008
    ring.update(capture, measured)
    with pytest.raises(BoundaryProjectionStop):
        g.select(.008, raw, raw)
    assert g.started_at is None


def test_physical_safety_has_priority_and_updates_once():
    g, ring = seeded()
    monitor = SafetyMonitor(g.hw, {"arm": ring}, boundary_config=config())
    monitor.boundary_projection = g
    count = []
    original = monitor._record
    monitor._record = lambda t, events: (count.append(t), original(t, events))
    ring.update(.008, P, qd=np.full(6, 5.0))
    ring.arm["protective_stop"][0] = True
    raw = P.copy()
    raw[1] = .176
    verdict = monitor.check(.008, raw)
    assert verdict.action == SafetyAction.PROTECTIVE_STOP
    assert count == [.008]
    assert "protective_stop" in [e.kind for e in verdict.events]
    assert g.started_at is None and g.last["reason"] == "physical_safety_preempted"


def test_ordinary_floor_clamp_remains_available_without_hidden_rearm():
    g, ring = seeded()
    raw, clamped = P.copy(), P.copy()
    raw[2], clamped[2] = -.1, .03
    ring.update(.008, P)
    selected = g.select(.008, raw, clamped)
    assert selected[2] == .03
    assert not g.raw_interior  # Workspace-clamped proposal cannot rearm.


def test_floor_clamp_does_not_launder_simultaneous_raw_upper_y_exit():
    g, ring = seeded()
    raw = P.copy()
    raw[1], raw[2] = .176, -.1
    clamped = raw.copy()
    clamped[2] = .03
    ring.update(.008, P)
    with pytest.raises(BoundaryProjectionStop, match="multiple_raw_faces"):
        g.select(.008, raw, clamped)


def test_ack_without_successful_submission_cannot_create_anchor_or_seal():
    g, ring = seeded()
    ring.update(.008, P)
    g.select(.008, P, P)
    g.request_finish(.008, .2, (1, 0., .2, False))
    g.verify_final(.008, P, Q, verified=True, dt=.008, previous_pose=P)
    with pytest.raises(BoundaryProjectionStop, match="ack_without_submission"):
        g.acknowledge(.008, P)
    assert g.seal is None


def test_sealed_selection_between_plane_and_ceiling_is_a_verified_hold_kind():
    g, ring = seeded()
    held = P.copy()
    held[1] = .1657
    ring.update(.008, held)
    g.select(.008, P, P)
    g.request_finish(.008, .2, (1, 0., .2, False))
    g.verify_final(.008, held, Q, verified=True, dt=.008, previous_pose=P)
    g.note_submission(.008, held, Q, mechanism="fake_cpu_submission")
    g.acknowledge(.008, held)
    ring.update(.016, held)
    selected = g.select(.016, held, held)
    assert g.last["selected_target_kind"] == "sealed_terminal_hold"
    audit = BoundaryPhysicsAudit(g, .00025, "a" * 64)
    audit.note_selected(selected, kind=g.last["selected_target_kind"])
    audit.sample(0, .016, held, held, Q, np.zeros(6), phase="finish_hold")
    assert not audit.counts


def test_replans_tangential_motion_and_interior_proposals_cannot_restart_budget():
    g, ring = seeded()
    projected(g, ring)
    start = g.started_at
    for t in np.arange(.016, 2.5, .008):
        raw = g.anchor.copy()
        raw[1] = .176 if int(t / .008) % 2 else .164
        raw[2] -= .00001  # Accepted tangent/descent progress is not a rearm.
        accept(g, ring, float(t), raw, measured=g.anchor.copy())
        assert g.started_at == start
    ring.update(2.508, g.anchor)
    interior = g.anchor.copy()
    interior[1] = .15
    with pytest.raises(BoundaryProjectionStop, match="timeout"):
        g.select(2.508, interior, interior)


def test_rearm_requires_raw_measured_final_ack_and_continuous_dwell():
    g, ring = seeded()
    projected(g, ring)
    # Move inward with feasible per-tick distance.
    for t, y in ((.016, .164), (.024, .1635)):
        raw = g.anchor.copy()
        raw[1] = y
        accept(g, ring, t, raw, measured=g.anchor.copy())
    interior = g.anchor.copy()
    rearmed = []
    for n in range(4, 38):
        t = n * .008
        accept(g, ring, t, interior)
        rearmed.append(bool(g.last.get("rearmed")))
    assert g.started_at is None and any(rearmed)


def test_final_fk_after_slide_cannot_cross_ceiling_or_replace_rate_anchor():
    for mode in ("ceiling", "rate", "anchor"):
        g, ring = seeded()
        projected(g, ring)
        ring.update(.016, g.anchor)
        raw = g.anchor.copy()
        raw[1] = .176
        selected = g.select(.016, raw, raw)
        sent, previous = selected.copy(), g.anchor.copy()
        if mode == "ceiling":
            sent[1] = .16581
        elif mode == "rate":
            sent[2] += .01
        else:
            previous[0] += .0001
        with pytest.raises(BoundaryProjectionStop):
            g.verify_final(.016, sent, Q, verified=True, dt=.008, previous_pose=previous)
        assert g.pending is None


def test_native_feedback_published_during_tick_is_causal_but_true_future_is_not():
    g, ring = seeded()
    g.decision_clock = lambda: .012
    ring.update(.011, P)
    assert np.array_equal(g.select(.008, P, P), P)
    ring.update(.013, P)
    with pytest.raises(BoundaryProjectionStop, match="stale_feedback"):
        g.select(.008, P, P)


def test_final_verification_rechecks_fresh_feedback_after_ik():
    g, ring = seeded()
    ring.update(.008, P)
    g.select(.008, P, P)
    ring.update(.016, [P[0], .182, *P[2:]])
    with pytest.raises(BoundaryProjectionStop, match="measured_envelope"):
        g.verify_final(.016, P, Q, verified=True, dt=.008, previous_pose=P)


def test_late_actual_ack_cannot_hide_command_gap_behind_a_new_anchor():
    g, ring = seeded()
    projected(g, ring)
    held, prior_ack = g.anchor.copy(), g.anchor_t
    ring.update(.020, held)
    g.select(.020, held, held)
    ring.update(.022, held)
    g.verify_final(.022, held, Q, verified=True, dt=.008, previous_pose=held)
    g.note_submission(.030, held, Q, mechanism="fake_cpu_submission")
    with pytest.raises(BoundaryProjectionStop, match="tick_overrun"):
        g.acknowledge(.030, held)
    assert g.anchor_t == prior_ack
    assert g.last["accepted_at_s"] == .030  # Actual ACK is still truthfully recorded.


def test_sealed_terminal_hold_retires_budget_but_never_rearms_or_licenses_motion():
    g, ring = seeded()
    projected(g, ring)
    held = g.anchor.copy()
    # The fixture's measured joint pose must agree with its final FK input.
    ring.update(.016, held)
    g.select(.016, held, held)
    g.request_finish(.016, .2, (1, .008, .2, False))
    g.verify_final(.016, held, Q, verified=True, dt=.008, previous_pose=held)
    g.note_submission(.016, held, Q, mechanism="fake_cpu_submission")
    g.acknowledge(.016, held)
    assert g.seal["sealed_at_s"] == .016
    assert g.started_at == .008  # Retired, never reset or extended.
    for t in (1., 2., 2.508, 3.):
        accept(g, ring, t, held, measured=held)
        assert g.seal["tcp_pose"] == held.tolist()
    changed = held.copy()
    changed[2] -= 1e-5
    ring.update(3.008, held)
    with pytest.raises(BoundaryProjectionStop, match="sealed_target_change"):
        g.select(3.008, changed, changed)


@pytest.mark.parametrize("fault", ["late_finish", "late_ack", "wrong_grip", "mismatched_ack"])
def test_finish_cannot_seal_without_causal_ack_before_deadline(fault):
    g, ring = seeded()
    projected(g, ring)
    held = g.anchor.copy()
    t = 2.508 if fault == "late_finish" else 2.500
    for n in range(2, 313):
        accept(g, ring, n * .008, held, measured=held)
    ring.update(t, held)
    with pytest.raises(BoundaryProjectionStop):
        g.select(t, held, held)
        ack = (1, t - .008, .21 if fault == "wrong_grip" else .2, False)
        g.request_finish(t, .2, ack)
        g.verify_final(t, held, Q, verified=True, dt=.008, previous_pose=held)
        g.note_submission(t, held, Q, mechanism="fake_cpu_submission")
        if fault == "mismatched_ack":
            held[2] += 1e-5
        g.acknowledge(2.508 if fault == "late_ack" else t, held)
    assert g.seal is None


def test_terminal_seal_continues_measured_containment_and_grip_guards():
    g, ring = seeded()
    ring.update(.008, P)
    g.select(.008, P, P)
    g.request_finish(.008, .2, (1, 0., .2, False))
    g.verify_final(.008, P, Q, verified=True, dt=.008, previous_pose=P)
    g.note_submission(.008, P, Q, mechanism="fake_cpu_submission")
    g.acknowledge(.008, P)
    with pytest.raises(BoundaryProjectionStop, match="sealed_grip_change"):
        g.check_grip(.21)


def test_every_step_audit_detects_between_control_contact_excursion_and_covers_hold():
    g, _ = guard()
    audit = BoundaryPhysicsAudit(g, .00025, "a" * 64)
    audit.note_selected(P)
    for step in range(9):
        measured = P.copy()
        if step == 3:
            measured[1] = .17581  # Would be missed by an 8 ms control trace.
        audit.sample(step, step * .00025, measured, P, Q, np.zeros(6),
                     phase="stop_hold" if step >= 4 else "active")
    r = audit.finalize(completed=True, expected_end_s=.002, stopped_reason="safety_stop",
                       completed_reason=None, completed_at_s=None)
    assert r["samples"] == 9 and r["contiguous"] and r["last_t_s"] == .002
    assert r["violations"]["hitbox_y_high"] == 1
    assert r["first_violation"]["step"] == 3 and r["phase_counts"]["stop_hold"] == 5


def test_every_step_audit_does_not_claim_missing_final_steps_or_nonfinite_valid():
    g, _ = guard()
    audit = BoundaryPhysicsAudit(g, .00025, "a" * 64)
    audit.sample(0, 0, P, P, Q, np.zeros(6), phase="active")
    audit.sample(2, .0005, np.full(6, np.nan), P, Q, np.zeros(6), phase="active")
    r = audit.finalize(completed=False, expected_end_s=2., stopped_reason=None,
                       completed_reason=None, completed_at_s=None)
    assert not r["finalized"] and not r["contiguous"] and r["nonfinite_samples"] == 1
