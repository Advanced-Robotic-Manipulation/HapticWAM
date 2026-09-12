"""Actual native driver and simulator feedback boundaries, fake devices only."""

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
from test_boundary_projection import P, Q, config, hardware, seeded
from test_sim_policy_adapter import Policy, observe, plan
from test_sim_campaign_servo_limiter import enabled_runtime
from test_sim_teacher_native_runner import runner
from test_ur_servo_limiter import FakeCtrl, make_arm

from phantom.deploy.boundary_projection import BoundaryProjectionStop
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.release_controller import PlacementReleaseConfig
from phantom.drivers.real import ur as ur_module
from phantom.drivers.servo_hold import ConstraintHoldBudget
from phantom.drivers.servo_limiter import ServoLimits, ServoStep
from phantom.sim.kinematics import forward_pose
from phantom.sim.policy_adapter import SimulationPolicyAdapter
from tools.sim.run_policy_campaign import simulation_command, write_controller_configs
from tools.sim.run_waffles import report_servo_limiter_execution


class NominalCtrl(FakeCtrl):
    def __init__(self):
        super().__init__()
        self.q = Q.copy()
        self.ik_calls = self.fk_calls = 0
        self.send_ok = True

    def getInverseKinematics(self, pose, qnear=None):
        self.ik_calls += 1
        return self.q.tolist()

    def getForwardKinematics(self, q):
        self.fk_calls += 1
        return forward_pose(q).tolist()

    def servoJ(self, *args):
        return super().servoJ(*args) if self.send_ok else False


def native(monkeypatch):
    arm = make_arm({"elbow_min_rad": .4, "servo_joint_speed_max_rad_s": 1.,
                    "servo_constraint_hold_s": 2.5})
    arm.hw = hardware()
    arm._ctrl = NominalCtrl()
    arm._last_qsol, arm._last_cmd_pose = Q.tolist(), P.copy()
    arm._diagnose_control_loss = lambda *_: {}
    clock = [.008]
    monkeypatch.setattr(ur_module, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    g, ring = seeded()
    return arm, g, ring, clock


def prepare(g, ring, t, raw_y=.176):
    raw = P.copy()
    raw[1] = raw_y
    ring.update(t, P)
    return g.select(t, raw, raw)


def test_native_calibrated_final_fk_rejects_before_servoj_and_ack(monkeypatch):
    arm, g, ring, _ = native(monkeypatch)
    target = prepare(g, ring, .008)
    arm._ctrl.q[0] = -.912  # Final FK violates inner ceiling despite legal selected pose.
    with pytest.raises(BoundaryProjectionStop, match="final_envelope"):
        arm.servo_l(target, .008, .1, 300, target_guard=g)
    assert arm._ctrl.fk_calls == 1 and not arm._ctrl.streamed
    assert g.last["accepted_tcp_pose"] is None
    np.testing.assert_array_equal(arm._last_qsol, Q)


def test_native_disabled_path_does_not_query_new_fk_or_change_behavior(monkeypatch):
    arm, _, _, _ = native(monkeypatch)
    result = arm.servo_l(P, .008, .1, 300)
    assert result.sent and arm._ctrl.fk_calls == 0


def test_native_failed_send_never_seals_or_records_ack(monkeypatch):
    arm, g, ring, _ = native(monkeypatch)
    target = prepare(g, ring, .008, P[1])
    g.request_finish(.008, .2, (1, 0., .2, False))
    arm._ctrl.send_ok = False
    with pytest.raises(RuntimeError, match="servoJ rejected"):
        arm.servo_l(target, .008, .1, 300, target_guard=g)
    assert g.seal is None and g.last["accepted_tcp_pose"] is None


def test_native_sealed_hold_streams_identical_joints_without_new_ik_past_deadline(monkeypatch):
    arm, g, ring, clock = native(monkeypatch)
    arm.servo_l(prepare(g, ring, .008), .008, .1, 300, target_guard=g)
    clock[0] = .016
    target = prepare(g, ring, .016, P[1])
    g.request_finish(.016, .2, (1, .008, .2, False))
    arm.servo_l(target, .008, .1, 300, target_guard=g)
    ik_at_seal = arm._ctrl.ik_calls
    assert g.seal is not None
    for t in (1., 2., 2.508, 3.):
        clock[0] = t
        ring.update(t, P)
        target = g.select(t, P, P)
        result = arm.servo_l(target, .008, .1, 300, target_guard=g)
        assert result.sent
        np.testing.assert_array_equal(arm._ctrl.streamed[-1], Q)
    assert arm._ctrl.ik_calls == ik_at_seal


def test_native_seal_keeps_measured_request_provenance_separate_from_exact_fk(monkeypatch):
    arm, g, ring, clock = native(monkeypatch)
    measured = P.copy()
    measured[0] += 1e-6  # Tool-body readback differs slightly from nominal/controller FK.
    ring.update(.008, measured)
    g.select(.008, measured, measured)
    request = g.request_finish(.008, .2, (1, 0., .2, False))
    arm.servo_l(request, .008, .1, 300, target_guard=g)
    assert g.finish_request["tcp_pose"] == measured.tolist()
    assert g.seal["tcp_pose"] == P.tolist()
    clock[0] = .016
    ring.update(.016, measured)
    selected_request = g.select(.016, measured, measured)
    np.testing.assert_array_equal(selected_request, measured)
    assert g.last["selected_tcp_pose"] == P.tolist()
    result = arm.servo_l(selected_request, .008, .1, 300, target_guard=g)
    np.testing.assert_array_equal(result.pose, P)


def sim(*, release_config=None):
    return SimulationPolicyAdapter(hardware(), Policy(), boundary_config=config(), release_config=release_config)


LIMITS = ServoLimits(elbow_min_rad=.4, joint_speed_max_rad_s=1.)


def sim_tick(ad, t, *, q=Q, gripper=.4, qd=None, submit_targets=None):
    observe(ad, t, pose=P, q=Q, qd=qd, grip=gripper,
            tactile={s.name: {"wrench": np.zeros(6)} for s in ad.hw.tactile.sensors})
    if t == 0:
        assert ad.submit(plan(t, tcp=P, delta=0., grip=gripper), t)
    cmd = ad.step(t)
    if cmd.stopped:
        return cmd, None, {}
    selection = ad.safety.boundary_projection.terminal_selection(P, Q, .008, LIMITS)
    if selection is None:
        selection = ServoStep(np.asarray(q), cmd.tcp_pose, "sent", "unchanged", 1., None, 1, True, True)
    telemetry = {}
    achieved, _ = report_servo_limiter_execution(
        ad, t, selection, cmd.gripper, 0, hold_budget=ConstraintHoldBudget(2.5),
        qref=Q, limits=LIMITS, dt=.008, telemetry=telemetry,
        submit_targets=submit_targets if submit_targets is not None else lambda _q, _g: None,
    )
    return cmd, achieved, telemetry


def test_sim_rejected_final_fk_retains_real_grip_ack_and_command_history():
    ad = sim()
    sim_tick(ad, 0)
    before = (ad._grip_ack, ad._last_grip, list(ad._grip_hist))
    bad = Q.copy()
    bad[0] = -.912
    cmd, achieved, telemetry = sim_tick(ad, .008, q=bad)
    assert achieved is None and telemetry["boundary_stop"] == "boundary_projection_final_envelope"
    assert (ad._grip_ack, ad._last_grip, list(ad._grip_hist)) == before
    assert ad._awaiting_feedback is None and ad.stopped_reason == telemetry["boundary_stop"]
    assert cmd.diagnostics["boundary_projection"]["accepted_tcp_pose"] is None


def test_existing_finish_transition_seals_sim_then_late_plan_is_discarded_without_motion():
    release = PlacementReleaseConfig(tuple(P[:3] - .1), tuple(P[:3] + .1),
                                     finish_after_release=True)
    ad = sim(release_config=release)
    sim_tick(ad, 0)
    # Reach the existing release state through its acknowledged policy-open
    # and unload logic; this test does not assign a synthetic FINISH flag.
    ad.release_controller.note_latch(.63)
    ad._grip_latch = .63
    finished = None
    for n in range(1, 150):
        t = n * .008
        if n % 50 == 0:
            ad.submit(plan(t, tcp=P, delta=0., grip=.4), t)
        cmd, achieved, _ = sim_tick(ad, t)
        assert not cmd.stopped and achieved is not None
        if ad.completed_reason:
            finished = t
            break
    assert finished is not None and ad.release_controller.finished
    seal = dict(ad.safety.boundary_projection.seal)
    assert seal["gripper_ack"][1] <= seal["finish_at_s"] <= seal["accepted_at_s"]
    assert not ad.submit(plan(finished, tcp=P, delta=.02, grip=.8), finished)
    for t in np.arange(finished + .008, finished + 2.1, .008):
        cmd, achieved, _ = sim_tick(ad, float(t))
        assert not cmd.stopped and cmd.gripper == seal["gripper_command"]
        np.testing.assert_array_equal(achieved, seal["tcp_pose"])
        assert ad.safety.boundary_projection.seal == seal
    cmd, _, _ = sim_tick(ad, finished + 2.11, qd=np.full(6, 5.))
    assert cmd.stopped and "joint_speed" in cmd.diagnostics["safety_events"]


def test_native_executor_boundary_fault_is_normal_stop_and_preserves_grip():
    ad = sim()
    calls = []
    arm = SimpleNamespace(stop=lambda value: calls.append(("stop", value)))
    ex = ChunkExecutor(ad.hw, arm, SimpleNamespace(move=lambda *a: calls.append(("grip", a))), ad.safety)
    ex._grip_target = ex._last_grip_command = .63
    ex._run = lambda: (_ for _ in ()).throw(BoundaryProjectionStop("final_envelope"))
    ex._run_guarded()
    assert ex.stopped_reason == "boundary_projection_final_envelope"
    assert calls == [("stop", 2.0)] and ex.crash_text is None
    assert ex._grip_target is None


def test_sim_callback_failure_cannot_create_ack_or_seal():
    ad = sim()
    sim_tick(ad, 0)
    before = ad._grip_ack, ad.safety.boundary_projection.anchor.copy()

    def fail_submit(_q, _grip):
        assert ad.safety.boundary_projection.pending is not None  # FK verified first.
        assert ad.safety.boundary_projection.last["drive_submission"] is None
        raise RuntimeError("fake SDK rejection")

    with pytest.raises(RuntimeError, match="fake SDK rejection"):
        sim_tick(ad, .008, submit_targets=fail_submit)
    assert ad._grip_ack == before[0]
    np.testing.assert_array_equal(ad.safety.boundary_projection.anchor, before[1])
    assert ad.safety.boundary_projection.last["accepted_tcp_pose"] is None
    assert ad.safety.boundary_projection.seal is None


def test_sim_actual_submission_precedes_grip_ack_and_boundary_ack():
    ad = sim()

    def submit(_q, _grip):
        assert ad._grip_ack is None
        assert ad.safety.boundary_projection.last["accepted_tcp_pose"] is None
        assert ad.safety.boundary_projection.last["final_verified_tcp_pose"] is not None

    sim_tick(ad, 0, submit_targets=submit)
    diag = ad.safety.boundary_projection.last
    assert diag["drive_submission"]["sequence"] == 1
    assert diag["accepted_tcp_pose"] is not None and ad._grip_ack is not None


def test_actual_native_materialization_matches_command_and_preserves_frozen_config(tmp_path):
    import json
    _, design, policy, condition, *_ = enabled_runtime()
    design["adapter_profile"]["boundary_projection"] = config().to_dict()
    args = Namespace(source=tmp_path, evidence=tmp_path / "evidence", output=tmp_path,
                     port=7799, hardware_config=tmp_path / "hardware.yaml")
    command = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    write_controller_configs(tmp_path, design["adapter_profile"], writer=runner.frozen_json)
    path = tmp_path / "runtime/boundary_projection.json"
    assert command[command.index("--boundary-projection-config") + 1] == str(path)
    assert json.loads(path.read_text()) == config().to_dict()
    design["adapter_profile"]["boundary_projection"]["maximum_excursion_m"] = .001
    with pytest.raises(RuntimeError, match="Frozen input differs"):
        write_controller_configs(tmp_path, design["adapter_profile"], writer=runner.frozen_json)


def test_native_first_tick_uses_controller_fk_for_the_previous_pose(monkeypatch):
    """Rig 2026-09-12: with no streamed pose yet, the driver compared the RTDE measured TCP against
    the controller FK of the solution; the calibration offset looked like a one-tick jump and the
    guard stopped the episode (final_rate) before the arm ever moved."""
    arm, g, ring, clock = native(monkeypatch)
    arm._last_qsol, arm._last_cmd_pose = None, None
    offset = np.array([.004, -.003, .005, .02, -.01, .015])       # measured vs model FK mismatch
    arm._recv = SimpleNamespace(getActualQ=lambda: Q.tolist(),
                                getActualTCPPose=lambda: (P + offset).tolist())
    ring.update(clock[0], P)
    target = P.copy(); target[2] += .0005
    selected = g.select(clock[0], target, target)
    res = arm.servo_l(selected, .008, .1, 300, target_guard=g)
    assert res.sent and g.failure is None
    assert np.allclose(arm._last_cmd_pose, forward_pose(Q))


def test_native_guard_fk_falls_back_to_nominal_when_controller_fk_is_garbage(monkeypatch):
    """Rig 2026-09-12: getForwardKinematics(q) returned a pose 1.7 m from the base; the guard
    compared real poses against it and stopped every episode before motion."""
    arm, g, ring, clock = native(monkeypatch)

    class GarbageCtrl(NominalCtrl):
        def getForwardKinematics(self, q, tcp_offset=None):
            self.fk_calls += 1
            return [-1.6588, -0.5221, 1.4755, 1.0983, -0.1307, -0.7024]

        def getTCPOffset(self):
            return [0, 0, .18, 0, 0, 0]

    arm._ctrl = GarbageCtrl()
    arm._last_qsol, arm._last_cmd_pose = None, None
    arm._recv = SimpleNamespace(getActualQ=lambda: Q.tolist(), getActualTCPPose=lambda: P.tolist())
    ring.update(clock[0], P)
    target = P.copy(); target[2] += .0005
    selected = g.select(clock[0], target, target)
    res = arm.servo_l(selected, .008, .1, 300, target_guard=g)
    assert res.sent and g.failure is None
    assert arm.guard_fk_source == "nominal_dh"
    assert np.allclose(arm._last_cmd_pose, forward_pose(Q))
    # a controller FK that reproduces the measured TCP is trusted
    arm2, g2, ring2, clock2 = native(monkeypatch)
    arm2._last_qsol, arm2._last_cmd_pose = None, None
    arm2._recv = SimpleNamespace(getActualQ=lambda: Q.tolist(), getActualTCPPose=lambda: P.tolist())
    ring2.update(clock2[0], P)
    selected2 = g2.select(clock2[0], target, target)
    assert arm2.servo_l(selected2, .008, .1, 300, target_guard=g2).sent
    assert arm2.guard_fk_source.startswith("controller_fk")
