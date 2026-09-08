"""Opt-in complete-client delivery with unchanged native action/CPK timing."""

from argparse import Namespace
from copy import deepcopy

import numpy as np
import pytest
from test_sim_policy_adapter import POSE, adapter, observe, plan, tick
from test_sim_policy_campaign import audited_runtime

from phantom.sim.policy_adapter import PolicyTimingError
from tools.sim.analyze_policy_campaign import (
    policy_delivery_clock,
    policy_delivery_timing_audit,
)
from tools.sim.run_policy_campaign import simulation_command
from tools.sim.run_waffles import PolicyAudit, arguments


def clock(*values):
    return iter(values).__next__


def test_native_default_does_not_sample_clock_and_retains_override_semantics():
    def unused_clock():
        raise AssertionError("native timing must not sample a new clock")

    ad = adapter(replan_clock=unused_clock)
    observe(ad)
    p = ad.replan(latency_s=0.4, inference_delay_add_s=0.1)
    assert p.latency_s == 0.4
    assert p.action_times[0] == 0.4
    assert ad._pending[0] == 0.5
    assert "sim_policy_replan_wall_time_s" not in p.diag


def test_rpc_delivery_skips_and_rebases_preserving_native_plan_and_cpk():
    ad = adapter(
        policy_delivery_clock="rpc_wall", replan_clock=clock(10, 10.5, 20, 20.4)
    )
    raw = plan(latency=0.2)
    raw.actions[:, 0] = np.arange(16) * 0.001
    raw._cpk_token = 72
    calls = []

    def infer(obs, prev, tcp):
        calls.append(prev)
        return raw if len(calls) == 1 else plan(obs.t, latency=0.2)

    ad.policy.replan = infer
    observe(ad)
    p = ad.replan(inference_delay_add_s=0.1)
    assert p is not raw and p._cpk_token == raw._cpk_token
    np.testing.assert_array_equal(p.action_times, raw.action_times)
    assert p.latency_s == raw.latency_s == 0.2
    assert raw.diag == {}
    assert p.diag["sim_policy_replan_wall_time_s"] == 0.5
    assert p.diag["sim_effective_delivery_delay_s"] == 0.6
    tick(ad, 0)
    observe(ad, 0.592)
    assert not tick(ad, 0.592).diagnostics["plan_activated"]
    observe(ad, 0.6)
    command = tick(ad, 0.6)
    assert command.diagnostics["plan_activated"]
    assert ad._play_time == pytest.approx(0.408)
    # Four elapsed native head steps are skipped; only the next8ms move plays.
    np.testing.assert_allclose(command.tcp_pose - POSE, [0.00032, 0, 0, 0, 0, 0])
    ad.replan()
    assert calls[-1] is ad._plan
    assert calls[-1].latency_s == 0.2
    assert calls[-1]._cpk_token == 72
    np.testing.assert_array_equal(calls[-1].action_times, raw.action_times)


def test_rpc_timer_excludes_observation_callback_and_expired_chunks_are_rejected():
    value = [0.0]
    deliveries = []
    ad = adapter(
        policy_delivery_clock="rpc_wall",
        replan_clock=lambda: value[0],
        observation_callback=lambda _snapshot: value.__setitem__(0, value[0] + 7),
        delivered_plan_callback=lambda p, s, t, activated: deliveries.append(activated),
    )

    def infer(obs, _prev, _tcp):
        value[0] += 3
        return plan(obs.t, latency=0.2)

    ad.policy.replan = infer
    observe(ad)
    p = ad.replan()
    assert p.diag["sim_policy_replan_wall_time_s"] == 3
    assert p.action_times[0] == 0.2
    assert ad._pending[0] == 3
    tick(ad, 0)
    observe(ad, 3)
    command = tick(ad, 3)
    assert not command.diagnostics["plan_activated"]
    assert deliveries == [False] and ad._plan is None
    np.testing.assert_array_equal(command.tcp_pose, POSE)


def test_rpc_override_fails_before_observation_or_inference():
    ad = adapter(policy_delivery_clock="rpc_wall", replan_clock=clock(0, 1))
    observe(ad)
    with pytest.raises(PolicyTimingError, match="override"):
        ad.replan(latency_s=0)
    assert ad.policy.calls == [] and ad._pending is None


@pytest.mark.parametrize(
    "values", [(0, float("nan")), (1, 0), (0, 0.1), (float("inf"), 1)]
)
def test_invalid_rpc_clock_is_infrastructure_invalid_and_never_submitted(values):
    ad = adapter(policy_delivery_clock="rpc_wall", replan_clock=clock(*values))
    observe(ad)
    with pytest.raises(PolicyTimingError) as error:
        ad.replan()
    assert error.value.infrastructure_invalid
    assert ad._pending is None and ad._plan is None


@pytest.mark.parametrize("latency", [-0.1, float("nan"), float("inf"), "invalid"])
def test_invalid_native_duration_in_rpc_mode_is_infrastructure_invalid(latency):
    ad = adapter(policy_delivery_clock="rpc_wall", replan_clock=clock(0, 0.5))
    raw = plan()
    raw.latency_s = latency
    ad.policy.replan = lambda *_args: raw
    observe(ad)
    with pytest.raises(PolicyTimingError):
        ad.replan()
    assert ad._pending is None


def test_safety_stop_cancels_rpc_pending_delivery():
    ad = adapter(policy_delivery_clock="rpc_wall", replan_clock=clock(0, 0.5))
    observe(ad)
    ad.replan()
    ad.request_stop("operator_stop")
    assert ad._pending is None
    observe(ad, 0.5)
    command = tick(ad, 0.5)
    assert command.stopped and not command.diagnostics["plan_activated"]


def rpc_timing_fixture(tmp_path):
    ad = adapter(policy_delivery_clock="rpc_wall", replan_clock=clock(0, 0.5))
    observe(ad)
    p = ad.replan(inference_delay_add_s=0.1)
    audit = PolicyAudit(tmp_path)
    index = audit.begin_replan(0)
    audit.finish_replan(index, p, 0.7)
    row = deepcopy(audit.plans[0])
    row["activated_at"] = 0.604
    audit.close()
    _, design, _, condition, _, _ = audited_runtime()
    design["adapter_profile"] = {"policy_delivery_clock": "rpc_wall"}
    condition["inference_delay_add_s"] = 0.1
    return design, condition, row


def test_audit_distinguishes_outer_runner_wall_and_native_cpk_latency(tmp_path):
    design, condition, row = rpc_timing_fixture(tmp_path)
    assert policy_delivery_timing_audit(design, condition, [row]) == []
    assert row["inference_wall_time_s"] == 0.7
    assert row["diagnostics"]["sim_policy_replan_wall_time_s"] == 0.5
    assert row["latency_s"] == 0.2
    (tmp_path / "failed").mkdir()
    audit = PolicyAudit(tmp_path / "failed")
    index = audit.begin_replan(0)
    audit.fail_replan(index, PolicyTimingError("clock moved backwards"))
    assert audit.plans[index]["error_category"] == "instrumentation_or_input_invalid"
    assert policy_delivery_timing_audit(design, condition, audit.plans) == [
        "policy_timing_instrumentation_invalid"
    ]
    audit.close()


@pytest.mark.parametrize(
    "corrupt",
    ["missing", "native", "grid", "early", "outer", "wall", "transport", "overhead"],
)
def test_scorer_rejects_missing_or_inconsistent_rpc_instrumentation(tmp_path, corrupt):
    design, condition, row = rpc_timing_fixture(tmp_path)
    if corrupt == "missing":
        row["diagnostics"].pop("sim_policy_replan_wall_time_s")
    elif corrupt == "native":
        row["latency_s"] = 0.5
    elif corrupt == "grid":
        row["action_times"] += 0.3
    elif corrupt == "early":
        row["activated_at"] = 0.59
    elif corrupt == "outer":
        row["inference_wall_time_s"] = 0.4
    elif corrupt == "wall":
        row["diagnostics"]["sim_policy_replan_wall_time_s"] = 0.1
    elif corrupt == "overhead":
        row["diagnostics"]["sim_rpc_minus_native_latency_s"] = 0.1
    else:
        condition["inference_delay_add_s"] = 0
    assert (
        "rpc_wall_plan_timing_missing_or_inconsistent"
        in policy_delivery_timing_audit(design, condition, [row])
    )


def test_default_metadata_compatibility_and_declared_rpc_mode_drift():
    audit, design, policy, condition, info, server = audited_runtime()
    design["adapter_profile"] = {"policy_delivery_clock": "native"}
    args = (design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None)
    assert audit(*args) == []
    design["adapter_profile"]["policy_delivery_clock"] = "rpc_wall"
    assert "effective_policy_delivery_clock_differs_from_campaign" in audit(*args)
    info["policy_delivery_clock"] = "rpc_wall"
    assert audit(*args) == []
    design["delivery_latency_s"] = 0
    with pytest.raises(ValueError, match="cannot be combined"):
        policy_delivery_clock(design)


def test_campaign_and_cli_forward_only_explicit_opt_in(tmp_path, monkeypatch):
    _, design, policy, condition, _, _ = audited_runtime()
    args = Namespace(
        source=tmp_path,
        evidence=tmp_path,
        output=tmp_path,
        port=7799,
        hardware_config=tmp_path / "hardware.yaml",
    )
    native = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    assert "--policy-delivery-clock" not in native
    design["adapter_profile"] = {"policy_delivery_clock": "rpc_wall"}
    rpc = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    assert rpc == native + ["--policy-delivery-clock", "rpc_wall"]
    base = [
        "run_waffles",
        "--mode",
        "policy",
        "--episode",
        "/none",
        "--output",
        "/none",
    ]
    monkeypatch.setattr("sys.argv", base)
    assert arguments().policy_delivery_clock == "native"
    monkeypatch.setattr(
        "sys.argv",
        base + ["--policy-delivery-clock", "rpc_wall", "--policy-latency", "0"],
    )
    with pytest.raises(SystemExit):
        arguments()
