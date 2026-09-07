#!/usr/bin/env python3
"""Exercise actual frozen adapters with deterministic CPU-only policy stubs."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--baseline-source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--native-only", action="store_true")
    args = p.parse_args()
    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    bridge = importlib.import_module("phantom.sim.policy_adapter")
    config = importlib.import_module("phantom.config.hardware")
    runner = importlib.import_module("tools.sim.run_waffles")
    hw = config.load_hardware(source / "configs/hardware.yaml", quiet=True)
    # Test fixture overrides suppress unrelated reach/workspace predicates only;
    # no scene, campaign, deployed hardware, model or source settings are changed.
    hw = hw.model_copy(
        update={
            "safety": hw.safety.model_copy(
                update={
                    "wrist_extension_stop_m": None,
                    "reach_clamp_m": None,
                    "workspace_m": config.WorkspaceBox(
                        x=[-0.7, 0.15], y=[-0.5, 0.3], z=[0.03, 0.8]
                    ),
                }
            )
        }
    )
    pose = np.array([-0.30, -0.12, 0.25, 0.0, np.pi, 0.0])
    q = np.array([0.0, -1.4, 1.5, -1.7, 1.4, 0.0])

    def plan(t=0.0, latency=0.2):
        actions = np.zeros((16, 7))
        actions[:, 0] = np.arange(16) * 0.001
        actions[:, 6] = 0.4
        return SimpleNamespace(
            t_created=t,
            t0_pose=pose.copy(),
            actions=actions,
            action_times=t + latency + np.arange(16) / 10,
            sigma=np.zeros(3),
            gate=0.0,
            p_evt=np.zeros(5),
            cpk=None,
            latency_s=latency,
            diag={},
            _cpk_token=72,
        )

    class Policy:
        def __init__(self):
            self.calls = []

        def replan(self, obs, prev, tcp):
            self.calls.append((obs, prev, tcp))
            return plan(obs.t)

    def adapter(**kw):
        return bridge.SimulationPolicyAdapter(hw, Policy(), mode="student", **kw)

    def observe(ad, t=0.0):
        ad.observe(
            t,
            rgb=np.zeros((12, 16, 3), np.uint8),
            q=q,
            qd=np.zeros(6),
            tcp_pose=pose,
            tcp_speed=np.zeros(6),
            gripper_state=[0.31, 3],
            wrist_ft=np.zeros(6),
        )

    def tick(ad, t):
        command = ad.step(t)
        ad.report_execution(t, accepted=True, gripper_command=command.gripper)
        return command

    def native_trace():
        def unused():
            raise AssertionError("Default native path sampled RPC clock")

        kw = {} if args.native_only else {"replan_clock": unused}
        ad = adapter(**kw)
        observe(ad)
        result = ad.replan(latency_s=0.4, inference_delay_add_s=0.1)
        commands = []
        for t in [0.0, 0.496, 0.5, 0.508, 0.516]:
            observe(ad, t)
            c = tick(ad, t)
            commands.append(
                {
                    "t": t,
                    "pose": c.tcp_pose.tolist(),
                    "gripper": c.gripper,
                    "stopped": c.stopped,
                    "reason": c.reason,
                    "diagnostics": c.diagnostics,
                }
            )
        return {
            "plan_actions": result.actions.tolist(),
            "action_times": result.action_times.tolist(),
            "latency_s": result.latency_s,
            "diag": result.diag,
            "token": result._cpk_token,
            "commands": commands,
        }

    native = native_trace()
    if args.native_only:
        result = native
    else:
        if args.baseline_source is None:
            raise ValueError("Actual frozen baseline source required")
        with tempfile.TemporaryDirectory(prefix="rpc_native_comparison_") as tmp:
            baseline = Path(tmp) / "native.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--source",
                    str(args.baseline_source),
                    "--out",
                    str(baseline),
                    "--native-only",
                ],
                check=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.DEVNULL,
            )
            assert native == json.loads(baseline.read_text())
        clocks = iter([10.0, 10.5, 20.0, 20.4])
        ad = adapter(
            policy_delivery_clock="rpc_wall", replan_clock=lambda: next(clocks)
        )
        raw = plan()
        calls = []

        def infer(obs, prev, tcp):
            calls.append(prev)
            return raw if len(calls) == 1 else plan(obs.t)

        ad.policy.replan = infer
        observe(ad)
        pending = ad.replan(inference_delay_add_s=0.1)
        assert pending is not raw and pending._cpk_token == 72
        np.testing.assert_array_equal(pending.action_times, raw.action_times)
        assert pending.latency_s == raw.latency_s == 0.2 and raw.diag == {}
        assert pending.diag["sim_policy_replan_wall_time_s"] == 0.5
        assert pending.diag["sim_effective_delivery_delay_s"] == 0.6
        tick(ad, 0.0)
        observe(ad, 0.592)
        assert not tick(ad, 0.592).diagnostics["plan_activated"]
        observe(ad, 0.6)
        c = tick(ad, 0.6)
        assert c.diagnostics["plan_activated"]
        np.testing.assert_allclose(
            c.tcp_pose - pose, [0.00032, 0, 0, 0, 0, 0], atol=1e-12
        )
        assert abs(ad._play_time - 0.408) < 1e-12
        ad.replan()
        assert calls[-1] is ad._plan and calls[-1]._cpk_token == 72
        np.testing.assert_array_equal(calls[-1].action_times, raw.action_times)
        assert calls[-1].latency_s == 0.2

        value = [0.0]
        delivered = []
        ad = adapter(
            policy_delivery_clock="rpc_wall",
            replan_clock=lambda: value[0],
            observation_callback=lambda _: value.__setitem__(0, value[0] + 7),
            delivered_plan_callback=lambda p, s, t, active: delivered.append(active),
        )

        def expired(obs, prev, tcp):
            value[0] += 3
            return plan(obs.t)

        ad.policy.replan = expired
        observe(ad)
        pending = ad.replan()
        assert pending.diag["sim_policy_replan_wall_time_s"] == 3
        tick(ad, 0.0)
        observe(ad, 3.0)
        assert not tick(ad, 3.0).diagnostics["plan_activated"]
        assert delivered == [False] and ad._plan is None

        for values in (
            [0.0, float("nan")],
            [1.0, 0.0],
            [0.0, 0.1],
            [float("inf"), 1.0],
        ):
            clocks = iter(values)
            ad = adapter(
                policy_delivery_clock="rpc_wall", replan_clock=lambda: next(clocks)
            )
            observe(ad)
            try:
                ad.replan()
                raise AssertionError("Invalid clock was accepted")
            except bridge.PolicyTimingError as error:
                assert error.infrastructure_invalid
            assert ad._pending is None and ad._plan is None

        ad = adapter(
            policy_delivery_clock="rpc_wall", replan_clock=iter([0.0, 0.5]).__next__
        )
        observe(ad)
        try:
            ad.replan(latency_s=0)
            raise AssertionError("RPC override was accepted")
        except bridge.PolicyTimingError:
            assert not ad.policy.calls
        ad.replan()
        ad.request_stop("operator_stop")
        assert ad._pending is None
        observe(ad, 0.5)
        c = tick(ad, 0.5)
        assert c.stopped and not c.diagnostics["plan_activated"]

        argv = sys.argv
        sys.argv = [
            "run_waffles",
            "--episode",
            "unused",
            "--output",
            "unused",
            "--mode",
            "policy",
        ]
        assert runner.arguments().policy_delivery_clock == "native"
        assert runner.arguments().servo_reach_limiter is False
        sys.argv += ["--policy-delivery-clock", "rpc_wall", "--servo-reach-limiter"]
        assert runner.arguments().policy_delivery_clock == "rpc_wall"
        assert runner.arguments().servo_reach_limiter is True
        sys.argv = argv
        with tempfile.TemporaryDirectory(prefix="rpc_audit_category_") as tmp:
            audit = runner.PolicyAudit(Path(tmp))
            index = audit.begin_replan(0.0)
            audit.fail_replan(index, bridge.PolicyTimingError("invalid test clock"))
            assert (
                audit.plans[index]["error_category"]
                == "instrumentation_or_input_invalid"
            )
            audit.close()
        result = {
            "status": "passed",
            "source": str(source),
            "baseline_source": str(args.baseline_source),
            "default_native_actual_adapter_trace_identical": True,
            "late_delivery_s": 0.6,
            "native_grid_origin_s": 0.2,
            "play_time_after_first_late_activation_s": 0.408,
            "first_late_command_translation_m": [0.00032, 0, 0],
            "cpk_token_and_native_latency_grid_preserved_for_next_request": True,
            "proposal_not_mutated": True,
            "observation_callback_excluded_from_rpc_timer": True,
            "expired_chunk_rejected": True,
            "invalid_clock_cases_rejected": 4,
            "override_rejected_before_request": True,
            "pending_delivery_canceled_on_stop": True,
            "infrastructure_error_category_recorded": True,
            "rpc_and_limiter_explicit_flags_parse": True,
            "GPU_model_hardware_calls": False,
            "limitation": "Deterministic CPU policy stub; no network/model/Isaac result or physical-success claim.",
        }
    with args.out.open("x") as f:
        f.write(
            json.dumps(
                result,
                indent=2,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                allow_nan=False,
            )
            + "\n"
        )
    if not args.native_only:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
