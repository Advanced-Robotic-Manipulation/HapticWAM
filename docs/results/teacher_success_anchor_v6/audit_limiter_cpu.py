#!/usr/bin/env python3
"""Read-only source checks and pure CPU feedback test of the assembled overlay."""

import argparse
import ast
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    runner = importlib.import_module("tools.sim.run_waffles")
    config = importlib.import_module("phantom.config.hardware")
    bridge = importlib.import_module("phantom.sim.policy_adapter")
    shared = importlib.import_module("phantom.drivers.servo_limiter")
    saved = sys.argv
    sys.argv = ["run_waffles", "--episode", "unused", "--output", "unused"]
    assert runner.arguments().servo_reach_limiter is False
    sys.argv.append("--servo-reach-limiter")
    assert runner.arguments().servo_reach_limiter is True
    sys.argv = saved
    hw = config.load_hardware(source / "configs/hardware.yaml", quiet=True)
    # Pure feedback fixture, not scene/hardware reconfiguration: suppress unrelated
    # workspace/reach predicates while keeping the actual frozen adapter code.
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
    ad = bridge.SimulationPolicyAdapter(hw, SimpleNamespace(), mode="student")
    pose = np.array([-0.30, -0.12, 0.25, 0.0, np.pi, 0.0])
    q = np.array([0.0, -1.4, 1.5, -1.7, 1.4, 0.0])

    def observe(t):
        ad.observe(
            t,
            rgb=np.zeros((12, 16, 3), np.uint8),
            q=q,
            qd=np.zeros(6),
            tcp_pose=pose,
            tcp_speed=np.zeros(6),
            gripper_state=[0.55, 3],
            wrist_ft=np.zeros(6),
        )

    rejected = shared.ServoStep(None, None, "limiter_hold", "hold", None, "elbow", 7)
    rejects = 0
    for i in range(25):
        t = i * 0.008
        observe(t)
        assert not ad.step(t).stopped
        achieved, rejects = runner.report_servo_limiter_execution(
            ad, t, rejected, 0.61, rejects
        )
        assert achieved is None and rejects == i + 1
        assert ad.stopped_reason == (None if i < 24 else "servo_limiter_stall")
    observe(0.2)
    stopped = ad.step(0.2)
    assert stopped.stopped and stopped.reason == "servo_limiter_stall"
    assert stopped.gripper == 0.61 and ad.ik_rejects == 25
    np.testing.assert_array_equal(stopped.tcp_pose, pose)
    rejects_before_stop_hold = ad.ik_rejects
    ad.report_execution(
        0.2, accepted=True, tcp_pose=pose, gripper_command=stopped.gripper
    )
    tree = ast.parse((source / "tools/sim/run_waffles.py").read_text())
    holds = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "command.stopped"
    ]
    assert len(holds) == 1
    stop_block = ast.unparse(holds[0])
    assert "terminal_until = min(t + 2.0, duration)" in stop_block
    assert "stop_after_step = True" in stop_block
    assert "servo_reach_limits is not None" in stop_block
    summary = {
        "status": "passed",
        "source": str(source),
        "cli_default_off": True,
        "cli_explicit_flag_parses": True,
        "consecutive_rejects_before_accepted_stop_hold": rejects_before_stop_hold,
        "counter_after_accepted_stop_hold": ad.ik_rejects,
        "stop_requested_at_s": 0.192,
        "next_stopped_tick_s": 0.2,
        "stop_reason": stopped.reason,
        "held_gripper_command": stopped.gripper,
        "runner_ordinary_stop_tail_ast": "min(t + 2.0, duration)",
        "ik_rejection_exception": False,
        "GPU_model_hardware_connections": False,
        "limitation": "CPU feedback/runner-source audit; actual Isaac dynamics and full scored stop validity require the separately authorized diagnostic.",
    }
    with args.out.open("x") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
