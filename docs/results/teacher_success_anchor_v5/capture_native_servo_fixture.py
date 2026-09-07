#!/usr/bin/env python3
"""Capture unchanged native driver decisions with CPU nominal IK, no connection."""

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
SOURCE = Path("/home/physicalai/phantom-icra-2027/phantom")
sys.path.insert(0, str(SOURCE))
import phantom

phantom.__path__.append(str(BASE / "source_teacher_v2_delivery/phantom"))

from phantom.config.hardware import load_hardware
from phantom.drivers.real.ur import URArm
from phantom.sim.kinematics import forward_pose, inverse_kinematics


class NominalCtrl:
    def __init__(self):
        self.calls = []
        self.streamed = []

    def isConnected(self):
        return True

    def getInverseKinematics(self, pose, qnear):
        result = inverse_kinematics(pose, qnear, max_joint_delta_rad=0.35)
        q = (
            result.q.tolist()
            if result.success or result.reason == "branch_guard"
            else []
        )
        self.calls.append({"pose": list(pose), "qref": list(qnear), "q": q})
        return q

    def servoJ(self, q, *_):
        self.streamed.append(q)
        return True


def main():
    logging.disable(logging.CRITICAL)
    folder = (
        BASE
        / "runs/teacher_success_anchor_v3/corrected_profile_bridge/rollouts/teacher__fixed_anchor__seed904301"
    )
    ex = [
        json.loads(line)
        for line in (folder / "execution_trace.jsonl").read_text().splitlines()
    ]
    times = [13.388, 14.764]
    output = {
        "source_driver_sha256": hashlib.sha256(
            (SOURCE / "phantom/drivers/real/ur.py").read_bytes()
        ).hexdigest(),
        "execution_sha256": hashlib.sha256(
            (folder / "execution_trace.jsonl").read_bytes()
        ).hexdigest(),
        "method": "No hardware: frozen URArm.servo_l with captured CPU nominal full UR3 IK answers. Callback queries and emitted joint/pose results preserved for extraction parity.",
        "cases": [],
    }
    for t in times:
        i = next(i for i, row in enumerate(ex) if abs(row["t"] - t) < 1e-8)
        previous_q = ex[i - 1]["target_q"]
        previous_pose = forward_pose(previous_q)
        target = forward_pose(ex[i]["target_q"])
        for enabled in (False, True):
            hw = load_hardware(SOURCE / "configs/hardware.nuc.mock.yaml", quiet=True)
            if enabled:
                hw = hw.model_copy(
                    update={
                        "safety": hw.safety.model_copy(
                            update={
                                "elbow_min_rad": 0.4,
                                "servo_joint_speed_max_rad_s": 1.0,
                            }
                        )
                    }
                )
            arm = URArm(hw)
            arm._ctrl = NominalCtrl()
            arm._last_qsol = list(previous_q)
            arm._last_cmd_pose = previous_pose.copy()
            result = arm.servo_l(target, 0.008, 0.1, 300)
            output["cases"].append(
                {
                    "t_s": t,
                    "enabled": enabled,
                    "qref": previous_q,
                    "previous_pose": previous_pose.tolist(),
                    "target_pose": target.tolist(),
                    "calls": arm._ctrl.calls,
                    "streamed": arm._ctrl.streamed,
                    "sent": result.sent,
                    "reason": result.reason,
                    "reported_pose": None
                    if result.pose is None
                    else np.asarray(result.pose).tolist(),
                    "limiter_hits": arm._limiter_hits,
                    "limiter_holds": arm._limiter_holds,
                }
            )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
