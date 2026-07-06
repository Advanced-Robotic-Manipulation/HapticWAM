"""Rig factory: build real or mock drivers per the hardware config's mode
section. All mocks in a rig share ONE ContactScenario so their signals
co-vary (wrist F/T leads tactile onset, camera tracks the blob, ...)."""

from __future__ import annotations

from phantom.config.hardware import HardwareConfig
from phantom.drivers.base import Rig
from phantom.drivers.mock.scenario import ContactScenario


def make_rig(hw: HardwareConfig, *, control: bool = False, seed: int = 0) -> Rig:
    scenario = ContactScenario(seed=seed)

    if hw.mode.resolve("arm") == "mock":
        from phantom.drivers.mock.ur import MockArm
        arm = MockArm(hw, scenario)
    else:
        from phantom.drivers.real.ur import URArm
        arm = URArm(hw)

    if hw.mode.resolve("gripper") == "mock":
        from phantom.drivers.mock.robotiq import MockGripper
        gripper = MockGripper(hw.gripper, scenario)
    else:
        from phantom.drivers.real.robotiq import RobotiqGripper
        gripper = RobotiqGripper(hw.gripper, hw.arm.ip)

    tactile = {}
    if hw.mode.resolve("tactile") == "mock":
        from phantom.drivers.mock.dmtac import MockTactileSensor
        for i, s in enumerate(hw.tactile.sensors):
            tactile[s.name] = MockTactileSensor(hw.tactile, s, scenario, finger_index=i)
    else:
        from phantom.drivers.real.dmtac import DmTacSensor
        for s in hw.tactile.sensors:
            tactile[s.name] = DmTacSensor(hw.tactile, s)

    cameras = {}
    cam_mode = hw.mode.resolve("cameras")
    for name, entry in (("scene", hw.cameras.scene), ("wrist", hw.cameras.wrist)):
        if not entry.enabled:
            continue
        if cam_mode == "mock":
            from phantom.drivers.mock.realsense import MockCamera
            cameras[name] = MockCamera(entry, name, scenario)
        else:
            from phantom.drivers.real.realsense import RealSenseCamera
            cameras[name] = RealSenseCamera(entry, name)

    return Rig(hw=hw, arm=arm, gripper=gripper, tactile=tactile, cameras=cameras,
               control=control)
