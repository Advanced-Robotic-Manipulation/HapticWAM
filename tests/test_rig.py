"""Rig aggregate: worker_owned_tactile must keep the parent from opening the
tactile sensors (a real DM-Tac is single-open; the SensorSession workers own
them in the recording/deploy path)."""

import numpy as np

from phantom.drivers.base import (Arm, ArmState, Camera, CameraFrame, Gripper,
                                  GripperState, Rig, TactileFrame, TactileSensor)
from phantom_test_utils import make_hw


class _Counter:
    def __init__(self):
        self.connects = 0
        self.disconnects = 0


class _Tac(TactileSensor):
    def __init__(self, cfg, sensor, ctr):
        super().__init__(cfg, sensor)
        self.ctr = ctr
    def connect(self): self.ctr.connects += 1
    def disconnect(self): self.ctr.disconnects += 1
    def read(self): raise NotImplementedError
    def reset_reference(self): ...


class _Arm(Arm):
    def connect(self, *, control=False): ...
    def disconnect(self): ...
    def get_state(self): raise NotImplementedError
    def servo_j(self, *a): ...
    def servo_l(self, *a): ...
    def speed_l(self, *a): ...
    def move_j(self, *a, **k): ...
    def stop(self, *a): ...
    def zero_ft(self): ...
    def is_protective_stopped(self): return False
    def servo_stop(self): ...


class _Grip(Gripper):
    def connect(self): ...
    def disconnect(self): ...
    def activate(self): ...
    def move(self, *a): ...
    def get_state(self): raise NotImplementedError


def _rig(hw, ctr, **kw):
    tac = {s.name: _Tac(hw.tactile, s, ctr) for s in hw.tactile.sensors}
    return Rig(hw=hw, arm=_Arm(hw), gripper=_Grip(hw.gripper), tactile=tac, **kw)


def test_parent_opens_tactile_by_default():
    hw = make_hw()
    ctr = _Counter()
    rig = _rig(hw, ctr)
    rig.connect_all()
    assert ctr.connects == len(hw.tactile.sensors)   # 2
    rig.disconnect_all()
    assert ctr.disconnects == len(hw.tactile.sensors)


def test_worker_owned_tactile_skips_parent_open():
    hw = make_hw()
    ctr = _Counter()
    rig = _rig(hw, ctr, worker_owned_tactile=True)
    rig.connect_all()
    rig.disconnect_all()
    # the SensorSession worker processes own the devices — parent never touches
    assert ctr.connects == 0 and ctr.disconnects == 0
