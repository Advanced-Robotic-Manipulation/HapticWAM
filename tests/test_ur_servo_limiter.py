"""Servo-level reach / joint-speed limiter in the real UR driver (09-04).

A fake RTDE control interface with a planar 2-link IK (a2, a3 from the DH
tuple; elbow angle from the law of cosines on the shoulder->TCP distance)
stands in for the controller. No robot is touched."""
import math
import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.drivers.real.ur import URArm

A2, A3, D1 = 0.24365, 0.21325, 0.1519


def planar_ik(pose, qref=None):
    p = np.asarray(pose[:3], dtype=float)
    r = float(np.linalg.norm(p - np.array([0.0, 0.0, D1])))
    c = (r * r - A2 * A2 - A3 * A3) / (2 * A2 * A3)
    if c > 1.0:
        return []                      # beyond reach
    elbow = math.acos(max(-1.0, min(1.0, c)))
    return [math.atan2(p[1], p[0]), -1.5, elbow, 0.7, 1.2, -3.1]


class FakeCtrl:
    def __init__(self):
        self.streamed = []
        self.flip = False

    def isConnected(self):
        return True

    def isProgramRunning(self):
        return True

    def getInverseKinematics(self, pose, qnear=None):
        q = planar_ik(pose, qnear)
        if self.flip and q:
            q = list(q); q[4] += 1.0   # a branch jump
        return q

    def servoJ(self, q, v, a, t, lookahead, gain):
        self.streamed.append(list(q))
        return True


class FakeRecv:
    def __init__(self, pose):
        self.pose = list(pose)

    def getActualQ(self):
        return planar_ik(self.pose)

    def getActualTCPPose(self):
        return list(self.pose)


def pose_at_reach(r, z=0.40):
    """A TCP pose whose shoulder distance is r, in the x<0 half like the rig."""
    dx = math.sqrt(max(r * r - (z - D1) ** 2, 0.0))
    return np.array([-dx, 0.0, z, -1.1, -1.8, 1.5])


def reach_of(q):
    return math.sqrt(A2 * A2 + A3 * A3 + 2 * A2 * A3 * math.cos(q[2]))


LIMITS = {"elbow_min_rad": 0.40, "servo_joint_speed_max_rad_s": 1.0}


def make_arm(limits: dict | None):
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    if limits:
        # the limiter ships OFF (SafetyConfig defaults None/None, 09-05 review);
        # these tests exercise it with the 09-04 rig values
        hw = hw.model_copy(update={"safety": hw.safety.model_copy(update=limits)})
    a = URArm(hw)
    a._ctrl = FakeCtrl()
    a._recv = FakeRecv(pose_at_reach(0.40))
    a._servo_active = True
    return a


@pytest.fixture
def arm():
    return make_arm(LIMITS)


def test_disabled_limiter_never_engages_even_on_empty_ik():
    """Defaults = off. An unreachable target (empty IK) must take the
    pre-limiter path: no limiter hit, no IndexError from the log line."""
    a = make_arm(None)
    assert a.hw.safety.elbow_min_rad is None
    prime(a, pose_at_reach(0.400))
    a.servo_l(pose_at_reach(0.60), 0.008, 0.1, 300)   # beyond a2+a3 -> []
    assert a._limiter_hits == 0 and a._limiter_holds == 0


def test_enabled_limiter_survives_empty_ik(arm):
    """09-05 finding 1: the throttled log line indexed q[2] of an
    empty solution and crashed the executor on exactly the tick the limiter
    exists for."""
    prime(arm, pose_at_reach(0.400))
    arm._limiter_log_t = -1e9                       # force the log line
    arm.servo_l(pose_at_reach(0.60), 0.008, 0.1, 300)
    assert arm._limiter_hits == 1


def prime(arm, pose):
    """Put the (fake) measured arm at `pose` and stream it once: the IK seed
    and the limiter anchor are then exactly this pose, like on the rig after
    homing."""
    arm._recv.pose = list(pose)
    arm.servo_l(pose, 0.008, 0.1, 300)


def test_inside_reach_streams_unchanged(arm):
    p0 = pose_at_reach(0.400)
    prime(arm, p0)
    p1 = pose_at_reach(0.4005)
    arm.servo_l(p1, 0.008, 0.1, 300)
    assert len(arm._ctrl.streamed) == 2
    assert arm._ctrl.streamed[-1] == pytest.approx(planar_ik(p1))
    assert arm._limiter_hits == 0


def test_elbow_clamp_shortens_step_towards_boundary(arm):
    emin = arm.hw.safety.elbow_min_rad
    r_lim = math.sqrt(A2 * A2 + A3 * A3 + 2 * A2 * A3 * math.cos(emin))
    prime(arm, pose_at_reach(r_lim - 0.0015))                     # just inside
    n0 = len(arm._ctrl.streamed)
    arm.servo_l(pose_at_reach(r_lim + 0.0010), 0.008, 0.1, 300)   # asks past the boundary
    assert len(arm._ctrl.streamed) == n0 + 1, "step must be shortened, not dropped"
    q = arm._ctrl.streamed[-1]
    assert abs(q[2]) >= emin, "streamed elbow must stay above elbow_min_rad"
    assert reach_of(q) > r_lim - 0.0015, "and still make progress towards the boundary"
    assert arm._limiter_hits == 1 and arm._limiter_holds == 0


def test_elbow_clamp_slides_tangentially(arm):
    emin = arm.hw.safety.elbow_min_rad
    r_lim = math.sqrt(A2 * A2 + A3 * A3 + 2 * A2 * A3 * math.cos(emin))
    p0 = pose_at_reach(r_lim - 0.0005)
    prime(arm, p0)
    # purely outward + lateral request: outward is impossible, lateral is not
    p1 = p0.copy(); p1[0] -= 0.004; p1[1] += 0.002
    arm.servo_l(p1, 0.008, 0.1, 300)
    q = arm._ctrl.streamed[-1]
    assert abs(q[2]) >= emin
    assert abs(q[0] - planar_ik(p0)[0]) > 1e-4, "lateral progress (base joint) must survive"


def test_joint_speed_limit_scales_step(arm):
    prime(arm, pose_at_reach(0.35))
    q0 = arm._ctrl.streamed[-1]
    n0 = len(arm._ctrl.streamed)
    arm.servo_l(pose_at_reach(0.352), 0.008, 0.1, 300)  # 2 mm in 8 ms: ~1.7 rad/s at the elbow
    assert len(arm._ctrl.streamed) == n0 + 1, "scaled, not held"
    q1 = arm._ctrl.streamed[-1]
    vmax = arm.hw.safety.servo_joint_speed_max_rad_s
    assert 0 < max(abs(a - b) for a, b in zip(q1, q0)) / 0.008 <= vmax + 1e-9
    assert arm._limiter_hits == 1 and arm._limiter_holds == 0


def test_branch_flip_is_still_rejected(arm):
    prime(arm, pose_at_reach(0.35))
    arm._ctrl.flip = True
    n0 = len(arm._ctrl.streamed)
    arm.servo_l(pose_at_reach(0.3502), 0.008, 0.1, 300)
    assert len(arm._ctrl.streamed) == n0
    assert arm._ik_rejects == 1


def test_servo_stop_resets_anchor(arm):
    prime(arm, pose_at_reach(0.35))
    assert arm._last_cmd_pose is not None
    arm._ctrl.servoStop = lambda: None
    arm.servo_stop()
    assert arm._last_cmd_pose is None and arm._limiter_hits == 0
