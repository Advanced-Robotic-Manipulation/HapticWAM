"""Physical extension is invariant to full elbow turns; joint speeds are not."""
import math
import pytest
from phantom.drivers.servo_limiter import ServoLimits, feasible, limit_violation

LIMITS=ServoLimits(elbow_min_rad=.4,joint_speed_max_rad_s=1.)
def q(elbow):return [0.,-1.,elbow,.5,1.,-2.]

@pytest.mark.parametrize('turn',[-2,-1,0,1,2])
@pytest.mark.parametrize('sign',[-1,1])
def test_wrapped_extension_and_safe_bend_have_same_guard(turn,sign):
    shift=turn*math.tau
    assert limit_violation(q(shift+sign*.39),None,.008,LIMITS)=='elbow'
    assert limit_violation(q(shift+sign*.41),None,.008,LIMITS) is None

@pytest.mark.parametrize('turn',[-1,0,1])
@pytest.mark.parametrize('sign',[-1,1])
def test_recovery_moves_away_from_extension_on_each_branch(turn,sign):
    shift=turn*math.tau
    ref=q(shift+sign*.3)
    assert feasible(q(shift+sign*.301),ref,.008,LIMITS)
    assert not feasible(q(shift+sign*.299),ref,.008,LIMITS)
    assert not feasible(q(shift+sign*.3),ref,.008,LIMITS)


def test_periodic_geometry_does_not_allow_full_turn_command_jump():
    ref=q(.5);target=q(.5+math.tau)
    assert limit_violation(target,ref,.008,LIMITS)=='joint_speed'
    assert not feasible(target,ref,.008,LIMITS)
