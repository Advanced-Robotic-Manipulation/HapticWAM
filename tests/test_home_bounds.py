"""--home-bounds clamps the sampled demo start into the region that works on the rig."""
import numpy as np
import pytest

from phantom.deploy.start_pose import clamp_start_target
import phantom.scripts.run_deploy as RD


def test_clamp_is_axis_wise_deterministic_and_noop_when_empty():
    t = np.array([-0.38, -0.25, 0.28, 0.0, 3.14, 0.0])
    b = {"y_max": -0.28, "z_min": 0.31, "z_max": 0.36}
    c = clamp_start_target(t, b)
    assert np.allclose(c[:3], [-0.38, -0.28, 0.31]) and np.allclose(c[3:], t[3:])
    assert clamp_start_target(t, None) is t and clamp_start_target(t, {}) is t
    inside = np.array([-0.38, -0.30, 0.34, 0, 0, 0])
    assert np.allclose(clamp_start_target(inside, b), inside)
    assert np.allclose(clamp_start_target(np.array([-0.38, -0.30, 0.40, 0, 0, 0]), b)[2], 0.36)
    with pytest.raises(ValueError):
        clamp_start_target(t, {"z_min": 0.4, "z_max": 0.3})


def test_parse_home_bounds():
    assert RD.parse_home_bounds("") == {}
    assert RD.parse_home_bounds("y_max=-0.28, z_min=0.31;z_max=0.36") == {"y_max": -0.28, "z_min": 0.31, "z_max": 0.36}
    with pytest.raises(SystemExit):
        RD.parse_home_bounds("w_max=1")
    with pytest.raises(SystemExit):
        RD.parse_home_bounds("y_max")


def test_run_deploy_accepts_the_flag():
    ap = RD.build_arg_parser() if hasattr(RD, "build_arg_parser") else None
    src = open(RD.__file__).read()
    assert '"--home-bounds"' in src and 'deploy_overrides["home_bounds"]' in src and "start_bounds=home_bounds" in src
