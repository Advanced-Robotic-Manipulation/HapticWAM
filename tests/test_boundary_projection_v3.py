"""upper_y_projection_v3: wider raw-request band, unchanged executed envelope (CPU only)."""
import numpy as np
import pytest

from phantom.deploy.boundary_projection import BoundaryProjectionConfig, BoundaryProjectionStop
from phantom_test_utils import make_small_hw
from test_boundary_projection import Ring, P, Q
from phantom.deploy.boundary_projection import UpperYBoundaryProjection


def cfg(variant, excursion, inset=.002):
    return BoundaryProjectionConfig(variant, .010, excursion, .0122, .016, inset, .016, .25, 2.5)


def hardware():
    # v2/v3 need a reach clamp for the solver interior (the v1 fixture has none)
    return make_small_hw(safety={
        "wrist_extension_stop_m": None, "reach_clamp_m": 1.0,
        "servo_constraint_hold_s": 2.5, "elbow_min_rad": .4,
        "servo_joint_speed_max_rad_s": 1.0,
        "workspace_m": {"x": [-.7, .15], "y": [-.5, .3], "z": [.03, .8]},
        "hitbox_m": {"x": [-.6, .1], "y": [-.4, .1758], "z": [.03, .75]},
    })


def make(variant, excursion):
    hw = hardware()
    ring = Ring()
    proj = UpperYBoundaryProjection(cfg(variant, excursion), hw, {"arm": ring})
    return proj, ring


def test_v2_rejects_wide_band_and_v3_accepts():
    with pytest.raises(ValueError):
        cfg("upper_y_projection_v2", .03)
    with pytest.raises(ValueError):
        cfg("upper_y_projection_v3", .031)
    cfg("upper_y_projection_v3", .03)
    cfg("upper_y_projection_v3", .002)


def test_v3_projects_far_raw_y_onto_plane_where_v2_stops():
    upper = .1758
    raw = P.copy(); raw[1] = upper + .02          # 2 cm past the wall
    clamped = raw.copy(); clamped[1] = upper       # ordinary hitbox clamp
    for variant, expect_stop in (("upper_y_projection_v2", True), ("upper_y_projection_v3", False)):
        proj, ring = make(variant, .002 if variant.endswith("v2") else .03)
        proj.anchor, proj.anchor_t, proj.anchor_q = P.copy(), 0.0, Q.copy()
        proj.anchor[1] = upper - .05
        if expect_stop:
            with pytest.raises(BoundaryProjectionStop) as err:
                proj.select(0.0, raw, clamped)
            assert err.value.reason.endswith("unsupported_excursion")
        else:
            out = proj.select(0.0, raw, clamped)
            assert out[1] <= proj.plane + 1e-12 and out[1] == pytest.approx(proj.solver_upper[1])
            assert bool(proj.last["projected"])


def test_v3_still_refuses_raw_beyond_band_and_multiple_faces():
    upper = .1758
    proj, ring = make("upper_y_projection_v3", .03)
    proj.anchor, proj.anchor_t, proj.anchor_q = P.copy(), 0.0, Q.copy(); proj.anchor[1] = upper - .05
    raw = P.copy(); raw[1] = upper + .04; clamped = raw.copy(); clamped[1] = upper
    with pytest.raises(BoundaryProjectionStop):
        proj.select(0.0, raw, clamped)
    proj, ring = make("upper_y_projection_v3", .03)
    proj.anchor, proj.anchor_t, proj.anchor_q = P.copy(), 0.0, Q.copy(); proj.anchor[1] = upper - .05
    raw = P.copy(); raw[1] = upper + .02; raw[2] = .76; clamped = raw.copy(); clamped[1] = upper; clamped[2] = .75
    with pytest.raises(BoundaryProjectionStop) as err:
        proj.select(0.0, raw, clamped)
    assert err.value.reason.endswith("multiple_raw_faces")


def test_v3_z_inset_caps_solver_upper_z_only():
    from dataclasses import replace
    base = cfg("upper_y_projection_v3", .03)
    with pytest.raises(ValueError):
        replace(cfg("upper_y_projection_v2", .002), z_inset_m=.02)
    with pytest.raises(ValueError):
        replace(base, z_inset_m=.06)
    hw = hardware(); ring = Ring()
    proj = UpperYBoundaryProjection(replace(base, z_inset_m=.02), hw, {"arm": ring})
    assert proj.solver_upper[2] == pytest.approx(.75 - .02)
    plain = UpperYBoundaryProjection(base, hw, {"arm": ring})
    assert plain.solver_upper[2] == pytest.approx(.75 - .002)
    assert proj.solver_upper[0] == plain.solver_upper[0] and proj.solver_upper[1] == plain.solver_upper[1]
    assert proj.ceiling == plain.ceiling  # the verified envelope is unchanged
    proj.anchor, proj.anchor_t, proj.anchor_q = P.copy(), 0.0, Q.copy()
    raw = P.copy(); raw[2] = .74; clamped = raw.copy()
    out = proj.select(0.0, raw, clamped)
    assert out[2] == pytest.approx(.73) and "z_high" in proj.last["solver_interior"]["active_constraints"]


def test_measured_feedback_outside_reach_ball_is_not_a_stop():
    """Rig 2026-09-12: auto-home start at 0.610 m from the base, reach clamp 0.6 m -> tick-1 abort."""
    from phantom_test_utils import make_small_hw
    hw = make_small_hw(safety={
        "wrist_extension_stop_m": None, "reach_clamp_m": .6,
        "servo_constraint_hold_s": 2.5, "elbow_min_rad": .4,
        "servo_joint_speed_max_rad_s": 1.0,
        "workspace_m": {"x": [-.7, .15], "y": [-.5, .3], "z": [.03, .8]},
        "hitbox_m": {"x": [-.6, .1], "y": [-.4, .1758], "z": [.03, .75]},
    })
    ring = Ring()
    stretched = np.array([-.386, -.315, .351, 0, 0, 0])   # |p| = 0.610 m, inside hitbox and workspace
    ring.update(0.0, stretched)
    proj = UpperYBoundaryProjection(cfg("upper_y_projection_v3", .03), hw, {"arm": ring})
    assert proj._envelope(stretched, reach=False) and not proj._envelope(stretched)
    assert np.allclose(proj._feedback(0.0)[:3], stretched[:3])      # measured: no stop
    outside = stretched.copy(); outside[0] = -.65                      # outside the hitbox: still a stop
    ring.update(0.0, outside)
    with pytest.raises(BoundaryProjectionStop) as err:
        proj._feedback(0.0)
    assert err.value.reason.endswith("measured_envelope")


def test_selection_currency_is_structural_not_wall_clock():
    """Rig 2026-09-12: select() at tick start, verify_final() after RTDE IK/FK solves; three aborts
    before the arm moved. The verified pose must belong to the latest selection (once); a wall-clock
    bound applies only when selection_max_age_s is configured."""
    from dataclasses import replace
    hw = hardware(); ring = Ring()
    base = cfg("upper_y_projection_v3", .03)
    assert BoundaryProjectionConfig.from_dict({k: v for k, v in base.to_dict().items()
                                               if k not in ("z_inset_m", "selection_max_age_s")}).selection_max_age_s is None
    with pytest.raises(ValueError):
        replace(base, selection_max_age_s=0.3)
    inside = P.copy(); inside[1] = -.10
    for age_cfg, delay, expect_stop in ((None, .05, False), (None, 1.5, False), (.1, .05, False), (.1, .2, True)):
        proj = UpperYBoundaryProjection(replace(base, selection_max_age_s=age_cfg), hw, {"arm": ring})
        proj.anchor, proj.anchor_t, proj.anchor_q = inside.copy(), 0.0, Q.copy()
        ring.update(0.0, inside)
        target = inside.copy(); target[2] += .001
        selected = proj.select(0.0, target, target)
        ring.update(delay, inside)
        if expect_stop:
            with pytest.raises(BoundaryProjectionStop) as err:
                proj.verify_final(delay, selected, Q, verified=True, dt=.008, previous_pose=inside)
            assert err.value.reason.endswith("missing_current_selection")
            assert proj.last["selection_age_s"] == pytest.approx(delay)
        else:
            proj.verify_final(delay, selected, Q, verified=True, dt=.008, previous_pose=inside)
            # the same selection cannot be verified twice, and no selection at all is refused
            with pytest.raises(BoundaryProjectionStop) as err:
                proj.verify_final(delay + .001, selected, Q, verified=True, dt=.008, previous_pose=inside)
            assert err.value.reason.endswith("missing_current_selection")
    fresh = UpperYBoundaryProjection(base, hw, {"arm": ring})
    fresh.anchor, fresh.anchor_t, fresh.anchor_q = inside.copy(), 0.0, Q.copy()
    ring.update(0.0, inside)
    with pytest.raises(BoundaryProjectionStop):
        fresh.verify_final(0.0, inside, Q, verified=True, dt=.008, previous_pose=inside)
