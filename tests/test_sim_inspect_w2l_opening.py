"""Numerical plane/aperture checks independent of Isaac or success outcomes."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from phantom.sim.gripper_articulation import joint_targets
from phantom.sim.gripper_visual import _rotation
from tools.sim.inspect_w2l_opening import inspect_config, inspect_surfaces, main

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/sim/waffles_w2l_articulated_provisional_20260908.json'


def surfaces(gap=.08, splay=.0):
    left=_rotation([0,1,0],splay);right=_rotation([0,1,0],-splay)
    left[:3,3]=[gap/2,0,0];right[:3,3]=[-gap/2,0,0]
    return left,right


def measure(left,right,**kwargs):
    arguments={'front_x_m':{'left':0.,'right':0.},
               'bounds_yz_m':{'left':[[-.03,-.05],[.03,.05]],'right':[[-.03,-.05],[.03,.05]]},
               'active_size_yz_m':[.027,.036],
               'depths_m':{'proximal':-.01,'center':0.,'distal':.01}}
    arguments.update(kwargs)
    return inspect_surfaces(left,right,**arguments)


@pytest.mark.parametrize('gap',[.035,.08,-.003])
def test_parallel_limit_has_equal_gap_everywhere_and_preserves_signed_overlap(gap):
    report=measure(*surfaces(gap))
    assert report['relative_face_angle_rad']==pytest.approx(0,abs=1e-12)
    for row in report['samples']:
        assert row['closing_axis_gap_m']==pytest.approx(gap,abs=1e-12)
        assert row['left_normal_separation_m']==pytest.approx(gap,abs=1e-12)
        assert row['right_normal_separation_m']==pytest.approx(gap,abs=1e-12)
        assert row['both_within_full_gel_bbox']
        # A bbox-only result cannot certify the finite front footprint.
        assert row['both_within_finite_planar_front'] is None
        assert row['positive_opening_between_finite_fronts'] is None


@pytest.mark.parametrize('splay',[math.radians(5),math.radians(15),math.radians(-8)])
def test_symmetric_splay_matches_independent_analytic_taper_and_normal_distance(splay):
    report=measure(*surfaces(.06,splay))
    assert report['relative_face_angle_rad']==pytest.approx(2*abs(splay),abs=1e-12)
    assert report['relative_distal_axis_angle_rad']==pytest.approx(2*abs(splay),abs=1e-12)
    for row in report['samples']:
        z=row['common_distal_depth_m'];expected=.06+2*z*math.tan(splay)
        assert row['closing_axis_gap_m']==pytest.approx(expected,abs=1e-12)
        assert row['left_normal_separation_m']==pytest.approx(expected*math.cos(splay),abs=1e-12)
        assert row['right_normal_separation_m']==pytest.approx(expected*math.cos(splay),abs=1e-12)
        for side in ['left','right']:
            assert row['surfaces'][side]['local_yz_m'][1]==pytest.approx(z/math.cos(splay),abs=1e-12)


def test_front_offsets_are_used_before_computing_common_frame_and_intersections():
    left,right=surfaces(.08,.12)
    report=measure(left,right,front_x_m={'left':-.002,'right':.002})
    # Face-centre separation and common origin both change under sloped normals.
    origin=np.asarray(report['common_frame']['origin_m'])
    assert origin[2]==pytest.approx(.002*math.sin(.12),abs=1e-12)
    center=report['samples'][1]
    assert center['closing_axis_gap_m']==pytest.approx(.08-.004*math.cos(.12),abs=1e-12)


def test_common_rigid_transform_preserves_metrics_and_transforms_actual_intersections():
    left,right=surfaces(.065,.14)
    # Include unequal mounting orientations and out-of-plane offsets.
    right=right@_rotation([1,0,0],.08);right[:3,3]+=[.004,.006,.008]
    before=measure(left,right)
    transform=_rotation([1,2,3],1.12);transform[:3,3]=[1.2,-.8,2.1]
    after=measure(transform@left,transform@right)
    assert after['relative_face_angle_rad']==pytest.approx(before['relative_face_angle_rad'],abs=1e-12)
    for a,b in zip(before['samples'],after['samples']):
        for field in ['closing_axis_gap_m','left_normal_separation_m','right_normal_separation_m']:
            assert b[field]==pytest.approx(a[field],abs=1e-12)
        for side in ['left','right']:
            point=np.asarray(a['surfaces'][side]['intersection_m'])
            np.testing.assert_allclose(b['surfaces'][side]['intersection_m'],transform[:3,:3]@point+transform[:3,3],atol=1e-12)
            np.testing.assert_allclose(b['surfaces'][side]['local_yz_m'],a['surfaces'][side]['local_yz_m'],atol=1e-12)


def test_finite_triangle_union_rejects_bbox_gap_without_inventing_contact_support():
    # Two disconnected finite pieces: their bbox contains the center, union does not.
    triangles=np.array([[[-.02,-.02],[-.01,-.02],[-.015,.02]],
                        [[.01,-.02],[.02,-.02],[.015,.02]]])
    report=measure(*surfaces(),front_triangles_yz_m={'left':triangles,'right':triangles})
    for row in report['samples']:
        assert row['both_within_full_gel_bbox']
        assert not row['both_within_finite_planar_front']
        assert not row['positive_opening_between_finite_fronts']
    # An actual supplied finite rectangle supports the same planar measurements.
    rectangle=np.array([[[-.02,-.02],[.02,-.02],[.02,.02]],
                        [[-.02,-.02],[.02,.02],[-.02,.02]]])
    report=measure(*surfaces(),front_triangles_yz_m={'left':rectangle,'right':rectangle})
    assert all(row['positive_opening_between_finite_fronts']for row in report['samples'])


def test_depth_outside_finite_extent_is_an_extrapolated_plane_measurement():
    report=measure(*surfaces(),depths_m={'outside':.06})
    row=report['samples'][0]
    assert row['closing_axis_gap_m']==pytest.approx(.08)
    assert not row['both_within_full_gel_bbox']
    assert not row['both_within_assumed_active_ellipse']


def test_opposed_axes_and_nonrigid_frames_fail_explicitly():
    left,right=surfaces();right=right@_rotation([0,0,1],math.pi)
    with pytest.raises(ValueError,match='bisector'):
        measure(left,right)
    left,right=surfaces();left[0,0]=2
    with pytest.raises(ValueError,match='proper rigid'):
        measure(left,right)


def test_actual_six_joint_feedback_keeps_asymmetric_follower_error_and_finite_cad_extent():
    cfg=json.loads(CONFIG.read_text());q=joint_targets(0,cfg)
    baseline=inspect_config(ROOT,cfg,q)
    np.testing.assert_allclose([v['closing_axis_gap_m']for v in baseline['samples']],.0802976626975951,atol=1e-10)
    assert baseline['gel_geometry']['finite_planar_front_triangle_count']==110
    assert baseline['samples'][2]['both_within_finite_planar_front']
    assert not baseline['samples'][0]['both_within_finite_planar_front']
    assert not baseline['samples'][-1]['both_within_finite_planar_front']
    q[2]+=.04
    observed=inspect_config(ROOT,cfg,q)
    assert observed['actual_joint_positions_rad'][2]==.04
    assert observed['coupling_residuals_rad']['left_inner_finger_joint']==.04
    assert observed['relative_face_angle_rad']==pytest.approx(.04,abs=1e-12)
    assert not np.isclose(observed['samples'][0]['closing_axis_gap_m'],observed['samples'][-1]['closing_axis_gap_m'])


def test_cli_records_supplied_joint_state_config_and_input_hashes(monkeypatch,tmp_path):
    output=tmp_path/'measurement.json'
    monkeypatch.setattr('sys.argv',['inspect_w2l_opening.py','--config',str(CONFIG),
        '--finger-q','0','0','0.04','0','0','0','--out',str(output)])
    main();result=json.loads(output.read_text())
    assert result['actual_joint_positions_rad']==[0,0,.04,0,0,0]
    assert result['effective_config']['gripper']['model']=='robotiq_2f85_w2l_articulated_v1'
    assert str(CONFIG) in result['input_sha256']
    assert 'provided --finger-q' in result['joint_state_source']
