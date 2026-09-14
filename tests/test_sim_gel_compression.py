"""Bounded recovery of native compliant gel contacts; no invented optical force."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tools.sim.gel_contact import GelSurfaceGeometry, select_gel_contacts


GEOMETRY = GelSurfaceGeometry(pad_thickness_m=.0017023373024049053,
                              inner_face_tolerance_m=.0003)


def data(point=(-.00045, 0., 0.), separation=-.0004, normal=(1., 0., 0.), force=5.):
    return (np.array([force]), np.array([point]), np.array([normal]),
            None if separation is None else np.array([separation]),
            np.array([[1]]), np.array([[0]]))


def select(contacts, *, compression=.0005, coverage='point', side='left', **kwargs):
    return select_gel_contacts(contacts, kwargs.pop('position', [0,0,0]),
                              kwargs.pop('quaternion', [1,0,0,0]), side=side,
                              coverage=coverage, geometry=replace(GEOMETRY,maximum_compression_m=compression),
                              **kwargs)


@pytest.mark.parametrize('coverage', ['point','manifold_patch','manifold_patch_v2'])
def test_real_native_probe_contact_recovers_force_lost_by_plane_tolerance(coverage):
    # Archived native3N probe at0.10025s, old plane depth0.322891mm and
    # actual separation0.250435mm. Native cooking shifts the surface~0.071mm.
    contacts = data(point=(-.0005282771307975054, .00016801829042378813,-.00029077514773234725),
                    separation=-.0002504349686205387,
                    normal=(-.9999947547912598,.001618200447410345,-.0028005586937069893),
                    force=2.9999494)
    old = select(contacts,compression=0.,coverage=coverage)
    fixed = select(contacts,coverage=coverage)
    assert old['normal_force_n']==0
    assert fixed['normal_force_n']==pytest.approx(2.99993366,abs=1e-6)
    assert fixed['inner_face_compression_model']['newly_eligible_inner_face_normal_force_n']==pytest.approx(2.9999494)
    assert fixed['normal_force_n'] <= fixed['observed_filtered_normal_force_n']


@pytest.mark.parametrize('side', ['left','right'])
@pytest.mark.parametrize('polarity', [-1,1])
def test_recovery_uses_physical_face_in_world_and_either_api_normal_polarity(side,polarity):
    contacts=list(data()); rotation=Rotation.from_euler('xyz',[.8,-.2,.4]);position=np.array([.3,.4,.1])
    if side=='right':contacts[1][:,:2]*=-1
    contacts[1]=rotation.apply(contacts[1])+position
    contacts[2]=rotation.apply(contacts[2])*polarity
    fixed=select(contacts,side=side,position=position,quaternion=rotation.as_quat()[[3,0,1,2]])
    assert fixed['normal_force_n']==pytest.approx(5.)
    np.testing.assert_allclose(fixed['contact_uv'],[0.,0.],atol=1e-12)


@pytest.mark.parametrize('contacts', [
    data(separation=None), data(separation=.0004), data(separation=0.),
    data(separation=-.0006), data(point=(-.0002,0.,0.),separation=-.0004),
    data(point=(-.00045,0.,0.),separation=-.00001),
    data(point=(.000851,0.,0.),separation=-.0004),
    data(normal=(0.,1.,0.)),
])
def test_unknown_speculative_deep_back_or_unaligned_contacts_remain_rejected(contacts):
    assert select(contacts)['normal_force_n']==0


def test_outside_optical_area_remains_excluded_with_separate_overlapping_reasons():
    fixed=select(data(point=(-.00045,.0138,.02035)))
    assert fixed['normal_force_n']==0
    reasons=fixed['independent_contact_rejection_n']
    assert reasons['outside_rigid_inner_face']==5.
    assert reasons['outside_corrected_inner_face']==0.
    assert reasons['outside_active_ellipse']==5.
    assert reasons['outside_rigid_face_and_active_ellipse']==5.
    assert fixed['inner_face_compression_model']['newly_eligible_inner_face_inside_active_ellipse_normal_force_n']==0


def test_default_zero_preserves_historical_plane_selection():
    assert GelSurfaceGeometry().maximum_compression_m==0
    old=select(data(),compression=0)
    assert old['normal_force_n']==0
    assert old['inner_face_compression_model']['newly_eligible_inner_face_normal_force_n']==0


@pytest.mark.parametrize('value',[-.001,np.nan,np.inf,.002])
def test_compression_cannot_exceed_declared_sensor_thickness(value):
    with pytest.raises(ValueError,match='maximum compression'):
        replace(GEOMETRY,maximum_compression_m=value)
