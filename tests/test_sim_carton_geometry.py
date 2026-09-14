"""A nonrectangular package must have honest volume and matching contact faces."""
from collections import Counter
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from phantom.sim.carton_geometry import carton_mesh, carton_envelope_volume, carton_face_uv
from phantom.sim.task_objects import object_config


def profile(sections=None):
    return {'model':'chamfered_sections_v1','corner_cut_xy_m':[.004,.003],
            'sections':sections or [[-.5,1,1],[.5,1,1]]}


def test_chamfered_profile_is_closed_oriented_and_removes_actual_corner_volume():
    size=np.array([.048,.046,.13]);v,f,labels=carton_mesh(size,profile())
    edges=Counter((int(a),int(b)) for face in f for a,b in zip(face,np.roll(face,-1)))
    assert all(n==1 and edges[(b,a)]==1 for (a,b),n in edges.items())
    np.testing.assert_allclose(np.ptp(v,axis=0),size,atol=1e-12)
    expected=(size[0]*size[1]-2*.004*.003)*size[2]
    assert carton_envelope_volume(size,profile())==pytest.approx(expected,rel=1e-12)
    assert len(f)==len(labels)
    assert set(labels)=={'top','bottom','front','back','left','right'}
    assert ConvexHull(v).volume==pytest.approx(expected,rel=1e-12)


def test_bulged_sections_preserve_declared_bounds_and_convexity():
    p=profile([[-.5,.96,.96],[0,1,1],[.4,.98,.98],[.5,.90,.90]])
    v,f,_=carton_mesh([.048,.046,.13],p)
    np.testing.assert_allclose(np.ptp(v,axis=0),[.048,.046,.13])
    assert 250<carton_envelope_volume([.048,.046,.13],p)*1e6<280
    # A waist would be filled by convex cooking and must not pass silently.
    with pytest.raises(ValueError,match='convex'):
        carton_mesh([.048,.046,.13],profile([[-.5,1,1],[0,.9,.9],[.5,1,1]]))


def test_shaped_capacity_guard_rejects_a_box_that_passes_only_its_bounding_volume():
    obj={'kind':'carton','size':[.05,.05,.104],'mass':.27,'nominal_capacity_ml':250,
         'carton_profile':{'model':'chamfered_sections_v1','corner_cut_xy_m':[.012,.012],
                           'sections':[[-.5,1,1],[.5,1,1]]}}
    assert np.prod(obj['size'])*1e6>250
    with pytest.raises(ValueError,match='shaped envelope volume'):
        object_config({'object':obj})


def test_profile_print_coordinates_follow_historical_face_orientation():
    size=np.array([.048,.046,.13]);p=np.array([[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]])*size/2
    np.testing.assert_allclose(carton_face_uv(p,size,'front'),[[0,0],[1,0],[1,1],[0,1]])
    assert np.isfinite(carton_face_uv(p,size,'top')).all()


@pytest.mark.parametrize('sections',[[[-.4,1,1],[.5,1,1]],[[-.5,.9,.9],[.5,.9,.9]],
                                     [[-.5,1,1],[.5,1.1,1]], [[-.5,1,1],[-.5,1,1]]])
def test_profile_rejects_invalid_sections(sections):
    with pytest.raises(ValueError,match='sections'):
        carton_mesh([.048,.046,.13],profile(sections))
