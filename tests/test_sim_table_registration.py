"""Metric lattice invariants independent of the camera fit and USD renderer."""
import numpy as np
from phantom.sim.geometry import table_hole_centers


def test_registered_holes_keep_metric_pitch_phase_and_table_bounds():
    cfg = dict(center=[-.2,.1,-.04], size=[.8,.6,.05], hole_pitch=.05,
               hole_radius=.006, grid_origin_xy=[-.531,.382], grid_yaw=.17)
    points = table_hole_centers(cfg)
    c, s = np.cos(cfg['grid_yaw']), np.sin(cfg['grid_yaw'])
    grid = (points-cfg['grid_origin_xy']) @ np.array([[c,-s],[s,c]]) / .05
    np.testing.assert_allclose(grid,np.round(grid),atol=1e-12)
    assert len(points)>100
    assert (np.abs(points-np.array(cfg['center'])[:2]) <= np.array(cfg['size'])[:2]/2-.006).all()
    distances = np.linalg.norm(points[:,None]-points[None,:],axis=2)
    distances[distances<1e-10] = np.inf
    np.testing.assert_allclose(distances.min(axis=1),.05,atol=1e-12)


def test_default_lattice_retains_historical_order_and_exact_centers():
    cfg = dict(center=[-.23,0,-.04],size=[1.7,.9,.05],hole_pitch=.05,hole_radius=.0055)
    expected = [[-.23-1.7/2+(i+.5)*.05,-.9/2+(j+.5)*.05]
                for i in range(int(1.7/.05)) for j in range(int(.9/.05))]
    np.testing.assert_array_equal(table_hole_centers(cfg),expected)
