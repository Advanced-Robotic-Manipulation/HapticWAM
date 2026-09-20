"""Geometric/dynamic preconditions for the new task scenes (CPU)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from phantom.sim.task_objects import egg_mesh, object_config, object_identity, orientation_wxyz

ROOT = Path(__file__).resolve().parents[1]


def test_egg_is_closed_convex_asymmetric_shell_with_physical_volume():
    vertices, faces = egg_mesh([.043, .043, .057])
    edges = Counter((int(a), int(b)) for f in faces for a,b in zip(f, np.roll(f,-1)))
    assert all(n == 1 and edges[(b,a)] == 1 for (a,b),n in edges.items())
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [.043,.043,.057], atol=1e-12)
    triangles = vertices[faces]
    volume = np.einsum('ij,ij->i', triangles[:,0], np.cross(triangles[:,1],triangles[:,2])).sum()/6
    assert volume == pytest.approx(ConvexHull(vertices).volume, rel=1e-10)
    assert 45e-6 < volume < 65e-6  # 45–65ml for the intended egg envelope
    radius = np.linalg.norm(vertices[:,:2], axis=1)
    assert vertices[radius.argmax(),2] < 0  # broad end differs from an ellipsoid


def test_scene_rig_is_identical_to_working_waffle_baseline():
    baseline = json.loads((ROOT/'tests/fixtures/reference/sim_zoo_20260912/inputs/scene.json').read_text())
    for task in ('carton','egg'):
        cfg = json.loads((ROOT/f'configs/sim/{task}.json').read_text())
        for key in ('camera','gripper','physics','robot_mount','table'):
            assert cfg[key] == baseline[key], (task,key)
        assert object_identity(cfg) == (task, '/World/'+task.title())
        assert object_config(cfg)['mass'] > 0
        assert np.linalg.norm(orientation_wxyz(object_config(cfg))) == pytest.approx(1)
    assert not cfg['bin']['enabled']  # egg scene must not add the waffle bin


def test_legacy_waffle_and_explicit_egg_pose_are_distinct():
    legacy = {'waffle': {'size':[.17,.035,.09], 'mass':.035, 'yaw':.3}}
    assert object_identity(legacy) == ('waffle','/World/Waffle')
    np.testing.assert_allclose(orientation_wxyz(legacy['waffle']), [np.cos(.15),0,0,np.sin(.15)])
    tilted = {'size':[.043,.043,.057], 'rpy':[np.pi/2,0,0]}
    np.testing.assert_allclose(orientation_wxyz(tilted), [np.sqrt(.5),np.sqrt(.5),0,0], atol=1e-15)
    with pytest.raises(ValueError, match='one task object'):
        object_config({**legacy,'object':tilted})


def test_printed_carton_capacity_rules_out_impossible_image_fit():
    carton = {'kind': 'carton', 'size': [.04047906, .0445497, .113196],
              'mass': .27, 'nominal_capacity_ml': 250}
    with pytest.raises(ValueError, match='exterior bounding volume'):
        object_config({'object': carton})
    # A taller old hypothesis is still too small; changing mass cannot fix it.
    with pytest.raises(ValueError, match='exterior bounding volume'):
        object_config({'object': {**carton, 'size': [.04047906, .0445497, .125099]}})
    plausible = {**carton, 'size': [.045, .05, .125]}
    assert object_config({'object': plausible}) == plausible


@pytest.mark.parametrize('capacity', [0, -250, float('nan'), float('inf')])
def test_carton_capacity_must_be_finite_and_positive(capacity):
    with pytest.raises(ValueError, match='finite positive capacity'):
        object_config({'object': {'kind': 'carton', 'size': [.05, .05, .12],
                                  'nominal_capacity_ml': capacity}})


def test_sdf_opt_in_keeps_egg_geometry_and_rejects_ignored_settings():
    egg = {"kind": "egg", "size": [.048, .048, .057], "mass": .06}
    assert object_config({"object": egg}) == egg
    compound = {**egg, "collision_approximation": "compound_convex_v1"}
    assert object_config({"object": compound}) == compound
    sdf = {**egg, "collision_approximation": "sdf", "sdf_resolution": 256}
    assert object_config({"object": sdf})["size"] == egg["size"]
    for invalid in ({**egg, "sdf_resolution": 256},
                    {**compound, "sdf_resolution": 256},
                    {**compound, "kind": "carton"},
                    {**egg, "collision_approximation": "unknown"},
                    {**sdf, "kind": "carton"},
                    {**sdf, "sdf_resolution": 0},
                    {**sdf, "sdf_resolution": 256.5},
                    {**sdf, "sdf_resolution": True}):
        with pytest.raises(ValueError):
            object_config({"object": invalid})
