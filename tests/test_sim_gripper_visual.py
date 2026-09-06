"""Detailed visual mesh import must preserve scale and avoid adding contacts."""

from pathlib import Path

import numpy as np
import pytest

from phantom.sim.gripper_visual import load_dae_meshes

ASSETS = Path(__file__).resolve().parents[1] / "assets/sim/robotiq"


def test_pinned_visual_mesh_units_and_indices():
    meshes = load_dae_meshes(ASSETS / "meshes/visual/robotiq_arg2f_85_base_link.dae")
    points = np.concatenate([m["points"] for m in meshes]) * 0.001
    assert 0.07 < np.ptp(points[:, 0]) < 0.09
    assert 0.09 < np.ptp(points[:, 2]) < 0.10
    assert len(meshes) == 6
    for mesh in meshes:
        assert len(mesh["indices"]) % 3 == 0
        assert len(mesh["normals"]) == len(mesh["indices"])
        assert np.isfinite(mesh["points"]).all()


def test_visual_linkage_closes_without_introducing_physics():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdPhysics

    from phantom.sim.gripper_visual import build

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Tool")
    skin = build(stage, "/World/Tool", ASSETS)
    distances = []
    for closure in [0, 0.2, 0.6, 0.9]:
        skin.update(closure)
        frames = skin.link_matrices(closure)
        left, right = frames["left_inner_finger_pad"], frames["right_inner_finger_pad"]
        distances.append(np.linalg.norm(left[:3, 3] - right[:3, 3]))
    assert np.all(np.diff(distances) < 0)
    assert distances[0] > 0.09
    assert distances[-1] < 0.009
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 14
    assert not any(
        prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
        for prim in stage.Traverse()
    )
