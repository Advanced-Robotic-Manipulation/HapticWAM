"""Bounds and attribution for the opt-in carton indentation hypothesis."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from phantom.sim.carton_compliance import compliance_spec, inset_core_mesh, outer_mass_properties
from phantom.sim.carton_geometry import carton_mesh
from phantom.sim.task_objects import object_config
from tools.sim.object_contacts import convex_support_paths

ROOT = Path(__file__).resolve().parents[1]


def carton():
    obj = json.loads((ROOT / "configs/sim/carton_teleop_side_profile_fit.json").read_text())["object"]
    obj["contact_compliance"] = dict(model="bounded_contact_v1", stiffness_n_m=4000., damping_n_s_m=15., max_indentation_m=.004)
    return obj


def test_core_preserves_normal_bound_at_sloped_folds_and_is_closed():
    obj = carton()
    outer, _, _ = carton_mesh(obj["size"], obj["carton_profile"])
    points, faces = inset_core_mesh(outer, .004)
    planes = ConvexHull(outer).equations
    distances = -(points @ planes[:, :3].T + planes[:, 3])
    assert distances.min() >= .004 - 1e-10
    np.testing.assert_allclose(distances.min(axis=0), .004, atol=1e-10)
    tri = points[faces]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()/6
    assert signed == pytest.approx(ConvexHull(points).volume)
    assert signed < ConvexHull(outer).volume
    edges = [(int(a), int(b)) for f in faces for a, b in zip(f, np.roll(f, -1))]
    assert all(edges.count((b, a)) == 1 for a, b in edges)


def test_nested_colliders_never_claim_single_convex_optical_patch():
    obj = carton()
    assert object_config({"object": obj}) is obj
    assert convex_support_paths("/World/Carton", obj) == ()
    obj.pop("contact_compliance")
    assert compliance_spec(obj) is None
    assert convex_support_paths("/World/Carton", obj) == ("/World/Carton",)


def test_nested_core_does_not_duplicate_mass_or_shrink_inertia():
    size = np.array([.046, .052, .13])
    points, faces, _ = carton_mesh(size, {"model": "rectangular_sections_v1", "sections": [[-.5, 1, 1], [.5, 1, 1]]})
    center, inertia, quat = outer_mass_properties(points, faces, .27)
    np.testing.assert_allclose(center, 0, atol=1e-15)
    np.testing.assert_allclose(inertia, .27/12*(np.sum(size**2) - size**2), atol=1e-15)
    np.testing.assert_allclose(quat, [1, 0, 0, 0], atol=1e-15)


@pytest.mark.parametrize("key,value", [("max_indentation_m", .02), ("max_indentation_m", 0), ("stiffness_n_m", -1), ("damping_n_s_m", float("nan")), ("model", "unknown")])
def test_invalid_compliance_cannot_be_silently_ignored(key, value):
    obj = carton()
    obj["contact_compliance"][key] = value
    with pytest.raises(ValueError):
        object_config({"object": obj})


def test_compliance_is_not_implicitly_applied_to_egg_or_legacy_carton():
    obj = carton()
    obj["kind"] = "egg"
    with pytest.raises(ValueError, match="profiled carton"):
        compliance_spec(obj)
    obj["kind"] = "carton"
    obj.pop("carton_profile")
    with pytest.raises(ValueError, match="profiled carton"):
        compliance_spec(obj)


def test_explicit_null_compliance_builds_the_same_rigid_usd_as_absent():
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from phantom.sim.task_object_usd import build_task_object

    layers = []
    for explicit_null in (False, True):
        obj = carton()
        obj.pop("contact_compliance")
        obj.pop("face_textures", None)
        if explicit_null:
            obj["contact_compliance"] = None
        obj = object_config({"object": obj})
        assert compliance_spec(obj) is None
        assert convex_support_paths("/World/Carton", obj) == ("/World/Carton",)
        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, "/World/Carton").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(obj["mass"])
        contact = UsdShade.Material.Define(stage, "/World/Looks/Contact")
        UsdPhysics.MaterialAPI.Apply(contact.GetPrim()).CreateStaticFrictionAttr(obj["static_friction"])

        def appearance(name, color, roughness):
            return UsdShade.Material.Define(stage, "/World/Looks/" + name)

        def collide(prim, material):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

        # Exercise the real folded-object USD builder and real USD schemas.
        # The box callback is unused for this profiled carton.
        build_task_object(stage, ROOT, "/World/Carton", obj, appearance, None, collide, contact)
        colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert [str(p.GetPath()) for p in colliders] == ["/World/Carton/Body"]
        mesh = UsdGeom.Mesh(colliders[0])
        assert len(mesh.GetPointsAttr().Get()) > 8  # Actual folded profile.
        assert UsdPhysics.MeshCollisionAPI(colliders[0]).GetApproximationAttr().Get() == "convexHull"
        bound, _ = UsdShade.MaterialBindingAPI(colliders[0]).ComputeBoundMaterial(materialPurpose="physics")
        assert bound.GetPath() == contact.GetPath()
        assert not stage.GetPrimAtPath("/World/Carton/CompressionStop")
        layers.append(stage.GetRootLayer().ExportToString())
    assert layers[0] == layers[1]
