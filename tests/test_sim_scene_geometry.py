"""Physical invariants for gripper collision assets, without an Isaac runtime."""

import struct
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from phantom.sim.scene import (
    elliptical_prism_mesh,
    gripper_urdf,
    set_forearm_collision_approximation,
)


def _signed_volume_and_inertia(vertices, faces, mass=1.0):
    """Integrate tetrahedra to check mass properties independently of primitives."""
    points = vertices[faces]
    volumes = (
        np.einsum("ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2])) / 6
    )
    sums = points.sum(axis=1)
    second_moments = (
        np.einsum("nki,nkj->nij", points, points) + np.einsum("ni,nj->nij", sums, sums)
    ) / 20
    volume = volumes.sum()
    moment = np.einsum("n,nij->ij", volumes, second_moments) * mass / volume
    return volume, np.trace(moment) * np.eye(3) - moment


def _assert_closed_convex_mesh(vertices, faces, size, tolerance=1e-12):
    assert np.isfinite(vertices).all()
    assert faces.min() >= 0 and faces.max() < len(vertices)
    np.testing.assert_allclose(
        vertices.min(axis=0), -np.array(size) / 2, atol=tolerance
    )
    np.testing.assert_allclose(vertices.max(axis=0), np.array(size) / 2, atol=tolerance)

    edges = Counter(
        (int(face[i]), int(face[(i + 1) % 3])) for face in faces for i in range(3)
    )
    # Each shared edge must have exactly one face on either side with opposite
    # direction. This also catches holes, duplicate triangles and flipped faces.
    assert all(count == 1 and edges[b, a] == 1 for (a, b), count in edges.items())
    assert len(vertices) - len(edges) // 2 + len(faces) == 2

    points = vertices[faces]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    assert (lengths > 1e-14).all()
    normals /= lengths[:, None]
    # Every vertex lies behind every outward face plane: the collider is
    # convex, correctly wound and contains its intended origin strictly inside.
    offsets = np.einsum("ij,ij->i", normals, points[:, 0])
    assert (offsets > 0).all()
    assert (normals @ vertices.T - offsets[:, None]).max() <= tolerance


@pytest.mark.parametrize(
    "size", [[0.012, 0.026, 0.055], [0.006, 0.030, 0.060], [0.008, 0.016, 0.060]]
)
@pytest.mark.parametrize("segments", [48, 96])
def test_rounded_contact_hulls_are_closed_convex_and_preserve_metric_bounds(
    size, segments
):
    vertices, faces = elliptical_prism_mesh(size, segments)
    _assert_closed_convex_mesh(vertices, faces, size)
    volume, inertia = _signed_volume_and_inertia(vertices, faces)
    analytic_volume = size[0] * np.pi * size[1] * size[2] / 4
    # Inscribed polygonal approximation must approach the solid ellipse from
    # below, with less than 0.3% volume loss at the production resolution.
    assert 0 < volume <= analytic_volume
    assert volume == pytest.approx(analytic_volume, rel=0.003)
    assert np.linalg.eigvalsh(inertia).min() > 0


def test_rounded_pad_removes_floor_blocking_corners_but_keeps_contact_width():
    size = np.array([0.012, 0.026, 0.055])
    vertices, _ = elliptical_prism_mesh(size)
    # Representative downward direction in the pad frame during the recorded
    # grasp: the old box corners struck the mat before the jaws could squeeze.
    downward = np.array([0.0766, -0.8835, -0.4621])
    downward /= np.linalg.norm(downward)
    box_support = np.dot(size / 2, np.abs(downward))
    rounded_support = (vertices @ downward).max()
    assert box_support - rounded_support > 0.006
    np.testing.assert_allclose(np.ptp(vertices[:, 0]), 0.012, atol=1e-12)


def _read_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + 50 * count
    records = [struct.unpack_from("<12fH", data, 84 + 50 * i) for i in range(count)]
    assert all(record[-1] == 0 for record in records)
    normals = np.array([record[:3] for record in records])
    triangles = np.array([record[3:12] for record in records]).reshape(-1, 3, 3)
    vertices, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, indices.reshape(-1, 3), normals


def _numbers(element, attribute):
    return np.fromstring(element.attrib[attribute], sep=" ")


@pytest.mark.parametrize("shape", [None, "box", "ellipse"])
@pytest.mark.parametrize("stroke", [0.070, 0.085])
def test_gripper_urdf_variants_preserve_mass_units_and_physical_aperture(
    tmp_path, shape, stroke
):
    source = Path(__file__).resolve().parents[1] / "assets/sim/ur3/ur3_cb3.urdf"
    original = source.read_bytes()
    gripper = {
        "housing_size": [0.065, 0.048, 0.070],
        "pad_size": [0.012, 0.026, 0.055],
        "pad_center_z": 0.16556,
        "stroke": stroke,
        "yaw": -0.377,
    }
    if shape is not None:
        gripper["contact_shape"] = shape
    destination = tmp_path / "generated" / "gripper.urdf"
    assert gripper_urdf(source, destination, {"gripper": gripper}) == destination
    assert source.read_bytes() == original
    root = ET.parse(destination).getroot()
    aperture_edges = {"closed": [], "open": []}
    tool_mass = 0
    for name, expected_mass in [
        ("gripper_housing", 0.94),
        ("left_pad", 0.13),
        ("right_pad", 0.13),
    ]:
        link = root.find(f"link[@name='{name}']")
        mass = float(link.find("inertial/mass").attrib["value"])
        assert mass == pytest.approx(expected_mass)
        tool_mass += mass
        values = link.find("inertial/inertia").attrib
        inertia = np.array(
            [
                [float(values["ixx"]), float(values["ixy"]), float(values["ixz"])],
                [float(values["ixy"]), float(values["iyy"]), float(values["iyz"])],
                [float(values["ixz"]), float(values["iyz"]), float(values["izz"])],
            ]
        )
        eigenvalues = np.linalg.eigvalsh(inertia)
        assert eigenvalues.min() > 0
        assert 2 * eigenvalues.max() <= eigenvalues.sum()

        pad = name.endswith("_pad")
        size = np.array(gripper["pad_size" if pad else "housing_size"])
        if shape == "ellipse" and pad:
            visual = link.find("visual/geometry/mesh")
            collision = link.find("collision/geometry/mesh")
            assert visual.attrib == collision.attrib
            np.testing.assert_array_equal(_numbers(collision, "scale"), [1, 1, 1])
            mesh_path = Path(collision.attrib["filename"])
            assert mesh_path.is_absolute() and mesh_path.is_file()
            vertices, faces, normals = _read_stl(mesh_path)
            _assert_closed_convex_mesh(vertices, faces, size, tolerance=5e-9)
            points = vertices[faces]
            geometric_normals = np.cross(
                points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
            )
            geometric_normals /= np.linalg.norm(geometric_normals, axis=1)[:, None]
            np.testing.assert_allclose(normals, geometric_normals, atol=1e-5)
            volume, integrated_inertia = _signed_volume_and_inertia(
                vertices, faces, mass
            )
            assert volume == pytest.approx(np.prod(size) * np.pi / 4, rel=0.003)
            # Independent solid integration catches mm/m and incorrect axis
            # scaling in inertia, allowing for polygonal approximation error.
            np.testing.assert_allclose(
                inertia, integrated_inertia, rtol=0.006, atol=1e-12
            )
        else:
            for kind in ("visual", "collision"):
                np.testing.assert_allclose(
                    _numbers(link.find(f"{kind}/geometry/box"), "size"), size
                )
                assert link.find(f"{kind}/geometry/mesh") is None

        if not pad:
            continue
        joint = root.find(f"joint[@name='{name.removesuffix('_pad')}_finger_joint']")
        assert joint.attrib["type"] == "prismatic"
        assert joint.find("parent").attrib["link"] == "gripper_housing"
        assert joint.find("child").attrib["link"] == name
        origin = _numbers(joint.find("origin"), "xyz")
        axis = _numbers(joint.find("axis"), "xyz")
        np.testing.assert_allclose(np.linalg.norm(axis), 1)
        np.testing.assert_allclose(origin[1:], [0, gripper["pad_center_z"]])
        limits = joint.find("limit").attrib
        assert float(limits["effort"]) > 0
        assert 0 < float(limits["velocity"]) <= 0.2
        for label, limit in [("closed", "lower"), ("open", "upper")]:
            center = origin + axis * float(limits[limit])
            # Inner contact surfaces, not pad centers, define the aperture.
            inner_x = center[0] - np.sign(axis[0]) * size[0] / 2
            aperture_edges[label].append(inner_x)

    assert tool_mass == pytest.approx(1.20)
    assert np.ptp(aperture_edges["closed"]) == pytest.approx(0, abs=1e-12)
    assert np.ptp(aperture_edges["open"]) == pytest.approx(stroke, abs=1e-12)
    for mesh in root.findall(".//mesh"):
        assert Path(mesh.attrib["filename"]).is_absolute()
        assert Path(mesh.attrib["filename"]).is_file()


@pytest.mark.parametrize(
    "size",
    [
        [0, 0.026, 0.055],
        [-0.012, 0.026, 0.055],
        [0.012, float("nan"), 0.055],
        [0.012, 0.026],
    ],
)
def test_nonphysical_collision_dimensions_are_rejected(size):
    with pytest.raises(ValueError, match="finite positive dimensions"):
        elliptical_prism_mesh(size)


@pytest.mark.parametrize("approximation", ["convexHull", "convexDecomposition"])
def test_forearm_cooking_override_preserves_other_colliders_and_contact_pairs(
    approximation,
):
    pytest.importorskip(
        "pxr", reason="CPU USD bindings optional; Isaac is not required"
    )
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Robot")
    for link in ("forearm_link", "wrist_2_link"):
        UsdGeom.Xform.Define(stage, f"/Robot/{link}")
        for role in ("visual", "collision"):
            mesh = UsdGeom.Mesh.Define(stage, f"/Robot/{link}/{role}")
            if role == "collision":
                UsdPhysics.CollisionAPI.Apply(
                    mesh.GetPrim()
                ).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(
                    mesh.GetPrim()
                ).CreateApproximationAttr("convexHull")
    paths = set_forearm_collision_approximation(stage, "/Robot", approximation)
    assert paths == ["/Robot/forearm_link/collision"]
    for link in ("forearm_link", "wrist_2_link"):
        prim = stage.GetPrimAtPath(f"/Robot/{link}/collision")
        expected = approximation if link == "forearm_link" else "convexHull"
        assert (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == expected
        )
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        assert not prim.HasAPI(UsdPhysics.FilteredPairsAPI)
        assert not stage.GetPrimAtPath(f"/Robot/{link}/visual").HasAPI(
            UsdPhysics.MeshCollisionAPI
        )
