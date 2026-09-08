"""Offline contract checks for supplier geometry before Isaac consumes it."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/sim/dmtac_w2l"


def load():
    return (json.loads((ASSETS / "geometry.json").read_text()),
            json.loads((ASSETS / "PROVENANCE.json").read_text()))


def mesh(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + count * 50
    return np.array([struct.unpack_from("<12fH", data, 84 + 50 * i)[3:12]
                     for i in range(count)]).reshape(-1, 3, 3)


def test_source_generator_and_export_bindings():
    geometry, provenance = load()
    for name in ["geometry.json", provenance["generator"]]:
        path = ASSETS / name if name == "geometry.json" else ROOT / name
        key = "geometry_sha256" if name == "geometry.json" else "generator_sha256"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[key]
    for path, row in provenance["meshes"].items():
        data = (ASSETS / path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        assert len(data) == row["bytes"]
    assert provenance["sources"]["sensor"]["sha256"] == "d54358f2f0222b66ab1eae314c4a0af22eb326303eadebbe645df490f21c690d"
    assert len(provenance["sensor_solids"]) == 41


def test_metre_conversion_and_supplier_housing_dimensions():
    geometry, _ = load()
    transform = np.array(geometry["canonical_frame"]["step_to_canonical_mm"])
    assert np.allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3))
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1)
    for part in geometry["parts"]:
        points = mesh(ROOT / part["visual_mesh"]).reshape(-1, 3)
        source = part["source_step_bounds"]
        # Canonical axes are a proper cyclic permutation; compare CAD extents
        # with exported float32 STL in metres, allowing stated tessellation.
        expected = np.array(source["size_mm"])[[2, 0, 1]] / 1000
        assert np.allclose(np.ptp(points, axis=0), expected, atol=.00011, rtol=0), part["id"]
    housing = next(p for p in geometry["parts"] if p["id"] == "housing")
    extent = np.ptp(mesh(ROOT / housing["visual_mesh"]).reshape(-1, 3), axis=0)
    assert np.allclose(extent, [.031, .046, .07614], atol=.00011, rtol=0)


def test_distinct_contact_mass_and_positive_inertia():
    geometry, _ = load()
    gels = [p for p in geometry["parts"] if p["role"] == "gel"]
    assert len(gels) == 1
    assert gels[0]["source_sensor_solid_indices"] == [6]
    assert np.isclose(sum(p["mass_kg"] for p in geometry["parts"] if p["role"] != "adapter"), .131)
    for part in geometry["parts"]:
        inertia = np.asarray(part["inertia_kg_m2"])
        assert np.allclose(inertia, inertia.T, atol=1e-15)
        assert np.linalg.eigvalsh(inertia).min() > 0
    assert np.isclose(geometry["gel"]["front_plane_x_m"], -geometry["gel"]["thickness_m"] / 2, atol=2e-10)
    assert "hypothesis" in geometry["gel"]["active_area_location_status"]


def test_collision_manifolds_convex_budget_and_separate_cover():
    geometry, provenance = load()
    for part in geometry["parts"]:
        for path in part["collision_meshes"]:
            triangles = mesh(ROOT / path)
            vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
            faces = inverse.reshape(-1, 3)
            assert 4 <= len(vertices) <= 240, path
            edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
            _, counts = np.unique(edges, axis=0, return_counts=True)
            assert np.all(counts == 2), path
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            signed = vertices @ normals.T - np.einsum("ij,ij->i", normals, triangles[:, 0])
            # Sub-micrometre support error allows skinny facets after binary
            # float32 export, well below the 25 micrometre CAD hull tolerance.
            assert signed.max() < 1e-6, path
            row = provenance["meshes"][str(Path(path).relative_to("assets/sim/dmtac_w2l"))]
            assert row["maximum_original_vertex_support_plane_error_m"] <= .000025
    cover = next(p for p in geometry["parts"] if p["id"] == "cover")
    assert len(cover["collision_meshes"]) > 1
    assert all(provenance["meshes"][str(Path(p).relative_to("assets/sim/dmtac_w2l"))]["method"] == "convex_hull_of_CAD_clipped_solid"
               for p in cover["collision_meshes"])
