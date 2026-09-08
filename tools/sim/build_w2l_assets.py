#!/usr/bin/env python3
"""Extract supplier DM-Tac W2L STEP geometry; no Isaac or robot connection.

Rebuild in an isolated environment::

    uv run --no-project --with cadquery==2.8.0 python tools/sim/build_w2l_assets.py

STEP inputs are deliberately not redistributed with these derivative meshes.
Their exact supplier paths and hashes are recorded in PROVENANCE.json. The
reference bracket registration is a geometric hypothesis, not an as-built fit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

import cadquery as cq
import numpy as np
from scipy.spatial import ConvexHull
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "artifacts/isaac_waffles/gripper_replication_audit_20260908/vendor_sources"
CAD = {
    "sensor": ("数模/传感器数模20260112/DM-Tac W2L_asm.stp",
               "d54358f2f0222b66ab1eae314c4a0af22eb326303eadebbe645df490f21c690d"),
    "base": ("数模/适配不同夹爪的参考连接件20260112/大号连接件/大号连接件/DM-Tac W2L base.STEP",
             "d883a7a2b8db49323c17893eb3a2c37290cb2674f9448280f762366a21d3df10"),
    "adapter": ("数模/适配不同夹爪的参考连接件20260112/大号连接件/大号连接件/ROBOTIQ 2F.STEP",
                "be085af9cc6c446a1a69e6a699195057325d098b2b16c1764f56b9ecd86036b9"),
}
ROTATION = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounds(s):
    b = s.BoundingBox()
    return {"min_mm": [b.xmin, b.ymin, b.zmin],
            "max_mm": [b.xmax, b.ymax, b.zmax],
            "size_mm": [b.xlen, b.ylen, b.zlen]}


def name(label):
    attr = TDataStd_Name()
    return str(attr.Get().ToExtString()) if label.FindAttribute(TDataStd_Name.GetID_s(), attr) else ""


def named_sensor(path):
    """Traverse XCAF instances without dropping duplicate screw names."""
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"Cannot read STEP: {path}")
    doc = TDocStd_Document(TCollection_ExtendedString("XCAF"))
    if not reader.Transfer(doc):
        raise ValueError("STEP transfer failed")
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    tool.GetFreeShapes(roots)
    result = []

    def visit(label, location):
        components = TDF_LabelSequence()
        tool.GetComponents_s(label, components)
        for component in components:
            ref = TDF_Label()
            tool.GetReferredShape_s(component, ref)
            placement = location.Multiplied(tool.GetLocation_s(component))
            if tool.IsAssembly_s(ref):
                visit(ref, placement)
            else:
                shape = cq.Shape.cast(tool.GetShape_s(ref)).moved(cq.Location(placement))
                result.append((shape, {"index": len(result), "instance_name": name(component),
                                      "product_name": name(ref), "volume_mm3": shape.Volume(),
                                      **bounds(shape)}))
    for root in roots:
        visit(root, TopLoc_Location())
    return result


def triangles(shape, origin, tolerance):
    points, faces = shape.tessellate(tolerance, .15)
    points = (np.array([v.toTuple() for v in points]) - origin) @ ROTATION.T / 1000.
    result = points[np.asarray(faces, dtype=int)]
    valid = np.linalg.norm(np.cross(result[:, 1] - result[:, 0], result[:, 2] - result[:, 0]), axis=1) > 1e-16
    return result[valid]


def hull_triangles(tris):
    # ConvexHull keeps the surface's actual sampled envelope; no invented box.
    vertices = np.unique(np.round(tris.reshape(-1, 3), 12), axis=0)
    full = ConvexHull(vertices)
    vertices = vertices[full.vertices]
    # Keep actual surface vertices. Incrementally add the most violated support
    # point until every original hull point is within 0.025 mm. This avoids
    # exceeding PhysX's hull budget and relying on undocumented cook reduction.
    directions = np.array([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1]
                           for z in [-1, 0, 1] if (x, y, z) != (0, 0, 0)])
    selected = sorted(set(np.argmax(vertices @ directions.T, axis=0).tolist()))
    while True:
        hull = ConvexHull(vertices[selected])
        distances = vertices @ hull.equations[:, :3].T + hull.equations[:, 3]
        worst = distances.max(axis=1)
        error = float(worst.max())
        if error <= .000025:
            break
        if len(selected) >= 240:
            raise ValueError(f"Convex hull needs more than 240 vertices for 0.025 mm support error: {error}")
        index = int(worst.argmax())
        assert index not in selected
        selected.append(index)
    result = vertices[selected][hull.simplices].copy()
    normals = np.cross(result[:, 1] - result[:, 0], result[:, 2] - result[:, 0])
    reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    result[reverse] = result[reverse, ::-1]
    return result, float(hull.volume), {"original_hull_vertices": len(vertices),
                                      "export_hull_vertices": len(hull.vertices),
                                      "maximum_original_vertex_support_plane_error_m": error,
                                      "reduction": "retained original CAD vertices; support-plane error <=0.025mm; <=240 vertices"}


def write_stl(path, tris):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"PHANTOM supplier W2L derivative; metres; see PROVENANCE.json".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for points in tris:
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            normal /= np.linalg.norm(normal)
            f.write(struct.pack("<12fH", *normal, *points.ravel(), 0))
    # Validate the actual exported float32 bytes, not just in-memory vertices.
    data = path.read_bytes()
    assert len(data) == 84 + 50 * len(tris)
    saved = np.array([struct.unpack_from("<12fH", data, 84 + 50 * i)[3:12]
                      for i in range(len(tris))]).reshape(-1, 3)
    return {"sha256": sha(path), "bytes": len(data), "triangles": len(tris),
            "bounds_m": {"min": saved.min(0).tolist(), "max": saved.max(0).tolist(),
                         "size": np.ptp(saved, axis=0).tolist()}}


def clipped_hulls(shape, origin, tolerance, split):
    """Use separate CAD-clipped cells so a cover hull cannot seal its aperture."""
    if not split:
        tris, volume, audit = hull_triangles(triangles(shape, origin, tolerance))
        yield (tris, volume), {"method": "per_component_convex_hull", **audit}
        return
    b = bounds(shape)
    lo, hi = np.array(b["min_mm"]), np.array(b["max_mm"])
    # Optical area comes from manufacturer 27 mm width ×36 mm height. These
    # planes only partition the original solid, never enlarge its geometry.
    edges = [sorted(set([lo[0] - .01, origin[0] - 13.5, origin[0] + 13.5, hi[0] + .01])),
             sorted(set([lo[1] - .01, origin[1] - 18., origin[1] + 18., hi[1] + .01]))]
    for i, (xmin, xmax) in enumerate(zip(edges[0], edges[0][1:])):
        for j, (ymin, ymax) in enumerate(zip(edges[1], edges[1][1:])):
            cutter = cq.Solid.makeBox(xmax - xmin, ymax - ymin, hi[2] - lo[2] + .02,
                                     cq.Vector(xmin, ymin, lo[2] - .01))
            intersection = shape.intersect(cutter)
            for k, solid in enumerate(intersection.Solids()):
                if solid.Volume() > 1e-6:
                    tris, volume, audit = hull_triangles(triangles(solid, origin, tolerance))
                    yield (tris, volume), {
                        "method": "convex_hull_of_CAD_clipped_solid", "grid_cell": [i, j, k],
                        "clip_step_xy_mm": [xmin, xmax, ymin, ymax], "source_piece_volume_mm3": solid.Volume(), **audit}


def mount_geometry(shapes, origin):
    # Select the independent bottom-cap mounting-hole row by exact radius,
    # direction and manufacturer ±4/±12 coordinates, not by mesh centroids.
    holes = []
    for face in shapes[3].Faces():
        if face.geomType() != "CYLINDER":
            continue
        cylinder = BRepAdaptor_Surface(face.wrapped).Cylinder()
        p, a = cylinder.Location(), cylinder.Axis().Direction()
        xyz = np.array([p.X(), p.Y(), p.Z()])
        if (abs(a.Y()) > .999 and abs(xyz[1] + 45.88) < .001 and
                (abs(abs(xyz[0]) - 4) < .001 or abs(abs(xyz[0]) - 12) < .001)):
            row = {"axis_centre_step_mm": xyz.tolist(), "radius_mm": cylinder.Radius()}
            if not any(np.allclose(xyz, h["axis_centre_step_mm"], atol=1e-6) for h in holes):
                holes.append(row)
    assert len(holes) == 4
    row_z = float(np.mean([h["axis_centre_step_mm"][2] for h in holes]))
    base_T = np.eye(4)
    base_T[:3, :3] = np.diag([1, -1, -1])
    base_T[:3, 3] = [0, -48.38, row_z]
    adapter_T = base_T.copy()
    adapter_T[:3, 3] += base_T[:3, :3] @ np.array([0, 2.5, 0])
    point_step = np.array([0., -57.38, row_z - 2.6])
    return base_T, adapter_T, {
        "status": "supplier_reference_hole_and_mating_plane_registration; physical_installation_unverified",
        "base_to_sensor_step_mm": base_T.tolist(), "adapter_to_sensor_step_mm": adapter_T.tolist(),
        "adapter_to_base_mm": [[1, 0, 0, 0], [0, 1, 0, 2.5], [0, 0, 1, 0], [0, 0, 0, 1]],
        "sensor_bottom_holes": holes,
        "evidence": "Sensor/base ±12 M3/clearance and ±4 dowels match; adapter/base transverse Ø3 axes at X±4.875,Y5.75 align after 2.5 mm slide and mating face alignment.",
        "gripper_facing_adapter_hole_row_center_step_mm": point_step.tolist(),
        "gripper_facing_adapter_hole_row_center_canonical_m": ((point_step-origin) @ ROTATION.T/1000).tolist(),
        "gripper_facing_adapter_plane_normal_canonical": [0, 0, -1],
        "gripper_side_holes": {"small_axis_x_step_mm": [-4.5, 4.5], "small_radius_mm": 1.05,
                               "central_radius_mm": 2.6, "counterbore_radius_mm": 5.1},
        "inner_finger_to_adapter": None,
        "inner_finger_to_adapter_status": "Requires matching supplier adapter pattern to mounted finger; no stock-pad-centroid equivalence assumed.",
    }


def nominal_interface_audit():
    """Measure hole loops on the pinned 2F85 finger face; do not force a mate."""
    path = ROOT / "assets/sim/robotiq/meshes/visual/robotiq_arg2f_85_inner_finger.dae"
    ns = {"d": "http://www.collada.org/2005/11/COLLADASchema"}
    meshes = []
    for element in ET.parse(path).getroot().findall("d:library_geometries/d:geometry/d:mesh", ns):
        sources = {}
        for source in element.findall("d:source", ns):
            array = source.find("d:float_array", ns)
            stride = int(source.find("d:technique_common/d:accessor", ns).get("stride", "3"))
            sources[source.get("id")] = np.fromstring(array.text, sep=" ").reshape(-1, stride)[:, :3]
        vertices = {v.get("id"): v.find("d:input[@semantic='POSITION']", ns).get("source")[1:]
                    for v in element.findall("d:vertices", ns)}
        for triangle in element.findall("d:triangles", ns):
            inputs = triangle.findall("d:input", ns)
            stride = max(int(i.get("offset", "0")) for i in inputs) + 1
            vertex = next(i for i in inputs if i.get("semantic") == "VERTEX")
            packed = np.fromstring(triangle.find("d:p", ns).text, sep=" ", dtype=int).reshape(-1, stride)
            ids = packed[:, int(vertex.get("offset", "0"))].reshape(-1, 3)
            meshes.append(sources[vertices[vertex.get("source")[1:]]][ids])
    tris = np.concatenate(meshes)
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    selected = tris[(normal[:, 1] < -.999) & (abs(tris[:, :, 1].mean(1) + 18.902) < .01)]
    points, indices = np.unique(np.round(selected.reshape(-1, 3), 4), axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    counts = Counter(tuple(sorted(edge)) for f in faces for edge in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])])
    adjacency = defaultdict(set)
    for (a, b), count in counts.items():
        if count == 1:
            adjacency[a].add(b)
            adjacency[b].add(a)
    circles = []
    while adjacency:
        todo, component = [next(iter(adjacency))], []
        while todo:
            i = todo.pop()
            if i in adjacency:
                component.append(i)
                todo.extend(adjacency.pop(i))
        loop = points[component]
        fit = np.linalg.lstsq(np.c_[2 * loop[:, 0], 2 * loop[:, 2], np.ones(len(loop))],
                             loop[:, 0]**2 + loop[:, 2]**2, rcond=None)[0]
        radius = np.sqrt(fit[2] + fit[0]**2 + fit[1]**2)
        residual = np.linalg.norm(loop[:, [0, 2]] - fit[:2], axis=1) - radius
        rms = float(np.sqrt(np.mean(residual**2)))
        if rms < .001:
            circles.append({"center_inner_finger_mm": [fit[0], float(loop[:, 1].mean()), fit[1]],
                            "radius_mm": float(radius), "circle_fit_rms_mm": rms,
                            "boundary_vertices": len(loop)})
    circles.sort(key=lambda row: row["center_inner_finger_mm"][2])
    assert len(circles) == 3
    separation = float(np.linalg.norm(np.array(circles[0]["center_inner_finger_mm"]) - circles[2]["center_inner_finger_mm"]))
    assert abs(separation - 16) < .01
    return {"nominal_mesh": str(path.relative_to(ROOT)), "sha256": sha(path),
            "method": "Boundary loops of planar -Y pad face; 0.0001 mm vertex welding then least-squares XZ circle fit",
            "hole_circles": circles, "outer_pair_separation_mm": separation,
            "supplier_adapter_small_pair_separation_mm": 9.,
            "rigid_pattern_match": False,
            "conclusion": "Supplier generic ROBOTIQ 2F adapter does not match the pinned 2F85 stock pad-face hole pattern under a rigid transform. The mounted bracket, attachment interface and physical variant require independent verification; no exact inner-finger-to-sensor transform is supplied."}


def build(source, out, tolerance):
    if not 0 < tolerance <= .15:
        raise ValueError("Use a tessellation tolerance between 0 and 0.15 mm")
    paths = {k: source / rel for k, (rel, _) in CAD.items()}
    for k, path in paths.items():
        if sha(path) != CAD[k][1]:
            raise ValueError(f"Supplier source hash mismatch: {path}")
        if 'SI_UNIT(.MILLI.,.METRE.)' not in ''.join(path.read_text(errors='replace').split()):
            raise ValueError(f"Missing millimetre declaration: {path}")
    named = named_sensor(paths["sensor"])
    assert len(named) == 41
    shapes = [s for s, _ in named]
    for index, token in [(3, "底盖"), (5, "透明层"), (6, "耐磨层"), (34, "外壳"), (35, "上盖")]:
        assert token in named[index][1]["instance_name"], (index, named[index][1])
    assert np.allclose(bounds(shapes[34])["size_mm"], [46, 76.14, 31], atol=.02, rtol=0)
    gel = shapes[6]
    front = max((f for f in gel.Faces() if f.geomType() == "PLANE" and f.normalAt().z < -.999), key=lambda f: f.Area())
    gb = bounds(gel)
    origin = np.array(front.Center().toTuple())
    origin[2] = (gb["min_mm"][2] + gb["max_mm"][2]) / 2
    canonical_T = np.eye(4)
    canonical_T[:3, :3] = ROTATION
    canonical_T[:3, 3] = -ROTATION @ origin
    base_T, adapter_T, mount = mount_geometry(shapes, origin)
    mount["nominal_2f85_interface_audit"] = nominal_interface_audit()
    base = cq.importers.importStep(str(paths["base"])).val()
    adapter = cq.importers.importStep(str(paths["adapter"])).val()
    base_in_sensor = base.rotate((0, 0, 0), (1, 0, 0), 180).translate(tuple(base_T[:3, 3]))
    adapter_in_sensor = adapter.rotate((0, 0, 0), (1, 0, 0), 180).translate(tuple(adapter_T[:3, 3]))
    mount["CAD_intersection_volumes_mm3"] = {
        "base_adapter": base.intersect(adapter.translate((0, 2.5, 0))).Volume(),
        "sensor_base": cq.Compound.makeCompound(shapes).intersect(base_in_sensor).Volume(),
        "sensor_adapter": cq.Compound.makeCompound(shapes).intersect(adapter_in_sensor).Volume()}
    assert all(abs(v) < .001 for v in mount["CAD_intersection_volumes_mm3"].values())
    total_volume = sum(s.Volume() for s in shapes)
    specs = [
        ("gel", "gel", [6], gel, False),
        ("transparent_layer", "body", [5], shapes[5], False),
        ("housing", "body", [34], shapes[34], True),
        ("cover", "body", [35], shapes[35], True),
        ("bottom_cap", "body", [3], shapes[3], False),
        ("connector", "body", [32], shapes[32], False),
        ("cable_stub", "body", [33], shapes[33], False),
    ]
    used = {i for _, _, ids, _, _ in specs for i in ids}
    other_ids = [i for i in range(41) if i not in used]
    specs += [("internals_and_fasteners", "body", other_ids,
               cq.Compound.makeCompound([shapes[i] for i in other_ids]), None),
              ("reference_base", "adapter", [], base_in_sensor, False),
              ("reference_adapter", "adapter", [], adapter_in_sensor, False)]
    out.mkdir(parents=True, exist_ok=True)
    mesh_root = out / "meshes"
    records, parts = {}, []
    for part_id, role, ids, shape, split in specs:
        print(f"Export {part_id}", flush=True)
        cad_bounds = bounds(shape)  # Snapshot before OCCT meshing changes cached bounds.
        visual = mesh_root / f"{part_id}_visual.stl"
        record = write_stl(visual, triangles(shape, origin, tolerance))
        records[str(visual.relative_to(out))] = record
        collisions = []
        if split is not None:
            for i, ((tris, volume), detail) in enumerate(clipped_hulls(shape, origin, tolerance, split)):
                collision = mesh_root / f"{part_id}_collision_{i:02d}.stl"
                records[str(collision.relative_to(out))] = {**write_stl(collision, tris),
                    "convex_volume_m3": volume, **detail}
                collisions.append(str(collision.relative_to(ROOT)) if collision.is_relative_to(ROOT) else str(collision))
        # Supplier only provides total module mass. Distribute it by CAD volume
        # for numerical inertias; the component masses/materials are not measured.
        mass = .131 * sum(shapes[i].Volume() for i in ids) / total_volume if ids else shape.Volume() * 1.2e-6
        centre = (np.array(shape.Center().toTuple()) - origin) @ ROTATION.T / 1000
        # OCC supplies the volume integral about the solid COM in mm^5.
        # Divide by mm^3 volume, multiply mass and convert mm^2 to m^2.
        inertia = ROTATION @ np.array(cq.Shape.matrixOfInertia(shape)) @ ROTATION.T
        inertia *= mass / shape.Volume() * 1e-6
        assert np.linalg.eigvalsh(inertia).min() > 0
        parts.append({"id": part_id, "role": role,
                      "visual_mesh": str(visual.relative_to(ROOT)) if visual.is_relative_to(ROOT) else str(visual),
                      "collision_meshes": collisions, "mass_kg": mass,
                      "mass_status": "total_sensor_mass_131g_allocated_by_CAD_volume" if ids else "reference_adapter_density_assumed_1200kg_per_m3",
                      "source_sensor_solid_indices": ids, "source_step_bounds": cad_bounds,
                      "bounding_box_m": {k: record["bounds_m"][k] for k in ["min", "max"]},
                      "center_of_mass_m": centre.tolist(),
                      "inertia_kg_m2": inertia.tolist(),
                      "inertia_status": "exact CAD volume integral under stated homogeneous component mass allocation; not measured density distribution",
                      "origin": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
                      "color_rgb": [0.18, .19, .18] if role == "gel" else ([.75, .76, .78] if role == "adapter" else [.89, .90, .87])})
    assert abs(sum(p["mass_kg"] for p in parts if p["role"] != "adapter") - .131) < 1e-12
    geometry = {
        "schema_version": 1, "units": "metres", "parts": parts,
        "canonical_frame": {
            "axes": {"x": "depth away from inner contact", "y": "sensor width", "z": "distal toward rounded tip"},
            "origin_step_mm": origin.tolist(), "step_to_canonical_mm": canonical_T.tolist(),
            "mesh_conversion": "p_mesh_m=(R*p_STEP_mm+t)/1000; proper rotation det(R)=+1",
            "right_mesh_rotation_rpy_rad": [0, 0, float(np.pi)],
            "right_frame_note": "Keep left_pad/right_pad link axes aligned with gripper. Rotate right-side component geometry 180 degrees about local Z; left inner face -X, right inner face +X."},
        "gel": {"part_id": "gel", "supplier_name": named[6][1]["instance_name"],
                "thickness_m": gb["size_mm"][2] / 1000,
                "front_plane_x_m": float((front.Center().z - origin[2]) / 1000),
                "front_plane_center_m": [float((front.Center().z-origin[2])/1000), 0, 0],
                "active_area_yz_m": [.027, .036],
                "active_area_location_status": "manufacturer size; centred on CAD front planar-face centroid as optical registration hypothesis",
                "active_area_shape_status": "manufacturer dimensions alone do not identify support shape; existing 27x36 mm ellipse may be retained only as an uncalibrated optical comparison proxy, independent of the actual CAD wear-layer collision outline",
                "material_status": "CAD identifies wear layer; stiffness, friction, optical deformation and measured force law remain uncalibrated"},
        "mount_reference": mount,
        "mass": {"sensor_total_kg": .131, "source": "supplier W2L manual",
                 "allocation": "CAD volume proportions are an explicit inertial approximation; not measured material densities",
                 "reference_adapter_density_kg_m3_assumed": 1200},
        "collision_policy": "Separate convex hull per CAD part; housing and cover split by optical-area boundary planes before hulling. Optical cavity is not filled by one complete cover/assembly hull. Internals are visual and mass only.",
        "limitations": ["Reference assembly is not independently verified as mounted hardware.",
                        "Physical Robotiq 85 versus 140 is not resolved by the supplier adapter filename.",
                        "CAD contact dimensions do not calibrate actuator position to gap, gel compliance or tactile images.",
                        "Component colours are approximations; mesh geometry is supplier derived."]}
    (out / "geometry.json").write_text(json.dumps(geometry, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    provenance = {
        "schema_version": 1, "source_type": "independent_supplier_STEP_derivative",
        "source_host_root": "compute3:/home/physicalai/kngn_ws/tacticle_sensor",
        "sources": {k: {"relative_path": rel, "sha256": digest, "bytes": paths[k].stat().st_size}
                    for k, (rel, digest) in CAD.items()},
        "supplier_hardware_manual_sha256": "70d66247ace6a6a7ff6510f1b5d65892b90b3caefbf5c066025c68d1f9865f53",
        "generator": "tools/sim/build_w2l_assets.py", "generator_sha256": sha(Path(__file__)),
        "packages": {k: importlib.metadata.version(k) for k in ["cadquery", "cadquery-ocp", "numpy", "scipy"]},
        "tessellation_tolerance_mm": tolerance, "angular_tolerance_rad": .15,
        "sensor_solids": [row for _, row in named], "meshes": records,
        "geometry_sha256": sha(out / "geometry.json"),
        "license_status": "Supplier integration package present in workspace; original license not provided in inspected package. Preserve supplier attribution and check redistribution terms before broader reuse.",
        "validation": {"all_source_hashes_match": True, "all_STEP_inputs_millimetres": True,
                       "housing_matches_manual_0_02mm": True, "separate_named_contact_layer": True,
                       "sensor_part_masses_sum_131g": True, "reference_mount_intersections_below_0_001mm3": True}}
    assert all(sha(paths[k]) == digest for k, (_, digest) in CAD.items())
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "parts": len(parts), "meshes": len(records),
                      "gel": geometry["gel"], "frame": geometry["canonical_frame"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "assets/sim/dmtac_w2l")
    parser.add_argument("--tolerance-mm", type=float, default=.10)
    args = parser.parse_args()
    build(args.source_root.resolve(), args.out.resolve(), args.tolerance_mm)
