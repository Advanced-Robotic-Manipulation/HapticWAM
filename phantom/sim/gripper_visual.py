"""Detailed Robotiq 2F85 visual skin; leaves the simulator's contacts unchanged.

Loads the pinned historical ROS-Industrial DAE meshes with stdlib XML + numpy,
without an asset converter or ROS. All created prims are visual-only. Original
nominal pads are omitted because the experiment has custom DM tactile pads.
Joint angles use nominal URDF mimic linkage, with an estimated closure mapping;
this module does not claim calibrated gripper kinematics or compliance.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NS = {"d": "http://www.collada.org/2005/11/COLLADASchema"}


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    result = np.eye(4)
    result[:3, :3] = (
        np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross
    )
    return result


def _origin(element):
    result = np.eye(4)
    if element is None:
        return result
    result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    result[:3, :3] = (
        _rotation([0, 0, 1], yaw)
        @ _rotation([0, 1, 0], pitch)
        @ _rotation([1, 0, 0], roll)
    )[:3, :3]
    return result


def load_dae_meshes(path):
    """Return triangulated DAE geometry dictionaries, in the file's units.

    Handles indexed normals and vertex positions for the pinned Robotiq assets.
    Their visual scenes contain only identity nodes. Reject other transforms
    rather than silently giving an incorrectly scaled or transformed asset.
    """
    root = ET.parse(path).getroot()
    scene = root.find("d:library_visual_scenes", _NS)
    if scene is not None:
        for name in ("matrix", "translate", "rotate", "scale"):
            if scene.findall(".//d:" + name, _NS):
                raise ValueError(f"{path}: unexpected visual-node transform {name}")
    effects = {}
    for effect in root.findall("d:library_effects/d:effect", _NS):
        color = effect.find(".//d:diffuse/d:color", _NS)
        effects[effect.get("id")] = (
            np.fromstring(color.text, sep=" ")[:3]
            if color is not None
            else np.array([0.08] * 3)
        )
    materials = {}
    for material in root.findall("d:library_materials/d:material", _NS):
        instance = material.find("d:instance_effect", _NS)
        materials[material.get("id")] = effects.get(
            instance.get("url", "")[1:], np.array([0.08] * 3)
        )
    bindings = {}
    for instance in root.findall(".//d:instance_geometry", _NS):
        bindings[instance.get("url", "")[1:]] = {
            entry.get("symbol"): materials.get(
                entry.get("target", "")[1:], np.array([0.08] * 3)
            )
            for entry in instance.findall(".//d:instance_material", _NS)
        }
    results = []
    for geometry in root.findall("d:library_geometries/d:geometry", _NS):
        mesh = geometry.find("d:mesh", _NS)
        sources = {}
        for source in mesh.findall("d:source", _NS):
            array = source.find("d:float_array", _NS)
            accessor = source.find("d:technique_common/d:accessor", _NS)
            stride = int(accessor.get("stride", "3"))
            sources[source.get("id")] = np.fromstring(array.text, sep=" ").reshape(
                -1, stride
            )[:, :3]
        vertices = {
            vertex.get("id"): vertex.find("d:input[@semantic='POSITION']", _NS).get(
                "source"
            )[1:]
            for vertex in mesh.findall("d:vertices", _NS)
        }
        if mesh.findall("d:polylist", _NS) or mesh.findall("d:polygons", _NS):
            raise ValueError(f"{path}: only pinned triangulated meshes are supported")
        for triangle in mesh.findall("d:triangles", _NS):
            inputs = triangle.findall("d:input", _NS)
            stride = max(int(entry.get("offset", "0")) for entry in inputs) + 1
            packed = np.fromstring(
                triangle.find("d:p", _NS).text, sep=" ", dtype=np.int32
            ).reshape(-1, stride)
            vertex = next(
                entry for entry in inputs if entry.get("semantic") == "VERTEX"
            )
            points = sources[vertices[vertex.get("source")[1:]]]
            indices = packed[:, int(vertex.get("offset", "0"))]
            if len(indices) % 3 or indices.min() < 0 or indices.max() >= len(points):
                raise ValueError(f"{path}: malformed triangle indices")
            normal_input = next(
                (entry for entry in inputs if entry.get("semantic") == "NORMAL"), None
            )
            normals = None
            if normal_input is not None:
                normals = sources[normal_input.get("source")[1:]][
                    packed[:, int(normal_input.get("offset", "0"))]
                ]
            results.append(
                {
                    "name": geometry.get("id"),
                    "points": points,
                    "indices": indices,
                    "normals": normals,
                    "color": bindings.get(geometry.get("id"), {}).get(
                        triangle.get("material"), np.array([0.08] * 3)
                    ),
                }
            )
    if not results:
        raise ValueError(f"{path}: no visual triangles found")
    return results


@dataclass
class _Joint:
    child: str
    parent: str
    origin: np.ndarray
    axis: np.ndarray
    multiplier: float
    offset: float


class RobotiqVisual:
    """Use build(), then update(recorded_or_measured_normalized_closure)."""

    def __init__(self, operations, joints, root_link, pad_touch_command):
        self._operations = operations
        self._joints = joints
        self._root_link = root_link
        self.pad_touch_command = float(pad_touch_command)
        self.last_closure = 0.0

    def link_matrices(self, closure):
        closure = float(closure)
        if not math.isfinite(closure):
            raise ValueError("closure must be finite")
        # Nominal ROS gripper angle .8 rad at close; rig's normalized .9 is pad touch.
        theta = 0.8 * float(np.clip(closure / self.pad_touch_command, 0, 1))
        transforms = {self._root_link: np.eye(4)}
        for joint in self._joints:
            local = joint.origin @ _rotation(
                joint.axis, joint.multiplier * theta + joint.offset
            )
            transforms[joint.child] = transforms[joint.parent] @ local
        return transforms

    def update(self, closure):
        from pxr import Gf

        matrices = self.link_matrices(closure)
        for name, operation in self._operations.items():
            # Gf uses row vectors; our FK uses column vectors.
            operation.Set(Gf.Matrix4d(matrices[name].T.tolist()))
        self.last_closure = float(closure)


def build(stage, tool_path, assets_dir, *, yaw_rad=-math.pi / 2, pad_touch_command=0.9):
    """Build detailed visual geometry below tool0, return an animated skin.

    assets_dir points to assets/sim/robotiq. Nominal fingers close along Y;
    yaw=-pi/2 aligns their closure with the simulator's X sliding pads. Add the
    configured physical gripper yaw to this value if the rig yaw is changed.
    This function does not hide or mutate any existing prims.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    assets = Path(assets_dir)
    urdf = ET.parse(assets / "robotiq_2f85.urdf").getroot()
    if not stage.GetPrimAtPath(str(tool_path)).IsValid():
        raise ValueError(f"No parent tool prim: {tool_path}")
    if pad_touch_command <= 0:
        raise ValueError("pad_touch_command must be positive")
    visual_path = str(tool_path).rstrip("/") + "/RobotiqVisual"
    if stage.GetPrimAtPath(visual_path).IsValid():
        raise ValueError(f"Visual already exists at {visual_path}")
    visual = UsdGeom.Xform.Define(stage, visual_path)
    visual.AddRotateZOp().Set(math.degrees(yaw_rad))
    visual.GetPrim().SetCustomDataByKey("phantom:visualOnly", True)
    visual.GetPrim().SetCustomDataByKey(
        "phantom:closureMapping", "estimated: joint_angle=.8*clip(closure/.9,0,1)"
    )

    materials = {}

    def material(color, metallic=0.25):
        key = tuple(float(v) for v in color) + (metallic,)
        if key not in materials:
            path = visual_path + f"/Looks/Material{len(materials)}"
            mat = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.32)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            mat.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            materials[key] = mat
        return materials[key]

    operations = {}
    mesh_cache = {}
    for link in urdf.findall("link"):
        name = link.get("name")
        transform = UsdGeom.Xform.Define(stage, visual_path + "/" + name)
        operations[name] = transform.AddTransformOp()
        if name.endswith("_pad"):
            continue  # User's custom tactile geometry provides the visible contact pads.
        for i, source in enumerate(link.findall("visual")):
            mesh_spec = source.find("geometry/mesh")
            if mesh_spec is None:
                continue
            mesh_file = (assets / mesh_spec.get("filename")).resolve()
            if mesh_file not in mesh_cache:
                mesh_cache[mesh_file] = load_dae_meshes(mesh_file)
            scale = np.fromstring(mesh_spec.get("scale", "1 1 1"), sep=" ")
            origin = _origin(source.find("origin"))
            for j, data in enumerate(mesh_cache[mesh_file]):
                mesh_path = visual_path + f"/{name}/Mesh{i}_{j}"
                mesh = UsdGeom.Mesh.Define(stage, mesh_path)
                points = (data["points"] * scale) @ origin[:3, :3].T + origin[:3, 3]
                mesh.CreatePointsAttr(points.astype(np.float32).tolist())
                mesh.CreateFaceVertexCountsAttr([3] * (len(data["indices"]) // 3))
                mesh.CreateFaceVertexIndicesAttr(data["indices"].tolist())
                mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                mesh.CreateDoubleSidedAttr(True)
                if data["normals"] is not None:
                    normal = (data["normals"] / scale) @ origin[:3, :3].T
                    normal /= np.maximum(
                        np.linalg.norm(normal, axis=1, keepdims=True), 1e-12
                    )
                    mesh.CreateNormalsAttr(normal.astype(np.float32).tolist())
                    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
                # Real fingers are black/anodized; base DAE keeps contrasting silver hardware.
                color = (
                    data["color"]
                    if name == "robotiq_arg2f_base_link"
                    else np.array([0.055, 0.06, 0.063])
                )
                if "outer_knuckle" in name:
                    color = np.array([0.42, 0.45, 0.47])
                metal = 0.55 if np.mean(color) > 0.25 else 0.15
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                    material(color, metal)
                )

    joints, pending = [], list(urdf.findall("joint"))
    root_link = "robotiq_arg2f_base_link"
    ready = {root_link}
    while pending:
        progressed = False
        for joint in pending[:]:
            parent, child = (
                joint.find("parent").get("link"),
                joint.find("child").get("link"),
            )
            if parent not in ready:
                continue
            axis = joint.find("axis")
            axis = (
                np.fromstring(axis.get("xyz"), sep=" ")
                if axis is not None
                else np.array([1.0, 0, 0])
            )
            mimic = joint.find("mimic")
            multiplier = 0.0 if joint.get("type") == "fixed" else 1.0
            offset = 0.0
            if mimic is not None:
                if mimic.get("joint") != "finger_joint":
                    raise ValueError("Unsupported nominal gripper mimic target")
                multiplier = float(mimic.get("multiplier", "1"))
                offset = float(mimic.get("offset", "0"))
            joints.append(
                _Joint(
                    child,
                    parent,
                    _origin(joint.find("origin")),
                    axis,
                    multiplier,
                    offset,
                )
            )
            ready.add(child)
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("Unresolved gripper URDF joint tree")
    skin = RobotiqVisual(operations, joints, root_link, pad_touch_command)
    skin.update(0.0)
    return skin
