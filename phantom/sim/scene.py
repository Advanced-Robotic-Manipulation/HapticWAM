"""Build the evidence-based waffles bench in USD/PhysX (import inside Isaac).

Official UR3 geometry is imported from the pinned, locally vendored URDF.
Historical scenes retain their estimated sliding pads. The opt-in W2L model
uses supplier sensor CAD attached to a shared articulated finger mechanism.
Mounted registration and contact compliance remain explicitly qualified.
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from phantom.sim.geometry import bin_geometry, mount_plate_geometry, table_hole_centers
from phantom.sim.task_objects import object_config, object_identity, orientation_wxyz


def elliptical_prism_mesh(size, segments=48):
    """Convex pad proxy: flat contact faces on X, rounded perimeter in YZ.

    This is an image-informed shape hypothesis, not measured tactile CAD.
    Returns vertices and outward-wound triangles, with exact XYZ bounds.
    """
    half = np.asarray(size, dtype=float) / 2
    if half.shape != (3,) or np.any(half <= 0) or not np.isfinite(half).all():
        raise ValueError("pad size must contain three finite positive dimensions")
    if segments < 8 or segments % 4:
        raise ValueError("ellipse segments must be >=8 and divisible by four")
    theta = np.arange(segments) * (math.tau / segments)
    ring = np.c_[np.zeros(segments), half[1] * np.cos(theta), half[2] * np.sin(theta)]
    left, right = ring.copy(), ring.copy()
    left[:, 0], right[:, 0] = -half[0], half[0]
    vertices = np.r_[left, right, [[-half[0], 0, 0], [half[0], 0, 0]]]
    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        triangles.extend(
            [
                [2 * segments, j, i],
                [2 * segments + 1, segments + i, segments + j],
                [i, j, segments + j],
                [i, segments + j, segments + i],
            ]
        )
    return vertices, np.asarray(triangles, dtype=np.int32)


def _write_ellipse_stl(path, size):
    vertices, faces = elliptical_prism_mesh(size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"PHANTOM estimated elliptical pad proxy; metres".ljust(80, b" "))
        stream.write(struct.pack("<I", len(faces)))
        for face in faces:
            points = vertices[face]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            normal /= np.linalg.norm(normal)
            stream.write(struct.pack("<12fH", *normal, *points.reshape(-1), 0))
    return path.resolve()


def gripper_urdf(source: Path, destination: Path, cfg: dict) -> Path:
    """Make a separate URDF with a UR-base world and pad articulation."""
    tree = ET.parse(source)
    root = tree.getroot()
    root.find("joint[@name='base_joint']/origin").set("rpy", f"0 0 {math.pi}")
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))

    from phantom.sim.gripper_articulation import is_articulated, append_gripper_urdf

    if is_articulated(cfg):
        append_gripper_urdf(root, Path(__file__).resolve().parents[2], cfg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(tree, space="  ")
        tree.write(destination, encoding="utf-8", xml_declaration=True)
        return destination

    def link(name, mass, size, color, xyz=(0, 0, 0), mesh_path=None):
        l = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(l, "inertial")
        ET.SubElement(inertial, "mass", value=str(mass))
        ET.SubElement(inertial, "origin", xyz=" ".join(map(str, xyz)))
        x, y, z = size
        # Elliptical cylinder moments; the box branch retains the original model.
        inertia = (
            [
                mass * (y * y + z * z) / 16,
                mass * (x * x / 12 + z * z / 16),
                mass * (x * x / 12 + y * y / 16),
            ]
            if mesh_path
            else [
                mass * (y * y + z * z) / 12,
                mass * (x * x + z * z) / 12,
                mass * (x * x + y * y) / 12,
            ]
        )
        ET.SubElement(
            inertial,
            "inertia",
            ixx=str(inertia[0]),
            iyy=str(inertia[1]),
            izz=str(inertia[2]),
            ixy="0",
            ixz="0",
            iyz="0",
        )
        for kind in ("visual", "collision"):
            e = ET.SubElement(l, kind)
            ET.SubElement(e, "origin", xyz=" ".join(map(str, xyz)))
            geometry = ET.SubElement(e, "geometry")
            if mesh_path:
                ET.SubElement(geometry, "mesh", filename=str(mesh_path), scale="1 1 1")
            else:
                ET.SubElement(geometry, "box", size=" ".join(map(str, size)))
            if kind == "visual":
                mat = ET.SubElement(e, "material", name=f"{name}_material")
                ET.SubElement(mat, "color", rgba=" ".join(map(str, (*color, 1))))
        return l

    g = cfg["gripper"]
    contact_shape = g.get("contact_shape", "box")
    if contact_shape not in ("box", "ellipse"):
        raise ValueError("gripper.contact_shape must be 'box' or 'ellipse'")
    pad_mesh = None
    if contact_shape == "ellipse":
        pad_mesh = _write_ellipse_stl(
            destination.parent / "tactile_pad_ellipse.stl", g["pad_size"]
        )
        root.insert(
            0,
            ET.Comment(
                "Estimated elliptical tactile proxy; uncalibrated contact shape"
            ),
        )
    link(
        "gripper_housing", 0.94, g["housing_size"], (0.025, 0.030, 0.035), (0, 0, 0.047)
    )
    j = ET.SubElement(root, "joint", name="tool_gripper", type="fixed")
    ET.SubElement(j, "parent", link="tool0")
    ET.SubElement(j, "child", link="gripper_housing")
    # 2F85 grasp axis in tool XY is estimated from RGB and is configurable.
    ET.SubElement(j, "origin", xyz="0 0 0", rpy=f"0 0 {g.get('yaw', 0)}")
    for side, sign in (("left", 1), ("right", -1)):
        link(f"{side}_pad", 0.13, g["pad_size"], (0.83, 0.91, 0.94), mesh_path=pad_mesh)
        j = ET.SubElement(root, "joint", name=f"{side}_finger_joint", type="prismatic")
        ET.SubElement(j, "parent", link="gripper_housing")
        ET.SubElement(j, "child", link=f"{side}_pad")
        ET.SubElement(
            j, "origin", xyz=f"{sign * g['pad_size'][0] / 2} 0 {g['pad_center_z']}"
        )
        ET.SubElement(j, "axis", xyz=f"{sign} 0 0")
        ET.SubElement(
            j,
            "limit",
            lower="0",
            upper=str(g["stroke"] / 2),
            effort="15",
            velocity="0.1",
        )
        ET.SubElement(j, "dynamics", damping="1", friction="0")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def import_robot(repo: Path, output: Path, cfg: dict):
    from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig

    src = gripper_urdf(
        repo / "assets/sim/ur3/ur3_cb3.urdf", output / "robot_source/waffles.urdf", cfg
    )
    config = URDFImporterConfig()
    config.urdf_path = str(src)
    config.usd_path = str(output / "robot_asset")
    config.fix_base = True
    config.merge_fixed_joints = False
    config.allow_self_collision = True
    config.collision_from_visuals = False
    config.joint_drive_type = "force"
    config.joint_target_type = "position"
    config.override_joint_stiffness = cfg["physics"]["joint_stiffness"]
    config.override_joint_damping = cfg["physics"]["joint_damping"]
    config.run_asset_transformer = False
    result = URDFImporter(config).import_urdf()
    if not result:
        raise RuntimeError("UR3 URDF import failed")
    return str(result)


def set_forearm_collision_approximation(stage, robot_path, approximation="convexHull"):
    """Override only forearm collider meshes, preserving every collision pair.

    A single convex hull fills the manufacturer's forearm mesh concavity and
    intersects wrist_2 at the recorded August start pose. Decomposition is a
    separately gated campaign geometry correction; old replay configs retain
    their original convex hull model. USD cooking uses default decomposition
    parameters unless the caller explicitly authors additional PhysX settings.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    if approximation not in ("convexHull", "convexDecomposition"):
        raise ValueError(
            "forearm collision approximation must be convexHull or convexDecomposition"
        )
    changed = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.GetName() != "forearm_link":
            continue
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh) and child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr(
                    approximation
                )
                changed.append(str(child.GetPath()))
    if not changed:
        raise RuntimeError(
            "No forearm collision meshes found for collision approximation override"
        )
    return changed


def build_bin_primitives(box, b: dict, blue, bin_phys):
    """Author bin primitives using the same cavity definition as task scoring.

    ``box`` is the scene's USD cube writer; accepting it here keeps metre-level
    physical/visual bounds independently testable without an Isaac runtime.
    Historical scenes keep the original decorative rims and authoring order.
    """
    geometry = bin_geometry(b)
    if geometry.model == "rectangular_envelope":
        # No decorative extension beyond the measured outer envelope. These
        # solid proxy sides are not a claim of measured plastic wall thickness.
        for part in geometry.collision_boxes():
            box(f"/World/Bin/{part.name}", part.center, part.size, blue, True, bin_phys)
        return
    bx, by, bz = b["center"]
    bsx, bsy, bsz = b["size"]
    wall = b["wall"]
    box(
        "/World/Bin/Bottom",
        [bx, by, bz + wall / 2],
        [bsx, bsy, wall],
        blue,
        True,
        bin_phys,
    )
    for side, sign in (("Left", -1), ("Right", 1)):
        box(
            f"/World/Bin/{side}",
            [bx + sign * (bsx - wall) / 2, by, bz + bsz / 2],
            [wall, bsy, bsz],
            blue,
            True,
            bin_phys,
        )
        box(
            f"/World/Bin/{side}Rim",
            [bx + sign * bsx / 2, by, bz + bsz],
            [0.011, bsy + 0.015, 0.01],
            blue,
        )
    for side, sign in (("Front", -1), ("Back", 1)):
        box(
            f"/World/Bin/{side}",
            [bx, by + sign * (bsy - wall) / 2, bz + bsz / 2],
            [bsx, wall, bsz],
            blue,
            True,
            bin_phys,
        )
        box(
            f"/World/Bin/{side}Rim",
            [bx, by + sign * bsy / 2, bz + bsz],
            [bsx + 0.015, 0.011, 0.01],
            blue,
        )
        for i in range(10):
            box(
                f"/World/Bin/{side}Rib_{i}",
                [
                    bx - bsx / 2 + 0.015 + i * (bsx - 0.03) / 9,
                    by + sign * (bsy + 0.004) / 2,
                    bz + bsz * 0.42,
                ],
                [0.005, 0.009, bsz * 0.8],
                blue,
            )


def build_scene(stage, repo: Path, output: Path, cfg: dict, robot_usd: str):
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Looks")

    def material(name, color, roughness=0.5, metallic=0, texture=None):
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Surface"))
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        if texture:
            tex = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Texture"))
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            uv = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("UV"))
            uv.CreateIdAttr("UsdPrimvarReader_float2")
            uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv.ConnectableAPI(), "result"
            )
            sh.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
        return mat

    steel = material("Steel", (0.22, 0.23, 0.21), 0.48, 0.65)
    dark = material("BlackPolymer", (0.018, 0.023, 0.028), 0.44)
    blue = material("BinBlue", (0.004, 0.15, 0.47), 0.27)
    black = material("HoleInterior", (0.008, 0.010, 0.012), 0.9)
    grid = material("MatGrid", (0.36, 0.39, 0.40), 0.75)
    yellow = material("CableLabels", (0.94, 0.72, 0.009), 0.4)
    gel = material("TactileShell", (0.79, 0.90, 0.92), 0.3)
    wafer = material("WaferWrapper", (0.82, 0.51, 0.27), 0.34)
    w = object_config(cfg)
    kind, object_path = object_identity(cfg)
    texpath = repo / w.get("texture", "assets/sim/waffles/packet_top.png")
    wafer_top = material(
        "WaferPrintedTop",
        (1, 1, 1),
        0.36,
        texture=texpath if texpath.exists() else None,
    )

    def physics_material(name, sf, df=None, rest=0, compliance=None):
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(float(sf))
        api.CreateDynamicFrictionAttr(float(sf if df is None else df))
        api.CreateRestitutionAttr(float(rest))
        pm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        pm.CreateFrictionCombineModeAttr("average")
        if compliance:
            pm.CreateCompliantContactStiffnessAttr(float(compliance[0]))
            pm.CreateCompliantContactDampingAttr(float(compliance[1]))
        return mat

    table_phys = physics_material("TableContact", cfg["table"]["friction"])
    mat_phys = physics_material("MatContact", cfg["mat"]["friction"])
    bin_phys = physics_material("BinContact", cfg["bin"]["friction"])
    object_phys = physics_material(
        "PacketContact", w["static_friction"], w["dynamic_friction"], w["restitution"]
    )
    pad_phys = physics_material(
        "PadContact",
        cfg["gripper"]["pad_friction"],
        compliance=(
            cfg["gripper"]["compliant_stiffness"],
            cfg["gripper"]["compliant_damping"],
        ),
    )

    def bind(prim, mat):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def collider(prim, mat=None):
        UsdPhysics.CollisionAPI.Apply(prim)
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(0.001)
        api.CreateRestOffsetAttr(0)
        if mat:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

    def box(path, pos, size, mat, collision=False, phys=None):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1)
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        bind(c.GetPrim(), mat)
        if collision:
            collider(c.GetPrim(), phys)
        return c.GetPrim()

    def ellipse(path, pos, size, mat, collision=False, phys=None):
        vertices, faces = elliptical_prism_mesh(size)
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(vertices.astype(np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(
            [Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))]
        )
        mesh.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        bind(mesh.GetPrim(), mat)
        if collision:
            collider(mesh.GetPrim(), phys)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
                "convexHull"
            )
        return mesh.GetPrim()

    def cylinder(path, pos, radius, height, mat, axis="Z"):
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(height)
        c.CreateAxisAttr(axis)
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        bind(c.GetPrim(), mat)
        return c.GetPrim()

    tab = cfg["table"]
    box("/World/Bench/Slab", tab["center"], tab["size"], steel, True, table_phys)
    # Recessed interiors and rim geometry preserve the optical breadboard pattern
    # while the underlying table uses a continuous collision surface.
    cx, cy, _ = tab["center"]
    sx, sy, _ = tab["size"]
    points = []
    counts = []
    indices = []
    for x, y in table_hole_centers(tab):
        start = len(points)
        points.extend(
            [(x, y, tab["top_z"] + 0.0001)]
            + [
                (
                    x + tab["hole_radius"] * math.cos(k * math.tau / 16),
                    y + tab["hole_radius"] * math.sin(k * math.tau / 16),
                    tab["top_z"] + 0.00012,
                )
                for k in range(16)
            ]
        )
        for k in range(16):
            counts.append(3)
            indices.extend([start, start + 1 + k, start + 1 + (k + 1) % 16])
    holes = UsdGeom.Mesh.Define(stage, "/World/Bench/Perforations")
    holes.CreatePointsAttr(points)
    holes.CreateFaceVertexCountsAttr(counts)
    holes.CreateFaceVertexIndicesAttr(indices)
    holes.CreateSubdivisionSchemeAttr("none")
    bind(holes.GetPrim(), black)
    for i, xx in enumerate([cx - sx / 2 + 0.05, cx + sx / 2 - 0.05]):
        for j, yy in enumerate([cy - sy / 2 + 0.05, cy + sy / 2 - 0.05]):
            box(f"/World/Bench/Leg_{i}_{j}", [xx, yy, -0.43], [0.05, 0.05, 0.74], dark)
    # Table panel seam and robot mounting plate/fasteners.
    box("/World/Bench/Seam", [cx, 0, tab["top_z"] + 0.0002], [sx, 0.001, 0.0002], black)
    mount = mount_plate_geometry(cfg.get("robot_mount", {}))
    if mount.yaw:
        # Rotate the plate and its fasteners together about the fixed UR origin.
        UsdGeom.Xform.Define(stage, "/World/Mount").AddRotateZOp().Set(
            math.degrees(mount.yaw)
        )
    box(
        "/World/Mount/Plate",
        mount.center,
        mount.size,
        steel,
        True,
        table_phys,
    )
    for i, x in enumerate([-0.057, 0.057]):
        for j, y in enumerate([-0.057, 0.057]):
            cylinder(f"/World/Mount/Bolt_{i}_{j}", [x, y, 0.0015], 0.007, 0.003, steel)
            cylinder(
                f"/World/Mount/Socket_{i}_{j}", [x, y, 0.0031], 0.0028, 0.0001, black
            )
    m = cfg["mat"]
    local_mat = m.get("frame_model") == "local_planar"
    if m.get("frame_model") not in (None, "local_planar"):
        raise ValueError("Unsupported mat.frame_model")
    if not local_mat and ("yaw" in m or "grid_origin_xy" in m):
        raise ValueError("Mat yaw/grid phase require frame_model='local_planar'")
    mat_center = m["center"]
    if local_mat:
        mat_frame = UsdGeom.Xform.Define(stage, "/World/Mat")
        mat_frame.AddTranslateOp().Set(Gf.Vec3d(*map(float, mat_center)))
        mat_frame.AddRotateZOp().Set(math.degrees(float(m.get("yaw", 0))))
        mat_center = [0., 0., 0.]
    box("/World/Mat/Base", mat_center, m["size"], dark, True, mat_phys)
    mx, my, mz = mat_center
    msx, msy, msz = m["size"]
    grid_start = m.get("grid_origin_xy", [-msx / 2, -msy / 2])
    def grid_positions(axis, extent):
        if not local_mat:
            return np.arange(-extent/2, extent/2+.00001, m["grid_pitch"])
        phase = float(grid_start[axis])
        first = math.ceil((-extent/2-phase)/m["grid_pitch"])
        last = math.floor((extent/2-phase)/m["grid_pitch"])
        return phase + np.arange(first,last+1)*m["grid_pitch"]
    for i, x in enumerate(grid_positions(0, msx)):
        box(
            f"/World/Mat/X_{i}",
            [mx + x, my, mz + msz / 2 + 0.00005],
            [0.00065, msy, 0.0001],
            grid,
        )
    for i, y in enumerate(grid_positions(1, msy)):
        box(
            f"/World/Mat/Y_{i}",
            [mx, my + y, mz + msz / 2 + 0.00005],
            [msx, 0.00065, 0.0001],
            grid,
        )
    b = cfg["bin"]
    if b.get("enabled", True) and bin_geometry(b).yaw:
        # Rotate all physical and visual pieces together; containment scoring
        # uses the same configured yaw in BinGeometry.to_interior_frame.
        bin_frame = UsdGeom.Xform.Define(stage, "/World/Bin")
        bin_frame.AddTranslateOp().Set(Gf.Vec3d(*map(float, b["center"])))
        bin_frame.AddRotateZOp().Set(math.degrees(float(b["yaw"])))
        b = {**b, "center": [0., 0., 0.]}
    support_paths = []
    if b.get("enabled", True):
        build_bin_primitives(box, b, blue, bin_phys)
        support_paths = [f"/World/Bin/{name}" for name in ("Bottom", "Left", "Right", "Front", "Back")]
    if cfg.get("egg_fixture"):
        from phantom.sim.egg_fixture import build_egg_fixture
        support_paths = build_egg_fixture(stage, cfg["egg_fixture"], material, box, collider, mat_phys)

    # Packet geometry has separate rendering and collision meshes.
    obj = UsdGeom.Xform.Define(stage, object_path)
    obj.AddTranslateOp().Set(Gf.Vec3d(*w["center"]))
    obj.AddOrientOp().Set(Gf.Quatf(*map(float, orientation_wxyz(w))))
    UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
    UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(w["mass"])
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(obj.GetPrim())
    rb.CreateEnableCCDAttr(True)
    rb.CreateLinearDampingAttr(0.015)
    rb.CreateAngularDampingAttr(0.02)
    if kind != "waffle":
        from phantom.sim.task_object_usd import build_task_object
        build_task_object(stage, repo, object_path, w, material, box, collider, object_phys)
    else:
        build_waffle_visual(stage, w, box, bind, wafer, wafer_top, object_phys)

    robot = stage.DefinePrim("/World/Robot", "Xform")
    robot.GetReferences().AddReference(robot_usd)
    # Resolve the seven CAD visual instances so appearance overrides can reach
    # their nested material inputs (ordinary traversal skips instance proxies).
    for _ in range(3):
        for prim in list(Usd.PrimRange(robot)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
    set_forearm_collision_approximation(
        stage,
        robot.GetPath(),
        cfg["physics"].get("forearm_collision_approximation", "convexHull"),
    )
    joint_paths = {}
    pad_paths = []
    tool_path = None
    housing_path = None
    from phantom.sim.gripper_articulation import is_articulated, configure_gripper_physics

    articulated = is_articulated(cfg)
    for prim in Usd.PrimRange(robot):
        name = prim.GetName()
        if prim.IsA(UsdShade.Material):
            # The DAE exporter stored very dark legacy Phong colors, which the
            # importer gamma-converts again. These are explicit RGB appearance
            # estimates from this rig, not a change to the CAD geometry.
            override = None
            if "blau" in name:
                override = ((0.32, 0.64, 0.78), 0.29, 0.0)
            elif "verbindung" in name or "Rohr" in name:
                override = ((0.36, 0.39, 0.40), 0.30, 0.55)
            elif "grey" in name:
                override = ((0.10, 0.12, 0.13), 0.4, 0.1)
            if override:
                ma = UsdShade.Material(prim)
                ma.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*override[0])
                )
                ma.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(override[1])
                ma.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(override[2])
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if mass is None or mass <= 0:
                # URDF frame-only links are fixed joints, not extra payloads.
                ma = UsdPhysics.MassAPI.Apply(prim)
                ma.CreateMassAttr(0.00001)
                ma.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0000001))
        if not articulated and name in ("left_finger_joint", "right_finger_joint"):
            drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            drive.CreateStiffnessAttr(cfg["physics"]["finger_stiffness"])
            drive.CreateDampingAttr(cfg["physics"]["finger_damping"])
            drive.CreateMaxForceAttr(cfg["physics"]["finger_force_N"])
            drive.CreateTargetPositionAttr(cfg["gripper"]["stroke"] / 2)
        if prim.IsA(UsdPhysics.Joint):
            joint_paths[name] = str(prim.GetPath())
        if name == "tool0" and prim.IsA(UsdGeom.Xform):
            tool_path = str(prim.GetPath())
        if name == "gripper_housing" and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            housing_path = str(prim.GetPath())
        if name in ("left_pad", "right_pad") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            pad_paths.append(str(prim.GetPath()))
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    collider(child, pad_phys)
            if articulated:
                # CAD sensor/adapter bodies already exist in the articulated
                # URDF. Never add the old backing, shell or hidden linkage.
                continue
            # Tactile electronics backing, sensor cap, linkage and visible screws.
            base = str(prim.GetPath())
            sgn = 1 if name == "left_pad" else -1
            pad_proxy = (
                ellipse
                if cfg["gripper"].get("contact_shape", "box") == "ellipse"
                else box
            )
            # Retain backing and linkage contacts. Rounded backing avoids the
            # same artificial YZ-corner floor blockage as the former pad box.
            pad_proxy(
                base + "/Backing",
                [sgn * 0.008, 0, 0],
                [0.006, 0.030, 0.060],
                dark,
                True,
                pad_phys,
            )
            pad_proxy(
                base + "/OuterShell",
                [sgn * 0.011, 0, -0.004],
                [0.002, 0.029, 0.045],
                gel,
            )
            pad_proxy(
                base + "/Linkage",
                [sgn * 0.012, 0, -0.051],
                [0.008, 0.016, 0.060],
                steel,
                True,
                table_phys,
            )
            for k, z in enumerate([-0.018, 0.018]):
                cylinder(
                    base + f"/Screw{k}", [sgn * 0.0125, 0, z], 0.0025, 0.003, steel, "X"
                )
            box(
                base + "/CableLabel",
                [sgn * 0.017, 0, -0.038],
                [0.003, 0.015, 0.028],
                yellow,
            )
    pad_paths.sort(key=lambda p: (0 if p.rsplit("/", 1)[-1] == "left_pad" else 1))
    if len(pad_paths) != 2 or housing_path is None:
        raise RuntimeError("Expected gripper housing and two named pad rigid bodies")
    gripper_body_paths = [
        str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(housing_path))
        if p.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    articulation_report = None
    if articulated:
        for p in Usd.PrimRange(stage.GetPrimAtPath(housing_path)):
            if not p.HasAPI(UsdPhysics.CollisionAPI):
                continue
            gel_body = any(str(p.GetPath()).startswith(pad + "/") for pad in pad_paths)
            collider(p, pad_phys if gel_body else table_phys)
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
        articulation_report = configure_gripper_physics(stage, joint_paths, cfg)
    # Estimated cable coil on bench, native curve geometry.
    curve = UsdGeom.BasisCurves.Define(stage, "/World/CableCoil")
    pts = []
    for theta in np.linspace(0, math.tau * 4, 180):
        pts.append(
            (
                0.01 + 0.060 * math.cos(theta),
                -0.21 + 0.045 * math.sin(theta),
                tab["top_z"] + 0.004 + 0.0005 * theta,
            )
        )
    curve.CreatePointsAttr(pts)
    curve.CreateCurveVertexCountsAttr([len(pts)])
    curve.CreateTypeAttr("linear")
    curve.CreateWidthsAttr([0.004])
    curve.SetWidthsInterpolation("constant")
    bind(curve.GetPrim(), dark)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr(cfg["lighting"]["dome_intensity"])
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.96, 1))
    key = UsdLux.RectLight.Define(stage, "/World/Lighting/Softbox")
    key.CreateIntensityAttr(cfg["lighting"]["key_intensity"])
    key.CreateWidthAttr(1.4)
    key.CreateHeightAttr(0.5)
    key.AddTranslateOp().Set(Gf.Vec3d(*cfg["lighting"]["key_position"]))
    return {
        "robot_path": "/World/Robot",
        "tool_path": tool_path,
        "pad_paths": pad_paths,
        "gripper_housing_path": housing_path,
        "gripper_body_paths": gripper_body_paths,
        "gripper_articulation": articulation_report,
        "joint_paths": joint_paths,
        "object_path": object_path,
        "object_kind": kind,
        "support_paths": support_paths,
        "environment_paths": ["/World/Bench/Slab", "/World/Mat/Base", *support_paths, object_path],
        "waffle_path": object_path,  # compatibility alias for existing trace consumers
    }


def build_waffle_visual(stage, w, box, bind, wafer, wafer_top, object_phys):
    """Preserve historical wafer mesh and wrapper appearance exactly."""
    from pxr import UsdGeom, Sdf
    box("/World/Waffle/Body", [0, 0, 0], w["size"], wafer, True, object_phys)
    wx, wy, wz = w["size"]
    top = UsdGeom.Mesh.Define(stage, "/World/Waffle/PrintedTop")
    top.CreatePointsAttr(
        [
            (-wx / 2, -wy / 2, wz / 2 + 0.0001),
            (wx / 2, -wy / 2, wz / 2 + 0.0001),
            (wx / 2, wy / 2, wz / 2 + 0.0001),
            (-wx / 2, wy / 2, wz / 2 + 0.0001),
        ]
    )
    top.CreateFaceVertexCountsAttr([4])
    top.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    top.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(top).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    bind(top.GetPrim(), wafer_top)
    for side, sign in (("Left", -1), ("Right", 1)):
        for i in range(12):
            box(
                f"/World/Waffle/{side}Crimp_{i}",
                [sign * (wx / 2 + 0.001), -wy / 2 + (i + 0.5) * wy / 12, 0],
                [0.004, wy / 24, wz * 0.8],
                wafer,
            )
