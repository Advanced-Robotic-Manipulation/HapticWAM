"""Image-estimated open pulp egg carton and four separate white egg holders.

These fixtures are static, watertight triangle shells. In particular their
collision approximation is ``none``: a convex hull would fill every cavity.
No pulp deformation, egg cracking, or measured material response is claimed.
An optional, separately parameterized compliant support pad occupies one source
cell only. Its contact indentation approximates softness; its visual mesh does
not deform. An explicit internal rigid stop bounds indentation before a contact
can penetrate through the pad's mid-plane; this is a foam-bottoming proxy.
All dimensions are metres. ``center`` is the tray bed's floor datum in world
coordinates; its local +Y follows the five-cell axis. Holder centers are world
floor positions and do not inherit tray yaw or translation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def support_pad_spec(cfg):
    """Validate the optional, image-estimated pad without changing old fixtures.

    Stiffness/damping are force-based *per-contact* PhysX parameters, not measured
    bulk foam properties. The ellipse is inside the 48-sided cup's floor even
    at its widest axis; no pad geometry spans another cup or the tray opening.
    The returned center is in tray-local coordinates, including the cup floor.
    """
    pad = cfg.get("support_pad")
    if pad is None or pad.get("enabled", True) is False:
        return None
    if pad.get("model", "compliant_contact_v1") != "compliant_contact_v1":
        raise ValueError("Unsupported egg support_pad model")
    cell = np.asarray(pad["cell"], float)
    counts = np.array([cfg.get("columns", 2), cfg.get("rows", 5)], int)
    if (cell.shape != (2,) or not np.isfinite(cell).all()
            or np.any(cell != np.floor(cell)) or np.any(cell < 0)
            or np.any(cell >= counts)):
        raise ValueError("support_pad.cell must index one valid [column,row]")
    pitch = np.asarray(cfg.get("cell_pitch_xy", [.057, .048]), float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("support_pad requires a positive finite cell pitch")
    footprint = np.asarray(pad["footprint_xy_m"], float)
    bottom_radius = _positive(cfg.get("cup_bottom_radius_m", .012), "bottom radius")
    if (footprint.shape != (2,) or not np.isfinite(footprint).all()
            or np.any(footprint <= 0)
            or footprint.max()/2 > bottom_radius*math.cos(math.pi/48)):
        raise ValueError("support_pad footprint must fit inside the polygonal cup floor")
    wall = _positive(cfg.get("wall_thickness_m", .0015), "wall")
    height = _positive(cfg.get("height_m", .032), "cup height")
    thickness = _positive(pad["thickness_m"], "support_pad thickness")
    if thickness > height-wall:
        raise ValueError("support_pad thickness must leave its top at or below the cup rim")
    max_compression = _positive(pad["max_compression_m"], "support_pad max compression")
    if max_compression >= thickness/2:
        raise ValueError("support_pad max compression must be less than half its thickness")
    stiffness = _positive(pad["stiffness_n_m"], "support_pad stiffness")
    damping = _positive(pad["damping_n_s_m"], "support_pad damping")
    friction = _positive(pad.get("friction", .55), "support_pad friction")
    color = np.asarray(pad.get("color", [.72, .71, .64]), float)
    if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
        raise ValueError("support_pad color must contain three values in [0,1]")
    return {
        "cell": cell.astype(int).tolist(),
        "center_local_m": [*((cell-(counts-1)/2)*pitch).tolist(), wall+thickness/2],
        "footprint_xy_m": footprint.tolist(), "thickness_m": thickness,
        "max_compression_m": max_compression,
        "bottoming_top_local_z_m": wall+thickness-max_compression,
        "stiffness_n_m": stiffness, "damping_n_s_m": damping,
        "friction": friction, "color": color.tolist(),
        "uncompressed_top_local_z_m": wall+thickness,
        "rigid_floor_local_z_m": wall,
    }


def build_support_pad(stage, cfg, material, collider):
    """Author a fixed compliant-contact pad, returning its exact fixture paths.

    The pad has no rigid body, articulation, drive or joint, and performs no
    writes to an egg. An internal hard stop approximates foam bottoming before
    deep interpenetration could flip the compliant convex contact normal.
    """
    spec = support_pad_spec(cfg)
    if spec is None:
        return []
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    column, row = spec["cell"]
    path = f"/World/EggFixture/Tray/Cell_{column}_{row}/SupportPad"
    appearance = material("EggSourceSupportPad", spec["color"], roughness=.95)
    contact = UsdShade.Material.Define(stage, "/World/Looks/EggSourceSupportPadContact")
    physics = UsdPhysics.MaterialAPI.Apply(contact.GetPrim())
    physics.CreateStaticFrictionAttr(spec["friction"])
    physics.CreateDynamicFrictionAttr(spec["friction"])
    physics.CreateRestitutionAttr(0.)
    compliant = PhysxSchema.PhysxMaterialAPI.Apply(contact.GetPrim())
    compliant.CreateCompliantContactStiffnessAttr(spec["stiffness_n_m"])
    compliant.CreateCompliantContactDampingAttr(spec["damping_n_s_m"])
    compliant.CreateCompliantContactAccelerationSpringAttr(False)
    pad = UsdGeom.Cylinder.Define(stage, path)
    pad.CreateRadiusAttr(1.)
    pad.CreateHeightAttr(spec["thickness_m"])
    pad.CreateAxisAttr("Z")
    pad.AddTranslateOp().Set(Gf.Vec3d(*spec["center_local_m"]))
    pad.AddScaleOp().Set(Gf.Vec3f(spec["footprint_xy_m"][0]/2,
                                   spec["footprint_xy_m"][1]/2, 1.))
    UsdShade.MaterialBindingAPI.Apply(pad.GetPrim()).Bind(appearance)
    collider(pad.GetPrim(), contact)
    pad.GetPrim().SetCustomDataByKey("support_model", "uncalibrated force-based compliant contact; static visual; internal stop bounds compression")
    hard = UsdShade.Material.Define(stage, "/World/Looks/EggSourceSupportPadStopContact")
    hard_physics = UsdPhysics.MaterialAPI.Apply(hard.GetPrim())
    hard_physics.CreateStaticFrictionAttr(spec["friction"])
    hard_physics.CreateDynamicFrictionAttr(spec["friction"])
    hard_physics.CreateRestitutionAttr(0.)
    stop_path = f"/World/EggFixture/Tray/Cell_{column}_{row}/SupportPadStop"
    stop_height = spec["thickness_m"]-spec["max_compression_m"]
    stop = UsdGeom.Cylinder.Define(stage, stop_path)
    stop.CreateRadiusAttr(1.)
    stop.CreateHeightAttr(stop_height)
    stop.CreateAxisAttr("Z")
    stop.AddTranslateOp().Set(Gf.Vec3d(*spec["center_local_m"][:2],
                                     spec["rigid_floor_local_z_m"]+stop_height/2))
    stop.AddScaleOp().Set(Gf.Vec3f(spec["footprint_xy_m"][0]/2,
                                  spec["footprint_xy_m"][1]/2, 1.))
    stop.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    collider(stop.GetPrim(), hard)
    stop.GetPrim().SetCustomDataByKey("support_model", "unmeasured foam-bottoming approximation; maximum contact indentation")
    return [path, stop_path]


def cup_shell_mesh(bottom_radius, top_radius, height, wall, segments=48):
    """Closed material shell around an open tapered cavity, Z=0 at underside.

The inner floor is at ``wall``. Ring winding gives positive solid volume and
inward-facing cavity normals; no surface spans the opening at ``height``.
"""
    rb = _positive(bottom_radius, "bottom radius")
    rt = _positive(top_radius, "top radius")
    h = _positive(height, "height")
    t = _positive(wall, "wall")
    if t >= h or rb > rt or segments < 8 or segments % 4:
        raise ValueError("Need wall < height, bottom radius <= top radius, and segments divisible by 4")
    # Profile follows the exposed surface from the underside to the cavity.
    profile = [(rb + t, 0.), (rt + t, h), (rt, h), (rb, t)]
    theta = np.arange(segments) * (math.tau / segments)
    vertices = [[r * math.cos(a), r * math.sin(a), z] for r, z in profile for a in theta]
    faces = []
    for ring in range(len(profile) - 1):
        for i in range(segments):
            j = (i + 1) % segments
            a, b = ring * segments + i, ring * segments + j
            c, d = b + segments, a + segments
            faces.extend([[a, b, c], [a, c, d]])
    bottom, floor = len(vertices), len(vertices) + 1
    vertices.extend([[0., 0., 0.], [0., 0., t]])
    last = (len(profile) - 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([[bottom, j, i], [floor, last + i, last + j]])
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32)


def _ring_solid(inner_xy, outer_xy, z, thickness):
    """Thin material band, retaining the inner polygon as a true opening."""
    inner_xy, outer_xy = np.asarray(inner_xy), np.asarray(outer_xy)
    n = len(inner_xy)
    rings = [(inner_xy, z), (outer_xy, z), (outer_xy, z - thickness), (inner_xy, z - thickness)]
    vertices = np.concatenate([np.c_[xy, np.full(n, zz)] for xy, zz in rings])
    faces = []
    for r in range(4):
        nr = (r + 1) % 4
        for i in range(n):
            j = (i + 1) % n
            faces.extend([[r*n+i, r*n+j, nr*n+j], [r*n+i, nr*n+j, nr*n+i]])
    # Ring enumeration runs inward/outward opposite the shell convention.
    return vertices, np.asarray(faces, dtype=np.int32)[:, ::-1]


def _round_rectangle(size, radius, segments=48):
    """Counterclockwise rounded rectangle with uniform angular correspondence."""
    half = np.asarray(size, float) / 2
    radius = min(float(radius), float(half.min()) * .95)
    points = []
    per_corner = segments // 4
    for quadrant in range(4):
        sign = [(1, 1), (-1, 1), (-1, -1), (1, -1)][quadrant]
        center = (half - radius) * sign
        angles = np.arange(per_corner) * math.pi / (2 * per_corner) + quadrant * math.pi / 2
        points.extend(center + radius * np.c_[np.cos(angles), np.sin(angles)])
    return np.asarray(points)


def tray_web_mesh(column, row, columns, rows, pitch, cup_outer_radius,
                  height, wall, corner_radius=.014, opening_half_xy=(.011, .014)):
    """Closed material web with rounded perimeter and open diamond junctions.

    The 2x5 tray's four dark inter-row diamonds are image-estimated apertures.
    Each neighboring cell loses one quarter of that opening. Cup centers,
    bowl walls, and their support heights are independent of this web. The
    exterior and collision mesh are identical; no convex hull fills the holes.
    Returned vertices use the cell's local XY origin and the tray's Z datum.
    """
    pitch, opening = np.asarray(pitch, float), np.asarray(opening_half_xy, float)
    if (pitch.shape != (2,) or opening.shape != (2,)
            or not np.isfinite(pitch).all() or not np.isfinite(opening).all()
            or np.any(pitch <= 0) or np.any(opening <= 0)
            or min(columns, rows) < 1 or not 0 <= column < columns
            or not 0 <= row < rows):
        raise ValueError("Need positive pitch/apertures and a valid tray cell")
    radius = _positive(cup_outer_radius, "cup outer radius")
    h, t = _positive(height, "height"), _positive(wall, "wall")
    rounding = _positive(corner_radius, "corner radius")
    if t >= h or radius >= pitch.min()/2:
        raise ValueError("Need wall < height and cup rim inside its cell")
    cell = (np.array([column, row]) - (np.array([columns, rows])-1)/2) * pitch
    # Outward half-plane normals. Every permitted point satisfies n.p <= d.
    normals = [np.array(x, float) for x in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    offsets = [pitch[0]/2, pitch[0]/2, pitch[1]/2, pitch[1]/2]
    outline = _round_rectangle(pitch * [columns, rows], rounding, 64) - cell
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        edge = b-a
        normal = np.array([edge[1], -edge[0]])
        normals.append(normal)
        offsets.append(float(normal @ a))
    for sx in (-1, 1):
        for sy in (-1, 1):
            junction = np.array([column + (sx+1)//2, row + (sy+1)//2])
            if 0 < junction[0] < columns and 0 < junction[1] < rows:
                normals.append(np.array([sx, sy]) / opening)
                offsets.append(float(np.sum(pitch / (2*opening)) - 1))
    normals, offsets = np.asarray(normals), np.asarray(offsets)
    if np.any(offsets <= radius * np.linalg.norm(normals, axis=1)):
        raise ValueError("Tray rounding or diamond aperture intersects a cup rim")

    polygon = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * pitch/2
    for normal, offset in zip(normals, offsets):
        clipped = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            da, db = normal @ a - offset, normal @ b - offset
            if da <= 1e-12:
                clipped.append(a)
            if (da < -1e-12 and db > 1e-12) or (da > 1e-12 and db < -1e-12):
                clipped.append(a + da/(da-db) * (b-a))
        polygon = np.asarray(clipped)
    # Include every true perimeter/aperture corner as well as circular-rim
    # samples. This avoids a coarse angular grid rounding over a diamond tip.
    angles = np.mod(np.arctan2(polygon[:, 1], polygon[:, 0]), math.tau)
    angles = np.unique(np.round(np.r_[angles, np.arange(48)*math.tau/48], 12))
    directions = np.c_[np.cos(angles), np.sin(angles)]
    projections = directions @ normals.T
    bounds = np.divide(offsets, projections, out=np.full_like(projections, np.inf),
                       where=projections > 1e-12)
    outer = directions * bounds.min(axis=1)[:, None]
    return _ring_solid(radius * directions, outer, h, t)


def build_egg_fixture(stage, cfg, material, box, collider, mat_phys):
    """Author the optional egg fixture and return its collider paths/estimates.

    ``cfg`` is the scene's ``egg_fixture`` configuration. ``material``, ``box``, and
    ``collider`` are the shared scene-builder helpers. The fixture defines a
    separate friction material, using its configured friction estimate.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    f = cfg
    # Validate before authoring any scene objects, including the disabled path.
    support_pad_spec(f)
    center = np.asarray(f["center"], float)
    yaw = float(f.get("yaw", 0.))
    pitch = np.asarray(f.get("cell_pitch_xy", [.057, .048]), float)
    columns, rows = int(f.get("columns", 2)), int(f.get("rows", 5))
    if center.shape != (3,) or pitch.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(pitch).all() or np.any(pitch <= 0) or not np.isfinite(yaw) or min(rows, columns) < 1:
        raise ValueError("Egg tray requires finite center/yaw, positive XY pitch, and positive row/column counts")
    radius = _positive(f.get("cup_radius_m", .022), "cup_radius_m")
    bottom_radius = _positive(f.get("cup_bottom_radius_m", .012), "cup_bottom_radius_m")
    height = _positive(f.get("height_m", .032), "height_m")
    wall = _positive(f.get("wall_thickness_m", .0015), "wall_thickness_m")
    if radius + wall >= pitch.min() / 2:
        raise ValueError("Egg cup outer rim must fit within its cell pitch")
    pulp_color = np.asarray(f.get("pulp_color", [.40, .52, .24]), float)
    if pulp_color.shape != (3,) or not np.isfinite(pulp_color).all() or np.any((pulp_color < 0) | (pulp_color > 1)):
        raise ValueError("pulp_color must be three values in [0,1]")
    pulp = material("EggTrayPulp", pulp_color.tolist(), roughness=.95)
    # Spatially varying vertex colors mimic the visible molded-pulp mottling.
    color_reader = UsdShade.Shader.Define(stage, pulp.GetPath().AppendChild("PulpColor"))
    color_reader.CreateIdAttr("UsdPrimvarReader_float3")
    color_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    color_reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*pulp_color))
    UsdShade.Shader(stage.GetPrimAtPath(pulp.GetPath().AppendChild("Surface"))).GetInput("diffuseColor").ConnectToSource(color_reader.ConnectableAPI(), "result")
    white = material("EggHolderWhite", [.88, .91, .90], roughness=.31)
    fixture_phys = UsdShade.Material.Define(stage, "/World/Looks/EggFixtureContact")
    physics = UsdPhysics.MaterialAPI.Apply(fixture_phys.GetPrim())
    friction = _positive(f.get("friction", .55), "fixture friction")
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction)
    physics.CreateRestitutionAttr(.01)
    UsdGeom.Xform.Define(stage, "/World/EggFixture")
    tray = UsdGeom.Xform.Define(stage, "/World/EggFixture/Tray")
    tray.AddTranslateOp().Set(Gf.Vec3d(*center))
    tray.AddRotateZOp().Set(math.degrees(yaw))
    collider_paths = []

    def mesh(path, vertices, faces, pos=(0., 0., 0.), appearance=pulp, physical=True):
        vertices = np.asarray(vertices, float)
        shape = UsdGeom.Mesh.Define(stage, path)
        shape.CreatePointsAttr(vertices.astype(np.float32).tolist())
        shape.CreateFaceVertexCountsAttr([3] * len(faces))
        shape.CreateFaceVertexIndicesAttr(np.asarray(faces).reshape(-1).tolist())
        shape.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        shape.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(appearance)
        if appearance == pulp:
            worldish = vertices + np.asarray(pos)
            phase = worldish @ np.array([713., 1021., 463.])
            mottling = .94 + .065 * np.sin(phase) + .035 * np.cos(phase * 2.371)
            UsdGeom.PrimvarsAPI(shape).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(np.clip(mottling[:, None] * pulp_color, 0., 1.).astype(np.float32).tolist())
        if physical:
            collider(shape.GetPrim(), fixture_phys)
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("none")
            collider_paths.append(path)
        return shape

    theta = np.arange(48) * math.tau / 48
    inner = (radius + wall) * np.c_[np.cos(theta), np.sin(theta)]
    direction = np.c_[np.cos(theta), np.sin(theta)]
    ray_scale = np.min(np.divide(pitch / 2, np.abs(direction), out=np.full_like(direction, np.inf), where=np.abs(direction) > 1e-12), axis=1)
    outer = direction * ray_scale[:, None]
    shell_v, shell_f = cup_shell_mesh(bottom_radius, radius, height, wall)
    web_v, web_f = _ring_solid(inner, outer, height, wall)
    web_model = f.get("web_model", "rectangular_web_v1")
    if web_model not in ("rectangular_web_v1", "rounded_perforated_v2"):
        raise ValueError(f"Unknown egg tray web_model: {web_model}")
    for row in range(rows):
        for column in range(columns):
            cell = [(column - (columns - 1) / 2) * pitch[0], (row - (rows - 1) / 2) * pitch[1], 0.]
            mesh(f"/World/EggFixture/Tray/Cell_{column}_{row}/Bowl", shell_v, shell_f, cell)
            if web_model == "rounded_perforated_v2":
                web_v, web_f = tray_web_mesh(
                    column, row, columns, rows, pitch, radius+wall, height, wall,
                    f.get("outer_corner_radius_m", .014),
                    f.get("diamond_opening_half_xy_m", [.011, .014]))
            mesh(f"/World/EggFixture/Tray/Cell_{column}_{row}/Web", web_v, web_f, cell)
    collider_paths.extend(build_support_pad(stage, f, material, collider))
    # Molded separators visible between the two rows of cups. Closed tapered
    # cones sit on the existing web; no sheet spans a cup's opening.
    peak_height = _positive(f.get("separator_height_m", .043), "separator_height_m")
    if web_model == "rectangular_web_v1" and peak_height > height:
        pv, pf = cup_shell_mesh(.003, .010, peak_height - height, min(wall, (peak_height-height)/3), 12)
        pv[:, 2] = peak_height - height - pv[:, 2]
        pf = pf[:, ::-1]
        for row in range(rows - 1):
            for col in range(columns - 1):
                xy = [(col - (columns-2)/2) * pitch[0], (row - (rows-2)/2) * pitch[1], height]
                mesh(f"/World/EggFixture/Tray/Separator_{col}_{row}", pv, pf, xy)

    lid_cfg = f.get("lid", {})
    if lid_cfg.get("enabled", True):
        lid_width = _positive(lid_cfg.get("width_m", .11), "lid width")
        lid_length = _positive(lid_cfg.get("length_m", rows * pitch[1]), "lid length")
        lid_height = _positive(lid_cfg.get("height_m", .018), "lid height")
        side = float(lid_cfg.get("side", -1))
        if side not in (-1., 1.):
            raise ValueError("lid.side must be -1 or 1")
        lid_center = [side * (columns*pitch[0] / 2 + lid_width/2), 0., 0.]
        lid_outline = _round_rectangle([lid_width, lid_length], .012)
        inner_outline = _round_rectangle([lid_width-2*wall, lid_length-2*wall], .010)
        lv, lf = _ring_solid(inner_outline, lid_outline, lid_height, lid_height)
        mesh("/World/EggFixture/Tray/Lid/Wall", lv, lf, lid_center)
        # Rounded floor triangulated as a fan; thickness is a second layer.
        n = len(lid_outline)
        lv = np.r_[np.c_[lid_outline, np.zeros(n)], np.c_[lid_outline, np.full(n, wall)], [[0.,0.,0.], [0.,0.,wall]]]
        lf = []
        for i in range(n):
            j = (i+1) % n
            lf.extend([[2*n,j,i], [2*n+1,n+i,n+j], [i,j,n+j], [i,n+j,n+i]])
        mesh("/World/EggFixture/Tray/Lid/Floor", lv, lf, lid_center)
        # Narrow hinge lies outside every bowl, at the adjoining outer edges.
        box("/World/EggFixture/Tray/Hinge", [side*columns*pitch[0]/2,0.,wall/2], [.005,lid_length*.96,wall], pulp)
        if lid_cfg.get("texture"):
            texture = Path(lid_cfg["texture"])
            if not texture.is_absolute():
                texture = Path(__file__).resolve().parents[2] / texture
            if not texture.is_file():
                raise FileNotFoundError(f"Egg tray lid evidence texture missing: {texture}")
            printed = material("EggTrayLidPrint", [1.,1.,1.], roughness=.95, texture=texture)
            x, y = lid_width * .43, lid_length * .45
            print_mesh = mesh("/World/EggFixture/Tray/Lid/Print", [[-x,-y,wall+.0001],[x,-y,wall+.0001],[x,y,wall+.0001],[-x,y,wall+.0001]], [[0,1,2],[0,2,3]], lid_center, printed, False)
            UsdGeom.PrimvarsAPI(print_mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set([(0.,0.),(1.,0.),(1.,1.),(0.,1.)])

    holders = f.get("holders", {})
    holder_centers = holders.get("centers", [])
    if holder_centers:
        outer_radius = _positive(holders.get("outer_radius_m", .0225), "holder outer radius")
        inner_radius = _positive(holders.get("inner_radius_m", .0205), "holder inner radius")
        holder_wall = _positive(holders.get("wall_thickness_m", .0015), "holder wall thickness")
        holder_height = _positive(holders.get("height_m", .020), "holder height")
        if inner_radius + holder_wall > outer_radius:
            raise ValueError("Holder outer radius must include inner radius plus wall thickness")
        hv,hf = cup_shell_mesh(holders.get("bottom_radius_m", .012), inner_radius, holder_height, holder_wall)
        # The shell already owns the inner rim band. Extend outward from its
        # exterior radius so duplicate coplanar top faces cannot add contacts.
        rv,rf = _ring_solid((inner_radius + holder_wall) * direction, outer_radius * direction, holder_height, holder_wall)
        for index, pos in enumerate(holder_centers):
            if np.asarray(pos).shape != (3,) or not np.isfinite(pos).all():
                raise ValueError("Holder center must contain three finite world coordinates")
            UsdGeom.Xform.Define(stage, f"/World/EggFixture/Holders/Cup_{index}")
            mesh(f"/World/EggFixture/Holders/Cup_{index}/Bowl", hv, hf, pos, white)
            mesh(f"/World/EggFixture/Holders/Cup_{index}/Rim", rv, rf, pos, white)
    return collider_paths
