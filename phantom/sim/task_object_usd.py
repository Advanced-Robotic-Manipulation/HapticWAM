"""USD appearance and collision surfaces for the recorded Carton/egg tasks."""
from __future__ import annotations

import numpy as np

from phantom.sim.task_objects import egg_mesh


def build_task_object(stage, repo, path, obj, material, box, collider, physics):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    kind = obj["kind"]
    if kind == "egg":
        vertices, faces = egg_mesh(obj["size"], taper=obj.get("taper", .17))
        mesh = UsdGeom.Mesh.Define(stage, path + "/Shell")
        mesh.CreatePointsAttr(vertices.astype(np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
        # Smooth normals affect light only; collisions use the same ovoid hull.
        triangles = vertices[faces]
        face_normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals = np.zeros_like(vertices)
        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        mesh.CreateNormalsAttr(normals.astype(np.float32).tolist())
        mesh.SetNormalsInterpolation("vertex")
        appearance = material("EggShell", obj.get("color", [.50, .235, .095]), .78)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(appearance)
        collider(mesh.GetPrim(), physics)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        return
    if kind != "carton":
        raise ValueError(f"Unsupported task object {kind}")
    size = np.asarray(obj["size"], float)
    appearance = material("CartonPaper", obj.get("color", [.85, .77, .63]), .63)
    box(path + "/Body", [0, 0, 0], size, appearance, True, physics)
    # Each face is independently rectified from recorded RGB when observed.
    # Hidden faces retain paper colour; no fabricated brand/printing evidence.
    corners = {
        "top": [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
        "bottom": [[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1]],
        "front": [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]],
        "back": [[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1]],
        "left": [[-1,1,-1],[-1,-1,-1],[-1,-1,1],[-1,1,1]],
        "right": [[1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]],
    }
    for face, texture in obj.get("face_textures", {}).items():
        if face not in corners:
            raise ValueError(f"Unknown carton texture face {face}")
        texpath = repo / texture
        if not texpath.is_file():
            raise FileNotFoundError(texpath)
        points = np.asarray(corners[face]) * (size/2+.00008)
        mesh = UsdGeom.Mesh.Define(stage, path + "/Print_" + face)
        mesh.CreatePointsAttr(points.tolist())
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set([(0,0), (1,0), (1,1), (0,1)])
        mat = material("CartonPrint_"+face, [1,1,1], .58, texture=texpath)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    # Millimetric folded-board seams are cosmetic estimates, not fake padding
    # added to the collision hull. Printed creases already occur in the RGB.
    seam = material("CartonFold", obj.get("seam_color", [.72, .62, .49]), .8)
    for i, sign in enumerate((-1, 1)):
        box(path+f"/Fold_{i}", [0, sign*size[1]/2, size[2]/2-.001],
            [size[0], .0005, .001], seam)
