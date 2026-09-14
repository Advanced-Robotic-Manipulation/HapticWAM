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
        # Smooth normals affect light only. Native collision cooking is
        # audited separately: the default GPU-compatible hull can be coarse.
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
        approximation = obj.get("collision_approximation", "convexHull")
        if approximation == "compound_convex_v1":
            from pxr import PhysxSchema
            from phantom.sim.egg_collision import egg_compound_collision, PHYSX_MESH_COORDINATE_SCALE

            # Partition an inscribed surface beneath the SAME rigid body.
            # Native-cook audit bounds the actual union error at 0.219 mm
            # for the nominal egg, with no exterior padding or open seams.
            # Larger local coordinates avoid small-mesh cooking loss; the
            # reciprocal transform preserves every physical dimension.
            UsdGeom.Xform.Define(stage, path + "/Collision")
            for piece in egg_compound_collision(obj["size"], obj.get("taper", .17)):
                collision_mesh = UsdGeom.Mesh.Define(stage, path + "/Collision/" + piece.name)
                points = piece.points * PHYSX_MESH_COORDINATE_SCALE
                collision_mesh.CreatePointsAttr(points.tolist())
                collision_mesh.CreateFaceVertexCountsAttr([len(face) for face in piece.faces])
                collision_mesh.CreateFaceVertexIndicesAttr([i for face in piece.faces for i in face])
                collision_mesh.CreateSubdivisionSchemeAttr("none")
                collision_mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
                collision_mesh.AddScaleOp().Set(Gf.Vec3f(1. / PHYSX_MESH_COORDINATE_SCALE))
                collision_mesh.MakeInvisible()
                collider(collision_mesh.GetPrim(), physics)
                UsdPhysics.MeshCollisionAPI.Apply(collision_mesh.GetPrim()).CreateApproximationAttr("convexHull")
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(collision_mesh.GetPrim()).CreateHullVertexLimitAttr(64)
            return
        collider(mesh.GetPrim(), physics)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
        if approximation == "sdf":
            from pxr import PhysxSchema

            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim()).CreateSdfResolutionAttr(
                obj.get("sdf_resolution", 256)
            )
        return
    if kind != "carton":
        raise ValueError(f"Unsupported task object {kind}")
    size = np.asarray(obj["size"], float)
    appearance = material("CartonPaper", obj.get("color", [.85, .77, .63]), .63)
    if "carton_profile" in obj:
        from phantom.sim.carton_geometry import carton_face_uv, carton_mesh

        points, triangles, face_labels = carton_mesh(size, obj["carton_profile"])
        mesh = UsdGeom.Mesh.Define(stage, path + "/Body")
        mesh.CreatePointsAttr(points.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
        mesh.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
        collider(mesh.GetPrim(), physics)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        binding.Bind(appearance)
        uv = np.empty((len(triangles), 3, 2))
        for face in np.unique(face_labels):
            indices = np.flatnonzero(face_labels == face)
            uv[indices] = carton_face_uv(points[triangles[indices]].reshape(-1, 3), size, face).reshape(-1, 3, 2)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        ).Set(uv.reshape(-1, 2).tolist())
        for face, texture in obj.get("face_textures", {}).items():
            if face not in ("top", "bottom", "front", "back", "left", "right"):
                raise ValueError(f"Unknown carton texture face {face}")
            texpath = repo / texture
            if not texpath.is_file():
                raise FileNotFoundError(texpath)
            subset = binding.CreateMaterialBindSubset("Print_" + face, np.flatnonzero(face_labels == face).tolist())
            mat = material("CartonPrint_" + face, [1, 1, 1], .58, texture=texpath)
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mat)
        binding.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.nonOverlapping)
        return
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
