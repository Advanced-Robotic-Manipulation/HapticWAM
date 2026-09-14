"""Opt-in bounded contact compliance for exploratory carton reconstruction.

One free rigid body retains the printed outer envelope. A compliant outer
collision surface and an eroded hard core approximate contact indentation.
This is not deformable paper, liquid slosh, or calibrated crushing mechanics.
The render surface does not deform. Core offset is measured normal to each
outer supporting plane, including sloped folds, rather than scaling XYZ.
The legacy ``max_indentation_m`` field specifies that plane-normal inset; it
does not bound arbitrary edge/corner or oblique contact travel. A compliant
contact partner can add further deflection even in a flat-face normal test.
"""
from __future__ import annotations

import numpy as np


def compliance_spec(obj):
    """Validate the contact hypothesis; legacy max_indentation_m is a core inset."""
    value = obj.get("contact_compliance")
    if value is None:
        return None
    if obj.get("kind") != "carton" or "carton_profile" not in obj:
        raise ValueError("contact_compliance requires a profiled carton")
    if value.get("model") != "bounded_contact_v1":
        raise ValueError("Unsupported carton contact compliance model")
    if value.get("restitution_combine", "average") not in ("average", "max"):
        raise ValueError("Carton restitution_combine must be average or max")
    result = {}
    for key in ("stiffness_n_m", "damping_n_s_m", "max_indentation_m"):
        number = float(value[key])
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"Carton {key} must be finite and positive")
        result[key] = number
    if result["max_indentation_m"] >= float(np.min(obj["size"])) / 4:
        raise ValueError("Carton core inset must be smaller than one quarter of every dimension")
    return result


def inset_core_mesh(points, inset_m):
    """Intersect inward-offset convex supporting halfspaces, with outward faces."""
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    points = np.asarray(points, dtype=float)
    inset = float(inset_m)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Core requires finite XYZ points")
    if not np.isfinite(inset) or inset <= 0:
        raise ValueError("Core inset must be finite and positive")
    hull = ConvexHull(points)
    planes = hull.equations.copy()
    planes[:, 3] += inset
    # Every supported profile is centered and centrally symmetric. Refuse a
    # collapsed/noncentered core instead of changing its plane inset implicitly.
    if np.any(planes[:, 3] >= -1e-9):
        raise ValueError("Inset leaves no strictly interior origin for carton core")
    vertices = HalfspaceIntersection(planes, np.zeros(3)).intersections
    core = ConvexHull(vertices)
    triangles = core.simplices.copy()
    tri = vertices[triangles]
    inward = np.einsum("ij,ij->i", np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), core.equations[:, :3]) < 0
    triangles[inward] = triangles[inward][:, [0, 2, 1]]
    if np.max(vertices @ planes[:, :3].T + planes[:, 3]) > 1e-9:
        raise ValueError("Core violates the declared plane-normal inset")
    return vertices, triangles


def outer_mass_properties(points, triangles, mass):
    """Uniform closed-envelope mass properties; nested core adds no mass."""
    from scipy.spatial.transform import Rotation

    tri = np.asarray(points, float)[np.asarray(triangles, int)]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6
    total = volume.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Mass properties require an outward closed envelope")
    sums = tri.sum(axis=1)
    center = (volume[:, None] * sums).sum(axis=0) / (4 * total)
    second = np.einsum("n,nij->ij", volume, np.einsum("nki,nkj->nij", tri, tri) + sums[:, :, None]*sums[:, None, :]) / (20 * total)
    covariance = second - np.outer(center, center)
    inertia = float(mass) * (np.trace(covariance)*np.eye(3) - covariance)
    # Keep canonical axes when symmetry makes this already diagonal.
    if np.max(abs(inertia - np.diag(np.diag(inertia)))) < 1e-14:
        diagonal, axes = np.diag(inertia), np.eye(3)
    else:
        diagonal, axes = np.linalg.eigh(inertia)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
    if np.min(diagonal) <= 0:
        raise ValueError("Mass inertia must be positive")
    return center, diagonal, Rotation.from_matrix(axes).as_quat()[[3, 0, 1, 2]]


def build_bounded_core(stage, path, obj, collider):
    """Author two materials and the hidden hard stop; return outer material."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    from phantom.sim.carton_geometry import carton_mesh

    spec = compliance_spec(obj)
    if spec is None:
        raise ValueError("Explicit carton compliance is required")

    def contact_material(name, compliant):
        mat = UsdShade.Material.Define(stage, "/World/Looks/" + name)
        material = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        material.CreateStaticFrictionAttr(float(obj["static_friction"]))
        material.CreateDynamicFrictionAttr(float(obj["dynamic_friction"]))
        material.CreateRestitutionAttr(float(obj["restitution"]))
        physics = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        physics.CreateFrictionCombineModeAttr("average")
        if compliant:
            physics.CreateCompliantContactStiffnessAttr(spec["stiffness_n_m"])
            physics.CreateCompliantContactDampingAttr(spec["damping_n_s_m"])
            physics.CreateCompliantContactAccelerationSpringAttr(False)
            # PhysX encodes compliant stiffness through negative restitution.
            # Native coupon tests establish that max selects the softer pair;
            # only this object's material overrides combination precedence.
            physics.CreateRestitutionCombineModeAttr(obj["contact_compliance"].get("restitution_combine", "average"))
        return mat

    hard = contact_material("CartonInternalStopContact", False)
    outer = contact_material("CartonCompliantOuterContact", True)
    points, outer_triangles, _ = carton_mesh(obj["size"], obj["carton_profile"])
    vertices, triangles = inset_core_mesh(points, spec["max_indentation_m"])
    core = UsdGeom.Mesh.Define(stage, path + "/CompressionStop")
    core.CreatePointsAttr(vertices.tolist())
    core.CreateFaceVertexCountsAttr([3] * len(triangles))
    core.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
    core.CreateSubdivisionSchemeAttr("none")
    core.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
    core.MakeInvisible()
    collider(core.GetPrim(), hard)
    UsdPhysics.MeshCollisionAPI.Apply(core.GetPrim()).CreateApproximationAttr("convexHull")
    core.GetPrim().SetCustomDataByKey("calibration_status", "Unmeasured internal compression bound; not a real rigid core")
    stage.GetPrimAtPath(path).SetCustomDataByKey("carton_contact_model", "Bounded contact indentation only; one free rigid body, no visual deformation or calibrated liquid/paper mechanics")
    # Without explicit inertia, PhysX counts both overlapping shape volumes,
    # changing the assumed mass distribution during a material comparison.
    center, diagonal, quat = outer_mass_properties(points, outer_triangles, obj["mass"])
    mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path))
    mass.CreateMassAttr(float(obj["mass"]))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*center))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(float(quat[0]), Gf.Vec3f(*quat[1:])))
    return outer
