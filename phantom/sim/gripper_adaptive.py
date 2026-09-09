"""Opt-in native 2F85 loop topology with rigid W2L sensor attachments.

The topology/origins come from the pinned Menagerie reference. Installed sensor
mounts, passive response and motor calibration remain provisional. This module
preserves PHANTOM motor settings and component masses; it does not import the
reference's actuator gains, force cap, armature or root mounting offset.

``joint_targets`` returns a closed-loop initialization/direct-replay pose.
``drive_targets`` returns actuator rest targets: passive coordinates must never
be reset to the initialization pose during physics stepping.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import root as solve_root

MODEL = "robotiq_2f85_w2l_adaptive_v2"
MASTER = "finger_joint"
JOINT_NAMES = (
    MASTER, "left_inner_knuckle_joint", "left_inner_finger_joint",
    "right_outer_knuckle_joint", "right_inner_knuckle_joint", "right_inner_finger_joint",
    "left_outer_finger_joint", "right_outer_finger_joint",
)
DRIVERS = (MASTER, "right_outer_knuckle_joint")
SPRINGS = ("left_inner_knuckle_joint", "right_inner_knuckle_joint")
FOLLOWERS = ("left_inner_finger_joint", "right_inner_finger_joint")
COUPLERS = ("left_outer_finger_joint", "right_outer_finger_joint")
LIMITS = {**dict.fromkeys(DRIVERS, (0., .8)),
          **dict.fromkeys(SPRINGS, (-.29670597283, .8)),
          **dict.fromkeys(FOLLOWERS, (-.872664, .872664)),
          **dict.fromkeys(COUPLERS, (-1.57, 0.))}
FOLLOWER_MESH_OFFSET = np.array([0., .018, -.0065])
LOOP_CONNECTIONS = tuple({
    "name": side + "_distal_loop", "body0": side + "_inner_finger",
    "body1": side + "_outer_finger",
    "local_pos0": [0., .018, -.0065],
    "local_pos1": [0., .0060989, .047596],
} for side in ("left", "right"))
REFERENCE_RELATIVE = "assets/sim/robotiq/adaptive_reference/2f85.xml"
REFERENCE_SHA256 = "d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de"


def loop_specs():
    return copy.deepcopy(list(LOOP_CONNECTIONS))


def _settings(cfg):
    if cfg["gripper"].get("model") != MODEL:
        raise ValueError(f"Expected gripper.model={MODEL}")
    value = cfg["gripper"].get("articulation")
    if not isinstance(value, dict):
        raise ValueError("Explicit adaptive articulation settings are required")  # noqa: TRY004 - public config validation
    return value


def _finite(value, label, minimum=None):
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return result


def _positive(value, label):
    result = _finite(value, label, 0.)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _motor_angle(closure, cfg):
    settings = _settings(cfg)
    closure = _finite(closure, "closure")
    touch = _positive(cfg["gripper"]["pad_touch_command"], "pad_touch_command")
    angle = _positive(settings["angle_at_touch_rad"], "angle_at_touch_rad")
    if angle > .8:
        raise ValueError("angle_at_touch_rad exceeds the native driver limit")
    return float(angle * np.clip(closure / touch, 0, 1))


def passive_settings(cfg):
    value = _settings(cfg).get("native_passive", {})
    return {
        "spring_stiffness_nm_rad": _finite(value.get("spring_stiffness_nm_rad", .05), "spring stiffness", 0.),
        "spring_damping_nm_s_rad": _finite(value.get("spring_damping_nm_s_rad", .00125), "spring damping", 0.),
        "spring_reference_rad": _finite(value.get("spring_reference_rad", 2.62), "spring reference"),
    }


def validate_physics_timestep(cfg):
    """Reject the timestep that caused native loop divergence in contact tests.

    This is a numerical prerequisite, not a calibration or grasp-success gate.
    The preserved 4 ms campaign remains reproducible from its frozen source.
    """
    step = _positive(cfg["physics"]["dt"], "physics timestep")
    if step > .001 + 1e-12:
        raise ValueError(
            "Native W2L requires physics.dt <= 0.001 s: the 0.004 s contact "
            "tests diverged. Use a dt1ms configuration; mechanical and tactile "
            "qualification still require measured-motion validation."
        )


def _rot2(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


@lru_cache(maxsize=2048)
def _initial_passive_angles(driver):
    """Exact planar source loop on the unloaded coupler-stop branch.

    This is a reproducible geometric seed, not a force-equilibrium prediction.
    Near the driver upper limit the spring reaches its limit first; allow the
    coupler to move instead of violating limits or leaving an open loop.
    """
    dbase, sbase = np.array([.0306011, .054904]), np.array([.0132, .0609])
    carrier, spring_pin = np.array([.0315, -.0041]), np.array([.037, .044])
    follower_pin = FOLLOWER_MESH_OFFSET[1:]
    coupler_pin = np.array(LOOP_CONNECTIONS[0]["local_pos1"])[1:]

    def residual(spring, follower, coupler):
        a = sbase + _rot2(spring) @ spring_pin + _rot2(spring + follower) @ follower_pin
        b = dbase + _rot2(driver) @ carrier + _rot2(driver + coupler) @ coupler_pin
        return a - b

    fit = solve_root(lambda q: residual(q[0], q[1], 0.), [driver, -driver], tol=1e-11)
    spring, follower, coupler = float(fit.x[0]), float(fit.x[1]), 0.
    if spring > LIMITS[SPRINGS[0]][1]:
        spring = LIMITS[SPRINGS[0]][1]
        fit = solve_root(lambda q: residual(spring, q[1], q[0]), [-.001, -driver], tol=1e-11)
        coupler, follower = map(float, fit.x)
    values = [spring, follower, coupler]
    for value, name in zip(values, (SPRINGS[0], FOLLOWERS[0], COUPLERS[0])):
        lo, hi = LIMITS[name]
        if not math.isfinite(value) or not lo - 1e-10 <= value <= hi + 1e-10:
            raise ValueError("No admissible native closed-loop initialization pose")
    if np.linalg.norm(residual(spring, follower, coupler)) > 1e-10:
        raise ValueError("Native initialization loop solve failed")
    return tuple(float(np.clip(v, *LIMITS[n])) for v, n in zip(values, (SPRINGS[0], FOLLOWERS[0], COUPLERS[0])))


def joint_targets(closure, cfg):
    driver = _motor_angle(closure, cfg)
    spring, follower, coupler = _initial_passive_angles(driver)
    return np.array([driver, spring, follower, driver, spring, follower, coupler, coupler])


def drive_targets(closure, cfg):
    driver = _motor_angle(closure, cfg)
    rest = passive_settings(cfg)["spring_reference_rad"]
    return np.array([driver, rest, 0., driver, rest, 0., 0., 0.])


def _positions(values, cfg):
    _settings(cfg)
    q = np.asarray(values, dtype=float)
    if q.shape != (8,) or not np.isfinite(q).all():
        raise ValueError("Expected eight finite native gripper joint angles in radians")
    return q


def closure_from_joint_positions(values, cfg):
    q = _positions(values, cfg)
    _motor_angle(0., cfg)
    return float(np.clip(q[0] / cfg["gripper"]["articulation"]["angle_at_touch_rad"] * cfg["gripper"]["pad_touch_command"], 0, 1))


def coupling_residuals(values, cfg):
    q = _positions(values, cfg)
    return {"right_outer_knuckle_joint": float(q[3] - q[0])}


def mechanical_diagnostics(values):
    """Check one measured eight-joint state before another physics step.

    Frozen qualification tolerances apply to actual positions, never spring
    drive references. Both mirrored chains share the same local YZ equations;
    their loop-error norms are invariant to the installation/world transform.
    Nonfinite or malformed input raises; finite invalid states return failures.
    This checks joint-space mechanics, not independent rigid-body pose readback.
    """
    q = np.asarray(values, dtype=float)
    if q.shape != (8,) or not np.isfinite(q).all():
        raise ValueError("Expected eight finite native gripper joint angles in radians")
    thresholds = {
        "coupling_max_abs_rad": .005,
        "joint_limit_violation_max_rad": .002,
        "loop_closure_max_m": .001,
    }
    violations = {
        name: float(max(lo - value, value - hi, 0.))
        for name, value in zip(JOINT_NAMES, q) for lo, hi in [LIMITS[name]]
    }
    loops = {}
    for side, indices in (("left", (0, 1, 2, 6)), ("right", (3, 4, 5, 7))):
        driver, spring, follower, coupler = q[list(indices)]
        follower_anchor = (np.array([.0132, .0609])
                           + _rot2(spring) @ np.array([.037, .044])
                           + _rot2(spring) @ _rot2(follower) @ FOLLOWER_MESH_OFFSET[1:])
        coupler_anchor = (np.array([.0306011, .054904])
                          + _rot2(driver) @ np.array([.0315, -.0041])
                          + _rot2(driver) @ _rot2(coupler)
                          @ np.asarray(LOOP_CONNECTIONS[0]["local_pos1"])[1:])
        loops[side + "_distal_loop"] = float(np.linalg.norm(follower_anchor - coupler_anchor))
    coupling, limit, loop = float(abs(q[3] - q[0])), max(violations.values()), max(loops.values())
    gates = {
        "joint_coupling": coupling <= thresholds["coupling_max_abs_rad"],
        "joint_limits": limit <= thresholds["joint_limit_violation_max_rad"],
        "adaptive_loop_closure": loop <= thresholds["loop_closure_max_m"],
    }
    return {
        "passed": all(gates.values()), "gates": gates, "thresholds": thresholds,
        "joint_state_finite": True, "joint_positions_rad": q.tolist(),
        "coupling_max_abs_rad": coupling,
        "joint_limit_violation_max_rad": limit,
        "joint_limit_violation_per_joint_rad": violations,
        "loop_closure_per_side_m": loops, "loop_closure_max_m": loop,
    }


def _verify_reference(repo):
    path = Path(repo) / REFERENCE_RELATIVE
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Adaptive reference XML differs from the pinned topology")
    return path


def append_gripper_urdf(root, repo, cfg):
    """Transfer MJCF topology to URDF tree, with loops authored later in USD.

    URDF follower link frames sit at the hinge. Its geometry/inertia and sensor
    mounts are translated by -MJCF hinge-position, preserving the source mesh
    frame. This avoids a massless extra carrier and retains all contact names.
    """
    from phantom.sim.gripper_articulation import MODEL as OLD_MODEL
    from phantom.sim.gripper_articulation import (
        _nominal,
        _part_link,
        _pose_element,
        load_sensor_geometry,
    )
    settings = _settings(cfg)
    _verify_reference(repo)
    mounts = settings.get("mounts", {})
    if any(s not in mounts for s in ("left", "right")):
        raise ValueError("Explicit left and right sensor mounts are required")
    if root.find("link[@name='tool0']") is None:
        raise ValueError("Robot URDF needs a tool0 link")
    nominal = _nominal(repo)
    validation_cfg = copy.deepcopy(cfg)
    validation_cfg["gripper"]["model"] = OLD_MODEL
    geometry = load_sensor_geometry(repo, validation_cfg)
    links = [copy.deepcopy(l) for l in nominal.findall("link") if not l.get("name").endswith("_pad")]
    base = next(l for l in links if l.get("name") == "robotiq_arg2f_base_link")
    base.set("name", "gripper_housing")
    moving_mass = sum(float(l.find("inertial/mass").get("value")) for l in links if l is not base)
    base_mass = _positive(_positive(settings["bare_gripper_mass_kg"], "bare_gripper_mass_kg") - moving_mass, "base mass")
    old_mass = float(base.find("inertial/mass").get("value"))
    base.find("inertial/mass").set("value", str(base_mass))
    inertia = base.find("inertial/inertia")
    for key, value in inertia.attrib.items():
        inertia.set(key, str(float(value) * base_mass / old_mass))
    existing = {n.get("name") for n in root if n.tag in ("joint", "link")}

    def append(element):
        name = element.get("name")
        if name in existing:
            raise ValueError(f"Duplicate gripper name {name}")
        existing.add(name)
        root.append(element)

    for link in links:
        if link.get("name") in ("left_inner_finger", "right_inner_finger"):
            for item in link:
                if item.tag in ("visual", "collision", "inertial"):
                    origin = item.find("origin")
                    if origin is None:
                        _pose_element(item, FOLLOWER_MESH_OFFSET)
                    else:
                        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") + FOLLOWER_MESH_OFFSET
                        origin.set("xyz", " ".join(map(str, xyz)))
        append(link)

    def joint(name, parent, child, xyz, rpy=(0,0,0), kind="revolute"):
        j = ET.Element("joint", name=name, type=kind)
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)
        _pose_element(j, xyz, rpy)
        if kind == "revolute":
            ET.SubElement(j, "axis", xyz="1 0 0")
            lo, hi = LIMITS[name]
            ET.SubElement(j, "limit", lower=str(lo), upper=str(hi), effort="1000", velocity="2.0")
            if name == DRIVERS[1]:
                ET.SubElement(j, "mimic", joint=MASTER, multiplier="1", offset="0")
        append(j)

    joint("tool_gripper", "tool0", "gripper_housing", [0,0,0], [0,0,-math.pi/2 + cfg["gripper"].get("yaw",0)], "fixed")
    for side, sign in (("left", -1), ("right", 1)):
        yaw = math.pi if sign < 0 else 0.
        joint(MASTER if side == "left" else DRIVERS[1], "gripper_housing", side+"_outer_knuckle", [0,sign*.0306011,.054904], [0,0,yaw])
        joint(side+"_outer_finger_joint", side+"_outer_knuckle", side+"_outer_finger", [0,.0315,-.0041])
        joint(side+"_inner_knuckle_joint", "gripper_housing", side+"_inner_knuckle", [0,sign*.0132,.0609], [0,0,yaw])
        joint(side+"_inner_finger_joint", side+"_inner_knuckle", side+"_inner_finger", [0,.037,.044])
    sensor_names = {}
    for side, parent in (("left", "right_inner_finger"), ("right", "left_inner_finger")):
        pose = mounts[side]
        sensor_names[side] = []
        for part in geometry["parts"]:
            link = _part_link(repo, side, part)
            name = link.get("name")
            append(link)
            sensor_names[side].append(name)
            joint(name+"_mount", parent, name, np.asarray(pose["xyz"]) + FOLLOWER_MESH_OFFSET, pose["rpy"], "fixed")
    return {
        "model": MODEL, "joint_names": list(JOINT_NAMES), "joint_units": "radians",
        "master_joint": MASTER, "pad_links": ["left_pad", "right_pad"], "sensor_links": sensor_names,
        "bare_gripper_mass_kg": moving_mass + base_mass, "mounts": copy.deepcopy(mounts),
        "loop_constraints": loop_specs(), "joint_limits_rad": dict(LIMITS),
        "follower_mesh_offset_m": FOLLOWER_MESH_OFFSET.tolist(), "native_reference_sha256": REFERENCE_SHA256,
        "mechanical_model": "Native adaptive loop topology; geometry seed distinct from passive force dynamics; installed mounts and response uncalibrated",
        "inertia_model": "Existing nominal moving-link masses/inertias and CAD sensor masses retained; base mass allocation unchanged; native MJCF armature not imported",
        "root_transform": "Existing PHANTOM tool installation yaw only; native reference base_mount Z offset omitted",
    }


def configure_gripper_physics(stage, joint_paths, cfg):
    """Eight tree revolutes, two excluded spherical loops, one driven master.

    USD angular drive positions use degrees and stiffness/damping use per-degree
    coefficients. The public command vectors and reference values use radians.
    """
    from pxr import Gf, Sdf, UsdPhysics
    settings, passive = _settings(cfg), passive_settings(cfg)
    stiffness = _positive(settings["drive_stiffness_nm_rad"], "motor stiffness")
    damping = _finite(settings["drive_damping_nm_s_rad"], "motor damping", 0.)
    torque = _positive(settings["drive_max_torque_nm"], "motor torque")
    paths = {n: joint_paths[n] for n in JOINT_NAMES}
    prims = {n: stage.GetPrimAtPath(str(p)) for n,p in paths.items()}
    if len(set(map(str,paths.values()))) != 8 or not all(p.IsA(UsdPhysics.RevoluteJoint) for p in prims.values()):
        raise ValueError("Expected eight distinct native revolute joints")
    axes = {n: UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get() for n,p in prims.items()}
    if any(a not in ("X","Y","Z") for a in axes.values()):
        raise ValueError("Unsupported native revolute joint axis")
    for name, prim in prims.items():
        prim.RemoveAppliedSchema("NewtonMimicAPI")
        for property_name in list(prim.GetPropertyNames()):
            if property_name.startswith(("newton:mimic", "physxMimicJoint:")):
                prim.RemoveProperty(property_name)
        authored = prim.GetMetadata("apiSchemas")
        for schema in list(authored.GetAppliedItems() if authored else []):
            if schema.startswith("PhysxMimicJointAPI:"):
                prim.RemoveAppliedSchema(schema)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateTargetVelocityAttr(0.)
        is_spring = name in SPRINGS
        k = stiffness if name == MASTER else passive["spring_stiffness_nm_rad"] if is_spring else 0.
        d = damping if name == MASTER else passive["spring_damping_nm_s_rad"] if is_spring else 0.
        drive.CreateStiffnessAttr(k * math.pi / 180)
        drive.CreateDampingAttr(d * math.pi / 180)
        drive.CreateTargetPositionAttr(math.degrees(passive["spring_reference_rad"]) if is_spring else 0.)
        drive.CreateMaxForceAttr(torque if name == MASTER else float("inf") if is_spring else 0.)
        if name == DRIVERS[1]:
            token = "rot" + axes[name]
            prim.AddAppliedSchema("PhysxMimicJointAPI:" + token)
            prefix = "physxMimicJoint:" + token + ":"
            for key, value in {"gearing":-1., "offset":0., "naturalFrequency":0., "dampingRatio":0.}.items():
                prim.CreateAttribute(prefix+key, Sdf.ValueTypeNames.Float).Set(value)
            prim.CreateAttribute(prefix+"referenceJointAxis", Sdf.ValueTypeNames.Token).Set("rot"+axes[MASTER])
            prim.CreateRelationship(prefix+"referenceJoint").SetTargets([str(paths[MASTER])])
    bodies = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.setdefault(prim.GetName(), []).append(prim)

    def body(name):
        if len(bodies.get(name, [])) != 1:
            raise ValueError(f"Missing or ambiguous native gripper body {name}")
        return bodies[name][0]

    loops = []
    parent = Sdf.Path(str(paths[MASTER])).GetParentPath()
    for spec in LOOP_CONNECTIONS:
        a, b = body(spec["body0"]), body(spec["body1"])
        path = parent.AppendChild(spec["name"])
        existing = stage.GetPrimAtPath(path)
        if existing and not existing.IsA(UsdPhysics.SphericalJoint):
            raise ValueError(f"Existing non-spherical native loop at {path}")
        loop = UsdPhysics.SphericalJoint.Define(stage, path)
        loop.CreateBody0Rel().SetTargets([a.GetPath()])
        loop.CreateBody1Rel().SetTargets([b.GetPath()])
        loop.CreateLocalPos0Attr(Gf.Vec3f(*spec["local_pos0"]))
        loop.CreateLocalPos1Attr(Gf.Vec3f(*spec["local_pos1"]))
        loop.CreateLocalRot0Attr(Gf.Quatf(1.))
        loop.CreateLocalRot1Attr(Gf.Quatf(1.))
        loop.CreateExcludeFromArticulationAttr(True)
        loop.CreateCollisionEnabledAttr(False)
        loops.append({**copy.deepcopy(spec), "path": str(path), "type": "spherical", "exclude_from_articulation": True})
    pairs = []
    for side in ("left", "right"):
        pairs += [("gripper_housing", side+"_outer_knuckle"), ("gripper_housing", side+"_inner_knuckle"),
                  (side+"_inner_finger", side+"_outer_finger"), (side+"_inner_finger", side+"_inner_knuckle")]
        names = [n for n in bodies if n == side+"_pad" or n.startswith(side+"_sensor_")]
        pairs.extend(itertools.combinations(sorted(names), 2))
    filtered = []
    for a, b in pairs:
        pa, pb = body(a), body(b)
        UsdPhysics.FilteredPairsAPI.Apply(pa).CreateFilteredPairsRel().AddTarget(pb.GetPath())
        filtered.append([str(pa.GetPath()), str(pb.GetPath())])
    return {
        "model": MODEL, "master_joint": MASTER, "mimic_joints": [DRIVERS[1]],
        "coupling": "Only driver symmetry; all native couplers/followers/spring links are passive",
        "loop_constraints": loops, "passive_spring": {**passive, "torque_cap": "unlimited native spring response"},
        "master_stiffness_nm_rad": stiffness, "master_damping_nm_s_rad": damping, "master_max_torque_nm": torque,
        "usd_stiffness_nm_degree": stiffness*math.pi/180, "usd_damping_nm_s_degree": damping*math.pi/180,
        "structural_collision_filters": filtered, "opposing_sensor_contacts_disabled": False,
        "solver_transfer": "Native point equality represented by excluded PhysX spherical joints; reference soft-constraint solver parameters and armature not copied",
    }
