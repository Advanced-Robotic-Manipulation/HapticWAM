"""Opt-in, coupled nominal 2F85 linkage with rigidly attached W2L components.

The constrained one-DOF mimic tree is the nominal parallel-pinch mechanism;
adaptive underactuation, motor/contact calibration and installed mount/85-vs-140
identity remain assumptions. Visual and collision meshes belong to the same
physical links. No separately animated skin or independent pad sliders are used.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from phantom.sim.gripper_visual import _origin, _rotation

MODEL = "robotiq_2f85_w2l_articulated_v1"
ADAPTIVE_MODEL = "robotiq_2f85_w2l_adaptive_v2"
MASTER = "finger_joint"
JOINT_MULTIPLIERS = {
    MASTER: 1.0,
    "left_inner_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "right_inner_finger_joint": -1.0,
}
# The nominal gripper closes along Y; its -pi/2 installation yaw exchanges
# nominal CAD left/right with the rig's positive/negative X pad naming.
PAD_PARENTS = {"left": "right_inner_finger", "right": "left_inner_finger"}


def is_articulated(cfg):
    model = cfg["gripper"].get("model")
    if model in (MODEL, ADAPTIVE_MODEL):
        return True
    if model is None or model == "legacy_sliding_pad_proxy":
        return False
    raise ValueError(f"Unknown gripper.model={model!r}; refusing silent legacy fallback")


def is_adaptive(cfg):
    is_articulated(cfg)  # Keep unknown-model rejection consistent.
    return cfg["gripper"].get("model") == ADAPTIVE_MODEL


def _settings(cfg):
    if not is_articulated(cfg):
        raise ValueError(f"Expected gripper.model={MODEL}")
    value = cfg["gripper"].get("articulation")
    if not isinstance(value, dict):
        raise ValueError("An explicit gripper.articulation configuration is required")
    return value


def _positive(value, label, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{label} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def finger_joint_names(cfg):
    if is_adaptive(cfg):
        from phantom.sim.gripper_adaptive import JOINT_NAMES
        return tuple(JOINT_NAMES)
    return tuple(JOINT_MULTIPLIERS) if is_articulated(cfg) else ("left_finger_joint", "right_finger_joint")


def finger_drive_type(cfg):
    return "angular" if is_articulated(cfg) else "linear"


def joint_targets(closure, cfg):
    """Full joint state in radians (candidate) or metres (legacy), never degrees."""
    if is_adaptive(cfg):
        from phantom.sim.gripper_adaptive import joint_targets as adaptive_seed
        return adaptive_seed(closure, cfg)
    closure = float(closure)
    if not math.isfinite(closure):
        raise ValueError("closure must be finite")
    g = cfg["gripper"]
    fraction = np.clip(closure / _positive(g["pad_touch_command"], "pad_touch_command"), 0, 1)
    if not is_articulated(cfg):
        return np.full(2, _positive(g["stroke"], "stroke") * .5 * (1-fraction))
    angle = _positive(_settings(cfg)["angle_at_touch_rad"], "angle_at_touch_rad")
    if angle > .8:
        raise ValueError("angle_at_touch_rad exceeds the pinned nominal 2F85 range")
    return angle * fraction * np.asarray(list(JOINT_MULTIPLIERS.values()))


def drive_targets(closure, cfg):
    """Drive references, distinct from passive-joint initialization positions."""
    if is_adaptive(cfg):
        from phantom.sim.gripper_adaptive import drive_targets as adaptive_drive
        return adaptive_drive(closure, cfg)
    return joint_targets(closure, cfg)


def closure_from_joint_positions(values, cfg):
    if is_adaptive(cfg):
        from phantom.sim.gripper_adaptive import closure_from_joint_positions as adaptive_closure
        return adaptive_closure(values, cfg)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(finger_joint_names(cfg)),) or not np.isfinite(values).all():
        raise ValueError("Expected one finite position per named gripper joint")
    touch = _positive(cfg["gripper"]["pad_touch_command"], "pad_touch_command")
    if is_articulated(cfg):
        # Motor/master feedback only. Follower residuals must be logged, not
        # averaged away into a fictitious symmetric mechanism state.
        angle = _positive(_settings(cfg)["angle_at_touch_rad"], "angle_at_touch_rad")
        return float(np.clip(values[0] / angle * touch, 0, 1))
    return float(np.clip((1-values.mean() / (cfg["gripper"]["stroke"]*.5))*touch, 0, 1))


def coupling_residuals(values, cfg):
    if is_adaptive(cfg):
        from phantom.sim.gripper_adaptive import coupling_residuals as adaptive_residuals
        return adaptive_residuals(values, cfg)
    values = np.asarray(values, dtype=float)
    closure_from_joint_positions(values, cfg)  # validates shape and finiteness
    if not is_articulated(cfg):
        return {}
    return {name: float(values[i]-multiplier*values[0]) for i,(name,multiplier) in enumerate(JOINT_MULTIPLIERS.items()) if name != MASTER}


def _rpy(matrix):
    pitch = math.atan2(-matrix[2,0], math.hypot(matrix[0,0], matrix[1,0]))
    if abs(math.cos(pitch)) < 1e-9:
        return [0., pitch, math.atan2(-matrix[0,1], matrix[1,1])]
    return [math.atan2(matrix[2,1], matrix[2,2]), pitch, math.atan2(matrix[1,0], matrix[0,0])]


def root_mount_pose(cfg):
    """Tool0-to-gripper pose, including an optional image-registered correction.

    The correction is XYZ metres followed by a rotation vector in radians,
    expressed AFTER the historical installation yaw in the nominal CAD root
    frame. It changes the common physical parent of all gripper components;
    arm kinematics/TCP and finger linkage coordinates are unaffected. Identity
    remains the default for historical scenes.
    """
    from scipy.spatial.transform import Rotation

    correction = np.asarray(_settings(cfg).get("root_correction_xyz_rotvec", [0.] * 6), float)
    if correction.shape != (6,) or not np.isfinite(correction).all():
        raise ValueError("root_correction_xyz_rotvec must contain finite XYZ metres and rotvec radians")
    yaw = -math.pi / 2 + float(cfg["gripper"].get("yaw", 0.))
    if not math.isfinite(yaw):
        raise ValueError("gripper yaw must be finite")
    installed = Rotation.from_euler("z", yaw).as_matrix()
    rotation = installed @ Rotation.from_rotvec(correction[3:]).as_matrix()
    return (installed @ correction[:3]).tolist(), _rpy(rotation)


def _pose_element(parent, xyz, rpy=(0,0,0)):
    numbers = np.r_[xyz, rpy].astype(float)
    if numbers.shape != (6,) or not np.isfinite(numbers).all():
        raise ValueError("origin requires finite xyz[3] and rpy[3]")
    return ET.SubElement(parent, "origin", xyz=" ".join(map(str,numbers[:3])), rpy=" ".join(map(str,numbers[3:])))


def _nominal(repo):
    path = Path(repo)/"assets/sim/robotiq/robotiq_2f85.urdf"
    # Moving-link inertials are present as XML comments in the pinned asset.
    source = ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
    for link in source.findall("link"):
        if link.find("inertial") is None and not link.get("name").endswith("_pad"):
            candidates = []
            for child in link:
                if child.tag is ET.Comment and "<inertial>" in (child.text or ""):
                    candidates.append(ET.fromstring(child.text.strip()))
            if len(candidates) != 1:
                raise ValueError(f"Missing nominal inertia for {link.get('name')}")
            link.append(candidates[0])
        for comment in list(link):
            if comment.tag is ET.Comment:
                link.remove(comment)
    for mesh in source.findall(".//mesh"):
        mesh.set("filename", str((path.parent/mesh.get("filename")).resolve()))
    return source


def load_sensor_geometry(repo, cfg):
    """Load immutable CAD data with optional scene-specific visual colors.

    Overrides affect only the URDF visual material consumed by ``_part_link``.
    The source manifest, collision meshes, masses and attachment frames retain
    their CAD values; omitted overrides preserve historical scene appearance.
    """
    settings = _settings(cfg)
    path = Path(repo)/settings.get("geometry_manifest", "assets/sim/dmtac_w2l/geometry.json")
    data = json.loads(path.read_text())
    parts = data["parts"]
    if not parts or sum(p["role"] == "gel" for p in parts) != 1:
        raise ValueError("Sensor geometry must identify exactly one gel component")
    ids = [p["id"] for p in parts]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", x) for x in ids):
        raise ValueError("Sensor part ids must be unique URDF-safe names")
    overrides = cfg["gripper"].get("visual_materials", {})
    if not isinstance(overrides, dict):
        raise ValueError("gripper.visual_materials must map sensor part IDs to RGBA")
    unknown = set(overrides) - set(ids)
    if unknown:
        raise ValueError(f"Unknown gripper.visual_materials part IDs: {sorted(map(str, unknown))}")
    for part in parts:
        if part["id"] not in overrides:
            continue
        try:
            color = np.asarray(overrides[part["id"]], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("gripper.visual_materials values must be finite normalized RGBA[4]") from error
        if color.shape != (4,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
            raise ValueError("gripper.visual_materials values must be finite normalized RGBA[4]")
        part["rgba"] = color.tolist()
    return data


def sensor_link_name(side, part):
    return f"{side}_pad" if part["role"] == "gel" else f"{side}_sensor_{part['id']}"


def _part_link(repo, side, part):
    link = ET.Element("link", name=sensor_link_name(side, part))
    mirror = _rotation([0,0,1], math.pi if side == "right" else 0)
    origin = part.get("origin", {"xyz":[0,0,0], "rpy":[0,0,0]})
    element = ET.Element("origin", xyz=" ".join(map(str, origin["xyz"])), rpy=" ".join(map(str,origin["rpy"])))
    transform = mirror @ _origin(element)
    for kind, paths in (("visual", [part["visual_mesh"]]), ("collision", part["collision_meshes"])):
        for relative in paths:
            mesh = (Path(repo)/relative).resolve()
            if not mesh.is_file():
                raise ValueError(f"Missing sensor {kind} mesh {mesh}")
            geometry = ET.SubElement(link, kind)
            _pose_element(geometry, transform[:3,3], _rpy(transform[:3,:3]))
            ET.SubElement(ET.SubElement(geometry,"geometry"), "mesh", filename=str(mesh), scale="1 1 1")
            if kind == "visual":
                color = np.asarray(part.get("rgba",[*part.get("color_rgb",[.82,.88,.90]),1.]),float)
                if color.shape != (4,) or not np.isfinite(color).all() or np.any((color<0)|(color>1)):
                    raise ValueError("Sensor color must be normalized RGBA")
                material = ET.SubElement(geometry,"material",name=side+"_w2l_"+part["id"])
                ET.SubElement(material,"color",rgba=" ".join(map(str,color)))
    mass = _positive(part["mass_kg"], "sensor part mass")
    bounds = part["bounding_box_m"]
    lo,hi = np.asarray(bounds["min"],float),np.asarray(bounds["max"],float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(np.r_[lo,hi]).all() or np.any(hi<=lo):
        raise ValueError("Sensor part bounds must have finite positive extent")
    if "center_of_mass_m" in part and "inertia_kg_m2" in part:
        center = np.asarray(part["center_of_mass_m"],float)
        tensor = np.asarray(part["inertia_kg_m2"],float)
        if (center.shape != (3,) or tensor.shape != (3,3)
                or not np.isfinite(np.r_[center,tensor.ravel()]).all()
                or not np.allclose(tensor,tensor.T,atol=1e-14)
                or np.linalg.eigvalsh(tensor).min() <= 0):
            raise ValueError("CAD inertia must be finite symmetric positive definite about a finite COM")
    else:
        center = (lo+hi)/2
        x,y,z = hi-lo
        tensor = np.diag(mass*np.array([y*y+z*z,x*x+z*z,x*x+y*y])/12)
    com = transform[:3,:3] @ center + transform[:3,3]
    inertia = transform[:3,:3] @ tensor @ transform[:3,:3].T
    inertial = ET.SubElement(link,"inertial")
    _pose_element(inertial,com)
    ET.SubElement(inertial,"mass",value=str(mass))
    ET.SubElement(inertial,"inertia",**{k:str(inertia[i,j]) for k,i,j in [("ixx",0,0),("iyy",1,1),("izz",2,2),("ixy",0,1),("ixz",0,2),("iyz",1,2)]})
    return link


def append_gripper_urdf(root, repo, cfg):
    """Append a physical nominal linkage and separately mounted sensor parts.

    Mounts are explicit parent-inner-finger -> aligned pad-frame xyz/rpy. Right
    component geometry is rotated pi around pad Z, retaining the old inner-face
    convention (-X left, +X right). Each part has its own fixed rigid body.
    """
    if is_adaptive(cfg):
        from phantom.sim.gripper_adaptive import append_gripper_urdf as append_adaptive
        return append_adaptive(root, repo, cfg)
    settings = _settings(cfg)
    if not isinstance(settings.get("mounts"),dict) or any(side not in settings["mounts"] for side in PAD_PARENTS):
        raise ValueError("Explicit left and right inner-finger mounts are required")
    if root.find("link[@name='tool0']") is None:
        raise ValueError("Robot URDF needs a tool0 link")
    source = _nominal(repo)
    geometry = load_sensor_geometry(repo,cfg)
    existing = {element.get("name") for element in root if element.tag in ("link","joint")}
    appended = []
    def append(element):
        if element.get("name") in existing:
            raise ValueError(f"Duplicate gripper name {element.get('name')}")
        existing.add(element.get("name")); root.append(element); appended.append(element)
    nominal_links = [copy.deepcopy(l) for l in source.findall("link") if not l.get("name").endswith("_pad")]
    base = next(l for l in nominal_links if l.get("name") == "robotiq_arg2f_base_link")
    base.set("name","gripper_housing")
    moving_mass = sum(float(l.find("inertial/mass").get("value")) for l in nominal_links if l is not base)
    base_mass = _positive(settings["bare_gripper_mass_kg"],"bare_gripper_mass_kg") - moving_mass
    _positive(base_mass,"base mass after nominal moving-link allocation")
    old_mass = float(base.find("inertial/mass").get("value"))
    base.find("inertial/mass").set("value",str(base_mass))
    inertia = base.find("inertial/inertia")
    for key,value in inertia.attrib.items():
        inertia.set(key,str(float(value)*base_mass/old_mass))
    for link in nominal_links:
        append(link)
    mount = ET.Element("joint",name="tool_gripper",type="fixed")
    ET.SubElement(mount,"parent",link="tool0"); ET.SubElement(mount,"child",link="gripper_housing")
    _pose_element(mount, *root_mount_pose(cfg))
    append(mount)
    for old in source.findall("joint"):
        if old.find("child").get("link").endswith("_pad"):
            continue
        joint = copy.deepcopy(old)
        if joint.find("parent").get("link") == "robotiq_arg2f_base_link":
            joint.find("parent").set("link","gripper_housing")
        if joint.get("name") in JOINT_MULTIPLIERS:
            multiplier = JOINT_MULTIPLIERS[joint.get("name")]
            lower,upper = sorted([0.,.8*multiplier])
            # Pinned visualization URDF has positive limits on negative mimics.
            # Physical limits must admit the nominal counter-rotation.
            joint.find("limit").set("lower",str(lower)); joint.find("limit").set("upper",str(upper))
        append(joint)
    sensor_names = {}
    for side,parent in PAD_PARENTS.items():
        pose = settings["mounts"][side]
        sensor_names[side] = []
        for part in geometry["parts"]:
            link = _part_link(repo,side,part); name = link.get("name")
            append(link); sensor_names[side].append(name)
            joint = ET.Element("joint",name=name+"_mount",type="fixed")
            ET.SubElement(joint,"parent",link=parent); ET.SubElement(joint,"child",link=name)
            _pose_element(joint,pose["xyz"],pose["rpy"]); append(joint)
    sensor_inertia = "CAD COM/tensors at declared mass allocation" if all("center_of_mass_m" in p and "inertia_kg_m2" in p for p in geometry["parts"]) else "bbox approximation for components without CAD inertia"
    return {"model":MODEL,"master_joint":MASTER,"joint_names":list(JOINT_MULTIPLIERS),"joint_units":"radians","pad_links":["left_pad","right_pad"],"sensor_links":sensor_names,"bare_gripper_mass_kg":moving_mass+base_mass,"inertia_model":"Nominal moving-link inertials restored from pinned URDF comments; base inertia scaled to declared remaining bare mass; sensor "+sensor_inertia,"mechanical_model":"One-DOF nominal mimic linkage, not calibrated adaptive/underactuated mechanics","mounts":copy.deepcopy(settings["mounts"])}


def link_transforms_from_joint_positions(repo, cfg, values):
    """Candidate physical/visual link FK in tool0, using each actual joint value."""
    closure_from_joint_positions(values,cfg)
    root = ET.Element("robot",name="gripper_fk"); ET.SubElement(root,"link",name="tool0")
    append_gripper_urdf(root,repo,cfg)
    positions = dict(zip(finger_joint_names(cfg),np.asarray(values,float)))
    transforms = {"tool0":np.eye(4)}
    pending = list(root.findall("joint"))
    while pending:
        progress = False
        for joint in pending[:]:
            parent,child = joint.find("parent").get("link"),joint.find("child").get("link")
            if parent not in transforms:
                continue
            transform = _origin(joint.find("origin"))
            if joint.get("type") == "revolute":
                transform = transform @ _rotation(np.fromstring(joint.find("axis").get("xyz"),sep=" "),positions[joint.get("name")])
            transforms[child] = transforms[parent] @ transform
            pending.remove(joint);progress=True
        if not progress:
            raise ValueError("Unresolved nominal gripper tree")
    return transforms


def configure_gripper_physics(stage, joint_paths, cfg):
    """Author PhysX couplings and exactly one drive, before physics initialization.

    Isaac6's URDF importer writes NewtonMimicAPI. Explicitly replace it here for
    this PhysX candidate, preventing duplicate mimic declarations. Generic USD
    authoring permits CPU tests without a PhysX extension/GPU.
    """
    if is_adaptive(cfg):
        from phantom.sim.gripper_adaptive import configure_gripper_physics as configure_adaptive
        return configure_adaptive(stage, joint_paths, cfg)
    from pxr import Sdf, UsdPhysics
    settings = _settings(cfg)
    stiffness = _positive(settings["drive_stiffness_nm_rad"],"drive_stiffness_nm_rad")
    damping = _positive(settings["drive_damping_nm_s_rad"],"drive_damping_nm_s_rad",zero=True)
    torque = _positive(settings["drive_max_torque_nm"],"drive_max_torque_nm")
    paths = {name:joint_paths[name] for name in JOINT_MULTIPLIERS}
    prims = {name:stage.GetPrimAtPath(str(path)) for name,path in paths.items()}
    if len(set(map(str,paths.values()))) != len(paths) or not all(p.IsA(UsdPhysics.RevoluteJoint) for p in prims.values()):
        raise ValueError("Expected six distinct physical revolute joints")
    axis = {name:UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get() for name,p in prims.items()}
    if any(a not in ("X","Y","Z") for a in axis.values()):
        raise ValueError("Unsupported imported revolute joint axis")
    for name,prim in prims.items():
        prim.RemoveAppliedSchema("NewtonMimicAPI")
        for property_name in list(prim.GetPropertyNames()):
            if property_name.startswith("newton:mimic"):
                prim.RemoveProperty(property_name)
        # Include authored schemas even when CPU usd-core does not load the
        # PhysX plugin; the saved listOp is consumed by the Isaac runtime.
        authored = prim.GetMetadata("apiSchemas")
        for applied in list(authored.GetAppliedItems() if authored else []):
            if applied.startswith("PhysxMimicJointAPI:"):
                prim.RemoveAppliedSchema(applied)
        drive = UsdPhysics.DriveAPI.Apply(prim,"angular")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(0.)
        drive.CreateStiffnessAttr(stiffness*math.pi/180 if name == MASTER else 0.)
        drive.CreateDampingAttr(damping*math.pi/180 if name == MASTER else 0.)
        drive.CreateMaxForceAttr(torque if name == MASTER else 0.)
        if name != MASTER:
            token = "rot"+axis[name]
            prim.AddAppliedSchema("PhysxMimicJointAPI:"+token)
            prefix = "physxMimicJoint:"+token+":"
            # PhysX: follower + gearing * reference + offset = 0.
            for key,value in {"gearing":-JOINT_MULTIPLIERS[name],"offset":0.,"naturalFrequency":0.,"dampingRatio":0.}.items():
                prim.CreateAttribute(prefix+key,Sdf.ValueTypeNames.Float).Set(value)
            prim.CreateAttribute(prefix+"referenceJointAxis",Sdf.ValueTypeNames.Token).Set("rot"+axis[MASTER])
            prim.CreateRelationship(prefix+"referenceJoint").SetTargets([str(paths[MASTER])])
    # Filter structural overlaps, not all gripper self-collision. Opposing pads
    # and housings retain contacts. Fixed sensor components are one rigid unit.
    body_by_name = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_by_name.setdefault(prim.GetName(),[]).append(prim)
    pairs = [(side+"_inner_knuckle",side+"_inner_finger") for side in ("left","right")]
    for side in ("left","right"):
        names = [name for name in body_by_name if name == side+"_pad" or name.startswith(side+"_sensor_")]
        pairs.extend(itertools.combinations(sorted(names),2))
    applied_pairs = []
    for a,b in pairs:
        if len(body_by_name.get(a,[])) != 1 or len(body_by_name.get(b,[])) != 1:
            raise ValueError(f"Missing or ambiguous structural gripper bodies: {a}, {b}")
        pa,pb = body_by_name[a][0],body_by_name[b][0]
        UsdPhysics.FilteredPairsAPI.Apply(pa).CreateFilteredPairsRel().AddTarget(pb.GetPath())
        applied_pairs.append([str(pa.GetPath()),str(pb.GetPath())])
    return {"master_joint":MASTER,"mimic_joints":list(JOINT_MULTIPLIERS)[1:],"coupling":"follower + gearing*master = 0; gearing=-URDF multiplier","drive_type":"force","master_stiffness_nm_rad":stiffness,"master_damping_nm_s_rad":damping,"master_max_torque_nm":torque,"usd_stiffness_nm_degree":stiffness*math.pi/180,"usd_damping_nm_s_degree":damping*math.pi/180,"structural_collision_filters":applied_pairs,"opposing_sensor_contacts_disabled":False}
