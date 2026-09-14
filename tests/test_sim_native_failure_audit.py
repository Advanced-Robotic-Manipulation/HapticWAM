"""Failure diagnostics remain bounded and cannot change physics or hide a gate."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.sim.native_failure_audit import (
    NativeFailureHistory, json_value, read_diagnostic, runtime_readback,
)


def test_history_retains_every_recent_step_and_copies_mutable_targets():
    history = NativeFailureHistory()
    names = [f"joint_{i}" for i in range(8)]
    q, qd, target = np.arange(8.), np.arange(8.) / 10, np.zeros(8)
    for step in range(1001):
        target[:] = step
        history.observe("execution", step * .0005, names, q, qd, target)
    target[:] = -999
    rows = history.report()["samples"]
    assert len(rows) == 201
    assert rows[0]["t_s"] == pytest.approx(.4)
    assert rows[-1]["desired_drive_references_rad"]["joint_7"] == 1000
    assert rows[-1]["finger_qd_rad_s"] == list(qd)
    history.observe("after_initialization_settling", 0, names, q, qd, target)
    assert len(history.report()["samples"]) == 1


def test_nonfinite_failure_evidence_and_getter_failure_are_json_safe():
    payload = {"q": np.array([np.nan, np.inf, -np.inf]), "count": np.int64(3)}
    assert json.loads(json.dumps(json_value(payload), allow_nan=False))["q"] == ["nan", "inf", "-inf"]
    report = read_diagnostic(lambda: (_ for _ in ()).throw(RuntimeError("unavailable tensor")))
    assert report == {"available": False, "error": "RuntimeError: unavailable tensor"}


@pytest.mark.parametrize("missing_iteration_getter", [False, True])
def test_runtime_readback_preserves_stage_and_names_effective_joint_limits(missing_iteration_getter):
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Robot", "Xform")
    root.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(64)
    scene = UsdPhysics.Scene.Define(stage, "/Scene").GetPrim()
    scene.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")
    names = [f"joint_{i}" for i in range(8)]
    paths = {}
    for name in names:
        prim = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/" + name).GetPrim()
        paths[name] = str(prim.GetPath())
        prim.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(114.591559)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr(12 * np.pi / 180 if name == names[0] else 0.)
    mimic = stage.GetPrimAtPath(paths[names[3]])
    mimic.CreateRelationship("physxMimicJoint:rotX:referenceJoint").SetTargets([paths[names[0]]])
    props = np.zeros(8, dtype=[("maxVelocity", float), ("stiffness", float)])
    props["maxVelocity"] = 2
    props["stiffness"][0] = 12
    robot = SimpleNamespace(
        dof_properties=props,
        get_solver_position_iteration_count=lambda: 64,
        get_solver_velocity_iteration_count=lambda: 8,
        get_articulation_controller=lambda: SimpleNamespace(get_gains=lambda: (np.ones(8), np.zeros(8))),
        get_applied_action=lambda: SimpleNamespace(joint_positions=np.zeros(8)),
        get_measured_joint_efforts=lambda joint_indices: np.ones(len(joint_indices)),
        get_applied_joint_efforts=lambda joint_indices: np.zeros(len(joint_indices)),
    )
    if missing_iteration_getter:
        del robot.get_solver_position_iteration_count
    before = stage.GetRootLayer().ExportToString()
    report = runtime_readback(stage, "/Robot", paths, robot, names, np.arange(8))
    assert stage.GetRootLayer().ExportToString() == before
    assert report["live_readbacks"]["dof_properties"]["value"][names[3]]["maxVelocity"] == 2
    assert report["joints"][names[3]]["relationships"]["physxMimicJoint:rotX:referenceJoint"] == [paths[names[0]]]
    assert report["usd_physics_scenes"]["/Scene"]["physxScene:solverType"]["value"] == "TGS"
    assert report["live_readbacks"]["position_iterations"]["available"] is not missing_iteration_getter
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad", [None, "getter", "nonfinite", "negative", "multiple_articulations"])
def test_armature_readback_uses_named_native_values_without_usd_fallback(bad):
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot", "Xform")
    names, indices = ["passive", "motor"], np.array([3, 1])
    paths = {}
    for name in names:
        prim = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/" + name).GetPrim()
        paths[name] = str(prim.GetPath())
        prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(99.)
    values = np.array([[0., .005, 0., .001]])
    if bad == "nonfinite":
        values[0, 3] = np.nan
    elif bad == "negative":
        values[0, 3] = -.001
    elif bad == "multiple_articulations":
        values = np.repeat(values, 2, axis=0)
    def getter():
        if bad == "getter":
            raise RuntimeError("native getter unavailable")
        return values
    robot = SimpleNamespace(_articulation_view=SimpleNamespace(
        _physics_view=SimpleNamespace(get_dof_armatures=getter)))
    before = stage.GetRootLayer().ExportToString()
    report = runtime_readback(stage, "/Robot", paths, robot, names, indices)
    assert stage.GetRootLayer().ExportToString() == before
    armature = report["live_readbacks"]["armatures_kg_m2"]
    if bad is None:
        assert armature == {"available": True, "value": {"passive": .001, "motor": .005}}
    else:
        assert armature["available"] is False
        assert "value" not in armature
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("velocity_error", [False, True])
def test_runner_preserves_mechanical_gate_when_readbacks_or_contacts_fail(tmp_path, monkeypatch, velocity_error):
    from phantom.sim import gripper_adaptive

    source = Path(__file__).resolve().parents[1] / "tools/sim/run_waffles.py"
    run = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "run")
    check = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == "check_native_mechanics")
    diagnostic = {"passed": False, "gates": {"joint_coupling": False}, "coupling_max_abs_rad": .006}
    monkeypatch.setattr(gripper_adaptive, "mechanical_diagnostics", lambda _: diagnostic)
    qd = np.arange(8.) / 10
    def get_velocity():
        if velocity_error:
            raise RuntimeError("velocity unavailable")
        return qd
    environment = {
        "adaptive_gripper": True, "np": np, "json": json,
        "robot": SimpleNamespace(get_joint_positions=lambda: np.zeros(8), get_joint_velocities=get_velocity),
        "fingers": np.arange(8), "finger_names": list(gripper_adaptive.JOINT_NAMES),
        "native_failure_history": NativeFailureHistory(), "native_mechanics_monitor": {"checks": 0},
        "json_value": json_value, "read_diagnostic": read_diagnostic,
        "runtime_readback": lambda *args: (_ for _ in ()).throw(RuntimeError("readback failed")),
        "stage": None, "roots": ["/Robot"], "paths": {"joint_paths": {}},
        "gel_views": SimpleNamespace(get_all=lambda _: (_ for _ in ()).throw(RuntimeError("contact failed"))),
        "robot_environment_views": None, "gripper_wrist": None, "dt": .0005,
        "args": SimpleNamespace(output=tmp_path),
    }
    exec(compile(ast.Module(body=[check], type_ignores=[]), str(source), "exec"), environment)
    with pytest.raises(RuntimeError, match="Native gripper mechanics failed"):
        environment["check_native_mechanics"]("execution", 12.869, np.ones(8) * .5)
    failure = json.loads((tmp_path / "native_mechanics_failure.json").read_text())
    assert failure["diagnostic"] == diagnostic
    assert failure["finger_qd_rad_s"] == (None if velocity_error else list(qd))
    assert failure["finger_qd_readback"]["available"] is not velocity_error
    if velocity_error:
        assert failure["recent_physics_steps"]["samples"][0]["finger_qd_read_error"] == "RuntimeError: velocity unavailable"
    assert failure["runtime_readback"]["available"] is False
    assert failure["failure_step_contacts"]["gel"]["available"] is False
    assert failure["desired_drive_references_rad"]["finger_joint"] == .5
    assert len(failure["recent_physics_steps"]["samples"]) == 1
