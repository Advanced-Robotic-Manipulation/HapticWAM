"""Physical/sensor isolation invariants for overlays of the actual frozen runner."""

import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sim.prepare_teacher_anchor_sensors import (
    BASE_RUNNER_SHA,
    LAUNCHER,
    RUNNER,
    patch_runner,
    prepare_overlay,
    sha,
)

REPO = Path(__file__).resolve().parents[1]
FROZEN = Path(__file__).parent / "fixtures/teacher_anchor_v1/run_waffles.py.txt"


def functions(raw):
    return {
        node.name: node
        for node in ast.walk(ast.parse(raw))
        if isinstance(node, ast.FunctionDef)
    }


def dump(node):
    return ast.dump(node, include_attributes=False)


def fixture_source(tmp_path):
    base = tmp_path / "frozen"
    files = {
        RUNNER: FROZEN.read_bytes(),
        LAUNCHER: b"#!/bin/sh\nexit 0\n",
        "phantom/sim/scene.py": b"# immutable physics\n",
        "phantom/sim/tactile_proxy.py": b"# immutable pressure synthesis\n",
        "phantom/sim/policy_adapter.py": b"# immutable controller\n",
        "tools/sim/gel_contact.py": b"# v1 sensor\n",
    }
    for name, raw in files.items():
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    freeze = tmp_path / "frozen_inputs.json"
    freeze.write_text(
        json.dumps(
            {
                "source_root": str(base),
                "source_sha256": {name: sha(base / name) for name in files},
            }
        )
    )
    return base, freeze


def test_fixture_is_exact_frozen_success_runner():
    assert sha(FROZEN) == BASE_RUNNER_SHA


def test_gel_variant_changes_only_cli_not_runtime_mechanics():
    old = functions(FROZEN.read_bytes())
    new = functions(patch_runner(FROZEN.read_bytes(), "gel_v2"))
    changed = [name for name in old if dump(old[name]) != dump(new[name])]
    assert changed == ["arguments"]
    assert dump(old["main"]) == dump(new["main"])


def test_wrist_variant_preserves_tactile_synthesis_and_all_articulation_writes():
    before = ast.parse(FROZEN.read_bytes())
    after = ast.parse(patch_runner(FROZEN.read_bytes(), "gripper_wrist"))
    assert dump(functions(FROZEN.read_bytes())["tactile_proxy"]) == dump(
        functions(ast.unparse(after))["tactile_proxy"]
    )

    def physical_writes(tree):
        methods = {
            "set_joint_positions",
            "set_joint_velocities",
            "apply_action",
            "set_world_pose",
            "set_linear_velocity",
            "set_angular_velocity",
            "step",
            "reset",
        }
        return [
            dump(n)
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in methods
        ]

    assert physical_writes(before) == physical_writes(after)
    # Both model input and execution telemetry read the same cached six-vector.
    assert "wrist_ft=measured_wrist" in ast.unparse(after)
    assert (
        '"measured_wrist_ft": measured_wrist'
        in patch_runner(FROZEN.read_bytes(), "gripper_wrist").decode()
    )


def test_wrist_readout_is_one_sample_and_returns_defensive_copies():
    nodes = functions(patch_runner(FROZEN.read_bytes(), "gripper_wrist"))
    preamble = ast.parse("""
def bind(gripper, output):
    wrist_value = None
    wrist_sample_t = None
    next_wrist_t = 0.0
    gripper_wrist = gripper
    wrist_contact_file = output
    wrist_bias = np.array([1.,2.,3.,4.,5.,6.])
    dt = .004
    control_dt = .008
    args = SimpleNamespace(wrist="gripper_contact_proxy")
""")
    binder = preamble.body[0]
    binder.body.extend([nodes["wrist_proxy"], nodes["sample_gripper_wrist"]])
    binder.body.extend(ast.parse("return wrist_proxy, sample_gripper_wrist").body)
    namespace = {"np": np, "json": json, "SimpleNamespace": SimpleNamespace}
    exec(  # noqa: S102 - execute only AST extracted from the hash-pinned test fixture
        compile(ast.fix_missing_locations(preamble), "sensor-cache-test", "exec"),
        namespace,
    )

    class Gripper:
        calls = 0

        def sample(self, pose, bias, physics_dt):
            self.calls += 1
            assert physics_dt == 0.004
            np.testing.assert_array_equal(bias, [1, 2, 3, 4, 5, 6])
            return bias + 10, {"sample": self.calls}

    gripper, output = Gripper(), io.StringIO()
    read, sample = namespace["bind"](gripper, output)
    with pytest.raises(RuntimeError, match="not been sampled"):
        read(np.zeros(6), [])
    sample(0.004, np.zeros(6))
    first = read(np.zeros(6), [])
    first[:] = -1
    np.testing.assert_array_equal(read(np.zeros(6), []), [11, 12, 13, 14, 15, 16])
    assert gripper.calls == 1
    assert json.loads(output.getvalue()) == {"t": 0.004, "sample": 1}


@pytest.mark.parametrize(
    "variant,changed",
    [
        ("gel_v2", {RUNNER, "tools/sim/gel_contact.py"}),
        (
            "gripper_wrist",
            {
                RUNNER,
                "tools/sim/gripper_wrist.py",
                "tools/sim/robot_environment_contacts.py",
            },
        ),
    ],
)
def test_overlay_isolates_changes_and_does_not_write_base(tmp_path, variant, changed):
    base, freeze = fixture_source(tmp_path)
    before = {str(p.relative_to(base)): sha(p) for p in base.rglob("*") if p.is_file()}
    output = tmp_path / "overlay"
    manifest = prepare_overlay(base, output, REPO, freeze, variant)
    assert set(manifest["changed_files"]) == changed
    assert manifest["required_environment"] == {"PYTHONDONTWRITEBYTECODE": "1"}
    assert not (output / LAUNCHER).is_symlink()
    assert (output / "phantom/sim/scene.py").is_symlink()
    for name, digest in before.items():
        assert sha(base / name) == digest
        if name not in changed:
            assert sha(output / name) == digest
    compile((output / RUNNER).read_bytes(), str(output / RUNNER), "exec")
    assert not list(tmp_path.rglob("*.pyc"))


def test_modified_base_fails_before_creating_overlay(tmp_path):
    base, freeze = fixture_source(tmp_path)
    (base / "phantom/sim/scene.py").write_text("modified material\n")
    output = tmp_path / "overlay"
    with pytest.raises(ValueError, match="frozen source hash mismatch"):
        prepare_overlay(base, output, REPO, freeze, "gel_v2")
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    base, freeze = fixture_source(tmp_path)
    output = tmp_path / "overlay"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(ValueError, match="output must be new"):
        prepare_overlay(base, output, REPO, freeze, "gel_v2")
    assert sentinel.read_bytes() == b"keep"


def test_unpinned_runner_and_unknown_combined_variant_are_rejected():
    with pytest.raises(ValueError, match="base runner hash"):
        patch_runner(FROZEN.read_bytes() + b"\n", "gel_v2")
    with pytest.raises(ValueError, match="one separate sensor variant"):
        patch_runner(FROZEN.read_bytes(), "combined")
