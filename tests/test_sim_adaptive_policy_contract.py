"""CPU preflight for explicitly exploratory native-gripper teacher experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from phantom.sim import gripper_adaptive as native
from phantom.sim.gripper_articulation import MODEL as LEGACY_MODEL
from tools.sim import run_waffles as runner

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """Parse the real CLI; keep this fixture independent of remote/Isaac tools."""
    cfg = json.loads(
        (REPO / "configs/sim/waffles_w2l_adaptive_parallel_20260909.json").read_text()
    )
    cfg["physics"]["dt"] = .001
    config = tmp_path / "scene.json"
    config.write_text(json.dumps(cfg))
    baseline = tmp_path / "no_contact.npz"
    # This contract hashes an input file; the tactile proxy separately validates
    # its array schema. A tiny NPZ suffices to test byte-bound provenance here.
    np.savez(baseline, provenance_marker=np.array([17]))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "nfe": 1,
        "guidance": 1.0,
        "k_seeds": 4,
        "parity_fixes": True,
        "persistent_noise": True,
        "task_text": "Pick up the waffles and place them in the box.",
        "drop_video": False,
        "close_p": .5,
        "action_time_origin": "inference_ready",
    }))
    monkeypatch.setattr(sys, "argv", [
        "run_waffles.py", "--episode", str(tmp_path / "episode"),
        "--output", str(tmp_path / "output"), "--config", str(config),
        "--mode", "policy", "--experimental-adaptive-policy",
        "--policy-mode", "teacher", "--tactile", "measured_baseline_proxy",
        "--wrist", "gripper_contact_proxy", "--tactile-baseline", str(baseline),
        "--policy-config", str(policy), "--gel-contact-coverage", "manifold_patch_v2",
        "--save-policy-observations", "--record-gel-contacts",
        "--record-packet-support", "--record-robot-environment-contacts",
    ])
    return runner.arguments(), cfg


def check(experiment):
    return runner.validate_adaptive_policy_experiment(*experiment)


def write_policy(experiment, **changes):
    args, _ = experiment
    settings = json.loads(args.policy_config.read_text())
    settings.update(changes)
    args.policy_config.write_text(json.dumps(settings))


def test_valid_contract_is_explicitly_unqualified_and_binds_exact_inputs(experiment):
    args, cfg = experiment
    saved = copy.deepcopy(cfg)
    result = check(experiment)
    assert result["status"] == "exploratory_uncalibrated_teacher_inference"
    assert result["hardware_transfer_qualified"] is False
    assert result["success_source"] == "Independent packet state and bin support, not controller FINISH"
    assert result["policy_config_sha256"] == hashlib.sha256(args.policy_config.read_bytes()).hexdigest()
    assert result["tactile_baseline_sha256"] == hashlib.sha256(args.tactile_baseline.read_bytes()).hexdigest()
    assert result["declared_policy_settings"] == json.loads(args.policy_config.read_text())
    assert cfg == saved
    assert not args.output.exists()
    with args.tactile_baseline.open("ab") as stream:
        stream.write(b"new provenance")
    changed = check(experiment)
    assert changed["tactile_baseline_sha256"] != result["tactile_baseline_sha256"]
    assert changed["policy_config_sha256"] == result["policy_config_sha256"]


@pytest.mark.parametrize("missing_attribute", [False, True])
def test_native_policy_without_explicit_optin_is_rejected(experiment, missing_attribute):
    args, _ = experiment
    if missing_attribute:
        del args.experimental_adaptive_policy
    else:
        args.experimental_adaptive_policy = False
    with pytest.raises(ValueError, match="requires --experimental-adaptive-policy"):
        check(experiment)


@pytest.mark.parametrize("mode", ["replay", "dynamics", "contact_probe", "command_replay"])
def test_optin_cannot_be_mislabeled_as_recorded_motion(experiment, mode):
    args, _ = experiment
    args.mode = mode
    with pytest.raises(ValueError, match="requires native W2L policy mode"):
        check(experiment)
    args.experimental_adaptive_policy = False
    assert check(experiment) is None


def test_legacy_gripper_cannot_claim_native_optin(experiment):
    args, cfg = experiment
    cfg["gripper"]["model"] = LEGACY_MODEL
    with pytest.raises(ValueError, match="requires native W2L policy mode"):
        check(experiment)
    args.experimental_adaptive_policy = False
    assert check(experiment) is None


@pytest.mark.parametrize("mode", ["student", "vision_only"])
def test_contract_covers_student_and_vision_only_modes(experiment, mode):
    # sim zoo 2026-09-12: the Cosmos students and the vision-only model run under the same
    # explicit contract as the teacher (the old teacher-only gate is gone)
    experiment[0].policy_mode = mode
    result = check(experiment)
    assert result["hardware_transfer_qualified"] is False


@pytest.mark.parametrize("field,value", [
    ("tactile", "zero_ablation"), ("tactile", "contact_proxy"),
    ("wrist", "zero_ablation"), ("wrist", "contact_proxy"),
    ("gel_contact_coverage", "point"), ("gel_contact_coverage", "manifold_patch"),
])
def test_missing_or_different_sensor_mapping_is_not_normal_teacher_input(experiment, field, value):
    setattr(experiment[0], field, value)
    with pytest.raises(ValueError, match="requires"):
        check(experiment)


@pytest.mark.parametrize("flag", [
    "save_policy_observations", "record_gel_contacts", "record_packet_support",
    "record_robot_environment_contacts",
])
def test_each_observation_object_and_collision_audit_is_required(experiment, flag):
    setattr(experiment[0], flag, False)
    with pytest.raises(ValueError, match="requires policy observations"):
        check(experiment)


@pytest.mark.parametrize("field", ["policy_config", "tactile_baseline"])
@pytest.mark.parametrize("invalid", ["unset", "absent", "directory"])
def test_inputs_must_be_existing_files(experiment, tmp_path, field, invalid):
    value = {"unset": None, "absent": tmp_path / "missing", "directory": tmp_path}[invalid]
    setattr(experiment[0], field, value)
    with pytest.raises(ValueError, match=f"existing {field} file"):
        check(experiment)


@pytest.mark.parametrize("value", ["{malformed", "[]", "null", "{}"])
def test_policy_json_cannot_silently_inherit_resident_server_settings(experiment, value):
    experiment[0].policy_config.write_text(value)
    with pytest.raises(ValueError, match="policy_config"):
        check(experiment)


@pytest.mark.parametrize("key", [
    "nfe", "guidance", "k_seeds", "parity_fixes", "persistent_noise",
    "task_text", "drop_video", "close_p", "action_time_origin",
])
def test_every_remote_inference_setting_must_be_explicit(experiment, key):
    path = experiment[0].policy_config
    settings = json.loads(path.read_text())
    del settings[key]
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="missing="):
        check(experiment)


def test_unknown_policy_option_is_rejected_before_remote_connection(experiment):
    write_policy(experiment, temperature=.8)
    with pytest.raises(ValueError, match="unknown=.*temperature"):
        check(experiment)


@pytest.mark.parametrize("key,value", [
    ("nfe", 0), ("nfe", -1), ("nfe", True), ("nfe", 1.0), ("nfe", "1"),
    ("k_seeds", 0), ("k_seeds", False), ("k_seeds", 2.5),
    ("guidance", float("nan")), ("guidance", float("inf")),
    ("guidance", -.01), ("guidance", True),
    ("close_p", float("nan")), ("close_p", float("inf")),
    ("close_p", -.01), ("close_p", 1.01), ("close_p", False),
    ("parity_fixes", 1), ("persistent_noise", "false"), ("drop_video", None),
    ("task_text", " "), ("task_text", None), ("task_text", 1),
])
def test_invalid_explicit_settings_are_rejected(experiment, key, value):
    write_policy(experiment, **{key: value})
    with pytest.raises(ValueError, match=f"policy_config {key}"):
        check(experiment)


@pytest.mark.parametrize("close_p", [0, 1])
def test_contract_allows_declared_ablation_recipes_without_selecting_a_winner(experiment, close_p):
    write_policy(experiment, nfe=5, k_seeds=1, guidance=0, parity_fixes=False,
                 persistent_noise=False, drop_video=True, close_p=close_p)
    assert check(experiment)["declared_policy_settings"]["close_p"] == close_p


@pytest.mark.parametrize("dt", [.004, .002, 0, -.001, float("nan"), float("inf")])
def test_main_optin_does_not_bypass_native_timestep_guard_or_start_kit(experiment, monkeypatch, dt):
    args, cfg = experiment
    cfg["physics"]["dt"] = dt
    args.config.write_text(json.dumps(cfg))
    monkeypatch.setattr(runner, "arguments", lambda: args)
    fake_isaac = ModuleType("isaacsim")

    def forbidden_start(*_args, **_kwargs):
        pytest.fail("invalid native timestep reached Kit startup")

    fake_isaac.SimulationApp = forbidden_start
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaac)
    with pytest.raises(ValueError, match="physics"):
        runner.main()
    assert not (args.output / "adaptive_policy_experiment.json").exists()


def test_exploratory_optin_does_not_reclassify_broken_native_mechanics(experiment):
    _, cfg = experiment
    assert check(experiment)["hardware_transfer_qualified"] is False
    native.validate_physics_timestep(cfg)
    healthy = native.joint_targets(.35, cfg)
    assert native.mechanical_diagnostics(healthy)["passed"]
    detached = healthy.copy()
    detached[6] = -.05
    result = native.mechanical_diagnostics(detached)
    assert result["gates"]["joint_coupling"]
    assert result["gates"]["joint_limits"]
    assert not result["gates"]["adaptive_loop_closure"]
    assert not result["passed"]
