"""Native profile activation and provenance, with all hardware connections blocked."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phantom.config.hardware import HardwareConfig, load_hardware
from phantom.deploy.reach_profile import PROFILE, PROFILE_LIMITS, apply_reach_profile, main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rig_hw():
    return load_hardware(ROOT / "configs/hardware.nuc.yaml", quiet=True)


def with_fields(hw, section, **fields):
    data = hw.model_dump(mode="python")
    data[section].update(fields)
    return HardwareConfig.model_validate(data)


def test_unselected_profile_is_identity_and_retains_custom_limits(rig_hw):
    custom = with_fields(rig_hw, "safety", elbow_min_rad=0.5)
    assert apply_reach_profile(custom, None) is custom


def test_bounded_changes_exactly_three_fields_preserving_base_and_all_guards(rig_hw):
    before = rig_hw.model_dump(mode="python")
    original_hash = rig_hw.config_hash()
    effective = apply_reach_profile(rig_hw, PROFILE)
    expected = rig_hw.model_dump(mode="python")
    expected["safety"].update(PROFILE_LIMITS)
    assert effective.model_dump(mode="python") == expected
    assert rig_hw.model_dump(mode="python") == before
    assert rig_hw.config_hash() == original_hash
    assert effective.config_hash() != original_hash
    assert effective.shape_relevant_fields() == rig_hw.shape_relevant_fields()
    assert apply_reach_profile(effective, PROFILE) == effective


@pytest.mark.parametrize("name,value", [
    ("elbow_min_rad", 0.5), ("elbow_min_rad", 0.2),
    ("servo_joint_speed_max_rad_s", 0.5), ("servo_joint_speed_max_rad_s", 1.5),
    ("servo_constraint_hold_s", 1.0), ("servo_constraint_hold_s", 4.0),
])
def test_conflicting_existing_package_is_not_overridden(rig_hw, name, value):
    custom = with_fields(rig_hw, "safety", **{name: value})
    before = custom.model_dump()
    with pytest.raises(ValueError, match="conflicts"):
        apply_reach_profile(custom, PROFILE)
    assert custom.model_dump() == before


@pytest.mark.parametrize("section,fields", [
    ("arm", {"model": "UR5"}), ("arm", {"model": "UR3e"}), ("arm", {"dof": 7}),
    ("safety", {"ur_dh_a2_a3_d4_m": (0.3, 0.21325, 0.11235)}),
    ("safety", {"ur_dh_d1_m": 0.15}),
    ("safety", {"wrist_extension_stop_m": None}),
    ("safety", {"wrist_extension_stop_m": 0.461}),
    ("safety", {"wrist_extension_stop_m": 0.48}),
    ("safety", {"wrist_extension_stop_m": float("nan")}),
    ("safety", {"joint_speed_stop_rad_s": 0.9}),
])
def test_incompatible_hardware_rejected(rig_hw, section, fields):
    with pytest.raises(ValueError, match="bounded_v1"):
        apply_reach_profile(with_fields(rig_hw, section, **fields), PROFILE)


def test_unknown_profile_rejected(rig_hw):
    with pytest.raises(ValueError, match="Unknown"):
        apply_reach_profile(rig_hw, "typo")


def test_file_only_check_reports_validated_effective_config(capsys):
    assert main(["--hardware", str(ROOT / "configs/hardware.nuc.yaml"),
                 "--profile", PROFILE]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["check"] == "configuration_only_no_hardware"
    assert report["effective_limits"] == PROFILE_LIMITS
    assert report["base_config_hash"] != report["effective_config_hash"]
    HardwareConfig.model_validate(report["effective_hardware"])


def test_file_only_invalid_config_fails_before_any_runtime(tmp_path, rig_hw, capsys):
    path = tmp_path / "wrong_arm.yaml"
    path.write_text(with_fields(rig_hw, "arm", model="UR5").snapshot_yaml())
    with pytest.raises(SystemExit) as error:
        main(["--hardware", str(path), "--profile", PROFILE])
    assert error.value.code == 2
    assert "requires the configured six-joint UR3" in capsys.readouterr().err


def test_native_parser_opt_in_default_unchanged():
    from phantom.scripts.run_deploy import build_parser
    parser = build_parser()
    args = ["--system", "teacher", "--task", "waffles"]
    assert parser.parse_args(args).servo_reach_profile is None
    assert parser.parse_args(args + ["--servo-reach-profile", PROFILE]).servo_reach_profile == PROFILE


@pytest.mark.parametrize("profile", [None, PROFILE])
def test_native_main_binds_profile_to_real_driver_and_records_effective_limits(
        monkeypatch, tmp_path, rig_hw, profile):
    """Run the actual runtime/factory/URArm constructor, stop before connect."""
    from phantom.deploy import runtime, start_pose
    from phantom.drivers.real.ur import URArm
    from phantom.scripts import run_deploy

    # Only the arm takes the real constructor; its imports/connections are lazy.
    base = with_fields(rig_hw, "mode", drivers="mock", overrides={"arm": "real"})
    monkeypatch.setattr(run_deploy, "load_hardware", lambda *_: base)
    monkeypatch.setattr(run_deploy, "load_paths", lambda: SimpleNamespace(
        validate=lambda **_: None, episodes_root=lambda: tmp_path))
    monkeypatch.setattr(run_deploy, "preflight_disk", lambda *_: 0)
    monkeypatch.setattr(run_deploy, "build_policy", lambda *args: SimpleNamespace(
        nfe=1, guidance=1.0, rf=SimpleNamespace()))
    monkeypatch.setattr(start_pose, "load_start_stats", lambda: {})

    def no_connection(*args, **kwargs):
        pytest.fail("test must not open any hardware connection")

    monkeypatch.setattr(URArm, "connect", no_connection)
    seen = {}

    class PreConnectionBoundary(Exception):
        pass

    def before_connect(rt):
        seen["runtime"] = rt
        raise PreConnectionBoundary

    monkeypatch.setattr(runtime.DeploymentRuntime, "__enter__", before_connect)
    argv = ["--system", "teacher", "--task", "waffles", "--tiny",
            "--no-hitbox", "--no-z-floor", "--allow-ood-start", "--out", str(tmp_path)]
    if profile:
        argv += ["--servo-reach-profile", profile]
    with pytest.raises(PreConnectionBoundary):
        run_deploy.main(argv)
    rt = seen["runtime"]
    assert isinstance(rt.rig.arm, URArm)
    assert rt.rig.arm.hw is rt.hw
    assert rt.base_config_hash == base.config_hash()
    assert rt.rig.arm._ctrl is None and rt.rig.arm._recv is None
    expected_limits = PROFILE_LIMITS if profile else {name: None for name in PROFILE_LIMITS}
    for name, value in expected_limits.items():
        assert getattr(rt.rig.arm.hw.safety, name) == value
        assert rt.deploy_overrides["safety_effective"][name] == value
    limits = rt.rig.arm._servo_limits()
    assert limits.elbow_min_rad == expected_limits["elbow_min_rad"]
    assert limits.joint_speed_max_rad_s == expected_limits["servo_joint_speed_max_rad_s"]
    if profile:
        assert rt.deploy_overrides["servo_reach_profile"] == {
            "id": PROFILE, "effective_limits": PROFILE_LIMITS}
    else:
        assert "servo_reach_profile" not in rt.deploy_overrides
    # CLI has established lift/TCP overrides; all other measured safety fields
    # must still be the base values, including force, geometry and workspace.
    actual = rt.hw.safety.model_dump()
    expected = base.safety.model_dump()
    for name in (*PROFILE_LIMITS, "lift_complete_z_m"):
        actual.pop(name)
        expected.pop(name)
    assert actual == expected


def test_native_main_rejects_before_paths_models_or_driver_factory(monkeypatch, rig_hw):
    from phantom.deploy import runtime
    from phantom.scripts import run_deploy

    bad = with_fields(rig_hw, "arm", model="UR5")
    monkeypatch.setattr(run_deploy, "load_hardware", lambda *_: bad)

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible hardware must fail before runtime preparation")

    monkeypatch.setattr(run_deploy, "load_paths", forbidden)
    monkeypatch.setattr(run_deploy, "build_policy", forbidden)
    monkeypatch.setattr(runtime, "make_rig", forbidden)
    assert run_deploy.main(["--system", "teacher", "--task", "waffles",
                            "--servo-reach-profile", PROFILE]) == 2
