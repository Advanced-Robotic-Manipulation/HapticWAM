"""Campaign declaration, frozen dependency and scorer integrity for the limiter."""

import copy
import json
from argparse import Namespace

import pytest

from tests.test_sim_policy_campaign import audited_runtime
from tools.sim.analyze_policy_campaign import (
    load_design,
    servo_reach_limiter_metadata,
)
from tools.sim.run_policy_campaign import simulation_command, source_manifest


def enabled_runtime():
    audit, design, policy, condition, info, server = audited_runtime()
    design["adapter_profile"] = {"servo_reach_limiter": True}
    safety = design["runtime_hardware"]["effective_model"]["safety"]
    safety.update(
        elbow_min_rad=0.40,
        servo_joint_speed_max_rad_s=1.0,
        wrist_extension_stop_m=0.468,
        ur_dh_d1_m=0.1519,
    )
    info["hardware_effective"] = copy.deepcopy(
        design["runtime_hardware"]["effective_model"]
    )
    info["servo_reach_limiter"] = servo_reach_limiter_metadata(design)
    return audit, design, policy, condition, info, server


def reasons(values, run=None, stop=None):
    audit, design, policy, condition, info, server = values
    return audit(
        design,
        policy,
        condition,
        info,
        server,
        run or {"duration_s": 30},
        [0, 30],
        stop,
    )


def test_matching_enabled_contract_is_valid_and_stall_is_not_infrastructure():
    values = enabled_runtime()
    info = values[4]
    run = {
        "duration_s": 30,
        "servo_reach_limiter": {
            **info["servo_reach_limiter"],
            "final_consecutive_rejects": 25,
        },
    }
    assert reasons(values, run, "servo_limiter_stall") == []


@pytest.mark.parametrize("reported", [None, False, True, {}, "enabled"])
def test_missing_or_malformed_enabled_metadata_is_rejected(reported):
    values = enabled_runtime()
    values[4]["servo_reach_limiter"] = reported
    assert any("servo_reach_limiter" in value for value in reasons(values))


@pytest.mark.parametrize(
    "field,wrong",
    [
        ("elbow_min_rad", 0.39),
        ("joint_speed_max_rad_s", 1.1),
        ("joint_speed_max_rad_s", True),
        ("branch_tolerance_rad", 0.4),
        ("bisection_iterations", 4),
        ("bisection_iterations", 3.0),
        ("shoulder_height_m", 0.16),
        ("algorithm", "other"),
        ("measured_wrist_extension_stop_m", 0.47),
        ("consecutive_reject_limit", 26),
        ("tracking_guarantee", 0),
        ("tracking_guarantee", True),
    ],
)
def test_metadata_cannot_silently_tune_constraint_or_claim_tracking(field, wrong):
    values = enabled_runtime()
    values[4]["servo_reach_limiter"][field] = wrong
    assert "effective_servo_reach_limiter_differs_from_campaign" in reasons(values)


@pytest.mark.parametrize("declared", [None, False])
def test_historical_absent_or_disabled_campaign_remains_valid(declared):
    values = audited_runtime()
    if declared is not None:
        values[1]["adapter_profile"] = {"servo_reach_limiter": declared}
    assert reasons(values) == []
    # An undeclared mapper cannot be hidden behind a matching old hardware
    # declaration, nor by omitting the policy-info copy while run.json has it.
    metadata = enabled_runtime()[4]["servo_reach_limiter"]
    values[4]["servo_reach_limiter"] = metadata
    assert "undeclared_servo_reach_limiter" in reasons(values)
    del values[4]["servo_reach_limiter"]
    assert "undeclared_servo_reach_limiter" in reasons(
        values, {"duration_s": 30, "servo_reach_limiter": metadata}
    )


@pytest.mark.parametrize("value", [1, 0, "true", "false", None, {}])
def test_opt_in_must_be_boolean_before_launch_or_scoring(tmp_path, value):
    _, design, *_ = enabled_runtime()
    design["adapter_profile"]["servo_reach_limiter"] = value
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design))
    with pytest.raises(TypeError, match="must be boolean"):
        load_design(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("elbow_min_rad", None),
        ("servo_joint_speed_max_rad_s", 1.2),
        ("servo_joint_speed_max_rad_s", True),
        ("wrist_extension_stop_m", 0.47),
        ("ur_dh_d1_m", 0.18),
    ],
)
def test_enabled_hardware_must_explicitly_declare_unchanged_contract(
    tmp_path, field, value
):
    _, design, *_ = enabled_runtime()
    design["runtime_hardware"]["effective_model"]["safety"][field] = value
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design))
    with pytest.raises(ValueError, match=field):
        load_design(path)


def test_launcher_adds_only_explicit_limiter_flag(tmp_path):
    _, design, policy, condition, *_ = enabled_runtime()
    args = Namespace(
        source=tmp_path,
        evidence=tmp_path / "evidence",
        output=tmp_path,
        port=7799,
        hardware_config=tmp_path / "hardware.yaml",
    )
    on = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    design["adapter_profile"]["servo_reach_limiter"] = False
    off = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    del design["adapter_profile"]["servo_reach_limiter"]
    absent = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    assert off == absent
    assert on == off + ["--servo-reach-limiter"]


def test_frozen_inventory_requires_shared_helper_only_when_enabled(
    tmp_path, monkeypatch
):
    _, design, *_ = enabled_runtime()
    args = Namespace(
        source=tmp_path / "source",
        live_repo=tmp_path / "live",
        evidence=tmp_path / "evidence",
        hardware_config=tmp_path / "hardware.yaml",
        robot_usd=None,
    )
    monkeypatch.setattr("tools.sim.run_policy_campaign.fingerprint", lambda p: str(p))
    with pytest.raises(FileNotFoundError, match="Enabled servo limiter source"):
        source_manifest(args, "design", design)
    helper = args.source / "phantom/drivers/servo_limiter.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# fixture")
    enabled = source_manifest(args, "design", design)
    assert "phantom/drivers/servo_limiter.py" in enabled["source_sha256"]
    design["adapter_profile"]["servo_reach_limiter"] = False
    disabled = source_manifest(args, "design", design)
    assert "phantom/drivers/servo_limiter.py" not in disabled["source_sha256"]
    del design["adapter_profile"]["servo_reach_limiter"]
    assert source_manifest(args, "design", design) == disabled


def test_enabled_metadata_does_not_hide_other_hardware_drift():
    values = enabled_runtime()
    values[4]["hardware_effective"]["safety"]["tactile_depth_limit_mm"] = 99
    assert "effective_hardware_differs_from_frozen_model" in reasons(values)


def test_run_copy_cannot_contradict_policy_info():
    values = enabled_runtime()
    run = {
        "duration_s": 30,
        "servo_reach_limiter": copy.deepcopy(values[4]["servo_reach_limiter"]),
    }
    run["servo_reach_limiter"]["measured_wrist_extension_stop_m"] = 0.5
    assert "run_servo_reach_limiter_differs_from_campaign" in reasons(values, run)


def test_false_declaration_does_not_activate_unrelated_legacy_profile_checks():
    values = audited_runtime()
    values[4].update(gel_contact_coverage="point", record_packet_support=False)
    assert reasons(values) == []
    values[1]["adapter_profile"] = {"servo_reach_limiter": False}
    assert reasons(values) == []
