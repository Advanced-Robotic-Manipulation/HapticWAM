"""Sharing model inference must not conceal simulator controller/config drift."""

import copy
from pathlib import Path

import pytest

from phantom.config.hardware import HardwareConfig, load_hardware
from tests.test_sim_campaign_constraint_hold import hold_runtime
from tests.test_sim_campaign_servo_limiter import reasons
from tools.sim.analyze_policy_campaign import load_design, shared_server_hardware


def shared_runtime():
    values = hold_runtime()
    design, info, server = values[1], values[4], values[5]
    model = load_hardware(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml", quiet=True).model_dump(mode="json")
    model["safety"].update(servo_constraint_hold_s=2.5, elbow_min_rad=0.40,
                           servo_joint_speed_max_rad_s=1.0, wrist_extension_stop_m=0.468,
                           ur_dh_d1_m=0.1519)
    client = HardwareConfig.model_validate(model)
    server_model = copy.deepcopy(model)
    server_model["safety"]["servo_constraint_hold_s"] = None
    server_hw = HardwareConfig.model_validate(server_model)
    design["runtime_hardware"] = {"sha256": "c" * 64, "config_hash": client.config_hash(),
                                   "effective_model": client.model_dump(mode="json")}
    design["shared_server_hardware"] = {"sha256": "a" * 64, "config_hash": server_hw.config_hash(),
                                        "effective_model": server_hw.model_dump(mode="json")}
    info["hardware_effective"] = client.model_dump(mode="json")
    server.update(hardware_sha256="a" * 64, hardware_config_hash=server_hw.config_hash())
    return values


def rehash(spec):
    spec["config_hash"] = HardwareConfig.model_validate(spec["effective_model"]).config_hash()


def test_declared_hold_only_difference_preserves_actual_server_identity():
    values = shared_runtime()
    before = copy.deepcopy(values[5])
    assert reasons(values) == []
    assert values[5] == before
    assert values[5]["hardware_sha256"] != values[1]["runtime_hardware"]["sha256"]


def test_missing_opt_in_retains_strict_historical_server_check():
    values = shared_runtime()
    values[1].pop("shared_server_hardware")
    assert "server_hardware_differs_from_frozen_configuration" in reasons(values)


@pytest.mark.parametrize("section,field,value", [
    ("safety", "wrench_limit_N", 44.0),
    ("safety", "wrench_baseline_mode", "episode_fixed"),
    ("arm", "payload_kg", 0.77),
    ("gripper", "max_close_cmd", 0.5),
    ("control", "action_rate_hz", 11.0),
])
def test_shape_preserving_semantic_changes_are_rejected(section, field, value):
    values = shared_runtime()
    spec = values[1]["shared_server_hardware"]
    spec["effective_model"][section][field] = value
    rehash(spec)
    with pytest.raises(ValueError, match="beyond the hold deadline"):
        shared_server_hardware(values[1])
    assert "shared_server_hardware_contract_invalid" in reasons(values)


@pytest.mark.parametrize("value", [0.5, 4.0, True])
def test_other_client_deadlines_cannot_expand_the_exception(value):
    values = shared_runtime()
    spec = values[1]["runtime_hardware"]
    spec["effective_model"]["safety"]["servo_constraint_hold_s"] = value
    rehash(spec)
    with pytest.raises(ValueError):
        shared_server_hardware(values[1])


def test_server_must_retain_legacy_deadline():
    values = shared_runtime()
    spec = values[1]["shared_server_hardware"]
    spec["effective_model"]["safety"]["servo_constraint_hold_s"] = 2.5
    rehash(spec)
    with pytest.raises(ValueError, match="legacy None"):
        shared_server_hardware(values[1])


@pytest.mark.parametrize("field,value", [("sha256", "wrong"), ("config_hash", "wrong"), ("effective_model", {})])
def test_malformed_contract_is_invalid(field, value):
    values = shared_runtime()
    values[1]["shared_server_hardware"][field] = value
    assert "shared_server_hardware_contract_invalid" in reasons(values)


def test_actual_server_hash_must_match_declared_server():
    values = shared_runtime()
    values[5]["hardware_sha256"] = values[1]["runtime_hardware"]["sha256"]
    assert "server_hardware_differs_from_frozen_configuration" in reasons(values)


def test_client_effective_hardware_remains_independently_checked():
    values = shared_runtime()
    values[4]["hardware_effective"] = values[1]["shared_server_hardware"]["effective_model"]
    assert "effective_hardware_differs_from_frozen_model" in reasons(values)


def test_loading_frozen_design_refuses_shared_hardware_drift(tmp_path):
    import json

    values = shared_runtime()
    values[1]["shared_server_hardware"]["config_hash"] = "wrong"
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(values[1]))
    with pytest.raises(ValueError, match="config hash disagrees"):
        load_design(path)
