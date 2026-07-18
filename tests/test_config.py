"""Hardware/paths config validators."""

import pytest
from pydantic import ValidationError

from phantom_test_utils import make_hw


def test_default_config_loads(default_hw):
    assert default_hw.tactile.field_ch == 8
    assert default_hw.n_fingers == 2
    assert default_hw.ur_state_dim == 2 * 6 + 6 + 6 + 2
    assert default_hw.contact_state_dim == 11
    assert default_hw.cpk_shape == (288 // 8, 384 // 8)


def test_cb3_requires_ft300s_and_low_rate():
    with pytest.raises(ValidationError, match="CB3|125"):
        make_hw(arm={"generation": "cb3"})   # 500 Hz + ur_internal both invalid
    hw = make_hw(arm={"generation": "cb3", "rtde_receive_hz": 125.0,
                      "rtde_control_hz": 125.0},
                 wrist_ft={"source": "ft300s", "rate_hz": 100.0},
                 control={"executor_rate_hz": 125.0})
    assert hw.arm.generation == "cb3"


def test_eseries_rate_whitelist():
    with pytest.raises(ValidationError, match="125/250/500"):
        make_hw(arm={"rtde_receive_hz": 300.0})


def test_field_ds_must_divide():
    with pytest.raises(ValidationError, match="divide"):
        make_hw(recording={"field_ds": {"h": 70, "w": 96}})


def test_channel_sum_must_be_8():
    with pytest.raises(ValidationError, match="sum to 8"):
        make_hw(tactile={"field_channels": {"deformation2d": 2, "depth": 2,
                                            "shear": 2, "dist_force": 3}})


def test_executor_rate_must_match_rtde():
    with pytest.raises(ValidationError, match="executor_rate_hz"):
        make_hw(control={"executor_rate_hz": 250.0})


def test_fz_contact_source_needs_calibration():
    with pytest.raises(ValidationError, match="calibrated"):
        make_hw(derived={"contact_source": "fz"})
    hw = make_hw(derived={"contact_source": "fz"},
                 tactile={"force_unit_to_N": 0.01, "dist_force_unit_to_N": 0.001})
    assert hw.tactile.force_calibrated


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        make_hw(tactile={"typo_field": 1})


def test_action_dim_tripwire():
    with pytest.raises(ValidationError):
        make_hw(control={"action_dim": 8})


def test_value_changes_are_fine(small_hw):
    """The user's core requirement: values change, ranks stay — must validate."""
    assert small_hw.tactile.field.hw == (48, 64)
    assert small_hw.cpk_shape == (12, 16)
    assert small_hw.tactile.field_ch == 8


def test_snapshot_and_hash_stable(default_hw):
    assert default_hw.config_hash() == default_hw.config_hash()
    hw2 = make_hw(tactile={"rate_hz": 60.0},
                  recording={"field_ds_rate_hz": 60.0, "infer_img_rate_hz": 30.0})
    assert hw2.config_hash() != default_hw.config_hash()
    # shape-relevant fields unchanged by a rate change
    assert hw2.shape_relevant_fields() == default_hw.shape_relevant_fields()


def test_paths_config_loads():
    from phantom.config.paths import load_paths
    p = load_paths()
    assert p.cosmos_repo.name == "cosmos-predict2.5"
    assert p.cosmos_checkpoint.suffix == ".pt"


def test_teleop_echo_section_loads(default_hw):
    """The repo yaml carries the Echo leader config (Denmark-rig facts)."""
    echo = default_hw.teleop.echo
    assert echo is not None
    assert (echo.vid, echo.pid, echo.baud) == (1603, 1868, 115200)
    assert len(echo.base_pose) == 6
    assert echo.sensitivity_divisors == (1.0, 1.25, 1.75)
    assert echo.gripper_open_tick != echo.gripper_closed_tick


def test_teleop_section_optional():
    """Older / partial configs without a teleop section keep validating."""
    raw_hw = make_hw()
    import yaml
    raw = yaml.safe_load(raw_hw.snapshot_yaml())
    del raw["teleop"]
    from phantom.config.hardware import HardwareConfig
    hw = HardwareConfig.model_validate(raw)
    assert hw.teleop is None


def test_tactile_network_fields_roundtrip(default_hw):
    """New network fields survive the spawn-worker snapshot_yaml path."""
    import yaml
    from phantom.config.hardware import HardwareConfig
    hw2 = HardwareConfig.model_validate(yaml.safe_load(default_hw.snapshot_yaml()))
    assert hw2 == default_hw
    left = hw2.tactile.sensors[0]
    assert left.dev_id == "L26050098"
    assert left.remote_addr == "192.168.127.10:50051"
    assert left.pc_port == 60001
    assert hw2.tactile.pc_host == "192.168.127.100"


def test_teleop_command_backward_compatible():
    """TeleopCommand still constructs without q_target (keyboard/spacemouse)."""
    import numpy as np
    from phantom.teleop.base import TeleopCommand
    cmd = TeleopCommand(dpose=np.zeros(6), gripper=0.5, buttons={})
    assert cmd.q_target is None
