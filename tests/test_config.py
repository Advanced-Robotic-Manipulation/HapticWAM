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
