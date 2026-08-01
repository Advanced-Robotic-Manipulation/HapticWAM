"""phantom.data_collect.config — data_collect.yaml loading + hardware ports patching."""

import pytest
import yaml
from pydantic import ValidationError

from phantom.data_collect.config import (DEFAULT_COLLECT_YAML, REPO_ROOT, CollectConfig,
                            load_collect)


def test_repo_data_collect_yaml_loads():
    cc, hw = load_collect(DEFAULT_COLLECT_YAML)
    # ports patched into the hardware config
    assert [s.name for s in hw.tactile.sensors] == ["left", "right"]
    assert hw.tactile.sensors[0].dev_id == cc.ports.dmtac_left
    assert hw.tactile.sensors[1].dev_id == cc.ports.dmtac_right
    assert hw.cameras.scene.serial == cc.ports.realsense_serial
    # echo tick calibration + filter tuning patched
    assert hw.teleop.echo.gripper_open_tick == cc.gripper.open_tick
    assert hw.teleop.echo.gripper_closed_tick == cc.gripper.closed_tick
    assert hw.teleop.echo.gripper_ema_alpha == cc.gripper.ema_alpha
    assert hw.teleop.echo.filter_min_cutoff == cc.teleop.filter_min_cutoff_hz
    assert hw.teleop.echo.filter_beta == cc.teleop.filter_beta
    assert hw.teleop.echo.filter_d_cutoff == cc.teleop.filter_d_cutoff_hz
    assert hw.teleop.echo.vid == cc.ports.echo_vid
    assert cc.storage.require_separate_device     # rig yaml: unmounted-drive guard on


def test_unknown_key_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT_COLLECT_YAML.read_text(encoding="utf-8"))
    raw["typo_section"] = {"x": 1}
    p = tmp_path / "data_collect.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_collect(p)


def test_local_overlay_merges(tmp_path):
    raw = yaml.safe_load(DEFAULT_COLLECT_YAML.read_text(encoding="utf-8"))
    raw["hardware"] = str(REPO_ROOT / raw["hardware"])   # keep resolvable from tmp
    p = tmp_path / "data_collect.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    (tmp_path / "data_collect.local.yaml").write_text(
        yaml.safe_dump({"ports": {"dmtac_left": "XLOCAL1"},
                        "storage": {"external_drive": "Z:/elsewhere"}}),
        encoding="utf-8")
    cc, hw = load_collect(p)
    assert cc.ports.dmtac_left == "XLOCAL1"
    assert hw.tactile.sensors[0].dev_id == "XLOCAL1"
    assert cc.storage.external_drive == "Z:/elsewhere"
    assert cc.ports.dmtac_right == raw["ports"]["dmtac_right"]  # untouched


def test_missing_teleop_section_is_loud(tmp_path):
    hw_raw = yaml.safe_load(
        (REPO_ROOT / "configs" / "hardware.yaml").read_text(encoding="utf-8"))
    hw_raw.pop("teleop")
    hw_p = tmp_path / "hw.yaml"
    hw_p.write_text(yaml.safe_dump(hw_raw), encoding="utf-8")
    raw = yaml.safe_load(DEFAULT_COLLECT_YAML.read_text(encoding="utf-8"))
    raw["hardware"] = str(hw_p)
    p = tmp_path / "data_collect.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="teleop.echo"):
        load_collect(p)


def test_defaults_sane():
    cc = CollectConfig.model_validate({
        "ports": {"dmtac_left": 0, "dmtac_right": 1},
        "storage": {"external_drive": "D:/x"}})
    assert cc.default_mode == "full"
    assert cc.safeguard.enabled and cc.safeguard.force_limit_n < 30.0
    assert cc.teleop.v_max_rad_s > 1.0          # the reference's 1.0 cap was a bug
    # one-euro must open up with motion (beta > the near-fixed 0.5) but NOT so
    # aggressively that it passes the exo's rest jitter as command buzz — beta=10
    # did that and drove the CB3 to C153A3. Keep it moderate.
    assert 1.0 <= cc.teleop.filter_beta <= 5.0
    # TRACK-path sqrt-braking accel bound: high enough for low lag, under the
    # ~88 rad/s^2 C153A3 ceiling (incl. the one-tick 2*a_max landing transient)
    assert 30.0 <= cc.teleop.track_a_max_rad_s2 <= 44.0
    # drop-out bridging must cover the ~100 ms Echo stalls with margin
    assert cc.teleop.extrap_cap_s >= 0.1
    assert cc.teleop.stale_reengage_s > cc.teleop.extrap_cap_s
    assert cc.gripper.speed == 1.0
