"""An optional hold budget must be declared, pinned and scored explicitly."""

from argparse import Namespace
import copy

import pytest

from tests.test_sim_campaign_servo_limiter import enabled_runtime, reasons
from tools.sim.analyze_policy_campaign import servo_reach_limiter_metadata
from tools.sim.run_policy_campaign import simulation_command, source_manifest
from tools.sim.run_waffles import arguments


def hold_runtime():
    values = enabled_runtime()
    design, info = values[1], values[4]
    design["adapter_profile"]["servo_constraint_hold_s"] = 2.5
    design["runtime_hardware"]["effective_model"]["safety"]["servo_constraint_hold_s"] = 2.5
    info["hardware_effective"] = copy.deepcopy(design["runtime_hardware"]["effective_model"])
    info["servo_reach_limiter"] = servo_reach_limiter_metadata(design)
    return values


def test_timeout_is_an_explicit_valid_controller_outcome():
    values = hold_runtime()
    assert reasons(values, stop="servo_constraint_hold_timeout") == []
    values[4]["servo_reach_limiter"]["constraint_hold_s"] = 10.0
    assert "effective_servo_reach_limiter_differs_from_campaign" in reasons(values)


@pytest.mark.parametrize("budget", [True, 0, -1, float("nan"), float("inf"), "2.5"])
def test_bad_deadlines_cannot_be_frozen(budget):
    values = hold_runtime()
    values[1]["adapter_profile"]["servo_constraint_hold_s"] = budget
    values[1]["runtime_hardware"]["effective_model"]["safety"]["servo_constraint_hold_s"] = budget
    with pytest.raises(ValueError, match="finite and positive"):
        servo_reach_limiter_metadata(values[1])


def test_hold_requires_matching_enabled_hardware():
    values = hold_runtime()
    design = values[1]
    design["runtime_hardware"]["effective_model"]["safety"]["servo_constraint_hold_s"] = 3.0
    with pytest.raises(ValueError, match="declared hardware"):
        servo_reach_limiter_metadata(design)
    design["runtime_hardware"]["effective_model"]["safety"]["servo_constraint_hold_s"] = 2.5
    design["adapter_profile"]["servo_reach_limiter"] = False
    with pytest.raises(ValueError, match="requires"):
        servo_reach_limiter_metadata(design)


@pytest.mark.parametrize("enabled", [True, False])
def test_hardware_cannot_enable_hold_without_profile(enabled):
    values = hold_runtime()
    values[1]["adapter_profile"].pop("servo_constraint_hold_s")
    values[1]["adapter_profile"]["servo_reach_limiter"] = enabled
    with pytest.raises(ValueError, match="declared hardware"):
        servo_reach_limiter_metadata(values[1])


def test_launcher_forwards_only_the_declared_deadline(tmp_path):
    _, design, policy, condition, *_ = hold_runtime()
    args = Namespace(source=tmp_path, evidence=tmp_path/"evidence", output=tmp_path,
                     port=7799, hardware_config=tmp_path/"hardware.yaml")
    command = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    assert command[-3:] == ["--servo-reach-limiter", "--servo-constraint-hold-s", "2.5"]


def test_hold_helper_is_a_required_frozen_input(tmp_path, monkeypatch):
    _, design, *_ = hold_runtime()
    args = Namespace(source=tmp_path/"source", live_repo=tmp_path/"live",
                     evidence=tmp_path/"evidence", hardware_config=tmp_path/"hardware.yaml",
                     robot_usd=None)
    monkeypatch.setattr("tools.sim.run_policy_campaign.fingerprint", lambda p: str(p))
    helper = args.source/"phantom/drivers/servo_limiter.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# fixture\n")
    with pytest.raises(FileNotFoundError, match="constraint-hold source"):
        source_manifest(args, "design", design)
    (helper.parent/"servo_hold.py").write_text("# fixture\n")
    frozen = source_manifest(args, "design", design)
    assert "phantom/drivers/servo_hold.py" in frozen["source_sha256"]


@pytest.mark.parametrize("extra,limiter,hold", [
    # replay/dynamics modes: no limiter, no hold
    ([], False, None),
    # policy mode: bounded reach + 2.5 s verified hold by default (09-11, rig parity)
    (["--mode", "policy"], True, 2.5),
    (["--mode", "policy", "--servo-reach-limiter"], True, 2.5),
    (["--mode", "policy", "--servo-constraint-hold-s", "0"], True, None),
    (["--mode", "policy", "--no-servo-reach-limiter"], False, None),
])
def test_cli_limiter_and_hold_defaults(monkeypatch, extra, limiter, hold):
    monkeypatch.setattr("sys.argv", ["run_waffles", "--episode", "unused", "--output", "unused", *extra])
    args = arguments()
    assert args.servo_reach_limiter is limiter
    assert args.servo_constraint_hold_s == hold


def test_cli_rejects_contradictory_limiter_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_waffles", "--episode", "unused", "--output", "unused",
                                   "--mode", "policy", "--servo-reach-limiter", "--no-servo-reach-limiter"])
    with pytest.raises(SystemExit):
        arguments()


def test_cli_refuses_a_hold_without_policy_limiter(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_waffles", "--episode", "unused", "--output", "unused",
                                   "--servo-constraint-hold-s", "2.5"])
    with pytest.raises(SystemExit):
        arguments()


@pytest.mark.parametrize("declared_hold", [None, 2.5])
def test_launcher_is_explicit_with_a_default_limiter_runner(tmp_path, declared_hold):
    """A runner that defaults to the limiter (09-11) gets the design's choice explicitly;
    a source without that runner keeps the exact historical CLI."""
    _, design, policy, condition, *_ = hold_runtime()
    if declared_hold is None:
        design["adapter_profile"].pop("servo_constraint_hold_s")
        design["runtime_hardware"]["effective_model"]["safety"]["servo_constraint_hold_s"] = None
    args = Namespace(source=tmp_path, evidence=tmp_path/"evidence", output=tmp_path,
                     port=7799, hardware_config=tmp_path/"hardware.yaml")
    old_on = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    runner = tmp_path / "tools/sim/run_waffles.py"
    runner.parent.mkdir(parents=True)
    runner.write_text('p.add_argument("--no-servo-reach-limiter", action="store_true")\n')
    new_on = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    if declared_hold is None:
        assert old_on[-1] == "--servo-reach-limiter"
        assert new_on == old_on + ["--servo-constraint-hold-s", "0"]
    else:
        assert new_on == old_on and new_on[-3:] == ["--servo-reach-limiter", "--servo-constraint-hold-s", "2.5"]
    design["adapter_profile"]["servo_reach_limiter"] = False
    design["adapter_profile"].pop("servo_constraint_hold_s", None)
    design["runtime_hardware"]["effective_model"]["safety"]["servo_constraint_hold_s"] = None
    new_off = simulation_command(args, design, policy, condition, 1, tmp_path, None)
    assert new_off[-1] == "--no-servo-reach-limiter"
    runner.unlink()
    assert "--no-servo-reach-limiter" not in simulation_command(args, design, policy, condition, 1, tmp_path, None)

