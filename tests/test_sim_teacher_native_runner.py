"""Frozen-input, physical-failure and resource gates for teacher development."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "native_teacher_runner", REPO / "tests/fixtures/reference/teacher_native_v11_20260909/run_trials.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def guarded_run(tmp_path):
    # The saved schema is copied from an actual guarded native execution;
    # values here isolate the orchestration checks from physical simulation.
    monitor = {
        "enabled": True, "checks": 10001, "last_phase": "execution", "last_t_s": 9.999,
        "observed_maxima": {
            "coupling_max_abs_rad": .00014,
            "joint_limit_violation_max_rad": .000003,
            "loop_closure_max_m": .0000041,
        },
        "last_diagnostic": {
            "passed": True, "joint_state_finite": True,
            "gates": {"joint_coupling": True, "joint_limits": True, "adaptive_loop_closure": True},
            "thresholds": {
                "coupling_max_abs_rad": .005,
                "joint_limit_violation_max_rad": .002,
                "loop_closure_max_m": .001,
            },
        },
    }
    run = {"duration_s": 9.999, "policy_stop_reason": "wrench_limit",
           "native_mechanics_monitor": monitor}
    (tmp_path / "native_mechanics_monitor.json").write_text(json.dumps(monitor))
    return tmp_path, run


def save_monitor(directory, run):
    (directory / "native_mechanics_monitor.json").write_text(
        json.dumps(run["native_mechanics_monitor"])
    )


def test_physical_guard_stop_does_not_make_valid_mechanics_invalid(guarded_run):
    assert runner.validate_native_monitor(*guarded_run) is None


@pytest.mark.parametrize("mutation", [
    "disabled", "no_checks", "bool_checks", "no_execution", "failed_last_gate",
    "nonfinite_joint", "changed_threshold", "excess_loop", "nan_loop",
    "missing_maximum", "missing_end", "short_monitor",
])
def test_invalid_native_monitor_stops_new_trials(guarded_run, mutation):
    directory, run = guarded_run
    monitor = run["native_mechanics_monitor"]
    if mutation == "disabled":
        monitor["enabled"] = False
    elif mutation == "no_checks":
        monitor["checks"] = 0
    elif mutation == "bool_checks":
        monitor["checks"] = True
    elif mutation == "no_execution":
        monitor["last_phase"] = "initialization_settling"
    elif mutation == "failed_last_gate":
        monitor["last_diagnostic"]["gates"]["adaptive_loop_closure"] = False
    elif mutation == "nonfinite_joint":
        monitor["last_diagnostic"]["joint_state_finite"] = False
    elif mutation == "changed_threshold":
        monitor["last_diagnostic"]["thresholds"]["loop_closure_max_m"] = .01
    elif mutation == "excess_loop":
        monitor["observed_maxima"]["loop_closure_max_m"] = .00101
    elif mutation == "nan_loop":
        monitor["observed_maxima"]["loop_closure_max_m"] = float("nan")
    elif mutation == "missing_maximum":
        del monitor["observed_maxima"]["loop_closure_max_m"]
    elif mutation == "missing_end":
        del monitor["last_t_s"]
    elif mutation == "short_monitor":
        monitor["last_t_s"] -= .01
    save_monitor(directory, run)
    with pytest.raises(RuntimeError, match="Native"):
        runner.validate_native_monitor(directory, run)


def test_failure_marker_cannot_be_hidden_by_later_good_monitor(guarded_run):
    directory, run = guarded_run
    (directory / "native_mechanics_failure.json").write_text('{"passed":false}')
    with pytest.raises(RuntimeError, match="failure marker"):
        runner.validate_native_monitor(directory, run)


def test_monitor_file_cannot_disagree_with_run_metadata(guarded_run):
    directory, run = guarded_run
    run = copy.deepcopy(run)
    run["native_mechanics_monitor"]["checks"] += 1
    with pytest.raises(RuntimeError, match="disagrees"):
        runner.validate_native_monitor(directory, run)


@pytest.mark.parametrize("free,passes", [
    (runner.MIN_FREE_BYTES - 1, False), (runner.MIN_FREE_BYTES, True),
])
def test_space_gate_never_deletes_existing_evidence(tmp_path, monkeypatch, free, passes):
    evidence = tmp_path / "preserved.npz"
    evidence.write_bytes(b"original physical failure")
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    if passes:
        assert runner.require_free_space(tmp_path) == free
    else:
        with pytest.raises(RuntimeError, match="10 GiB"):
            runner.require_free_space(tmp_path)
    assert evidence.read_bytes() == b"original physical failure"


@pytest.mark.parametrize("status,valid", [("scored", True), ("invalid", False), ("scored", False)])
def test_only_technical_validity_controls_continuation(tmp_path, status, valid):
    path = tmp_path / "analysis/trials/teacher__start__seed1.json"
    path.parent.mkdir(parents=True)
    record = {"status": status, "metrics": {
        "valid_for_scoring": valid, "outcomes": {"full_task": False, "lift": False},
        "invalid_reasons": [] if valid else ["effective_policy_delivery_clock_differs_from_campaign"],
    }}
    path.write_text(json.dumps(record))
    if valid and status == "scored":
        assert runner.validate_scored_trial(tmp_path, path.stem) == record
    else:
        with pytest.raises(RuntimeError, match="Preserved technically invalid"):
            runner.validate_scored_trial(tmp_path, path.stem)


def test_hardware_hash_and_disabled_lift_autostop_are_both_required(tmp_path):
    args = SimpleNamespace(hardware_config=tmp_path / "hardware.yaml")
    args.hardware_config.write_text("safety:\n  lift_complete_z_m: 0\n")
    design = {"runtime_hardware": {"sha256": runner.campaign.fingerprint(args.hardware_config)}}
    assert runner.validate_hardware(args, design) is None
    args.hardware_config.write_text("safety:\n  lift_complete_z_m: 0.32\n")
    with pytest.raises(ValueError, match="differs from the frozen"):
        runner.validate_hardware(args, design)
    design["runtime_hardware"]["sha256"] = runner.campaign.fingerprint(args.hardware_config)
    with pytest.raises(ValueError, match="disable lift-complete"):
        runner.validate_hardware(args, design)


def test_protocol_ready_and_runner_are_bound_beyond_base_source_manifest(tmp_path, monkeypatch):
    args = SimpleNamespace(server_dir=tmp_path, protocol=tmp_path / "protocol.json")
    ready = tmp_path / "ready.json"
    ready.write_text('{"pid":123}')
    args.protocol.write_text('{"seed":1}')
    monkeypatch.setattr(runner.campaign, "source_manifest", lambda *_: {"base": "unchanged"})
    before = runner.experiment_manifest(args, "digest", {})
    assert before["controller_sha256"] == runner.campaign.fingerprint(Path(runner.__file__))
    args.protocol.write_text('{"seed":2}')
    after = runner.experiment_manifest(args, "digest", {})
    assert after["protocol_sha256"] != before["protocol_sha256"]
    assert after["base"] == before["base"]
    ready.write_text('{"pid":124}')
    changed = runner.experiment_manifest(args, "digest", {})
    assert changed["server_ready_sha256"] != after["server_ready_sha256"]
