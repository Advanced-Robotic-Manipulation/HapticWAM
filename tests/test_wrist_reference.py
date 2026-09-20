"""Opt-in reference behavior, legacy hashes and unchanged input/stop semantics."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from phantom_test_utils import make_small_hw
from pydantic import ValidationError

from phantom.config.hardware import HardwareConfig, load_hardware
from phantom.deploy.safety import (
    SafetyAction,
    SafetyMonitor,
    apply_wrench_baseline_mode,
)

ROOT = Path(__file__).resolve().parents[1]
POSE = np.array([0.0, -0.45, 0.25, 0.0, 3.14, 0.0])
BIAS = np.array([2.0, -15.0, 100.0, 4.0, -3.0, -4.0])


class Ring:
    def __init__(self):
        self.ts, self.data = np.zeros(0), {}

    def push(self, t, ft, pstop=False):
        self.ts = np.array([t])
        self.data = {"ft": np.asarray(ft)[None], "protective_stop": [pstop]}

    def latest(self, _):
        return self.ts, self.data


def monitor(mode="episode_fixed"):
    hw = apply_wrench_baseline_mode(
        make_small_hw(
            safety={
                "wrench_limit_N": 60,
                "wrench_limit_Nm": 15,
            }
        ),
        mode,
    )
    ring = Ring()
    return SafetyMonitor(hw, {"arm": ring}), ring


def check(mon, ring, t, ft=BIAS, *, sample_t=None, pstop=False):
    ring.push(t if sample_t is None else sample_t, ft, pstop)
    return mon.check(t, POSE)


def test_legacy_hashes_and_explicit_default_round_trip():
    for file, digest in (
        ("hardware.frozen.yaml", "6ab949e3fea83bd3"),
        ("hardware.nuc.frozen.yaml", "7ecb586f6666f291"),
    ):
        original = load_hardware(
            ROOT / "tests/fixtures/reference/wrist_baseline_20260907" / file, quiet=True
        )
        assert original.config_hash() == digest
        assert original.safety.wrench_baseline_mode == "rolling_calm"
        assert apply_wrench_baseline_mode(original, None) is original
        explicit = apply_wrench_baseline_mode(original, "rolling_calm")
        assert explicit.snapshot_yaml() == original.snapshot_yaml()
        fixed = apply_wrench_baseline_mode(original, "episode_fixed")
        assert fixed.config_hash() != digest
        assert (
            yaml.safe_load(fixed.snapshot_yaml())["safety"]["wrench_baseline_mode"]
            == "episode_fixed"
        )
        assert (
            HardwareConfig.model_validate(yaml.safe_load(fixed.snapshot_yaml()))
            == fixed
        )
        before, after = original.model_dump(), fixed.model_dump()
        after["safety"]["wrench_baseline_mode"] = "rolling_calm"
        assert before == after  # shapes, thresholds and timing all unchanged
    with pytest.raises(ValidationError):
        apply_wrench_baseline_mode(original, "fixed_typo")


def test_current_nuc_guards_and_opt_in_hold_snapshot_remain_distinct():
    current = load_hardware(ROOT / "configs/hardware.nuc.yaml", quiet=True)
    assert current.safety.wrench_limit_N == 45
    assert current.safety.wrench_debounce_ticks / current.control.action_rate_hz == 0.1
    raw = yaml.safe_load(current.snapshot_yaml())
    assert "servo_constraint_hold_s" not in raw["safety"]
    assert HardwareConfig.model_validate(raw) == current
    raw["safety"]["servo_constraint_hold_s"] = None
    assert HardwareConfig.model_validate(raw).config_hash() == current.config_hash()
    raw["safety"]["servo_constraint_hold_s"] = 2.5
    enabled = HardwareConfig.model_validate(raw)
    assert enabled.config_hash() != current.config_hash()
    assert yaml.safe_load(enabled.snapshot_yaml())["safety"]["servo_constraint_hold_s"] == 2.5
    assert HardwareConfig.model_validate(yaml.safe_load(enabled.snapshot_yaml())) == enabled


def test_native_cli_respects_yaml_until_explicit_override():
    # Parse the real parser without importing Torch or opening any drivers.
    tree = ast.parse((ROOT / "phantom/scripts/run_deploy.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_parser"
    )
    planner = ast.parse((ROOT / "phantom/deploy/planner.py").read_text())
    modes = next(
        n
        for n in planner.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SYSTEM_MODES" for t in n.targets)
    )
    default = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_MAX_REPLANS" for t in n.targets
        )
    )
    policy_src = ast.parse((ROOT / "phantom/inference/policy.py").read_text())
    null_modes = next(
        n
        for n in policy_src.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "NULL_IMAGINATION_MODES"
            for t in n.targets
        )
    )
    ns = {
        "argparse": argparse,
        "Path": Path,
        "SYSTEM_MODES": ast.literal_eval(modes.value),
        "DEFAULT_MAX_REPLANS": ast.literal_eval(default.value),
        "NULL_IMAGINATION_MODES": ast.literal_eval(null_modes.value),
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "run_deploy.py", "exec"), ns)  # noqa: S102 -- trusted local parser, no model import
    parser = ns["build_parser"]()
    base = ["--system", "teacher", "--task", "waffles"]
    assert parser.parse_args(base).wrench_baseline_mode is None
    assert (
        parser.parse_args(
            base + ["--wrench-baseline-mode", "episode_fixed"]
        ).wrench_baseline_mode
        == "episode_fixed"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--wrench-baseline-mode", "unknown"])


def test_fixed_preserves_large_initial_bias_and_reference_copy():
    mon, ring = monitor()
    for t in np.arange(0, 10, 0.008):
        assert check(mon, ring, t).action == SafetyAction.OK
    report = mon.wrench_diagnostics()
    assert report["capture_t_s"] == 0
    assert report["force_deviation_n"] == report["torque_deviation_nm"] == 0
    assert not report["baseline_adapted"]
    report["initial_reference"][0] = 999
    report["reference_after"][0] = 999
    np.testing.assert_array_equal(mon._wrench_base, BIAS)
    np.testing.assert_array_equal(mon.wrench_diagnostics()["initial_reference"], BIAS)


def test_slow_external_load_cannot_enter_fixed_reference():
    fixed, f_ring = monitor()
    rolling, r_ring = monitor("rolling_calm")
    fixed_stops, rolling_stops = [], []
    for t in np.arange(0, 12, 0.008):
        force = BIAS + np.array([0, 0, min(100, 10 * t), 0, 0, 0])
        if check(fixed, f_ring, t, force).action == SafetyAction.STOP_EPISODE:
            fixed_stops.append(t)
        if check(rolling, r_ring, t, force).action == SafetyAction.STOP_EPISODE:
            rolling_stops.append(t)
    assert 6.3 < fixed_stops[0] < 6.32
    assert not rolling_stops
    assert rolling._wrench_base[2] - BIAS[2] > 90
    np.testing.assert_array_equal(fixed._wrench_base, BIAS)
    # Counterfactual continuation after a fixed stop is a guard replay only.
    for t in np.arange(12, 12.4, 0.008):
        f = check(fixed, f_ring, t)
        r = check(rolling, r_ring, t)
    assert f.action == SafetyAction.OK
    assert r.action == SafetyAction.STOP_EPISODE


def test_signed_torque_and_transient_debounce_reset():
    mon, ring = monitor()
    assert check(mon, ring, 0).action == SafetyAction.OK
    high = BIAS + [0, 0, 0, -16, 0, 0]
    assert check(mon, ring, 0.008, high).action == SafetyAction.OK
    assert check(mon, ring, 0.2).action == SafetyAction.OK
    assert check(mon, ring, 0.3, high).action == SafetyAction.OK
    assert check(mon, ring, 0.592, high).action == SafetyAction.OK
    verdict = check(mon, ring, 0.608, high)
    assert [e.kind for e in verdict.events] == ["wrench_limit"]
    assert mon.wrench_diagnostics()["torque_deviation_nm"] == 16
    assert mon.wrench_diagnostics()["over_since_s"] == 0.3


def test_recovery_and_event_auto_clear_never_retare(monkeypatch):
    mon, ring = monitor()
    check(mon, ring, 0)
    high = BIAS + [70, 0, 0, 0, 0, 0]
    check(mon, ring, 0.01, high)
    assert check(mon, ring, 0.32, high).action == SafetyAction.STOP_EPISODE
    monkeypatch.setattr("phantom.deploy.safety.time.perf_counter", lambda: 0.4)
    ring.push(0.4, BIAS + [49, 0, 0, 0, 0, 0])
    assert not mon.recovered()
    ring.push(0.4, BIAS + [47, 0, 0, 0, 0, 0])
    assert mon.recovered()
    assert check(mon, ring, 0.408).action == SafetyAction.OK
    assert "wrench_limit" not in mon._active
    assert mon.wrench_diagnostics()["capture_t_s"] == 0
    np.testing.assert_array_equal(mon._wrench_base, BIAS)
    fresh, fresh_ring = monitor()
    check(fresh, fresh_ring, 0.408, high)
    np.testing.assert_array_equal(fresh._wrench_base, high)  # only NEW episode re-tares


@pytest.mark.parametrize(
    "value,stamp", [(np.full(6, np.nan), 0), (np.ones(5), 0), (BIAS, -1), (BIAS, 1)]
)
def test_invalid_initial_reference_stops_without_capture(value, stamp, monkeypatch):
    mon, ring = monitor()
    verdict = check(mon, ring, 0, value, sample_t=stamp)
    assert verdict.action == SafetyAction.STOP_EPISODE
    assert "wrench_invalid" in [e.kind for e in verdict.events]
    assert mon._wrench_base is None
    assert not mon.wrench_diagnostics()["valid"]
    monkeypatch.setattr("phantom.deploy.safety.time.perf_counter", lambda: 0)
    assert not mon.recovered()


def test_fixed_missing_then_protective_stop_wins_and_valid_receive_race():
    mon, ring = monitor()
    assert mon.check(0, POSE).action == SafetyAction.STOP_EPISODE
    assert mon._wrench_base is None
    assert (
        check(mon, ring, 0, np.full(6, np.nan), pstop=True).action
        == SafetyAction.PROTECTIVE_STOP
    )
    # Real receive thread may publish after the executor's clock read.
    assert check(mon, ring, 0, sample_t=0.00001).action == SafetyAction.OK
    assert mon.wrench_diagnostics()["capture_t_s"] == 0.00001


def test_sim_finish_latch_clear_and_teacher_input_preserve_reference():
    from test_sim_placement_release import INSIDE, arm, plan, setup, tick
    from test_sim_release_finish import FINISH

    from phantom.deploy.release_controller import (
        PlacementReleaseConfig,
        PlacementReleaseController,
    )

    ad = setup()
    ad.hw = apply_wrench_baseline_mode(ad.hw, "episode_fixed")
    ad.safety = SafetyMonitor(ad.hw, ad.rings)
    ad.release_controller = PlacementReleaseController(PlacementReleaseConfig(**FINISH))
    arm(ad)
    original = ad.safety.wrench_diagnostics()
    assert ad.submit(plan(0.008, 0.4), 0.008)
    for i in range(2, 28):
        tick(ad, i * 0.008)
    for i in range(28, 160):
        command = tick(ad, i * 0.008, load=0, measured=0.3)
        if ad.completed_reason:
            break
    assert ad.completed_reason == "placement_release_finished"
    assert (
        command.diagnostics["wrist_guard"]["initial_reference"]
        == original["initial_reference"]
    )
    ad.clear_grip_latch()
    assert ad.safety.wrench_diagnostics()["capture_t_s"] == original["capture_t_s"]
    # Raw model windows still carry the actual wrist, never the guard residual.
    t = command.t + 0.008
    ad.observe(
        t,
        rgb=np.zeros((12, 16, 3), np.uint8),
        q=np.zeros(6),
        qd=np.zeros(6),
        tcp_pose=INSIDE,
        tcp_speed=np.zeros(6),
        gripper_state=np.array([0.3, 0]),
        wrist_ft=BIAS,
    )
    raw_before = ad.rings["arm"].latest(1)[1]["ft"].copy()
    window_before = ad.snapshot(t).wrist_window.copy()
    ad.safety.check(t, INSIDE)
    np.testing.assert_array_equal(ad.rings["arm"].latest(1)[1]["ft"], raw_before)
    np.testing.assert_array_equal(ad.snapshot(t).wrist_window, window_before)
    np.testing.assert_array_equal(window_before[-1], BIAS.astype(np.float32))
    assert (
        ad.safety.wrench_diagnostics()["initial_reference"]
        == original["initial_reference"]
    )


def test_native_fixed_wrench_stop_preempts_finish_and_preserves_halt_decision(
    monkeypatch,
):
    from test_sim_placement_release import INSIDE, observe
    from test_sim_release_finish import native

    from phantom.deploy import executor as executor_module

    ex, ad = native()
    ex.hw = ad.hw = apply_wrench_baseline_mode(ad.hw, "episode_fixed")
    ex.safety = ad.safety = SafetyMonitor(ex.hw, ad.rings)
    observe(ad, 0, load=0, measured=0.3)
    assert ex.safety.check(0, INSIDE).action == SafetyAction.OK
    high = np.array([100, 0, 0, 0, 0, 0])
    ex.safety._check_wrench(0.1, 0.1, high)
    observe(ad, 0.404, load=0, measured=0.3)
    ad.rings["arm"].rows[-1][1]["ft"] = high.copy()
    ex.completed_reason = "placement_release_finished"
    ex.completed_at_s = 0.05
    ex._finish_pose, ex._finish_grip = INSIDE.copy(), 0.4
    released, servoed = [], []
    ex.gripper.move = lambda grip, *_: released.append(grip)
    ex.arm.stop = lambda *_: None
    ex.arm.servo_l = lambda *args: servoed.append(args)
    ex.arm.get_state = lambda: SimpleNamespace(
        q=np.zeros(6), qd=np.zeros(6), tcp_pose=INSIDE
    )
    monkeypatch.setattr(executor_module.time, "perf_counter", lambda: 0.404)
    ex._run()
    assert ex.stopped_reason == "safety_stop"
    assert ex.completed_reason == "placement_release_finished"
    assert not servoed and released == [ex.open_aperture]
    assert ex.halt_state["wrist_guard"]["force_deviation_n"] == 100
    assert ex.halt_state["wrist_guard"]["capture_t_s"] == 0
    np.testing.assert_array_equal(ex.safety._wrench_base, np.zeros(6))


def test_saved_6273_wrist_trace_matches_frozen_default_and_fixed_counterfactual():
    import importlib.util

    path = ROOT / "tests/fixtures/reference/wrist_baseline_20260907/replay_6273.py"
    spec = importlib.util.spec_from_file_location("replay_6273", path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    result = replay.audit()
    assert result["default_parity"]["bitwise_numeric_equal_all_2247_rows"]
    rolling, fixed = (
        result["modes"][mode] for mode in ("rolling_calm", "episode_fixed")
    )
    assert rolling["first_debounced_stop"]["t_s"] == 17.972
    assert fixed["first_debounced_stop"]["t_s"] == 10.460
    assert rolling["at_recorded_stop"]["force_deviation_n"] == pytest.approx(
        101.53942679740246
    )
    assert fixed["at_recorded_stop"]["force_deviation_n"] == pytest.approx(
        0.09362201032470081
    )
    assert fixed["reference_force_drift_max_n"] == 0
