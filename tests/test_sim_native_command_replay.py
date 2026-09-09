"""Native mechanics playback retains submitted targets, units and source clocks."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.sim.command_replay import (
    NativeFingerHoldReplay,
    NativeRecordedDriveCommands,
    native_drive_manifest,
)
from phantom.sim.gripper_adaptive import drive_targets, joint_targets
from tools.sim.run_waffles import load_command_replay

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/sim/waffles_w2l_adaptive_parallel_dt1ms_20260909.json").read_text())


def command(t, closure, cfg, *, q=0.3):
    return {
        "t": t, "target_q": [q] * 6,
        "target_finger_q": drive_targets(closure, cfg).tolist(),
        "gripper_command": closure, "status": "drive_submitted",
        "measured_q": [99] * 6, "requested_tcp": [88] * 6,
        "measured_finger_q": joint_targets(closure, cfg).tolist(),
    }


def files(tmp_path, cfg, rows, *, declaration=None):
    trace = tmp_path / "execution_trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest = tmp_path / "native_commands.json"
    manifest.write_text(json.dumps(native_drive_manifest(trace, cfg) if declaration is None else declaration))
    return trace, manifest


def test_native_targets_and_timestamps_survive_both_physics_rates(tmp_path):
    cfg = config()
    rows = [command(0.001, 0.4525251053273678, cfg), command(0.009000000000000001, 0.4242098033428192, cfg, q=-0.4)]
    trace, manifest = files(tmp_path, cfg, rows)
    original = trace.read_bytes()
    for dt in (0.001, 0.0005):
        variant = deepcopy(cfg)
        variant["physics"]["dt"] = dt
        replay = NativeRecordedDriveCommands(trace, manifest_path=manifest, cfg=variant)
        assert replay.at(0) is None
        assert replay.at(0.0005) is None
        np.testing.assert_array_equal(replay.t, [row["t"] for row in rows])
        for t in np.arange(0.001, 0.016, dt):
            row = rows[0 if t + 1e-10 < 0.009 else 1]
            actual_q, actual_finger_refs = replay.at(t)
            np.testing.assert_array_equal(actual_q, row["target_q"])
            np.testing.assert_array_equal(actual_finger_refs, row["target_finger_q"])
        np.testing.assert_array_equal(replay.at(8)[0], rows[-1]["target_q"])
        assert replay.at(8)[1][1] == 2.62  # Spring rest, not measured joint angle.
        replay.at(8)[0][:] = 0
        np.testing.assert_array_equal(replay.at(8)[0], rows[-1]["target_q"])
    assert trace.read_bytes() == original


@pytest.mark.parametrize("bad", ["hash", "arm_order", "finger_order", "units", "semantics", "config", "missing"])
def test_native_manifest_refuses_unpinned_or_mislabelled_commands(tmp_path, bad):
    cfg = config()
    trace, manifest = files(tmp_path, cfg, [command(0.001, 0.4, cfg)])
    declaration = json.loads(manifest.read_text())
    if bad == "hash":
        declaration["trace_sha256"] = "0" * 64
    elif bad == "arm_order":
        declaration["arm_joint_names"].reverse()
    elif bad == "finger_order":
        declaration["finger_joint_names"].reverse()
    elif bad == "units":
        declaration["finger_target_units"] = "m"
    elif bad == "semantics":
        declaration["finger_target_semantics"] = "measured_joint_positions"
    elif bad == "config":
        cfg["gripper"]["articulation"]["native_passive"]["spring_reference_rad"] += 0.1
    else:
        declaration.pop("finger_joint_names")
    manifest.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="manifest"):
        NativeRecordedDriveCommands(trace, manifest_path=manifest, cfg=cfg)


@pytest.mark.parametrize("bad", ["legacy_meters", "measured_positions", "changed_spring", "changed_motor", "missing_targets", "proposal", "duplicate_time", "nonfinite", "empty"])
def test_even_hash_pinned_trace_must_contain_valid_submitted_drive_references(tmp_path, bad):
    cfg = config()
    row = command(0.001, 0.4, cfg)
    rows = [row]
    if bad == "legacy_meters":
        row["target_finger_q"] = [0.03, 0.03]
    elif bad == "measured_positions":
        row["target_finger_q"] = row["measured_finger_q"]
    elif bad == "changed_spring":
        row["target_finger_q"][1] = 0.4
    elif bad == "changed_motor":
        row["target_finger_q"][0] += 0.01
    elif bad == "missing_targets":
        del row["target_q"]
    elif bad == "proposal":
        row["status"] = "proposal"
    elif bad == "duplicate_time":
        rows.append(deepcopy(row))
    elif bad == "nonfinite":
        row["target_q"][0] = float("nan")
    else:
        rows = []
    trace, manifest = files(tmp_path, cfg, rows)
    with pytest.raises(ValueError):
        NativeRecordedDriveCommands(trace, manifest_path=manifest, cfg=cfg)


def test_runner_requires_manifest_for_native_and_preserves_legacy_format(tmp_path):
    cfg = config()
    trace, manifest = files(tmp_path, cfg, [command(0.001, 0.4, cfg)])
    args = SimpleNamespace(command_trace=trace, policy_initial_state=tmp_path / "initial.json", command_trace_manifest=None)
    with pytest.raises(ValueError, match="command-trace-manifest"):
        load_command_replay(args, cfg)
    args.command_trace_manifest = manifest
    assert isinstance(load_command_replay(args, cfg), NativeRecordedDriveCommands)
    legacy = {"gripper": {"stroke": 0.085}}
    with pytest.raises(ValueError, match="legacy prismatic"):
        load_command_replay(args, legacy)
    args.command_trace_manifest = None
    trace.write_text(json.dumps({"t": 0.001, "status": "drive_submitted", "target_q": [0] * 6, "target_finger_q": [0.03] * 2}) + "\n")
    replay = load_command_replay(args, legacy)
    np.testing.assert_array_equal(replay.at(0.001)[1], [0.03, 0.03])


def test_finger_hold_overlay_is_explicit_causal_and_preserves_arm_stream_and_source(tmp_path):
    cfg = config()
    rows = [command(.0005, .1, cfg), command(.0085, .61, cfg, q=.5),
            command(.0165, .232, cfg, q=.6), command(.0245, .2, cfg, q=.7)]
    trace, manifest = files(tmp_path, cfg, rows)
    original = trace.read_bytes()
    source = NativeRecordedDriveCommands(trace, manifest_path=manifest, cfg=cfg)
    replay = NativeFingerHoldReplay(source, hold_at_s=.0165)
    assert replay.at(0) is None
    for t in [0., .0005, .008, .0085, .016, .0165, .024, .0245, 2.0165]:
        actual, expected = replay.at(t), source.at(t)
        if expected is None:
            assert actual is None
            continue
        np.testing.assert_array_equal(actual[0], expected[0])
        target = expected[1] if t < .0165 else rows[1]["target_finger_q"]
        np.testing.assert_array_equal(actual[1], target)
    # The loaded command is finite-P drive truth, never measured finger angles,
    # the event's release target, or a later submitted command.
    held = replay.at(.0165)[1]
    assert not np.array_equal(held, rows[1]["measured_finger_q"])
    assert not np.array_equal(held, rows[2]["target_finger_q"])
    assert not np.array_equal(held, rows[3]["target_finger_q"])
    held[:] = 99
    np.testing.assert_array_equal(replay.at(99)[1], rows[1]["target_finger_q"])
    np.testing.assert_array_equal(source.fingers, [row["target_finger_q"] for row in rows])
    assert trace.read_bytes() == original
    assert "intervention" not in source.metadata
    detail = replay.metadata["intervention"]
    assert detail["start_s"] == .0165
    assert detail["source_command_s"] == .0085
    assert detail["source_row_index"] == 1
    assert detail["superseded_event_row_index"] == 2
    assert detail["source_finger_targets_rad"] == rows[1]["target_finger_q"]
    assert "counterfactual" in detail["input_trace"]
    assert "not a policy score" in replay.metadata["semantics"]
    assert replay.metadata["sha256"] == source.metadata["sha256"]
    json.dumps(replay.metadata, allow_nan=False)


@pytest.mark.parametrize("time", [-1., float("nan"), float("inf"), 0., .0005, .006, .02])
def test_finger_hold_rejects_missing_predecessor_nonfinite_or_unrecorded_event(tmp_path, time):
    cfg = config()
    trace, manifest = files(tmp_path, cfg, [command(.0005, .6, cfg), command(.0085, .2, cfg)])
    replay = NativeRecordedDriveCommands(trace, manifest_path=manifest, cfg=cfg)
    with pytest.raises(ValueError, match="Finger-hold"):
        NativeFingerHoldReplay(replay, hold_at_s=time)


def test_finger_hold_requires_native_replay_and_explicit_runner_mode(tmp_path):
    cfg = config()
    trace, manifest = files(tmp_path, cfg, [command(.0005, .6, cfg), command(.0085, .2, cfg)])
    args = SimpleNamespace(command_trace=trace, policy_initial_state=tmp_path / "initial.json",
                           command_trace_manifest=manifest, command_finger_hold_at=.0085,
                           mode="policy")
    with pytest.raises(ValueError, match="requires command_replay"):
        load_command_replay(args, cfg)
    args.mode = "command_replay"
    assert isinstance(load_command_replay(args, cfg), NativeFingerHoldReplay)
    with pytest.raises(ValueError, match="requires native adaptive"):
        load_command_replay(args, {"gripper": {"stroke": .085}})
    with pytest.raises(ValueError, match="requires native recorded"):
        NativeFingerHoldReplay(object(), hold_at_s=.0085)
