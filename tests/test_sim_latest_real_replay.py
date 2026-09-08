"""No GPU/hardware: protect the latest-real diagnostic's execution boundary."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("latest_real_replay", ROOT / "docs/results/teacher_scene_v10_20260908/run_latest_real_replay.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def progress(tmp_path, n=60, status="all_trials_completed", analyses=5):
    p = tmp_path / "campaign/progress.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"status": status, "trials": {str(i): {"status": "completed"} for i in range(n)}, "analysis": {str(i): {"status": "completed"} for i in range(analyses)}}))
    return p


def test_does_not_overlap_incomplete_policy_campaign(tmp_path):
    progress(tmp_path, n=59, status="running")
    with pytest.raises(RuntimeError, match="not complete"):
        MODULE.baseline_finished(tmp_path)
    progress(tmp_path, n=59)
    with pytest.raises(RuntimeError, match="all60"):
        MODULE.baseline_finished(tmp_path)


def test_waits_for_all_policy_analyses(tmp_path):
    progress(tmp_path, analyses=4)
    with pytest.raises(RuntimeError, match="analyses still incomplete"):
        MODULE.baseline_finished(tmp_path)
    progress(tmp_path)
    assert len(MODULE.baseline_finished(tmp_path)["trials"]) == 60


def test_frozen_source_hash_rejects_code_change_before_gpu(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    file = source / "run.py"
    file.write_text("# frozen\n")
    args = SimpleNamespace(source=source, out=tmp_path / "out")
    protocol = {"source": str(source), "output": str(args.out), "runner_sha256": MODULE.sha(MODULE.__file__), "source_files_sha256": {"run.py": MODULE.sha(file)}}
    file.write_text("# altered\n")
    with pytest.raises(ValueError, match="Frozen simulator source changed"):
        MODULE.validate(args, protocol)


def test_diagnostic_does_not_claim_placement_or_native_hotfix(tmp_path):
    import numpy as np
    from phantom.sim.kinematics import forward_pose

    case = tmp_path / "dynamics"
    reference = tmp_path / "reference"
    case.mkdir()
    reference.mkdir()
    q = np.array([.1, -1.3, 1.7, .1, 1.4, -3.])
    time = np.arange(0, 19.201, .04)
    pose = forward_pose(q)
    xyz = np.tile([-.39, -.27, .0355], (len(time), 1))
    trace = dict(t=time, physics_t=time, q=np.tile(q, (len(time), 1)), tcp=np.tile(pose, (len(time), 1)), waffle_position=xyz,
                 pad_packet_normal_force=np.zeros((len(time), 2)), packet_robot_normal_force=np.zeros(len(time)), packet_bin_normal_force=np.zeros(len(time)))
    np.savez(case / "sim_trace.npz", **trace)
    np.savez(reference / "replay.npz", t=time, q=trace["q"], tcp=trace["tcp"])
    cfg = ROOT / "configs/sim/waffles_d435_factory_20260908_r4.json"
    (case / "effective_config.json").write_bytes(cfg.read_bytes())
    (case / "run.json").write_text(json.dumps({"camera_projection": {"readback_matches_config": True}, "object_dynamics": {"rigid_body_dynamic": True, "kinematic": False, "attachments": [], "pose_writes_after_initialization": 0}}))
    rows = [{"t": float(t), "per_pad": [{"per_filter_contacts": [{"filter_path": "/World/Waffle", "gel_compression_n": 0.}]}] * 2} for t in time]
    (case / "gel_contact_trace.json").write_text(json.dumps(rows))
    protocol = {"kinematic_gates": {"dynamics_joint_rmse_rad": .05, "dynamics_tcp_rmse_m": .020, "coverage_end_tolerance_s": .1},
                "physical_diagnostic_thresholds": {"bilateral_packet_normal_n": .05, "bilateral_onset_hold_s": .2, "lift_onset_m": .02, "lift_onset_hold_s": .15, "minimum_peak_lift_m": .1, "bilateral_carry_fraction_min": .85, "maximum_bilateral_carry_gap_s": .3, "retention_tcp_relative_p95_m": .02, "retention_tcp_relative_max_m": .05},
                "time_origin": {"original_elapsed_to_export_elapsed_offset_s": 0.}, "real_adjudication": {"diagnostic_carry_original_elapsed_interval_s": [16., 18.75]}}
    result = MODULE.diagnostics(SimpleNamespace(out=tmp_path), protocol, "dynamics")
    assert result["kinematic_gates_passed"]
    assert not result["physical_pick_carry_reproduced"]
    assert not result["placement_claim"]
    assert not result["native_hotfix_validation_claim"]
    assert result["gel_any_packet_contact_rows"] == 0
    assert result["body_bilateral_contact_onset_s"] is None
