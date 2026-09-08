"""CPU-only checks for honest outcome and startup-retry accounting."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docs.results.teacher_scene_v10_20260908 import plot_study as plot
from docs.results.teacher_scene_v10_20260908 import summarize_study as summary
from docs.results.teacher_scene_v10_20260908 import verify_completion as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_first_stop_keeps_simultaneous_guards_without_later_relabeling():
    first = {"t": 2., "stopped": True, "stop_reason": "safety_stop",
             "diagnostics": {"safety_events": ["joint_speed", "workspace_clamp"]}}
    later = {"t": 3., "stopped": True, "stop_reason": "safety_stop",
             "diagnostics": {"safety_events": ["wrist_force"]}}
    result = summary.first_stop_diagnostic([{"t": 1., "stopped": False}, first, later])
    assert result["safety_events"] == ["joint_speed", "workspace_clamp"]
    assert result["t"] == 2.
    assert "not individually assigned causal priority" in result["interpretation"]
    assert summary.first_stop_diagnostic([{"t": 1., "stopped": False}]) is None


def mock_outcomes():
    return {stage: False for stage in (*summary.STAGES, *summary.EXTRA)}


def matched_summary():
    trials = [{"variant": f"v{v}", "condition": f"start{c}", "seed": s,
               "status": "scored", "outcomes": mock_outcomes()}
              for c in range(6) for s in (1, 2) for v in range(5)]
    # Retain invalid rows even if a stale caller supplies successful values.
    trials[0].update(status="missing_score", outcomes=dict.fromkeys(mock_outcomes()))
    trials[1].update(status="invalid", outcomes=dict.fromkeys(mock_outcomes(), True))
    variants = {}
    for v in range(5):
        selected = [r for r in trials if r["variant"] == f"v{v}"]
        safe = [{"outcomes": r["outcomes"] if r["status"] == "scored" else dict.fromkeys(mock_outcomes())} for r in selected]
        variants[f"v{v}"] = {"outcomes": {stage: summary.outcome_summary(safe, stage) for stage in plot.STAGES}}
    return {"planned_trials": 60, "variants": variants, "trials": trials}


def test_matrix_retains_all_sixty_rows_and_invalid_scores_are_unresolved():
    data = matched_summary()
    variants, rows, lookup, counts = plot.prepare(data)
    assert len(variants) == 5 and len(rows) == 12 and len(lookup) == 60
    assert plot.category(data["trials"][0]) == "unresolved"
    assert plot.category(data["trials"][1]) == "unresolved"
    assert counts["v1"]["full_task"] == {"true": 0, "false": 11, "unresolved": 1}
    for stages in counts.values():
        assert all(sum(value.values()) == 12 for value in stages.values())
    with pytest.raises(ValueError, match="Every planned trial"):
        plot.prepare({**data, "trials": data["trials"][:-1]})
    with pytest.raises(ValueError, match="Duplicate trial"):
        plot.prepare({**data, "trials": data["trials"][:-1] + [data["trials"][0]]})


def test_matrix_distinguishes_initial_sustained_and_clean():
    row = {"status": "scored", "outcomes": mock_outcomes()}
    row["outcomes"]["full_task"] = True
    row["outcomes"][plot.SUSTAINED] = None
    assert plot.category(row) == "full_task"
    row["outcomes"][plot.SUSTAINED] = False
    assert plot.category(row) == "full_task"
    row["outcomes"][plot.SUSTAINED] = True
    assert plot.category(row) == plot.SUSTAINED
    row["outcomes"][plot.CLEAN] = True
    assert plot.category(row) == plot.CLEAN


def test_render_does_not_label_unresolved_stale_score_as_a_drop(tmp_path, monkeypatch):
    axes = pytest.importorskip("matplotlib.axes")
    data = matched_summary()
    data["trials"][2]["outcomes"]["dropped"] = True
    cells = []
    original = axes.Axes.text

    def capture(self, x, y, text, *args, **kwargs):
        if kwargs.get("wrap"):
            cells.append(text)
        return original(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(axes.Axes, "text", capture)
    manifest = plot.render(data, tmp_path)
    assert len(cells) == 60 and manifest["planned_trials"] == 60
    assert sum("\nDROP" in cell for cell in cells) == 1
    assert all("DROP" not in cell for cell in cells if cell.startswith("Unresolved"))
    assert (tmp_path / "outcome_matrix.svg").is_file()


@pytest.fixture
def recovery_fixture(tmp_path):
    schedule = [{"variant": f"v{v}", "condition": f"start{c}", "seed": s}
                for c in range(6) for s in (1, 2) for v in range(5)]
    study_path = tmp_path / "study.json"
    write(study_path, {"schedule": schedule})
    args = SimpleNamespace(output=tmp_path / "campaign", source=Path(__file__).resolve().parents[1])
    archive = tmp_path / "amendments/startup_timeout_case28_attempt1"
    proof_path = tmp_path / "amendments/startup_timeout_recovery.json"

    def case(index):
        row = schedule[index]
        return {"study_sha256": audit.fingerprint(study_path), "campaign_sha256": "campaign",
                "schedule_index": index, "attempt": 1, "variant": row["variant"],
                "condition_id": row["condition"], "sampling_seed": row["seed"], "policy_id": "teacher",
                "checkpoint_sha256": "checkpoint", "intended_parameters": {}, "intended_scene_sha256": "scene",
                "scoring_thresholds": {}}

    def name(index):
        row = schedule[index]
        return row["variant"] + "/" + audit.campaign.trial_id("teacher", row["condition"], row["seed"])

    directories = {}
    for index in range(28):
        variant, case_id = name(index).split("/")
        directory = args.output / variant / "rollouts" / case_id
        write(directory / "case.json", case(index))
        write(directory / "run_status.json", {"status": "completed", "started_unix_s": 101})
        (directory / "raw.bin").write_bytes(bytes([index]))
        directories[index] = directory
    original = archive / "attempt1"
    command = ["sim", "--sampling-seed", str(schedule[27]["seed"]), "--output", str(directories[27])]
    write(directories[27] / "command.json", command)
    write(original / "case.json", case(27))
    write(original / "command.json", command)
    write(original / "run_status.json", {"status": "timeout"})
    for file in ("server_ready.json", "resources_before.json", "effective_config.json"):
        write(original / file, {})
    (original / "isaac.log").touch()
    proof = {"kind": "explicit_startup_infrastructure_retry", "study_sha256": audit.fingerprint(study_path),
             "runner_sha256": audit.fingerprint(audit.run_study.__file__),
             "recovery_helper_sha256": audit.fingerprint(args.source / "docs/results/teacher_scene_v10_20260908/recover_startup_timeout.py"),
             "archive_verified": True, "completed_trials_preserved": 27, "planned_trials_unchanged": 60,
             "schedule_index": 27, "archived_attempt": 1, "next_infrastructure_attempt": 2,
             "archive": str(archive), "trial": name(27), "canonical_retry_directory": str(directories[27]),
             "command_sha256": audit.fingerprint(original / "command.json"),
             "created_at_utc": "1970-01-01T00:01:40+00:00"}

    def refresh():
        proof["attempt_file_sha256"] = audit.full_file_hashes(original)
        proof["archive_file_sha256"] = audit.full_file_hashes(archive)
        proof["completed_trial_file_sha256"] = {name(i): audit.full_file_hashes(directories[i]) for i in range(27)}
        write(proof_path, proof)
    refresh()
    return study_path, args, directories, original, proof, refresh


def test_recovery_audit_reports_true_second_attempt_with_identical_seed(recovery_fixture):
    path, args, *_ = recovery_fixture
    result = audit.infrastructure_recovery_audit(path, args)
    assert result["valid"] and result["retry_count"] == 1
    assert result["preserved_completed_trials"] == 27 and result["schedule_index"] == 27
    assert result["same_command_and_seed"] is True
    assert "directory-local field, not total launch count" in result["interpretation"]


def test_recovery_audit_rejects_preserved_trial_bound_to_wrong_schedule(recovery_fixture):
    path, args, directories, _, _, refresh = recovery_fixture
    case = audit.read(directories[0] / "case.json")
    case["schedule_index"] = 1
    write(directories[0] / "case.json", case)
    refresh()  # Even a self-consistent hash manifest cannot change the schedule.
    with pytest.raises(ValueError, match="identity differs from the frozen schedule"):
        audit.infrastructure_recovery_audit(path, args)


def test_recovery_audit_rejects_index_27_in_preserved_prefix(recovery_fixture):
    path, args, directories, _, _, refresh = recovery_fixture
    case = audit.read(directories[0] / "case.json")
    case["schedule_index"] = 27
    write(directories[0] / "case.json", case)
    refresh()
    with pytest.raises(ValueError, match="not among the first 27"):
        audit.infrastructure_recovery_audit(path, args)


def test_recovery_audit_rejects_retry_bound_to_wrong_schedule(recovery_fixture):
    path, args, directories, original, _, refresh = recovery_fixture
    for directory in (original, directories[27]):
        case = audit.read(directory / "case.json")
        case["schedule_index"] = 28
        write(directory / "case.json", case)
    refresh()
    with pytest.raises(ValueError, match="not the original scheduled case 27"):
        audit.infrastructure_recovery_audit(path, args)


def test_recovery_audit_rejects_changed_raw_bytes_or_retry_command(recovery_fixture):
    path, args, directories, _, _, _ = recovery_fixture
    original_bytes = (directories[0] / "raw.bin").read_bytes()
    (directories[0] / "raw.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="completed pre-recovery trial changed"):
        audit.infrastructure_recovery_audit(path, args)
    (directories[0] / "raw.bin").write_bytes(original_bytes)
    write(directories[27] / "command.json", ["changed_sampling_seed"])
    with pytest.raises(ValueError, match="Retry command differs"):
        audit.infrastructure_recovery_audit(path, args)


@pytest.mark.parametrize("retry_valid", [False, True])
def test_completion_subtracts_only_a_valid_retried_case(tmp_path, retry_valid):
    schedule = [{k: row[k] for k in ("variant", "condition", "seed")} for row in matched_summary()["trials"]]
    study = {"schedule": schedule}
    path = tmp_path / "study.json"
    write(path, study)
    args = SimpleNamespace(output=tmp_path / "campaign", source=tmp_path, port=7799)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    variants = {f"v{i}": {"args": SimpleNamespace(output=args.output / f"v{i}"),
                          "policy": {"id": "teacher", "checkpoint": str(checkpoint),
                                     "checkpoint_sha256": audit.fingerprint(checkpoint)}} for i in range(5)}

    def case_result(index, *_):
        if index == 27 and not retry_valid:
            raise ValueError("Retry has no valid physical evidence yet")
        return {"valid": True, "server_pid": 123, "stop_reason": "safety_stop", "started_unix_s": 100}

    with patch.object(audit.run_study, "load_study", return_value=(study, args, variants)), \
         patch.object(audit, "source_audit", return_value={"valid": True}), \
         patch.object(audit, "replay_audit", return_value={"valid": False}), \
         patch.object(audit, "infrastructure_recovery_audit", return_value={"valid": True, "retry_count": 1, "schedule_index": 27}), \
         patch.object(audit, "case_audit", side_effect=case_result), \
         patch.object(audit, "process_diagnostics", return_value={"valid": True}):
        report = audit.audit(path, tmp_path / "replay.json")
    assert report["counts"]["valid_completed"] == (60 if retry_valid else 59)
    assert report["counts"]["valid_without_startup_retry"] == 59
    assert report["counts"]["documented_startup_retries"] == 1
