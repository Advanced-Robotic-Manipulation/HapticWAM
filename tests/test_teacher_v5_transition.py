"""CPU-only handoff guards; no subprocess or socket construction in tests."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "teacher_v5_transition",
    ROOT / "docs/results/teacher_success_anchor_v5/transition_to_confirmation.py",
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def fake_process(proc, *, start=None, state="S", argv=None):
    p = proc / str(m.PID)
    p.mkdir(parents=True)
    fields = [state] + ["0"] * 18 + [start or m.START_TICKS] + ["0"] * 5
    (p / "stat").write_text(f"{m.PID} (python worker) " + " ".join(fields))
    (p / "cmdline").write_bytes(
        b"\0".join(a.encode() for a in (argv or ["python"])) + b"\0"
    )
    boot = proc / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text(m.BOOT_ID)
    return p


def test_exact_screen_identity_and_zombie_exit(tmp_path):
    p = fake_process(tmp_path, argv=["venv/python", "frozen/run.py"])
    assert m.screen_running(["venv/python", "frozen/run.py"], tmp_path)
    (p / "stat").write_text((p / "stat").read_text().replace(") S ", ") Z "))
    (p / "cmdline").write_bytes(b"")
    assert not m.screen_running(["venv/python", "frozen/run.py"], tmp_path)


@pytest.mark.parametrize("change", ["start", "argv", "boot"])
def test_pid_reuse_exec_replacement_and_reboot_abort(tmp_path, change):
    fake_process(
        tmp_path,
        start="999" if change == "start" else None,
        argv=["different"] if change == "argv" else ["python"],
    )
    if change == "boot":
        (tmp_path / "sys/kernel/random/boot_id").write_text("other boot")
    with pytest.raises(RuntimeError):
        m.screen_running(["python"], tmp_path)


def complete_fixture(folder):
    design = {
        "policies": [{"id": f"p{i}"} for i in range(4)],
        "conditions": [{"id": "anchor"}],
        "sampling_seeds": list(range(6)),
        "thresholds": {"hold_s": 0.5},
    }
    keys = m.trial_keys(design)
    progress = {
        "status": "all_trials_completed",
        "campaign_sha256": "frozen",
        "controller_pid": m.PID,
        "trials": {},
    }
    for key in keys:
        status = {"status": "completed", "exit_code": 0}
        progress["trials"][key] = status
        put(folder / "rollouts" / key / "run_status.json", status)
        put(
            folder / "analysis/trials" / f"{key}.json",
            {
                "status": "scored",
                "case": {"campaign_sha256": "frozen"},
                "metrics": {
                    "valid_for_scoring": True,
                    "thresholds": design["thresholds"],
                },
            },
        )
    put(folder / "progress.json", progress)
    return design, sorted(keys)[0]


def test_complete_matched_grid_requires_all24_valid_and_no_reuse(tmp_path):
    design, key = complete_fixture(tmp_path)
    assert (
        m.require_complete(tmp_path, design, "frozen", m.PID)["completed_valid_trials"]
        == 24
    )
    path = tmp_path / "analysis/trials" / f"{key}.json"
    original = m.read(path)
    changed = copy.deepcopy(original)
    changed["metrics"]["valid_for_scoring"] = False
    put(path, changed)
    with pytest.raises(RuntimeError, match="Invalid"):
        m.require_complete(tmp_path, design, "frozen", m.PID)
    put(path, original)
    put(
        tmp_path / "rollouts" / key / "run_status.json",
        {"status": "completed", "exit_code": 0, "reused": True},
    )
    with pytest.raises(RuntimeError, match="reused"):
        m.require_complete(tmp_path, design, "frozen", m.PID)


def test_completed_header_cannot_hide_missing_or_extra_case(tmp_path):
    design, key = complete_fixture(tmp_path)
    path = tmp_path / "progress.json"
    original = m.read(path)
    changed = copy.deepcopy(original)
    del changed["trials"][key]
    put(path, changed)
    with pytest.raises(RuntimeError, match="Incomplete"):
        m.require_complete(tmp_path, design, "frozen", m.PID)
    put(path, original)
    (tmp_path / "rollouts/unplanned").mkdir()
    with pytest.raises(RuntimeError, match="Unexpected"):
        m.require_complete(tmp_path, design, "frozen", m.PID)


def test_confirmation_argv_only_changes_campaign_and_output_preserving_venv():
    command = [
        "/repo/.venv/bin/python",
        "/driver/run_policy_campaign.py",
        "--campaign",
        "screen.json",
        "--source",
        "frozen_v1",
        "--output",
        "screen",
        "--port",
        "7799",
        "--execute",
    ]
    saved = list(command)
    result = m.confirmation_command(command, "confirm.json", "confirmation")
    assert command == saved
    assert (
        result
        == saved[:3] + ["confirm.json"] + saved[4:7] + ["confirmation"] + saved[8:]
    )
    assert result[0] == "/repo/.venv/bin/python"
    with pytest.raises(RuntimeError, match="Ambiguous"):
        m.confirmation_command(command + ["--output", "elsewhere"], "c", "o")


def test_selection_requires_complete_hash_bound_stage(tmp_path):
    p = tmp_path / "selection.json"
    selected = {
        "status": "complete_valid_matched_stage",
        "expected_trials": 24,
        "available_trials": 24,
        "missing_trial_keys": [],
        "invalid_trial_keys": [],
        "campaign_sha256": "frozen",
        "protocol_sha256": m.PINS[m.PROTOCOL],
        "stage": "screen",
    }
    put(p, selected)
    assert m.require_selection(p, "frozen", "screen") == selected
    for field, value in (
        ("campaign_sha256", "changed"),
        ("stage", "confirmation"),
        ("invalid_trial_keys", [["p", "anchor", 1]]),
        ("available_trials", 23),
    ):
        put(p, {**selected, field: value})
        with pytest.raises(RuntimeError):
            m.require_selection(p, "frozen", "screen")
