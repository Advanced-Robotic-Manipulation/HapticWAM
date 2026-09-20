import copy
import json
from pathlib import Path

import pytest

from tools.sim.audit_teacher_anchor_v4_bridge import evaluate_bridge
from tools.sim.freeze_teacher_anchor_v4_confirmation import derive_confirmation
from tools.sim.run_teacher_anchor_v4 import blocks, inference_config
from tools.sim.select_teacher_anchor_v4 import COUNTS, evaluate_selection
from tools.sim.teacher_anchor_design import reference_path
from tools.sim.teacher_anchor_v4_design import (
    PROTOCOL_SHA,
    checked_json,
    digest,
    load_protocol,
    make_screen,
    shared_digest,
    validate_design,
    verify_execution_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def screen():
    return json.loads(
        (ROOT / "configs/sim/teacher_success_anchor_v4_screen.json").read_text()
    )


def grid(design):
    return [
        dict(
            policy_id=p["id"],
            condition_id=c["id"],
            sampling_seed=s,
            valid_for_selection=True,
            **dict.fromkeys(COUNTS, False),
        )
        for p in design["policies"]
        for c in design["conditions"]
        for s in design["sampling_seeds"]
    ]


def confirmation():
    d = screen()
    rows = grid(d)
    result = evaluate_selection(load_protocol(), d, rows, "screen")
    result.update(campaign_sha256="screen", trials=rows)
    return derive_confirmation(
        load_protocol(), d, result, screen_sha="screen", selection_sha="selection"
    )


def bridge():
    p = load_protocol()
    d = checked_json(p["bridge"]["campaign_path"], p["bridge"]["campaign_sha256"])
    rows = []
    for s in d["sampling_seeds"]:
        rows.append(
            {
                "policy_id": d["policies"][0]["id"],
                "condition_id": d["conditions"][0]["id"],
                "sampling_seed": s,
                "valid_for_selection": True,
                "strict_full_place": True,
                "final_supported_placement": True,
                "completed_reason": "placement_release_finished",
                "actual_stop": False,
                "raw_trace_end_s": 59.996,
                "reported_duration_s": 59.996,
            }
        )
    return p, d, rows


def test_v4_preserves_legacy_freeze_and_exact_physics_while_binding_combined_bridge():
    p = load_protocol()
    d = screen()
    assert (
        digest(ROOT / reference_path(p["supersedes"]["protocol_path"]))
        == p["supersedes"]["protocol_sha256"]
    )
    assert (
        digest(ROOT / reference_path(p["supersedes"]["legacy_screen_path"]))
        == p["supersedes"]["legacy_screen_sha256"]
    )
    assert make_screen(p) == d
    validate_design(p, d, "screen")
    b = checked_json(p["bridge"]["campaign_path"], p["bridge"]["campaign_sha256"])
    from tools.sim.analyze_policy_campaign import condition_scene

    assert d["nominal_scene"] == condition_scene(b, b["conditions"][0])
    assert d["conditions"][0]["object_offset_m"] == [0, 0]
    assert d["adapter_profile"] == b["adapter_profile"]
    assert d["adapter_profile"]["wrist_model"] == "gripper_contact_proxy"
    assert d["adapter_profile"]["terminal_veto_feedback_source"] == "current_delivery"
    assert d["adapter_profile"]["placement_release"]["finish_after_release"] is True
    assert d["runtime_contract"]["source"].endswith("/source_teacher_v2_delivery")


def test_v4_same_prospective_grid_and_per_policy_nfe_as_accepted_parent():
    d = screen()
    assert len(blocks(d)) == 24
    assert d["sampling_seeds"] == list(range(904401, 904407))
    assert [inference_config(d, p)["nfe"] for p in d["policies"]] == [1, 1, 1, 5]
    assert all(inference_config(d, p)["k_seeds"] == 4 for p in d["policies"])
    c = confirmation()
    assert c["sampling_seeds"] == list(range(904501, 904513))
    assert c["adapter_profile"] == d["adapter_profile"]
    assert sum(len(b["trials"]) for b in blocks(c)) == 24


def test_two_clean_bridge_trials_require_actual_full_horizon():
    p, d, rows = bridge()
    result = evaluate_bridge(p, d, rows)
    assert result["status"] == "passed"
    assert result["horizon_tolerance_s"] == pytest.approx(0.005)
    assert result["interpretation"].startswith("Both reused development seeds")


@pytest.mark.parametrize(
    "change",
    [
        "one_failure",
        "short_finish",
        "late_safety",
        "no_support",
        "no_finish",
        "invalid",
        "missing",
        "duration_lie",
    ],
)
def test_bridge_failure_blocks_screen_without_fallback_or_replacement(change):
    p, d, rows = bridge()
    if change == "one_failure":
        rows[1]["strict_full_place"] = False
    if change == "short_finish":
        rows[1]["raw_trace_end_s"] = rows[1]["reported_duration_s"] = 27.2
    if change == "late_safety":
        rows[1]["actual_stop"] = True
    if change == "no_support":
        rows[1]["final_supported_placement"] = False
    if change == "no_finish":
        rows[1]["completed_reason"] = None
    if change == "invalid":
        rows[1]["valid_for_selection"] = False
    if change == "missing":
        rows.pop()
    if change == "duration_lie":
        rows[1]["reported_duration_s"] = 60
        rows[1]["raw_trace_end_s"] = 40
    result = evaluate_bridge(p, d, rows)
    assert result["status"] == "blocked_incomplete_or_failed_bridge"
    assert result["decision"] == "stop_at_diagnosis_no_fallback_no_seed_search"


def test_duplicate_bridge_attempt_cannot_be_chosen_implicitly():
    p, d, rows = bridge()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_bridge(p, d, rows)


@pytest.mark.parametrize(
    "change", ["gel_only", "legacy_source", "threshold", "scene", "seed", "nfe"]
)
def test_rehash_does_not_allow_favored_profile_or_numerical_changes(change):
    d = screen()
    if change == "gel_only":
        d["adapter_profile"]["wrist_model"] = "contact_proxy"
    if change == "legacy_source":
        d["runtime_contract"]["source"] = d["runtime_contract"]["source"].replace(
            "source_teacher_v2_delivery", "source_teacher_pick_place_v1"
        )
    if change == "threshold":
        d["thresholds"]["lift_height_m"] = 0.001
    if change == "scene":
        d["nominal_scene"]["waffle"]["center"][0] += 0.01
    if change == "seed":
        d["sampling_seeds"][0] = 4242
    if change == "nfe":
        d["policies"][0]["inference_settings"]["nfe"] = 3
    d["anchor_shared_configuration_sha256"] = shared_digest(d)
    with pytest.raises(ValueError):
        validate_design(load_protocol(), d, "screen")


def test_v4_physical_winner_tests_remain_exact_and_distinct_from_clean_completion():
    d = confirmation()
    rows = grid(d)
    for row in rows[:8]:
        row["strict_full_place"] = True
    result = evaluate_selection(load_protocol(), d, rows, "confirmation")
    assert result["clear_simulator_winner"] == d["policies"][0]["id"]
    assert result["ranking"][0]["clean_place"] == 0
    for row in rows[12:16]:
        row["strict_full_place"] = True
    result = evaluate_selection(load_protocol(), d, rows, "confirmation")
    assert result["paired_comparison"]["estimate"]["exact_two_sided_mcnemar_p"] == 0.125
    assert result["clear_simulator_winner"] is None


def test_v4_execution_requires_actual_passed_bridge_audit(tmp_path):
    with pytest.raises(ValueError, match="stage-gate-audit"):
        verify_execution_gate(screen(), None, Path("/source"))
    path = tmp_path / "failed.json"
    path.write_text(json.dumps({"status": "failed", "protocol_sha256": PROTOCOL_SHA}))
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_execution_gate(screen(), path, Path("/source"))


def test_v4_gate_rechecks_raw_trials_and_corrected_source_identity(
    tmp_path, monkeypatch
):
    from tools.sim import audit_teacher_anchor_v4_bridge as audit_module
    from tools.sim import teacher_anchor_v4_design as design_module

    d = screen()
    p, b, rows = bridge()
    result = evaluate_bridge(p, b, rows)
    result["runs"] = "/synthetic_bridge"
    path = tmp_path / "passed.json"
    path.write_text(json.dumps(result))
    raw_result = copy.deepcopy(result)
    monkeypatch.setattr(
        audit_module, "audit_bridge", lambda *_: copy.deepcopy(raw_result)
    )
    real_digest = design_module.digest
    hashes = {
        r["path"]: r["sha256"] for r in d["runtime_contract"]["source_and_input_hashes"]
    }
    monkeypatch.setattr(
        design_module, "digest", lambda path: hashes.get(str(path)) or real_digest(path)
    )
    for row in result["trials"]:
        row["input_sha256"] = {"run.json": "fixture"}
    raw_result = copy.deepcopy(result)
    path.write_text(json.dumps(result))
    gate = verify_execution_gate(d, path, Path(d["runtime_contract"]["source"]))
    assert gate["kind"] == "two_complete_clean_policy_bridges"
    raw_result["trials"][0]["strict_full_place"] = False
    with pytest.raises(ValueError, match="raw evidence changed"):
        verify_execution_gate(d, path, Path(d["runtime_contract"]["source"]))
    raw_result = copy.deepcopy(result)
    first = d["runtime_contract"]["source_and_input_hashes"][0]["path"]
    hashes[first] = "changed"
    with pytest.raises(ValueError, match="source/input hash changed"):
        verify_execution_gate(d, path, Path(d["runtime_contract"]["source"]))


def test_v4_runner_blocks_missing_gate_before_any_child_process(tmp_path, monkeypatch):
    from argparse import Namespace
    from tools.sim import run_teacher_anchor_v4 as runner

    d = screen()
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--campaign",
            str(ROOT / "configs/sim/teacher_success_anchor_v4_screen.json"),
            "--source",
            d["runtime_contract"]["source"],
            "--evidence",
            "/evidence",
            "--hardware-config",
            "/hardware",
            "--output",
            str(tmp_path),
            "--execute",
        ],
    )

    def unexpected_child(*_args, **_kwargs):
        pytest.fail("Gate failure must precede every model/simulator process")

    monkeypatch.setattr(runner.subprocess, "Popen", unexpected_child)
    with pytest.raises(
        ValueError, match="V4 model execution requires --stage-gate-audit"
    ):
        runner.main()
    command = runner.analysis_command(
        Namespace(server_python="/venv/python", output=tmp_path),
        tmp_path / "campaign.json",
    )
    assert command[1] == str(ROOT / "tools/sim/analyze_policy_campaign.py")
