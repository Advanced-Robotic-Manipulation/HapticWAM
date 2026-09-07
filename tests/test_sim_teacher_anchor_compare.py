import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.sim.teacher_anchor_compare import (
    COUNTS,
    derive_confirmation,
    digest,
    evaluate_selection,
    legacy_completion_evidence,
    validate_design,
)

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    p = ROOT / "docs/results/teacher_success_anchor_v5/protocol.json"
    d = ROOT / "configs/sim/teacher_success_anchor_v5_screen.json"
    return json.loads(p.read_text()), digest(p), json.loads(d.read_text())


def grid(d):
    return [
        dict(
            policy_id=p["id"],
            condition_id=c["id"],
            sampling_seed=s,
            valid_for_selection=True,
            **dict.fromkeys(COUNTS, False),
        )
        for p in d["policies"]
        for c in d["conditions"]
        for s in d["sampling_seeds"]
    ]


def finished_fixture(full=True):
    record = {
        "policy_id": "teacher",
        "condition_id": "anchor",
        "sampling_seed": 4242,
        "directory": "/fixture",
        "status": "scored",
        "metrics": {
            "valid_for_scoring": True,
            "invalid_reasons": [],
            "outcomes": dict(
                full_task=full,
                acquired=full,
                lifted=full,
                carried=full,
                released_in_bin=full,
                dropped=False,
                reach_before_closure=True,
            ),
            "event_times_s": {"full_task": 25.2 if full else None},
            "object": {"final_inside_bin": True, "final_contacts_unloaded": True},
            "placement_support": {"required": True, "verified_placement": full},
            "control": {},
        },
    }
    run = {}
    execution = [
        {
            "t": 26.0,
            "stopped": False,
            "diagnostics": {
                "completed_reason": "placement_release_finished",
                "completed_at_s": 26.0,
                "completion_hold": True,
            },
        },
        {
            "t": 59.992,
            "stopped": False,
            "diagnostics": {
                "completed_reason": "placement_release_finished",
                "completed_at_s": 26.0,
                "completion_hold": True,
            },
        },
    ]
    trace = {
        "packet_robot_normal_force": np.array([0.0]),
        "packet_bin_normal_force": np.array([0.34]),
    }
    return record, run, trace, execution


def test_legacy_finish_can_be_read_without_top_level_header_or_raw_mutation():
    _, _, d = inputs()
    record, run, trace, execution = finished_fixture()
    saved = copy.deepcopy((record, run, execution))
    row = legacy_completion_evidence(record, run, trace, execution, [], d["thresholds"])
    assert (
        row["valid_for_selection"] and row["clean_place"] and row["strict_full_place"]
    )
    assert row["completed_reason"] == "placement_release_finished"
    assert row["completion_reason_source"] == "execution_only"
    assert not row["raw_run_has_completion_field"]
    assert (record, run, execution) == saved


def test_finish_diagnostics_never_supply_physical_success():
    _, _, d = inputs()
    r, run, t, e = finished_fixture(full=False)
    row = legacy_completion_evidence(r, run, t, e, [], d["thresholds"])
    assert (
        row["valid_for_selection"]
        and row["completed_reason"] == "placement_release_finished"
    )
    assert not row["clean_place"] and not row["strict_full_place"]


@pytest.mark.parametrize("conflict", ["explicit_header", "execution"])
def test_conflicting_completion_records_are_invalid_not_silently_fixed(conflict):
    _, _, d = inputs()
    r, run, t, e = finished_fixture()
    if conflict == "explicit_header":
        run["policy_completed_reason"] = None
    else:
        e[0]["diagnostics"]["completed_reason"] = "another_reason"
    row = legacy_completion_evidence(r, run, t, e, [], d["thresholds"])
    assert not row["valid_for_selection"]


def test_v5_has_no_reference_teacher_win_gate_and_keeps_matched_prospective_grid():
    p, sha, d = inputs()
    validate_design(p, sha, d, "screen")
    assert "stage_gate" not in d
    assert d["execution_admission"]["reference_teacher_success_required"] is False
    assert d["sampling_seeds"] == list(range(904401, 904407))
    rows = grid(d)
    result = evaluate_selection(p, d, rows, "screen")
    result.update(campaign_sha256="screen", trials=rows)
    c = derive_confirmation(p, sha, d, "screen", result, "selection")
    assert c["sampling_seeds"] == list(range(904501, 904513))
    assert c["adapter_profile"] == d["adapter_profile"]
    assert c["runtime_contract"] == d["runtime_contract"]


@pytest.mark.parametrize("field", ["profile", "scene", "recipe", "seed"])
def test_profile_or_geometry_or_recipe_changes_cannot_enter_frozen_comparison(field):
    p, sha, d = inputs()
    if field == "profile":
        d["adapter_profile"]["gel_contact_coverage"] = "manifold_patch"
    if field == "scene":
        d["nominal_scene"]["waffle"]["center"][0] -= 0.01
    if field == "recipe":
        d["policies"][0]["inference_settings"]["nfe"] = 9
    if field == "seed":
        d["sampling_seeds"][0] = 4242
    with pytest.raises(ValueError):
        validate_design(p, sha, d, "screen")


def test_confirmation_retains_exact_small_sample_and_physical_primary_rules():
    p, sha, d = inputs()
    rows = grid(d)
    selection = evaluate_selection(p, d, rows, "screen")
    selection.update(campaign_sha256="screen", trials=rows)
    c = derive_confirmation(p, sha, d, "screen", selection, "selection")
    rows = grid(c)
    for r in rows[:8]:
        r["strict_full_place"] = True
    win = evaluate_selection(p, c, rows, "confirmation")
    assert win["clear_simulator_winner"] == c["policies"][0]["id"]
    for r in rows[12:16]:
        r["strict_full_place"] = True
    weak = evaluate_selection(p, c, rows, "confirmation")
    assert weak["paired_comparison"]["estimate"]["exact_two_sided_mcnemar_p"] == 0.125
    assert weak["clear_simulator_winner"] is None
