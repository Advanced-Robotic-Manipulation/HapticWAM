import ast
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from tools.sim.freeze_teacher_anchor_confirmation import derive_confirmation
from tools.sim.run_policy_campaign import (
    analysis_command,
    blocks,
    inference_config,
    simulation_command,
)
from tools.sim.select_teacher_anchor import (
    COUNTS,
    anchor_trial_evidence,
    evaluate_selection,
    paired_seed_statistics,
)
from tools.sim.teacher_anchor_design import (
    load_protocol,
    shared_digest,
    validate_design,
    verify_execution_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def screen():
    return json.loads(
        (ROOT / "configs/sim/teacher_success_anchor_v3_screen.json").read_text()
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


def physical_trial(full=True, later_stop=True, stop_t=31.38):
    record = {
        "policy_id": "teacher",
        "condition_id": "anchor",
        "sampling_seed": 4242,
        "directory": "/fixture",
        "status": "scored",
        "metrics": {
            "valid_for_scoring": True,
            "invalid_reasons": [],
            "outcomes": {
                "full_task": full,
                "acquired": full,
                "lifted": full,
                "carried": full,
                "released_in_bin": full,
                "dropped": False,
                "reach_before_closure": True,
            },
            "event_times_s": {"full_task": 25.2 if full else None},
            "object": {"final_inside_bin": full, "final_contacts_unloaded": full},
            "placement_support": {"required": True, "verified_placement": full},
            "control": {},
        },
    }
    run = {"policy_stop_reason": "safety_stop" if later_stop else None}
    execution = [
        {
            "t": 20.0,
            "stopped": False,
            "diagnostics": {"safety_events": ["workspace_clamp"]},
        }
    ]
    if later_stop:
        execution.append(
            {
                "t": stop_t,
                "stopped": True,
                "stop_reason": "safety_stop",
                "diagnostics": {"safety_events": ["wrist_extension"]},
            }
        )
    trace = {
        "packet_robot_normal_force": np.array([0.0]),
        "packet_bin_normal_force": np.array([0.34]),
    }
    return record, run, trace, execution


def test_later_stop_does_not_erase_strict_placement_or_become_preplacement_failure():
    record, run, trace, execution = physical_trial()
    row = anchor_trial_evidence(
        record, run, trace, execution, [], screen()["thresholds"]
    )
    assert row["valid_for_selection"] and row["strict_full_place"]
    assert row["post_placement_safety_stop"] and not row["pre_placement_safety_stop"]
    assert row["final_supported_placement"] and not row["clean_place"]
    assert row["terminal_safety_events"] == ["wrist_extension"]


def test_depth_failure_before_placement_is_preplacement_force_stop():
    record, run, trace, execution = physical_trial(full=False, stop_t=6.0)
    execution[-1]["diagnostics"]["safety_events"] = ["tactile_depth"]
    row = anchor_trial_evidence(
        record, run, trace, execution, [], screen()["thresholds"]
    )
    assert row["pre_placement_force_limit_stop"]
    assert row["tactile_force_limit_stop"] and not row["post_placement_safety_stop"]


def test_controller_completion_cannot_supply_physical_task_truth():
    record, run, trace, execution = physical_trial(full=False, later_stop=False)
    run["policy_completed_reason"] = "placement_release_finished"
    execution[-1]["diagnostics"]["completed_reason"] = "placement_release_finished"
    row = anchor_trial_evidence(
        record, run, trace, execution, [], screen()["thresholds"]
    )
    assert not row["strict_full_place"] and not row["clean_place"]


def test_missing_stop_time_blocks_selection_instead_of_hiding_force_event():
    record, run, trace, execution = physical_trial()
    del execution[-1]["t"]
    row = anchor_trial_evidence(
        record, run, trace, execution, [], screen()["thresholds"]
    )
    assert not row["valid_for_selection"]
    assert "actual_stop_time_missing" in row["invalid_reasons"]


def test_fixed_grid_and_legacy_scene_are_derived_once():
    d = screen()
    validate_design(load_protocol(), d, "screen")
    from tools.sim.analyze_policy_campaign import condition_scene

    assert (
        condition_scene(d, d["conditions"][0])
        == load_protocol()["common_inputs"]["fixed_scene"]
    )
    schedule = blocks(d)
    assert len(schedule) == 24
    assert all(len(b["trials"]) == 1 for b in schedule)
    assert [b["trials"][0][1] for b in schedule[:4]] == [904401] * 4
    assert len({(b["policy_id"], *b["trials"][0]) for b in schedule}) == 24
    assert inference_config(d, d["policies"][3])["nfe"] == 5
    assert inference_config(d, d["policies"][0])["k_seeds"] == 4


@pytest.mark.parametrize("change", ["seed", "policy", "duplicate"])
def test_phase_grid_rejects_same_count_wrong_cells(change):
    d = screen()
    if change == "seed":
        d["execution_phases"][0]["sampling_seeds"] = [999]
    if change == "policy":
        d["execution_phases"][0]["policy_ids"][0] = "unplanned"
    if change == "duplicate":
        d["execution_phases"][1]["sampling_seeds"] = [904401]
    with pytest.raises(ValueError, match="grid"):
        blocks(d)


def test_rank_primary_physical_success_above_clean_flag_and_later_stops():
    d = screen()
    rows = grid(d)
    rows[0]["strict_full_place"] = True
    rows[0]["post_placement_safety_stop"] = True
    rows[0]["actual_stop"] = rows[0]["safety_stop"] = True
    rows[6]["clean_place"] = rows[6]["lifted"] = rows[6]["acquired"] = True
    # The pure selector assumes audited rows, so this synthetic clean flag is
    # deliberately lower-tier: actual extraction independently prevents it.
    result = evaluate_selection(load_protocol(), d, rows, "screen")
    assert result["selected_ids"][0] == d["policies"][0]["id"]
    assert result["clear_simulator_winner"] is None


@pytest.mark.parametrize("invalid", [False, True])
def test_incomplete_or_invalid_rows_block_selection(invalid):
    d = screen()
    rows = grid(d)
    if invalid:
        rows[0]["valid_for_selection"] = False
    else:
        rows.pop()
    result = evaluate_selection(load_protocol(), d, rows, "screen")
    assert result["status"] == "incomplete_or_invalid_matched_stage"
    assert result["selected_ids"] == []


def test_exact_guard_rejects_four_favorable_discordant_pairs_despite_bootstrap_lower():
    rule = load_protocol()["confirmation_rule"]
    result = paired_seed_statistics([1] * 8 + [0] * 4, [1] * 4 + [0] * 8, rule)
    assert result["interval"][0] > 0
    assert result["exact_two_sided_mcnemar_p"] == 0.125
    d = confirmation()
    rows = grid(d)
    for row in rows[:8]:
        row["strict_full_place"] = True
    for row in rows[12:16]:
        row["strict_full_place"] = True
    result = evaluate_selection(load_protocol(), d, rows, "confirmation")
    assert not result["winner_gates"]["exact_two_sided_p_le_0_05"]
    assert result["clear_simulator_winner"] is None


def test_confirmation_winner_requires_no_extra_drops_or_preplacement_force_stops():
    d = confirmation()
    rows = grid(d)
    for row in rows[:8]:
        row["strict_full_place"] = True
    result = evaluate_selection(load_protocol(), d, rows, "confirmation")
    assert result["clear_simulator_winner"] == d["policies"][0]["id"]
    assert result["paired_comparison"]["estimate"]["paired_seed_count"] == 12
    rows[0]["pre_placement_force_limit_stop"] = True
    result = evaluate_selection(load_protocol(), d, rows, "confirmation")
    assert result["clear_simulator_winner"] is None
    rows[0]["pre_placement_force_limit_stop"] = False
    rows[0]["dropped"] = True
    assert (
        evaluate_selection(load_protocol(), d, rows, "confirmation")[
            "clear_simulator_winner"
        ]
        is None
    )


def test_zero_paired_interval_is_not_equivalence():
    r = paired_seed_statistics([0] * 12, [0] * 12, load_protocol()["confirmation_rule"])
    assert r["interval"] == [0.0, 0.0] and r["exact_two_sided_mcnemar_p"] == 1
    assert "does not establish equivalence" in r["interpretation"]


def test_confirmation_reserved_seeds_order_and_inputs_unchanged():
    d = confirmation()
    s = screen()
    validate_design(load_protocol(), d, "confirmation")
    assert d["sampling_seeds"] == list(range(904501, 904513))
    assert not set(d["sampling_seeds"]) & set(s["sampling_seeds"])
    assert (
        d["nominal_scene"] == s["nominal_scene"]
        and d["adapter_profile"] == s["adapter_profile"]
    )
    assert d["execution_phases"][1]["policy_ids"] == list(
        reversed(d["execution_phases"][0]["policy_ids"])
    )
    assert sum(len(b["trials"]) for b in blocks(d)) == 24


def test_confirmation_refuses_forged_selection():
    d = screen()
    rows = grid(d)
    s = evaluate_selection(load_protocol(), d, rows, "screen")
    s.update(campaign_sha256="right", trials=rows)
    with pytest.raises(ValueError, match="hash"):
        derive_confirmation(
            load_protocol(), d, s, screen_sha="wrong", selection_sha="x"
        )
    s["selected_ids"].reverse()
    with pytest.raises(ValueError, match="ranking"):
        derive_confirmation(
            load_protocol(), d, s, screen_sha="right", selection_sha="x"
        )


@pytest.mark.parametrize(
    "field", ["adapter_profile", "runtime_hardware", "nominal_scene"]
)
def test_shared_rehash_cannot_legitimize_changed_parent_inputs(field):
    d = screen()
    if field == "adapter_profile":
        d[field]["placement_release"]["finish_after_release"] = True
    if field == "runtime_hardware":
        d[field]["effective_model"]["safety"]["wrench_limit_N"] = 1000
    if field == "nominal_scene":
        d[field]["waffle"]["center"][0] -= 0.01
    d["anchor_shared_configuration_sha256"] = shared_digest(d)
    with pytest.raises(ValueError, match="(mismatch|differ)"):
        validate_design(load_protocol(), d, "screen")


def test_execution_cannot_bypass_pending_gate():
    with pytest.raises(ValueError, match="stage-gate-audit"):
        verify_execution_gate(screen(), None, Path("/frozen"))


def test_external_analyzer_and_old_cli_compatibility(tmp_path):
    d = screen()
    args = Namespace(
        source=Path(d["runtime_contract"]["source"]),
        evidence=Path("/evidence"),
        output=tmp_path,
        port=7799,
        hardware_config=Path("/hardware"),
        server_python=Path("/live/.venv/bin/python"),
    )
    command = simulation_command(
        args,
        d,
        d["policies"][3],
        d["conditions"][0],
        904401,
        tmp_path / "trial",
        Path("/robot"),
    )
    old_tree = ast.parse(
        (ROOT / "tests/fixtures/teacher_anchor_v1/run_waffles.py.txt").read_text()
    )
    allowed = {
        arg.value
        for node in ast.walk(old_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value.startswith("--")
    }
    assert {x for x in command if x.startswith("--")} <= allowed
    assert "--policy-latency" not in command
    assert command[command.index("--gel-contact-coverage") + 1] == "manifold_patch"
    assert command[command.index("--wrist") + 1] == "contact_proxy"
    assert analysis_command(args, tmp_path / "campaign.json")[1] == str(
        ROOT / "tools/sim/analyze_policy_campaign.py"
    )


def passed_gate_fixture(tmp_path, monkeypatch):
    from tools.sim import teacher_anchor_design as module

    d = screen()
    real_digest = module.digest
    pinned = {
        x["path"]: x["sha256"] for x in d["runtime_contract"]["source_and_input_hashes"]
    }
    monkeypatch.setattr(
        module, "digest", lambda p: pinned.get(str(p)) or real_digest(p)
    )
    payloads = {
        "historical_controls": {},
        "first_divergence_review": {},
        "mechanics_metrics": {
            "thresholds": d["thresholds"],
            "valid_for_scoring": False,
            "invalid_reasons": ["run_mode_is_not_policy"],
            "outcomes": {"full_task": True},
            "placement_support": {"required": True, "verified_placement": True},
        },
        "mechanics_run": {
            "mode": "command_replay",
            "object_dynamics": {
                "rigid_body_dynamic": True,
                "kinematic": False,
                "attachments": [],
                "pose_writes_after_initialization": 0,
            },
            "command_replay": {
                "path": str(
                    Path(d["runtime_contract"]["source"]).parent
                    / "runs/teacher_pick_place_v1/campaign/rollouts"
                    / load_protocol()["anchor_reference"]["case"]
                    / "execution_trace.jsonl"
                ),
                "sha256": "historical_commands",
            },
        },
    }
    pinned[payloads["mechanics_run"]["command_replay"]["path"]] = "historical_commands"
    evidence = {}
    for name, value in payloads.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        evidence[name] = {"path": str(path), "sha256": real_digest(path)}
    audit = {
        "status": "passed",
        "protocol_sha256": d["protocol"]["sha256"],
        "amendment_sha256": d["stage_gate"]["amendment_sha256"],
        "checks": dict.fromkeys(
            [
                "all_four_historical_controls_preserved",
                "source_inputs_match_anchor",
                "first_divergence_review_complete",
                "timing_variance_documented",
                "mechanics_integrity_passed",
                "fresh_screen_not_used_to_approve_gate",
            ],
            True,
        ),
        "mechanics_reproduction_kind": "command_replay",
        "evidence": evidence,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(audit))
    return d, path, audit


def test_mechanics_command_replay_opens_gate_without_a_policy_win(
    tmp_path, monkeypatch
):
    d, path, _ = passed_gate_fixture(tmp_path, monkeypatch)
    gate = verify_execution_gate(d, path, Path(d["runtime_contract"]["source"]))
    assert gate["mechanics_reproduction_kind"] == "command_replay"
    assert "no policy success" in gate["interpretation"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_review",
        "broken_hash",
        "fake_policy",
        "other_invalid_reason",
        "threshold",
    ],
)
def test_gate_rejects_missing_provenance_or_invalid_reproduction(
    tmp_path, monkeypatch, change
):
    from tools.sim.teacher_anchor_design import digest

    d, path, audit = passed_gate_fixture(tmp_path, monkeypatch)
    if change == "missing_review":
        audit["checks"]["first_divergence_review_complete"] = False
    if change == "broken_hash":
        audit["evidence"]["mechanics_metrics"]["sha256"] = "wrong"
    if change == "fake_policy":
        audit["mechanics_reproduction_kind"] = "policy"
    if change in (
        "other_invalid_reason",
        "threshold",
        "animated_object",
        "wrong_replay_mode",
    ):
        p = Path(audit["evidence"]["mechanics_metrics"]["path"])
        m = json.loads(p.read_text())
        if change == "other_invalid_reason":
            m["invalid_reasons"].append("physics_clock_mismatch")
        else:
            m["thresholds"]["lift_height_m"] = 0.0001
        p.write_text(json.dumps(m))
        audit["evidence"]["mechanics_metrics"]["sha256"] = digest(p)
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        verify_execution_gate(d, path, Path(d["runtime_contract"]["source"]))


@pytest.mark.parametrize("change", ["animated_object", "wrong_mode", "wrong_commands"])
def test_command_replay_gate_requires_free_body_and_exact_historical_commands(
    tmp_path, monkeypatch, change
):
    from tools.sim.teacher_anchor_design import digest

    d, path, audit = passed_gate_fixture(tmp_path, monkeypatch)
    p = Path(audit["evidence"]["mechanics_run"]["path"])
    run = json.loads(p.read_text())
    if change == "animated_object":
        run["object_dynamics"]["pose_writes_after_initialization"] = 1
    if change == "wrong_mode":
        run["mode"] = "policy"
    if change == "wrong_commands":
        run["command_replay"]["sha256"] = "different"
    p.write_text(json.dumps(run))
    audit["evidence"]["mechanics_run"]["sha256"] = digest(p)
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        verify_execution_gate(d, path, Path(d["runtime_contract"]["source"]))


def test_confirmation_cli_refuses_overwrite_before_reading_or_rescoring(
    tmp_path, monkeypatch
):
    from tools.sim import freeze_teacher_anchor_confirmation as module

    target = tmp_path / "confirmation.json"
    target.write_text("preserved")
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--protocol",
            "/missing",
            "--screen",
            "/missing",
            "--selection",
            "/missing",
            "--out",
            str(target),
        ],
    )
    with pytest.raises(FileExistsError):
        module.main()
    assert target.read_text() == "preserved"
