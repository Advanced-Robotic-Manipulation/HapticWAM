import copy
import json
from pathlib import Path

import pytest

from tools.sim.freeze_teacher_confirmation import (
    derive_confirmation,
    main,
    shared_configuration_sha256,
)
from tools.sim.run_policy_campaign import blocks
from tools.sim.select_teacher_candidate import evaluate_selection


def inputs():
    root = Path(__file__).resolve().parents[1]
    parent = json.loads((root / "configs/sim/teacher_v2_protocol.json").read_text())
    screen = json.loads((root / "configs/sim/teacher_v2_screen.json").read_text())
    parent["status"] = screen["status"] = "frozen"
    parent["shared_configuration_sha256"] = shared_configuration_sha256(screen)
    screen["protocol"].update(sha256="parent", stage="screen")
    rows = []
    for p in screen["policies"]:
        for c in screen["conditions"]:
            for seed in screen["sampling_seeds"]:
                rows.append(
                    {
                        "policy_id": p["id"],
                        "condition_id": c["id"],
                        "sampling_seed": seed,
                        "valid_for_selection": True,
                        **dict.fromkeys(
                            (
                                "clean_place",
                                "strict_full_place",
                                "lifted",
                                "acquired",
                                "carried",
                                "released_in_bin",
                                "safety_stop",
                                "actual_stop",
                                "dropped",
                                "wrench_limit_stop",
                                "tactile_force_limit_stop",
                            ),
                            False,
                        ),
                    }
                )
    selection = evaluate_selection(screen, rows, "screen")
    selection.update(campaign_sha256="screen", trials=rows)
    return parent, screen, selection


def derive(values):
    return derive_confirmation(
        *values, protocol_sha="parent", screen_sha="screen", selection_sha="selection"
    )


def test_derived_confirmation_exactly_preserves_shared_parameters_and_balances_order():
    parent, screen, selection = values = inputs()
    result = derive(values)
    assert result["conditions"] == parent["confirmation"]["conditions"]
    assert result["sampling_seeds"] == [903201, 903202]
    assert shared_configuration_sha256(result) == shared_configuration_sha256(screen)
    assert result["primary_policy_order"] == selection["selected_ids"]
    assert sum(len(b["trials"]) for b in blocks(result)) == 24
    assert [p["policy_ids"][0] for p in result["execution_phases"]] == selection[
        "selected_ids"
    ] * 3
    assert result["confirmation_provenance"]["screen_selection_sha256"] == "selection"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p, s, r: p.update(status="draft"),
        lambda p, s, r: s["protocol"].update(sha256="other"),
        lambda p, s, r: r.update(campaign_sha256="other"),
        lambda p, s, r: r.update(stage="confirmation"),
        lambda p, s, r: s["nominal_scene"]["waffle"].update(mass=0.9),
        lambda p, s, r: s["runtime_hardware"].update(sha256="other"),
        lambda p, s, r: r.update(selected_ids=list(reversed(r["selected_ids"]))),
        lambda p, s, r: r["trials"].pop(),
        lambda p, s, r: r["trials"][0].update(valid_for_selection=False),
        lambda p, s, r: p["confirmation"].update(sampling_seeds=s["sampling_seeds"]),
        lambda p, s, r: p["confirmation"]["conditions"][0].update(
            initial_state=copy.deepcopy(s["conditions"][0]["initial_state"])
        ),
    ],
)
def test_changed_parent_screen_selection_or_reserved_inputs_are_rejected(mutation):
    values = inputs()
    mutation(*values)
    with pytest.raises(ValueError):
        derive(values)


def test_cli_refuses_to_overwrite_before_reading_inputs(tmp_path, monkeypatch):
    out = tmp_path / "confirmation.json"
    out.write_text("preserve")
    monkeypatch.setattr(
        "sys.argv",
        [
            "builder",
            "--protocol",
            "missing",
            "--screen",
            "missing",
            "--selection",
            "missing",
            "--out",
            str(out),
        ],
    )
    with pytest.raises(FileExistsError):
        main()
    assert out.read_text() == "preserve"
