#!/usr/bin/env python3
"""Derive reserved fixed-anchor confirmation from a complete matched screen."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_selection = importlib.import_module("tools.sim.select_teacher_anchor")
collect_trials, evaluate_selection = (
    _selection.collect_trials,
    _selection.evaluate_selection,
)
_design = importlib.import_module("tools.sim.teacher_anchor_design")
digest, load_protocol = _design.digest, _design.load_protocol
phases, validate_design = _design.phases, _design.validate_design


def derive_confirmation(protocol, screen, selection, *, screen_sha, selection_sha):
    validate_design(protocol, screen, "screen")
    if (
        selection.get("campaign_sha256") != screen_sha
        or selection.get("stage") != "screen"
    ):
        raise ValueError("Selection screen hash/stage mismatch")
    if selection.get("status") != "complete_valid_matched_stage":
        raise ValueError("Complete valid matched screen required")
    recomputed = evaluate_selection(
        protocol, screen, selection.get("trials", []), "screen"
    )
    chosen = selection.get("selected_ids", [])
    if (
        recomputed["status"] != "complete_valid_matched_stage"
        or chosen != recomputed["selected_ids"]
        or len(chosen) != 2
        or selection.get("ranking") != recomputed["ranking"]
    ):
        raise ValueError("Selection differs from rederived complete-screen ranking")
    result = copy.deepcopy(screen)
    confirm = protocol["stages"]["confirmation"]
    policies = {p["id"]: p for p in screen["policies"]}
    result.update(
        campaign_id=protocol["protocol_id"] + "_confirmation",
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        freeze_authorization="Mechanically derived from complete valid screen and immutable parent; no numerical or ranking changes.",
        sampling_seeds=list(confirm["sampling_seeds"]),
        policies=[copy.deepcopy(policies[p]) for p in chosen],
        primary_policy_order=list(chosen),
        planned_counts={
            "conditions": 1,
            "seeds_per_condition": 12,
            "per_policy": 12,
            "primary": 24,
            "secondary": 0,
            "total": 24,
        },
        execution_phases=phases(protocol, "confirmation", chosen),
        execution_order="Six paired-seed blocks, selected A/B then B/A alternating exactly as frozen.",
        protocol={**screen["protocol"], "stage": "confirmation"},
        confirmation_provenance={
            "protocol_sha256": screen["protocol"]["sha256"],
            "screen_campaign_sha256": screen_sha,
            "screen_selection_sha256": selection_sha,
            "selected_ids": list(chosen),
        },
    )
    result["analysis"]["weighting"] = (
        "One fixed physical condition with twelve disjoint reserved matched sampling seeds per selected teacher."
    )
    validate_design(protocol, result, "confirmation")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite confirmation: {args.out}")
    protocol = load_protocol(args.protocol)
    screen = json.loads(args.screen.read_text())
    selection = json.loads(args.selection.read_text())
    screen_sha, selection_sha = digest(args.screen), digest(args.selection)
    result = derive_confirmation(
        protocol, screen, selection, screen_sha=screen_sha, selection_sha=selection_sha
    )
    # Do not let stale edited report rows choose candidates. Re-audit original raw cases.
    with tempfile.TemporaryDirectory(prefix="anchor_confirmation_audit_") as temporary:
        fresh = collect_trials(
            Path(selection["runs"]), screen, screen_sha, Path(temporary)
        )
        audited = evaluate_selection(protocol, screen, fresh, "screen")
        if any(
            audited.get(k) != selection.get(k)
            for k in ("status", "selected_ids", "ranking")
        ):
            raise ValueError("Raw screen rescore differs from the selected report")
        old_by_key = {
            (r["policy_id"], r["sampling_seed"]): r for r in selection["trials"]
        }
        for row in fresh:
            old = old_by_key[row["policy_id"], row["sampling_seed"]]
            if old.get("input_sha256") != row.get("input_sha256") or old.get(
                "case_sha256"
            ) != row.get("case_sha256"):
                raise ValueError("Raw screen inputs changed since selection")
    result["confirmation_provenance"].update(
        screen_campaign_path=str(args.screen.absolute()),
        screen_selection_path=str(args.selection.absolute()),
        raw_screen_rescore="all24 rows audited; score/ranking/input hashes agree",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "path": str(args.out),
                "sha256": digest(args.out),
                "selected_ids": result["confirmation_provenance"]["selected_ids"],
                "trials": 24,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
