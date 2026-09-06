#!/usr/bin/env python3
"""Derive reserved teacher confirmation from a frozen protocol and audited screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.sim.select_teacher_candidate import evaluate_selection

SHARED_FIELDS = (
    "horizon_s",
    "delivery_latency_s",
    "early_termination",
    "nominal_scene_source",
    "nominal_scene_source_sha256",
    "nominal_scene",
    "prepared_episode",
    "inference_settings",
    "observation_delay_semantics",
    "thresholds",
    "runtime_hardware",
    "initialization_requirements",
    "post_stop_observation_s",
    "post_stop_observation_semantics",
    "adapter_profile",
)


def shared_configuration_sha256(screen):
    payload = {k: screen[k] for k in SHARED_FIELDS}
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_confirmation(
    protocol, screen, selection, *, protocol_sha, screen_sha, selection_sha
):
    """Validate all links and rederive ranking from the selected report's rows."""
    if protocol.get("status") != "frozen" or screen.get("status") != "frozen":
        raise ValueError("Parent protocol and screen must be frozen")
    link = screen.get("protocol", {})
    if link.get("sha256") != protocol_sha or link.get("stage") != "screen":
        raise ValueError("Screen parent hash/stage mismatch")
    if protocol.get("shared_configuration_sha256") != shared_configuration_sha256(
        screen
    ):
        raise ValueError("Shared configuration differs from frozen parent digest")
    for field in (
        "horizon_s",
        "delivery_latency_s",
        "adapter_profile",
        "thresholds",
        "initialization_requirements",
    ):
        if screen.get(field) != protocol.get(field):
            raise ValueError(f"Shared parent field mismatch: {field}")
    declared = protocol["screen"]
    if (
        screen["conditions"] != declared["conditions"]
        or screen["sampling_seeds"] != declared["sampling_seeds"]
        or screen["policies"] != protocol["candidates"]
        or [p["id"] for p in screen["policies"]] != declared["candidates"]
        or screen["planned_counts"]["total"] != declared["trials"]
    ):
        raise ValueError("Screen grid/candidates differ from frozen parent")
    if (
        selection.get("campaign_sha256") != screen_sha
        or selection.get("stage") != "screen"
    ):
        raise ValueError("Selection screen hash/stage mismatch")
    if selection.get("status") != "complete_valid_matched_stage":
        raise ValueError("Selection requires a complete valid matched screen")
    recomputed = evaluate_selection(screen, selection.get("trials", []), "screen")
    chosen = selection.get("selected_ids", [])
    if (
        recomputed["status"] != "complete_valid_matched_stage"
        or len(chosen) != 2
        or len(set(chosen)) != 2
        or chosen != recomputed["selected_ids"]
    ):
        raise ValueError("Selected IDs do not match complete valid screen ranking")
    available = {p["id"]: p for p in protocol["candidates"]}
    if not set(chosen) <= set(available):
        raise ValueError("Selected candidates absent from frozen parent")
    confirm = protocol["confirmation"]
    conditions, seeds = confirm["conditions"], confirm["sampling_seeds"]
    if (
        len(conditions) != 6
        or len({c["id"] for c in conditions}) != 6
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or confirm["trials"] != 24
        or conditions[0]["family"] != "nominal"
    ):
        raise ValueError(
            "Confirmation must contain six starts and two seeds, 24 trials"
        )
    if {c["id"] for c in conditions} & {c["id"] for c in screen["conditions"]} or set(
        seeds
    ) & set(screen["sampling_seeds"]):
        raise ValueError("Confirmation starts and seeds must be reserved")
    screen_initials = {c["initial_state"]["sha256"] for c in screen["conditions"]}
    if screen_initials & {c["initial_state"]["sha256"] for c in conditions}:
        raise ValueError("Confirmation reuses a discovery initial-state payload")
    result = copy.deepcopy(screen)
    result.update(
        campaign_id=protocol["protocol_id"] + "_confirmation",
        status="frozen",
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        pending_inputs=[],
        freeze_authorization="Mechanical derivation from predeclared frozen parent and complete valid screen; no new numerical parameters.",
        sampling_seeds=copy.deepcopy(seeds),
        conditions=copy.deepcopy(conditions),
        policies=[copy.deepcopy(available[p]) for p in chosen],
        primary_policy_order=list(chosen),
        secondary_policy_order=[],
        planned_counts={
            "conditions": 6,
            "seeds_per_condition": 2,
            "per_policy": 12,
            "primary": 24,
            "secondary": 0,
            "total": 24,
        },
        execution_order=confirm["execution"],
        execution_phases=[
            {
                "id": f"confirmation_start_{i + 1}",
                "policy_ids": chosen if i % 2 == 0 else list(reversed(chosen)),
                "condition_ids": [c["id"]],
            }
            for i, c in enumerate(conditions)
        ],
        protocol={
            "path": screen["protocol"].get("path"),
            "sha256": protocol_sha,
            "stage": "confirmation",
            "status": "frozen_parent_verified",
        },
        confirmation_provenance={
            "protocol_sha256": protocol_sha,
            "screen_campaign_sha256": screen_sha,
            "screen_selection_sha256": selection_sha,
            "selected_ids": chosen,
            "shared_configuration_sha256": shared_configuration_sha256(screen),
        },
        development_split_note="Six parent-reserved measured starts and fresh matched seeds; simulator-study confirmation, not proven training-unseen data.",
    )
    if shared_configuration_sha256(result) != protocol["shared_configuration_sha256"]:
        raise AssertionError("Confirmation changed shared numerical configuration")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "screen", "selection", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    result = derive_confirmation(
        json.loads(args.protocol.read_text()),
        json.loads(args.screen.read_text()),
        json.loads(args.selection.read_text()),
        protocol_sha=digest(args.protocol),
        screen_sha=digest(args.screen),
        selection_sha=digest(args.selection),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": digest(args.out),
                "selected_ids": result["primary_policy_order"],
                "trials": 24,
            }
        )
    )


if __name__ == "__main__":
    main()
