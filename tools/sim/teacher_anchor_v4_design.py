"""Immutable v4 combined-profile derivation; keeps the superseded v3 tools intact."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from tools.sim.teacher_anchor_design import (
    RECORDED_PREFIX,
    REFERENCE_ROOT,
    SHARED_FIELDS,
    candidate_policy,
    digest,
    phases,
    reference_path,
    shared_digest,
)

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / (REFERENCE_ROOT + "teacher_success_anchor_v4/protocol.json")
PROTOCOL_SHA = "29d921b476807e68e129472d4090f47920af52ce60b6dc999d00f7fef820fe36"


def load_protocol(path=PROTOCOL):
    if digest(path) != PROTOCOL_SHA:
        raise ValueError("Unexpected frozen v4 protocol hash")
    return json.loads(Path(path).read_text())


def checked_json(relative_path, sha):
    path = REPO / reference_path(relative_path)
    if digest(path) != sha:
        raise ValueError(f"Frozen v4 derivation input changed: {relative_path}")
    return json.loads(path.read_text())


def make_screen(protocol):
    old = checked_json(
        protocol["supersedes"]["legacy_screen_path"],
        protocol["supersedes"]["legacy_screen_sha256"],
    )
    bridge = checked_json(
        protocol["bridge"]["campaign_path"], protocol["bridge"]["campaign_sha256"]
    )
    audit = checked_json(
        protocol["source_audit"]["path"], protocol["source_audit"]["sha256"]
    )
    if audit["status"] != "passed" or not all(r["match"] for r in audit["rows"]):
        raise ValueError("Corrected source audit failed")
    result = copy.deepcopy(old)
    common = protocol["common_inputs"]
    policies = [candidate_policy(p) for p in protocol["candidates"]]
    result.update(
        campaign_id=protocol["protocol_id"] + "_screen",
        status="frozen",
        frozen_at_utc=protocol["frozen_at_utc"],
        freeze_authorization="Development-informed v4 conditional freeze before combined bridge and all prospective model outcomes. Execution requires2/2 clean bridge trials; no legacy fallback.",
        adapter_profile=copy.deepcopy(bridge["adapter_profile"]),
        policies=policies,
        primary_policy_order=[p["id"] for p in policies],
        nominal_scene=copy.deepcopy(common["fixed_scene"]),
        protocol={
            "path": RECORDED_PREFIX + "teacher_success_anchor_v4/protocol.json",
            "sha256": PROTOCOL_SHA,
            "stage": "screen",
        },
        stage_gate={
            "required": True,
            "protocol_sha256": PROTOCOL_SHA,
            "bridge_campaign_path": protocol["bridge"]["campaign_path"],
            "bridge_campaign_sha256": protocol["bridge"]["campaign_sha256"],
        },
        execution_phases=phases(protocol, "screen", []),
        comparability_requirements=[
            "All prospective teachers use the same bridge-validated corrected profile, fixed historical scene/start and September4 baseline.",
            "Physical placement is primary; later stops and clean completion remain separate.",
            "The source and numerical profile are fixed before bridge outcomes; bridge failure blocks model scoring with no fallback.",
        ],
        development_split_note="V4 amendment used prior gel-only development outcomes. The two combined bridge seeds are reused development inputs and excluded from model ranking; all24 screen and24 confirmation outcomes are prospective.",
        post_stop_observation_semantics="Only actual stops shorten the full60s physics horizon. FINISH prevents replans, returns stopped=False and remains under safety monitoring through the horizon.",
    )
    result["runtime_contract"]["source"] = common["source"]
    result["runtime_contract"]["source_and_input_hashes"] = [
        {"path": r["path"], "sha256": r["sha256"], "group": r["group"]}
        for r in audit["rows"]
    ] + [
        r
        for r in old["runtime_contract"]["source_and_input_hashes"]
        if r["group"] != "source_sha256"
    ]
    result["analysis"]["reproduction_note"] = (
        "All historical controls, components and combined-profile bridge trials are excluded from prospective model ranking."
    )
    result["anchor_shared_configuration_sha256"] = shared_digest(result)
    return result


def validate_design(protocol, design, stage):
    if design.get("status") != "frozen" or stage not in ("screen", "confirmation"):
        raise ValueError("Frozen v4 screen or confirmation required")
    if (
        design.get("protocol", {}).get("sha256") != PROTOCOL_SHA
        or design["protocol"].get("stage") != stage
    ):
        raise ValueError("V4 protocol link/stage mismatch")
    expected = make_screen(protocol)
    candidates = {p["id"]: p for p in expected["policies"]}
    ids = [p["id"] for p in design["policies"]]
    declared = protocol["stages"]["teacher_screen" if stage == "screen" else stage]
    if (
        len(ids) != (4 if stage == "screen" else 2)
        or len(set(ids)) != len(ids)
        or not set(ids) <= set(candidates)
    ):
        raise ValueError("Frozen v4 teacher candidate set mismatch")
    if stage == "screen" and ids != declared["candidates"]:
        raise ValueError("V4 screen candidate order mismatch")
    if design["policies"] != [candidates[p] for p in ids]:
        raise ValueError("V4 checkpoint or inference recipe changed")
    if design["sampling_seeds"] != declared["sampling_seeds"]:
        raise ValueError("V4 prospective/reserved seeds changed")
    if any(design.get(k) != expected[k] for k in SHARED_FIELDS):
        raise ValueError(
            "Shared v4 scene/profile/hardware differs from frozen bridge derivation"
        )
    if design.get("anchor_shared_configuration_sha256") != shared_digest(design):
        raise ValueError("V4 shared-input digest mismatch")
    n = 6 if stage == "screen" else 12
    if design["planned_counts"] != {
        "conditions": 1,
        "seeds_per_condition": n,
        "per_policy": n,
        "primary": 24,
        "secondary": 0,
        "total": 24,
    }:
        raise ValueError("V4 matched planned counts differ")
    if design["execution_phases"] != phases(protocol, stage, ids):
        raise ValueError("V4 frozen execution order changed")


def verify_execution_gate(design, audit_path, source):
    from tools.sim.audit_teacher_anchor_v4_bridge import audit_bridge

    protocol = load_protocol()
    validate_design(protocol, design, design["protocol"]["stage"])
    if audit_path is None:
        raise ValueError(
            "V4 model execution requires --stage-gate-audit; no legacy fallback"
        )
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "passed"
        or audit.get("protocol_sha256") != PROTOCOL_SHA
        or audit.get("bridge_campaign_sha256") != protocol["bridge"]["campaign_sha256"]
    ):
        raise ValueError("V4 bridge gate status or frozen identity mismatch")
    if Path(source).resolve() != Path(protocol["common_inputs"]["source"]).resolve():
        raise ValueError("V4 requires exact corrected source_teacher_v2_delivery")
    with tempfile.TemporaryDirectory(prefix="v4_bridge_gate_check_") as temporary:
        recomputed = audit_bridge(protocol, Path(audit["runs"]), Path(temporary))
    if recomputed["status"] != "passed" or recomputed["trials"] != audit["trials"]:
        raise ValueError("V4 bridge raw evidence changed or no longer passes all gates")
    for spec in design["runtime_contract"]["source_and_input_hashes"]:
        if digest(spec["path"]) != spec["sha256"]:
            raise ValueError(f"V4 corrected source/input hash changed: {spec['path']}")
    return {
        "path": str(audit_path.absolute()),
        "sha256": digest(audit_path),
        "kind": "two_complete_clean_policy_bridges",
        "protocol_sha256": PROTOCOL_SHA,
        "bridge_campaign_sha256": protocol["bridge"]["campaign_sha256"],
        "raw_trial_inputs": [r["input_sha256"] for r in audit["trials"]],
        "interpretation": "Two reused development seeds validate this common profile only; neither enters prospective model selection.",
    }
