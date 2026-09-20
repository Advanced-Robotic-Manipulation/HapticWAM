"""CPU-only derivation and execution gates for the frozen v3 success anchor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The frozen study artefacts now live under tests/fixtures/reference/.  The frozen
# records themselves must stay byte-identical, so they still carry the directory
# name they were written with; resolve that name to where the files are today.
RECORDED_PREFIX = "docs/results/"
REFERENCE_ROOT = "tests/fixtures/reference/"


def reference_path(recorded):
    """Map a path recorded in a frozen artefact onto its current location."""
    text = str(recorded)
    if text.startswith(RECORDED_PREFIX):
        return REFERENCE_ROOT + text[len(RECORDED_PREFIX):]
    return text


PROTOCOL = REPO / (REFERENCE_ROOT + "teacher_success_anchor_v3/protocol.json")
PROTOCOL_SHA = "2e02af9dd1a5bf7b9059d1a71d1a8aec1f95618f491a35e50c455c70bf765cd9"
TEMPLATE_SHA = "ce088ff3aefeccdb43a6597f51466060cdba1a8ab7cfdd685509e2b3ca0c191a"
AMENDMENT_SHA = "c61f7d1b2d310686cb0e53aeb1101aba8864f09a388aff289e1534e5b0938312"
SHARED_FIELDS = (
    "nominal_scene",
    "prepared_episode",
    "horizon_s",
    "delivery_latency_s",
    "adapter_profile",
    "thresholds",
    "runtime_hardware",
    "inference_settings",
    "initialization_requirements",
    "post_stop_observation_s",
    "early_termination",
    "runtime_contract",
    "stage_gate",
    "conditions",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def shared_digest(design):
    return payload_digest({k: design[k] for k in SHARED_FIELDS})


def load_protocol(path=PROTOCOL):
    if digest(path) != PROTOCOL_SHA:
        raise ValueError("Unexpected frozen anchor protocol hash")
    return json.loads(Path(path).read_text())


def candidate_policy(candidate):
    return {
        "id": candidate["id"],
        "architecture": "teacher",
        "policy_mode": "teacher",
        "checkpoint": candidate["checkpoint"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "cohort": "primary_conditional_anchor",
        "inference_settings": {
            "use_ema": candidate["weights"] == "EMA",
            "nfe": candidate["nfe"],
            "k_seeds": candidate["k_seeds"],
            "guidance": candidate["guidance"],
            "parity": candidate["parity_fixes"],
            "persistent_noise": candidate["persistent_noise"],
            "task_text": candidate["task_text"],
            "max_play": 10,
        },
    }


def phases(protocol, stage, chosen):
    declared = protocol["stages"]["teacher_screen" if stage == "screen" else stage]
    result = []
    for index, block in enumerate(declared["execution_blocks"]):
        order = block["candidate_order"]
        if stage == "confirmation":
            order = chosen if order == "AB" else list(reversed(chosen))
        result.append(
            {
                "id": f"{stage}_block{index + 1}",
                "policy_ids": list(order),
                "condition_ids": ["successful_anchor"],
                "sampling_seeds": (
                    [block["sampling_seed"]]
                    if stage == "screen"
                    else list(block["sampling_seeds"])
                ),
            }
        )
    return result


def make_screen(protocol, template, dependency, amendment, *, amendment_sha):
    """Derive all numerical inputs from the accepted parent, never a trial outcome."""
    result = copy.deepcopy(template)
    common = protocol["common_inputs"]
    policies = [candidate_policy(c) for c in protocol["candidates"]]
    declared = protocol["stages"]["teacher_screen"]
    result.update(
        campaign_id=protocol["protocol_id"] + "_screen",
        status="frozen",
        frozen_at_utc=amendment["created_at_utc"],
        freeze_authorization="Mechanical derivation of the accepted 64-trial parent; execution remains gated by a separate mechanics audit.",
        nominal_scene=copy.deepcopy(common["fixed_scene"]),
        nominal_scene_source=dependency["launch_dependencies"]["--config"]["path"],
        nominal_scene_source_sha256=dependency["launch_dependencies"]["--config"][
            "sha256"
        ],
        prepared_episode="evidence/fit/ep_waffles_1787395928_000",
        sampling_seeds=list(declared["sampling_seeds"]),
        policies=policies,
        primary_policy_order=[p["id"] for p in policies],
        secondary_policy_order=[],
        conditions=[
            {
                "id": "successful_anchor",
                "family": "nominal",
                "object_offset_m": [0.0, 0.0],
                "friction_scale": 1.0,
                "camera_override": None,
                "observation_delay_s": 0.0,
                "inference_delay_add_s": 0.0,
                "intended": "Exact already-offset historical physical cell; apply no further placement shift.",
            }
        ],
        horizon_s=common["nominal_horizon_s"],
        delivery_latency_s=None,
        adapter_profile=copy.deepcopy(common["original_adapter_profile"]),
        thresholds=copy.deepcopy(common["thresholds"]),
        planned_counts={
            "conditions": 1,
            "seeds_per_condition": 6,
            "per_policy": 6,
            "primary": 24,
            "secondary": 0,
            "total": 24,
        },
        execution_phases=phases(protocol, "screen", []),
        execution_order="One paired sampling-seed block at a time, candidate order counterbalanced exactly as frozen; model reloads and native latency retained.",
        development_split_note="Historical controls and component diagnostics are excluded. Six prospective screen seeds select two candidates; twelve disjoint seeds confirm at this same fixed physical anchor.",
        comparability_requirements=[
            "Same fixed historical scene, initial state, baseline, hardware and frozen v1 runtime for all teachers.",
            "Physical placement is primary; later controller stops and normal completion are separate.",
            "NFE5/K4 differs only in NFE from reference NFE1/K4; native inference latency is part of recipe performance.",
        ],
        protocol={
            "path": RECORDED_PREFIX + "teacher_success_anchor_v3/protocol.json",
            "sha256": PROTOCOL_SHA,
            "stage": "screen",
        },
        stage_gate={
            "amendment_path": RECORDED_PREFIX + "teacher_success_anchor_v3/stage_gate_amendment.json",
            "amendment_sha256": amendment_sha,
            "protocol_sha256": PROTOCOL_SHA,
            "required": True,
        },
        runtime_contract={
            "source": common["source"],
            "hardware_path": common["runtime_hardware"]["path"],
            "robot_usd": common["robot_usd"],
            "source_and_input_hashes": [
                {"path": r["path"], "sha256": r["expected_sha256"], "group": r["group"]}
                for r in dependency["source_and_input_checks"]
            ],
        },
    )
    result["runtime_hardware"]["source"] = common["runtime_hardware"]["path"]
    for key in ("amendments", "preflight_provenance", "execution_order_seed"):
        result.pop(key, None)
    result["analysis"].update(
        primary_outcome="full_task",
        primary_comparison=None,
        exploratory_comparisons=[],
        secondary_comparisons=[],
        weighting="One fixed physical condition with six matched prospective sampling seeds per teacher.",
        bootstrap_seed=20260907,
        paired_interval_method="Generic scenario comparison disabled: tools/sim/select_teacher_anchor.py resamples matched SEED pairs, not the single scene row.",
        reproduction_note="All historical controls and ablations excluded from ranking.",
    )
    result["inference_settings"].update(
        use_ema=True,
        nfe=1,
        k_seeds=4,
        guidance=1.0,
        persistent_noise=True,
        parity=True,
        max_play=10,
        task_text="waffles",
    )
    result["anchor_shared_configuration_sha256"] = shared_digest(result)
    return result


def validate_design(protocol, design, stage):
    if design.get("status") != "frozen" or stage not in ("screen", "confirmation"):
        raise ValueError("Frozen screen or confirmation required")
    if (
        design.get("protocol", {}).get("sha256") != PROTOCOL_SHA
        or design["protocol"].get("stage") != stage
    ):
        raise ValueError("Anchor protocol link/stage mismatch")
    common = protocol["common_inputs"]
    declared = protocol["stages"]["teacher_screen" if stage == "screen" else stage]
    candidates = {c["id"]: candidate_policy(c) for c in protocol["candidates"]}
    ids = [p["id"] for p in design["policies"]]
    if (
        len(ids) != (4 if stage == "screen" else 2)
        or len(set(ids)) != len(ids)
        or not set(ids) <= set(candidates)
    ):
        raise ValueError("Frozen teacher candidate set mismatch")
    if stage == "screen" and ids != declared["candidates"]:
        raise ValueError("Screen candidate order mismatch")
    if design["policies"] != [candidates[p] for p in ids]:
        raise ValueError("Candidate checkpoint/recipe mismatch")
    if design["sampling_seeds"] != declared["sampling_seeds"]:
        raise ValueError("Reserved sampling seeds mismatch")
    for key, value in (
        ("nominal_scene", common["fixed_scene"]),
        ("thresholds", common["thresholds"]),
        ("adapter_profile", common["original_adapter_profile"]),
        ("horizon_s", common["nominal_horizon_s"]),
        ("delivery_latency_s", None),
    ):
        if design.get(key) != value:
            raise ValueError(f"Frozen parent input mismatch: {key}")
    conditions = design["conditions"]
    if len(conditions) != 1 or conditions[0]["id"] != "successful_anchor":
        raise ValueError("One fixed successful-anchor condition required")
    condition = conditions[0]
    if (
        condition["family"] != "nominal"
        or condition["object_offset_m"] != [0, 0]
        or condition["friction_scale"] != 1
        or condition["observation_delay_s"] != 0
        or condition["inference_delay_add_s"] != 0
        or condition.get("camera_override") is not None
        or condition.get("initial_state")
    ):
        raise ValueError("Anchor perturbation/double-offset prohibited")
    if design["execution_phases"] != phases(protocol, stage, ids):
        raise ValueError("Frozen seed/candidate execution order mismatch")
    expected_n = 6 if stage == "screen" else 12
    if design["planned_counts"] != {
        "conditions": 1,
        "seeds_per_condition": expected_n,
        "per_policy": expected_n,
        "primary": 24,
        "secondary": 0,
        "total": 24,
    }:
        raise ValueError("Frozen matched counts mismatch")
    if design["runtime_hardware"]["sha256"] != common["runtime_hardware"]["sha256"]:
        raise ValueError("Frozen hardware identity mismatch")
    if design["runtime_contract"]["source"] != common["source"]:
        raise ValueError("Legacy v1 runtime required")
    if design.get("anchor_shared_configuration_sha256") != shared_digest(design):
        raise ValueError("Shared input digest mismatch")
    root = PROTOCOL.parent
    files = {
        "reproduction_campaign.json": TEMPLATE_SHA,
        "dependency_audit.json": protocol["dependency_audit"]["sha256"],
        "stage_gate_amendment.json": AMENDMENT_SHA,
    }
    for name, sha in files.items():
        if digest(root / name) != sha:
            raise ValueError(f"Frozen derivation input changed: {name}")
    expected = make_screen(
        protocol,
        json.loads((root / "reproduction_campaign.json").read_text()),
        json.loads((root / "dependency_audit.json").read_text()),
        json.loads((root / "stage_gate_amendment.json").read_text()),
        amendment_sha=AMENDMENT_SHA,
    )
    if any(design[k] != expected[k] for k in SHARED_FIELDS):
        raise ValueError("Shared runtime inputs differ from frozen derivation")


def verify_execution_gate(design, audit_path, source):
    """Require a separately signed-off, evidence-linked mechanics gate before launch."""
    gate = design["stage_gate"]
    validate_design(load_protocol(), design, design["protocol"]["stage"])
    if audit_path is None:
        raise ValueError(
            "Anchor execution requires --stage-gate-audit; no success-hunting retries"
        )
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "passed"
        or audit.get("protocol_sha256") != PROTOCOL_SHA
        or audit.get("amendment_sha256") != gate["amendment_sha256"]
    ):
        raise ValueError("Mechanics gate status/protocol/amendment mismatch")
    required_checks = (
        "all_four_historical_controls_preserved",
        "source_inputs_match_anchor",
        "first_divergence_review_complete",
        "timing_variance_documented",
        "mechanics_integrity_passed",
        "fresh_screen_not_used_to_approve_gate",
    )
    if any(audit.get("checks", {}).get(k) is not True for k in required_checks):
        raise ValueError("Mechanics gate review is incomplete")
    if Path(source).resolve() != Path(design["runtime_contract"]["source"]).resolve():
        raise ValueError("Anchor comparison must use exact frozen legacy v1 source")
    evidence = audit.get("evidence", {})
    required_evidence = {
        "historical_controls",
        "first_divergence_review",
        "mechanics_metrics",
        "mechanics_run",
    }
    if not required_evidence <= set(evidence):
        raise ValueError("Missing mechanics-gate evidence links")
    for name, spec in evidence.items():
        if digest(spec["path"]) != spec["sha256"]:
            raise ValueError(f"Mechanics-gate evidence hash changed: {name}")
    metrics = json.loads(Path(evidence["mechanics_metrics"]["path"]).read_text())
    metrics = metrics.get("metrics", metrics)
    if metrics.get("thresholds") != design["thresholds"]:
        raise ValueError("Mechanics gate thresholds differ from the frozen study")
    support = metrics.get("placement_support", {})
    if not (
        metrics.get("outcomes", {}).get("full_task") is True
        and support.get("verified_placement") is True
        and support.get("required") is True
    ):
        raise ValueError(
            "Mechanics gate requires strict support-verified physical placement"
        )
    kind = audit.get("mechanics_reproduction_kind")
    run = json.loads(Path(evidence["mechanics_run"]["path"]).read_text())
    if kind == "policy":
        if metrics.get("valid_for_scoring") is not True or run.get("mode") != "policy":
            raise ValueError("Invalid policy reproduction")
    elif kind != "command_replay":
        raise ValueError("Mechanics reproduction must be command_replay or policy")
    elif metrics.get("invalid_reasons") not in ([], ["run_mode_is_not_policy"]):
        raise ValueError("Command replay has non-mode scoring integrity failures")
    if kind == "command_replay":
        dynamics = run.get("object_dynamics", {})
        if (
            run.get("mode") != "command_replay"
            or dynamics.get("rigid_body_dynamic") is not True
            or dynamics.get("kinematic") is not False
            or dynamics.get("attachments") != []
            or dynamics.get("pose_writes_after_initialization") != 0
        ):
            raise ValueError(
                "Command replay requires free dynamic object and zero post-initialization pose writes"
            )
        commands = run.get("command_replay", {})
        expected_commands = (
            Path(design["runtime_contract"]["source"]).parent
            / "runs/teacher_pick_place_v1/campaign/rollouts"
            / load_protocol()["anchor_reference"]["case"]
            / "execution_trace.jsonl"
        )
        if commands.get("path") != str(expected_commands) or digest(
            expected_commands
        ) != commands.get("sha256"):
            raise ValueError(
                "Command replay must link unchanged historical accepted commands"
            )
    for spec in design["runtime_contract"]["source_and_input_hashes"]:
        if digest(spec["path"]) != spec["sha256"]:
            raise ValueError(f"Historical source/input changed: {spec['path']}")
    return {
        "path": str(audit_path.absolute()),
        "sha256": digest(audit_path),
        "mechanics_reproduction_kind": kind,
        "evidence": evidence,
        "interpretation": "Command replay, when used, validates mechanics only and earns no policy success.",
    }
