#!/usr/bin/env python3
"""Guarded compact classification after all 32+24 corrected-study cases finish.

Keep this file beside audit_first_start.py on compute3. Prints JSON only;
never writes raw trial folders, imports Isaac, runs inference, or ranks policies.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import audit_first_start as audit
import numpy as np
import yaml


def compact_contact(contact):
    return {
        key: contact[key]
        for key in [
            "actor_path",
            "filter_path",
            "normal_force_magnitude_n",
            "normal_wrench_world_n_nm",
            "loaded_centroid_world_m",
        ]
    }


def compact_case(result, folder):
    stop = result["first_safety_stop"]
    all_first = result["per_body_first_force_gt_point1_n_s"]
    first_contact = min(all_first.values(), default=None)
    contact_rows = []
    if stop:
        contact_rows = result["wrist_snapshots"]["stop"]["contacts"]
    dominant = max(
        contact_rows, key=lambda c: c["normal_force_magnitude_n"], default=None
    )
    # Precontact tracking is descriptive, not a new success/safety gate.
    precontact = []
    for row in audit.read_jsonl(folder / "execution_trace.jsonl"):
        if row["stopped"] or row.get("accepted_tcp") is None:
            continue
        if first_contact is not None and row["t"] >= first_contact:
            continue
        precontact.append(
            float(
                np.linalg.norm(
                    np.asarray(row["accepted_tcp"][:3]) - row["measured_tcp"][:3]
                )
            )
        )
    metrics = result["metrics"]
    outcomes = metrics["outcomes"]
    furthest = next(
        (
            stage
            for stage in [
                "full_task",
                "released_in_bin",
                "carried",
                "lifted",
                "acquired",
            ]
            if outcomes[stage]
        ),
        "no_acquisition",
    )
    return {
        "case_id": result["case_id"],
        "directory": result["directory"],
        "valid_for_scoring": metrics["valid_for_scoring"],
        "invalid_reasons": metrics.get("invalid_reasons", []),
        "outcomes": outcomes,
        "event_times_s": metrics["event_times_s"],
        "furthest_object_stage": furthest,
        "object": metrics["object"],
        "reach_diagnostic": metrics["reach_diagnostic"],
        "execution_end_s": result["execution_end_s"],
        "safety_stop": None
        if stop is None
        else {
            "t_s": stop["t"],
            "reason": stop["stop_reason"],
            "events": stop["diagnostics"]["safety_events"],
            "measured_tcp": stop["measured_tcp"],
            "requested_tcp": stop["requested_tcp"],
            "measured_closure": stop["measured_gripper"][0],
            "commanded_closure": stop["gripper_command"],
        },
        "contacts_at_first_stop": [compact_contact(c) for c in contact_rows],
        "dominant_contact_at_first_stop": None
        if dominant is None
        else compact_contact(dominant),
        "first_external_contact_gt_point1_n_s": first_contact,
        "per_pair_first_external_contact_gt_point1_n_s": all_first,
        "per_pair_peak_contacts_before_stop": [
            {"t_s": c["t_s"], **compact_contact(c)}
            for c in result["per_body_peak_contact_before_stop"]
        ],
        "maximum_precontact_tracking_error_m": max(precontact, default=None),
        "tracking": {
            k: v
            for k, v in result["tracking"].items()
            if k != "maximum_tracking_error_row"
        },
        "wrench_guard": result["wrench_subguard_reconstruction"],
        "wrench_guard_stop_matches_log": result["wrench_subguard_stop_matches_log"],
        "gel_peak_before_stop_n": result["gel"]["peak_per_pad_before_stop_n"],
        "gel_peak_rejection_budgets": {
            key: {
                "t_s": value["t_s"],
                "normal_force_n": value["normal_force_n"],
                "gel_force_n": value["gel_force_n"],
                "pad_rejection_budgets_n": value["pad_rejection_budgets_n"],
            }
            for key, value in result["gel"]["per_body_peaks"].items()
        },
        "latch_observed": result["latch_observed"],
        "delivered_veto_action_counts": result["delivered_veto_action_counts"],
        "wrist_input_audit": result["input_audit"],
        "input_sha256": result["input_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=audit.ROOT.parent)
    parser.add_argument("--screen-root", type=Path)
    parser.add_argument("--confirmation-root", type=Path)
    args = parser.parse_args()
    marker = args.study_root / "study_complete.json"
    if not marker.is_file():
        raise SystemExit("Not ready: study_complete.json is absent; no cases analyzed.")
    roots = [
        args.screen_root or args.study_root / "screen",
        args.confirmation_root or args.study_root / "confirmation",
    ]
    studies = []
    for root, expected_count in zip(roots, [32, 24]):
        design = json.loads((root / "campaign_snapshot.json").read_text())
        progress = json.loads((root / "progress.json").read_text())
        expected = [
            f"{policy['id']}__{condition['id']}__seed{seed}"
            for policy in design["policies"]
            for condition in design["conditions"]
            for seed in design["sampling_seeds"]
        ]
        if len(expected) != expected_count or set(expected) != set(progress["trials"]):
            raise SystemExit(
                f"Not ready: {root.name} does not have exactly the expected identities."
            )
        if progress.get("status") != "all_trials_completed":
            raise SystemExit(f"Not ready: {root.name} is not all_trials_completed.")
        if any(
            progress["trials"][case].get("status") != "completed" for case in expected
        ):
            raise SystemExit(f"Not ready: {root.name} contains an incomplete case.")
        studies.append((root, design, progress, expected))
    hardware_path = (
        audit.BASE / "runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml"
    )
    hardware = yaml.safe_load(hardware_path.read_text())
    results, manifests = [], []
    for root, design, progress, cases in studies:
        audit.ROOT = root
        manifests.append(
            {
                "root": str(root),
                "campaign_snapshot_sha256": audit.sha(root / "campaign_snapshot.json"),
                "expected_case_count": len(cases),
            }
        )
        for case in cases:
            folder = root / "rollouts" / case
            if progress["trials"][case].get("exit_code") != 0:
                results.append(
                    {
                        "case_id": case,
                        "phase": root.name,
                        "valid_for_scoring": False,
                        "diagnostic_status": "completed_nonzero_exit_preserved_as_explicit_failure",
                        "progress_entry": progress["trials"][case],
                    }
                )
                continue
            row = audit.summarize_case(case, design, progress, hardware)
            results.append({"phase": root.name, **compact_case(row, folder)})
    print(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "study_completion_marker_sha256": audit.sha(marker),
                "campaigns": manifests,
                "case_count": len(results),
                "trials": results,
                "scope": "Post-completion descriptive classification of all corrected screen and confirmation cases. No ranking, score changes, superseded pooling, or simulator calls.",
                "definitions": {
                    "first_external_contact": "First sampled >0.1N sum of positive normal-force magnitudes for any instrumented distal actor/environment pair; diagnostic threshold only.",
                    "precontact_tracking": "Maximum accepted-minus-measured TCP position error on active executor rows strictly before first instrumented contact; not a success gate and not proof of no unobserved arm/self collision.",
                    "dominant_contact": "Largest loaded pair at the first safety-stop sample. Association only; the reconstructed wrench guard establishes whether a debounced wrench deviation explains the stop.",
                    "force_limits": "Uncalibrated normal-contact proxy; no gravity/inertia/friction, proximal arm or self-contact coverage. Magnitudes and contact geometry remain uncertain.",
                },
                "source_sha256": {
                    name: audit.sha(audit.SOURCE / name)
                    for name in [
                        "tools/sim/gripper_wrist.py",
                        "tools/sim/gel_contact.py",
                        "phantom/sim/policy_metrics.py",
                        "phantom/deploy/safety.py",
                    ]
                },
                "hardware_sha256": audit.sha(hardware_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
