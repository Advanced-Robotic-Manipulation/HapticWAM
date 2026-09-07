#!/usr/bin/env python3
"""Independent read-only raw rescore of the completed v5 screen.

Uses the pinned CPU review layer, writes only a new audit file and temporary
score files. Never changes primary scores, launches a policy, or selects early.
"""

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v5"
REVIEW = BASE / "review_teacher_anchor_v5"
DRIVER = BASE / "source_teacher_anchor_driver_v5"
SCREEN_SHA = "3392e9f703ef008f2a16d6c133870192ddb1642526ef7adb3702b2e8a6c9b9c5"
PROTOCOL_SHA = "f67fe79a5267c6d253d7d90f67f2a367c8899f030959c8f42fa714d072ab7ac2"
REVIEW_SHA = "25e342f304ab0a3b81ffcb5ca8cd24dd4657cf1560aef2e0e379429dd8e3d45f"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve existing independent audit")
    progress = read(ROOT / "screen/progress.json")
    if (
        progress["status"] != "all_trials_completed"
        or len(progress["trials"]) != 24
        or any(
            v.get("status") != "completed"
            or v.get("exit_code") != 0
            or v.get("reused", False)
            for v in progress["trials"].values()
        )
    ):
        raise RuntimeError("Wait for all24 original completed cases")
    manifest = read(REVIEW / "review_manifest.json")
    if sha(REVIEW / "review_manifest.json") != REVIEW_SHA:
        raise RuntimeError("Review manifest changed")
    for path, expected in manifest["source_sha256"].items():
        if sha(REVIEW / path) != expected:
            raise RuntimeError(f"Review module changed: {path}")
    sys.path.insert(0, str(REVIEW))
    helper = importlib.import_module("tools.sim.teacher_anchor_compare")
    campaign = ROOT / "screen/campaign_snapshot.json"
    protocol_file = DRIVER / "docs/results/teacher_success_anchor_v5/protocol.json"
    if sha(campaign) != SCREEN_SHA or sha(protocol_file) != PROTOCOL_SHA:
        raise RuntimeError("Screen/protocol binding changed")
    design, protocol = read(campaign), read(protocol_file)
    helper.validate_design(protocol, PROTOCOL_SHA, design, "screen")
    path = ROOT / "screen/selection/selection.json"
    selected = read(path)
    if selected.get("status") != "complete_valid_matched_stage":
        raise RuntimeError("Authoritative selection not completed/valid")
    with tempfile.TemporaryDirectory(prefix="v5_independent_screen_") as temp:
        fresh = helper.collect(ROOT / "screen/rollouts", design, SCREEN_SHA, Path(temp))
    before = {
        (r["policy_id"], r["condition_id"], r["sampling_seed"]): r
        for r in selected["trials"]
    }
    errors = []
    if len(fresh) != 24 or len(before) != 24:
        errors.append("Expected24 distinct fresh and recorded rows")
    for row in fresh:
        key = row["policy_id"], row["condition_id"], row["sampling_seed"]
        original = before.get(key, {})
        for field, value in row.items():
            if original.get(field) != value:
                errors.append({"trial": list(key), "field": field})
        if row.get("valid_for_selection") is not True:
            errors.append({"trial": list(key), "invalid": row.get("invalid_reasons")})
    totals = []
    for order, candidate in enumerate(protocol["candidates"]):
        rows = [r for r in fresh if r["policy_id"] == candidate["id"]]
        count = {k: sum(bool(r[k]) for r in rows) for k in helper.COUNTS}
        ranking_key = [
            -count[k]
            for k in (
                "strict_full_place",
                "final_supported_placement",
                "clean_place",
                "lifted",
                "acquired",
            )
        ] + [count["pre_placement_safety_stop"], order]
        def mean(name, data=rows):
            values = [r[name]["mean"] for r in data if r[name]["mean"] is not None]
            return sum(values) / len(values) if values else None
        totals.append(
            {
                "policy_id": candidate["id"],
                "trials": len(rows),
                **count,
                "ranking_key": ranking_key,
                "terminal_stop_reasons": dict(
                    Counter(r["stop_reason"] or "horizon_no_stop" for r in rows)
                ),
                "terminal_safety_events_overlapping": dict(
                    Counter(e for r in rows for e in r["terminal_safety_events"])
                ),
                "mean_of_episode_native_latency_s": mean("native_inference_latency_s"),
                "mean_of_episode_delivery_latency_s": mean("delivery_latency_s"),
                "mean_of_episode_wall_latency_s": mean("inference_wall_time_s"),
            }
        )
    ranked = sorted(totals, key=lambda row: row["ranking_key"])
    ids = [r["policy_id"] for r in ranked[:2]]
    if ids != selected["selected_ids"]:
        errors.append("Independent frozen-tier ranking differs")
    for fresh_total, original in zip(ranked, selected["ranking"]):
        if any(
            fresh_total[k] != original[k]
            for k in (*helper.COUNTS, "policy_id", "ranking_key", "trials")
        ):
            errors.append({"ranking_total": fresh_total["policy_id"]})
    derived_path = ROOT / "confirmation_campaign.json"
    derived = read(derived_path)
    helper.validate_design(protocol, PROTOCOL_SHA, derived, "confirmation")
    if (
        derived["sampling_seeds"] != list(range(904501, 904513))
        or [p["id"] for p in derived["policies"]] != ids
        or derived["confirmation_provenance"]["screen_selection_sha256"] != sha(path)
        or any(derived[k] != design[k] for k in helper.SHARED)
    ):
        errors.append("Derived confirmation seed/selection/shared-profile mismatch")
    transition = read(ROOT / "transition/ledger.json")
    summary = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "screen_campaign_sha256": SCREEN_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "review_manifest_sha256": REVIEW_SHA,
        "selection_sha256": sha(path),
        "confirmation_campaign_sha256": sha(derived_path),
        "raw_trials_rescored": len(fresh),
        "errors": errors,
        "policies": totals,
        "independent_selected_ids": ids,
        "confirmation_seed_count": 12,
        "transition_status_at_audit": transition["status"],
        "screen_verdict": "All24 valid matched screen cases are eligible only for candidate selection; no screen winner is declared.",
        "limitations": [
            "A fixed-anchor screen ranking is selection-biased; reserved confirmation establishes the separate winner gates.",
            "Terminal-event counts may overlap; tactile-force-limit labels include depth. Later stops do not erase physical placement.",
            "Native and delivery-inclusive latency use captured per-plan diagnostics; wall inference time is separate.",
            "No raw video, source, controller, threshold, primary score, model or process was changed by this audit.",
        ],
        "trials": fresh,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status",
                    "raw_trials_rescored",
                    "errors",
                    "policies",
                    "independent_selected_ids",
                    "transition_status_at_audit",
                )
            },
            indent=2,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
