#!/usr/bin/env python3
"""Independent reserved-seed raw rescore and winner-gate arithmetic; CPU only."""

import argparse
import importlib
import json
import math
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_completed_screen import (
    DRIVER,
    PROTOCOL_SHA,
    REVIEW,
    REVIEW_SHA,
    ROOT,
    read,
    sha,
)

CAMPAIGN_SHA = "fe62f45fcaf2e71e4df890ff7303bf70a2b49e35fc2a745f6f7492a5f200b855"
SCREEN_AUDITOR_SHA = "d8fc6a03617ac43f9d22c41be9f4f92334966c15231923c52c278a25703e916f"


def paired_arithmetic(a, b, rule):
    a, b = np.asarray(a, int), np.asarray(b, int)
    if a.shape != (12,) or b.shape != (12,) or not np.isin(np.r_[a, b], [0, 1]).all():
        raise ValueError("Exactly twelve binary paired seeds required")
    rng = np.random.default_rng(rule["bootstrap_seed"])
    indexes = rng.integers(0, 12, (rule["bootstrap_replicates"], 12))
    delta = a - b
    alpha = (1 - rule["confidence_level"]) / 2
    bounds = np.quantile(delta[indexes].mean(axis=1), [alpha, 1 - alpha])
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    n = a_only + b_only
    p = (
        min(
            1, 2 * sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / (2**n)
        )
        if n
        else 1.0
    )
    return {
        "difference_a_minus_b": float(delta.mean()),
        "interval": bounds.tolist(),
        "discordant_a_only": a_only,
        "discordant_b_only": b_only,
        "exact_two_sided_mcnemar_p": p,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve existing independent confirmation audit")
    if sha(Path(__file__).with_name("audit_completed_screen.py")) != SCREEN_AUDITOR_SHA:
        raise RuntimeError("Pinned shared read-only utility changed")
    transition = read(ROOT / "transition/final_status.json")
    progress = read(ROOT / "confirmation/progress.json")
    if (
        transition["status"] != "study_complete"
        or progress["status"] != "all_trials_completed"
        or len(progress["trials"]) != 24
        or any(
            v.get("status") != "completed"
            or v.get("exit_code") != 0
            or v.get("reused", False)
            for v in progress["trials"].values()
        )
    ):
        raise RuntimeError("Wait for completed original24 and final transition scoring")
    if sha(REVIEW / "review_manifest.json") != REVIEW_SHA:
        raise RuntimeError("Review manifest changed")
    for path, expected in read(REVIEW / "review_manifest.json")[
        "source_sha256"
    ].items():
        if sha(REVIEW / path) != expected:
            raise RuntimeError(f"Review module changed: {path}")
    sys.path.insert(0, str(REVIEW))
    helper = importlib.import_module("tools.sim.teacher_anchor_compare")
    campaign = ROOT / "confirmation_campaign.json"
    protocol_path = DRIVER / "docs/results/teacher_success_anchor_v5/protocol.json"
    if sha(campaign) != CAMPAIGN_SHA or sha(protocol_path) != PROTOCOL_SHA:
        raise RuntimeError("Campaign/protocol changed")
    design, protocol = read(campaign), read(protocol_path)
    helper.validate_design(protocol, PROTOCOL_SHA, design, "confirmation")
    primary = read(ROOT / "confirmation/selection/selection.json")
    screen = read(ROOT / "screen/selection/selection.json")
    if (
        primary["status"] != "complete_valid_matched_stage"
        or primary["campaign_sha256"] != CAMPAIGN_SHA
        or primary["protocol_sha256"] != PROTOCOL_SHA
        or primary["stage"] != "confirmation"
        or sha(ROOT / "confirmation/campaign_snapshot.json") != CAMPAIGN_SHA
        or design["sampling_seeds"] != list(range(904501, 904513))
        or [p["id"] for p in design["policies"]] != screen["selected_ids"]
    ):
        raise RuntimeError("Invalid stage or changed reserved grid")
    with tempfile.TemporaryDirectory(prefix="v5_independent_confirmation_") as tmp:
        fresh = helper.collect(
            ROOT / "confirmation/rollouts", design, CAMPAIGN_SHA, Path(tmp)
        )
    originals = {
        (r["policy_id"], r["condition_id"], r["sampling_seed"]): r
        for r in primary["trials"]
    }
    errors = []
    if len(fresh) != 24 or len(originals) != 24:
        errors.append("Expected24 distinct raw and selected trials")
    for row in fresh:
        key = row["policy_id"], row["condition_id"], row["sampling_seed"]
        for field, value in row.items():
            if originals.get(key, {}).get(field) != value:
                errors.append({"trial": list(key), "field": field})
        if row.get("valid_for_selection") is not True:
            errors.append({"trial": list(key), "invalid": row.get("invalid_reasons")})
    totals = []
    candidate_order = {p["id"]: i for i, p in enumerate(protocol["candidates"])}
    for policy in design["policies"]:
        rows = [r for r in fresh if r["policy_id"] == policy["id"]]
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
        ] + [count["pre_placement_safety_stop"], candidate_order[policy["id"]]]
        totals.append(
            {
                "policy_id": policy["id"],
                "trials": len(rows),
                **count,
                "ranking_key": ranking_key,
                "terminal_stop_reasons": dict(
                    Counter(r["stop_reason"] or "horizon_no_stop" for r in rows)
                ),
                "terminal_safety_events_overlapping": dict(
                    Counter(e for r in rows for e in r["terminal_safety_events"])
                ),
                "native_latency_episode_means_s": [
                    r["native_inference_latency_s"]["mean"] for r in rows
                ],
                "delivery_latency_episode_means_s": [
                    r["delivery_latency_s"]["mean"] for r in rows
                ],
            }
        )
    ranked = sorted(totals, key=lambda r: r["ranking_key"])
    for new, old in zip(ranked, primary["ranking"]):
        if any(
            new[k] != old[k]
            for k in (*helper.COUNTS, "policy_id", "trials", "ranking_key")
        ):
            errors.append({"ranking_total": new["policy_id"]})
    by_key = {(r["policy_id"], r["sampling_seed"]): r for r in fresh}
    vectors = [
        [
            int(by_key[row["policy_id"], s]["strict_full_place"])
            for s in design["sampling_seeds"]
        ]
        for row in ranked
    ]
    rule = protocol["confirmation_rule"]
    paired = paired_arithmetic(*vectors, rule)
    if any(
        primary["paired_comparison"]["estimate"].get(k) != v for k, v in paired.items()
    ):
        errors.append("Independent paired bootstrap/McNemar arithmetic differs")
    leader, other = ranked
    gates = {
        "physical_place_at_least_8_of_12": leader["strict_full_place"]
        >= rule["minimum_physical_full_tasks"],
        "paired_95pct_lower_bound_positive": paired["interval"][0] > 0,
        "exact_two_sided_p_le_0_05": paired["exact_two_sided_mcnemar_p"]
        <= rule["small_sample_guard"]["alpha"],
        "no_increase_in_drops": leader["dropped"] <= other["dropped"],
        "no_increase_in_pre_placement_force_limit_stops": leader[
            "pre_placement_force_limit_stop"
        ]
        <= other["pre_placement_force_limit_stop"],
    }
    winner = leader["policy_id"] if all(gates.values()) else None
    if gates != primary["winner_gates"] or winner != primary["clear_simulator_winner"]:
        errors.append("Independent winner gates differ")
    if winner != transition["events"][-1]["clear_simulator_winner"]:
        errors.append("Transition winner differs from independent raw scoring")
    result = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "raw_trials_rescored": len(fresh),
        "policies": totals,
        "paired": paired,
        "winner_gates": gates,
        "clear_simulator_winner": winner,
        "campaign_sha256": CAMPAIGN_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "selection_sha256": sha(ROOT / "confirmation/selection/selection.json"),
        "transition_final_status_sha256": sha(ROOT / "transition/final_status.json"),
        "trials": fresh,
        "interpretation": "Matched sampling-seed uncertainty at one fixed simulator state; no hardware or varied-start population claim. Degenerate intervals do not establish equivalence.",
        "source_or_process_changes": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
