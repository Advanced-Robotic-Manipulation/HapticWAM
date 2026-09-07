"""Read-only compact corrected teacher-v2 final-only summary; JSON stdout, no outcome recomputation."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    # No partial ranking: primary screen selection is frozen-tool work only.
    for stage, expected in (("screen", 32), ("confirmation", 24)):
        selection_path = root / stage / "selection/selection.json"
        progress_path = root / stage / "progress.json"
        if not selection_path.is_file() or not progress_path.is_file():
            raise ValueError(
                "Final report waits for both completed stages and selection reports"
            )
        report = json.loads(selection_path.read_text())
        progress = json.loads(progress_path.read_text())
        if (
            report.get("status") != "complete_valid_matched_stage"
            or report.get("available_trials") != expected
            or report.get("expected_trials") != expected
            or report.get("missing_trial_keys")
            or report.get("invalid_trial_keys")
            or progress.get("status") != "all_trials_completed"
        ):
            raise ValueError(
                "Final report refuses incomplete or invalid stage coverage"
            )
        snapshot = root / stage / "campaign_snapshot.json"
        design = json.loads(snapshot.read_text())
        if report.get("campaign_sha256") != digest(snapshot):
            raise ValueError("Selector report does not match stage snapshot")
        if (
            design.get("protocol", {}).get("sha256")
            != "5cba45bfb944d621a2b5ffc2850366aba83c60b0c95f013b3593d37fc5ff8b70"
        ):
            raise ValueError("Not the corrected frozen parent protocol")
        if (
            stage == "screen"
            and digest(snapshot)
            != "5ea36cfb5404b6e20461a251beb1173c9e0ee809e279dfca4733e6acc502bda0"
        ):
            raise ValueError("Not the corrected frozen screen")
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_root": str(root),
        "status": "complete_valid_56_trial_study",
        "excludes": "Original ten halted trials, all preflights and optional RGB proposal diagnostic; no hardware success inference.",
        "stages": {},
        "interpretation": "Fixed August packet with authentic recorded robot/gripper/wrist starts. Conditional simulator study; no proven training-unseen or hardware probability claim.",
    }
    for stage in ("screen", "confirmation"):
        path = root / stage / "selection/selection.json"
        if not path.exists():
            statuses = Counter()
            for p in (root / stage / "rollouts").glob("*/run_status.json"):
                statuses[json.loads(p.read_text()).get("status", "unknown")] += 1
            result["stages"][stage] = {
                "status": "selection_not_available",
                "raw_trial_statuses": dict(statuses),
            }
            continue
        report = json.loads(path.read_text())
        stage_result = {
            k: report.get(k)
            for k in (
                "status",
                "campaign_sha256",
                "expected_trials",
                "available_trials",
                "missing_trial_keys",
                "invalid_trial_keys",
                "ranking",
                "selected_ids",
                "clear_simulator_winner",
                "conclusion",
                "comparison",
                "confirmation_gates",
            )
        }
        stage_result["report"] = str(path)
        stage_result["report_sha256"] = digest(path)
        stage_result["candidates"] = {}
        for pid in sorted({r["policy_id"] for r in report.get("trials", [])}):
            rows = [r for r in report["trials"] if r["policy_id"] == pid]
            valid = [r for r in rows if r.get("valid_for_selection") is True]
            counts = {
                key: sum(r.get(key) is True for r in valid)
                for key in (
                    "clean_place",
                    "strict_full_place",
                    "acquired",
                    "lifted",
                    "carried",
                    "released_in_bin",
                    "dropped",
                    "actual_stop",
                    "safety_stop",
                    "wrench_limit_stop",
                    "tactile_force_limit_stop",
                )
            }
            reasons = Counter(r.get("stop_reason") or "no_stop" for r in valid)
            events = Counter(
                e for r in valid for e in r.get("terminal_safety_events", [])
            )
            count = sum(
                r.get("native_inference_latency_s", {}).get("count", 0) for r in valid
            )
            weighted = sum(
                r.get("native_inference_latency_s", {}).get("count", 0)
                * (r.get("native_inference_latency_s", {}).get("mean") or 0.0)
                for r in valid
            )
            latency_fields = {}
            for key in (
                "native_inference_latency_s",
                "delivery_latency_s",
                "inference_wall_time_s",
                "activation_delay_s",
            ):
                observed = [r.get(key) or {} for r in valid]
                n = sum(s.get("count", 0) for s in observed)
                latency_fields[key] = {
                    "count": n,
                    "event_weighted_mean_s": sum(
                        s.get("count", 0) * (s.get("mean") or 0) for s in observed
                    )
                    / n
                    if n
                    else None,
                }
            stage_result["candidates"][pid] = {
                "latency": latency_fields,
                "valid": len(valid),
                "available": len(rows),
                "outcome_counts": counts,
                "terminal_controller_reasons": dict(reasons),
                "events_logged_on_first_stopped_tick_including_nonterminal": dict(
                    events
                ),
                "native_latency_plan_count": count,
                "native_latency_plan_weighted_mean_s": weighted / count
                if count
                else None,
                "strict_placements_with_later_stop": sum(
                    r.get("strict_full_place") is True and r.get("actual_stop") is True
                    for r in valid
                ),
            }
        result["stages"][stage] = stage_result
    if all(
        r.get("status") == "complete_valid_matched_stage"
        for r in result["stages"].values()
    ):
        result["status"] = "complete_valid_56_trial_study"
        result["clear_simulator_winner"] = result["stages"]["confirmation"][
            "clear_simulator_winner"
        ]
        result["conclusion"] = result["stages"]["confirmation"]["conclusion"]
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--root",
    type=Path,
    default=Path(
        "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_robustness_v2_delivery"
    ),
)
args = parser.parse_args()
print(json.dumps(summarize(args.root), indent=2, allow_nan=False))
