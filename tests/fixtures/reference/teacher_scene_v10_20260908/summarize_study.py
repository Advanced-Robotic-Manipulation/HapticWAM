#!/usr/bin/env python3
"""Summarize every frozen V10 trial without changing campaign scores (CPU only).

Run the existing analyze_policy_campaign.py for each variant first. This helper
reads those score records and their hashed raw inputs, then adds a stricter
through-horizon placement check. It never launches inference or device drivers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from phantom.sim.geometry import bin_geometry  # noqa: E402
from phantom.sim.policy_metrics import DEFAULT_THRESHOLDS, _rotations  # noqa: E402

STAGES = ("acquired", "lifted", "carried", "released_in_bin", "full_task", "dropped")
EXTRA = ("sustained_supported_placement_through_horizon", "clean_placement_through_horizon")
REPLICATES, BOOTSTRAP_SEED = 10000, 20260908
CONTRASTS = (
    ("scene_with_legacy_hold", "primary", {"r4_legacy": 1, "old_legacy": -1}),
    ("scene_with_bounded_hold", "primary", {"r4_bounded": 1, "old_bounded": -1}),
    ("hold_in_old_scene", "primary", {"old_bounded": 1, "old_legacy": -1}),
    ("hold_in_r4_scene", "primary", {"r4_bounded": 1, "r4_legacy": -1}),
    ("scene_controller_interaction", "primary", {
        "r4_bounded": 1, "r4_legacy": -1, "old_bounded": -1, "old_legacy": 1}),
    ("play12_in_r4_bounded", "secondary", {"r4_bounded_play12": 1, "r4_bounded": -1}),
)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    if not path.is_file():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = read_json(path)
    if isinstance(value, list):
        return value
    for key in ("plans", "events", "rows"):
        if isinstance(value.get(key), list):
            return value[key]
    raise ValueError(f"Unrecognized row schema: {path}")


def numeric(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return {"count": len(a), **{key: float(fn(a)) if len(a) else None for key, fn in (
        ("mean", np.mean), ("median", np.median),
        ("p95", lambda x: np.percentile(x, 95)), ("max", np.max))}}


def latency_summary(plans):
    """Summarize action-bearing plans without mixing first and steady timing."""
    return {
        "native_inference_s": numeric([row.get("diagnostics", {}).get("sim_native_inference_latency_s", row.get("latency_s")) for row in plans]),
        "full_client_rpc_s": numeric([row.get("diagnostics", {}).get("sim_policy_replan_wall_time_s") for row in plans]),
        "runner_inference_wall_s": numeric([row.get("inference_wall_time_s") for row in plans]),
        "effective_delivery_delay_s": numeric([row.get("diagnostics", {}).get("sim_effective_delivery_delay_s") for row in plans]),
    }


def path_from(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def through_horizon(trace, config, metrics, run, execution, horizon):
    """Check ALL sampled states after first confirmed placement, not just last.

    Support forces are positive normal-force magnitudes in N, exactly matching
    policy_metrics.py. This checks sampled containment, all-robot unloading,
    bin support and settling. It cannot certify unobserved contact transients.
    """
    result = dict.fromkeys(EXTRA)
    result.update(status="not_evaluated", reasons=[], horizon_s=horizon,
                  initial_placement=bool(metrics.get("outcomes", {}).get("full_task")),
                  original_final_inside_unloaded=bool(
                      metrics.get("object", {}).get("final_inside_bin")
                      and metrics.get("object", {}).get("final_contacts_unloaded")))
    result["semantics"] = (
        "Every sampled state from original full_task confirmation through the planned horizon: "
        "oriented containment, both pads unloaded, ALL robot normal force <= threshold, "
        "bin normal force > threshold, linear/angular settling limits. Clean additionally "
        "requires no recorded policy/controller stop after placement. No whole-robot "
        "collision-free certification is inferred.")
    if metrics.get("valid_for_scoring") is not True:
        result["reasons"].append("original_score_invalid_or_missing")
        return result
    limits = {**DEFAULT_THRESHOLDS, **metrics.get("thresholds", {})}
    placed = metrics.get("event_times_s", {}).get("full_task")
    stop_events = [{"t": float(row["t"]), "reason": row.get("stop_reason")}
                   for row in execution if row.get("stopped") and row.get("t") is not None]
    stop_events += [{"t": float(row["t"]), "reason": row.get("reason", row.get("event"))}
                    for row in run.get("events", [])
                    if row.get("event") in ("policy_stop", "controller_stop")
                    and row.get("t") is not None]
    result["first_recorded_stop_s"] = min((row["t"] for row in stop_events), default=None)
    horizon_tolerance = float(config["physics"]["dt"]) + limits["clock_tolerance_s"]
    later = [row for row in stop_events if placed is not None and placed <= row["t"] <= horizon + horizon_tolerance]
    result["post_placement_stop"] = min(later, key=lambda row: row["t"]) if later else None
    if not result["initial_placement"]:
        result.update(status="no_initial_placement", **dict.fromkeys(EXTRA, False))
        return result
    if placed is None or not np.isfinite(placed):
        result["reasons"].append("missing_finite_original_placement_time")
        return result
    result["confirmation_time_s"] = float(placed)
    t = np.asarray(trace.get("t", []), dtype=float)
    if t.ndim != 1 or len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        result["reasons"].append("invalid_raw_trace_times")
        return result
    required = {"physics_t": (len(t),), "waffle_position": (len(t), 3),
                "waffle_orientation_wxyz": (len(t), 4),
                "pad_packet_normal_force": (len(t), 2),
                "packet_robot_normal_force": (len(t),), "packet_bin_normal_force": (len(t),)}
    arrays = {key: np.asarray(trace.get(key, []), dtype=float) for key in required}
    for key, shape in required.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            result["reasons"].append(f"missing_or_invalid:{key}")
    if result["reasons"]:
        result["status"] = "initial_placement_and_final_state_only"
        return result
    force_keys = ("pad_packet_normal_force", "packet_robot_normal_force", "packet_bin_normal_force")
    if any(np.any(arrays[key] < -1e-8) for key in force_keys):
        result["reasons"].append("negative_normal_force_magnitude")
    if np.max(np.diff(t)) > limits["max_sample_gap_s"] + 1e-9:
        result["reasons"].append("trace_sample_gap_exceeds_resolution")
    if np.max(np.abs(arrays["physics_t"] - t)) > limits["clock_tolerance_s"]:
        result["reasons"].append("physics_clock_mismatch")
    q = arrays["waffle_orientation_wxyz"]
    norms = np.linalg.norm(q, axis=1)
    if not np.allclose(norms, 1, atol=1e-3, rtol=0):
        result["reasons"].append("nonunit_object_quaternion")
    if result["reasons"]:
        return result
    q = q / norms[:, None]
    p = arrays["waffle_position"]
    corners = np.asarray(list(product([-1, 1], repeat=3))) * np.asarray(config["waffle"]["size"]) / 2
    world = np.einsum("nij,kj->nki", _rotations(q), corners) + p[:, None]
    geometry = bin_geometry(config["bin"])
    lower, upper = geometry.interior_bounds
    local, tol = geometry.to_interior_frame(world), limits["bin_tolerance_m"]
    inside = (((local[:, :, :2] >= lower[:2] - tol) &
               (local[:, :, :2] <= upper[:2] + tol)).all(axis=(1, 2)) &
              (world[:, :, 2].min(axis=1) >= lower[2] - tol) &
              (world[:, :, 2].max(axis=1) <= upper[2] + tol))
    speed = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)]
    angular = np.r_[0, 2 * np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", q[1:], q[:-1])), 0, 1)) / np.diff(t)]
    checks = {
        "contained": inside,
        "pads_unloaded": (arrays["pad_packet_normal_force"] <= limits["contact_force_n"]).all(axis=1),
        "all_robot_unloaded": arrays["packet_robot_normal_force"] <= limits["support_robot_force_max_n"],
        "bin_supported": arrays["packet_bin_normal_force"] > limits["support_bin_force_min_n"],
        "linear_settled": speed <= limits["settle_speed_m_s"],
        "angular_settled": angular <= limits["settle_angular_speed_rad_s"],
    }
    # Runner traces can end one physics step before the requested horizon.
    window = (t >= placed - 1e-9) & (t <= horizon + horizon_tolerance)
    covered = bool(t[0] <= horizon_tolerance and t[-1] >= horizon - horizon_tolerance)
    result.update(trace_last_s=float(t[-1]), horizon_coverage=covered,
                  horizon_tolerance_s=horizon_tolerance, checked_rows=int(window.sum()))
    if not window.any():
        result["reasons"].append("no_samples_after_placement")
        return result
    result["post_placement_checks"] = {
        key: {"passed": bool(value[window].all()),
              "failed_samples": int((~value[window]).sum()),
              "first_failure_s": float(t[window & ~value][0]) if (window & ~value).any() else None}
        for key, value in checks.items()}
    result["post_placement_bin_force_min_n"] = float(arrays["packet_bin_normal_force"][window].min())
    result["post_placement_robot_force_max_n"] = float(arrays["packet_robot_normal_force"][window].max())
    physical_pass = all(check["passed"] for check in result["post_placement_checks"].values())
    # Observed failure remains false even when the horizon was not completed.
    physical = False if not physical_pass else True if covered else None
    result[EXTRA[0]] = physical
    unknown_stop_time = bool(run.get("policy_stop_reason") and not stop_events)
    result["stop_time_unresolved"] = unknown_stop_time
    result[EXTRA[1]] = (False if physical is False or later else
                        None if physical is None or unknown_stop_time else True)
    if not covered:
        result["reasons"].append("planned_horizon_not_observed")
    if unknown_stop_time:
        result["reasons"].append("recorded_stop_without_timestamp")
    result["status"] = "passed" if result[EXTRA[1]] is True else "failed" if result[EXTRA[1]] is False else "unverified"
    return result


def contact_diagnostic(path):
    if not path.is_file():
        return {"status": "not_recorded", "collision_count": None}
    rows, peaks, sample_times = 0, {}, []
    for row in read_rows(path):
        rows += 1
        if row.get("t") is not None:
            sample_times.append(float(row["t"]))
        for key, value in row.get("normal_force_by_environment_n", {}).items():
            peaks[key] = max(peaks.get(key, 0), float(value))
    return {"status": "recorded", "rows": rows, "peak_normal_force_by_environment_n": peaks,
            "collision_count": None,
            "maximum_sample_gap_s": max(np.diff(sample_times)).item() if len(sample_times) > 1 else None,
            "interpretation": "Sampled normal-contact magnitudes, not classified collision events. Packet contacts may be intentional; no collision count is fabricated."}


def first_stop_diagnostic(execution):
    """Retain the first stop's actual guard labels, not later hold diagnostics."""
    row = next((row for row in execution if row.get("stopped")), None)
    if row is None:
        return None
    diagnostics = row.get("diagnostics") or {}
    return {"t": row.get("t"), "reason": row.get("stop_reason"),
            "safety_events": list(diagnostics.get("safety_events") or []),
            "wrist_guard": diagnostics.get("wrist_guard"),
            "measured_q": row.get("measured_q"), "measured_qd": row.get("measured_qd"),
            "measured_tcp": row.get("measured_tcp"),
            "ik_reason": row.get("ik_reason"),
            "interpretation": "Labels recorded at the first stopped executor row; simultaneous labels are not individually assigned causal priority."}


def load_trial(spec, variant, campaign, campaign_sha, runs):
    policy = campaign["policies"][0]["id"]
    case_id = f"{policy}__{spec['condition']}__seed{spec['seed']}"
    directory = runs / variant["id"] / "rollouts" / case_id
    score_path = runs / variant["id"] / "analysis" / "trials" / f"{case_id}.json"
    record = {**spec, "case_id": case_id, "directory": str(directory), "score_path": str(score_path),
              "status": "missing_score", "invalid_reasons": [], "original_metrics": None,
              "outcomes": dict.fromkeys(STAGES + EXTRA)}
    if not score_path.is_file():
        record["run_present"] = (directory / "run.json").is_file()
        return record
    score = read_json(score_path)
    record.update(score_sha256=sha256(score_path), original_score_status=score.get("status"))
    metrics = score.get("metrics") or {}
    record["original_metrics"] = metrics
    record["invalid_reasons"] = list(metrics.get("invalid_reasons", []))
    if score.get("case", {}).get("campaign_sha256") != campaign_sha:
        record["invalid_reasons"].append("score_campaign_hash_mismatch")
    if not score.get("input_sha256"):
        record["invalid_reasons"].append("score_raw_input_hash_manifest_missing")
    if (score.get("policy_id"), score.get("condition_id"), score.get("sampling_seed")) != (policy, spec["condition"], spec["seed"]):
        record["invalid_reasons"].append("score_identity_mismatch")
    for name, expected in score.get("input_sha256", {}).items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            record["invalid_reasons"].append(f"score_raw_input_hash_mismatch:{name}")
    valid = score.get("status") == "scored" and metrics.get("valid_for_scoring") is True and not record["invalid_reasons"]
    record["status"] = "scored" if valid else "invalid"
    if not valid:
        return record
    record["outcomes"].update({stage: bool(metrics["outcomes"][stage]) for stage in STAGES})
    run, config = read_json(directory / "run.json"), read_json(directory / "effective_config.json")
    execution, planners = read_rows(directory / "execution_trace.jsonl"), read_rows(directory / "planner_trace.json")
    with np.load(directory / "sim_trace.npz", allow_pickle=False) as trace:
        sustained = through_horizon(trace, config, metrics, run, execution, float(campaign["horizon_s"]))
    record["through_horizon"] = sustained
    record["outcomes"].update({name: sustained[name] for name in EXTRA})
    record["stop_reason"] = metrics.get("control", {}).get("stop_reason")
    record["first_stop_diagnostic"] = first_stop_diagnostic(execution)
    record["completed_reason"] = run.get("policy_completed_reason")
    record["reach_diagnostic"] = metrics.get("reach_diagnostic", {})
    action_plans = [row for row in planners if "actions" in row]
    record["latency"] = {
        **latency_summary(action_plans),
        "action_plan_count": len(action_plans),
        "first_plan": latency_summary(action_plans[:1]),
        "steady_plans": latency_summary(action_plans[1:]),
        "phase_definition": "First action-bearing plan in the recorded planner order versus all subsequent action-bearing plans; inference errors remain separately reported",
    }
    held = [bool(row.get("servo_reach_limiter", {}).get("constraint_hold", {}).get("held")) for row in execution]
    times = np.asarray([row["t"] for row in execution], dtype=float)
    record["controller"] = {**metrics.get("control", {}), "constraint_hold_rows": sum(held),
                            "constraint_hold_duration_s": float(np.dot(np.diff(times), held[:-1])) if len(times) > 1 else 0}
    contact_path = directory / "robot_environment_contact_trace.json"
    try:
        record["robot_environment_contacts"] = contact_diagnostic(contact_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record["robot_environment_contacts"] = {"status": "diagnostic_error", "collision_count": None,
                                                "error": f"{type(exc).__name__}: {exc}"}
    if contact_path.is_file():
        record["robot_environment_contacts"]["sha256"] = sha256(contact_path)
    return record


def outcome_summary(records, name):
    values = [row["outcomes"].get(name) for row in records]
    true, false = sum(value is True for value in values), sum(value is False for value in values)
    unknown, planned = len(values) - true - false, len(values)
    return {"planned": planned, "true": true, "false": false, "unresolved": unknown,
            "rate_among_resolved": true / (true + false) if true + false else None,
            "planned_denominator_bounds": [true / planned, (true + unknown) / planned] if planned else None}


def paired(records, coefficients, outcome):
    relevant = [row for row in records if row["variant"] in coefficients]
    index = {(row["variant"], row["condition"], row["seed"]): row for row in relevant}
    cells = list(dict.fromkeys((row["condition"], row["seed"]) for row in relevant))
    rows, clusters, excluded = [], [], []
    for condition, seed in cells:
        values = {variant: index.get((variant, condition, seed), {}).get("outcomes", {}).get(outcome) for variant in coefficients}
        delta = sum(coefficients[key] * int(value) for key, value in values.items()) if all(value is not None for value in values.values()) else None
        rows.append({"condition": condition, "seed": seed, "values": values, "difference": delta})
    for condition in dict.fromkeys(row["condition"] for row in rows):
        values = [row["difference"] for row in rows if row["condition"] == condition]
        if all(value is not None for value in values):
            clusters.append({"condition": condition, "seed_differences": values, "mean": float(np.mean(values))})
        else:
            excluded.append(condition)
    estimate = None
    estimation_rows = sum(len(row["seed_differences"]) for row in clusters)
    if clusters:
        values = np.asarray([row["mean"] for row in clusters])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = values[rng.integers(0, len(values), size=(REPLICATES, len(values)))].mean(axis=1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
        estimate = {"difference": float(values.mean()), "confidence_level": 0.95,
                    "interval": interval, "start_clusters": len(clusters),
                    "matched_rows": estimation_rows,
                    "bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                    "degenerate_interval": bool(interval[0] == interval[1]),
                    "interpretation": "Descriptive paired start-cluster resampling with both seeds retained; six starts do not support population or hardware claims. A zero-width interval does not establish equivalence."}
    resolved = [row["difference"] for row in rows if row["difference"] is not None]
    low, high = sum(min(value, 0) for value in coefficients.values()), sum(max(value, 0) for value in coefficients.values())
    unknown = len(rows) - len(resolved)
    return {"coefficients": coefficients, "outcome": outcome, "planned_matched_rows": len(rows),
            "available_matched_rows": len(resolved), "complete_start_clusters": clusters,
            "estimation_matched_rows": estimation_rows,
            "status": "complete_matched_suite" if clusters and not excluded else "partial_matched_suite" if clusters else "no_complete_start_clusters",
            "excluded_start_clusters": excluded, "rows": rows, "estimate": estimate,
            "positive_discordances": sum(value > 0 for value in resolved),
            "negative_discordances": sum(value < 0 for value in resolved),
            "planned_difference_bounds": [(sum(resolved) + unknown * low) / len(rows),
                                           (sum(resolved) + unknown * high) / len(rows)] if rows else None}


def report(summary):
    lines = [f"# {summary['study_id']} results", "", f"Status: **{summary['status']}**. Every one of {summary['planned_trials']} planned cases remains in the denominator.", "",
             "Original full-task scores are preserved. Sustained placement additionally requires every sampled state after placement to remain contained, unloaded, bin-supported and settled through 60 seconds. Clean placement additionally excludes a later recorded controller/safety stop. Missing observations remain unverified.", "",
             "| Variant | Valid/planned | Acquire | Lift | Carry | Release | Initial placement | Sustained | Clean | Drops |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in summary["variants"].items():
        def cell(key):
            item = value["outcomes"][key]
            return f"{item['true']}/{item['planned']}" + (f" ({item['unresolved']} unknown)" if item["unresolved"] else "")
        lines.append(f"| {name} | {value['statuses'].get('scored', 0)}/{value['planned']} | " + " | ".join(cell(key) for key in STAGES[:5] + EXTRA + ("dropped",)) + " |")
    lines += ["", "Paired differences below are percentage points, treatment minus reference. Bootstrap samples whole arm starts, retaining both policy seeds. Rows used in each estimate exclude a whole start when either seed is unresolved; all individually resolved pairs and scheduled-denominator bounds remain in summary.json. The anchor and varied-start counts are also separate there.", "", "| Contrast | Outcome | Rows used/planned | Status | Difference | Descriptive 95% interval |", "|---|---|---:|---|---:|---:|"]
    for contrast in summary["contrasts"]:
        if contrast["outcome"] not in ("full_task", EXTRA[1]):
            continue
        est = contrast["estimate"]
        delta = f"{100 * est['difference']:+.1f}" if est else "unavailable"
        ci = f"[{100 * est['interval'][0]:+.1f}, {100 * est['interval'][1]:+.1f}]" if est else "unavailable"
        lines.append(f"| {contrast['id']} | {contrast['outcome']} | {contrast['estimation_matched_rows']}/{contrast['planned_matched_rows']} | {contrast['status']} | {delta} | {ci} |")
    lines += ["", "Teacher tactile/wrist proxies remain uncalibrated. This is a fixed-object, measured-arm-start comparison, not training-held-out evaluation. Measured RPC delivery means equal sampling seeds need not yield identical action activation times. Contact diagnostics report sampled forces, not certified collision counts.", "", "All original scores, input hashes, stop reasons, latency summaries, reach diagnostics, post-placement failures and missing/invalid cases are retained in summary.json and trials.csv."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, help="Explicit relocation of runtime.output, e.g. downloaded raw evidence")
    args = parser.parse_args()
    study = read_json(args.study)
    if study.get("status") != "frozen":
        raise ValueError("Study must be frozen")
    variants = {row["id"]: row for row in study["variants"]}
    if len(variants) != len(study["variants"]):
        raise ValueError("Duplicate variant IDs")
    runs = args.runs or path_from(args.study.parent, study["runtime"]["output"])
    campaign_paths = {name: path_from(args.study.parent, row["campaign"]) for name, row in variants.items()}
    if args.runs:
        campaign_paths = {name: path if path.is_file() else runs / name / "campaign_snapshot.json"
                          for name, path in campaign_paths.items()}
    campaigns = {name: read_json(path) for name, path in campaign_paths.items()}
    campaign_hashes = {name: sha256(path) for name, path in campaign_paths.items()}
    expected = {(name, condition["id"], seed) for name, campaign in campaigns.items()
                for condition in campaign["conditions"] for seed in campaign["sampling_seeds"]}
    keys = [(row["variant"], row["condition"], row["seed"]) for row in study["schedule"]]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Frozen schedule must cover every campaign case exactly once")
    records = []
    for row in study["schedule"]:
        try:
            records.append(load_trial(row, variants[row["variant"]], campaigns[row["variant"]], campaign_hashes[row["variant"]], runs))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            records.append({**row, "status": "analysis_error", "invalid_reasons": [f"{type(exc).__name__}: {exc}"],
                            "outcomes": dict.fromkeys(STAGES + EXTRA), "original_metrics": None})
    groups = {}
    for name, variant in variants.items():
        selected = [row for row in records if row["variant"] == name]
        groups[name] = {"factors": variant.get("factors"), "planned": len(selected),
                        "statuses": dict(Counter(row["status"] for row in selected)),
                        "outcomes": {key: outcome_summary(selected, key) for key in STAGES + EXTRA},
                        "stop_reasons": dict(Counter(row.get("stop_reason") or "none" for row in selected if row["status"] == "scored")),
                        "first_stop_safety_events": dict(Counter(event for row in selected for event in set((row.get("first_stop_diagnostic") or {}).get("safety_events", [])))),
                        "subsets": {label: {key: outcome_summary([row for row in selected if (row["condition"] == "sept4_anchor") == anchor], key) for key in STAGES + EXTRA} for label, anchor in (("sept4_anchor", True), ("varied_aug22_starts", False))},
                        "latency_trial_means": {key: numeric([row.get("latency", {}).get(key, {}).get("mean") for row in selected]) for key in ("native_inference_s", "full_client_rpc_s", "runner_inference_wall_s", "effective_delivery_delay_s")},
                        "latency_trial_means_by_phase": {
                            phase: {key: numeric([row.get("latency", {}).get(phase, {}).get(key, {}).get("mean") for row in selected])
                                    for key in ("native_inference_s", "full_client_rpc_s", "runner_inference_wall_s", "effective_delivery_delay_s")}
                            for phase in ("first_plan", "steady_plans")}}
    contrasts = [{"id": name, "tier": tier, **paired(records, coefficients, outcome)}
                 for name, tier, coefficients in CONTRASTS if set(coefficients) <= set(variants)
                 for outcome in STAGES + EXTRA]
    summary = {"schema_version": 1, "study_id": study["study_id"], "study_sha256": sha256(args.study),
               "analysis_source_sha256": sha256(__file__), "runtime_output_read": str(runs),
               "campaign_sha256": campaign_hashes,
               "frozen_analysis_plan": study.get("analysis", {}),
               "sustained_placement_fully_resolved": all(row["outcomes"][EXTRA[0]] is not None for row in records),
               "planned_trials": len(keys), "status": "complete" if all(row["status"] == "scored" for row in records) else "incomplete",
               "variants": groups, "contrasts": contrasts, "trials": records,
               "limitations": ["Six selected starts with two seeds are not a population sample or a new training-held-out split.",
                               "Teacher tactile and wrist proxy transfer is unvalidated.",
                               "Sampled support and contact cannot exclude between-frame transients or certify collision-free operation.",
                               "Measured inference delivery timing may differ before any nominal treatment becomes active.",
                               "Missing and invalid original scores stay unresolved in scheduled-denominator bounds."]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    columns = ["variant", "condition", "seed", "status", *STAGES, *EXTRA, "stop_reason", "first_stop_safety_events", "first_stop_s", "post_placement_stop_s", "initial_closure", "initial_closure_above_absolute_threshold", "reach_error_before_first_closing_motion_m", "first_closing_motion_s", "native_latency_mean_s", "rpc_latency_mean_s", "first_native_latency_s", "steady_native_latency_mean_s", "first_rpc_latency_s", "steady_rpc_latency_mean_s", "first_delivery_delay_s", "steady_delivery_delay_mean_s", "ik_rejects", "constraint_hold_rows", "constraint_hold_duration_s", "invalid_reasons", "directory"]
    with (args.out / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            reach, through, control = row.get("reach_diagnostic", {}), row.get("through_horizon", {}), row.get("controller", {})
            flat = {key: row.get(key) for key in columns}
            flat.update(row["outcomes"])
            flat["first_stop_safety_events"] = json.dumps((row.get("first_stop_diagnostic") or {}).get("safety_events", []))
            flat.update(first_stop_s=through.get("first_recorded_stop_s"), post_placement_stop_s=(through.get("post_placement_stop") or {}).get("t"), initial_closure=reach.get("initial_measured_closure"), initial_closure_above_absolute_threshold=reach.get("absolute_threshold_already_exceeded_at_start"), reach_error_before_first_closing_motion_m=reach.get("reach_error_before_first_closing_motion_m"), first_closing_motion_s=reach.get("first_closing_motion_s"), native_latency_mean_s=row.get("latency", {}).get("native_inference_s", {}).get("mean"), rpc_latency_mean_s=row.get("latency", {}).get("full_client_rpc_s", {}).get("mean"), ik_rejects=control.get("ik_rejects"), constraint_hold_rows=control.get("constraint_hold_rows"), constraint_hold_duration_s=control.get("constraint_hold_duration_s"), invalid_reasons=json.dumps(row.get("invalid_reasons", [])))
            for first_column, steady_column, metric in (
                ("first_native_latency_s", "steady_native_latency_mean_s", "native_inference_s"),
                ("first_rpc_latency_s", "steady_rpc_latency_mean_s", "full_client_rpc_s"),
                ("first_delivery_delay_s", "steady_delivery_delay_mean_s", "effective_delivery_delay_s"),
            ):
                flat[first_column] = row.get("latency", {}).get("first_plan", {}).get(metric, {}).get("mean")
                flat[steady_column] = row.get("latency", {}).get("steady_plans", {}).get(metric, {}).get("mean")
            writer.writerow(flat)
    (args.out / "report.md").write_text(report(summary))
    print(json.dumps({"status": summary["status"], "planned": len(keys), "out": str(args.out)}))


if __name__ == "__main__":
    main()
