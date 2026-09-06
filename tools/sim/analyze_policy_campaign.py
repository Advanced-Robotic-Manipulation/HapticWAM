#!/usr/bin/env python3
"""Score a frozen matched policy campaign from raw free-body traces, CPU only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUTCOMES = (
    "reach_before_closure",
    "acquired",
    "lifted",
    "carried",
    "released_in_bin",
    "full_task",
    "dropped",
)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_settings(design, policy=None):
    """Resolve declared recipe variants without changing checkpoint-owned config."""
    settings = copy.deepcopy(design["inference_settings"])
    overrides = (policy or {}).get("inference_settings", {})
    allowed = {
        "use_ema",
        "nfe",
        "guidance",
        "persistent_noise",
        "parity",
        "k_seeds",
        "max_play",
        "task_text",
    }
    if set(overrides) - allowed:
        raise ValueError(
            f"Unsupported policy inference overrides: {sorted(set(overrides) - allowed)}"
        )
    settings.update(overrides)
    for field in ("nfe", "k_seeds", "max_play"):
        value = settings[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("use_ema", "persistent_noise", "parity"):
        if not isinstance(settings[field], bool):
            raise TypeError(f"{field} must be boolean")
    if not np.isfinite(settings["guidance"]) or settings["guidance"] < 0:
        raise ValueError("guidance must be finite and nonnegative")
    return settings


def adapter_input(design, condition, field):
    """Only measured startup state can vary by start-condition in this version."""
    if field == "initial_state" and condition.get(field) is not None:
        return condition[field]
    return design.get("adapter_profile", {}).get(field)


def load_design(path):
    design = json.loads(Path(path).read_text())
    if design.get("status") != "frozen":
        raise ValueError("Campaign must be frozen before policy scoring")
    conditions = design["conditions"]
    if len({c["id"] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate condition IDs")
    if not conditions or conditions[0]["family"] != "nominal":
        raise ValueError("Nominal condition must be first")
    seeds = design["sampling_seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct matched sampling seeds are required")
    policies = design["policies"]
    if len({p["id"] for p in policies}) != len(policies):
        raise ValueError("Duplicate policy IDs")
    expected = len(conditions) * len(seeds)
    if design["planned_counts"]["per_policy"] != expected:
        raise ValueError("Frozen planned count disagrees with condition/seed grid")
    for policy in policies:
        policy_settings(design, policy)
    latency = design.get("delivery_latency_s")
    if latency is not None and (not np.isfinite(latency) or latency < 0):
        raise ValueError("delivery_latency_s must be finite and nonnegative")
    return design, fingerprint(path)


def planned_keys(design):
    return [
        (p["id"], c["id"], int(seed))
        for p in design["policies"]
        for c in design["conditions"]
        for seed in design["sampling_seeds"]
    ]


def condition_scene(design, condition):
    """Materialize one frozen scene; shared with the campaign controller."""
    scene = copy.deepcopy(design["nominal_scene"])
    for axis, offset in enumerate(condition["object_offset_m"]):
        scene["waffle"]["center"][axis] += offset
    for name in ("static_friction", "dynamic_friction"):
        scene["waffle"][name] *= condition["friction_scale"]
    if condition["camera_override"] is not None:
        scene["camera"] = copy.deepcopy(condition["camera_override"])
    return scene


def scene_mismatches(expected, actual):
    """Compare physical/sensor parameters; diagnostic prose and media rate excluded."""
    result = []

    def visit(want, got, path):
        if isinstance(want, dict):
            for key, value in want.items():
                if key in ("render_hz", "source", "geometry_source"):
                    continue
                visit(
                    value,
                    got.get(key) if isinstance(got, dict) else None,
                    path + "." + key,
                )
        elif isinstance(want, (int, float, list)):
            try:
                a, b = np.asarray(want, float), np.asarray(got, float)
                match = a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-9)
            except (TypeError, ValueError):
                match = want == got
            if not match:
                result.append(path)
        elif want != got:
            result.append(path)

    for section in ("physics", "table", "mat", "bin", "waffle", "gripper", "camera"):
        visit(expected[section], actual.get(section), section)
    return result


def read_rows(path):
    if not path.exists():
        return None
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return value
    for key in ("plans", "events", "rows"):
        if isinstance(value.get(key), list):
            return value[key]
    raise ValueError(f"Cannot identify rows in {path}")


def runtime_audit(design, policy, condition, info, server, run, times, stop):
    """Reject recipe drift and incomplete time coverage before outcome scoring."""
    reasons = []
    settings = policy_settings(design, policy)
    expected = {
        "nfe": settings["nfe"],
        "guidance": settings["guidance"],
        "k_seeds": settings["k_seeds"],
        "parity_fixes": settings["parity"],
        "persistent_noise": settings["persistent_noise"],
        "task_text": settings["task_text"],
        "drop_video": False,
        "close_p": 0.5,
    }
    for source, reported in (("policy", info), ("server", server)):
        for field, value in expected.items():
            if reported.get("effective", {}).get(field) != value:
                reasons.append(f"{source}_effective_{field}_differs_from_campaign")
    if info.get("ckpt_sha") != policy["checkpoint_sha256"]:
        reasons.append("remote_checkpoint_identity_not_verified")
    if server.get("checkpoint_sha256") != policy["checkpoint_sha256"]:
        reasons.append("server_checkpoint_identity_not_verified")
    if (
        server.get("system") != policy["architecture"]
        or info.get("mode") != policy["policy_mode"]
    ):
        reasons.append("effective_policy_architecture_differs_from_campaign")
    if server.get("weights") != ("EMA" if settings["use_ema"] else "raw"):
        reasons.append("effective_ema_setting_differs_from_campaign")
    hardware = design["runtime_hardware"]
    if info.get("hardware_effective") != hardware["effective_model"]:
        reasons.append("effective_hardware_differs_from_frozen_model")
    if (
        server.get("hardware_sha256") != hardware["sha256"]
        or server.get("hardware_config_hash") != hardware["config_hash"]
    ):
        reasons.append("server_hardware_differs_from_frozen_configuration")
    for field in ("observation_delay_s", "inference_delay_add_s"):
        if info.get(field) != condition[field]:
            reasons.append(f"effective_{field}_differs_from_condition")
    if info.get("max_play_steps") != settings["max_play"]:
        reasons.append("effective_max_play_differs_from_campaign")
    if info.get("policy_latency_override_s") != design.get("delivery_latency_s"):
        reasons.append("effective_delivery_latency_override_differs_from_campaign")
    profile = design.get("adapter_profile", {})
    veto = profile.get("terminal_veto", False)
    for field, expected_value in (
        ("planner_stall_watchdog", True),
        ("terminal_veto", veto),
        ("phantom_recovery", bool(veto and veto["implementation"] == "live")),
        ("tactile_model", profile.get("tactile_model", "contact_proxy")),
        ("wrist_model", "contact_proxy"),
    ):
        if info.get(field) != expected_value:
            reasons.append(f"effective_{field}_differs_from_campaign")
    if profile or condition.get("initial_state"):
        for field in (
            "placement_release",
            "gel_contact_coverage",
            "record_packet_support",
            "save_policy_observations",
        ):
            if info.get(field) != profile.get(field):
                reasons.append(f"effective_{field}_differs_from_campaign")
        for field, reported_field in (
            ("initial_state", "policy_initial_state_provenance"),
            ("tactile_baseline", "tactile_baseline_provenance"),
        ):
            expected_input = adapter_input(design, condition, field)
            if (
                expected_input
                and info.get(reported_field, {}).get("sha256")
                != expected_input["sha256"]
            ):
                reasons.append(f"effective_{field}_hash_differs_from_campaign")
    times = np.asarray(times, float)
    tolerance = design["thresholds"]["max_sample_gap_s"]
    if times.ndim != 1 or not times.size or not np.isfinite(times).all():
        reasons.append("raw_trace_clock_missing_or_invalid")
        return reasons
    if abs(times[0]) > tolerance:
        reasons.append("raw_trace_initial_time_missing")
    duration = float(run.get("duration_s", float("nan")))
    if not np.isfinite(duration) or abs(times[-1] - duration) > tolerance:
        reasons.append("raw_trace_end_disagrees_with_reported_duration")
    if times[-1] > design["horizon_s"] + tolerance:
        reasons.append("raw_trace_exceeds_frozen_horizon")
    if times[-1] + tolerance < design["horizon_s"]:
        if not stop:
            reasons.append("short_horizon_without_recorded_termination")
        elif stop in ("lift_complete", "success", "achieved"):
            reasons.append("early_success_termination_forbidden")
    return reasons


def load_trials(runs, design, design_sha, out):
    from phantom.sim.policy_metrics import evaluate_policy_trace

    expected = set(planned_keys(design))
    policies = {p["id"]: p for p in design["policies"]}
    conditions = {c["id"]: c for c in design["conditions"]}
    found = {}
    for manifest_path in sorted(Path(runs).rglob("case.json")):
        case = json.loads(manifest_path.read_text())
        if case.get("campaign_sha256") != design_sha:
            raise ValueError(f"Frozen campaign hash mismatch: {manifest_path}")
        key = (case["policy_id"], case["condition_id"], int(case["sampling_seed"]))
        if key not in expected:
            raise ValueError(f"Unplanned trial key {key}")
        if key in found:
            raise ValueError(
                f"Duplicate trial key {key}; resolve explicit attempt provenance first"
            )
        folder = manifest_path.parent
        record = {
            "policy_id": key[0],
            "condition_id": key[1],
            "sampling_seed": key[2],
            "directory": str(folder),
            "case_sha256": fingerprint(manifest_path),
            "case": case,
            "status": "missing_inputs",
            "metrics": None,
        }
        found[key] = record
        required = [
            folder / name
            for name in (
                "run.json",
                "sim_trace.npz",
                "effective_config.json",
                "execution_trace.jsonl",
                "planner_trace.json",
                "policy_info.json",
                "server_ready.json",
            )
        ]
        absent = [p.name for p in required if not p.is_file()]
        if absent:
            record["missing_inputs"] = absent
            continue
        run = json.loads(required[0].read_text())
        effective = json.loads(required[2].read_text())
        with np.load(required[1], allow_pickle=False) as trace:
            metrics = evaluate_policy_trace(
                trace,
                effective,
                design["thresholds"],
                run=run,
                execution_trace=read_rows(folder / "execution_trace.jsonl"),
                planner_trace=read_rows(folder / "planner_trace.json"),
            )
            times = np.asarray(trace["t"]).copy() if "t" in trace else []
        mismatches = scene_mismatches(
            condition_scene(design, conditions[key[1]]), effective
        )
        if mismatches:
            metrics["invalid_reasons"].append(
                "effective_scene_differs_from_frozen_condition"
            )
            metrics["valid_for_scoring"] = False
            record["effective_scene_mismatches"] = mismatches
        actual_sha = case.get("checkpoint_sha256")
        if actual_sha != policies[key[0]]["checkpoint_sha256"]:
            metrics["invalid_reasons"].append(
                "missing_or_mismatched_checkpoint_identity"
            )
            metrics["valid_for_scoring"] = False
        info_path = folder / "policy_info.json"
        info = json.loads(info_path.read_text())
        server_path = folder / "server_ready.json"
        server = json.loads(server_path.read_text())
        if case.get("server_ready_sha256") != fingerprint(server_path):
            metrics["invalid_reasons"].append("server_ready_snapshot_hash_mismatch")
        metrics["invalid_reasons"].extend(
            runtime_audit(
                design,
                policies[key[0]],
                conditions[key[1]],
                info,
                server,
                run,
                times,
                metrics.get("control", {}).get("stop_reason"),
            )
        )
        metrics["valid_for_scoring"] = not metrics["invalid_reasons"]
        record["status"] = "scored" if metrics["valid_for_scoring"] else "invalid"
        record["metrics"] = metrics
        record["input_sha256"] = {p.name: fingerprint(p) for p in required}
        score_path = out / "trials" / f"{key[0]}__{key[1]}__seed{key[2]}.json"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        record["score_path"] = str(score_path)
    return found


def outcome_summary(records, scheduled, outcome):
    valid = [
        r
        for r in records
        if r.get("status") == "scored"
        and r.get("metrics", {}).get("valid_for_scoring") is True
    ]
    successes = sum(bool(r["metrics"]["outcomes"][outcome]) for r in valid)
    unresolved = scheduled - len(valid)
    return {
        "scheduled": scheduled,
        "valid_scored": len(valid),
        "observed_true": successes,
        "observed_false": len(valid) - successes,
        "missing_or_invalid": unresolved,
        "rate_among_valid_only": successes / len(valid) if valid else None,
        "scheduled_denominator_rate_bounds": [
            successes / scheduled,
            (successes + unresolved) / scheduled,
        ],
    }


def paired_cluster_interval(a, b, *, confidence=0.95, replicates=10000, seed=20260906):
    """Resample matched configurations; never split the seed outcomes in a row."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (
        a.ndim != 2
        or a.shape != b.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Paired arrays must be finite matching configurations×seeds")
    if np.any((a < 0) | (a > 1) | (b < 0) | (b > 1)):
        raise ValueError("Outcome values must lie in [0,1]")
    difference = (a - b).mean(axis=1)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(difference), size=(replicates, len(difference)))
    bootstrap = difference[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    limits = np.quantile(bootstrap, [alpha, 1 - alpha])
    return {
        "difference_a_minus_b": float(difference.mean()),
        "confidence_level": confidence,
        "interval": limits.tolist(),
        "configuration_clusters": len(difference),
        "seeds_per_cluster": a.shape[1],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "degenerate_interval": bool(limits[0] == limits[1]),
        "interpretation": "Descriptive paired scenario-resampling interval; not a hardware/population generalization bound. A degenerate interval does not establish equivalence.",
    }


def matched_comparison(records, design, policy_a, policy_b, outcome="full_task"):
    seeds = design["sampling_seeds"]
    a, b, kept, excluded = [], [], [], []
    for condition in design["conditions"]:
        cid = condition["id"]
        rows = [
            [records.get((policy, cid, int(seed))) for seed in seeds]
            for policy in (policy_a, policy_b)
        ]
        complete = all(
            r is not None
            and r.get("status") == "scored"
            and r["metrics"]["valid_for_scoring"]
            for group in rows
            for r in group
        )
        if not complete:
            excluded.append(cid)
            continue
        a.append([float(r["metrics"]["outcomes"][outcome]) for r in rows[0]])
        b.append([float(r["metrics"]["outcomes"][outcome]) for r in rows[1]])
        kept.append(cid)
    settings = design["analysis"]
    result = {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "outcome": outcome,
        "status": "complete_matched_suite" if not excluded else "partial_matched_suite",
        "included_condition_ids": kept,
        "excluded_condition_ids": excluded,
        "matched_rollouts_per_policy": len(kept) * len(seeds),
    }
    result["estimate"] = (
        paired_cluster_interval(
            a,
            b,
            confidence=settings["confidence_level"],
            replicates=settings["bootstrap_replicates"],
            seed=settings["bootstrap_seed"],
        )
        if kept
        else None
    )
    return result


def numeric_summary(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "max": max(values) if values else None,
    }


def summarize_policy(records, design, policy):
    selected = [r for key, r in records.items() if key[0] == policy]
    scheduled = len(design["conditions"]) * len(design["sampling_seeds"])
    valid = [r["metrics"] for r in selected if r["status"] == "scored"]
    result = {
        "scheduled": scheduled,
        "present": len(selected),
        "valid_scored": len(valid),
        "invalid": sum(r["status"] == "invalid" for r in selected),
        "missing": scheduled
        - len(selected)
        + sum(r["status"] == "missing_inputs" for r in selected),
        "outcomes": {
            name: outcome_summary(selected, scheduled, name) for name in OUTCOMES
        },
        "families": {},
        "sampling_seeds": {},
        "control": {
            "stop_count": sum(bool(m["control"].get("stop_reason")) for m in valid),
            "ik_rejects": numeric_summary(
                [m["control"].get("ik_rejects") for m in valid]
            ),
            "hold_duration_s": numeric_summary(
                [m["control"].get("hold_duration_s") for m in valid]
            ),
            "effective_latency_mean_s": numeric_summary(
                [m["control"].get("effective_latency_s", {}).get("mean") for m in valid]
            ),
            "replans": numeric_summary([m["control"].get("replans") for m in valid]),
            "inference_error_count": sum(
                m["control"].get("inference_errors", 0) for m in valid
            ),
        },
        "collisions": {
            "observability": "Partial pad/environment forces and reported events; no whole-robot collision-free certification.",
            "reported_count": sum(
                m["collisions"].get("reported_count", 0) for m in valid
            ),
            "pad_environment_force_peak_n": numeric_summary(
                [m["collisions"].get("pad_environment_force_peak_n") for m in valid]
            ),
        },
    }
    for family in dict.fromkeys(c["family"] for c in design["conditions"]):
        ids = {c["id"] for c in design["conditions"] if c["family"] == family}
        subset = [r for r in selected if r["condition_id"] in ids]
        result["families"][family] = {
            name: outcome_summary(
                subset, len(ids) * len(design["sampling_seeds"]), name
            )
            for name in OUTCOMES
        }
    for seed in design["sampling_seeds"]:
        subset = [r for r in selected if r["sampling_seed"] == seed]
        result["sampling_seeds"][str(seed)] = outcome_summary(
            subset, len(design["conditions"]), "full_task"
        )
    complete = []
    for condition in design["conditions"]:
        rows = [
            records.get((policy, condition["id"], int(seed)))
            for seed in design["sampling_seeds"]
        ]
        if all(r is not None and r["status"] == "scored" for r in rows):
            complete.append(
                len({r["metrics"]["outcomes"]["full_task"] for r in rows}) > 1
            )
    result["seed_disagreement"] = {
        "complete_conditions": len(complete),
        "disagreements": sum(complete),
        "fraction": float(np.mean(complete)) if complete else None,
    }
    complete_outcomes = []
    for condition in design["conditions"]:
        rows = [
            records.get((policy, condition["id"], int(seed)))
            for seed in design["sampling_seeds"]
        ]
        if all(r is not None and r["status"] == "scored" for r in rows):
            complete_outcomes.append(
                [float(r["metrics"]["outcomes"]["full_task"]) for r in rows]
            )
    if complete_outcomes:
        values = np.asarray(complete_outcomes)
        settings = design["analysis"]
        interval = paired_cluster_interval(
            values,
            np.zeros_like(values),
            confidence=settings["confidence_level"],
            replicates=settings["bootstrap_replicates"],
            seed=settings["bootstrap_seed"],
        )
        result["scenario_success_interval"] = {
            "mean": interval["difference_a_minus_b"],
            "interval": interval["interval"],
            "confidence_level": interval["confidence_level"],
            "complete_configurations": len(values),
            "method": "Bootstrap configuration clusters, retaining both sampling seeds; descriptive sensitivity-suite uncertainty, not a hardware/population confidence bound",
            "degenerate_interval": interval["degenerate_interval"],
        }
    return result


def analyze(design, design_sha, records):
    policies = {
        p["id"]: summarize_policy(records, design, p["id"]) for p in design["policies"]
    }
    comparisons = []
    plan = design["analysis"]
    for tier, pairs in (
        (
            "primary",
            [plan["primary_comparison"]] if plan.get("primary_comparison") else [],
        ),
        ("exploratory", plan["exploratory_comparisons"]),
        ("secondary", plan["secondary_comparisons"]),
    ):
        for a, b in pairs:
            row = matched_comparison(records, design, a, b)
            row["tier"] = tier
            comparisons.append(row)
    complete = all(p["valid_scored"] == p["scheduled"] for p in policies.values())
    return {
        "schema_version": 1,
        "campaign_id": design["campaign_id"],
        "campaign_sha256": design_sha,
        "status": "complete" if complete else "incomplete",
        "analysis_plan": design["analysis"],
        "thresholds": design["thresholds"],
        "policies": policies,
        "paired_comparisons": comparisons,
        "limitations": [
            "This fixed perturbation suite does not estimate a natural deployment distribution.",
            "Only two sampling seeds per configuration are planned; scenario-resampling intervals do not establish policy equivalence.",
            "Teacher tactile inputs are uncalibrated proxies; teacher contrasts are exploratory.",
            "Physical task completion and safety termination are reported separately.",
            "Missing/invalid trials remain in scheduled-denominator bounds and are listed explicitly.",
        ],
        "trials": [
            {k: v for k, v in records[key].items() if k not in ("metrics", "case")}
            for key in sorted(records)
        ],
        "missing_trial_keys": [
            list(key) for key in planned_keys(design) if key not in records
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, default=REPO / "configs/sim/policy_campaign.json"
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    design, design_sha = load_design(args.campaign)
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_trials(args.runs, design, design_sha, args.out)
    result = analyze(design, design_sha, records)
    (args.out / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "campaign_sha256": design_sha,
                "policies": {
                    k: {
                        f: v[f]
                        for f in ("scheduled", "valid_scored", "invalid", "missing")
                    }
                    for k, v in result["policies"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
