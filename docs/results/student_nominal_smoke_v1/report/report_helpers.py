#!/usr/bin/env python3
"""CPU-only rendering of frozen teacher campaign scores; never rescore trials.

Usage: python report_campaign.py --campaign-root ROOT --out ROOT/report
Use --no-plots in a Python environment without Matplotlib. NumPy is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUTCOMES = (
    "reach_before_closure",
    "acquired",
    "lifted",
    "carried",
    "released_in_bin",
    "full_task",
    "dropped",
)
STAGES = (*OUTCOMES[:-1], "support_verified")
LABELS = ("Reach", "Acquire", "Lift", "Carry", "Release", "Full task", "Bin support")
REQUIRED_REPORT_INPUTS = (
    "run.json",
    "execution_trace.jsonl",
    "planner_trace.json",
    "policy_info.json",
)


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.is_file()
        else None
    )


def stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def fmt(value, digits=3):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def latency_samples(plans):
    result = {key: [] for key in ("native", "delivery", "activation", "wall")}
    for p in plans or []:
        if p.get("status") == "inference_error":
            continue
        d = p.get("diagnostics", {})
        request = d.get("sim_inference_request_t", p.get("t_created", p.get("t")))
        active = p.get("activated_at")
        result["native"].append(
            d.get("sim_native_inference_latency_s", p.get("latency_s"))
        )
        result["delivery"].append(d.get("sim_effective_inference_latency_s"))
        result["activation"].append(
            active - request if active is not None and request is not None else None
        )
        result["wall"].append(p.get("inference_wall_time_s"))
    return result


def control_diagnostics(execution, plans, info):
    result = {
        key: None
        for key in (
            "first_stop_time_s",
            "terminal_event_kinds",
            "logged_safety_event_counts",
            "placement_release_commits",
            "placement_release_commit_times_s",
            "placement_release_events",
            "playback_cap_dwell_rows",
            "playback_cap_dwell_s",
            "last_release_phase",
        )
    }
    if execution is None:
        return result
    stops = [r for r in execution if r.get("stopped")]
    first = stops[0] if stops else None
    result["first_stop_time_s"] = first.get("t") if first else None
    result["terminal_event_kinds"] = (
        ";".join(first.get("diagnostics", {}).get("safety_events", [])) if first else ""
    )
    events = Counter(
        str(e)
        for r in execution
        for e in r.get("diagnostics", {}).get("safety_events", [])
    )
    result["logged_safety_event_counts"] = json.dumps(dict(events), sort_keys=True)
    release = [
        (r["t"], r["diagnostics"]["placement_release"])
        for r in execution
        if isinstance(r.get("diagnostics", {}).get("placement_release"), dict)
    ]
    if release:
        commits = sorted(
            {
                float(d["committed_at_s"])
                for _, d in release
                if d.get("committed_at_s") is not None
            }
        )
        for t, d in release:
            if d.get("event") == "policy_release_committed" and not any(
                abs(t - old) < 1e-6 for old in commits
            ):
                commits.append(float(t))
        result.update(
            placement_release_commits=len(commits),
            placement_release_commit_times_s=json.dumps(sorted(commits)),
            placement_release_events=json.dumps(
                [{"t": t, "event": d["event"]} for t, d in release if d.get("event")]
            ),
            last_release_phase=release[-1][1].get("phase"),
        )
    rate = info.get("hardware_effective", {}).get("control", {}).get("action_rate_hz")
    cap_limit = info.get("max_play_steps")
    if plans is None or rate is None:
        return result
    lookup = {p["replan_id"]: p for p in plans if "replan_id" in p}
    capped = []
    for r in execution:
        d = r.get("diagnostics", {})
        p = lookup.get(r.get("active_replan_id"))
        cap = len(p.get("actions", [])) if p else 0
        if cap_limit is not None:
            cap = min(cap, cap_limit)
        play = d.get("play_time_s")
        capped.append(
            bool(
                cap
                and play is not None
                and not r.get("stopped")
                and not d.get("stale_plan_hold")
                and play * rate >= cap - 1e-5
            )
        )
    times = np.asarray([r.get("t", np.nan) for r in execution], float)
    if np.isfinite(times).all() and (len(times) < 2 or np.all(np.diff(times) > 0)):
        result["playback_cap_dwell_rows"] = sum(capped)
        result["playback_cap_dwell_s"] = float(np.dot(np.diff(times), capped[:-1]))
    return result


def trial_row(root, policy, condition, seed, score, hint, campaign_hash):
    case_id = f"{policy}__{condition['id']}__seed{seed}"
    folder = root / "rollouts" / case_id
    # Local mirrors retain the same rollouts names; do not follow remote absolute paths.
    if (
        not folder.is_dir()
        and score
        and score.get("directory")
        and Path(score["directory"]).is_dir()
    ):
        folder = Path(score["directory"])
    score = score or {}
    metric = score.get("metrics") or {}
    if score and score.get("case", {}).get("campaign_sha256") != campaign_hash:
        raise ValueError(f"Frozen hash mismatch in {case_id}")
    valid = score.get("status") == "scored" and metric.get("valid_for_scoring") is True
    if valid and any(
        not isinstance(metric.get("outcomes", {}).get(k), bool) for k in OUTCOMES
    ):
        raise ValueError(
            f"Scored trial has missing/non-boolean physical outcome: {case_id}"
        )
    row = {
        "case_id": case_id,
        "policy_id": policy,
        "condition_id": condition["id"],
        "family": condition["family"],
        "sampling_seed": seed,
        "score_status": score.get("status", hint.get("status", "missing_score")),
        "valid_for_scoring": valid,
        "invalid_reasons": ";".join(metric.get("invalid_reasons", [])),
        "missing_scoring_inputs": ";".join(
            score.get("missing_inputs", hint.get("missing_inputs", []))
        ),
        "missing_report_inputs": ";".join(
            name for name in REQUIRED_REPORT_INPUTS if not (folder / name).is_file()
        ),
        "raw_directory": str(folder),
        "authoritative_directory": score.get("directory", hint.get("directory")),
        "observation_delay_s": condition.get("observation_delay_s"),
        "inference_delay_add_s": condition.get("inference_delay_add_s"),
        "collision_coverage": metric.get("collisions", {}).get("observability"),
        "sensor_uncertainty": "Uncalibrated tactile and wrist proxies; visual/contact geometry estimated",
    }
    for name, expected in score.get("input_sha256", {}).items():
        f = folder / name
        if f.is_file() and sha(f) != expected:
            raise ValueError(f"Raw input changed since scoring: {f}")
    for key in OUTCOMES:
        observed = metric.get("outcomes", {}).get(key)
        row[f"observed_{key}"] = observed
        row[f"scored_{key}"] = observed if valid else None
    support = metric.get("placement_support", {})
    row.update(
        observed_support_verified=support.get("verified_placement"),
        scored_support_verified=support.get("verified_placement") if valid else None,
        support_required=support.get("required"),
        support_telemetry_status=support.get("telemetry_status"),
        support_confirmation_s=support.get("confirmation_time_s"),
        packet_robot_contact_peak_n=support.get("robot_contact_peak_n"),
        packet_bin_contact_peak_n=support.get("bin_contact_peak_n"),
    )
    for key, val in metric.get("event_times_s", {}).items():
        row[f"event_{key}_s"] = val
    for key in (
        "max_lift_m",
        "carry_distance_m",
        "final_inside_bin",
        "final_contacts_unloaded",
        "final_speed_m_s",
        "final_angular_speed_rad_s",
        "minimum_pad_midpoint_to_object_m",
        "bilateral_contact_duration_s",
        "pad_packet_normal_force_peak_n",
    ):
        row[key] = metric.get("object", {}).get(key)
    reach = metric.get("reach_diagnostic", {})
    row["reach_distance_before_first_additional_closure_m"] = reach.get(
        "reach_error_before_first_closing_motion_m"
    )
    row["first_additional_closing_command_s"] = reach.get("first_closing_motion_s")
    for key in (
        "stop_reason",
        "ik_rejects",
        "hold_duration_s",
        "stale_plan_hold_rows",
        "replans",
        "plans_activated",
        "inference_errors",
        "execution_rows",
    ):
        row[key] = metric.get("control", {}).get(key)
    row["collision_reported_count"] = metric.get("collisions", {}).get("reported_count")
    row["pad_environment_force_peak_n"] = metric.get("collisions", {}).get(
        "pad_environment_force_peak_n"
    )
    row["physics_clock_max_error_s"] = metric.get("trace_sampling", {}).get(
        "physics_clock_max_error_s"
    )
    run = read(folder / "run.json", {})
    execution = load_jsonl(folder / "execution_trace.jsonl")
    plans = read(folder / "planner_trace.json")
    info = read(folder / "policy_info.json", {})
    row["duration_s"] = run.get("duration_s")
    row["runtime_status"] = read(folder / "run_status.json", {}).get("status")
    row.update(control_diagnostics(execution, plans, info))
    if row["stop_reason"] is None:
        row["stop_reason"] = run.get("policy_stop_reason")
    row["terminal_cause"] = row["terminal_event_kinds"] or row["stop_reason"]
    if row["terminal_cause"] is None and run:
        row["terminal_cause"] = "horizon_no_controller_stop"
    completion = metric.get("event_times_s", {}).get("full_task")
    stopped_at = row["first_stop_time_s"]
    row["physical_completion_before_later_stop"] = (
        bool(
            completion is not None
            and stopped_at is not None
            and stopped_at > completion
        )
        if valid and execution is not None
        else None
    )
    row["completion_to_later_stop_s"] = (
        stopped_at - completion
        if row["physical_completion_before_later_stop"]
        else None
    )
    final_flags = (row["final_inside_bin"], row["final_contacts_unloaded"])
    row["full_task_final_inside_and_unloaded"] = (
        bool(row["scored_full_task"] and all(final_flags))
        if valid and all(v is not None for v in final_flags)
        else None
    )
    row["packet_robot_contact_final_n"] = None
    row["packet_bin_contact_final_n"] = None
    trace_path = folder / "sim_trace.npz"
    if trace_path.is_file():
        with np.load(trace_path, allow_pickle=False) as trace:
            for source, target in (
                ("packet_robot_normal_force", "packet_robot_contact_final_n"),
                ("packet_bin_normal_force", "packet_bin_contact_final_n"),
            ):
                if source in trace and len(trace[source]):
                    row[target] = float(trace[source][-1])
    support_final = (
        row["packet_robot_contact_final_n"],
        row["packet_bin_contact_final_n"],
    )
    limits = metric.get("thresholds", {})
    row["full_task_final_robot_unloaded_bin_supported"] = (
        bool(
            row["full_task_final_inside_and_unloaded"]
            and support_final[0] <= limits.get("support_robot_force_max_n", 0.1)
            and support_final[1] > limits.get("support_bin_force_min_n", 0.1)
        )
        if valid and all(v is not None for v in support_final)
        else None
    )
    row["packet_robot_peak_over_100n"] = (
        row["packet_robot_contact_peak_n"] > 100
        if row["packet_robot_contact_peak_n"] is not None
        else None
    )
    samples = latency_samples(plans)
    for kind, values in samples.items():
        for key, value in stats(values).items():
            row[
                f"latency_{kind}_{key}_s" if key != "count" else f"latency_{kind}_count"
            ] = value if plans is not None else None
    # Do not mislabel the frozen evaluator's effective_latency_s as delivery latency.
    if plans is None:
        native = metric.get("control", {}).get("effective_latency_s", {})
        for key in ("count", "mean", "p95", "max"):
            row[
                f"latency_native_{key}_s" if key != "count" else "latency_native_count"
            ] = native.get(key)
    candidates = [
        root.parent / "video_reviews" / case_id / "policy_review.mp4",
        root / "video_reviews" / case_id / "policy_review.mp4",
        folder / "policy_review.mp4",
    ]
    video = next((p for p in candidates if p.is_file()), candidates[0])
    row["video_path"] = str(video)
    row["video_present"] = video.is_file()
    row["scene_video_path"] = str(folder / "sim.mp4")
    row["scene_video_present"] = (folder / "sim.mp4").is_file()
    return row, samples


def aggregate(rows):
    valid = [r for r in rows if r["valid_for_scoring"]]
    counts = {key: sum(r[f"scored_{key}"] is True for r in valid) for key in STAGES}
    successes = counts["full_task"]
    return {
        "scheduled": len(rows),
        "valid": len(valid),
        "invalid": sum(r["score_status"] == "invalid" for r in rows),
        "missing_or_pending": sum(
            not r["valid_for_scoring"] and r["score_status"] != "invalid" for r in rows
        ),
        "stage_counts": counts,
        "support_known": sum(r["scored_support_verified"] is not None for r in valid),
        "drops": sum(r["scored_dropped"] is True for r in valid),
        "controller_stops": sum(bool(r["stop_reason"]) for r in valid),
        "physical_completions_before_later_stop": sum(
            r.get("physical_completion_before_later_stop") is True for r in valid
        ),
        "full_tasks_final_inside_and_unloaded": sum(
            r.get("full_task_final_inside_and_unloaded") is True for r in valid
        ),
        "completed_then_stopped_and_final_inside_unloaded": sum(
            r.get("physical_completion_before_later_stop") is True
            and r.get("full_task_final_inside_and_unloaded") is True
            for r in valid
        ),
        "packet_robot_peak_over_100n_trials": sum(
            r.get("packet_robot_peak_over_100n") is True for r in valid
        ),
        "full_tasks_final_robot_unloaded_bin_supported": sum(
            r.get("full_task_final_robot_unloaded_bin_supported") is True for r in valid
        ),
        "final_support_known": sum(
            r.get("full_task_final_robot_unloaded_bin_supported") is not None
            for r in valid
        ),
        "full_task_rate_valid": successes / len(valid) if valid else None,
        "scheduled_rate_bounds": [
            successes / len(rows),
            (successes + len(rows) - len(valid)) / len(rows),
        ]
        if rows
        else None,
    }


def selected_videos(rows):
    selected = []
    for role, candidates in (
        ("nominal", [r for r in rows if r["family"] == "nominal"]),
        (
            "earliest_full_success",
            [r for r in rows if r["scored_full_task"] is True][:1],
        ),
        ("first_drop", [r for r in rows if r["scored_dropped"] is True][:1]),
    ):
        for row in candidates:
            selected.append(
                {
                    "role": role,
                    "case_id": row["case_id"],
                    "video_path": row["video_path"],
                    "video_relative_path": row.get("video_relative_path"),
                    "present": row["video_present"],
                }
            )
    return selected


def seed_disagreement(rows, conditions, seeds):
    lookup = {(r["condition_id"], r["sampling_seed"]): r for r in rows}
    result = {}
    for outcome in STAGES:
        complete, disagree, excluded = [], [], []
        for c in conditions:
            group = [lookup[(c["id"], s)] for s in seeds]
            if all(
                r["valid_for_scoring"] and r[f"scored_{outcome}"] is not None
                for r in group
            ):
                complete.append(c["id"])
                if len({r[f"scored_{outcome}"] for r in group}) > 1:
                    disagree.append(c["id"])
            else:
                excluded.append(c["id"])
        result[outcome] = {
            "complete_conditions": len(complete),
            "disagreements": len(disagree),
            "condition_ids": disagree,
            "excluded_condition_ids": excluded,
        }
    return result


def make_plots(rows, design, summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    conditions, seeds = design["conditions"], design["sampling_seeds"]
    lookup = {(r["condition_id"], r["sampling_seed"]): r for r in rows}
    colors = [
        "#e5e7eb",
        "#dd87a0",
        "#485260",
        "#719dc6",
        "#4aa9b9",
        "#418881",
        "#8171b1",
        "#dbad53",
        "#50a764",
    ]
    names = [
        "Missing / pending",
        "Invalid",
        "No stage",
        "Reach",
        "Acquire",
        "Lift",
        "Carry",
        "Release",
        "Full task",
    ]
    matrix = np.zeros((len(conditions), len(seeds)), int)
    text = {}
    for i, c in enumerate(conditions):
        for j, seed in enumerate(seeds):
            r = lookup[(c["id"], seed)]
            if not r["valid_for_scoring"]:
                value = 1 if r["score_status"] == "invalid" else 0
                annotation = "Invalid" if value == 1 else "Pending"
            else:
                achieved = [
                    k for k, field in enumerate(OUTCOMES[:-1]) if r[f"scored_{field}"]
                ]
                value = 3 + max(achieved) if achieved else 2
                annotation = names[value]
                if r["scored_dropped"]:
                    annotation += " · D"
                if r["scored_support_verified"]:
                    annotation += " · S"
                if r["stop_reason"]:
                    annotation += "\nstop"
            matrix[i, j], text[i, j] = value, annotation
    fig, ax = plt.subplots(figsize=(9.5, max(7, 0.37 * len(conditions) + 2.2)))
    ax.imshow(
        matrix,
        cmap=ListedColormap(colors),
        norm=BoundaryNorm(np.arange(-0.5, 9.5), 9),
        aspect="auto",
    )
    ax.set_xticks(range(len(seeds)), [f"Seed {s}" for s in seeds])
    ax.set_yticks(range(len(conditions)), [c["id"] for c in conditions], fontsize=8)
    for (i, j), annotation in text.items():
        ax.text(
            j,
            i,
            annotation,
            ha="center",
            va="center",
            fontsize=8,
            color="white" if matrix[i, j] in (2, 5, 6) else "#14202b",
        )
    for i in range(1, len(conditions)):
        if conditions[i]["family"] != conditions[i - 1]["family"]:
            ax.axhline(i - 0.5, color="white", lw=2.5)
    total = summary["overall"]
    ax.set_title(
        f"Teacher pick-to-box: {total['valid']}/{total['scheduled']} valid\nHighest observed stage; D = drop, S = verified bin support",
        fontsize=12,
        pad=16,
    )
    fig.legend(
        [Patch(facecolor=c) for c in colors],
        names,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"outcome_heatmap.{ext}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, name in zip(axes, ("nominal", "stress"), strict=True):
        data = summary[name]
        counts = [data["stage_counts"][key] for key in STAGES]
        bars = ax.bar(range(len(STAGES)), counts, color="#4c86a3")
        for i, bar in enumerate(bars):
            label = (
                "NA"
                if i == len(STAGES) - 1 and data["support_known"] == 0
                else str(counts[i])
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025 * max(1, data["valid"]),
                label,
                ha="center",
                fontsize=9,
            )
        ax.set_xticks(range(len(STAGES)), LABELS, rotation=30, ha="right")
        ax.set_ylim(0, max(1, data["valid"]) * 1.2)
        ax.set_ylabel("Valid trials reaching stage")
        ax.set_title(f"{name.capitalize()}: {data['valid']}/{data['scheduled']} valid")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Physical outcomes; missing/invalid excluded from counts and retained in planned denominators"
    )
    for ext in ("png", "pdf"):
        fig.savefig(out / f"stage_counts.{ext}", dpi=220)
    plt.close(fig)


def markdown(design, summary, rows, out):
    total = summary["overall"]
    policy = next(
        (p for p in design["policies"] if p["id"] == summary["policy_id"]), {}
    )
    settings = design.get("inference_settings", {})
    setup_note = (
        "The excluded release-controller setup achieved strict full placement at 30.268 s. "
        f"The scored nominal pair achieved {summary['nominal']['stage_counts']['full_task']}/{summary['nominal']['scheduled']}. "
        "Matching the sampling seed does not guarantee bitwise identical closed-loop trajectories. Measured latency and physics can vary, but this difference is not attributed to latency without a paired ablation."
        if design.get("campaign_id") == "teacher_pick_place_v1"
        else "Setup provenance is preserved in the frozen design where available."
    )
    lines = [
        f"# {design['campaign_id']}: teacher pick-to-box",
        "",
        f"**{summary['report_status']}** — {total['valid']}/{total['scheduled']} planned trials valid; {total['invalid']} invalid; {total['missing_or_pending']} missing or pending. This report renders frozen scores and never changes outcome thresholds.",
        "",
        f"Frozen design: [`{summary['campaign_sha256']}`](../campaign_snapshot.json). Checkpoint: `{policy.get('checkpoint', 'NA')}`; SHA256 `{policy.get('checkpoint_sha256', 'NA')}`. Task text `{settings.get('task_text', 'NA')}`, NFE {settings.get('nfe', 'NA')}, K {settings.get('k_seeds', 'NA')}, EMA {settings.get('use_ema', 'NA')}; planned horizon {design.get('horizon_s', 'NA')} s. Exact inference, controller, scene and threshold settings are preserved in the snapshot.",
        "",
        "Only the scheduled campaign trials enter these results; setup and replay preflights are excluded. A physical completion followed by later safety termination remains a completion with a separately reported stop. Post-stop physics observation is capped by the fixed horizon.",
        "",
        setup_note,
        "",
        f"Full task: **{total['stage_counts']['full_task']}/{total['valid']} valid trials**. Scheduled-denominator bounds: {total['scheduled_rate_bounds']}. Unknown trials are not automatically failures. Placement requires the frozen ordered physical stages and verified robot unloading/bin support; controller release commits are only diagnostics.",
        "",
        f"Physical completion preceded a later controller stop in **{total['physical_completions_before_later_stop']}** trials. Of these, **{total['completed_then_stopped_and_final_inside_unloaded']}** still ended inside the bin with pads unloaded. Across all full successes, **{total['full_tasks_final_inside_and_unloaded']}** had that final state. A later stop does not erase earlier verified placement; time ordering and final-state columns are preserved in the CSV.",
        "",
        f"Final-sample whole-robot unloading and positive bin support are additionally verified for **{total['full_tasks_final_robot_unloaded_bin_supported']}** full successes ({total['final_support_known']} valid trials have the required final-force telemetry). This final-sample diagnostic is separate from the scorer's sustained support window.",
        "",
        "| Group | Valid / planned | Reach | Acquire | Lift | Carry | Release | Full task | Verified support / known | Drops | Controller stops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in [
        ("Nominal", summary["nominal"]),
        ("All stress", summary["stress"]),
        *summary["families"].items(),
    ]:
        if name == "nominal":
            continue
        counts = " | ".join(str(group["stage_counts"][s]) for s in OUTCOMES[:-1])
        lines.append(
            f"| {name} | {group['valid']} / {group['scheduled']} | {counts} | {group['stage_counts']['support_verified']} / {group['support_known']} | {group['drops']} | {group['controller_stops']} |"
        )
    lines += ["", "## Uncertainty and seed dependence", ""]
    interval = summary.get("frozen_scenario_success_interval")
    if interval:
        lines.append(
            f"Frozen analyzer descriptive scenario interval: mean **{fmt(interval.get('mean'))}**, interval **{interval.get('interval')}** at confidence level {fmt(interval.get('confidence_level'))}, using {interval.get('complete_configurations')} complete configuration clusters. This value is copied unchanged from analysis/summary.json."
        )
    else:
        lines.append(
            "No scenario-cluster interval is available in the current frozen analyzer summary (NA). No interval is invented or recomputed here."
        )
    lines += [
        "",
        "Clusters retain both episode seeds. This fixed perturbation suite does not estimate a natural hardware/deployment distribution. A degenerate interval does not establish robustness or equivalence. Nominal has only the planned seed replicates; stress families are kept separate.",
        "",
        design.get("development_split_note", "Development-split provenance: NA."),
        "",
        "| Stage | Seed-disagreeing / complete conditions | Condition IDs |",
        "|---|---:|---|",
    ]
    for key, data in summary["seed_disagreement"].items():
        lines.append(
            f"| {key} | {data['disagreements']} / {data['complete_conditions']} | {', '.join(data['condition_ids']) or '—'} |"
        )
    lines += [
        "",
        "## Execution and latency",
        "",
        "Terminal causes are separated from nonterminal logged safety events. `safety_stop` alone is expanded using the first stopped execution row when available. Every nonempty controller stop counts, including non-safety termination. Missing diagnostic fields remain NA.",
        "",
        "| Terminal cause | Valid trials |",
        "|---|---:|",
    ]
    for cause, count in summary["terminal_causes"].items():
        lines.append(f"| {cause} | {count} |")
    lines += ["", "| Metric | Total / known trials |", "|---|---:|"]
    for key, value in summary["control_totals"].items():
        lines.append(f"| {key} | {fmt(value['sum'])} / {value['known_trials']} |")
    lines += [
        "",
        "| Latency source | Requests | Mean | Median | p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, values in summary["latency_pooled"].items():
        lines.append(
            f"| {kind} | {values['count']} | {fmt(values['mean'])} | {fmt(values['median'])} | {fmt(values['p95'])} |"
        )
    lines += [
        "",
        "Latency pools available valid-trial planner requests: native is measured inference, delivery adds the declared response delay, activation is activation time minus request time, and wall measures remote call overhead. The evaluator's field `control.effective_latency_s` is native planner latency, not delayed delivery. CSV per-trial native summaries can fall back to that field if raw planners are unavailable; pooled figures do not synthesize samples.",
        "",
        "Compute was shared rather than GPU-isolated. Frozen policy/controller settings and matched episode seeds therefore do not imply identical inference latency or feedback trajectories. The timing of any resident-workload change is unknown; no in-campaign change is asserted. Native latency is measured per trial, and no resident process was restarted by this report or monitor.",
        "",
        "Capped playback dwell integrates the logged clock at the action cap, excluding stopped/stale rows. It is distinct from rejected/stale hold duration and is not proof of stationary motors. Release-commit counts are unique logged commit times; a commit is not physical placement.",
        "",
        "## Completeness and limits",
        "",
        "Collision coverage is partial. Packet-to-robot and packet-to-bin positive normal magnitudes verify placement support; they do not certify absence of every robot/environment collision. Gel and wrist inputs remain uncalibrated proxies, and scene/contact geometry remains estimated. This is a teacher plus explicit placement-controller experiment, not an untouched-controller or real-hardware success claim.",
        "",
        f"Packet-to-robot positive normal contact exceeded 100 N in **{total['packet_robot_peak_over_100n_trials']}** valid trials. Such peaks can occur in failed approaches. Rigid-packet geometry and contact-force realism remain uncertain; the wrist-wrench proxy does not observe every robot-body contact. These diagnostics do not change the frozen scores.",
        "",
        "| Case | Status | Invalid / missing details |",
        "|---|---|---|",
    ]
    unresolved = [r for r in rows if not r["valid_for_scoring"]]
    for r in unresolved:
        lines.append(
            f"| {r['case_id']} | {r['score_status']} | {r['invalid_reasons'] or r['missing_scoring_inputs'] or 'No completed score'} |"
        )
    if not unresolved:
        lines.append("| All planned cases | Valid | No unresolved scoring cases |")
    incomplete_raw = [r["case_id"] for r in rows if r["missing_report_inputs"]]
    lines += [
        "",
        f"Locally incomplete raw diagnostic inputs: **{len(incomplete_raw)}** trials; CSV lists the missing filenames. Frozen score validity is preserved while missing raw diagnostics stay NA.",
        "",
        "## Representative videos",
        "",
        "Selection is deterministic in frozen condition/seed order: both nominal trials, first valid full success, and first valid drop. All planned video paths remain in the CSV; unavailable local files are labeled.",
        "",
    ]
    for video in summary["representative_videos"]:
        path = Path(video["video_path"])
        target = video.get("video_relative_path") or os.path.relpath(path, out)
        lines.append(
            f"- {video['role']}: [{video['case_id']}]({target}) — {'available' if video['present'] else 'not present in this mirror'}. "
        )
    lines += [
        "",
        "## All planned trial videos",
        "",
        "Relative links remain valid when the campaign and adjacent video_reviews directory are mirrored together. Availability was checked on the authoritative report host at generation time.",
        "",
        "| Case | Video | Availability |",
        "|---|---|---|",
    ]
    for row in rows:
        target = row.get("video_relative_path") or os.path.relpath(
            Path(row["video_path"]), out
        )
        lines.append(
            f"| {row['case_id']} | [Review]({target}) | {'Present' if row['video_present'] else 'Pending / unavailable'} |"
        )
    lines += [
        "",
        "[All trials CSV](per_trial.csv) · [Machine-readable report](report_summary.json) · [Provenance](report_manifest.json)",
        "",
    ]
    if summary["plots_generated"]:
        lines += [
            "[Outcome heatmap](outcome_heatmap.png) ([PDF](outcome_heatmap.pdf)) · [Stage counts](stage_counts.png) ([PDF](stage_counts.pdf))",
            "",
        ]
    return "\n".join(lines)


def generate(root, out, policy="teacher", no_plots=False):
    snapshot = root / "campaign_snapshot.json"
    design = read(snapshot)
    if design is None:
        raise FileNotFoundError(snapshot)
    if policy not in {p["id"] for p in design["policies"]}:
        raise ValueError(f"Policy {policy} absent from frozen design")
    campaign_hash = sha(snapshot)
    frozen = read(root / "analysis/summary.json", {})
    if frozen and frozen.get("campaign_sha256") != campaign_hash:
        raise ValueError("Analyzer summary differs from frozen campaign hash")
    scores = {}
    score_files = []
    for path in sorted((root / "analysis/trials").glob("*.json")):
        s = read(path)
        if s.get("policy_id") != policy:
            continue
        key = (s["condition_id"], int(s["sampling_seed"]))
        if key in scores:
            raise ValueError(f"Duplicate score: {key}")
        scores[key] = s
        score_files.append(path)
    expected = {
        (c["id"], int(s))
        for c in design["conditions"]
        for s in design["sampling_seeds"]
    }
    if scores.keys() - expected:
        raise ValueError("Unplanned score identities")
    hints = {
        (s["condition_id"], int(s["sampling_seed"])): s
        for s in frozen.get("trials", [])
        if s.get("policy_id") == policy
    }
    rows, samples = [], {}
    for condition in design["conditions"]:
        for seed in design["sampling_seeds"]:
            key = (condition["id"], int(seed))
            row, timing = trial_row(
                root,
                policy,
                condition,
                seed,
                scores.get(key),
                hints.get(key, {}),
                campaign_hash,
            )
            row["video_relative_path"] = os.path.relpath(Path(row["video_path"]), out)
            row["scene_video_relative_path"] = os.path.relpath(
                Path(row["scene_video_path"]), out
            )
            rows.append(row)
            samples[row["case_id"]] = timing
    overall = aggregate(rows)
    frozen_policy = frozen.get("policies", {}).get(policy, {})
    if frozen_policy and overall["valid"] != frozen_policy["valid_scored"]:
        raise ValueError(
            "Scores and summary are from different snapshots; retry after analyzer completes"
        )
    if (
        frozen_policy
        and overall["stage_counts"]["full_task"]
        != frozen_policy["outcomes"]["full_task"]["observed_true"]
    ):
        raise ValueError("Full-task count differs from frozen summary")
    valid = [r for r in rows if r["valid_for_scoring"]]
    control_totals = {}
    for key in (
        "ik_rejects",
        "hold_duration_s",
        "stale_plan_hold_rows",
        "playback_cap_dwell_rows",
        "playback_cap_dwell_s",
        "placement_release_commits",
    ):
        values = [r[key] for r in valid if r[key] is not None]
        control_totals[key] = {
            "sum": sum(values) if values else None,
            "known_trials": len(values),
        }
    summary = {
        "campaign_id": design["campaign_id"],
        "campaign_sha256": campaign_hash,
        "policy_id": policy,
        "report_status": "Complete"
        if overall["valid"] == overall["scheduled"]
        else "Partial / unresolved",
        "overall": overall,
        "nominal": aggregate([r for r in rows if r["family"] == "nominal"]),
        "stress": aggregate([r for r in rows if r["family"] != "nominal"]),
        "families": {
            f: aggregate([r for r in rows if r["family"] == f])
            for f in dict.fromkeys(r["family"] for r in rows)
        },
        "seed_disagreement": seed_disagreement(
            rows, design["conditions"], design["sampling_seeds"]
        ),
        "frozen_scenario_success_interval": frozen_policy.get(
            "scenario_success_interval"
        ),
        "control_totals": control_totals,
        "terminal_causes": dict(Counter(r["terminal_cause"] or "NA" for r in valid)),
        "latency_pooled": {
            kind: stats([value for r in valid for value in samples[r["case_id"]][kind]])
            for kind in ("native", "delivery", "activation", "wall")
        },
        "representative_videos": selected_videos(rows),
        "plots_generated": not no_plots,
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_trial.csv").open("w", newline="") as handle:
        columns = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {k: "NA" if r.get(k) is None else r.get(k) for k in columns} for r in rows
        )
    if not no_plots:
        make_plots(rows, design, summary, out)
    (out / "report_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (out / "report.md").write_text(markdown(design, summary, rows, out))
    inputs = [snapshot, *score_files]
    if (root / "analysis/summary.json").is_file():
        inputs.append(root / "analysis/summary.json")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(root),
        "report_policy": policy,
        "expected_rows": len(expected),
        "actual_rows": len(rows),
        "source_sha256": sha(Path(__file__)),
        "report_csv_sha256": sha(out / "per_trial.csv"),
        "input_sha256": {str(p): sha(p) for p in inputs},
        "missing_values": "NA in CSV; null in JSON. Zero only when observed count is zero.",
        "scoring": "Frozen analyzer values only; no thresholds or interval calculations changed.",
        "selection_order": "Frozen conditions then sampling_seeds; earliest means first in this planned order, not best-looking video.",
    }
    (out / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def plots_only(root, out):
    """Render authoritative compact results without touching partial raw mirrors."""
    design = read(root / "campaign_snapshot.json")
    summary = read(out / "report_summary.json")
    manifest = read(out / "report_manifest.json")
    if summary["campaign_sha256"] != sha(root / "campaign_snapshot.json"):
        raise ValueError("Plot snapshot/report hash mismatch")
    if manifest.get("report_csv_sha256") != sha(out / "per_trial.csv"):
        raise ValueError("Authoritative report CSV hash mismatch")
    with (out / "per_trial.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if value == "NA":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
        row["sampling_seed"] = int(row["sampling_seed"])
    if len(rows) != manifest["expected_rows"]:
        raise ValueError("Plot CSV row count differs from frozen report")
    make_plots(rows, design, summary, out)
    summary["plots_generated"] = True
    (out / "report_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    doc = (out / "report.md").read_text()
    if "outcome_heatmap.png" not in doc:
        doc += "\n[Outcome heatmap](outcome_heatmap.png) ([PDF](outcome_heatmap.pdf)) · [Stage counts](stage_counts.png) ([PDF](stage_counts.pdf))\n"
        (out / "report.md").write_text(doc)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", default="teacher")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Render from authoritative compact report files without reading raw trial mirrors",
    )
    args = parser.parse_args()
    result = (
        plots_only(args.campaign_root.resolve(), args.out.resolve())
        if args.plots_only
        else generate(
            args.campaign_root.resolve(), args.out.resolve(), args.policy, args.no_plots
        )
    )
    print(
        json.dumps(
            {
                "report_status": result["report_status"],
                "overall": result["overall"],
                "out": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
