#!/usr/bin/env python3
"""Independently rescore the complete frozen four-cell RPC development diagnostic."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v7"
DRIVER = BASE / "source_teacher_anchor_driver_v7"
SOURCE = BASE / "source_teacher_anchor_rpc_v7"
COMPLETION_HELPER = (
    BASE / "review_teacher_anchor_v5/tools/sim/teacher_anchor_compare.py"
)
CAMPAIGN_SHA = "e68ca523862e805538b9f150c8394363091aa9d9dac6bd950b6215fc148d739f"
PINS = {
    COMPLETION_HELPER: "989f7af2dfbbd42a1e0f748e2553cf2776c87518513168465411e8a6a6f06067",
    ROOT
    / "external_driver_manifest.json": "fdeed0882f7606944540df4dd372293b4d2baecc4ebb659ea2b0adaf935c0deb",
    ROOT
    / "preflight_inputs.json": "323779a4ed6a17e4605491720b364594b89a4f890bfb520c9483d8ab30d54416",
    ROOT
    / "source_audit/source_manifest.json": "1165cb34c53444c89ea76a78c9687e508adfd70b22570c2278060e83afd37084",
    DRIVER
    / "tools/sim/analyze_policy_campaign.py": "11b88baedfa6bed8696abb87fb30a51677be56538267c9fef3ddc60743d307a4",
    DRIVER
    / "tools/sim/run_policy_campaign.py": "235db3d485412da413c9abf866d22c0ff63522fdba70de65b6f7d5234b79aefa",
    DRIVER / "configs/sim/teacher_success_anchor_v7_diagnostic.json": CAMPAIGN_SHA,
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite reported timing")
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve existing independent diagnostic audit")
    case_root = ROOT / "diagnostic"
    design = read(DRIVER / "configs/sim/teacher_success_anchor_v7_diagnostic.json")
    expected = {
        f"{p['id']}__successful_anchor__seed{s}"
        for p in design["policies"]
        for s in [904301, 904302]
    }
    progress = read(case_root / "progress.json")
    if (
        progress.get("status") != "all_trials_completed"
        or progress.get("campaign_sha256") != CAMPAIGN_SHA
        or set(progress.get("trials", {})) != expected
        or len(expected) != 4
        or any(
            r.get("status") != "completed"
            or r.get("exit_code") != 0
            or r.get("reused", False)
            for r in progress["trials"].values()
        )
    ):
        raise RuntimeError("Wait for all four original completed cases")
    checks = 0

    def check(path, expected_sha):
        nonlocal checks
        if sha(path) != expected_sha:
            raise RuntimeError(f"Changed frozen input: {path}")
        checks += 1

    for path, digest in PINS.items():
        check(path, digest)
    check(case_root / "campaign_snapshot.json", CAMPAIGN_SHA)
    driver_manifest = read(ROOT / "external_driver_manifest.json")
    if driver_manifest["source"] != str(DRIVER):
        raise RuntimeError("Driver source identity changed")
    for name, digest in driver_manifest["sha256"].items():
        check(DRIVER / name, digest)
    for name, digest in read(ROOT / "source_audit/source_manifest.json")[
        "output_sha256"
    ].items():
        check(SOURCE / name, digest)
    frozen = read(ROOT / "preflight_inputs.json")
    for hashes, root in [
        ("source_sha256", "source_root"),
        ("external_controller_source_sha256", "external_controller_source_root"),
        ("live_core_sha256", "live_repository"),
        ("episode_sha256", "prepared_episode"),
    ]:
        for name, digest in frozen[hashes].items():
            check(Path(frozen[root]) / name, digest)
    for item in frozen["adapter_profile_inputs"].values():
        check(item["path"], item["sha256"])
    check(frozen["hardware_config"], frozen["hardware_sha256"])
    check(frozen["robot_usd"], frozen["robot_usd_sha256"])
    for policy in design["policies"]:
        check(policy["checkpoint"], policy["checkpoint_sha256"])
    assert design["sampling_seeds"] == [904301, 904302]
    assert {
        p["id"]: p["inference_settings"]["k_seeds"] for p in design["policies"]
    } == {
        "fta1500_nfe1_k4": 4,
        "fta1500_nfe1_k1": 1,
    }
    assert all(p["inference_settings"]["nfe"] == 1 for p in design["policies"])
    assert design["adapter_profile"]["policy_delivery_clock"] == "rpc_wall"
    assert design["adapter_profile"]["servo_reach_limiter"] is False
    sys.path.insert(0, str(DRIVER))
    analyzer = importlib.import_module("tools.sim.analyze_policy_campaign")
    # Reuse only the hash-pinned old-run FINISH reader. Its imports resolve to
    # the already selected v7 driver, including the new RPC-aware analyzer.
    spec = importlib.util.spec_from_file_location(
        "v7_completion_review", COMPLETION_HELPER
    )
    completion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(completion)
    if completion.load_trials is not analyzer.load_trials:
        raise RuntimeError("Completion reader bound a different analyzer")
    if (
        Path(analyzer.__file__).resolve()
        != DRIVER / "tools/sim/analyze_policy_campaign.py"
    ):
        raise RuntimeError("Wrong analyzer imported")
    errors, trials = [], []
    with tempfile.TemporaryDirectory(prefix="v7_independent_four_") as tmp:
        fresh = analyzer.load_trials(
            case_root / "rollouts", design, CAMPAIGN_SHA, Path(tmp)
        )
        if len(fresh) != 4:
            errors.append("Raw score count differs from four")
        for record in fresh.values():
            folder = Path(record["directory"])
            primary_path = case_root / "analysis/trials" / (folder.name + ".json")
            primary = read(primary_path)
            for field in ("status", "metrics", "input_sha256", "case_sha256", "case"):
                if record.get(field) != primary.get(field):
                    errors.append({"case": folder.name, "field": field})
            if (
                record.get("status") != "scored"
                or not record["metrics"]["valid_for_scoring"]
            ):
                errors.append(
                    {
                        "case": folder.name,
                        "invalid": record["metrics"].get("invalid_reasons"),
                    }
                )
            run = read(folder / "run.json")
            execution = completion.read_rows(folder / "execution_trace.jsonl")
            plans = completion.read_rows(folder / "planner_trace.json")
            with np.load(folder / "sim_trace.npz", allow_pickle=False) as trace:
                row = completion.legacy_completion_evidence(
                    record, run, trace, execution, plans, design["thresholds"]
                )
                physics_error = float(np.max(np.abs(trace["physics_t"] - trace["t"])))
                end_time = float(trace["t"][-1])
            if not row["valid_for_selection"]:
                errors.append(
                    {"case": folder.name, "completion_invalid": row["invalid_reasons"]}
                )
            received = [
                p
                for p in plans
                if p.get("status") != "error"
                and p.get("diagnostics", {}).get("sim_policy_delivery_clock")
                == "rpc_wall"
            ]
            native = [
                p["diagnostics"]["sim_native_inference_latency_s"] for p in received
            ]
            rpc = [p["diagnostics"]["sim_policy_replan_wall_time_s"] for p in received]
            outer = [p["inference_wall_time_s"] for p in received]
            for plan in received:
                diag = plan["diagnostics"]
                if not np.isclose(
                    plan["latency_s"],
                    diag["sim_native_inference_latency_s"],
                    atol=1e-9,
                    rtol=0,
                ):
                    errors.append(
                        {
                            "case": folder.name,
                            "native_plan_latency_changed": plan["replan_id"],
                        }
                    )
                if (
                    plan.get("activated_at") is not None
                    and plan["activated_at"] + 1e-9
                    < plan["t"] + diag["sim_effective_delivery_delay_s"]
                ):
                    errors.append(
                        {"case": folder.name, "early_activation": plan["replan_id"]}
                    )
            row.update(
                raw_score_sha256=sha(primary_path),
                physics_clock_max_error_s=physics_error,
                actual_trace_end_s=end_time,
                native_latency_s=stats(native),
                full_client_rpc_s=stats(rpc),
                rpc_minus_native_s=stats(np.asarray(rpc) - np.asarray(native)),
                outer_runner_wall_s=stats(outer),
                outer_minus_rpc_s=stats(np.asarray(outer) - np.asarray(rpc)),
                full_client_timer_plans=len(received),
                received_unactivated_plans=sum(
                    p.get("activated_at") is None for p in received
                ),
            )
            trials.append(row)
    totals = []
    for policy in design["policies"]:
        rows = [r for r in trials if r["policy_id"] == policy["id"]]
        totals.append(
            {
                "policy_id": policy["id"],
                "inference_settings": policy["inference_settings"],
                "trials": len(rows),
                **{k: sum(bool(r[k]) for r in rows) for k in completion.COUNTS},
                "terminal_events_overlapping": dict(
                    Counter(e for r in rows for e in r["terminal_safety_events"])
                ),
                "mean_of_episode_native_s": float(
                    np.mean([r["native_latency_s"]["mean"] for r in rows])
                ),
                "mean_of_episode_full_rpc_s": float(
                    np.mean([r["full_client_rpc_s"]["mean"] for r in rows])
                ),
            }
        )
    result = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "raw_cases_rescored": len(trials),
        "hash_checks": checks,
        "campaign_sha256": CAMPAIGN_SHA,
        "frozen_analyzer_sha256": PINS[DRIVER / "tools/sim/analyze_policy_campaign.py"],
        "policies": totals,
        "trials": trials,
        "clear_winner_claim": False,
        "frozen_prose_erratum": {
            "field": "comparability_requirements",
            "retained_original_text": design["comparability_requirements"],
            "qualification": "Inherited references to four prospective candidates and v3 ranking/winner criteria are stale prose. The pinned executable design has exactly two ftA1500 recipes, K1/K4 at NFE1, times two reused development seeds: four cases. This audit applies no teacher selection or winner test and does not amend the frozen file.",
        },
        "interpretation": "Two reused development seeds per recipe at one fixed simulator anchor. No winner, reliability estimate, automatic expansion or change to v5 reserved selection. Native model duration, complete client call, and outer runner bookkeeping are distinct clocks.",
        "GPU_model_hardware_or_existing_process_actions": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
