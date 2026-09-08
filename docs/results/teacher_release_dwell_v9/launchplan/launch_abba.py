#!/usr/bin/env python3
"""Four release-dwell diagnostic cases; CPU plan only unless explicitly executed."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
ORDER = [
    ("control_904301", 904301, 0.2),
    ("dwell100_904301", 904301, 0.1),
    ("dwell100_904302", 904302, 0.1),
    ("control_904302", 904302, 0.2),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def exclusive(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def verify_files(binding):
    count = 0
    for item in binding["files"]:
        require(sha(item["path"]) == item["sha256"], f"changed: {item['path']}")
        count += 1
    for item in binding["inventories"]:
        require(sha(item["manifest"]) == item["sha256"], "manifest changed")
        values = read(item["manifest"])
        for key in item["files_key"]:
            values = values[key]
        require(isinstance(values, dict) and values, "empty inventory")
        for relative, expected in values.items():
            require(sha(Path(item["root"]) / relative) == expected, relative)
            count += 1
    return count


def validate_design(design, baseline, block_id, seed, dwell):
    require(design["status"] == "frozen", "unfrozen design")
    require(design["sampling_seeds"] == [seed], "wrong single-trial seed")
    require(design["planned_counts"]["total"] == 1, "wrong count")
    require(
        design["execution_phases"]
        == [
            {
                "id": block_id,
                "policy_ids": ["fta1500_nfe1_k4"],
                "condition_ids": ["successful_anchor"],
                "sampling_seeds": [seed],
            }
        ],
        "wrong execution phase",
    )
    # All behavior-affecting fields must match the frozen V8 bounded-hold case,
    # except one declared release dwell. Administrative study fields may change.
    administrative = {
        "campaign_id",
        "frozen_at_utc",
        "freeze_authorization",
        "sampling_seeds",
        "planned_counts",
        "execution_order",
        "comparability_requirements",
        "analysis",
        "execution_phases",
        "development_split_note",
    }
    candidate = json.loads(json.dumps(design))
    candidate["adapter_profile"]["placement_release"]["opening_hold_s"] = 0.2
    require(
        design["adapter_profile"]["placement_release"]["opening_hold_s"] == dwell,
        "wrong release dwell",
    )
    require(set(candidate) == set(baseline), "undeclared design fields")
    for key in candidate:
        if key not in administrative:
            require(candidate[key] == baseline[key], f"undeclared difference: {key}")
    require(design["thresholds"] == baseline["thresholds"], "changed scoring")
    require(design["horizon_s"] == 60, "wrong horizon")
    require(design["post_stop_observation_s"] == 2, "wrong stop tail")


def complete(output, campaign_sha):
    progress = read(output / "progress.json")
    require(progress["status"] == "all_trials_completed", "controller incomplete")
    require(progress["campaign_sha256"] == campaign_sha, "campaign mismatch")
    require(len(progress["trials"]) == 1, "wrong completed denominator")
    require(
        all(x["status"] == "completed" for x in progress["trials"].values()),
        "incomplete trial",
    )
    summary = read(output / "analysis/summary.json")
    require(
        summary["status"] == "complete" and len(summary["trials"]) == 1,
        "incomplete score index",
    )
    require(not summary["missing_trial_keys"], "missing score")
    score = read(summary["trials"][0]["score_path"])
    require(score["metrics"]["valid_for_scoring"], "invalid trial; no retry")
    return {
        "status": "complete_valid",
        "trial_count": 1,
        "campaign_sha256": campaign_sha,
        "summary_sha256": sha(output / "analysis/summary.json"),
        "primary_score_sha256": sha(summary["trials"][0]["score_path"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    binding = read(args.bindings)
    require(binding["status"] == "frozen", "unfrozen binding")
    if args.execute:
        require(binding["cpu_preflight_passed"] is True, "CPU preflight required")
        require(
            binding["root_execution_authorized"] is True,
            "root explicit execution authorization required",
        )
    require(binding["port"] == 7799, "dedicated port required")
    require(
        [x["id"] for x in binding["cases"]] == [x[0] for x in ORDER],
        "ABBA order changed",
    )
    count = verify_files(binding)
    baseline = read(binding["baseline_campaign"])
    driver_root = Path(binding["driver"])
    sys.path.insert(0, str(driver_root))
    driver = module(driver_root / "tools/sim/run_policy_campaign.py", "dwell_driver")
    plans = []
    for row, (block_id, seed, dwell) in zip(binding["cases"], ORDER, strict=True):
        campaign = Path(row["campaign"])
        require(sha(campaign) == row["campaign_sha256"], "campaign digest mismatch")
        design, digest = driver.load_design(campaign)
        validate_design(design, baseline, block_id, seed, dwell)
        require(
            sum(len(x["trials"]) for x in driver.blocks(design)) == 1,
            "driver did not produce exactly one trial",
        )
        output = Path(binding["output_root"]) / block_id
        require(not output.exists(), f"refuse existing output: {output}")
        command = [
            binding["server_python"],
            str(driver_root / "tools/sim/run_policy_campaign.py"),
            "--campaign",
            str(campaign),
            "--source",
            binding["runtime"],
            "--live-repo",
            binding["live_repo"],
            "--server-python",
            binding["server_python"],
            "--evidence",
            binding["evidence"],
            "--hardware-config",
            row["hardware"],
            "--robot-usd",
            binding["robot_usd"],
            "--output",
            str(output),
            "--port",
            str(binding["port"]),
        ]
        parsed = driver.parser().parse_args(command[2:])
        inputs = driver.source_manifest(parsed, digest, design)
        require(
            inputs["hardware_sha256"] == design["runtime_hardware"]["sha256"],
            "effective hardware mismatch",
        )
        plans.append(
            {
                "id": block_id,
                "sampling_seed": seed,
                "opening_hold_s": dwell,
                "command": command,
                "inputs": inputs,
                "output": str(output),
                "campaign_sha256": digest,
            }
        )
    result = {
        "status": "plan_only",
        "planned_trials": 4,
        "order": "ABBA",
        "verified_files": count,
        "binding_sha256": sha(args.bindings),
        "plans": plans,
        "source_changed_from_v8": False,
    }
    if not args.execute:
        print(json.dumps(result, indent=2))
        return
    output_root = Path(binding["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    exclusive(output_root / "launch_plan.json", result)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    for plan in plans:
        verify_files(binding)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", binding["port"]))
        with (output_root / f"{plan['id']}.log").open("x") as log:
            process = subprocess.Popen(
                plan["command"] + ["--execute"],
                cwd=driver_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            exclusive(
                output_root / f"{plan['id']}_launch.json",
                {
                    "pid": process.pid,
                    "command": plan["command"] + ["--execute"],
                    "started_at_utc": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                },
            )
            rc = process.wait()
        require(rc == 0, f"controller exit {rc}; no retry")
        exclusive(
            output_root / f"{plan['id']}_completed.json",
            complete(Path(plan["output"]), plan["campaign_sha256"]),
        )
    exclusive(
        output_root / "complete.json",
        {
            "status": "four_valid_trials_completed",
            "trials": 4,
            "interpretation": "Two reused development seeds; no reliable-winner or hardware claim",
        },
    )


if __name__ == "__main__":
    main()
