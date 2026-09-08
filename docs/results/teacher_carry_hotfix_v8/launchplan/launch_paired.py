#!/usr/bin/env python3
"""Specific six-case carry-hold diagnostic; plan-only unless --execute.

Root supplies reviewed frozen bindings after integration tests. No hardware
commands, retries, source mutations, or process killing are implemented here.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CHECKPOINT = "67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e"
SEEDS = [904301, 904302]
CENTER = [-0.3937067184864266, -0.2930692769792478, 0.07000000000000008]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exclusive(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def verify_files(binding):
    count = 0
    for item in binding["files"]:
        require(sha(item["path"]) == item["sha256"], f"file changed: {item['path']}")
        count += 1
    for item in binding["inventories"]:
        require(
            sha(item["manifest"]) == item["sha256"],
            f"manifest changed: {item['manifest']}",
        )
        values = read(item["manifest"])
        for key in item["files_key"]:
            values = values[key]
        require(isinstance(values, dict) and values, "empty inventory")
        for relative, expected in values.items():
            require(
                sha(Path(item["root"]) / relative) == expected,
                f"inventory changed: {relative}",
            )
            count += 1
    return count


def validate_design(design, hold, limiter=True):
    require(design["status"] == "frozen", "campaign must be frozen")
    require(design["sampling_seeds"] == SEEDS, "wrong paired seeds")
    require(design["horizon_s"] == 60, "full60s horizon required")
    require(design["post_stop_observation_s"] == 2, "preserve actual-stop tail")
    require(len(design["conditions"]) == 1, "one fixed condition required")
    c = design["conditions"][0]
    require(
        c["object_offset_m"] == [0, 0] and c["family"] == "nominal",
        "do not apply object offset twice",
    )
    require(
        c["friction_scale"] == 1 and c["camera_override"] is None,
        "unchanged scene required",
    )
    require(
        c["observation_delay_s"] == c["inference_delay_add_s"] == 0, "no injected delay"
    )
    require(
        design["nominal_scene"]["waffle"]["center"] == CENTER, "wrong packet anchor"
    )
    require(len(design["policies"]) == 1, "teacher only")
    p = design["policies"][0]
    require(
        p["checkpoint_sha256"] == CHECKPOINT and p["architecture"] == "teacher",
        "wrong teacher",
    )
    recipe = dict(design["inference_settings"])
    recipe.update(p.get("inference_settings", {}))
    for k, v in {
        "use_ema": True,
        "nfe": 1,
        "k_seeds": 4,
        "guidance": 1.0,
        "parity": True,
        "persistent_noise": True,
        "task_text": "waffles",
        "max_play": 10,
    }.items():
        require(recipe[k] == v, f"wrong inference setting {k}")
    profile = design["adapter_profile"]
    require(
        profile["policy_delivery_clock"] == "rpc_wall", "full client clock required"
    )
    require(profile["servo_reach_limiter"] is limiter, "wrong limiter condition")
    require(
        profile.get("servo_constraint_hold_s") == hold, "wrong bounded-hold setting"
    )
    require(
        profile["placement_release"]["finish_after_release"] is True, "FINISH required"
    )
    safety = design["runtime_hardware"]["effective_model"]["safety"]
    for key, val in {
        "elbow_min_rad": 0.40 if limiter else None,
        "servo_joint_speed_max_rad_s": 1.0 if limiter else None,
        "wrist_extension_stop_m": 0.468,
        "servo_constraint_hold_s": hold,
    }.items():
        require(safety.get(key) == val, f"wrong hardware setting {key}")
    require(
        design["thresholds"]["require_support_verified_release"] is True,
        "strict support scorer required",
    )


def matched(a, b):
    # Exact hardware equality apart from the declared optional hold field.
    ah = json.loads(json.dumps(a["runtime_hardware"]["effective_model"]))
    bh = json.loads(json.dumps(b["runtime_hardware"]["effective_model"]))
    for key in (
        "servo_constraint_hold_s",
        "elbow_min_rad",
        "servo_joint_speed_max_rad_s",
    ):
        ah["safety"].pop(key, None)
        bh["safety"].pop(key, None)
    require(ah == bh, "undeclared hardware difference")
    ap, bp = dict(a["adapter_profile"]), dict(b["adapter_profile"])
    for key in ("servo_constraint_hold_s", "servo_reach_limiter"):
        ap.pop(key, None)
        bp.pop(key, None)
    require(ap == bp, "undeclared adapter difference")
    for key in [
        "nominal_scene",
        "prepared_episode",
        "policies",
        "inference_settings",
        "thresholds",
        "sampling_seeds",
        "conditions",
        "initialization_requirements",
    ]:
        require(a[key] == b[key], f"undeclared paired difference: {key}")


def complete(output, campaign_sha):
    progress = read(output / "progress.json")
    require(progress["status"] == "all_trials_completed", "controller incomplete")
    require(progress["campaign_sha256"] == campaign_sha, "wrong completed design")
    require(
        len(progress["trials"]) == 2
        and all(v["status"] == "completed" for v in progress["trials"].values()),
        "incomplete two-case grid",
    )
    summary = read(output / "analysis/summary.json")
    require(
        summary["status"] == "complete" and len(summary["trials"]) == 2,
        "analysis incomplete",
    )
    require(not summary["missing_trial_keys"], "missing scored trial")
    for row in summary["trials"]:
        # The summary is an INDEX, never treat it as embedded physical metrics.
        score = read(row["score_path"])
        require(
            score["metrics"]["valid_for_scoring"], f"invalid trial: {row['directory']}"
        )
    return {
        "status": "complete_valid",
        "campaign_sha256": campaign_sha,
        "trial_count": 2,
        "summary_sha256": sha(output / "analysis/summary.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    b = read(args.bindings)
    require(b["status"] == "frozen", "draft bindings cannot launch")
    require(
        b["cpu_tests_passed"] is True and b["root_review_approved"] is True,
        "green tests/review required",
    )
    require(b["port"] == 7799, "use dedicated study port7799")
    require(
        len(b["cases"]) == 3
        and [x["id"] for x in b["cases"]]
        == ["plain_k4", "legacy_limiter", "bounded_hold"],
        "three conditions required",
    )
    require(
        {b["runtime"], b["driver"]}.issubset({x["root"] for x in b["inventories"]}),
        "both full runtime and driver inventories are required",
    )
    verified = verify_files(b)
    driver_root = Path(b["driver"])
    sys.path.insert(0, str(driver_root))
    driver = module(
        driver_root / "tools/sim/run_policy_campaign.py", "carry_hold_driver"
    )
    plans = []
    designs = []
    for row, hold, limiter in zip(
        b["cases"], [None, None, 2.5], [False, True, True], strict=True
    ):
        campaign = Path(row["campaign"])
        require(sha(campaign) == row["campaign_sha256"], "campaign digest mismatch")
        design, digest = driver.load_design(campaign)
        validate_design(design, hold, limiter)
        require(
            sum(len(x["trials"]) for x in driver.blocks(design)) == 2,
            "wrong runtime trial count",
        )
        output = Path(b["output_root"]) / row["id"]
        require(not output.exists(), f"refuse existing output: {output}")
        command = [
            b["server_python"],
            str(driver_root / "tools/sim/run_policy_campaign.py"),
            "--campaign",
            str(campaign),
            "--source",
            b["runtime"],
            "--live-repo",
            b["live_repo"],
            "--server-python",
            b["server_python"],
            "--evidence",
            b["evidence"],
            "--hardware-config",
            row["hardware"],
            "--robot-usd",
            b["robot_usd"],
            "--output",
            str(output),
            "--port",
            str(b["port"]),
        ]
        parsed = driver.parser().parse_args(command[2:])
        inputs = driver.source_manifest(parsed, digest, design)
        require(
            inputs["hardware_sha256"] == design["runtime_hardware"]["sha256"],
            "wrong effective hardware bytes",
        )
        plans.append(
            {
                "id": row["id"],
                "command": command,
                "inputs": inputs,
                "output": str(output),
                "campaign_sha256": digest,
            }
        )
        designs.append(design)
    for other in designs[1:]:
        matched(designs[0], other)
    result = {
        "status": "plan_only",
        "verified_files": verified,
        "binding_sha256": sha(args.bindings),
        "plans": plans,
    }
    if not args.execute:
        print(json.dumps(result, indent=2))
        return
    root = Path(b["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    exclusive(root / "launch_plan.json", result)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    for plan in plans:
        # Recheck immutable inputs before each block; failures remain preserved.
        verify_files(b)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", b["port"]))
        with (root / f"{plan['id']}.log").open("x") as log:
            process = subprocess.Popen(
                plan["command"] + ["--execute"],
                cwd=driver_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            exclusive(
                root / f"{plan['id']}_launch.json",
                {"pid": process.pid, "command": plan["command"] + ["--execute"]},
            )
            rc = process.wait()
        require(rc == 0, f"controller exit{rc}; no retry")
        checked = complete(Path(plan["output"]), plan["campaign_sha256"])
        exclusive(root / f"{plan['id']}_completed.json", checked)
    exclusive(
        root / "complete.json",
        {
            "status": "six_valid_trials_completed",
            "trials": 6,
            "interpretation": "two development seeds; no reliable-winner or hardware claim",
        },
    )


if __name__ == "__main__":
    main()
