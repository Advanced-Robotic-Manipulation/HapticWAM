#!/usr/bin/env python3
"""Execute a frozen teacher block against a dedicated, already owned server.

The source snapshot and server ready marker are immutable inputs. Failed
attempts are retained; an interrupted trial cannot be silently retried.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from tools.sim import run_policy_campaign as campaign

MIN_FREE_BYTES = 10 * 1024**3
REQUIRED_OUTPUTS = (
    "run.json", "sim_trace.npz", "effective_config.json", "policy_info.json",
    "execution_trace.jsonl", "planner_trace.json", "sim.mp4", "policy_tactile.npz",
    "robot_environment_contact_trace.json", "adaptive_policy_experiment.json",
    "native_mechanics_monitor.json",
)


def frozen_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError(f"Frozen input differs: {path}")
    if not path.exists():
        campaign.write_json(path, value)


def experiment_manifest(args, digest, design):
    value = campaign.source_manifest(args, digest, design)
    value.update(
        server_ready_sha256=campaign.fingerprint(args.server_dir / "ready.json"),
        controller_sha256=campaign.fingerprint(Path(__file__)),
        protocol_sha256=campaign.fingerprint(args.protocol),
    )
    return value


def validate_hardware(args, design):
    import yaml

    if campaign.fingerprint(args.hardware_config) != design["runtime_hardware"]["sha256"]:
        raise ValueError("Hardware file differs from the frozen experiment")
    hardware = yaml.safe_load(args.hardware_config.read_text())
    if hardware.get("safety", {}).get("lift_complete_z_m") != 0:
        raise ValueError("Experiment must explicitly disable lift-complete auto-stop")


def require_free_space(output):
    free = shutil.disk_usage(output).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Preserving existing data: only {free / 1024**3:.2f} GiB free; "
            "each new rollout requires at least 10 GiB free"
        )
    return free


def validate_native_monitor(directory, run):
    """Check the actual guarded-runtime schema; physical failures are permitted."""
    if (directory / "native_mechanics_failure.json").exists():
        raise RuntimeError("Native mechanical failure marker is present")
    monitor = json.loads((directory / "native_mechanics_monitor.json").read_text())
    if run.get("native_mechanics_monitor") != monitor:
        raise RuntimeError("Native monitor file disagrees with run metadata")
    if (monitor.get("enabled") is not True or type(monitor.get("checks")) is not int
            or monitor["checks"] < 1 or monitor.get("last_phase") != "execution"):
        raise RuntimeError("Native instantaneous mechanical monitoring is incomplete")
    last = monitor.get("last_diagnostic", {})
    thresholds = {
        "coupling_max_abs_rad": .005,
        "joint_limit_violation_max_rad": .002,
        "loop_closure_max_m": .001,
    }
    if (last.get("passed") is not True or last.get("joint_state_finite") is not True
            or last.get("thresholds") != thresholds
            or last.get("gates") != {
                "joint_coupling": True, "joint_limits": True, "adaptive_loop_closure": True,
            }):
        raise RuntimeError("Native mechanical gates or their frozen thresholds differ")
    for name, threshold in thresholds.items():
        value = monitor.get("observed_maxima", {}).get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= threshold:
            raise RuntimeError(f"Native mechanical maximum is invalid: {name}")
    final_t = monitor.get("last_t_s")
    duration = run.get("duration_s")
    if (type(final_t) not in (int, float) or not math.isfinite(final_t)
            or type(duration) not in (int, float) or not math.isfinite(duration)
            or abs(final_t - duration) > .001 + 1e-9):
        raise RuntimeError("Native monitor does not cover the reported rollout end")


def validate_scored_trial(output, name):
    record = json.loads((output / "analysis" / "trials" / f"{name}.json").read_text())
    if record.get("status") != "scored" or record.get("metrics", {}).get("valid_for_scoring") is not True:
        raise RuntimeError(
            f"Preserved technically invalid trial {name}: "
            f"{record.get('metrics', {}).get('invalid_reasons', record.get('status'))}"
        )
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--server-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, default=REPO)
    p.add_argument("--live-repo", type=Path, required=True,
                   help="Frozen inference repository, not hardware checkout")
    p.add_argument("--server-python", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--hardware-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=7799)
    p.add_argument("--rollout-timeout", type=float, default=1800)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    for name in ("campaign", "protocol", "server_dir", "source", "live_repo",
                 "evidence", "hardware_config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.server_python = args.server_python.absolute()
    if not 1024 <= args.port <= 65535 or 7777 <= args.port <= 7785:
        raise ValueError("Use a dedicated simulator port outside rig ports 7777–7785")
    if not math.isfinite(args.rollout_timeout) or args.rollout_timeout <= 0:
        raise ValueError("rollout-timeout must be finite and positive")
    args.robot_usd = None
    design, digest = campaign.load_design(args.campaign)
    protocol = json.loads(args.protocol.read_text())
    policy, = design["policies"]
    if policy["architecture"] != "teacher" or policy["policy_mode"] != "teacher":
        raise ValueError("This experiment is teacher-only")
    conditions = {c["id"]: c for c in design["conditions"]}
    schedule = protocol["development"]["execution_order"]
    actual = [(policy["id"], r["condition"], r["seed"]) for r in schedule]
    expected = campaign._analysis.planned_keys(design)
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("Execution order does not cover the frozen grid exactly")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_json(args.output / "plan.json", {
        "campaign_sha256": digest, "protocol_sha256": campaign.fingerprint(args.protocol),
        "schedule": schedule, "server_dir": str(args.server_dir),
    })
    if not args.execute:
        print(json.dumps({"planned": len(schedule), "schedule": schedule}, indent=2))
        return
    with (args.output / "controller.lock").open("w") as lock, (
        args.server_dir / "simulation_lease.lock"
    ).open("w") as lease:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready_path = args.server_dir / "ready.json"
        ready = json.loads(ready_path.read_text())
        owned = json.loads((args.server_dir / "owned_pid.json").read_text())
        if (ready["pid"] != owned["pid"] or ready["port"] != args.port
                or not ready["dedicated_simulation_server"]
                or ready["checkpoint_sha256"] != policy["checkpoint_sha256"]):
            raise ValueError("Dedicated server identity differs from pinned experiment")
        os.kill(ready["pid"], 0)
        validate_hardware(args, design)
        source = experiment_manifest(args, digest, design)
        frozen_json(args.output / "frozen_inputs.json", source)
        frozen_json(args.output / "campaign_snapshot.json", design)
        # Preserve exact campaign bytes for the analyzer's SHA identity.
        (args.output / "campaign_snapshot.json").write_bytes(args.campaign.read_bytes())
        profile = design["adapter_profile"]
        campaign.write_controller_configs(args.output, profile, writer=frozen_json)
        frozen_json(args.output / "runtime" / f"inference_{policy['id']}.json",
                    campaign.inference_config(design, policy))
        for condition in conditions.values():
            frozen_json(args.output / "conditions" / f"{condition['id']}.json",
                        campaign.condition_scene(design, condition))
        ledger = {"controller_pid": os.getpid(), "campaign_sha256": digest,
                  "status": "running", "trials": {}}
        ledger_path = args.output / "progress.json"
        child = None

        def stop(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        try:
            for row in schedule:
                condition, seed = conditions[row["condition"]], row["seed"]
                directory = args.output / "rollouts" / campaign.trial_id(
                    policy["id"], condition["id"], seed)
                if experiment_manifest(args, digest, design) != source:
                    raise RuntimeError("Source or frozen experiment inputs changed")
                if campaign.completed_case(directory, digest, policy, condition, seed):
                    if not all((directory / name).is_file() for name in REQUIRED_OUTPUTS):
                        raise RuntimeError(f"Completed rollout has missing evidence: {directory}")
                    validate_native_monitor(directory, json.loads((directory / "run.json").read_text()))
                    validate_scored_trial(args.output, directory.name)
                    ledger["trials"][directory.name] = {"status": "completed", "reused": True}
                    continue
                free_bytes = require_free_space(args.output)
                directory.mkdir(parents=True, exist_ok=True)
                campaign.write_json(directory / "server_ready.json", ready)
                case = {
                    "schema_version": 1, "campaign_sha256": digest,
                    "policy_id": policy["id"], "condition_id": condition["id"],
                    "sampling_seed": seed, "checkpoint_sha256": ready["checkpoint_sha256"],
                    "attempt": 1, "phase": "development", "intended_parameters": condition,
                    "server_metadata": str(args.server_dir / "server.json"),
                    "server_ready_sha256": campaign.fingerprint(directory / "server_ready.json"),
                    "scoring_thresholds": design["thresholds"],
                }
                campaign.write_json(directory / "case.json", case)
                command = campaign.simulation_command(
                    args, design, policy, condition, seed, directory, None)
                command += ["--experimental-adaptive-policy", "--record-gel-contacts",
                            "--record-robot-environment-contacts"]
                if profile.get("placement_controller_profile"):
                    command += ["--placement-controller-profile", profile["placement_controller_profile"]]
                if profile.get("grip_play_steps") is not None:
                    command += ["--grip-play-steps", str(profile["grip_play_steps"])]
                status = {"status": "running", "command": command, "started_unix_s": time.time(),
                          "prelaunch_free_bytes": free_bytes}
                campaign.write_json(directory / "run_status.json", status)
                ledger["trials"][directory.name] = status
                campaign.write_json(ledger_path, ledger)
                print(f"Starting {directory.name}", flush=True)
                with (directory / "isaac.log").open("w") as log:
                    child = subprocess.Popen(command, cwd=args.source, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = child.wait(timeout=args.rollout_timeout)
                    except BaseException:
                        campaign.stop_owned(child)
                        status.update(status="interrupted_or_timeout", exit_code=child.returncode,
                                      finished_unix_s=time.time())
                        campaign.write_json(directory / "run_status.json", status)
                        raise
                child = None
                complete = code == 0 and all((directory / name).is_file() for name in REQUIRED_OUTPUTS)
                status.update(status="physical_complete_pending_audit" if complete else "runtime_error", exit_code=code,
                              finished_unix_s=time.time())
                campaign.write_json(directory / "run_status.json", status)
                campaign.write_json(ledger_path, ledger)
                if not complete:
                    raise RuntimeError(f"Preserved incomplete attempt: {directory}")
                run = json.loads((directory / "run.json").read_text())
                if experiment_manifest(args, digest, design) != source:
                    raise RuntimeError("Source or frozen experiment inputs changed during rollout")
                validate_native_monitor(directory, run)
                case["effective_parameters"] = {
                    "scene_sha256": campaign.fingerprint(directory / "effective_config.json"),
                    "policy_info_sha256": campaign.fingerprint(directory / "policy_info.json"),
                    "duration_s": run["duration_s"], "stop_reason": run.get("policy_stop_reason"),
                    "robot_usd_sha256": campaign.fingerprint(run["robot_usd"]),
                }
                campaign.write_json(directory / "case.json", case)
                with (args.output / f"analysis__{directory.name}.log").open("w") as log:
                    subprocess.run(campaign.analysis_command(args, args.output / "campaign_snapshot.json"),
                                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
                validate_scored_trial(args.output, directory.name)
                status.update(status="completed", audited_unix_s=time.time())
                campaign.write_json(directory / "run_status.json", status)
                campaign.write_json(ledger_path, ledger)
                print(f"Completed {directory.name}: duration={run['duration_s']:.3f}s "
                      f"stop={run.get('policy_stop_reason')}", flush=True)
            ledger["status"] = "all_trials_completed"
        except BaseException as error:
            ledger.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            campaign.stop_owned(child)
            ledger["finished_unix_s"] = time.time()
            campaign.write_json(ledger_path, ledger)


if __name__ == "__main__":
    main()
