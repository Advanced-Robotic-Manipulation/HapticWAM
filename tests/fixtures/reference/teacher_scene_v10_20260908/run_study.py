#!/usr/bin/env python3
"""Run the frozen, counterbalanced teacher scene/controller study.

Default: validate and write a plan. --execute owns one inference-only server
and sequential Isaac children. Existing completed trials are reused only with
matching frozen provenance; incomplete attempts are never silently replaced.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from phantom.config.hardware import load_hardware
from tools.sim import run_policy_campaign as campaign
from tools.sim.analyze_policy_campaign import policy_delivery_clock, shared_server_hardware
from tools.sim.deployment_filters import TerminalVetoFilter

FT_A_SHA256 = "67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e"
ROLLOUT_TIMEOUT_S = 900
MIN_FREE_BYTES = 10 * 1024**3
REQUIRED = (
    "run.json", "sim_trace.npz", "effective_config.json", "policy_info.json",
    "execution_trace.jsonl", "planner_trace.json", "server_ready.json", "sim.mp4",
    "policy_tactile.npz", "robot_environment_contact_trace.json",
)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Unsafe identifier: {value!r}")
    return value


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Frozen inputs changed; amend the study explicitly: {path}")
    else:
        campaign.write_json(path, value)


def immutable_copy(source, target):
    if target.exists():
        if campaign.fingerprint(source) != campaign.fingerprint(target):
            raise RuntimeError(f"Frozen snapshot differs: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def resource_snapshot(output):
    disk = shutil.disk_usage(output)
    value = {"unix_s": time.time(), "disk_free_bytes": disk.free,
             "disk_total_bytes": disk.total, "controller_pid": os.getpid()}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        value["gpu_query"] = {"exit_code": result.returncode, "stdout": result.stdout,
                              "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        value["gpu_query_error"] = str(error)
    return value


def load_study(path):
    study = json.loads(path.read_text())
    if study.get("status") != "frozen":
        raise ValueError("Study must be frozen before planning or execution")
    identifier(study["study_id"])
    runtime = study["runtime"]
    args = argparse.Namespace(**{
        name: Path(runtime[name]).expanduser().absolute()
        for name in ("source", "live_repo", "server_python", "evidence", "robot_usd", "output")
    }, port=runtime["port"])
    if isinstance(args.port, bool) or not isinstance(args.port, int) or not (
        1024 <= args.port <= 65535 and not 7777 <= args.port <= 7783
    ):
        raise ValueError("Use a dedicated port outside reserved rig ports 7777–7783")
    if len(study["variants"]) not in (4, 5):
        raise ValueError("This bounded study supports four primary variants and one optional fifth")
    variants = {}
    expected = set()
    reference_hw = reference_shapes = reference_policy = reference_inference = None
    reference_grid = reference_shared = reference_timing = reference_thresholds = None
    reference_conditions = reference_episode = reference_baseline = None
    for variant in study["variants"]:
        vid = identifier(variant["id"])
        if vid in variants:
            raise ValueError("Duplicate study variant")
        local = copy.copy(args)
        local.campaign = Path(variant["campaign"]).absolute()
        local.hardware_config = Path(variant["hardware"]).absolute()
        local.output = args.output / vid
        design, sha = campaign.load_design(local.campaign)
        campaign.blocks(design)
        if len(design["policies"]) != 1:
            raise ValueError("Each variant must contain exactly one ftA teacher policy")
        policy = design["policies"][0]
        identifier(policy["id"])
        if policy["architecture"] != "teacher" or policy["checkpoint_sha256"] != FT_A_SHA256:
            raise ValueError("Study must use the pinned ftA teacher checkpoint")
        if policy.get("policy_mode") != "teacher":
            raise ValueError("Study must use teacher observation inputs")
        identity = {k: policy[k] for k in ("id", "checkpoint", "checkpoint_sha256", "architecture", "policy_mode")}
        settings = campaign.policy_settings(design, policy)
        if settings["max_play"] not in (10, 12):
            raise ValueError("Study allows only the frozen 10 versus 12 played-action contrast")
        inference = {k: v for k, v in settings.items() if k != "max_play"}
        hw = load_hardware(local.hardware_config, quiet=True)
        hardware = design["runtime_hardware"]
        if (campaign.fingerprint(local.hardware_config) != hardware["sha256"]
                or hw.config_hash() != hardware["config_hash"]
                or hw.model_dump(mode="json") != hardware["effective_model"]):
            raise ValueError(f"Hardware file differs from frozen effective hardware: {vid}")
        if hw.safety.lift_complete_z_m != 0:
            raise ValueError("Full-task study must explicitly disable lift-complete auto-stop")
        shared = shared_server_hardware(design)
        if "shared_server_hardware" not in design:
            raise ValueError("Shared inference server hardware must be explicitly declared")
        common_hw = hw.model_dump(mode="json")
        common_hw["safety"].pop("servo_constraint_hold_s")
        shapes = hw.shape_relevant_fields()
        conditions = {identifier(c["id"]): c for c in design["conditions"]}
        grid = {(cid, int(seed)) for cid in conditions for seed in design["sampling_seeds"]}
        if len(grid) != 12 or len(conditions) != 6 or len(design["sampling_seeds"]) != 2:
            raise ValueError("Study requires six measured starts and two matched sampling seeds")
        timing = (policy_delivery_clock(design), design.get("delivery_latency_s"))
        if timing != ("rpc_wall", None):
            raise ValueError("Study uses measured RPC wall delivery timing without latency override")
        if design.get("stage_gate"):
            raise ValueError("Run a separately verified gate before freezing this study; campaign gates cannot be bypassed")
        profile = design.get("adapter_profile", {})
        actual_feedback_source = TerminalVetoFilter.feedback_source(SimpleNamespace(
            release_controller=object() if profile.get("placement_release") else None,
        ))
        if profile.get("terminal_veto_feedback_source") != actual_feedback_source:
            raise ValueError(
                f"Frozen terminal_veto_feedback_source for {vid} disagrees with the runtime: "
                f"declared={profile.get('terminal_veto_feedback_source')!r}, "
                f"actual={actual_feedback_source!r}"
            )
        if not profile.get("save_policy_observations") or not profile.get("record_packet_support"):
            raise ValueError("Study requires policy observations and packet support instrumentation")
        if reference_hw is None:
            reference_hw, reference_shapes = common_hw, shapes
            reference_policy, reference_inference = identity, inference
            reference_grid, reference_shared = grid, shared
            reference_timing, reference_thresholds = timing, design["thresholds"]
            reference_conditions = conditions
            reference_episode = design["prepared_episode"]
            reference_baseline = profile.get("tactile_baseline")
            if hw.safety.servo_constraint_hold_s is not None:
                raise ValueError("First variant must use legacy None hold for the shared inference server")
            if any(shared[k] != hardware[k] for k in ("sha256", "config_hash", "effective_model")):
                raise ValueError("Shared-server declaration must match first variant hardware")
        elif (common_hw != reference_hw or shapes != reference_shapes
              or identity != reference_policy or inference != reference_inference
              or grid != reference_grid or shared != reference_shared
              or timing != reference_timing or design["thresholds"] != reference_thresholds
              or conditions != reference_conditions
              or design["prepared_episode"] != reference_episode
              or profile.get("tactile_baseline") != reference_baseline):
            raise ValueError(f"Unmatched policy, observation, hardware, grid, timing or scoring inputs: {vid}")
        variants[vid] = {"args": local, "design": design, "sha256": sha,
                         "policy": policy, "conditions": conditions}
        expected.update((vid, cid, seed) for cid, seed in grid)
    actual = []
    for row in study["schedule"]:
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise ValueError("Schedule sampling seeds must be integers")
        actual.append((identifier(row["variant"]), identifier(row["condition"]), row["seed"]))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Schedule must cover each variant/condition/seed exactly once")
    # Require complete paired blocks so slow temporal drift cannot track variants.
    width = len(variants)
    for start in range(0, len(actual), width):
        block = actual[start:start + width]
        if len({(cid, seed) for _, cid, seed in block}) != 1 or {v for v, _, _ in block} != set(variants):
            raise ValueError("Every sequential block must contain all variants at one matched start/seed")
    return study, args, variants


def prepare(path, study, args, variants):
    study_sha = campaign.fingerprint(path)
    immutable_copy(path, args.output / "study_snapshot.json")
    frozen = {"study_sha256": study_sha, "runner_sha256": campaign.fingerprint(__file__),
              "variants": {}}
    for vid, data in variants.items():
        local, design, policy = data["args"], data["design"], data["policy"]
        sources = campaign.source_manifest(local, data["sha256"], design)
        sources["study_sha256"] = study_sha
        sources["shared_server_hardware"] = design["shared_server_hardware"]
        immutable_json(local.output / "frozen_inputs.json", sources)
        snapshot = local.output / "campaign_snapshot.json"
        immutable_copy(local.campaign, snapshot)
        immutable_json(local.output / "runtime" / f"inference_{policy['id']}.json",
                       campaign.inference_config(design, policy))
        for field in ("terminal_veto", "placement_release"):
            value = design.get("adapter_profile", {}).get(field)
            if value:
                immutable_json(local.output / "runtime" / f"{field}.json", value)
        for cid, condition in data["conditions"].items():
            immutable_json(local.output / "conditions" / f"{cid}.json",
                           campaign.condition_scene(design, condition))
        frozen["variants"][vid] = sources
    immutable_json(args.output / "frozen_inputs.json", frozen)
    plan = {"study_sha256": study_sha, "planned_trials": len(study["schedule"]),
            "schedule": study["schedule"], "variants": [
                {"id": vid, "campaign_sha256": data["sha256"],
                 "hardware_sha256": data["design"]["runtime_hardware"]["sha256"],
                 "inference": campaign.policy_settings(data["design"], data["policy"])}
                for vid, data in variants.items()],
            "server_hardware_variant": next(iter(variants)),
            "server_transport": f"127.0.0.1:{args.port}",
            "rollout_timeout_s": ROLLOUT_TIMEOUT_S,
            "min_free_disk_bytes": MIN_FREE_BYTES,
            "common_simulation_flags": ["--record-robot-environment-contacts"]}
    immutable_json(args.output / "plan.json", plan)
    return plan


def run_study(study, args, variants, plan):
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, interrupted)
    server = child = None
    active_status = active_directory = None
    server_log = None
    failure = None
    ledger = {"study_sha256": plan["study_sha256"], "controller_pid": os.getpid(),
              "status": "running", "started_unix_s": time.time(), "trials": {},
              "analysis": {}}
    ledger_path = args.output / "progress.json"
    campaign.write_json(ledger_path, ledger)
    try:
        pending = []
        for index, row in enumerate(study["schedule"]):
            data = variants[row["variant"]]
            condition = data["conditions"][row["condition"]]
            policy = data["policy"]
            directory = data["args"].output / "rollouts" / campaign.trial_id(
                policy["id"], row["condition"], row["seed"])
            key = f"{row['variant']}/{directory.name}"
            if campaign.completed_case(directory, data["sha256"], policy, condition, row["seed"]):
                missing = [name for name in REQUIRED if not (directory / name).is_file()]
                case = json.loads((directory / "case.json").read_text())
                audit_path = directory / "runtime_audit.json"
                if not audit_path.is_file() or json.loads(audit_path.read_text()).get("valid") is not True:
                    missing.append("passed_runtime_audit")
                if missing or case.get("study_sha256") != plan["study_sha256"] or case.get("schedule_index") != index:
                    raise RuntimeError(f"Completed case lacks matching study provenance/artifacts: {directory}")
                ledger["trials"][key] = {"status": "completed", "reused": True}
            else:
                pending.append((index, row, data, condition, directory, key))
        campaign.write_json(ledger_path, ledger)
        campaign.write_json(args.output / "startup_resources.json", resource_snapshot(args.output))
        if pending:
            with socket.socket() as port_check:
                port_check.bind(("127.0.0.1", args.port))
            first = next(iter(variants.values()))
            server_dir = args.output / "servers" / f"shared_teacher_{time.time_ns()}"
            server_dir.mkdir(parents=True)
            command = campaign.server_command(first["args"], first["design"], first["policy"], server_dir)
            campaign.write_json(server_dir / "command.json", command)
            server_log = (server_dir / "server.log").open("w")
            server = subprocess.Popen(command, cwd=args.live_repo, stdout=server_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            ready = campaign.wait_for_server(server, server_dir, FT_A_SHA256, 600)
            declared = first["design"]["shared_server_hardware"]
            if (ready.get("hardware_sha256") != declared["sha256"]
                    or ready.get("hardware_config_hash") != declared["config_hash"]):
                raise RuntimeError("Owned shared server loaded different hardware than its frozen declaration")
            ledger["server"] = {"directory": str(server_dir), "pid": server.pid}
        for index, row, data, condition, directory, key in pending:
            if server.poll() is not None:
                raise RuntimeError(f"Owned shared server exited {server.returncode}")
            resources = resource_snapshot(args.output)
            if resources["disk_free_bytes"] < MIN_FREE_BYTES:
                raise RuntimeError("Insufficient free output disk; require at least 10 GiB before each trial")
            directory.mkdir(parents=True, exist_ok=True)
            campaign.write_json(directory / "resources_before.json", resources)
            campaign.write_json(directory / "server_ready.json", ready)
            local, design, policy = data["args"], data["design"], data["policy"]
            case = {
                "schema_version": 1, "study_sha256": plan["study_sha256"],
                "variant": row["variant"], "schedule_index": index,
                "campaign_sha256": data["sha256"], "policy_id": policy["id"],
                "condition_id": condition["id"], "sampling_seed": row["seed"],
                "checkpoint_sha256": ready["checkpoint_sha256"], "attempt": 1,
                "phase": f"matched_block_{index // len(variants):02d}",
                "server_metadata": str(server_dir / "server.json"),
                "server_ready_sha256": campaign.fingerprint(directory / "server_ready.json"),
                "intended_parameters": condition,
                "intended_scene_sha256": campaign.fingerprint(local.output / "conditions" / f"{condition['id']}.json"),
                "effective_parameters": None, "scoring_thresholds": design["thresholds"],
                "common_simulation_flags": plan["common_simulation_flags"],
            }
            campaign.write_json(directory / "case.json", case)
            command = campaign.simulation_command(local, design, policy, condition, row["seed"], directory, args.robot_usd)
            command += plan["common_simulation_flags"]
            status = {"status": "running", "command": command, "started_unix_s": time.time()}
            active_directory, active_status = directory, status
            campaign.write_json(directory / "command.json", command)
            campaign.write_json(directory / "run_status.json", status)
            ledger["active_trial"] = key
            ledger["trials"][key] = status
            campaign.write_json(ledger_path, ledger)
            print(f"Starting {index + 1}/{len(study['schedule'])}: {key}", flush=True)
            with (directory / "isaac.log").open("w") as log:
                child = subprocess.Popen(command, cwd=args.source, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    exit_code = child.wait(timeout=ROLLOUT_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    campaign.stop_owned(child)
                    status.update(status="timeout", finished_unix_s=time.time(), exit_code=child.returncode)
                    campaign.write_json(directory / "run_status.json", status)
                    raise
            child = None
            missing = [name for name in REQUIRED if not (directory / name).is_file()]
            if exit_code or missing:
                status.update(status="runtime_error", exit_code=exit_code, missing_artifacts=missing,
                              finished_unix_s=time.time())
                campaign.write_json(directory / "run_status.json", status)
                raise RuntimeError(f"Trial failed before complete evidence: {directory}")
            run = json.loads((directory / "run.json").read_text())
            info = json.loads((directory / "policy_info.json").read_text())
            case["effective_parameters"] = {
                "scene_sha256": campaign.fingerprint(directory / "effective_config.json"),
                "policy_info_sha256": campaign.fingerprint(directory / "policy_info.json"),
                "checkpoint_sha256": info.get("ckpt_sha"),
                "observation_delay_s": info.get("observation_delay_s"),
                "inference_delay_add_s": info.get("inference_delay_add_s"),
                "max_play_steps": info.get("max_play_steps"),
                "policy_latency_override_s": info.get("policy_latency_override_s"),
                "policy_delivery_clock": info.get("policy_delivery_clock", "native"),
                "policy_initial_state_provenance": info.get("policy_initial_state_provenance"),
                "duration_s": run["duration_s"], "stop_reason": run.get("policy_stop_reason"),
                "completed_reason": run.get("policy_completed_reason"),
            }
            campaign.write_json(directory / "case.json", case)
            with np.load(directory / "sim_trace.npz", allow_pickle=False) as trace:
                times = np.asarray(trace["t"]).copy() if "t" in trace else []
            audit_reasons = campaign._analysis.runtime_audit(
                design, policy, condition, info, ready, run, times, run.get("policy_stop_reason"))
            audit_reasons += campaign._analysis.policy_delivery_timing_audit(
                design, condition, campaign._analysis.read_rows(directory / "planner_trace.json"))
            scene_differences = campaign._analysis.scene_mismatches(
                campaign.condition_scene(design, condition),
                json.loads((directory / "effective_config.json").read_text()))
            if scene_differences:
                audit_reasons.append("effective_scene_differs_from_frozen_condition")
            campaign.write_json(directory / "runtime_audit.json", {
                "valid": not audit_reasons, "reasons": sorted(set(audit_reasons)),
                "scene_differences": scene_differences,
            })
            if audit_reasons:
                status.update(status="instrumentation_error", audit_reasons=sorted(set(audit_reasons)),
                              exit_code=exit_code, finished_unix_s=time.time())
                campaign.write_json(directory / "run_status.json", status)
                raise RuntimeError(f"Runtime contract audit failed; preserve trial and stop: {directory}")
            status.update(status="completed", exit_code=exit_code, finished_unix_s=time.time())
            campaign.write_json(directory / "run_status.json", status)
            ledger.pop("active_trial", None)
            campaign.write_json(ledger_path, ledger)
            active_directory = active_status = None
            print(f"Completed {key}; duration={run['duration_s']:.3f}s stop={run.get('policy_stop_reason')}", flush=True)
        ledger["status"] = "all_trials_completed"
    except BaseException as error:
        failure = error
        state = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        ledger.update(status=state, error=f"{type(error).__name__}: {error}")
        if active_status is not None and active_status["status"] == "running":
            active_status.update(status=state, error=ledger["error"], finished_unix_s=time.time())
            campaign.write_json(active_directory / "run_status.json", active_status)
    finally:
        try:
            campaign.stop_owned(child)
        finally:
            try:
                campaign.stop_owned(server)
            finally:
                if server_log is not None:
                    server_log.close()
                ledger["finished_unix_s"] = time.time()
                campaign.write_json(ledger_path, ledger)
                signal.signal(signal.SIGTERM, old_handler)
    # Analyze partial evidence as well, while preserving any original failure.
    for vid, data in variants.items():
        local = data["args"]
        command = campaign.analysis_command(local, local.output / "campaign_snapshot.json")
        campaign.write_json(local.output / "analysis_command.json", command)
        try:
            with (local.output / "analysis.log").open("w") as log:
                result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=300, check=True)
            ledger["analysis"][vid] = {"status": "completed", "exit_code": result.returncode}
        except (OSError, subprocess.SubprocessError) as error:
            ledger["analysis"][vid] = {"status": "error", "error": str(error)}
            if failure is None:
                failure = error
                ledger.update(status="analysis_error", error=str(error))
        campaign.write_json(ledger_path, ledger)
    if failure is not None:
        raise failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    cli = parser.parse_args()
    path = cli.study.resolve()
    study, args, variants = load_study(path)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(path, study, args, variants)
        print(json.dumps(plan, indent=2), flush=True)
        if cli.execute:
            run_study(study, args, variants, plan)


if __name__ == "__main__":
    main()
