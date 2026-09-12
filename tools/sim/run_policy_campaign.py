#!/usr/bin/env python3
"""Plan or execute the frozen simulator campaign using dedicated owned servers.

Default mode writes a reviewable plan only. --execute starts one inference
server at a time and sequential Isaac subprocesses. No hardware launcher is
used, no external process is terminated, and completed failures are preserved.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_analysis = importlib.import_module("tools.sim.analyze_policy_campaign")
condition_scene = _analysis.condition_scene
fingerprint = _analysis.fingerprint
load_design = _analysis.load_design
policy_settings = _analysis.policy_settings
adapter_input = _analysis.adapter_input
servo_reach_limiter_metadata = _analysis.servo_reach_limiter_metadata


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_controller_configs(output, profile, *, writer=write_json):
    """Materialize the exact optional JSONs referenced by simulation_command."""
    for field in ("terminal_veto", "placement_release", "boundary_projection"):
        if profile.get(field):
            writer(output / "runtime" / f"{field}.json", profile[field])


def blocks(design):
    result = []
    phase_ids = set()
    for phase in design["execution_phases"]:
        if phase["id"] in phase_ids:
            raise ValueError("Duplicate execution phase ID")
        phase_ids.add(phase["id"])
        seeds = phase.get("sampling_seeds", design["sampling_seeds"])
        if not seeds or not phase["policy_ids"] or not phase["condition_ids"]:
            raise ValueError("Execution phases must not contain empty blocks")
        for policy in phase["policy_ids"]:
            trials = [
                (condition, int(seed))
                for condition in phase["condition_ids"]
                for seed in seeds
            ]
            result.append(
                {
                    "id": f"{phase['id']}__{policy}",
                    "policy_id": policy,
                    "trials": trials,
                }
            )
    keys = [
        (b["policy_id"], condition, seed)
        for b in result
        for condition, seed in b["trials"]
    ]
    expected = set(_analysis.planned_keys(design))
    if (
        len(keys) != len(set(keys))
        or set(keys) != expected
        or len(keys) != design["planned_counts"]["total"]
    ):
        raise ValueError("Execution phases do not cover the frozen grid exactly once")
    return result


def trial_id(policy, condition, seed):
    return f"{policy}__{condition}__seed{seed}"


def inference_config(design, policy=None):
    settings = policy_settings(design, policy)
    config = {
        "nfe": settings["nfe"],
        "guidance": settings["guidance"],
        "k_seeds": settings["k_seeds"],
        "parity_fixes": settings["parity"],
        "persistent_noise": settings["persistent_noise"],
        "task_text": settings["task_text"],
        "drop_video": False,
        "close_p": 0.5,
    }
    if "action_time_origin" in settings:
        config["action_time_origin"] = settings["action_time_origin"]
    return config


def server_command(args, design, policy, directory):
    settings = policy_settings(design, policy)
    command = [
        str(args.server_python),
        str(args.source / "tools/sim/policy_server.py"),
        "--repo",
        str(args.live_repo),
        "--ckpt",
        policy["checkpoint"],
        "--expected-sha256",
        policy["checkpoint_sha256"],
        "--system",
        policy["architecture"],
        "--port",
        str(args.port),
        "--out",
        str(directory),
        "--hardware",
        str(args.hardware_config),
        "--nfe",
        str(settings["nfe"]),
        "--guidance",
        str(settings["guidance"]),
        "--k-seeds",
        str(settings["k_seeds"]),
        "--task-text",
        settings["task_text"],
        "--parity-fixes" if settings["parity"] else "--no-parity-fixes",
        "--persistent-noise"
        if settings["persistent_noise"]
        else "--no-persistent-noise",
    ]
    if "action_time_origin" in settings:
        command.extend(["--action-time-origin", settings["action_time_origin"]])
    if not settings["use_ema"]:
        command.append("--raw")
    return command


def simulation_command(args, design, policy, condition, seed, directory, robot_usd):
    profile = design.get("adapter_profile", {})
    delivery_clock = _analysis.policy_delivery_clock(design)
    settings = policy_settings(design, policy)
    relative_episode = Path(design["prepared_episode"]).relative_to("evidence")
    command = [
        str(args.source / "tools/sim/launch_waffles.sh"),
        "--mode",
        "policy",
        "--episode",
        str(args.evidence / relative_episode),
        "--output",
        str(directory),
        "--config",
        str(args.output / "conditions" / f"{condition['id']}.json"),
        "--duration",
        str(design["horizon_s"]),
        "--seed",
        str(seed),
        "--policy-server",
        f"127.0.0.1:{args.port}",
        "--policy-mode",
        policy["policy_mode"],
        "--policy-config",
        str(args.output / "runtime" / f"inference_{policy['id']}.json"),
        "--hardware-config",
        str(args.hardware_config),
        "--ignore-episode-overrides",
        "--max-play-steps",
        str(settings["max_play"]),
        "--observation-delay-s",
        str(condition["observation_delay_s"]),
        "--inference-delay-add-s",
        str(condition["inference_delay_add_s"]),
        "--tactile",
        profile.get("tactile_model", "contact_proxy"),
        "--wrist",
        profile.get("wrist_model", "contact_proxy"),
        "--skip-stage-export",
    ]
    if robot_usd:
        command += ["--robot-usd", str(robot_usd)]
    for field, flag in (
        ("initial_state", "--policy-initial-state"),
        ("tactile_baseline", "--tactile-baseline"),
    ):
        spec = adapter_input(design, condition, field)
        if spec:
            command += [flag, spec["path"]]
    if design.get("delivery_latency_s") is not None:
        command += ["--policy-latency", str(design["delivery_latency_s"])]
    if delivery_clock != "native":
        command += ["--policy-delivery-clock", delivery_clock]
    for field, flag in (
        ("terminal_veto", "--terminal-veto-config"),
        ("placement_release", "--placement-release-config"),
        ("boundary_projection", "--boundary-projection-config"),
    ):
        if profile.get(field):
            command += [flag, str(args.output / "runtime" / f"{field}.json")]
    if "gel_contact_coverage" in profile:
        command += ["--gel-contact-coverage", profile["gel_contact_coverage"]]
    if profile.get("save_policy_observations"):
        command += ["--save-policy-observations"]
    if profile.get("record_packet_support"):
        command += ["--record-packet-support"]
    if servo_reach_limiter_metadata(design) is not None:
        command += ["--servo-reach-limiter"]
        if profile.get("servo_constraint_hold_s") is not None:
            command += ["--servo-constraint-hold-s", str(profile["servo_constraint_hold_s"])]
    return command


def stop_owned(process, timeout=20):
    """Only accept Popen objects created by this controller, never ledger PIDs."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=timeout)
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def wait_for_server(process, directory, expected_sha, timeout):
    deadline = time.monotonic() + timeout
    ready = directory / "ready.json"
    next_update = time.monotonic()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Owned policy server exited {process.returncode}; see {directory / 'server.log'}"
            )
        if ready.exists():
            value = json.loads(ready.read_text())
            if (
                value.get("pid") != process.pid
                or value.get("checkpoint_sha256") != expected_sha
            ):
                raise RuntimeError(
                    "Ready marker does not identify the owned expected-checkpoint process"
                )
            return value
        if time.monotonic() >= next_update:
            print(f"Waiting for owned policy server pid={process.pid}", flush=True)
            next_update = time.monotonic() + 30
        time.sleep(0.25)
    raise TimeoutError(f"Policy server did not become ready within {timeout}s")


def source_manifest(args, design_sha, design):
    paths = set()
    for pattern in (
        "phantom/sim/*.py",
        "phantom/deploy/*.py",
        "phantom/scripts/run_deploy.py",
        "tools/sim/*.py",
        "tools/sim/*.sh",
        "configs/sim/*.json",
    ):
        paths.update(args.source.glob(pattern))
    paths.update(
        p
        for p in (args.source / "assets/sim").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    core_names = [
        "phantom/deploy/safety.py",
        "phantom/deploy/governor.py",
        "phantom/deploy/executor.py",
        "phantom/deploy/planner.py",
        "phantom/data/derived.py",
        "phantom/config/hardware.py",
        "configs/hardware.nuc.yaml",
    ]
    paths.update(args.source / name for name in core_names)
    if servo_reach_limiter_metadata(design) is not None:
        limiter_path = args.source / "phantom/drivers/servo_limiter.py"
        if not limiter_path.is_file():
            raise FileNotFoundError(
                "Enabled servo limiter source is missing: " + str(limiter_path)
            )
        paths.add(limiter_path)
        hold_path = args.source / "phantom/drivers/servo_hold.py"
        if design.get("adapter_profile", {}).get("servo_constraint_hold_s") is not None:
            if not hold_path.is_file():
                raise FileNotFoundError("Enabled constraint-hold source is missing: " + str(hold_path))
            paths.add(hold_path)
    episode = args.evidence / Path(design["prepared_episode"]).relative_to("evidence")
    profile_inputs = {}
    for field in ("initial_state", "tactile_baseline"):
        spec = design.get("adapter_profile", {}).get(field)
        if spec:
            actual = fingerprint(spec["path"])
            if actual != spec["sha256"]:
                raise ValueError(f"Frozen adapter {field} input hash differs")
            profile_inputs[field] = {"path": spec["path"], "sha256": actual}
    condition_inputs = {}
    for condition in design.get("conditions", []):
        spec = condition.get("initial_state")
        if spec:
            actual = fingerprint(spec["path"])
            if actual != spec["sha256"]:
                raise ValueError(
                    f"Frozen condition {condition['id']} initial-state hash differs"
                )
            condition_inputs[condition["id"]] = {"path": spec["path"], "sha256": actual}
    return {
        "campaign_sha256": design_sha,
        "hardware_config": str(args.hardware_config),
        "hardware_sha256": fingerprint(args.hardware_config),
        "source_root": str(args.source),
        "external_controller_source_root": str(REPO),
        "external_controller_source_sha256": {
            str(path.relative_to(REPO)): fingerprint(path)
            for path in sorted(
                set((REPO / "tools/sim").glob("*.py"))
                | set((REPO / "phantom/sim").glob("*.py"))
                | ({REPO / "phantom/deploy/boundary_projection.py"}
                   if design.get("adapter_profile", {}).get("boundary_projection") else set())
            )
        },
        "live_repository": str(args.live_repo),
        "source_sha256": {
            str(path.relative_to(args.source)): fingerprint(path)
            for path in sorted(paths)
        },
        "live_core_sha256": {
            name: fingerprint(args.live_repo / name) for name in core_names
        },
        "prepared_episode": str(episode),
        "episode_sha256": {
            name: fingerprint(episode / name)
            for name in ("replay.npz", "manifest.json")
        },
        "robot_usd": str(args.robot_usd) if args.robot_usd else None,
        "robot_usd_sha256": fingerprint(args.robot_usd) if args.robot_usd else None,
        "adapter_profile_inputs": profile_inputs,
        "condition_initial_state_inputs": condition_inputs,
    }


def analysis_command(args, snapshot):
    """Audit per-policy overrides with the controller's own versioned analyzer."""
    return [
        str(args.server_python),
        str(REPO / "tools/sim/analyze_policy_campaign.py"),
        "--campaign",
        str(snapshot),
        "--runs",
        str(args.output / "rollouts"),
        "--out",
        str(args.output / "analysis"),
    ]


def completed_case(directory, design_sha, policy, condition, seed):
    if not directory.exists():
        return False
    manifest_path = directory / "case.json"
    if not manifest_path.exists():
        if any(directory.iterdir()):
            raise RuntimeError(f"Nonempty unrecognized trial directory: {directory}")
        return False
    case = json.loads(manifest_path.read_text())
    expected = (
        design_sha,
        policy["id"],
        condition["id"],
        seed,
        policy["checkpoint_sha256"],
    )
    actual = tuple(
        case.get(k)
        for k in (
            "campaign_sha256",
            "policy_id",
            "condition_id",
            "sampling_seed",
            "checkpoint_sha256",
        )
    )
    if actual != expected:
        raise RuntimeError(
            f"Existing trial belongs to different frozen inputs: {directory}"
        )
    status_path = directory / "run_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get("status") == "completed" and all(
        (directory / name).exists()
        for name in (
            "run.json",
            "sim_trace.npz",
            "effective_config.json",
            "policy_info.json",
            "execution_trace.jsonl",
            "planner_trace.json",
            "server_ready.json",
        )
    ):
        return True
    raise RuntimeError(
        f"Preserving incomplete/failed attempt {directory}; no automatic outcome replacement. Inspect run_status and explicit infrastructure provenance before resuming."
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--campaign", type=Path, default=REPO / "configs/sim/policy_campaign.json"
    )
    p.add_argument("--source", type=Path, default=REPO)
    p.add_argument(
        "--live-repo",
        type=Path,
        default=Path("/home/physicalai/phantom-icra-2027/phantom"),
    )
    p.add_argument("--server-python", type=Path)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--hardware-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--robot-usd", type=Path)
    p.add_argument(
        "--stage-gate-audit",
        type=Path,
        help="Passed, hash-linked mechanics/reproduction audit when required by the design",
    )
    p.add_argument("--port", type=int, default=7792)
    p.add_argument("--server-timeout", type=float, default=600)
    p.add_argument("--rollout-timeout", type=float, default=900)
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually start owned server/Isaac subprocesses; default writes plan only",
    )
    return p


def main():
    args = parser().parse_args()
    for name in (
        "campaign",
        "source",
        "live_repo",
        "evidence",
        "hardware_config",
        "output",
        "robot_usd",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.server_python = (
        args.server_python or args.live_repo / ".venv/bin/python"
    ).absolute()
    if not 1024 <= args.port <= 65535 or 7777 <= args.port <= 7783:
        raise ValueError("Dedicated port must be outside reserved rig ports7777–7783")
    design, design_sha = load_design(args.campaign)
    policy_map = {p["id"]: p for p in design["policies"]}
    condition_map = {c["id"]: c for c in design["conditions"]}
    schedule = blocks(design)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {
            "campaign_sha256": design_sha,
            "executing": args.execute,
            "checkpoint_blocks": schedule,
            "planned_trials": design["planned_counts"]["total"],
            "policy_server_transport": f"127.0.0.1:{args.port}",
            "server_python": str(args.server_python),
            "effective_inference_by_policy": {
                policy["id"]: policy_settings(design, policy)
                for policy in design["policies"]
            },
            "initial_state_by_condition": {
                condition["id"]: adapter_input(design, condition, "initial_state")
                for condition in design["conditions"]
            },
            "delivery_latency_override_s": design.get("delivery_latency_s"),
            "policy_delivery_clock": _analysis.policy_delivery_clock(design),
        }
        write_json(args.output / "plan.json", plan)
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return
        gate = None
        if design.get("stage_gate"):
            from tools.sim.teacher_anchor_design import verify_execution_gate

            gate = verify_execution_gate(design, args.stage_gate_audit, args.source)
            episode = args.evidence / Path(design["prepared_episode"]).relative_to(
                "evidence"
            )
            contract = design["runtime_contract"]
            original_episode = (
                Path(contract["source"]).parent / design["prepared_episode"]
            )
            if episode.resolve() != original_episode.resolve():
                raise ValueError(
                    "Anchor episode path differs from the frozen prepared input"
                )
            if (
                args.robot_usd is None
                or fingerprint(args.robot_usd) != contract["robot_usd"]["sha256"]
            ):
                raise ValueError("Anchor requires the original hash-pinned robot asset")
        import yaml

        hardware = yaml.safe_load(args.hardware_config.read_text())
        if fingerprint(args.hardware_config) != design["runtime_hardware"]["sha256"]:
            raise ValueError(
                "Hardware file differs from the frozen GO_ANY configuration"
            )
        if hardware.get("safety", {}).get("lift_complete_z_m") != 0:
            raise ValueError(
                "Campaign hardware config must explicitly disable lift-complete auto-stop"
            )
        current_sources = source_manifest(args, design_sha, design)
        if gate is not None:
            current_sources["stage_gate_audit"] = gate
        sources_path = args.output / "frozen_inputs.json"
        if (
            sources_path.exists()
            and json.loads(sources_path.read_text()) != current_sources
        ):
            raise RuntimeError(
                "Frozen source/hardware/scene inputs changed; create an explicitly amended campaign"
            )
        write_json(sources_path, current_sources)
        snapshot = args.output / "campaign_snapshot.json"
        if snapshot.exists() and fingerprint(snapshot) != design_sha:
            raise RuntimeError("Campaign snapshot differs from current frozen design")
        snapshot.write_bytes(args.campaign.read_bytes())
        write_json(args.output / "runtime/inference.json", inference_config(design))
        for policy in design["policies"]:
            write_json(
                args.output / "runtime" / f"inference_{policy['id']}.json",
                inference_config(design, policy),
            )
        write_controller_configs(args.output, design.get("adapter_profile", {}))
        for condition in design["conditions"]:
            write_json(
                args.output / "conditions" / f"{condition['id']}.json",
                condition_scene(design, condition),
            )

        def interrupt(_signal, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        server = child = None
        robot_usd = args.robot_usd
        ledger_path = args.output / "progress.json"
        ledger = {
            "campaign_sha256": design_sha,
            "controller_pid": os.getpid(),
            "status": "running",
            "trials": {},
        }
        write_json(ledger_path, ledger)
        try:
            for block_index, block in enumerate(schedule):
                policy = policy_map[block["policy_id"]]
                pending = []
                for cid, seed in block["trials"]:
                    condition = condition_map[cid]
                    directory = (
                        args.output / "rollouts" / trial_id(policy["id"], cid, seed)
                    )
                    if completed_case(directory, design_sha, policy, condition, seed):
                        ledger["trials"][directory.name] = {
                            "status": "completed",
                            "reused": True,
                        }
                        if robot_usd is None:
                            robot_usd = json.loads(
                                (directory / "run.json").read_text()
                            ).get("robot_usd")
                    else:
                        pending.append((condition, seed, directory))
                if not pending:
                    continue
                # Refuse an occupied endpoint rather than replacing its owner.
                with socket.socket() as port_check:
                    port_check.bind(("127.0.0.1", args.port))
                server_dir = (
                    args.output
                    / "servers"
                    / f"{block_index:02d}_{block['id']}_{time.time_ns()}"
                )
                server_dir.mkdir(parents=True)
                command = server_command(args, design, policy, server_dir)
                write_json(server_dir / "command.json", command)
                with (server_dir / "server.log").open("w") as server_log:
                    server = subprocess.Popen(
                        command,
                        cwd=args.live_repo,
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    ready = wait_for_server(
                        server,
                        server_dir,
                        policy["checkpoint_sha256"],
                        args.server_timeout,
                    )
                    for condition, seed, directory in pending:
                        directory.mkdir(parents=True, exist_ok=True)
                        write_json(directory / "server_ready.json", ready)
                        case = {
                            "schema_version": 1,
                            "campaign_sha256": design_sha,
                            "policy_id": policy["id"],
                            "condition_id": condition["id"],
                            "sampling_seed": seed,
                            "checkpoint_sha256": ready["checkpoint_sha256"],
                            "attempt": 1,
                            "phase": block["id"],
                            "server_metadata": str(server_dir / "server.json"),
                            "server_ready_sha256": fingerprint(
                                directory / "server_ready.json"
                            ),
                            "intended_parameters": condition,
                            "intended_scene_sha256": fingerprint(
                                args.output / "conditions" / f"{condition['id']}.json"
                            ),
                            "effective_parameters": None,
                            "scoring_thresholds": design["thresholds"],
                        }
                        write_json(directory / "case.json", case)
                        command = simulation_command(
                            args, design, policy, condition, seed, directory, robot_usd
                        )
                        status = {
                            "status": "running",
                            "command": command,
                            "started_unix_s": time.time(),
                        }
                        write_json(directory / "run_status.json", status)
                        print(f"Starting {directory.name}", flush=True)
                        with (directory / "isaac.log").open("w") as log:
                            child = subprocess.Popen(
                                command,
                                cwd=args.source,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            try:
                                exit_code = child.wait(timeout=args.rollout_timeout)
                            except subprocess.TimeoutExpired:
                                stop_owned(child)
                                status.update(
                                    status="timeout",
                                    finished_unix_s=time.time(),
                                    exit_code=child.returncode,
                                )
                                write_json(directory / "run_status.json", status)
                                raise
                        child = None
                        complete = exit_code == 0 and all(
                            (directory / name).is_file()
                            for name in (
                                "run.json",
                                "sim_trace.npz",
                                "effective_config.json",
                                "policy_info.json",
                                "execution_trace.jsonl",
                                "planner_trace.json",
                            )
                        )
                        status.update(
                            status="completed" if complete else "runtime_error",
                            exit_code=exit_code,
                            finished_unix_s=time.time(),
                        )
                        write_json(directory / "run_status.json", status)
                        ledger["trials"][directory.name] = status
                        write_json(ledger_path, ledger)
                        if not complete:
                            raise RuntimeError(
                                f"Trial failed before producing complete evidence: {directory}"
                            )
                        run = json.loads((directory / "run.json").read_text())
                        info = json.loads((directory / "policy_info.json").read_text())
                        case["effective_parameters"] = {
                            "scene_sha256": fingerprint(
                                directory / "effective_config.json"
                            ),
                            "policy_info_sha256": fingerprint(
                                directory / "policy_info.json"
                            ),
                            "checkpoint_sha256": info.get("ckpt_sha"),
                            "observation_delay_s": info.get("observation_delay_s"),
                            "inference_delay_add_s": info.get("inference_delay_add_s"),
                            "max_play_steps": info.get("max_play_steps"),
                            "policy_latency_override_s": info.get(
                                "policy_latency_override_s"
                            ),
                            "policy_delivery_clock": info.get(
                                "policy_delivery_clock", "native"
                            ),
                            "policy_initial_state_provenance": info.get(
                                "policy_initial_state_provenance"
                            ),
                            "duration_s": run["duration_s"],
                            "stop_reason": run.get("policy_stop_reason"),
                            "completed_reason": run.get("policy_completed_reason"),
                        }
                        write_json(directory / "case.json", case)
                        if robot_usd is None:
                            robot_usd = run.get("robot_usd")
                        print(
                            f"Completed {directory.name}; duration={run['duration_s']:.3f}s stop={run.get('policy_stop_reason')}",
                            flush=True,
                        )
                    stop_owned(server)
                    server = None
                subprocess.run(analysis_command(args, snapshot), cwd=REPO, check=True)
            ledger["status"] = "all_trials_completed"
        except BaseException as error:
            ledger.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "error",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            try:
                stop_owned(child)
            finally:
                try:
                    stop_owned(server)
                finally:
                    ledger["finished_unix_s"] = time.time()
                    write_json(ledger_path, ledger)


if __name__ == "__main__":
    main()
