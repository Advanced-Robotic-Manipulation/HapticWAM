#!/usr/bin/env python3
"""Read-only completion/provenance audit for the 60-trial teacher study.

Writes only --out. It starts no simulation, inference or hardware runtime and
never terminates a process. Physical task failures are valid experimental
outcomes; this audit checks evidence completeness and declared-input fidelity.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tests.fixtures.reference.teacher_scene_v10_20260908 import run_study
from tools.sim import run_policy_campaign as campaign


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def differences(expected, actual, prefix=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            name = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                result.append({"field": name, "expected": expected.get(key), "actual": actual.get(key)})
            else:
                result += differences(expected[key], actual[key], name)
        return result
    return [] if expected == actual else [{"field": prefix, "expected": expected, "actual": actual}]


def full_file_hashes(directory):
    return {str(path.relative_to(directory)): fingerprint(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def infrastructure_recovery_audit(study_path, args):
    """Expose the one startup retry even though the frozen runner uses attempt=1."""
    proof_path = study_path.parent / "amendments/startup_timeout_recovery.json"
    archive = study_path.parent / "amendments/startup_timeout_case28_attempt1"
    if not proof_path.exists():
        check(not archive.exists(), "Startup archive exists without its completed recovery proof")
        return {"valid": True, "retry_count": 0}
    proof = read(proof_path)
    check(proof["kind"] == "explicit_startup_infrastructure_retry", "Unknown recovery kind")
    check(proof["study_sha256"] == fingerprint(study_path), "Recovery study differs")
    check(proof["runner_sha256"] == fingerprint(Path(run_study.__file__)), "Recovery runner differs")
    helper = args.source / "tests/fixtures/reference/teacher_scene_v10_20260908/recover_startup_timeout.py"
    check(proof["recovery_helper_sha256"] == fingerprint(helper), "Recovery helper changed")
    check(proof["archive_verified"] is True and proof["completed_trials_preserved"] == 27,
          "Recovery did not preserve the 27 completed outcomes")
    check(proof["planned_trials_unchanged"] == 60 and proof["schedule_index"] == 27
          and proof["archived_attempt"] == 1 and proof["next_infrastructure_attempt"] == 2,
          "Recovery changed the declared grid/attempt accounting")
    check(Path(proof["archive"]).resolve() == archive.resolve(), "Unexpected startup archive")
    check(full_file_hashes(archive) == proof["archive_file_sha256"], "Startup recovery archive changed")
    original = archive / "attempt1"
    check(full_file_hashes(original) == proof["attempt_file_sha256"], "Startup attempt bytes changed")
    required = {"server_ready.json", "resources_before.json", "case.json", "isaac.log",
                "effective_config.json", "run_status.json", "command.json"}
    check(set(proof["attempt_file_sha256"]) == required and (original / "isaac.log").stat().st_size == 0,
          "Original attempt was not the documented startup-only timeout")
    check(read(original / "run_status.json")["status"] == "timeout", "Original timeout was relabeled")
    preserved = proof["completed_trial_file_sha256"]
    check(len(preserved) == 27, "Recovery preserved-trial manifest has a different count")
    schedule = read(study_path)["schedule"]
    check(len(schedule) == 60, "Recovery schedule does not contain sixty planned cases")
    preserved_indices = []
    for name, expected in preserved.items():
        variant, case_id = name.split("/")
        directory = args.output / variant / "rollouts" / case_id
        check(full_file_hashes(directory) == expected,
              f"A completed pre-recovery trial changed: {name}")
        case = read(directory / "case.json")
        index = case["schedule_index"]
        check(type(index) is int and 0 <= index < 27,
              f"Preserved trial is not among the first 27 scheduled cases: {name}")
        row = schedule[index]
        check(case["variant"] == variant == row["variant"]
              and case["condition_id"] == row["condition"]
              and case["sampling_seed"] == row["seed"]
              and case_id == campaign.trial_id(case["policy_id"], row["condition"], row["seed"]),
              f"Preserved trial identity differs from the frozen schedule: {name}")
        preserved_indices.append(index)
    check(sorted(preserved_indices) == list(range(27)),
          "Recovery manifest does not preserve each scheduled case 0 through 26 exactly once")
    variant, case_id = proof["trial"].split("/")
    canonical = args.output / variant / "rollouts" / case_id
    check(canonical.resolve() == Path(proof["canonical_retry_directory"]).resolve(), "Retry directory differs")
    check(fingerprint(canonical / "command.json") == proof["command_sha256"]
          == fingerprint(original / "command.json"), "Retry command differs from archived startup command")
    old_case, new_case = read(original / "case.json"), read(canonical / "case.json")
    row = schedule[27]
    check(old_case["schedule_index"] == new_case["schedule_index"] == 27
          and old_case["variant"] == variant == row["variant"]
          and old_case["condition_id"] == row["condition"]
          and old_case["sampling_seed"] == row["seed"]
          and case_id == campaign.trial_id(old_case["policy_id"], row["condition"], row["seed"]),
          "Startup retry is not the original scheduled case 27")
    for field in ("study_sha256", "campaign_sha256", "sampling_seed", "condition_id", "policy_id", "variant",
                  "schedule_index", "checkpoint_sha256", "intended_parameters", "intended_scene_sha256", "scoring_thresholds"):
        check(old_case[field] == new_case[field], f"Retry changed the scheduled case: {field}")
    created = datetime.fromisoformat(proof["created_at_utc"]).timestamp()
    check(read(canonical / "run_status.json")["started_unix_s"] >= created,
          "Canonical run does not follow the disclosed recovery")
    return {"valid": True, "retry_count": 1, "trial": proof["trial"], "schedule_index": 27,
            "proof": str(proof_path), "proof_sha256": fingerprint(proof_path),
            "preserved_completed_trials": 27, "same_command_and_seed": True,
            "original_timeout_had_physical_evidence": False,
            "interpretation": "60 planned rows; case28 has two infrastructure attempts. Its first attempt remains an unscored startup timeout, and its retried physical result is explicitly identified. Canonical attempt=1 is the unchanged runner's directory-local field, not total launch count."}


def source_audit(study_path, study, args, variants):
    expected = read(args.output / "frozen_inputs.json")
    current = {"study_sha256": fingerprint(study_path),
               "runner_sha256": fingerprint(Path(run_study.__file__)), "variants": {}}
    variant_differences = {}
    for vid, data in variants.items():
        sources = campaign.source_manifest(data["args"], data["sha256"], data["design"])
        sources.update(study_sha256=current["study_sha256"],
                       shared_server_hardware=data["design"]["shared_server_hardware"])
        current["variants"][vid] = sources
        variant_differences[vid] = differences(read(data["args"].output / "frozen_inputs.json"), sources)
        check(fingerprint(data["args"].output / "campaign_snapshot.json") == data["sha256"],
              f"Campaign snapshot changed: {vid}")
    check(fingerprint(args.output / "study_snapshot.json") == current["study_sha256"],
          "Study snapshot differs from supplied frozen study")
    return {"valid": expected == current and not any(variant_differences.values()),
            "frozen_manifest": str(args.output / "frozen_inputs.json"),
            "frozen_manifest_sha256": fingerprint(args.output / "frozen_inputs.json"),
            "differences": differences(expected, current),
            "per_variant_differences": variant_differences}


def replay_audit(path, args):
    audit = read(path)
    check(audit.get("gate_status") == "passed" and audit.get("policy_launch_prerequisite_passed") is True,
          "Replay prerequisite was not passed")
    protocol_path = path.parent / "replay_protocol.json"
    check(fingerprint(protocol_path) == audit["protocol_sha256"], "Replay protocol hash differs")
    check(read(protocol_path) == audit["protocol"] == audit["progress"]["protocol"],
          "Replay protocol declarations disagree")
    protocol = audit["protocol"]
    expected = {(scene, ep, mode) for scene in protocol["scenes"]
                for ep in protocol["episodes"] for mode in protocol["modes"]}
    keys = [(row["scene"], row["episode"], row["mode"]) for row in audit["cases"]]
    check(len(keys) == len(set(keys)) == 8 and set(keys) == expected,
          "Replay prerequisite does not cover the eight frozen cases exactly once")
    progress = {(r["scene"], r["episode"], r["mode"]): r for r in audit["progress"]["cases"]}
    results = []
    for row in audit["cases"]:
        directory = Path(row["directory"])
        key = (row["scene"], row["episode"], row["mode"])
        recorded = progress[key]
        check(recorded["status"] == "completed" and recorded["returncode"] == 0,
              f"Replay execution was incomplete: {key}")
        check(row["gates"] and all(v is True for v in row["gates"].values()),
              f"A mechanics replay gate was not passed: {key}")
        hashes = {name: fingerprint(directory / name) for name in row["sha256"]}
        check(hashes == row["sha256"], f"Replay raw evidence changed: {key}")
        command = recorded["command"]
        reference = Path(command[command.index("--episode") + 1]) / "replay.npz"
        check(fingerprint(reference) == row["reference_replay_sha256"],
              f"Replay reference recording changed: {key}")
        check(Path(command[command.index("--output") + 1]) == directory,
              f"Replay command output differs: {key}")
        scene_file = Path(command[command.index("--config") + 1])
        scene_key = "r4" if row["scene"] == "r4_d435" else "old_scene.replay.json"
        check(fingerprint(scene_file) == protocol["hashes"][scene_key],
              f"Replay frozen scene changed: {key}")
        results.append({"scene": row["scene"], "episode": row["episode"], "mode": row["mode"],
                        "directory": str(directory), "hashes_verified": len(hashes),
                        "object_replay_verdict": row.get("object_replay", {}).get("verdict")})
    source_hashes = {name: fingerprint(args.source / name) for name in audit["source_hashes"]}
    check(source_hashes == audit["source_hashes"], "Replay mechanics source changed after its gate")
    return {"valid": True, "path": str(path), "sha256": fingerprint(path),
            "audited_at_utc": audit["audited_at_utc"], "cases": results,
            "interpretation": audit["policy_outcome_interpretation"]}


def verify_adoption(directory, case, study_path):
    adoption = case.get("metadata_only_adoption")
    if not adoption:
        return None
    proof_path = Path(adoption["proof"])
    check(fingerprint(proof_path) == adoption["sha256"], "Metadata-adoption proof hash differs")
    proof = read(proof_path)
    original = Path(proof["original_trial"])
    check(Path(proof["adopted_trial"]).resolve() == directory.resolve(), "Adopted-trial path differs")
    check(proof["corrected_study_sha256"] == fingerprint(study_path), "Adoption names a different study")
    check(proof["extra_simulations"] == 0 and proof["simulation_command_identical"] is True,
          "Adoption did not declare a no-rerun metadata correction")
    for name, field in (("case.json", "original_case_sha256"),
                        ("run_status.json", "original_status_sha256"),
                        ("runtime_audit.json", "original_audit_sha256")):
        check(fingerprint(original / name) == proof[field], f"Archived original changed: {name}")
    check(adoption["original_case_sha256"] == proof["original_case_sha256"],
          "Case and proof original-case fingerprints differ")
    mutable = {"case.json", "run_status.json", "runtime_audit.json"}
    raw_names = {str(p.relative_to(original)) for p in original.rglob("*")
                 if p.is_file() and p.name not in mutable}
    check(raw_names == set(proof["raw_unchanged_sha256"]), "Adoption proof does not cover every archived raw file")
    for name, sha in proof["raw_unchanged_sha256"].items():
        check(fingerprint(original / name) == sha == fingerprint(directory / name),
              f"Original/adopted raw evidence differs: {name}")
    old_status, status = read(original / "run_status.json"), read(directory / "run_status.json")
    for value in (old_status, status):
        for name in ("status", "audit_reasons", "metadata_only_adoption"):
            value.pop(name, None)
    check(old_status == status, "Adoption changed original launch command/timestamps/exit metadata")
    old_case, new_case = read(original / "case.json"), deepcopy(case)
    for value in (old_case, new_case):
        for name in ("study_sha256", "campaign_sha256", "server_metadata", "metadata_only_adoption"):
            value.pop(name, None)
    check(old_case == new_case, "Adoption changed non-amended original case metadata")
    old_audit = read(original / "runtime_audit.json")
    check(old_audit == {"valid": False,
                       "reasons": ["effective_terminal_veto_feedback_source_differs_from_campaign"],
                       "scene_differences": []}, "Original audit failure had a different cause")
    check(read(directory / "policy_info.json")["terminal_veto_feedback_source"] == "current_delivery",
          "Adopted trial did not actually use current-delivery feedback")
    run = read(directory / "run.json")
    check(proof["original_outcome_preserved"] == {
        "duration_s": run["duration_s"], "stop_reason": run.get("policy_stop_reason")},
        "Adoption outcome declaration differs from unchanged raw run")
    relocation = proof["server_metadata_relocation"]
    check(Path(case["server_metadata"]) == Path(relocation["archived"]), "Archived server metadata path differs")
    check(fingerprint(relocation["archived"]) == relocation["sha256"], "Archived server metadata changed")
    archive_inputs = proof_path.parent / "metadata_v0/inputs"
    old_study = read(archive_inputs / "study.json")
    check(fingerprint(archive_inputs / "study.json") == proof["original_study_sha256"],
          "Archived original study changed")
    current_study = read(study_path)
    for value in (old_study, current_study):
        value.pop("frozen_at_utc", None)
    check(old_study == current_study, "Metadata amendment changed study factors or ordering")
    for variant in current_study["variants"]:
        vid = variant["id"]
        archived = archive_inputs / "campaigns" / Path(variant["campaign"]).name
        current = Path(variant["campaign"])
        hashes = proof["campaign_bindings"][vid]
        check(fingerprint(archived) == hashes["original_sha256"] and fingerprint(current) == hashes["corrected_sha256"],
              f"Amended campaign binding changed: {vid}")
        old, new = read(archived), read(current)
        check(old["adapter_profile"]["terminal_veto_feedback_source"] == "request_snapshot_historical",
              "Archived campaign does not contain the stated original declaration")
        old["adapter_profile"]["terminal_veto_feedback_source"] = "current_delivery"
        old.pop("frozen_at_utc", None); new.pop("frozen_at_utc", None)
        check(old == new, f"Campaign amendment changed runtime settings: {vid}")
    return {"valid": True, "proof": str(proof_path), "proof_sha256": fingerprint(proof_path),
            "raw_files_identical": len(raw_names), "original_trial": str(original),
            "original_timestamps_unchanged": True, "extra_simulations": 0,
            "server_original_directory": str(Path(relocation["original"]).parent)}


def case_audit(index, row, data, args, study_path):
    local, design, policy = data["args"], data["design"], data["policy"]
    condition = data["conditions"][row["condition"]]
    directory = local.output / "rollouts" / campaign.trial_id(policy["id"], row["condition"], row["seed"])
    required = (*run_study.REQUIRED, "case.json", "command.json", "runtime_audit.json", "run_status.json")
    check(all((directory / name).is_file() for name in required), "Missing required case artifact")
    case, status, stored_audit = (read(directory / name) for name in ("case.json", "run_status.json", "runtime_audit.json"))
    expected = {"study_sha256": fingerprint(study_path), "campaign_sha256": data["sha256"],
                "policy_id": policy["id"], "condition_id": condition["id"], "sampling_seed": row["seed"],
                "checkpoint_sha256": policy["checkpoint_sha256"], "variant": row["variant"], "schedule_index": index}
    check(all(case.get(k) == value for k, value in expected.items()), "Case identity differs from frozen schedule")
    check(case.get("attempt") == 1, "Unexpected simulation replacement attempt")
    check(status.get("status") == "completed" and status.get("exit_code") == 0, "Case execution did not complete")
    check(stored_audit.get("valid") is True and not stored_audit.get("reasons") and not stored_audit.get("scene_differences"),
          "Stored runtime audit was not valid")
    command = campaign.simulation_command(local, design, policy, condition, row["seed"], directory, args.robot_usd)
    command += ["--record-robot-environment-contacts"]
    check(command == read(directory / "command.json") == status["command"], "Actual simulation command differs from frozen recipe")
    for flag in ("--episode", "--config", "--policy-config", "--hardware-config", "--robot-usd",
                 "--policy-initial-state", "--tactile-baseline", "--terminal-veto-config", "--placement-release-config"):
        if flag in command:
            check(Path(command[command.index(flag) + 1]).exists(), f"Command input path does not resolve: {flag}")
    ready, info, run = (read(directory / name) for name in ("server_ready.json", "policy_info.json", "run.json"))
    check(case["server_ready_sha256"] == fingerprint(directory / "server_ready.json"), "Saved server identity hash differs")
    check(case["intended_scene_sha256"] == fingerprint(local.output / "conditions" / f"{condition['id']}.json"),
          "Intended scene hash differs")
    effective = case["effective_parameters"]
    check(effective["scene_sha256"] == fingerprint(directory / "effective_config.json")
          and effective["policy_info_sha256"] == fingerprint(directory / "policy_info.json"),
          "Effective evidence differs from completion-time hashes")
    with np.load(directory / "sim_trace.npz", allow_pickle=False) as trace:
        times = np.asarray(trace["t"]).copy()
    reasons = campaign._analysis.runtime_audit(design, policy, condition, info, ready, run, times, run.get("policy_stop_reason"))
    reasons += campaign._analysis.policy_delivery_timing_audit(design, condition, campaign._analysis.read_rows(directory / "planner_trace.json"))
    scene_errors = campaign._analysis.scene_mismatches(campaign.condition_scene(design, condition), read(directory / "effective_config.json"))
    check(not reasons and not scene_errors, f"Recomputed runtime audit failed: {reasons}; scene={scene_errors}")
    adoption = verify_adoption(directory, case, study_path)
    server_path = Path(case["server_metadata"])
    server = read(server_path)
    old, current = deepcopy(ready), deepcopy(server)
    old.pop("status", None); current.pop("status", None)
    check(old == current, "Server metadata differs from the actual saved ready snapshot beyond lifecycle status")
    check(server["status"] == "stopped", f"Owned server did not record normal cleanup: {server['status']}")
    if adoption:
        server_command_directory = Path(adoption["server_original_directory"])
    else:
        server_command_directory = server_path.parent
    check(server_path.parent.name == server_command_directory.name, "Server archive relocated to a different owner directory")
    first = next(iter(run_study.load_study(study_path)[2].values()))
    expected_server = campaign.server_command(first["args"], first["design"], first["policy"], server_command_directory)
    check(read(server_path.parent / "command.json") == expected_server, "Server launch command differs from the shared teacher recipe")
    for name, sha in ready.get("inference_source_sha256", {}).items():
        check(fingerprint(args.live_repo / name) == sha, f"Model inference source changed: {name}")
    return {"valid": True, "directory": str(directory), "identity": expected,
            "case_sha256": fingerprint(directory / "case.json"), "raw_file_sha256": full_file_hashes(directory),
            "raw_hash_interpretation": "Current full evidence inventory; preexisting completion hashes and adoption proof checked separately",
            "stop_reason": run.get("policy_stop_reason"), "completed_reason": run.get("policy_completed_reason"),
            "started_unix_s": status["started_unix_s"], "finished_unix_s": status["finished_unix_s"],
            "server_metadata": str(server_path), "server_metadata_sha256": fingerprint(server_path),
            "server_pid": server.get("pid"), "metadata_only_adoption": adoption}


def process_diagnostics(args, progress, server_pids):
    processes = []
    tracked = {progress.get("controller_pid"), *server_pids} - {None}
    for path in Path("/proc").glob("[0-9]*"):
        try:
            argv = (path / "cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
            pid = int(path.name)
            relevant = [arg for arg in argv if arg.endswith(("run_study.py", "policy_server.py", "run_waffles.py", "launch_waffles.sh"))]
            owned = any(str(args.source) in arg for arg in relevant)
            if relevant or pid in tracked:
                processes.append({"pid": pid, "owned_study_command": owned,
                                  "tracked_pid": pid in tracked,
                                  "command": argv if owned else relevant,
                                  "executable": argv[0] if argv else None,
                                  "note": "PID alone does not establish ownership; reused/unrelated PIDs are not treated as study children"})
        except (OSError, ValueError):
            continue
    listeners = []
    for name in ("tcp", "tcp6"):
        path = Path("/proc/net") / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].rsplit(":", 1)[1], 16) == args.port and fields[3] == "0A":
                listeners.append({"table": name, "address": fields[1], "socket_inode": fields[9]})
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
        gpu = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu = {"query_error": str(error)}
    return {"valid": not listeners and not any(p["owned_study_command"] for p in processes),
            "owned_processes_stopped": not any(p["owned_study_command"] for p in processes),
            "dedicated_port": args.port, "port_listeners": listeners, "related_or_tracked_processes": processes,
            "gpu_processes_read_only": gpu, "processes_terminated": 0}


def audit(study_path, replay_path):
    study, args, variants = run_study.load_study(study_path)
    report = {"schema_version": 1, "study_sha256": fingerprint(study_path), "study": str(study_path),
              "auditor_sha256": fingerprint(__file__), "audited_unix_s": time.time(),
              "scope": "Evidence/input fidelity and process cleanup; physical success is not a completion prerequisite",
              "planned_cases": len(study["schedule"]), "cases": [], "errors": []}
    check(len(study["schedule"]) == 60, "Expected all sixty planned cases")
    for name, function in (("source_inputs", lambda: source_audit(study_path, study, args, variants)),
                           ("replay_prerequisite", lambda: replay_audit(replay_path, args)),
                           ("infrastructure_recovery", lambda: infrastructure_recovery_audit(study_path, args))):
        try:
            report[name] = function()
            if not report[name]["valid"]:
                report["errors"].append(f"{name}: verification failed")
        except Exception as error:
            report[name] = {"valid": False, "error": f"{type(error).__name__}: {error}"}
            report["errors"].append(f"{name}: {error}")
    checkpoints = {}
    for data in variants.values():
        policy = data["policy"]
        checkpoint = policy["checkpoint"]
        if checkpoint not in checkpoints:
            try:
                sha = fingerprint(checkpoint)
                checkpoints[checkpoint] = {"sha256": sha, "expected_sha256": policy["checkpoint_sha256"],
                                           "valid": sha == policy["checkpoint_sha256"]}
            except OSError as error:
                checkpoints[checkpoint] = {"valid": False, "error": str(error)}
    report["checkpoints"] = checkpoints
    if not all(item["valid"] for item in checkpoints.values()):
        report["errors"].append("Checkpoint content differs or is inaccessible")
    for index, row in enumerate(study["schedule"]):
        try:
            result = case_audit(index, row, variants[row["variant"]], args, study_path)
        except Exception as error:
            result = {"valid": False, "error": f"{type(error).__name__}: {error}"}
            report["errors"].append(f"Case {index}: {error}")
        report["cases"].append({"schedule_index": index, **row, **result})
        print(f"Audited {index + 1}/60 {row['variant']} {row['condition']} seed{row['seed']}: valid={result['valid']}", flush=True)
    actual = {p.resolve() for p in args.output.glob("*/rollouts/*/case.json")}
    expected = {(data["args"].output / "rollouts" / campaign.trial_id(data["policy"]["id"], row["condition"], row["seed"]) / "case.json").resolve()
                for row in study["schedule"] for data in [variants[row["variant"]]]}
    report["grid"] = {"valid": actual == expected, "missing": sorted(map(str, expected - actual)),
                      "unexpected": sorted(map(str, actual - expected))}
    if actual != expected:
        report["errors"].append("Output cases differ from the exact planned grid")
    progress = {}
    try:
        progress = read(args.output / "progress.json")
        report["progress"] = {"valid": False, "sha256": fingerprint(args.output / "progress.json"), **progress}
        check(progress.get("study_sha256") == report["study_sha256"] and progress.get("status") == "all_trials_completed",
              "Controller progress does not report this study complete")
        check(len(progress.get("trials", {})) == 60 and all(r.get("status") == "completed" for r in progress["trials"].values()),
              "Controller progress does not contain sixty completed outcomes")
        check(len(progress.get("analysis", {})) == len(variants)
              and all(r.get("status") == "completed" for r in progress["analysis"].values()),
              "Per-campaign analysis did not complete")
        report["progress"] = {"valid": True, "sha256": fingerprint(args.output / "progress.json"), **progress}
    except Exception as error:
        report["errors"].append(f"progress: {error}")
    try:
        pids = [r.get("server_pid") for r in report["cases"] if r["valid"]]
        report["cleanup"] = process_diagnostics(args, progress, pids)
        if not report["cleanup"]["valid"]:
            report["errors"].append("Study process or dedicated port remains active")
    except Exception as error:
        report["cleanup"] = {"valid": False, "error": str(error)}
        report["errors"].append(f"cleanup: {error}")
    if report.get("replay_prerequisite", {}).get("valid"):
        gate_t = datetime.fromisoformat(report["replay_prerequisite"]["audited_at_utc"].replace("Z", "+00:00")).timestamp()
        starts = [r["started_unix_s"] for r in report["cases"] if r["valid"]]
        report["replay_preceded_policy_trials"] = bool(starts) and gate_t <= min(starts)
        if not report["replay_preceded_policy_trials"]:
            report["errors"].append("Replay prerequisite was not audited before the original first policy trial")
    report["counts"] = {"valid_completed": sum(r["valid"] for r in report["cases"]),
                         "invalid_or_missing": sum(not r["valid"] for r in report["cases"]),
                         "stop_reasons": dict(Counter(r.get("stop_reason") for r in report["cases"] if r["valid"])),
                         "adopted_without_rerun": sum(bool(r.get("metadata_only_adoption")) for r in report["cases"])}
    recovery = report.get("infrastructure_recovery", {})
    report["counts"]["documented_startup_retries"] = recovery.get("retry_count")
    retried_valid = sum(r["valid"] and r["schedule_index"] == recovery.get("schedule_index") for r in report["cases"])
    report["counts"]["valid_without_startup_retry"] = (report["counts"]["valid_completed"] - retried_valid
        if recovery.get("valid") else None)
    report["valid"] = not report["errors"]
    report["finished_unix_s"] = time.time()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON audit report; raw evidence remains read-only")
    parser.add_argument("--replay-audit", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.study.resolve(), args.replay_audit.resolve())
    except Exception as error:
        report = {"valid": False, "fatal_error": f"{type(error).__name__}: {error}",
                  "study": str(args.study.resolve()), "audited_unix_s": time.time()}
    campaign.write_json(args.out.resolve(), report)
    print(json.dumps({"valid": report["valid"], "report": str(args.out.resolve()), "counts": report.get("counts")}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
