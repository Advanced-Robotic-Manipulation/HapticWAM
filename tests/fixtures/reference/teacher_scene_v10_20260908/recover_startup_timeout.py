#!/usr/bin/env python3
"""Explicit, file-only recovery of the V10 case-28 startup timeout.

This never executes a rollout. It preserves the empty startup attempt before
making its original schedule slot available to the unchanged frozen runner.
The recorded infrastructure failure remains part of the experiment history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket

EXPECTED_STUDY = "76634921fcb61b9a2aebf6530df44dad6168036c12cfefdb8a20c3d67d11268b"
EXPECTED_RUNNER = "cebca2b621bbf47e182ea190254a8dcb030d422b47fe5fe53f4bb1d21f09c084"
TRIAL = "r4_legacy/fta1500_nfe1_k4__aug22_1787396461_000__seed908101"
ALLOWED_FILES = {"server_ready.json", "resources_before.json", "case.json", "isaac.log",
                 "effective_config.json", "run_status.json", "command.json"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def hashes(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refuse indirect evidence path: {path}")
        if path.is_file():
            result[str(path.relative_to(directory))] = digest(path)
    return result


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def preflight(base):
    study_path = base / "study.json"
    ensure(digest(study_path) == EXPECTED_STUDY, "Unexpected frozen study")
    study = read(study_path)
    output = Path(study["runtime"]["output"])
    ensure(output.resolve() == (base / "campaign").resolve(), "Unexpected output directory")
    runner = Path(study["runtime"]["source"]) / "tests/fixtures/reference/teacher_scene_v10_20260908/run_study.py"
    ensure(digest(runner) == EXPECTED_RUNNER, "Frozen runner changed")
    frozen = read(output / "frozen_inputs.json")
    ensure(frozen["runner_sha256"] == EXPECTED_RUNNER and frozen["study_sha256"] == EXPECTED_STUDY,
           "Frozen source identity differs")
    progress = read(output / "progress.json")
    ensure(progress["status"] == "error" and progress["active_trial"] == TRIAL,
           "Not the known case-28 timeout")
    ensure(len(progress["trials"]) == 28 and sum(v["status"] == "completed" for v in progress["trials"].values()) == 27,
           "Recovery only supports 27 completed cases and one startup timeout")
    variant, trial = TRIAL.split("/")
    directory = output / variant / "rollouts" / trial
    ensure(set(p.name for p in directory.iterdir()) == ALLOWED_FILES,
           "Attempt has unexpected or physical rollout evidence; refuse replacement")
    ensure((directory / "isaac.log").stat().st_size == 0, "Isaac startup log is not empty")
    status, case = read(directory / "run_status.json"), read(directory / "case.json")
    ensure(status["status"] == "timeout" and status["exit_code"] == -15,
           "Not the known owned 900-second timeout termination")
    elapsed = status["finished_unix_s"] - status["started_unix_s"]
    ensure(900 <= elapsed < 905, "Unexpected timeout duration")
    ensure(case["study_sha256"] == EXPECTED_STUDY and case["schedule_index"] == 27 and case["attempt"] == 1,
           "Unexpected attempt provenance")
    command = read(directory / "command.json")
    ensure(command == status["command"], "Command records disagree")
    ensure(command[command.index("--output") + 1] == str(directory), "Unexpected command output")
    config = Path(command[command.index("--config") + 1])
    ensure(read(config) == read(directory / "effective_config.json"), "Effective scene differs")
    server = Path(case["server_metadata"]).parent
    server_text = (server / "server.log").read_text()
    connections = []
    for line in server_text.splitlines():
        if " connected" in line:
            match = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})", line)
            ensure(match is not None, "Unrecognized server connection timestamp")
            # compute3 records Python logging timestamps in MSK, verified by
            # date/time.tzname. run_status Unix timestamps remain UTC-based.
            connections.append(datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f").replace(
                tzinfo=timezone(timedelta(hours=3))).timestamp())
    ensure(connections and max(connections) < status["started_unix_s"],
           "A policy client connected during this attempt; refuse startup-only recovery")
    completed = {}
    for key, value in progress["trials"].items():
        if value["status"] != "completed":
            continue
        vid, tid = key.split("/")
        target = output / vid / "rollouts" / tid
        ensure(read(target / "run_status.json")["status"] == "completed", "Completed status disagreement")
        ensure(read(target / "runtime_audit.json")["valid"] is True, "Completed runtime audit not valid")
        completed[key] = hashes(target)
    return study, output, directory, server, completed, {
        "timeout_elapsed_wall_s": elapsed,
        "last_policy_client_connected_unix_s": max(connections),
        "server_log_timezone": "MSK, UTC+03:00 (host verified)",
        "empty_isaac_log": True,
        "physical_rollout_artifacts_absent": True,
        "resources_before": read(directory / "resources_before.json"),
    }


def recover(base, execute=False):
    study, output, directory, server, completed, evidence = preflight(base)
    archive = base / "amendments/startup_timeout_case28_attempt1"
    ensure(not archive.exists(), "Recovery already archived; do not repeat")
    proof = {
        "schema_version": 1, "kind": "explicit_startup_infrastructure_retry",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_sha256": EXPECTED_STUDY, "runner_sha256": EXPECTED_RUNNER,
        "recovery_helper_sha256": digest(Path(__file__)), "trial": TRIAL,
        "schedule_index": 27, "planned_trials_unchanged": 60,
        "archived_attempt": 1, "next_infrastructure_attempt": 2,
        "canonical_retry_directory": str(directory), "archive": str(archive),
        "reason": "900-second simulator startup timeout; no policy connection or physical rollout evidence; exact lower-level cause unknown",
        "authorization": "User requested resuming the experiments; explicit one-time startup retry disclosed before execution",
        "scoring": "Original timeout remains an infrastructure failure, not a physical success/failure. Same scheduled row and seed; report retry history and original-attempt completeness separately.",
        "case_attempt_field": "Unchanged frozen runner writes attempt=1 in each fresh canonical directory; this recovery ledger records the true second infrastructure attempt.",
        "changes": "Archive startup-only directory; no edits to runner, scene, policy, hardware, thresholds, timeout, schedule or completed physical outcomes",
        "attempt_file_sha256": hashes(directory),
        "completed_trial_file_sha256": completed,
        "frozen_inputs_sha256": digest(output / "frozen_inputs.json"),
        "command_sha256": digest(directory / "command.json"),
        "evidence": evidence,
    }
    if not execute:
        return proof
    # A free controller lock is held by main. Refuse any remaining owned child
    # or policy service before freeing this slot for the unchanged runner.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            parts = (proc / "cmdline").read_bytes().split(b"\0")
        except (OSError, ProcessLookupError):
            continue
        exact = [p.decode(errors="replace") for p in parts]
        if any(p.endswith(("/run_study.py", "/policy_server.py", "/run_waffles.py")) for p in exact):
            ensure(not any(str(study["runtime"]["source"]) in p for p in exact),
                   f"Study process still running: {proc.name}")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", int(study["runtime"]["port"])))
    archive.mkdir(parents=True)
    for name in ["progress.json", "frozen_inputs.json", "plan.json", "startup_resources.json"]:
        shutil.copy2(output / name, archive / name)
    shutil.copy2(base / "execute_after_gpu_free.log", archive / "execute_after_gpu_free.log")
    shutil.copytree(server, archive / "server")
    shutil.copy2(Path(study["runtime"]["source"]) / "tests/fixtures/reference/teacher_scene_v10_20260908/run_study.py", archive / "run_study.py")
    (archive / "recovery_pending.json").write_text(json.dumps(proof, indent=2) + "\n")
    directory.rename(archive / "attempt1")
    ensure(hashes(archive / "attempt1") == proof["attempt_file_sha256"], "Archive bytes changed")
    for key, expected in completed.items():
        vid, tid = key.split("/")
        ensure(hashes(output / vid / "rollouts" / tid) == expected, f"Completed trial changed: {key}")
    proof["completed_trials_preserved"] = 27
    proof["archive_verified"] = True
    proof["archive_file_sha256"] = hashes(archive)
    (base / "amendments/startup_timeout_recovery.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", action="store_true", help="Archive the reviewed startup attempt; does not execute sim")
    args = parser.parse_args()
    base = args.root.resolve()
    with (base / "campaign/controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = recover(base, args.archive)
    print(json.dumps({k: v for k, v in result.items() if k not in {"completed_trial_file_sha256", "archive_file_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
