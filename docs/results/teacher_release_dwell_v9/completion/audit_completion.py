#!/usr/bin/env python3
"""Read completed V9 scores, source pins and owned process records; no scoring or launch."""

import datetime
import importlib.util
import json
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
ROOT = Path(
    "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_release_dwell_v9"
)


def read(path):
    return json.loads(path.read_text())


def process_record(role, pid, **extra):
    stat = Path(f"/proc/{pid}/stat")
    state = stat.read_text().split()[2] if stat.exists() else None
    return {
        "role": role,
        "pid": pid,
        "proc_state": state,
        "running": state not in (None, "Z"),
        **extra,
    }


def main():
    spec = importlib.util.spec_from_file_location(
        "v9launcher", ROOT / "launch_dwell.py"
    )
    launch = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launch
    spec.loader.exec_module(launch)
    binding = read(ROOT / "bindings.frozen.json")
    marker = read(ROOT / "paired/complete.json")
    assert marker["status"] == "four_valid_trials_completed" and marker["trials"] == 4
    launcher_record = read(ROOT / "launcher_record.json")
    launcher_completed = read(ROOT / "launcher_completed.json")
    assert launcher_completed["launcher_exit_code"] == 0
    assert launcher_record["binding_sha256"] == launch.sha(
        ROOT / "bindings.frozen.json"
    )
    verified = launch.verify_files(binding)
    processes = [
        process_record("launcher", launcher_record["pid"], exit_code=0),
        process_record("supervisor", launcher_record["supervisor_pid"]),
    ]
    trials = []
    for block in binding["cases"]:
        block_id = block["id"]
        output = ROOT / "paired" / block_id
        checked = launch.complete(output, block["campaign_sha256"])
        completion = read(ROOT / "paired" / f"{block_id}_completed.json")
        assert completion["controller_exit_code"] == 0
        assert completion["summary_sha256"] == checked["summary_sha256"]
        progress = read(output / "progress.json")
        processes.append(
            process_record(
                block_id + "_controller", progress["controller_pid"], exit_code=0
            )
        )
        servers = list((output / "servers").glob("*/server.json"))
        assert len(servers) == 1
        server = read(servers[0])
        assert server["status"] == "stopped"
        processes.append(
            process_record(
                block_id + "_server", server["pid"], recorded_status=server["status"]
            )
        )
        for row in read(output / "analysis/summary.json")["trials"]:
            score_path = Path(row["score_path"])
            score = read(score_path)
            metrics = score["metrics"]
            assert metrics["valid_for_scoring"]
            trial = {
                "block": block_id,
                "case": Path(row["directory"]).name,
                "valid_for_scoring": True,
                "score_path": str(score_path),
                "score_sha256": launch.sha(score_path),
                "opening_hold_s": block["opening_hold_s"],
            }
            for key in ["outcomes", "event_times_s", "object", "control"]:
                trial[key] = metrics.get(key)
            trials.append(trial)
    with socket.socket() as sock:
        sock.settimeout(1)
        port_free = sock.connect_ex(("127.0.0.1", binding["port"])) != 0
    all_stopped = all(not row["running"] for row in processes)
    assert len(trials) == 4 and all_stopped and port_free
    result = {
        "status": "passed",
        "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "planned": 4,
        "valid": 4,
        "all_trials_complete": True,
        "owned_processes_stopped": all_stopped,
        "dedicated_port_free": port_free,
        "binding_sha256": launch.sha(ROOT / "bindings.frozen.json"),
        "verified_pinned_files": verified,
        "launcher_exit_code": 0,
        "processes": processes,
        "trials": trials,
        "notes": [
            "Existing authoritative scores read, never rescored.",
            "AABB setting blocks with actual shared-GPU timing; two reused development seeds.",
            "No winner or hardware-validation claim.",
        ],
    }
    with (ROOT / "completion_audit.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
