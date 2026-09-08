#!/usr/bin/env python3
"""Record the owned diagnostic launcher's exit status; no retries or process killing."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.bindings.read_text())
    if binding.get("root_execution_authorized") is not True:
        raise RuntimeError("Root explicit GO required before execution")
    root = Path(binding["output_root"]).parent
    if (root / "launcher_record.json").exists() or (
        root / "launcher_completed.json"
    ).exists():
        raise RuntimeError("Refuse existing launch record; no retry")
    command = [
        sys.executable,
        str(root / "launch_dwell.py"),
        "--bindings",
        str(args.bindings),
        "--execute",
    ]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    with (root / "launcher.log").open("x") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write(
            root / "launcher_record.json",
            {
                "status": "launched",
                "pid": process.pid,
                "supervisor_pid": os.getpid(),
                "command": command,
                "started_at_utc": now(),
                "binding_sha256": hashlib.sha256(
                    args.bindings.read_bytes()
                ).hexdigest(),
                "planned_trials": 4,
                "order": "AABB",
                "root_authorized": True,
            },
        )
        rc = process.wait()
    write(
        root / "launcher_completed.json",
        {
            "launcher_exit_code": rc,
            "finished_at_utc": now(),
            "status": "exited_zero" if rc == 0 else "failed_no_retry",
        },
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
