#!/usr/bin/env python3
"""Launch ONLY an explicitly prepared latency plan; never called by preparation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sha(args.plan) != args.expected_plan_sha256:
        raise ValueError("Plan hash differs from the reviewed preparation")
    plan = json.loads(args.plan.read_text())
    source = Path(plan["source"])
    if Path(plan["output"]).exists():
        raise FileExistsError("Diagnostic output already exists; preserve all attempts")
    for record in plan["inputs"].values():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError(f"Input changed: {record['path']}")
    for relative, expected in plan["source_files_sha256"].items():
        if sha(source / relative) != expected:
            raise ValueError(f"Prepared source changed: {relative}")
    if args.dry_run:
        print(json.dumps(plan["commands"], indent=2))
        return
    # Reuse the reviewed managed lifecycle; only its own Popen handles can be
    # terminated. This imports no Isaac, policy model or real hardware driver.
    spec = importlib.util.spec_from_file_location(
        "anchor_managed", source / "tools/sim/run_teacher_pick.py"
    )
    managed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(managed)
    request = SimpleNamespace(
        source=source,
        live_repo=Path(plan["live_repo"]),
        output=Path(plan["output"]),
        port=plan["port"],
        server_timeout=600.0,
        rollout_timeout=2400.0,
    )
    managed.check_port(request.port)
    result = managed.run_managed(request, plan)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": result["output"],
                "cleanup": result["cleanup"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
