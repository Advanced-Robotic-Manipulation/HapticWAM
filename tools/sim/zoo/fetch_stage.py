#!/usr/bin/env python3
"""Fetch a zoo stage's raw root from compute3 into the local (untracked) artifacts tree and run the report.

Default fetch is the small per-trial summaries (run_status, trial_result, run,
effective_config, planner_trace, lane logs, server receipts); ``--full`` also
pulls the traces, observations and videos (~100 MB per trial) for preservation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HOST = "compute3"
REMOTE_RAW = "/dev/shm/phantom_sim_zoo_raw_20260912"
LOCAL_RAW = REPO / "artifacts/isaac_waffles/sim_zoo_20260912/raw"
SMALL = ["run_status.json", "trial_result.json", "run.json", "effective_config.json", "planner_trace.json",
         "policy_info.json", "initialization.json", "adaptive_policy_experiment.json", "boundary_physics_audit.json",
         "sim_first.png", "sim_last.png"]


def rsync(stage: str, full: bool) -> None:
    dst = LOCAL_RAW / stage
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--no-perms", "--no-owner", "--no-group", "--chmod=ugo=rwX"]
    if not full:
        cmd += ["--include=*/", "--include=lane*.log", "--include=ledger.json", "--include=server__*/**"]
        cmd += [f"--include=rollouts/*/{n}" for n in SMALL]
        cmd += ["--exclude=*"]
    cmd += [f"{HOST}:{REMOTE_RAW}/{stage}/", str(dst) + "/"]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--report-out", type=Path, default=None)
    ap.add_argument("--pair", nargs=2, action="append", default=[])
    args = ap.parse_args()
    rsync(args.stage, args.full)
    out = args.report_out or REPO / "docs/results/sim_zoo_20260912/report" / args.stage
    cmd = [sys.executable, str(REPO / "tools/sim/zoo/report.py"), "--study", str(args.study),
           "--raw", str(LOCAL_RAW / args.stage), "--output", str(out)]
    for a, b in args.pair:
        cmd += ["--pair", a, b]
    subprocess.run(cmd, check=True)
    print(json.dumps({"stage": args.stage, "local_raw": str(LOCAL_RAW / args.stage), "report": str(out), "full": args.full}))


if __name__ == "__main__":
    main()
