#!/usr/bin/env python3
"""Run the fixed waffles replay/dynamics validation, optionally a real policy.

This is an offline simulator campaign. It never invokes a hardware launcher.
Use a new output directory; completed cases are reused only after verifying
their input episode and mode match the requested case.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--policy-server",
        help="Opt in to a resident checkpoint test, e.g.127.0.0.1:7777",
    )
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.evidence = args.evidence.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    fit = args.evidence / "fit/ep_teacher_waffles_1788535016_005"
    cases = [("replay_fit", "replay", fit, [])]
    for episode in sorted((args.evidence / "heldout").glob("ep_*/replay.npz")):
        suffix = episode.parent.name.removeprefix("ep_teacher_waffles_")
        cases.append((f"replay_heldout_{suffix}", "replay", episode.parent, []))
    cases.extend(
        [
            ("dynamics_fit", "dynamics", fit, []),
            (
                "dynamics_push",
                "dynamics",
                fit,
                ["--push-at", "3", "--push-velocity", ".4", "0", "0"],
            ),
        ]
    )
    if args.policy_server:
        cases.append(
            ("policy_v5_6", "policy", fit, ["--policy-server", args.policy_server])
        )
    robot_usd = None
    results = []
    for name, mode, episode, extra in cases:
        out = args.output / name
        out.mkdir(parents=True, exist_ok=True)
        report_file = out / "run.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            if report["episode"] != str(episode) or report["mode"] != mode:
                raise RuntimeError(f"Existing run {out} belongs to different inputs")
        else:
            command = [
                str(repo / "tools/sim/launch_waffles.sh"),
                "--episode",
                str(episode),
                "--output",
                str(out),
                "--mode",
                mode,
                *extra,
            ]
            if robot_usd:
                command += ["--robot-usd", robot_usd]
            print(f"Running {name}", flush=True)
            with (out / "isaac.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            if not report_file.exists() or (out / "FAILED.txt").exists():
                raise RuntimeError(f"Simulator did not complete cleanly: {out}")
            report = json.loads(report_file.read_text())
        if name == "replay_fit":
            robot_usd = report["robot_usd"]
        if mode != "policy":
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "tools/sim/compare_replay.py"),
                    "--reference",
                    str(episode),
                    "--sim-trace",
                    str(out / "sim_trace.npz"),
                    "--sim-video",
                    str(out / "sim.mp4"),
                    "--out",
                    str(out / "comparison"),
                ],
                check=True,
            )
        results.append(
            {
                "case": name,
                "mode": mode,
                "output": str(out),
                "frames": report["frames"],
                "duration_s": report["duration_s"],
            }
        )
        (args.output / "campaign.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Completed {name}: {report['frames']} frames", flush=True)


if __name__ == "__main__":
    main()
