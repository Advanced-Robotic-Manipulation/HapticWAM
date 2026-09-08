#!/usr/bin/env python3
"""Compare historical and repeat first observations on one owned teacher server.

This is a proposal diagnostic, not a policy rollout or success score. It neither
launches a server nor connects to hardware. Every call resets seed4242.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.sim.diagnose_rgb_transfer import (
    acquire_diagnostic_policy,
    array_sha,
    compare_proposals,
    file_sha,
    native_observation,
    proposal,
    validate_owned_server,
    write_json,
)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def calls(policy, sources, out):
    reference = sources["historical"]
    for label, arrays in sources.items():
        if set(arrays) != set(reference) or any(
            array_sha(arrays[key]) != array_sha(reference[key])
            for key in reference
            if key != "rgb"
        ):
            raise ValueError(f"Non-RGB input differs: {label}")
    results = {}
    for label in [*sources, "historical_repeat"]:
        arrays = reference if label == "historical_repeat" else sources[label]
        policy.remote_reset(4242)
        plan = policy.replan(
            native_observation(arrays),
            None,
            np.asarray(reference["ur_state"][12:18], float),
        )
        results[label] = proposal(plan)
        write_json(out / f"{label}.json", results[label])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--repeat1", type=Path, required=True)
    parser.add_argument("--repeat2", type=Path, required=True)
    parser.add_argument("--server-ready", type=Path, required=True)
    parser.add_argument("--owned-server-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new diagnostic directory")
    paths = {
        label: getattr(args, label) for label in ("historical", "repeat1", "repeat2")
    }
    original = json.loads(
        (args.historical.parent.parent / "server_ready.json").read_text()
    )
    ready = validate_owned_server(args.server_ready, args.owned_server_pid, original)
    sources = {label: load_arrays(path) for label, path in paths.items()}
    for path in paths.values():
        if args.out.resolve().is_relative_to(path.parent.parent.resolve()):
            raise ValueError("Diagnostic output cannot modify a source rollout")
    args.out.mkdir(parents=True)
    write_json(
        args.out / "manifest.json",
        {
            "scope": "Four reset first-plan calls; no controller, physics or success score",
            "seed": 4242,
            "inputs": {
                label: {
                    "path": str(path),
                    "sha256": file_sha(path),
                    "arrays": {
                        key: array_sha(value) for key, value in sources[label].items()
                    },
                }
                for label, path in paths.items()
            },
            "server_ready": ready,
            "driver_sha256": file_sha(Path(__file__)),
            "helper_sha256": file_sha(REPO / "tools/sim/diagnose_rgb_transfer.py"),
        },
    )
    with acquire_diagnostic_policy(ready, args.out) as policy:
        results = calls(policy, sources, args.out)
    comparisons = {
        label: compare_proposals(results["historical"], value)
        for label, value in results.items()
        if label != "historical"
    }
    write_json(args.out / "comparisons.json", comparisons)
    print(json.dumps({"status": "complete", "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
