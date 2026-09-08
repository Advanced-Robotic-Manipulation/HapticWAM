#!/usr/bin/env python3
"""Prepare V9 files locally from recovered frozen V8 evidence; never execute trials."""

import copy
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V8 = HERE.parents[1] / "teacher_carry_hotfix_v8" / "launchplan"
REMOTE = Path(
    "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_release_dwell_v9"
)
SEEDS = [904301, 904302]
ORDER = [("control_dwell200", 0.2), ("treatment_dwell100", 0.1)]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    baseline = json.loads((V8 / "campaigns/bounded_hold.json").read_text())
    old = json.loads((V8 / "bindings.frozen.json").read_text())
    cases = []
    pins = copy.deepcopy(old["files"])
    for block_id, dwell in ORDER:
        design = copy.deepcopy(baseline)
        design.update(
            {
                "campaign_id": "teacher_release_dwell_v9_" + block_id,
                "frozen_at_utc": now,
                "freeze_authorization": "Root requested four-case preparation, two blocks required by unchanged driver; execution awaits explicit root GO.",
                "sampling_seeds": SEEDS,
                "planned_counts": {
                    "conditions": 1,
                    "seeds_per_condition": 2,
                    "per_policy": 2,
                    "primary": 2,
                    "secondary": 0,
                    "total": 2,
                },
                "execution_order": "AABB: control0.2 seeds904301/904302, treatment0.1 seeds904301/904302. Frozen driver requires complete two-seed campaigns; ABBA rejected during CPU preflight before any policy execution.",
                "comparability_requirements": [
                    "Four reused-seed development trials only; no retries, expansion, or model ranking.",
                    "Frozen V8 bounded-hold runtime, inference, teacher, hardware, scene, sensors and RPC timing; only placement_release.opening_hold_s differs.",
                    "Sequential two-trial setting blocks; actual shared-GPU timing and setting order remain possible confounds.",
                ],
                "execution_phases": [
                    {
                        "id": block_id,
                        "policy_ids": ["fta1500_nfe1_k4"],
                        "condition_ids": ["successful_anchor"],
                        "sampling_seeds": SEEDS,
                    }
                ],
                "development_split_note": "Both sampling seeds already observed in V7/V8; V9 is a separate release-dwell diagnostic, not held-out confirmation.",
            }
        )
        design["adapter_profile"]["placement_release"]["opening_hold_s"] = dwell
        design["analysis"].update(
            {
                "weighting": "Two matched reused sampling seeds; one arm start and one object pose.",
                "early_failure": "Keep every executed failure; this diagnostic performs no retry or replacement.",
                "reproduction_note": "Four development cells, two release dwell settings, AABB order; physical stages scored independently of FINISH.",
            }
        )
        local = HERE / "campaigns" / (block_id + ".json")
        write(local, design)
        remote = REMOTE / "campaigns" / local.name
        pins.append({"path": str(remote), "sha256": sha(local)})
        cases.append(
            {
                "id": block_id,
                "campaign": str(remote),
                "campaign_sha256": sha(local),
                "hardware": baseline["runtime_hardware"]["source"],
                "sampling_seeds": SEEDS,
                "opening_hold_s": dwell,
            }
        )
    for name in ["launch_dwell.py", "prepare_twoblock.py", "run_supervised.py"]:
        pins.append({"path": str(REMOTE / name), "sha256": sha(HERE / name)})
    pins.append(
        {
            "path": str(Path(old["output_root"]).parent / "bindings.frozen.json"),
            "sha256": sha(V8 / "bindings.frozen.json"),
        }
    )
    binding = {
        key: copy.deepcopy(old[key])
        for key in [
            "schema_version",
            "runtime",
            "driver",
            "live_repo",
            "original_live_repo",
            "server_python",
            "evidence",
            "robot_usd",
            "port",
            "inventories",
        ]
    }
    binding.update(
        {
            "status": "frozen",
            "cpu_preflight_passed": False,
            "root_execution_authorized": False,
            "frozen_at_utc": now,
            "output_root": str(REMOTE / "paired"),
            "baseline_campaign": str(
                Path(old["output_root"]).parent / "campaigns/bounded_hold.json"
            ),
            "files": pins,
            "cases": cases,
            "planned_trials": 4,
            "interpretation": "Separate four-cell release-dwell diagnostic, AABB order; no retraining, source changes, retries or winner claim.",
        }
    )
    write(HERE / "bindings.prepared.json", binding)
    print(
        json.dumps(
            {
                "status": "prepared_not_executed",
                "cases": cases,
                "binding_sha256": sha(HERE / "bindings.prepared.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
