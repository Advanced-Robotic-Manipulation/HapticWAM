#!/usr/bin/env python3
"""Read-only final V8 audit; writes a compact artifact after all six trials."""

import hashlib
import importlib.util
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
R = B / "runs/teacher_carry_hotfix_v8"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mean(values):
    return sum(values) / len(values) if values else None


def hold_audit(rows):
    groups = {}
    previous = None
    for row in rows:
        h = row.get("servo_reach_limiter", {}).get("constraint_hold", {})
        start = h.get("started_at_s")
        if start is not None:
            g = groups.setdefault(
                str(start),
                {
                    "onset_s": start,
                    "last_s": None,
                    "maximum_elapsed_s": 0,
                    "held_rows": 0,
                    "timed_out": False,
                    "activated_plans": [],
                    "clear_s": None,
                    "clear_progress_rad": None,
                },
            )
            g["last_s"] = row["t"]
            g["maximum_elapsed_s"] = max(g["maximum_elapsed_s"], h["elapsed_s"])
            g["held_rows"] += bool(h.get("held"))
            g["timed_out"] |= bool(h["timed_out"])
            if row.get("diagnostics", {}).get("plan_activated"):
                g["activated_plans"].append(
                    {"t": row["t"], "id": row.get("active_replan_id")}
                )
        if previous is not None and start is None and not h.get("active"):
            groups[previous]["clear_s"] = row["t"]
            groups[previous]["clear_progress_rad"] = h.get("progress_rad")
        previous = str(start) if start is not None else None
    duration = sum(
        (b["t"] - a["t"])
        for a, b in zip(rows, rows[1:])
        if a.get("servo_reach_limiter", {}).get("constraint_hold", {}).get("held")
    )
    return {
        "episodes": list(groups.values()),
        "accepted_held_rows": sum(g["held_rows"] for g in groups.values()),
        "accepted_held_duration_s": duration,
        "notes": "Accepted verified holds differ from rejected/stale hold_duration_s in primary metrics; duration uses left-held execution intervals.",
    }


complete = read(R / "paired/complete.json")
assert complete["status"] == "six_valid_trials_completed" and complete["trials"] == 6
spec = importlib.util.spec_from_file_location(
    "v8launcher_audit", R / "launch_paired.py"
)
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)
binding = read(R / "bindings.frozen.json")
verified = launcher.verify_files(binding)
result = {
    "status": "passed",
    "at_utc": datetime.now(timezone.utc).isoformat(),
    "binding_sha256": sha(R / "bindings.frozen.json"),
    "verified_pinned_files": verified,
    "planned": 6,
    "valid": 0,
    "trials": [],
    "processes": [],
    "notes": [
        "Six development cases, not a reliable-winner comparison.",
        "Plain, legacy limiter, bounded hold use sequential blocks and actual shared-GPU timing.",
        "Detached launcher OS wait status is unavailable after reap; complete marker and each child controller completion are verified.",
    ],
}
for block in binding["cases"]:
    name = block["id"]
    root = R / "paired" / name
    finished = launcher.complete(root, block["campaign_sha256"])
    assert finished["status"] == "complete_valid"
    control = read(R / "paired" / f"{name}_launch.json")
    result["processes"].append(
        {
            "role": name + "_controller",
            "pid": control["pid"],
            "present": Path(f"/proc/{control['pid']}").exists(),
            "exit_code": 0,
            "exit_code_evidence": "launcher complete marker is only reached after process.wait()==0 for this block",
        }
    )
    summary = read(root / "analysis/summary.json")
    for index in summary["trials"]:
        record = read(Path(index["score_path"]))
        m = record["metrics"]
        assert m["valid_for_scoring"]
        folder = Path(record["directory"])
        run = read(folder / "run.json")
        execution = [
            json.loads(line) for line in (folder / "execution_trace.jsonl").open()
        ]
        plans = read(folder / "planner_trace.json")
        stopped = next((x for x in execution if x.get("stopped")), None)
        completed = next(
            (x for x in execution if x.get("diagnostics", {}).get("completed_reason")),
            None,
        )
        entry = {
            "condition": name,
            "seed": record["sampling_seed"],
            "valid_for_scoring": True,
            "primary_score_sha256": sha(Path(index["score_path"])),
            "duration_s": run["duration_s"],
            "outcomes": m["outcomes"],
            "event_times_s": m["event_times_s"],
            "object": m["object"],
            "control": m["control"],
            "first_stop_t": stopped["t"] if stopped else None,
            "first_stop_reason": stopped.get("stop_reason") if stopped else None,
            "first_stop_events": stopped.get("diagnostics", {}).get("safety_events", [])
            if stopped
            else [],
            "completion_reason": completed["diagnostics"]["completed_reason"]
            if completed
            else None,
            "completion_t": completed["t"] if completed else None,
            "hold_audit": hold_audit(execution),
            "latency_means_s": {
                k: mean(
                    [
                        p["diagnostics"][field]
                        for p in plans
                        if field in p.get("diagnostics", {})
                    ]
                )
                for k, field in [
                    ("native", "sim_native_inference_latency_s"),
                    ("full_client_rpc", "sim_policy_replan_wall_time_s"),
                    ("effective_delivery", "sim_effective_delivery_delay_s"),
                ]
            },
        }
        result["trials"].append(entry)
        result["valid"] += 1
    for p in root.glob("servers/**/server.json"):
        s = read(p)
        result["processes"].append(
            {
                "role": name + "_server",
                "pid": s["pid"],
                "present": Path(f"/proc/{s['pid']}").exists(),
                "recorded_status": s["status"],
            }
        )
launch = read(R / "launcher_record.json")
result["processes"].append(
    {
        "role": "launcher",
        "pid": launch["pid"],
        "present": Path(f"/proc/{launch['pid']}").exists(),
        "exit_code": None,
        "completion_marker": complete,
    }
)
assert all(not p["present"] for p in result["processes"])
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 7799))
result["owned_port_7799_free"] = True
assert result["valid"] == 6
(R / "completion_audit.json").write_text(
    json.dumps(result, indent=2, allow_nan=False) + "\n"
)
print(
    json.dumps(
        {
            "status": result["status"],
            "valid": 6,
            "verified": verified,
            "port_free": True,
            "processes": result["processes"],
            "trials": [
                {
                    "condition": x["condition"],
                    "seed": x["seed"],
                    "stop": x["first_stop_events"] or x["first_stop_reason"],
                    "accepted_held_rows": x["hold_audit"]["accepted_held_rows"],
                    "held_duration_s": x["hold_audit"]["accepted_held_duration_s"],
                }
                for x in result["trials"]
            ],
        },
        indent=2,
    )
)
