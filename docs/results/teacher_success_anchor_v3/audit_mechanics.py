import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
OLD = (
    BASE
    / "runs/teacher_pick_place_v1/campaign/rollouts/teacher__placement_xm10_ym10mm__seed4242"
)
NEW = BASE / "runs/teacher_success_anchor_v3/accepted_command_replay"
SOURCE = BASE / "source_teacher_anchor_commands_v3"
sys.path.insert(0, str(SOURCE))
from phantom.sim.command_replay import RecordedDriveCommands
from phantom.sim.policy_metrics import evaluate_policy_trace


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


a, b = np.load(OLD / "sim_trace.npz"), np.load(NEW / "sim_trace.npz")
cfg = json.loads((NEW / "effective_config.json").read_text())
run = json.loads((NEW / "run.json").read_text())
orun = json.loads((OLD / "run.json").read_text())
design = json.loads(
    (BASE / "runs/teacher_pick_place_v1/campaign/campaign_snapshot.json").read_text()
)
metrics = evaluate_policy_trace(b, cfg, design["thresholds"], run=run)
assert metrics["invalid_reasons"] == ["run_mode_is_not_policy"]
assert metrics["placement_support"]["verified_placement"]
idx = {round(float(t), 9): i for i, t in enumerate(a["t"])}
pairs = [
    (idx[round(float(t), 9)], j)
    for j, t in enumerate(b["t"])
    if round(float(t), 9) in idx
]
aa, bb = map(np.asarray, zip(*pairs))
comparison = {}
for key in sorted(set(a.files) & set(b.files)):
    x, y = a[key][aa], b[key][bb]
    delta = x.astype(float) - y.astype(float)
    finite = np.isfinite(delta)
    comparison[key] = {
        "shared_rows": len(x),
        "bitwise_equal": x.dtype == y.dtype
        and x.shape == y.shape
        and x.tobytes() == y.tobytes(),
        "equal_values_with_nan": bool(np.array_equal(x, y, equal_nan=True)),
        "max_abs_delta": float(np.max(np.abs(delta[finite]))) if finite.any() else None,
    }
commands = RecordedDriveCommands(
    OLD / "execution_trace.jsonl", finger_limit_m=cfg["gripper"]["stroke"] / 2
)
rows = [json.loads(x) for x in (OLD / "execution_trace.jsonl").read_text().splitlines()]
ct = np.array([round(r["t"] * 1e9) for r in rows])
clock = (
    np.arange(round(float(b["t"][-1]) / cfg["physics"]["dt"]) + 1)
    * cfg["physics"]["dt"]
)
target_checks = 0
for t in clock:
    eligible = np.flatnonzero(ct <= round(float(t) * 1e9))
    value = commands.at(float(t))
    if not len(eligible):
        assert value is None
        continue
    expected = rows[int(eligible[-1])]
    assert np.array_equal(value[0], expected["target_q"])
    assert np.array_equal(value[1], expected["target_finger_q"])
    target_checks += 1
trace_errors = []
for j, t in enumerate(b["t"]):
    value = commands.at(float(t))
    if value is not None:
        trace_errors.append(
            float(np.abs(b["target_q"][j] - value[0].astype(b["target_q"].dtype)).max())
        )
assert max(trace_errors) == 0
window = (b["t"] >= 25.2) & (b["t"] <= 27.2)
support = {
    k: {
        "min": float(b[k][window].min()),
        "max": float(b[k][window].max()),
        "mean": float(b[k][window].mean()),
    }
    for k in ["packet_robot_normal_force", "packet_bin_normal_force"]
}
assert run["packet_support_filter_paths"] == orun["packet_support_filter_paths"]
oldscore = json.loads(
    (
        BASE
        / "runs/teacher_pick_place_v1/campaign/analysis/trials/teacher__placement_xm10_ym10mm__seed4242.json"
    ).read_text()
)["metrics"]
assert metrics["outcomes"] == oldscore["outcomes"]
assert metrics["event_times_s"] == oldscore["event_times_s"]
out = {
    "scope": "Independent read-only CPU audit. Recorded-command mechanics reproduction only, not fresh policy success. All comparisons at exact shared simulation timestamps.",
    "original_rows": len(a["t"]),
    "replay_rows": len(b["t"]),
    "shared_time_rows": len(pairs),
    "original_last_s": float(a["t"][-1]),
    "replay_last_s": float(b["t"][-1]),
    "unmatched_original_times_s": [
        float(t)
        for t in a["t"]
        if round(float(t), 9) not in {round(float(s), 9) for s in b["t"]}
    ],
    "unmatched_replay_times_s": [
        float(t) for t in b["t"] if round(float(t), 9) not in idx
    ],
    "arrays": comparison,
    "all_shared_arrays_bitwise_equal": all(
        v["bitwise_equal"] for v in comparison.values()
    ),
    "command_alignment": {
        "source_rows": len(rows),
        "all_physics_ticks_checked": len(clock),
        "ticks_with_recorded_command": target_checks,
        "method": "Independent integer-nanosecond causal eligibility, then exact q/finger comparison with recorded reader; sampled target_q matches cast source target exactly. Before0.004s no command; after31.38s final targets held.",
        "sampled_target_q_max_delta_after_dtype_cast": max(trace_errors),
    },
    "physics_configuration_equal": cfg
    == json.loads((OLD / "effective_config.json").read_text()),
    "initialization_equal": json.loads((NEW / "initialization.json").read_text())
    == json.loads((OLD / "initialization.json").read_text()),
    "score": {
        "valid_for_policy_scoring": metrics["valid_for_scoring"],
        "invalid_reasons": metrics["invalid_reasons"],
        "outcomes": metrics["outcomes"],
        "event_times_s": metrics["event_times_s"],
        "placement_support": metrics["placement_support"],
    },
    "support_paths_identical": True,
    "robot_body_count": len(run["packet_support_filter_paths"]["robot"]),
    "bin_body_count": len(run["packet_support_filter_paths"]["bin"]),
    "settled_support_window_s": [25.2, 27.2],
    "settled_support_force_n": support,
    "packet_mg_n": cfg["waffle"]["mass"] * 9.81,
    "input_sha256": {
        "original_trace": sha(OLD / "sim_trace.npz"),
        "replayed_trace": sha(NEW / "sim_trace.npz"),
        "original_execution": sha(OLD / "execution_trace.jsonl"),
        "replay_run": sha(NEW / "run.json"),
        "replay_config": sha(NEW / "effective_config.json"),
        "command_reader_source": sha(SOURCE / "phantom/sim/command_replay.py"),
        "scorer_source": sha(SOURCE / "phantom/sim/policy_metrics.py"),
    },
}
print(json.dumps(out, indent=2, allow_nan=False))
