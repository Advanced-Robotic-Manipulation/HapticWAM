import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
old = (
    BASE
    / "runs/teacher_pick_place_v1/campaign/rollouts/teacher__placement_xm10_ym10mm__seed4242"
)
new = BASE / "runs/teacher_success_anchor_v3/reproduction/rollouts"


def read(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


a = read(old / "execution_trace.jsonl")
ma = {round(r["t"], 9): r for r in a}
pa = json.loads((old / "planner_trace.json").read_text())
out = []
for path in [
    old,
    new / "teacher__anchor_repeat1__seed4242",
    new / "teacher__anchor_repeat2__seed4242",
]:
    rows = read(path / "execution_trace.jsonl")
    plans = json.loads((path / "planner_trace.json").read_text())
    gel = json.loads((path / "gel_contact_trace.json").read_text())
    stop = next(r for r in rows if r["stopped"])
    gel = [r for r in gel if r["t"] <= stop["t"]]
    f = np.asarray([r["normal_force_n"] for r in gel])
    latch = next(
        (r for r in rows if r["diagnostics"].get("grip_latch") is not None), None
    )
    first = {}
    for key in [
        "requested_tcp",
        "target_q",
        "target_finger_q",
        "measured_q",
        "measured_qd",
        "measured_tcp",
        "measured_gripper",
    ]:
        first[key] = next(
            (
                r["t"]
                for r in rows
                if round(r["t"], 9) in ma
                and r.get(key) is not None
                and ma[round(r["t"], 9)].get(key) is not None
                and not np.array_equal(r[key], ma[round(r["t"], 9)][key])
            ),
            None,
        )
    act = np.asarray(plans[0]["actions"])
    ref = np.asarray(pa[0]["actions"])
    delta = act - ref
    out.append(
        {
            "case_id": path.name,
            "first_executor_value_difference_s": first,
            "first_head_delta_per_channel_max_abs": np.abs(delta).max(axis=0).tolist(),
            "first_head_translation_delta_max_row_norm_m": float(
                np.linalg.norm(delta[:, :3], axis=1).max()
            ),
            "first_head_rotvec_delta_max_row_norm_rad": float(
                np.linalg.norm(delta[:, 3:6], axis=1).max()
            ),
            "first_head_closure_delta_max_abs": float(np.abs(delta[:, 6]).max()),
            "first_head_translation_sum_delta_m": delta[:, :3].sum(axis=0).tolist(),
            "first_latch_s": None if latch is None else latch["t"],
            "gel_peak_per_pad_n": f.max(axis=0).tolist(),
            "maximum_concurrent_weaker_gel_pad_n": float(f.min(axis=1).max()),
            "samples_both_gel_ge_2point5": int(np.sum((f >= 2.5).all(axis=1))),
            "gel_source_sha256": hashlib.sha256(
                (path / "gel_contact_trace.json").read_bytes()
            ).hexdigest(),
            "gel_max_weaker_sample": gel[int(np.argmax(f.min(axis=1)))],
            "stop_time_s": stop["t"],
        }
    )
print(json.dumps(out, indent=2, allow_nan=False))
