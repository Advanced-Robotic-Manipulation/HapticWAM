import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v3/reproduction"
OLD = (
    BASE
    / "runs/teacher_pick_place_v1/campaign/rollouts/teacher__placement_xm10_ym10mm__seed4242"
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def readrows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def array_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    out = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "array_bytes_sha256_a": hashlib.sha256(
            np.ascontiguousarray(a).tobytes()
        ).hexdigest(),
        "array_bytes_sha256_b": hashlib.sha256(
            np.ascontiguousarray(b).tobytes()
        ).hexdigest(),
    }
    if a.shape != b.shape:
        return out
    equal = (
        (a == b) | (np.isnan(a) & np.isnan(b))
        if np.issubdtype(a.dtype, np.floating)
        else a == b
    )
    delta = a.astype(float) - b.astype(float)
    finite = np.isfinite(delta)
    out.update(
        bitwise_equal=a.dtype == b.dtype and a.tobytes() == b.tobytes(),
        equal_values_including_nan=bool(equal.all()),
        changed_values=int((~equal).sum()),
        max_abs_delta=float(np.max(np.abs(delta[finite]))) if finite.any() else None,
        rms_delta=float(np.sqrt(np.mean(delta[finite] ** 2))) if finite.any() else None,
    )
    return out


def data(p):
    return {
        "path": p,
        "config": json.loads((p / "effective_config.json").read_text()),
        "init": json.loads((p / "initialization.json").read_text()),
        "info": json.loads((p / "policy_info.json").read_text()),
        "exe": readrows(p / "execution_trace.jsonl"),
        "plans": json.loads((p / "planner_trace.json").read_text()),
        "delivered": readrows(p / "delivered_plans.jsonl"),
        "trace": np.load(p / "sim_trace.npz"),
    }


a = data(OLD)
reports = []
for name in ["teacher__anchor_repeat1__seed4242", "teacher__anchor_repeat2__seed4242"]:
    p = ROOT / "rollouts" / name
    status = json.loads((p / "run_status.json").read_text())
    assert status["status"] == "completed" and status["exit_code"] == 0
    b = data(p)
    result = {
        "case_id": name,
        "reference": str(OLD),
        "new_directory": str(p),
        "effective_config_equal": a["config"] == b["config"],
        "effective_config_hash_equal": sha(OLD / "effective_config.json")
        == sha(p / "effective_config.json"),
        "initialization_differences": {},
        "observations": [],
        "first_five_plans": [],
        "trace_common_time_comparison": {},
        "first_safety_stop": next((x for x in b["exe"] if x["stopped"]), None),
    }
    for key in [
        "configured_robot_q",
        "settled_robot_q",
        "settled_robot_qd",
        "configured_packet_center_m",
        "settled_packet_center_m",
        "packet_velocity_m_s",
    ]:
        result["initialization_differences"][key] = array_diff(
            a["init"][key], b["init"][key]
        )
    for i in range(3):
        oa = np.load(OLD / "observations" / f"{i:04d}.npz")
        ob = np.load(p / "observations" / f"{i:04d}.npz")
        result["observations"].append(
            {
                "index": i,
                "file_sha256_old": sha(OLD / "observations" / f"{i:04d}.npz"),
                "file_sha256_new": sha(p / "observations" / f"{i:04d}.npz"),
                "time_old_s": float(oa["t"]),
                "time_new_s": float(ob["t"]),
                "array_comparison": {
                    key: array_diff(oa[key], ob[key]) for key in oa.files
                },
                "old_metadata": json.loads(
                    (OLD / "observations" / f"{i:04d}.json").read_text()
                ),
                "new_metadata": json.loads(
                    (p / "observations" / f"{i:04d}.json").read_text()
                ),
            }
        )
    for i in range(min(5, len(b["plans"]))):
        pa, pb = a["plans"][i], b["plans"][i]
        result["first_five_plans"].append(
            {
                "index": i,
                "capture_old_s": pa["t"],
                "capture_new_s": pb["t"],
                "activation_old_s": pa.get("activated_at"),
                "activation_new_s": pb.get("activated_at"),
                "native_latency_old_s": pa["latency_s"],
                "native_latency_new_s": pb["latency_s"],
                "raw_actions": array_diff(pa["actions"], pb["actions"]),
                "head_old": pa["actions"][0],
                "head_new": pb["actions"][0],
                "old_k_pick": pa["diagnostics"].get("k_pick"),
                "new_k_pick": pb["diagnostics"].get("k_pick"),
            }
        )
    ta, tb = a["trace"]["t"], b["trace"]["t"]
    ia = {round(float(t), 9): i for i, t in enumerate(ta)}
    indices = [
        (ia[round(float(t), 9)], j)
        for j, t in enumerate(tb)
        if round(float(t), 9) in ia
    ]
    aa, bb = map(np.array, zip(*indices))
    result["common_scene_rows"] = len(indices)
    for key in sorted(set(a["trace"].files) & set(b["trace"].files)):
        va, vb = a["trace"][key][aa], b["trace"][key][bb]
        if va.shape != vb.shape:
            continue
        diff = array_diff(va, vb)
        eq = (
            (va == vb) | (np.isnan(va) & np.isnan(vb))
            if np.issubdtype(va.dtype, np.floating)
            else va == vb
        )
        row_equal = eq.reshape(len(eq), -1).all(axis=1)
        first = np.flatnonzero(~row_equal)
        diff["first_unequal_value_time_s"] = (
            float(tb[bb[first[0]]]) if len(first) else None
        )
        pre = tb[bb] <= 1.0
        diff["up_to_one_second"] = array_diff(va[pre], vb[pre])
        result["trace_common_time_comparison"][key] = diff
    result["first_deliveries"] = [
        {
            "snapshot_t": x["captured_snapshot_t"],
            "delivery_t": x["delivery_t"],
            "veto": x["diagnostics"].get("terminal_veto"),
        }
        for x in b["delivered"][:3]
    ]
    score = ROOT / "analysis/trials" / f"{name}.json"
    if score.exists():
        result["score"] = {
            k: v
            for k, v in json.loads(score.read_text())["metrics"].items()
            if k
            in [
                "valid_for_scoring",
                "invalid_reasons",
                "outcomes",
                "event_times_s",
                "object",
            ]
        }
    stop = result["first_safety_stop"]
    st = stop["t"] if stop else b["exe"][-1]["t"]
    j = int(np.argmin(np.abs(tb - st)))
    result["scene_near_stop"] = {
        "t_s": float(tb[j]),
        **{
            k: b["trace"][k][j].tolist()
            for k in [
                "tcp",
                "waffle_position",
                "pad_force",
                "pad_packet_normal_force",
                "packet_robot_normal_force",
                "packet_bin_normal_force",
            ]
            if k in b["trace"].files
        },
    }
    result["input_sha256"] = {
        f: sha(p / f)
        for f in [
            "effective_config.json",
            "initialization.json",
            "policy_info.json",
            "execution_trace.jsonl",
            "planner_trace.json",
            "delivered_plans.jsonl",
            "sim_trace.npz",
        ]
    }
    reports.append(result)
# Raw first observation / first plan equality between independently reset new repeats.
b1 = data(ROOT / "rollouts/teacher__anchor_repeat1__seed4242")
b2 = data(ROOT / "rollouts/teacher__anchor_repeat2__seed4242")
n1 = np.load(b1["path"] / "observations/0000.npz")
n2 = np.load(b2["path"] / "observations/0000.npz")
repeat_pair = {
    "observations0": {k: array_diff(n1[k], n2[k]) for k in n1.files},
    "raw_plan0": array_diff(b1["plans"][0]["actions"], b2["plans"][0]["actions"]),
}
print(
    json.dumps(
        {
            "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope": "Completed seed4242 repeats only. Read-only CPU; no simulator/model calls or data/source mutation. Later raw heads are compared by ordinal with their differing capture times exposed, not asserted to be equal-input tests.",
            "cases": reports,
            "between_new_repeats": repeat_pair,
        },
        indent=2,
        allow_nan=False,
    )
)
