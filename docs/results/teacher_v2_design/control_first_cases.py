#!/usr/bin/env python3
"""Read-only first-case veto audit; emit JSON, never infer or advance physics."""

import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
SOURCE = ROOT / "source_teacher_v2"
BASE = ROOT / "runs/teacher_robustness_v2/screen/rollouts"
sys.path.insert(0, str(SOURCE))

from phantom.config.hardware import HardwareConfig
from phantom.deploy.release_controller import original_policy_grip
from tools.sim.deployment_filters import TerminalVetoFilter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    text = path.read_text()
    return (
        [json.loads(line) for line in text.splitlines()]
        if path.suffix == ".jsonl"
        else json.loads(text)
    )


class Ring:
    def __init__(self, ts, wrench, clock):
        self.ts, self.wrench, self.clock = ts, wrench, clock

    def latest(self, n):
        end = int(np.searchsorted(self.ts, self.clock[0] + 1e-9, side="right"))
        return self.ts[max(0, end - n) : end], {
            "wrench": self.wrench[max(0, end - n) : end]
        }


def audit(seed):
    path = BASE / f"fta1500_nfe1_k4__start_1787395928__seed{seed}"
    raw = read(path / "planner_trace.json")
    delivered = read(path / "delivered_plans.jsonl")
    execution = read(path / "execution_trace.jsonl")
    info = read(path / "policy_info.json")
    run = read(path / "run.json")
    config = read(path / "effective_config.json")
    hw = HardwareConfig.model_validate(info["hardware_effective"])
    et = np.array([r["t"] for r in execution])
    history = []
    for row in execution:
        if not row["stopped"] and (
            not history or row["gripper_command"] != history[-1][1]
        ):
            history.append((row["t"], row["gripper_command"]))
    with np.load(path / "policy_tactile.npz", allow_pickle=False) as tac:
        tt, normal = np.asarray(tac["t"]), np.asarray(tac["gel_normal_force"])
    with np.load(
        info["tactile_baseline_provenance"]["path"], allow_pickle=False
    ) as baseline:
        baseline_wrench = np.stack(
            [baseline[f"{side}_wrench"] for side in ("left", "right")]
        )
    phases = Counter(
        row["diagnostics"]["placement_release"]["phase"] for row in execution
    )
    assert set(phases) == {"unarmed"}, (
        "release-mask restoration needs separate reconstruction"
    )
    filters = {}
    clock = [0.0]
    for mode in ("captured", "delivery"):
        rings = {}
        for i, side in enumerate(("left", "right")):
            wrench = np.tile(baseline_wrench[i], (len(tt), 1))
            wrench[:, 2] -= normal[:, i]
            rings[f"tactile_{side}"] = Ring(tt, wrench, clock)
        feedback = SimpleNamespace(
            rings=rings,
            _last_observe_t=0.0,
            entered_grip_after=lambda after: [
                (t, g) for t, g in history if after < t < clock[0]
            ],
            clear_grip_latch=lambda: None,
            request_stop=lambda why: None,
            placement_release_opening_mask=lambda values: np.zeros(len(values), bool),
        )
        filters[mode] = (
            TerminalVetoFilter(
                hw, info["terminal_veto"]["config"], implementation="live"
            ),
            feedback,
        )
    plans = []
    original_rows = Counter()
    for item in delivered:
        proposal = next(
            r for r in raw if abs(r["t"] - item["captured_snapshot_t"]) < 1e-8
        )
        n = proposal["replan_id"]
        a, b = np.asarray(proposal["actions"]), np.asarray(item["actions"])
        i = int(np.searchsorted(et, item["delivery_t"]))
        current = execution[i]
        assert abs(current["t"] - item["delivery_t"]) < 1e-8
        with np.load(path / "observations" / f"{n:04d}.npz", allow_pickle=False) as obs:
            ur = np.array(obs["ur_state"], copy=True)
        outputs = {}
        for mode, (filt, feedback) in filters.items():
            u = ur.copy()
            if mode == "delivery":
                u[12:18] = current["measured_tcp"]
                u[-2:] = current["measured_gripper"]
            snapshot = SimpleNamespace(t=proposal["t"], ur_state=u)
            clock[0] = feedback._last_observe_t = item["delivery_t"]
            plan = SimpleNamespace(
                actions=a.copy(),
                p_evt=np.asarray(proposal["p_evt"]),
                cpk=None,
                diag=deepcopy(proposal["diagnostics"]),
            )
            result = filt(plan, snapshot, feedback)
            outputs[mode] = (result.actions.copy(), result.diag["terminal_veto"])
        assert np.array_equal(outputs["captured"][0], b), (
            seed,
            n,
            "captured replay mismatch",
        )
        assert (
            outputs["captured"][1]["action"]
            == item["diagnostics"]["terminal_veto"]["action"]
        )
        entered = [
            r for r in execution if r["active_replan_id"] == n and not r["stopped"]
        ]
        plan = SimpleNamespace(actions=b, diag=item["diagnostics"])
        for row in entered:
            original_rows[
                str(
                    original_policy_grip(
                        plan, row["diagnostics"]["play_time_s"], 10, 10
                    )
                )
            ] += 1
        plans.append(
            {
                "id": n,
                "request_t": proposal["t"],
                "delivery_t": item["delivery_t"],
                "native_latency_s": proposal["latency_s"],
                "rpc_wall_s": proposal["inference_wall_time_s"],
                "request_tcp": ur[12:18].tolist(),
                "delivery_tcp": current["measured_tcp"],
                "request_g": float(ur[-2]),
                "delivery_g": current["measured_gripper"][0],
                "raw_grip_min_max": [float(a[:, 6].min()), float(a[:, 6].max())],
                "raw_head10_delta_xyz_mm": (a[:10, :3].sum(0) * 1000).tolist(),
                "actual_veto": item["diagnostics"]["terminal_veto"],
                "raw_to_delivered_max_abs_by_channel": np.max(
                    abs(a - b), axis=0
                ).tolist(),
                "captured_counterfactual_reproduces_logged_actions": True,
                "delivery_feedback_counterfactual_veto": outputs["delivery"][1],
                "delivery_feedback_counterfactual_max_abs_action_change": float(
                    np.max(abs(outputs["delivery"][0] - b))
                ),
                "played_time_max_s": max(
                    [r["diagnostics"]["play_time_s"] for r in entered], default=None
                ),
                "sent_grip_min_max": [
                    min([r["gripper_command"] for r in entered], default=None),
                    max([r["gripper_command"] for r in entered], default=None),
                ],
            }
        )
    stops = [r for r in execution if r["stopped"]]
    additional = [
        r
        for r in execution
        if r["gripper_command"] > execution[0]["measured_gripper"][0] + 2 / 255
        and not r["stopped"]
    ]
    return {
        "seed": seed,
        "path": str(path),
        "source_hashes": {
            name: sha(path / name)
            for name in (
                "planner_trace.json",
                "delivered_plans.jsonl",
                "execution_trace.jsonl",
                "run.json",
                "policy_info.json",
                "effective_config.json",
                "policy_tactile.npz",
            )
        },
        "veto_settings": info["terminal_veto"],
        "scene_geometry": {
            k: v
            for k, v in config.items()
            if k in ("table", "waffle", "gripper", "robot")
        },
        "request_count": len(raw),
        "delivered_count": len(delivered),
        "actual_veto_counts": dict(Counter(p["actual_veto"]["action"] for p in plans)),
        "actual_rewritten_plan_count": sum(
            any(v > 0 for v in p["raw_to_delivered_max_abs_by_channel"]) for p in plans
        ),
        "counterfactual_rewritten_plan_count": sum(
            p["delivery_feedback_counterfactual_max_abs_action_change"] > 0
            for p in plans
        ),
        "counterfactual_decision_difference_ids": [
            p["id"]
            for p in plans
            if p["actual_veto"]["action"]
            != p["delivery_feedback_counterfactual_veto"]["action"]
        ],
        "max_request_delivery_z_difference_mm": max(
            abs(p["request_tcp"][2] - p["delivery_tcp"][2]) * 1000 for p in plans
        ),
        "max_request_delivery_grip_difference": max(
            abs(p["request_g"] - p["delivery_g"]) for p in plans
        ),
        "max_gel_normal_force_n_proxy": normal.max(axis=0).tolist(),
        "release_phases": dict(phases),
        "original_policy_grip_reconstructed_active_rows": dict(original_rows),
        "first_additional_close_row": additional[0] if additional else None,
        "stop_row": stops[0] if stops else None,
        "max_requested_measured_tcp_translation_gap_mm": max(
            float(
                np.linalg.norm(np.array(r["requested_tcp"][:3]) - r["measured_tcp"][:3])
                * 1000
            )
            for r in execution
        ),
        "final_tcp": execution[-1]["measured_tcp"],
        "run_duration_s": run["duration_s"],
        "plans": plans,
    }


result = {
    "scope": "Read-only logged-trajectory controller audit; no GPU inference or physics. Counterfactual keeps recorded observations/actions/accepted history fixed, so it is not a counterfactual closed-loop outcome.",
    "frozen_source": str(SOURCE),
    "frozen_filter_sha256": sha(SOURCE / "tools/sim/deployment_filters.py"),
    "frozen_native_planner_sha256": sha(SOURCE / "phantom/deploy/planner.py"),
    "cases": [audit(seed) for seed in (903101, 903102)],
}
print(json.dumps(result, indent=2, allow_nan=False))
