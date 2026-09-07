#!/usr/bin/env python3
"""Read-only scalar trace for the start6273 v5_6 native baseline diagnostic."""

import json

import audit_first_start as audit
import numpy as np
import yaml


def main():
    case = "v5_6_nfe1_k4__start_1787396273__seed903101"
    root = audit.ROOT / "rollouts" / case
    execution = audit.read_jsonl(root / "execution_trace.jsonl")
    wrist = {
        round(r["t"], 9): r
        for r in audit.read_jsonl(root / "wrist_contact_trace.jsonl")
    }
    hardware = yaml.safe_load(
        (
            audit.BASE / "runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml"
        ).read_text()
    )
    safety = hardware["safety"]
    tau = safety["wrench_baseline_tau_s"]
    debounce = safety["wrench_debounce_ticks"] / hardware["control"]["action_rate_hz"]
    series = {
        k: []
        for k in [
            "t_s",
            "wrist_fz_n",
            "rolling_baseline_fz_n",
            "force_deviation_n",
            "torque_deviation_nm",
            "contact_fz_n",
            "packet_normal_sum_n",
            "mat_normal_sum_n",
            "bench_normal_sum_n",
            "other_normal_sum_n",
            "baseline_update",
            "over_limit",
        ]
    }
    baseline, last, over_since, trigger = None, None, None, None
    for row in execution:
        t = row["t"]
        ft = np.asarray(row["measured_wrist_ft"])
        sample = wrist[round(row["wrist_capture_t"], 9)]
        assert np.array_equal(ft, sample["wrist_ft"])
        if baseline is None:
            baseline = ft.copy()
        dev = ft - baseline
        fm, tm = float(np.linalg.norm(dev[:3])), float(np.linalg.norm(dev[3:]))
        over = fm > safety["wrench_limit_N"] or tm > safety["wrench_limit_Nm"]
        dt = 0 if last is None else t - last
        last = t
        if over:
            if over_since is None:
                over_since = t
        else:
            over_since = None
        totals = dict.fromkeys(["packet", "mat", "bench", "other"], 0.0)
        for actor in sample["per_actor"]:
            for contact in actor["contacts"]:
                key = {
                    "/World/Waffle": "packet",
                    "/World/Mat/Base": "mat",
                    "/World/Bench/Slab": "bench",
                }.get(contact["filter_path"], "other")
                totals[key] += contact["normal_force_magnitude_n"]
        values = [
            t,
            float(ft[2]),
            float(baseline[2]),
            fm,
            tm,
            sample["normal_wrench_world"][2],
            totals["packet"],
            totals["mat"],
            totals["bench"],
            totals["other"],
            bool(not over and dt > 0),
            over,
        ]
        for key, value in zip(series, values):
            series[key].append(value)
        if over and t - over_since >= debounce and trigger is None:
            trigger = {
                "t_s": t,
                "over_since_s": over_since,
                "baseline_n_nm": baseline.tolist(),
                "actual_wrist_n_nm": ft.tolist(),
                "deviation_n_nm": dev.tolist(),
                "positive_contact_normal_sum_n": sum(totals.values()),
                "events": row["diagnostics"]["safety_events"],
            }
        if not over and dt > 0:
            baseline += min(1.0, dt / tau) * dev
        if row["stopped"]:
            assert (
                trigger and trigger["t_s"] == t and "wrench_limit" in trigger["events"]
            )
            break
    print(
        json.dumps(
            {
                "case_id": case,
                "source_directory": str(root),
                "input_sha256": {
                    name: audit.sha(root / name)
                    for name in [
                        "execution_trace.jsonl",
                        "wrist_contact_trace.jsonl",
                        "run.json",
                    ]
                },
                "safety_source_sha256": audit.sha(
                    audit.SOURCE / "phantom/deploy/safety.py"
                ),
                "recorded_initial_bias_n_nm": next(iter(wrist.values()))[
                    "recorded_bias"
                ],
                "force_limit_n": safety["wrench_limit_N"],
                "torque_limit_nm": safety["wrench_limit_Nm"],
                "tau_s": tau,
                "debounce_s": debounce,
                "trigger": trigger,
                "series": series,
                "interpretation": "Native calm-only EMA reproduced using actual measured wrist samples. Gradual contact can be incorporated while deviation is below limits; unloading can then exceed the learned baseline. This is the unchanged native rule, not a proposed modified guard or proof of hardware calibration.",
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
