#!/usr/bin/env python3
"""Read-only diagnosis of the four frozen screening acquisitions; JSON stdout.

Run on compute3 with audit_first_start.py available on PYTHONPATH. No simulator,
inference, policy ranking, source modifications, or rescoring thresholds.
"""

import json
from datetime import datetime, timezone

import audit_first_start as audit
import numpy as np
import yaml


def intervals(t, mask):
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [
        {
            "start_s": float(t[a]),
            "end_s": float(t[b - 1]),
            "sampled_span_s": float(t[b - 1] - t[a]),
        }
        for a, b in zip(bounds[::2], bounds[1::2])
    ]


def causal(rows, t):
    return max((row for row in rows if row["t"] <= t + 1e-9), key=lambda row: row["t"])


def case_summary(case, design, progress, hardware):
    result = audit.summarize_case(case, design, progress, hardware)
    assert result["metrics"]["outcomes"]["acquired"], case
    folder = audit.ROOT / "rollouts" / case
    z = np.load(folder / "sim_trace.npz")
    executions = audit.read_jsonl(folder / "execution_trace.jsonl")
    gel = json.loads((folder / "gel_contact_trace.json").read_text())
    acquisition = result["metrics"]["event_times_s"]["acquisition"]
    end_t = result["execution_end_s"]
    active = [r for r in executions if not r["stopped"]]
    after = [r for r in active if r["t"] >= acquisition]
    base = causal(active, acquisition)
    measured_z = np.array([r["measured_tcp"][2] for r in after])
    target_z = np.array([r["accepted_tcp"][2] for r in after])
    commanded_closure = np.array([r["gripper_command"] for r in after])
    measured_closure = np.array([r["measured_gripper"][0] for r in after])
    ftimes = z["t"]
    before_stop = ftimes <= end_t + 1e-9
    bilateral = (
        z["pad_packet_normal_force"] > design["thresholds"]["contact_force_n"]
    ).all(axis=1)
    gel_active = [r for r in gel if r["t"] <= end_t + 1e-9]
    gel_forces = np.array([r["normal_force_n"] for r in gel_active])
    gel_times = np.array([r["t"] for r in gel_active])
    gel_during_bilateral = []
    for i in np.flatnonzero(bilateral & before_stop):
        row = causal(gel_active, float(ftimes[i]))
        gel_during_bilateral.append(
            {
                "t_s": float(ftimes[i]),
                "gel_capture_t_s": row["t"],
                "physical_pad_packet_normal_n": z["pad_packet_normal_force"][
                    i
                ].tolist(),
                "gel_normal_force_n": row["normal_force_n"],
            }
        )
    latch_rows = [
        r for r in executions if r["diagnostics"].get("grip_latch") is not None
    ]
    first_target_rise = next(
        (r for r in after if r["accepted_tcp"][2] >= base["accepted_tcp"][2] + 0.03),
        None,
    )
    first_measured_rise = next(
        (r for r in after if r["measured_tcp"][2] >= base["measured_tcp"][2] + 0.03),
        None,
    )
    checkpoints = {}
    for label, time in [
        ("acquisition", acquisition),
        ("acquisition_plus_half_second", min(acquisition + 0.5, end_t)),
        ("maximum_gel", float(gel_times[np.argmax(np.sum(gel_forces, axis=1))])),
        (
            "first_30mm_commanded_tcp_rise",
            None if first_target_rise is None else first_target_rise["t"],
        ),
        ("stop_or_horizon", end_t),
    ]:
        if time is None:
            continue
        e, g = causal(executions, time), causal(gel_active, time)
        i = max(0, int(np.searchsorted(ftimes, time, side="right") - 1))
        checkpoints[label] = {
            "t_s": time,
            "execution": {
                k: e.get(k)
                for k in [
                    "t",
                    "requested_tcp",
                    "accepted_tcp",
                    "measured_tcp",
                    "gripper_command",
                    "measured_gripper",
                    "measured_qd",
                    "diagnostics",
                ]
            },
            "scene_t_s": float(ftimes[i]),
            "packet_position_m": z["waffle_position"][i].tolist(),
            "packet_quaternion_wxyz": z["waffle_orientation_wxyz"][i].tolist(),
            "packet_normal_force_n": z["pad_packet_normal_force"][i].tolist(),
            "gel_capture_t_s": g["t"],
            "gel_force_n": g["normal_force_n"],
            "gel_per_pad": g["per_pad"],
        }
    # Plot data retain separate clocks: 15Hz scene, 8Hz gel, and125Hz executor.
    # Decimate execution display only; extrema/timing above use every record.
    display_e = active[::8]
    display_e += [active[-1]] if display_e[-1] is not active[-1] else []
    return {
        "case_id": case,
        "metrics": result["metrics"],
        "input_sha256": result["input_sha256"],
        "first_safety_stop": result["first_safety_stop"],
        "execution_end_s": end_t,
        "physical_bilateral_intervals": intervals(
            ftimes[before_stop], bilateral[before_stop]
        ),
        "gel_bilateral_above_2p5n_intervals": intervals(
            gel_times, (gel_forces > 2.5).all(axis=1)
        ),
        "gel_peak_n": gel_forces.max(axis=0).tolist(),
        "gel_during_physical_bilateral": gel_during_bilateral,
        "latch": {
            "ever_observed": bool(latch_rows),
            "first_t_s": latch_rows[0]["t"] if latch_rows else None,
            "first_value": latch_rows[0]["diagnostics"]["grip_latch"]
            if latch_rows
            else None,
            "last_value": latch_rows[-1]["diagnostics"]["grip_latch"]
            if latch_rows
            else None,
        },
        "post_acquisition_motion": {
            "reference_execution_t_s": base["t"],
            "reference_measured_tcp_z_m": base["measured_tcp"][2],
            "reference_accepted_tcp_z_m": base["accepted_tcp"][2],
            "maximum_measured_tcp_rise_m": float(
                measured_z.max() - base["measured_tcp"][2]
            ),
            "maximum_accepted_tcp_rise_m": float(
                target_z.max() - base["accepted_tcp"][2]
            ),
            "minimum_measured_tcp_delta_z_m": float(
                measured_z.min() - base["measured_tcp"][2]
            ),
            "minimum_accepted_tcp_delta_z_m": float(
                target_z.min() - base["accepted_tcp"][2]
            ),
            "first_commanded_30mm_tcp_rise_s": None
            if first_target_rise is None
            else first_target_rise["t"],
            "first_measured_30mm_tcp_rise_s": None
            if first_measured_rise is None
            else first_measured_rise["t"],
            "reference_commanded_closure": base["gripper_command"],
            "reference_measured_closure": base["measured_gripper"][0],
            "commanded_closure_min_max": [
                float(commanded_closure.min()),
                float(commanded_closure.max()),
            ],
            "measured_closure_min_max": [
                float(measured_closure.min()),
                float(measured_closure.max()),
            ],
        },
        "tracking": result["tracking"],
        "wrench_guard": result["wrench_subguard_reconstruction"],
        "contact_at_stop": result["wrist_snapshots"].get("stop"),
        "per_body_peak_contact": result["per_body_peak_contact_before_stop"],
        "checkpoints": checkpoints,
        "plot": {
            "scene_t_s": ftimes[before_stop].tolist(),
            "object_z_m": z["waffle_position"][before_stop, 2].tolist(),
            "physical_pad_packet_normal_n": z["pad_packet_normal_force"][
                before_stop
            ].tolist(),
            "gel_t_s": gel_times.tolist(),
            "gel_force_n": gel_forces.tolist(),
            "execution_t_s": [r["t"] for r in display_e],
            "accepted_tcp_z_m": [r["accepted_tcp"][2] for r in display_e],
            "measured_tcp_z_m": [r["measured_tcp"][2] for r in display_e],
            "commanded_closure": [r["gripper_command"] for r in display_e],
            "measured_closure": [r["measured_gripper"][0] for r in display_e],
            "latch": [r["diagnostics"].get("grip_latch") for r in display_e],
        },
    }


def main():
    design = json.loads((audit.ROOT / "campaign_snapshot.json").read_text())
    progress = json.loads((audit.ROOT / "progress.json").read_text())
    assert progress["status"] == "all_trials_completed"
    hardware_path = (
        audit.BASE / "runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml"
    )
    hardware = yaml.safe_load(hardware_path.read_text())
    cases = [
        "fta3000_nfe1_k4__start_1787395963__seed903101",
        "v5_6_nfe1_k4__start_1787396028__seed903101",
        "v5_6_nfe1_k4__start_1787396273__seed903101",
        "fta1500_nfe5_k1__start_1787396273__seed903102",
    ]
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Four acquisitions identified by the authoritative completed corrected-screen selector. Descriptive diagnosis only; no ranking, frozen score changes, or superseded runs.",
        "campaign_sha256": audit.sha(audit.ROOT / "campaign_snapshot.json"),
        "selector_sha256": audit.sha(audit.ROOT / "selection/selection.json"),
        "trials": [case_summary(case, design, progress, hardware) for case in cases],
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
