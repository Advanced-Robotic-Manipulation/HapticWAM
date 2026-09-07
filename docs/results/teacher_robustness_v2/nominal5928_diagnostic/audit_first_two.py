#!/usr/bin/env python3
"""Read completed first reference-teacher screen cases; no tuning or inference."""

import base64
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path("/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_v2")
sys.path.insert(0, str(SOURCE))
from phantom.sim.policy_metrics import evaluate_policy_trace

ROOT = Path(
    "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_robustness_v2/screen"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    design = json.loads((ROOT / "campaign_snapshot.json").read_text())
    progress = json.loads((ROOT / "progress.json").read_text())
    out = []
    for seed in [903101, 903102]:
        case = f"fta1500_nfe1_k4__start_1787395928__seed{seed}"
        assert progress["trials"][case]["status"] == "completed"
        assert progress["trials"][case]["exit_code"] == 0
        folder = ROOT / "rollouts" / case
        z = np.load(folder / "sim_trace.npz")
        run = json.loads((folder / "run.json").read_text())
        config = json.loads((folder / "effective_config.json").read_text())
        executions = [
            json.loads(line)
            for line in (folder / "execution_trace.jsonl").read_text().splitlines()
            if line.strip()
        ]
        plans = json.loads((folder / "planner_trace.json").read_text())
        delivered = [
            json.loads(line)
            for line in (folder / "delivered_plans.jsonl").read_text().splitlines()
            if line.strip()
        ]
        gel = json.loads((folder / "gel_contact_trace.json").read_text())
        info = json.loads((folder / "policy_info.json").read_text())
        metrics = evaluate_policy_trace(
            z,
            config,
            design["thresholds"],
            run=run,
            execution_trace=executions,
            planner_trace=plans,
        )
        peaks = {}
        for row in gel:
            for side, pad in zip(["left", "right"], row["per_pad"]):
                for record in pad["per_filter_contacts"]:
                    key = side + ":" + record["filter_path"]
                    if (
                        key not in peaks
                        or record["normal_force_magnitude_n"]
                        > peaks[key]["normal_force_n"]
                    ):
                        peaks[key] = {
                            "t_s": row["t"],
                            "side": side,
                            "normal_force_n": record["normal_force_magnitude_n"],
                            "gel_force_n": record["gel_compression_n"],
                            "body_contact": record,
                            "rejection_budgets": pad["ignored_by_reason_n"],
                        }
        snapshots = []
        for t in [
            0.004,
            1.5,
            3,
            4.5,
            5.1,
            6,
            7,
            8,
            9,
            10,
            10.916,
            12,
            20,
            26.34,
            33.38,
            54,
            59.996,
        ]:
            if t > executions[-1]["t"] + 1e-8:
                continue
            row = min(executions, key=lambda r: abs(r["t"] - t))
            snapshots.append(
                {
                    key: row.get(key)
                    for key in [
                        "t",
                        "measured_q",
                        "target_q",
                        "measured_qd",
                        "measured_tcp",
                        "requested_tcp",
                        "accepted_tcp",
                        "measured_gripper",
                        "gripper_command",
                        "measured_wrist_ft",
                        "diagnostics",
                        "active_replan_id",
                    ]
                }
            )
        active = [
            e
            for e in executions
            if not e["stopped"] and e.get("accepted_tcp") is not None
        ]
        ik_residual = np.array(
            [
                np.linalg.norm(
                    np.array(e["accepted_tcp"])[:3] - np.array(e["requested_tcp"])[:3]
                )
                for e in active
            ]
        )
        track_residual = np.array(
            [
                np.linalg.norm(
                    np.array(e["accepted_tcp"])[:3] - np.array(e["measured_tcp"])[:3]
                )
                for e in active
            ]
        )
        measured_wrist = np.array([e["measured_wrist_ft"] for e in executions])
        initial_wrist = np.array(info["initial_recorded_wrist_bias"])
        first_stop = next((e for e in executions if e["stopped"]), None)
        video = cv2.VideoCapture(str(folder / "sim.mp4"))
        frames = []
        times = [0, 5.1, 10.916, 54] if seed == 903101 else [0, 4.5, 6, 10.02]
        for requested in times:
            i = int(np.argmin(abs(z["t"] - requested)))
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = video.read()
            if not ok:
                raise RuntimeError("Could not decode completed review frame")
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            assert ok
            frames.append(
                {
                    "requested_t_s": requested,
                    "actual_t_s": float(z["t"][i]),
                    "frame_index": i,
                    "jpeg_sha256": hashlib.sha256(encoded).hexdigest(),
                    "jpeg_base64": base64.b64encode(encoded).decode(),
                }
            )
        video.release()
        out.append(
            {
                "case_id": case,
                "directory": str(folder),
                "input_sha256": {
                    p.name: sha(p)
                    for p in folder.iterdir()
                    if p.name
                    in [
                        "run.json",
                        "sim_trace.npz",
                        "execution_trace.jsonl",
                        "planner_trace.json",
                        "delivered_plans.jsonl",
                        "gel_contact_trace.json",
                        "effective_config.json",
                        "policy_info.json",
                        "sim.mp4",
                    ]
                },
                "metrics": metrics,
                "per_body_peak_contacts": peaks,
                "execution_snapshots": snapshots,
                "first_safety_stop": first_stop,
                "largest_accepted_to_requested_position_difference_m": float(
                    ik_residual.max()
                ),
                "largest_accepted_to_measured_position_difference_m": float(
                    track_residual.max()
                ),
                "first_accepted_to_measured_error_gt_20mm_s": next(
                    (e["t"] for e, err in zip(active, track_residual) if err > 0.02),
                    None,
                ),
                "mean_initial_wrist_bias_n_nm": initial_wrist.tolist(),
                "initial_wrist_force_norm_n": float(np.linalg.norm(initial_wrist[:3])),
                "maximum_raw_wrist_force_norm_n": float(
                    np.linalg.norm(measured_wrist[:, :3], axis=1).max()
                ),
                "maximum_wrist_force_change_from_initial_n": float(
                    np.linalg.norm(
                        measured_wrist[:, :3] - initial_wrist[:3], axis=1
                    ).max()
                ),
                "maximum_gel_force_per_pad_n": np.max(
                    [r["normal_force_n"] for r in gel], axis=0
                ).tolist(),
                "latch_observed": any(
                    e["diagnostics"].get("grip_latch") is not None for e in executions
                ),
                "veto_action_counts": dict(
                    Counter(
                        r["diagnostics"]
                        .get("terminal_veto", {})
                        .get("action", "missing")
                        for r in delivered
                    )
                ),
                "frames": frames,
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "Descriptive audit of the first two completed cases in the subsequently halted original screen; not a teacher comparison or a new frozen study result.",
                "source_files_sha256": {
                    name: sha(SOURCE / name)
                    for name in [
                        "phantom/deploy/safety.py",
                        "tools/sim/run_waffles.py",
                        "tools/sim/gel_contact.py",
                        "phantom/sim/policy_metrics.py",
                    ]
                },
                "campaign_snapshot_sha256": sha(ROOT / "campaign_snapshot.json"),
                "trials": out,
                "safety_baseline_semantics": "Native supervisor initializes a wrist baseline from the first sample, subtracts its calm-only EMA, and debounces limits; policy sees raw wrist input. Proxy adds whole-pad net forces and approximate moments to fixed measured bias. Other robot-body environment contacts are not included in this proxy.",
                "limits": [
                    "No new simulations, policy samples, geometry or threshold changes.",
                    "Force proxies and collision geometry remain uncalibrated. No full robot/self-contact attribution is available in these traces.",
                    "First-additional-close diagnostic uses executed closure above initial measured closure+2/255; it is not the first rise after any later opening. Absolute .25 threshold already exceeded at t0.",
                    "A safety stop is a completed policy/safety failure in this conditional simulator, not proof of real-hardware failure or an infrastructure error.",
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
