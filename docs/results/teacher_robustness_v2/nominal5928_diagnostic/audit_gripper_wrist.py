#!/usr/bin/env python3
"""Independent CPU validation of cached contact-point wrist signals."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
FRESH = BASE / "runs/teacher_v2_preflights/first_seed903102_gripper_wrist"
ORIGINAL = (
    BASE
    / "runs/teacher_robustness_v2/screen/rollouts/fta1500_nfe1_k4__start_1787395928__seed903102"
)
OBSERVER = BASE / "runs/teacher_v2_preflights/first_seed903102_full_robot_contacts"
SOURCE = BASE / "source_teacher_v2_delivery"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert "PHANTOM_COMPLETE" in FRESH.with_suffix(".log").read_text()
    arrays, original = (
        np.load(FRESH / "sim_trace.npz"),
        np.load(ORIGINAL / "sim_trace.npz"),
    )
    comparisons = {
        key: bool(np.array_equal(arrays[key], original[key], equal_nan=True))
        for key in original.files
    }
    rows = [
        json.loads(line)
        for line in (FRESH / "wrist_contact_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    observer = json.loads(
        (OBSERVER / "robot_environment_contact_trace.json").read_text()
    )
    observer_times = {round(r["t"], 8): r for r in observer}
    initial = json.loads((FRESH / "initialization.json").read_text())
    bias = np.array(initial["policy_initial_state"]["wrist_ft"])
    t = np.array([r["t"] for r in rows])
    error = {
        "impulse_conversion_n": 0.0,
        "per_actor_force_n": 0.0,
        "per_actor_moment_nm": 0.0,
        "total_contact_wrench": 0.0,
        "bias_addition": 0.0,
        "same_time_observer_force_n": 0.0,
    }
    scopes = set()
    observer_matches = 0
    per_sample = []
    for row in rows:
        assert np.array_equal(row["recorded_bias"], bias)
        assert row["physics_dt_s"] == 0.004 and row["api_dt_argument"] == 1.0
        assert len(row["per_actor"]) == 3
        assert {a["actor_path"].rsplit("/", 1)[-1] for a in row["per_actor"]} == {
            "gripper_housing",
            "left_pad",
            "right_pad",
        }
        torque_origin = np.array(row["tcp_pose"][:3])
        total = np.zeros(6)
        for actor in row["per_actor"]:
            scopes.add(actor["actor_path"])
            value = np.zeros(6)
            for contact in actor["contacts"]:
                assert contact["actor_path"] == actor["actor_path"]
                assert not contact["filter_path"].startswith("/World/Robot")
                scalar = (
                    np.array(contact["normal_impulse_signed_ns"]) / row["physics_dt_s"]
                )
                error["impulse_conversion_n"] = max(
                    error["impulse_conversion_n"],
                    float(abs(scalar - contact["normal_force_signed_n"]).max()),
                )
                xyz = np.array(contact["points_world_m"])
                direction = np.array(contact["normals_world"])
                forces = scalar[:, None] * direction
                value[:3] += forces.sum(axis=0)
                value[3:] += np.cross(xyz - torque_origin, forces).sum(axis=0)
            logged = np.array(actor["normal_wrench_world"])
            error["per_actor_force_n"] = max(
                error["per_actor_force_n"], float(abs(value[:3] - logged[:3]).max())
            )
            error["per_actor_moment_nm"] = max(
                error["per_actor_moment_nm"], float(abs(value[3:] - logged[3:]).max())
            )
            total += value
        error["total_contact_wrench"] = max(
            error["total_contact_wrench"],
            float(abs(total - row["normal_wrench_world"]).max()),
        )
        error["bias_addition"] = max(
            error["bias_addition"], float(abs(bias + total - row["wrist_ft"]).max())
        )
        if round(row["t"], 8) in observer_times:
            obs = observer_times[round(row["t"], 8)]
            by_path = {a["actor_path"]: a for a in obs["per_actor"]}
            for actor in row["per_actor"]:
                got = sum(
                    abs(np.array(c["normal_force_signed_n"])).sum()
                    for c in actor["contacts"]
                )
                expected = by_path[actor["actor_path"]]["normal_force_magnitude_n"]
                error["same_time_observer_force_n"] = max(
                    error["same_time_observer_force_n"], abs(got - expected)
                )
            observer_matches += 1
        per_sample.append(
            {
                "t_s": row["t"],
                "normal_wrench_world": row["normal_wrench_world"],
                "wrist_ft": row["wrist_ft"],
            }
        )
    expected_times = np.arange(len(rows)) * 0.008
    grid_error = float(abs(t - expected_times).max())
    values = np.array([r["wrist_ft"] for r in rows])
    contact_wrench = np.array([r["normal_wrench_world"] for r in rows])
    cache_indices = np.searchsorted(t, arrays["t"] + 1e-10, side="right") - 1
    expected_cache_t = t[cache_indices]
    cache_error = float(abs(arrays["wrist_ft"] - values[cache_indices]).max())
    cache_t_error = float(abs(arrays["wrist_capture_t"] - expected_cache_t).max())
    force_deviation = np.linalg.norm(values[:, :3] - bias[:3], axis=1)
    first_threshold_i = int(np.flatnonzero(force_deviation > 60)[0])
    peak_i = int(np.argmax(force_deviation))
    example = rows[peak_i]
    original_exec = [
        json.loads(line)
        for line in (ORIGINAL / "execution_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    first_block = next(
        e["t"]
        for e in original_exec
        if not e["stopped"]
        and e.get("accepted_tcp") is not None
        and np.linalg.norm(np.array(e["accepted_tcp"][:3]) - e["measured_tcp"][:3])
        > 0.02
    )
    out = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_directory": str(FRESH),
        "source_directory": str(SOURCE),
        "source_sha256": {
            name: sha(SOURCE / name)
            for name in [
                "tools/sim/gripper_wrist.py",
                "tools/sim/robot_environment_contacts.py",
                "tools/sim/run_waffles.py",
            ]
        },
        "input_sha256": {
            label: {
                p.name: sha(p)
                for p in folder.iterdir()
                if p.name
                in [
                    "sim_trace.npz",
                    "wrist_contact_trace.jsonl",
                    "robot_environment_contact_trace.json",
                    "initialization.json",
                    "effective_config.json",
                    "command_replay.json",
                    "run.json",
                    "execution_trace.jsonl",
                ]
            }
            for label, folder in [
                ("fresh", FRESH),
                ("original", ORIGINAL),
                ("observer", OBSERVER),
            ]
        },
        "physics_arrays_identical": comparisons,
        "all_original_21_arrays_identical": all(comparisons.values()),
        "additional_arrays": sorted(set(arrays.files) - set(original.files)),
        "scene_configuration_identical": json.loads(
            (FRESH / "effective_config.json").read_text()
        )
        == json.loads((ORIGINAL / "effective_config.json").read_text()),
        "sampling": {
            "rows": len(rows),
            "first_t_s": float(t[0]),
            "last_t_s": float(t[-1]),
            "expected_hz": 125,
            "maximum_grid_error_s": grid_error,
            "maximum_scene_cache_value_error": cache_error,
            "maximum_scene_cache_t_error_s": cache_t_error,
            "maximum_cache_age_s": float((arrays["t"] - expected_cache_t).max()),
            "same_time_matches_to_separate_observer": observer_matches,
        },
        "exact_initial_bias_n_nm": bias.tolist(),
        "initial_contact_wrench": rows[0]["normal_wrench_world"],
        "initial_wrist_equals_bias_exactly": bool(np.array_equal(values[0], bias)),
        "actor_scope": sorted(scopes),
        "maximum_arithmetic_error": error,
        "first_force_deviation_gt60n": rows[first_threshold_i],
        "peak_force_deviation": example,
        "first_force_deviation_gt60n_t_s": float(t[first_threshold_i]),
        "first_tracking_error_gt20mm_t_s": first_block,
        "first_threshold_precedes_tracking_error": bool(
            t[first_threshold_i] < first_block
        ),
        "peak_force_deviation_n": float(force_deviation[peak_i]),
        "peak_contact_torque_norm_nm": float(
            np.linalg.norm(contact_wrench[:, 3:], axis=1).max()
        ),
        "normal_force_sign_evidence": "At top-of-bin housing contact, raw API impulse is positive and world normal is +Z; reconstructed housing force is upward, opposing descent. Scalars and normals are never abs-flipped for wrench aggregation; magnitudes remain separate diagnostic accounting.",
        "samples": per_sample,
        "limitations": [
            "Idealized normal-contact wrench about measured TCP in world/base axes, not calibrated current-based UR3 sensor response.",
            "Tangential/friction forces, gravity, inertia, self contact and proximal arm contact are deliberately omitted.",
            "Initial-bias threshold crossing is a signal-availability check; command replay does not execute the policy safety guard or establish the actual stop time under the changed sensor.",
            "Recorded first-case outcomes remain unchanged and belong to the halted screen.",
        ],
    }
    out["audit_pass"] = (
        out["all_original_21_arrays_identical"]
        and max(error.values()) < 1e-8
        and max(grid_error, cache_error, cache_t_error) < 1e-8
        and out["initial_wrist_equals_bias_exactly"]
        and out["first_threshold_precedes_tracking_error"]
    )
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
