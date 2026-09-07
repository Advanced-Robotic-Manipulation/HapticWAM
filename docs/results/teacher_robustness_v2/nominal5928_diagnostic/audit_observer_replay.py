#!/usr/bin/env python3
"""Verify optional observer invariance and explicit blocking-body attribution."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
FRESH = BASE / "runs/teacher_v2_preflights/first_seed903102_full_robot_contacts"
ORIGINAL = (
    BASE
    / "runs/teacher_robustness_v2/screen/rollouts/fta1500_nfe1_k4__start_1787395928__seed903102"
)
SOURCE = BASE / "source_teacher_v2_contact_diagnostic"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    log = FRESH.with_suffix(".log").read_text()
    assert "PHANTOM_COMPLETE" in log
    new, old = np.load(FRESH / "sim_trace.npz"), np.load(ORIGINAL / "sim_trace.npz")
    comparisons = {
        k: bool(np.array_equal(new[k], old[k], equal_nan=True)) for k in old.files
    }
    assert set(new.files) == set(old.files)
    rows = json.loads((FRESH / "robot_environment_contact_trace.json").read_text())
    run = json.loads((FRESH / "run.json").read_text())
    init = json.loads((FRESH / "initialization.json").read_text())
    initial_contacts = init["robot_environment_contacts"]
    paths = run["robot_environment_contact_diagnostic"]
    command = json.loads((FRESH / "command_replay.json").read_text())
    assert command["sha256"] == sha(ORIGINAL / "execution_trace.jsonl")
    assert len(paths["robot_paths"]) == 13 and len(paths["environment_paths"]) == 8
    assert np.array_equal([r["t"] for r in rows], new["t"])
    peaks, series, first_loaded, deepest_loaded, deepest_unloaded = (
        {},
        [],
        {},
        None,
        None,
    )
    max_force_error, max_budget_error, max_contacts, count, unloaded_count = (
        0.0,
        0.0,
        0,
        0,
        0,
    )
    for row in rows:
        assert row["robot_paths"] == paths["robot_paths"]
        assert row["environment_paths"] == paths["environment_paths"]
        assert len(row["per_actor"]) == 13
        assert row["api_dt_argument"] == 1.0
        sample = {"t_s": row["t"], "normal_force_by_pair_n": {}}
        actors_total = 0.0
        for actor_index, actor in enumerate(row["per_actor"]):
            assert actor["actor_path"] == paths["robot_paths"][actor_index]
            assert actor["sensor_rows"] == 1 and actor["filter_columns"] == 8
            max_contacts = max(max_contacts, actor["populated_unique_contact_count"])
            actors_total += actor["normal_force_magnitude_n"]
            total = 0.0
            for contact in actor["contacts"]:
                assert contact["actor_path"] == actor["actor_path"]
                assert (
                    paths["environment_paths"][contact["filter_column"]]
                    == contact["filter_path"]
                )
                impulses = np.array(contact["normal_impulse_signed_ns"])
                forces = np.array(contact["normal_force_signed_n"])
                count += len(forces)
                unloaded_count += int((impulses == 0).sum())
                max_force_error = max(
                    max_force_error,
                    float(abs(impulses / row["physics_dt_s"] - forces).max()),
                )
                max_budget_error = max(
                    max_budget_error,
                    abs(float(abs(forces).sum()) - contact["normal_force_magnitude_n"]),
                )
                total += contact["normal_force_magnitude_n"]
                key = actor["actor_path"] + " -> " + contact["filter_path"]
                sample["normal_force_by_pair_n"][key] = contact[
                    "normal_force_magnitude_n"
                ]
                if (
                    key not in peaks
                    or contact["normal_force_magnitude_n"]
                    > peaks[key]["normal_force_n"]
                ):
                    peaks[key] = {
                        "t_s": row["t"],
                        "normal_force_n": contact["normal_force_magnitude_n"],
                        "contact": contact,
                    }
                if (
                    contact["normal_force_magnitude_n"] > 0.1
                    and key not in first_loaded
                ):
                    first_loaded[key] = row["t"]
                for i, sep in enumerate(contact["signed_separation_m"]):
                    value = {
                        "t_s": row["t"],
                        "actor_path": actor["actor_path"],
                        "filter_path": contact["filter_path"],
                        "signed_separation_m": sep,
                        "normal_force_n": abs(float(forces[i])),
                        "point_world_m": contact["points_world_m"][i],
                    }
                    if forces[i] != 0 and (
                        deepest_loaded is None
                        or sep < deepest_loaded["signed_separation_m"]
                    ):
                        deepest_loaded = value
                    if forces[i] == 0 and (
                        deepest_unloaded is None
                        or sep < deepest_unloaded["signed_separation_m"]
                    ):
                        deepest_unloaded = value
            max_budget_error = max(
                max_budget_error, abs(total - actor["normal_force_magnitude_n"])
            )
        max_budget_error = max(
            max_budget_error, abs(actors_total - row["normal_force_magnitude_n"])
        )
        series.append(sample)
    out = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_directory": str(FRESH),
        "original_directory": str(ORIGINAL),
        "source_directory": str(SOURCE),
        "mode": run["mode"],
        "source_sha256": {
            name: sha(SOURCE / name)
            for name in [
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
                    "run.json",
                    "sim_trace.npz",
                    "robot_environment_contact_trace.json",
                    "effective_config.json",
                    "initialization.json",
                    "command_replay.json",
                    "execution_trace.jsonl",
                ]
            }
            for label, folder in [("fresh", FRESH), ("original", ORIGINAL)]
        },
        "array_comparison_bitwise": comparisons,
        "all_21_arrays_identical": all(comparisons.values()),
        "frames": len(rows),
        "first_t_s": rows[0]["t"],
        "last_t_s": rows[-1]["t"],
        "scene_configuration_identical": json.loads(
            (FRESH / "effective_config.json").read_text()
        )
        == json.loads((ORIGINAL / "effective_config.json").read_text()),
        "initial_total_robot_environment_normal_force_n": initial_contacts[
            "normal_force_magnitude_n"
        ],
        "robot_paths": paths["robot_paths"],
        "environment_paths": paths["environment_paths"],
        "contact_populated_count": count,
        "zero_impulse_populated_count": unloaded_count,
        "maximum_contacts_per_actor": max_contacts,
        "maximum_impulse_to_force_conversion_error_n": max_force_error,
        "maximum_accounting_error_n": max_budget_error,
        "first_pair_load_above_0_1n_s": first_loaded,
        "pair_peaks": peaks,
        "deepest_loaded_constraint": deepest_loaded,
        "deepest_unloaded_constraint": deepest_unloaded,
        "series": series,
        "diagnosis": "Gripper housing on top edge of bin front supplies the principal blocking load. Housing peak precedes accepted/measured TCP error >20mm at5.3s. Frozen wrist proxy omits housing forces, so wrist guard cannot see this modeled tool-body load. Pad-only load is much smaller. No collision changes or new safety decisions in diagnostic replay.",
        "limitations": [
            "Exact actor/filter attribution does not calibrate physical force magnitude, housing shape or bin pose.",
            "Scene-rate force samples can miss transient peaks; raw impulse belongs to the last physics step, not the whole interval between rendered samples.",
            "Robot/environment contacts are observed; robot self-contact is not.",
            "Fixed robot base overlaps the raised table and creates zero-impulse geometric constraints; this is an existing mounting approximation, not positive dynamic load.",
            "No teacher ranking or new policy score is inferred from this separate command replay.",
        ],
    }
    out["observer_audit_pass"] = (
        out["all_21_arrays_identical"]
        and max_force_error < 1e-8
        and max_budget_error < 1e-8
    )
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
