#!/usr/bin/env python3
"""Compare mapper outputs on identical saved contacts, without rolling physics."""

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
SOURCE = BASE / "source_teacher_v2_delivery"
sys.path.insert(0, str(SOURCE))
from tools.sim.gel_contact import select_gel_contacts

ROOT = BASE / "runs/teacher_success_anchor_v3/components"
LEGACY = BASE / "source_teacher_pick_place_v1/tools/sim/gel_contact.py"
spec = importlib.util.spec_from_file_location("anchor_frozen_legacy_gel", LEGACY)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(pad, paths):
    """Saved local points/normals permit identity-transform recomputation.

    Do not supply a fabricated separation for missing archived evidence.
    """
    force, points, normals, separation = [], [], [], []
    counts, starts = np.zeros((1, len(paths)), int), np.zeros((1, len(paths)), int)
    available = True
    for pair in pad["per_filter_contacts"]:
        col = pair["filter_column"]
        counts[0, col] = len(pair["normal_force_by_contact_n"])
        starts[0, col] = len(force)
        force.extend(pair["normal_force_by_contact_n"])
        points.extend(pair["contact_points_pad_m"])
        normals.extend(pair["contact_normals_pad"])
        sep = pair.get("separation_by_contact_m")
        if sep is None:
            available = False
        else:
            separation.extend(sep)
    return (
        np.asarray(force),
        np.asarray(points).reshape(-1, 3),
        np.asarray(normals).reshape(-1, 3),
        np.asarray(separation) if available else None,
        counts,
        starts,
    )


def main():
    outputs = []
    for seed in (904301, 904302):
        path = ROOT / "gel_v2_only/rollouts" / f"teacher__fixed_anchor__seed{seed}"
        baseline = ROOT / "baseline/rollouts" / path.name
        initial_a = np.load(baseline / "observations/0000.npz")
        initial_b = np.load(path / "observations/0000.npz")
        initial = {}
        for key in initial_a.files:
            a, b = initial_a[key], initial_b[key]
            difference = a.astype(float) - b.astype(float)
            initial[key] = {
                "bitwise_equal": a.dtype == b.dtype and a.tobytes() == b.tobytes(),
                "rms_delta": float(np.sqrt(np.mean(difference * difference))),
                "maximum_absolute_delta": float(np.abs(difference).max()),
            }
        plan_a = json.loads((baseline / "planner_trace.json").read_text())[0]
        plan_b = json.loads((path / "planner_trace.json").read_text())[0]
        actions_delta = np.asarray(plan_a["actions"]) - plan_b["actions"]
        startup = {
            "observation0": initial,
            "initialization_equal": json.loads(
                (baseline / "initialization.json").read_text()
            )
            == json.loads((path / "initialization.json").read_text()),
            "scene_config_bytes_equal": sha(baseline / "effective_config.json")
            == sha(path / "effective_config.json"),
            "first_activation_baseline_s": plan_a["activated_at"],
            "first_activation_v2_s": plan_b["activated_at"],
            "first_native_latency_baseline_s": plan_a["latency_s"],
            "first_native_latency_v2_s": plan_b["latency_s"],
            "first_action_translation_max_row_delta_m": float(
                np.linalg.norm(actions_delta[:, :3], axis=1).max()
            ),
            "first_action_closure_max_delta": float(abs(actions_delta[:, 6]).max()),
            "observation_sha256": {
                str(p): sha(p)
                for p in (
                    baseline / "observations/0000.npz",
                    path / "observations/0000.npz",
                )
            },
        }
        logs = json.loads((path / "gel_contact_trace.json").read_text())
        executions = [
            json.loads(line)
            for line in (path / "execution_trace.jsonl").read_text().splitlines()
        ]
        stop = next(row for row in executions if row["stopped"])
        logs = [row for row in logs if row["t"] <= stop["t"]]
        states, metadata = [], []
        conservation_error, reconstruction_error, legacy_error = 0.0, 0.0, 0.0
        support_routes = Counter()
        for row in logs:
            values, details = [], []
            for i, side in enumerate(("left", "right")):
                data = reconstruct(row["per_pad"][i], row["filter_paths"][i])
                assert data[3] is not None, "No invented missing separation permitted"
                result = {}
                for mode in ("manifold_patch", "manifold_patch_v2"):
                    result[mode] = select_gel_contacts(
                        data,
                        np.zeros(3),
                        [1, 0, 0, 0],
                        side=side,
                        filter_paths=row["filter_paths"][i],
                        coverage=mode,
                    )
                v1, v2 = result["manifold_patch"], result["manifold_patch_v2"]
                frozen_v1 = legacy.select_gel_contacts(
                    data,
                    np.zeros(3),
                    [1, 0, 0, 0],
                    side=side,
                    filter_paths=row["filter_paths"][i],
                    coverage="manifold_patch",
                )
                legacy_error = max(
                    legacy_error,
                    abs(frozen_v1["normal_force_n"] - v1["normal_force_n"]),
                )
                reconstruction_error = max(
                    reconstruction_error,
                    abs(v2["normal_force_n"] - row["normal_force_n"][i]),
                )
                conservation_error = max(
                    conservation_error,
                    abs(
                        v2["observed_filtered_normal_force_n"]
                        - v2["normal_force_n"]
                        - v2["ignored_contact_normal_force_n"]
                        - v2["unprojected_accepted_normal_force_n"]
                    ),
                )
                values.append([v1["normal_force_n"], v2["normal_force_n"]])
                pairs = [
                    p
                    for p in v2["per_filter_contacts"]
                    if p["filter_path"] == "/World/Waffle"
                ]
                pair = None if not pairs else pairs[0]
                if pair:
                    coverage = pair["coverage"]
                    support_routes[
                        (side, coverage["method"], coverage.get("fallback_reason"))
                    ] += 1
                    details.append(
                        {
                            "side": side,
                            "packet_total_normal_n": pair["normal_force_magnitude_n"],
                            "v1_gel_n": v1["normal_force_n"],
                            "v2_gel_n": v2["normal_force_n"],
                            "coverage": coverage,
                            "point_force_n": pair["normal_force_by_contact_n"],
                            "points_pad_m": pair["contact_points_pad_m"],
                            "separation_m": pair["separation_by_contact_m"],
                        }
                    )
                else:
                    details.append(None)
            states.append(values)
            metadata.append(details)
        states = np.asarray(states)
        assert reconstruction_error < 1e-9, reconstruction_error
        assert conservation_error < 1e-9, conservation_error
        assert legacy_error < 1e-9, legacy_error
        v1_both = (states[:, :, 0] >= 2.5).all(axis=1)
        v2_both = (states[:, :, 1] >= 2.5).all(axis=1)
        flips = np.flatnonzero(v2_both & ~v1_both)

        def first(mask, logs=logs):
            found = np.flatnonzero(mask)
            return None if not len(found) else logs[int(found[0])]["t"]

        examples = []
        for i in list(flips[:1]) + (
            [int(np.argmax((states[:, :, 1] - states[:, :, 0]).sum(axis=1)))]
            if len(states)
            else []
        ):
            if not any(x["t_s"] == logs[i]["t"] for x in examples):
                examples.append({"t_s": logs[i]["t"], "pads": metadata[i]})
        outputs.append(
            {
                "seed": seed,
                "matched_seed_live_startup_confounds": startup,
                "folder": str(path),
                "stop_t_s": stop["t"],
                "samples_before_stop": len(logs),
                "first_simultaneous_2point5N_v1_on_same_contacts_s": first(v1_both),
                "first_simultaneous_2point5N_v2_s": first(v2_both),
                "v1_bilateral_threshold_samples": int(v1_both.sum()),
                "v2_bilateral_threshold_samples": int(v2_both.sum()),
                "v2_threshold_but_v1_below_samples": len(flips),
                "maximum_v2_reconstruction_error_n": reconstruction_error,
                "maximum_legacy_branch_vs_frozen_v1_error_n": legacy_error,
                "maximum_force_partition_residual_n": conservation_error,
                "routes": [
                    {"side": k[0], "method": k[1], "fallback": k[2], "samples": n}
                    for k, n in support_routes.items()
                ],
                "first_actual_latch_s": next(
                    (
                        r["t"]
                        for r in executions
                        if r["diagnostics"].get("grip_latch") is not None
                    ),
                    None,
                ),
                "examples": examples,
                "input_sha256": {
                    name: sha(path / name)
                    for name in (
                        "gel_contact_trace.json",
                        "execution_trace.jsonl",
                        "sim_trace.npz",
                    )
                },
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "method": "Reconstruct populated per-filter buffers from saved local coordinates/normals/forces/separations, identity pad transform; evaluate preserved v1 branch and v2 branch on identical inputs. No physics or closed-loop rerun.",
                "limits": [
                    "Same-contact differences establish a mapper mechanism, not the cause of unmatched live policy outcomes.",
                    "All positive/zero separations are actual saved API data; speculative geometry is not measured gel indentation.",
                    "Uniform patch pressure and declared convex packet support remain uncalibrated assumptions.",
                    "Baseline already latched and retained its packet at the reach guard; do not describe its failure as missing latch.",
                    "No threshold changes, invented load, pressure-area inflation, or hardware sensor calibration.",
                ],
                "mapper_source": str(SOURCE / "tools/sim/gel_contact.py"),
                "mapper_source_sha256": sha(SOURCE / "tools/sim/gel_contact.py"),
                "legacy_source": str(LEGACY),
                "legacy_source_sha256": sha(LEGACY),
                "cases": outputs,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
