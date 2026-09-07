#!/usr/bin/env python3
"""Read-only CPU audit; run on compute3 and redirect JSON to a new artifact."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(pad, paths):
    """Rebuild only populated entries, retaining archived buffer/filter identity."""
    records = pad["per_filter_contacts"]
    ids = sorted({i for r in records for i in r["populated_buffer_indices"]})
    size = max(ids, default=-1) + 1
    force = np.full(size, np.nan)
    points = np.full((size, 3), np.nan)
    normals = np.full((size, 3), np.nan)
    separation = np.full(size, np.nan)
    counts = np.zeros((1, len(paths)), dtype=int)
    starts = np.zeros_like(counts)
    seen = set()
    for r in records:
        idx = np.array(r["populated_buffer_indices"], dtype=int)
        column = r["filter_column"]
        assert r["filter_path"] == paths[column]
        assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))
        assert r["contact_count"] == len(idx)
        assert r["separation_by_contact_m"] is not None
        values = (
            r["normal_force_by_contact_n"],
            r["contact_points_pad_m"],
            r["contact_normals_pad"],
            r["separation_by_contact_m"],
        )
        for buffer, values_in in zip((force, points, normals, separation), values):
            values_in = np.asarray(values_in)
            for i, value in zip(idx, values_in):
                if i in seen:
                    assert np.array_equal(buffer[i], value)
                buffer[i] = value
        seen.update(idx)
        counts[0, column], starts[0, column] = len(idx), idx[0]
    assert pad["populated_buffer_indices"] == ids
    assert len(ids) == pad["populated_unique_contact_count"]
    assert np.isfinite(separation[ids]).all()
    return force, points, normals, separation, counts, starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from tools.sim.gel_contact import select_gel_contacts

    fresh, reference = args.fresh, args.reference
    run = json.loads((fresh / "run.json").read_text())
    old_run = json.loads((reference / "run.json").read_text())
    a, b = np.load(fresh / "sim_trace.npz"), np.load(reference / "sim_trace.npz")
    comparisons = {}
    for key in sorted(b.files):
        if key not in a or key not in b:
            comparisons[key] = {"identical": False, "reason": "missing_array"}
            continue
        same_shape = a[key].shape == b[key].shape
        finite = np.isfinite(a[key]) & np.isfinite(b[key]) if same_shape else None
        comparisons[key] = {
            "identical": same_shape
            and bool(np.array_equal(a[key], b[key], equal_nan=True)),
            "shape": list(a[key].shape),
            "max_absolute_difference": float(
                np.max(abs(a[key][finite] - b[key][finite]))
            )
            if same_shape and finite.any()
            else None,
        }
    rows = json.loads((fresh / "gel_contact_trace.json").read_text())
    statistics = Counter()
    reasons = Counter()
    by_body = Counter()
    separation_sets = {"all": [], "loaded": [], "zero_impulse": []}
    max_errors = Counter()
    maxima = Counter()
    examples = []
    deepest_contact = None
    per_sample = []
    for row in rows:
        sample = {
            "t_s": row["t"],
            "v1_gel_n": [],
            "v2_gel_n": [],
            "physical_filtered_normal_n": [],
        }
        for i, (side, pad) in enumerate(zip(("left", "right"), row["per_pad"])):
            paths = row["filter_paths"][i]
            assert pad["coverage_mode"] == "manifold_patch_v2"
            assert pad["separation_status"] == "available"
            data = restore(pad, paths)
            kwargs = {"side": side, "filter_paths": paths}
            v1 = select_gel_contacts(
                data, [0, 0, 0], [1, 0, 0, 0], coverage="manifold_patch", **kwargs
            )
            v2 = select_gel_contacts(
                data, [0, 0, 0], [1, 0, 0, 0], coverage="manifold_patch_v2", **kwargs
            )
            for key in (
                "normal_force_n",
                "observed_filtered_normal_force_n",
                "accepted_contact_normal_magnitude_n",
                "ignored_contact_normal_force_n",
                "unprojected_accepted_normal_force_n",
            ):
                max_errors["saved_recomputation_" + key] = max(
                    max_errors["saved_recomputation_" + key], abs(v2[key] - pad[key])
                )
            max_errors["saved_recomputation_uv"] = max(
                max_errors["saved_recomputation_uv"],
                float(np.max(abs(v2["contact_uv"] - pad["contact_uv"]))),
            )
            budget = (
                v2["accepted_contact_normal_magnitude_n"]
                + v2["ignored_contact_normal_force_n"]
                - v2["observed_filtered_normal_force_n"]
            )
            projection = (
                v2["normal_force_n"]
                + v2["unprojected_accepted_normal_force_n"]
                - v2["accepted_contact_normal_magnitude_n"]
            )
            ignored = (
                sum(v2["ignored_by_reason_n"].values())
                - v2["ignored_contact_normal_force_n"]
            )
            for name, val in (
                ("force_budget_n", budget),
                ("projection_budget_n", projection),
                ("ignored_reason_budget_n", ignored),
            ):
                max_errors[name] = max(max_errors[name], abs(val))
            assert v2["normal_force_n"] <= v2["observed_filtered_normal_force_n"] + 1e-9
            statistics["pad_readouts"] += 1
            statistics["populated_contacts"] += pad["populated_unique_contact_count"]
            statistics["duplicate_filter_indices"] += pad[
                "duplicate_filter_contact_indices"
            ]
            maxima["populated_contacts_one_pad"] = max(
                maxima["populated_contacts_one_pad"],
                pad["populated_unique_contact_count"],
            )
            for raw, recomputed in zip(
                pad["per_filter_contacts"], v2["per_filter_contacts"]
            ):
                assert raw["coverage"]["method"] == recomputed["coverage"]["method"]
                assert (
                    raw["coverage"]["fallback_reason"]
                    == recomputed["coverage"]["fallback_reason"]
                )
                c = raw["coverage"]
                statistics["method_" + c["method"]] += 1
                reasons[str(c["fallback_reason"])] += 1
                by_body[raw["filter_path"]] += 1
                force = np.array(raw["normal_force_by_contact_n"])
                sep = np.array(raw["separation_by_contact_m"])
                minimum_index = int(np.argmin(sep))
                if (
                    deepest_contact is None
                    or sep[minimum_index] < deepest_contact["separation_m"]
                ):
                    deepest_contact = {
                        "t_s": row["t"],
                        "side": side,
                        "body": raw["filter_path"],
                        "separation_m": float(sep[minimum_index]),
                        "normal_force_n": float(force[minimum_index]),
                        "point_pad_m": raw["contact_points_pad_m"][minimum_index],
                        "normal_pad": raw["contact_normals_pad"][minimum_index],
                    }
                for name, mask in (
                    ("all", np.ones(len(sep), dtype=bool)),
                    ("loaded", force > 0),
                    ("zero_impulse", force == 0),
                ):
                    separation_sets[name].extend(sep[mask].tolist())
                statistics["loaded_contacts"] += int((force > 0).sum())
                statistics["zero_impulse_contacts"] += int((force == 0).sum())
                statistics["positive_separation_contacts"] += int((sep > 0).sum())
                statistics["outside_fixed_2mm_envelope_contacts"] += int(
                    (sep > 0.002).sum()
                )
                statistics["zero_impulse_geometric_vertices"] += c[
                    "zero_impulse_geometric_count"
                ]
                if c["method"] == "manifold_patch_v2":
                    assert raw["filter_path"] == "/World/Waffle"
                    assert c["minimum_pairwise_normal_agreement"] >= 0.99
                    assert 0 <= c["overlap_fraction"] <= 1
                    statistics["patches_using_zero_impulse_geometry"] += (
                        c["zero_impulse_geometric_count"] > 0
                    )
            delta = v2["normal_force_n"] - v1["normal_force_n"]
            statistics["pad_readouts_changed_gt_1e_8_n"] += abs(delta) > 1e-8
            sample["v1_gel_n"].append(v1["normal_force_n"])
            sample["v2_gel_n"].append(v2["normal_force_n"])
            sample["physical_filtered_normal_n"].append(
                v2["observed_filtered_normal_force_n"]
            )
            if abs(delta) > 1e-8:
                examples.append(
                    {
                        "t_s": row["t"],
                        "side": side,
                        "v1_gel_n": v1["normal_force_n"],
                        "v2_gel_n": v2["normal_force_n"],
                        "delta_n": delta,
                        "physical_filtered_normal_n": v2[
                            "observed_filtered_normal_force_n"
                        ],
                        "coverage": [r["coverage"] for r in v2["per_filter_contacts"]],
                        "per_body_force_n": [
                            {
                                k: r[k]
                                for k in (
                                    "filter_path",
                                    "normal_force_magnitude_n",
                                    "gel_compression_n",
                                    "ignored_normal_force_n",
                                    "separation_by_contact_m",
                                    "normal_force_by_contact_n",
                                )
                            }
                            for r in v2["per_filter_contacts"]
                        ],
                    }
                )
        per_sample.append(sample)
    v1forces = np.array([s["v1_gel_n"] for s in per_sample])
    v2forces = np.array([s["v2_gel_n"] for s in per_sample])
    times = np.array([s["t_s"] for s in per_sample])
    metrics_path = reference / "strict_metrics.json"
    reference_metrics = (
        json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    )
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_directory": str(fresh),
        "reference_directory": str(reference),
        "source_directory": str(args.source),
        "source_sha256": {
            "tools/sim/gel_contact.py": sha(args.source / "tools/sim/gel_contact.py")
        },
        "input_sha256": {
            label: {
                p.name: sha(p)
                for p in folder.iterdir()
                if p.name
                in {
                    "sim_trace.npz",
                    "run.json",
                    "effective_config.json",
                    "gel_contact_trace.json",
                    "initialization.json",
                    "strict_metrics.json",
                }
            }
            for label, folder in (("fresh", fresh), ("reference", reference))
        },
        "physics_invariance": {
            "all_arrays_identical": all(x["identical"] for x in comparisons.values()),
            "comparison_scope": "all arrays present in the reference; added telemetry listed separately",
            "additional_fresh_arrays": sorted(set(a.files) - set(b.files)),
            "arrays": comparisons,
            "effective_config_bytes_identical": (
                fresh / "effective_config.json"
            ).read_bytes()
            == (reference / "effective_config.json").read_bytes(),
            "initialization_bytes_identical": (
                fresh / "initialization.json"
            ).read_bytes()
            == (reference / "initialization.json").read_bytes(),
            "camera_intrinsics_identical": run["camera_intrinsics_px"]
            == old_run["camera_intrinsics_px"],
            "robot_asset_identical_path": run["robot_usd"] == old_run["robot_usd"],
            "support_filters_identical": run["packet_support_filter_paths"]
            == old_run["packet_support_filter_paths"]
            if "packet_support_filter_paths" in old_run
            else None,
            "robot_filter_count": len(run["packet_support_filter_paths"]["robot"]),
            "bin_filter_count": len(run["packet_support_filter_paths"]["bin"]),
            "recorded_mode": run["mode"],
            "reference_strict_metrics": reference_metrics,
        },
        "contact_validation": {
            "scene_samples": len(times),
            "statistics": dict(statistics),
            "maxima": dict(maxima),
            "fallback_reasons": dict(reasons),
            "populated_filter_readouts_by_body": dict(by_body),
            "separation_m": {
                k: {
                    "count": len(v),
                    "min": min(v) if v else None,
                    "max": max(v) if v else None,
                }
                for k, v in separation_sets.items()
            },
            "maximum_error": dict(max_errors),
            "deepest_populated_contact": deepest_contact,
            "gel_timestamps_equal_scene_t": bool(np.array_equal(times, a["t"])),
            "gel_timestamps_strictly_increasing": bool(np.all(np.diff(times) > 0)),
            "v1_peak_per_pad_n": v1forces.max(axis=0).tolist(),
            "v2_peak_per_pad_n": v2forces.max(axis=0).tolist(),
            "v1_bilateral_ge_2_5n_scene_samples": int(
                np.all(v1forces >= 2.5, axis=1).sum()
            ),
            "v2_bilateral_ge_2_5n_scene_samples": int(
                np.all(v2forces >= 2.5, axis=1).sum()
            ),
            "max_abs_v1_v2_difference_n": float(abs(v2forces - v1forces).max()),
            "largest_changed_examples": sorted(
                examples, key=lambda x: abs(x["delta_n"]), reverse=True
            )[:6],
            "samples": per_sample,
        },
        "limitations": [
            "Measured joint replay observes a fixed trajectory; these are not closed-loop policy success results.",
            "Scene-rate samples are not the policy's causal 8 Hz observation schedule; threshold counts do not prove a live latch transition.",
            "Area coverage assumes uniform pressure on a connected convex support patch. No calibrated gel pressure, friction/shear, body patch IDs or deformation is measured.",
            "Positive PhysX separation is a speculative contact gap inside the fixed 2 mm contactOffset envelope, not gel indentation.",
            "Unknown and nonconvex body filters retain point fallback. Backing and tangent contacts remain physical safety loads without being painted into gel.",
        ],
    }
    output["audit_pass"] = (
        output["physics_invariance"]["all_arrays_identical"]
        and max(max_errors.values()) < 1e-8
        and output["contact_validation"]["gel_timestamps_equal_scene_t"]
    )
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
