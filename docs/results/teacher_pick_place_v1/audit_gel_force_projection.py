#!/usr/bin/env python3
"""Read-only post-experiment audit; run on the host retaining raw tactile NPZ."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from tools.sim.gel_contact import GelSurfaceGeometry, select_gel_contacts

    geometry = GelSurfaceGeometry()
    result = {
        "classification": "post-experiment read-only diagnostic; no rescoring or source changes",
        "root": str(args.root),
        "geometry": asdict(geometry),
        "units_and_axes": {
            "normal_force": "newtons from PhysX contact impulse divided by physics dt; not calibrated real SDK force",
            "physical_packet_force": "sum of magnitudes of each populated pad/packet normal-contact constraint, including backing/linkage contacts",
            "gel_force": "sum(abs(normal_constraint_N) * abs(unit_normal_pad_X) * active_area_fraction)",
            "coordinates": "contact positions in pad local metres; pad X closing axis; left inner face X=-0.006m, right X=+0.006m, tolerance0.002m",
            "active_ellipse": "(padY/0.0135)^2 + (padZ/0.018)^2 <= 1",
            "sdk": "gel columns=+padZ; rows=+padY; positive compression is added as negative SDK wrench Z; orientation is an explicit uncalibrated estimate",
            "shear": "not observed by this normal-contact API; not reconstructed from pad world net forces",
        },
        "source_sha256": {
            rel: sha(args.source / rel)
            for rel in [
                "tools/sim/gel_contact.py",
                "phantom/sim/tactile_proxy.py",
                "tools/sim/run_waffles.py",
            ]
        },
        "cases": {},
        "interpretation": {
            "nominal": "The deterministic sparse-manifold fallback rejects both loaded left points and the dominant right point as outside the active ellipse. Two additional zero-force manifold vertices on each pad are excluded before hull construction. This is not shear, net-force cancellation, normal-axis sign error, or a stale-sample explanation.",
            "success": "Four strictly positive loaded points per pad permit a convex support patch; gel force follows the active ellipse overlap fraction. The small normal-projection loss is insufficient to explain the nominal deficit.",
            "physical_limit": "Sparse solver constraints do not establish a calibrated pressure field. Numerical agreement with the frozen mapper does not validate zero physical pressure in the gel region. Existing frozen scores and all40 outcomes remain unchanged.",
            "next_fix_hypothesis": "In an isolated future mapper version, assess whether valid zero-force geometric manifold vertices should bound support when the same body has positive total load, and compare against measured pressure/contact evidence. This may remove the two-point discontinuity, but zero-force vertices alone do not prove loaded area and may overestimate it. A calibrated compliant-contact footprint or an explicitly uncertain sparse-line model is needed; do not replace the active-area test with whole-pad net force.",
        },
    }
    for case, video_t in [
        ("teacher__nominal__seed4242", 17.0),
        ("teacher__placement_xm10_ym10mm__seed4242", 18.0),
    ]:
        directory = args.root / "campaign" / "rollouts" / case
        contacts = json.loads((directory / "gel_contact_trace.json").read_text())
        tactile = np.load(directory / "policy_tactile.npz")
        trace = np.load(directory / "sim_trace.npz")
        sample = int(np.searchsorted(tactile["t"], video_t + 1e-9, side="right") - 1)
        frame = int(np.searchsorted(trace["t"], video_t + 1e-9, side="right") - 1)
        row = contacts[sample]
        pads = []
        maximum_reconstruction_error = 0.0
        # Replay only the numerical contact selection in pad coordinates, with
        # identity pose. No scene/physics/model inference is constructed.
        for contact_row in contacts:
            for side, pad in zip(("left", "right"), contact_row["per_pad"]):
                assert pad["duplicate_filter_contact_indices"] == 0
                forces, positions, normals, counts, starts, paths = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for body in pad["per_filter_contacts"]:
                    starts.append(len(forces))
                    counts.append(body["contact_count"])
                    paths.append(body["filter_path"])
                    forces.extend(body["normal_force_by_contact_n"])
                    positions.extend(body["contact_points_pad_m"])
                    normals.extend(body["contact_normals_pad"])
                data = (
                    np.asarray(forces),
                    np.asarray(positions).reshape(-1, 3),
                    np.asarray(normals).reshape(-1, 3),
                    None,
                    np.asarray(counts)[None, :],
                    np.asarray(starts)[None, :],
                )
                rebuilt = select_gel_contacts(
                    data,
                    [0, 0, 0],
                    [1, 0, 0, 0],
                    side=side,
                    geometry=geometry,
                    filter_paths=paths,
                    coverage=contact_row["coverage_mode"],
                )
                maximum_reconstruction_error = max(
                    maximum_reconstruction_error,
                    abs(rebuilt["normal_force_n"] - pad["normal_force_n"]),
                )
        for side, pad in zip(("left", "right"), row["per_pad"]):
            enriched = {"side": side, "recorded": pad, "bodies": []}
            for body in pad["per_filter_contacts"]:
                points = np.asarray(body["contact_points_pad_m"])
                normals = np.asarray(body["contact_normals_pad"])
                forces = np.asarray(body["normal_force_by_contact_n"])
                alignments = abs(normals[:, 0]) / np.linalg.norm(normals, axis=1)
                fractions = np.asarray(body["gel_force_fraction_by_contact"])
                enriched["bodies"].append(
                    {
                        "path": body["filter_path"],
                        "positive_force_contact_count": int((forces > 0).sum()),
                        "zero_force_contact_count": int((forces == 0).sum()),
                        "ellipse_radius_squared": np.sum(
                            (points[:, 1:] / [0.0135, 0.018]) ** 2, axis=1
                        ).tolist(),
                        "normal_alignment_abs_pad_X": alignments.tolist(),
                        "independent_formula_gel_force_n": float(
                            (forces * alignments * fractions).sum()
                        ),
                        "projection_loss_before_any_area_filter_n": float(
                            (forces * (1 - alignments)).sum()
                        ),
                    }
                )
            pads.append(enriched)
        result["cases"][case] = {
            "video_t_s": video_t,
            "scene_capture_t_s": float(trace["t"][frame]),
            "scene_packet_normal_force_n": trace["pad_packet_normal_force"][
                frame
            ].tolist(),
            "tactile_capture_t_s": float(tactile["t"][sample]),
            "tactile_age_s": float(video_t - tactile["t"][sample]),
            "tactile_sample_index": sample,
            "saved_gel_normal_force_n": tactile["gel_normal_force"][sample].tolist(),
            "saved_pad_world_net_force_n": tactile["pad_force"][sample].tolist(),
            "saved_gel_shape": list(tactile["gel"].shape),
            "saved_gel_dtype": str(tactile["gel"].dtype),
            "saved_gel_frame_sha256": [
                hashlib.sha256(a.tobytes()).hexdigest() for a in tactile["gel"][sample]
            ],
            "input_sha256": {
                name: sha(directory / name)
                for name in (
                    "policy_tactile.npz",
                    "gel_contact_trace.json",
                    "sim_trace.npz",
                    "effective_config.json",
                    "policy_info.json",
                    "run.json",
                )
            },
            "all_samples_crosscheck": {
                "sample_count": len(contacts),
                "max_npz_vs_contact_json_timestamp_difference_s": float(
                    np.max(abs(tactile["t"] - [r["t"] for r in contacts]))
                ),
                "max_npz_vs_contact_json_force_difference_n": float(
                    np.max(
                        abs(
                            tactile["gel_normal_force"]
                            - [r["normal_force_n"] for r in contacts]
                        )
                    )
                ),
                "max_reconstructed_mapper_force_difference_n": maximum_reconstruction_error,
            },
            "selected_sample_pads": pads,
            "neighboring_tactile_samples": [
                {
                    "t": r["t"],
                    "gel_normal_force_n": r["normal_force_n"],
                    "all_contact_normal_force_n": [
                        p["observed_filtered_normal_force_n"] for p in r["per_pad"]
                    ],
                    "body_coverage": [
                        [
                            {"body": b["filter_path"], **b["coverage"]}
                            for b in p["per_filter_contacts"]
                        ]
                        for p in r["per_pad"]
                    ],
                }
                for r in contacts[max(0, sample - 2) : sample + 2]
            ],
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
